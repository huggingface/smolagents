#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MCP Application Firewall
========================
A multi-layer application firewall for smolagents MCP tool ingestion.
Prevents tool poisoning, prompt injection, rug-pull attacks, PII leakage,
and rate abuse via MCP servers.

Eight security layers
---------------------
  1. TrustVerifier        — pre-flight reputation/trust-score check on MCP
                            server URLs/commands before any TCP connection.
  2. MCPPayloadValidator  — validates tool metadata (name, description, schema)
                            to block prompt injection and resource exhaustion.
  3. MCPToolFingerprinter — SHA-256 lockfile detects rug-pull attacks (tool
                            definitions changing between connections).
  4. MCPCallSentinel      — pre/post-call inspection; blocks credential
                            exfiltration in args and injection in responses.
  5. MCPAuditLogger       — structured JSONL audit log of every tool call.
  6. MCPToolAllowlist     — whitelist mode; blocks calls to unapproved tools.
  7. MCPResponseSanitizer — strips PII (emails, cards, SSNs, JWTs) from
                            responses before they reach the LLM context.
  8. MCPRateLimiter       — sliding-window call budget per server and tool.

Main entry point
----------------
  ``MCPFirewall`` is the facade that wires all eight layers together.
  Use a preset or build from a config file::

      fw = MCPFirewall.preset("strict")
      fw = MCPFirewall.from_yaml("firewall.yml")
      fw = MCPFirewall.from_env()

      with MCPClient(server_params, **fw.as_kwargs()) as tools:
          ...

  See ``MCPFirewall.PRESETS`` for available presets and ``from_config()``
  for the full configuration reference.
"""

from __future__ import annotations

import builtins
import hashlib as _hashlib
import json as _json
import keyword
import logging
import re
import threading as _threading
import time as _time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import ast as _ast

logger = logging.getLogger(__name__)

__all__ = [
    # Exceptions
    "MCPServerUntrustedError",
    "MCPPayloadValidationError",
    "MCPRugPullDetectedError",
    "MCPCallInterceptedError",
    "MCPToolBlockedError",
    "MCPRateLimitExceededError",
    # Trust verification
    "TrustVerificationResult",
    "TrustVerifier",
    "StaticTrustVerifier",
    "CompositeTrustVerifier",
    # Payload & AST validation
    "MCPPayloadValidator",
    "_validate_tool_code_ast",
    # Runtime guardian
    "MCPToolFingerprint",
    "MCPToolFingerprinter",
    "MCPCallSentinel",
    "MCPAuditLogger",
    "MCPAuditLogReader",
    "wrap_tool_with_guardian",
    # Allowlist & sanitizer
    "MCPToolAllowlist",
    "MCPResponseSanitizer",
    # Rate limiting
    "MCPRateLimiter",
    # Security event hooks
    "MCPSecurityHook",
    "MCPConsoleHook",
    "MCPFileHook",
    "MCPCallbackHook",
    # Analytics
    "MCPCallStats",
    "MCPSecurityReport",
    # Firewall facade (the main entry point)
    "MCPFirewall",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPServerUntrustedError(RuntimeError):
    """Raised when an MCP server fails the pre-flight trust verification.

    Attributes:
        server_id: Human-readable identifier of the server that was rejected.
        trust_score: The computed trust score (0.0 = fully untrusted).
        reasons: List of human-readable explanations for the rejection.
    """

    def __init__(self, server_id: str, trust_score: float, reasons: list[str]):
        self.server_id = server_id
        self.trust_score = trust_score
        self.reasons = reasons
        reasons_text = "; ".join(reasons) if reasons else "no reasons given"
        super().__init__(
            f"MCP server '{server_id}' failed trust verification "
            f"(score={trust_score:.2f}): {reasons_text}"
        )


class MCPPayloadValidationError(ValueError):
    """Raised when an MCP server's tool metadata fails safety validation.

    Attributes:
        tool_name: The name of the tool that triggered the error (may be raw/unsafe).
        field: Which field failed (e.g. 'description', 'name', 'inputs').
        detail: Specific reason for the failure.
    """

    def __init__(self, tool_name: str, field: str, detail: str):
        self.tool_name = tool_name
        self.field = field
        self.detail = detail
        super().__init__(
            f"MCP tool payload validation failed for tool '{tool_name}' "
            f"in field '{field}': {detail}"
        )


# ---------------------------------------------------------------------------
# TrustVerificationResult
# ---------------------------------------------------------------------------


@dataclass
class TrustVerificationResult:
    """Result of a pre-flight trust check on an MCP server.

    Attributes:
        trusted: Whether the server is considered safe to connect to.
        trust_score: Numeric score from 0.0 (fully untrusted) to 1.0 (fully trusted).
        server_id: Human-readable identifier of the server that was evaluated.
        reasons: Ordered list of human-readable explanations for each scoring decision.
    """

    trusted: bool
    trust_score: float
    server_id: str
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TrustVerifier — Abstract Base Class
# ---------------------------------------------------------------------------


class TrustVerifier(ABC):
    """Abstract interface for MCP server trust verification.

    Subclass this to implement custom reputation checks:
    - Query an external reputation API
    - Check against a company CMDB
    - Validate HuggingFace Hub MCP server listings
    - Apply organisation-specific allowlists

    The ``verify`` method is called *before* any TCP connection is established.
    """

    @abstractmethod
    def verify(self, server_parameters: Any) -> TrustVerificationResult:
        """Evaluate the trustworthiness of an MCP server before connecting.

        Args:
            server_parameters: The same value passed to ``MCPClient`` or
                ``ToolCollection.from_mcp()``. Can be a
                ``mcp.StdioServerParameters``, a ``dict`` with a ``url`` key,
                or a list of either.

        Returns:
            TrustVerificationResult with ``trusted=True`` if the server may be
            connected to, or ``trusted=False`` if it should be blocked.
        """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# These IP ranges must never be reached via an MCP URL regardless of config.
# 169.254.169.254 is the AWS/GCP/Azure instance metadata endpoint — exfiltrating
# credentials from it is the most common cloud supply-chain attack.
_HARD_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"169\.254\.169\.254"),  # cloud metadata endpoints
    re.compile(r"100\.100\.100\.200"),  # Alibaba Cloud metadata
    re.compile(r"^javascript:", re.IGNORECASE),
    re.compile(r"^file://", re.IGNORECASE),
    re.compile(r"^data:", re.IGNORECASE),
]

_LOCALHOST_HOSTS = frozenset(
    ["localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"]
)


def _extract_server_id(server_parameters: Any) -> str:
    """Return a human-readable identifier string from server_parameters."""
    if isinstance(server_parameters, list):
        ids = [_extract_server_id(p) for p in server_parameters]
        return "[" + ", ".join(ids) + "]"
    if isinstance(server_parameters, dict):
        return server_parameters.get("url", repr(server_parameters))
    # StdioServerParameters or similar: try .command attribute
    command = getattr(server_parameters, "command", None)
    if command is not None:
        args = getattr(server_parameters, "args", [])
        return f"stdio:{command} {' '.join(str(a) for a in args)}".strip()
    return repr(server_parameters)


def _extract_urls(server_parameters: Any) -> list[str]:
    """Return all URL strings present in server_parameters."""
    if isinstance(server_parameters, list):
        urls: list[str] = []
        for p in server_parameters:
            urls.extend(_extract_urls(p))
        return urls
    if isinstance(server_parameters, dict):
        url = server_parameters.get("url")
        if url:
            return [url]
    return []


def _is_stdio(server_parameters: Any) -> bool:
    """Return True if server_parameters represents a stdio (subprocess) server."""
    if isinstance(server_parameters, list):
        return all(_is_stdio(p) for p in server_parameters)
    return not isinstance(server_parameters, dict)


def _is_localhost_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.hostname in _LOCALHOST_HOSTS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# StaticTrustVerifier
# ---------------------------------------------------------------------------


class StaticTrustVerifier(TrustVerifier):
    """Default trust verifier based on static rules — no external calls.

    Scoring logic (in priority order):
    1. Hard-blocked patterns (cloud metadata endpoints, file://, etc.) → 0.0, blocked.
    2. User-supplied blocklist match → 0.0, blocked.
    3. Allowlist configured but no match → 0.0, blocked.
    4. Allowlist match → 1.0, trusted.
    5. stdio server (subprocess) → 0.65, trusted (user explicitly configured command).
    6. Localhost HTTP/HTTPS → 0.75, trusted.
    7. HTTPS non-localhost → 0.85, trusted.
    8. HTTP non-localhost + require_https=True → 0.0, blocked.
    9. HTTP non-localhost + require_https=False → 0.55, trusted.

    Any computed score below ``min_trust_score`` is treated as untrusted even
    if the rule would otherwise permit the server.

    Args:
        blocklist: Iterable of regex pattern strings. Any URL or server ID
            matching one of these patterns is immediately blocked.
        allowlist: Optional iterable of regex pattern strings. When set, only
            servers matching at least one pattern are trusted.
        require_https: Reject plain-HTTP connections to non-localhost servers.
            Defaults to True.
        min_trust_score: Minimum score required for a server to be trusted.
            Defaults to 0.5.
    """

    def __init__(
        self,
        blocklist: list[str] | None = None,
        allowlist: list[str] | None = None,
        require_https: bool = True,
        min_trust_score: float = 0.5,
    ):
        self._blocklist: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in (blocklist or [])
        ]
        self._allowlist: list[re.Pattern[str]] | None = (
            [re.compile(p, re.IGNORECASE) for p in allowlist]
            if allowlist is not None
            else None
        )
        self.require_https = require_https
        self.min_trust_score = min_trust_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, server_parameters: Any) -> TrustVerificationResult:
        server_id = _extract_server_id(server_parameters)
        reasons: list[str] = []

        # Flatten to a list for uniform processing
        params_list = server_parameters if isinstance(server_parameters, list) else [server_parameters]

        individual_scores: list[float] = []
        for params in params_list:
            result = self._verify_single(params, reasons)
            individual_scores.append(result.trust_score)
            if not result.trusted:
                return TrustVerificationResult(
                    trusted=False,
                    trust_score=result.trust_score,
                    server_id=server_id,
                    reasons=reasons,
                )

        # All individual params passed — use the minimum score collected above
        final_score = min(individual_scores)

        if final_score < self.min_trust_score:
            reasons.append(
                f"computed score {final_score:.2f} is below min_trust_score {self.min_trust_score:.2f}"
            )
            return TrustVerificationResult(
                trusted=False,
                trust_score=final_score,
                server_id=server_id,
                reasons=reasons,
            )

        reasons.append(f"all checks passed (score={final_score:.2f})")
        return TrustVerificationResult(
            trusted=True,
            trust_score=final_score,
            server_id=server_id,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_single(self, params: Any, reasons: list[str]) -> TrustVerificationResult:
        """Verify a single (non-list) server parameters object."""
        server_id = _extract_server_id(params)

        # 1. Hard-blocked patterns
        candidate = server_id
        for pat in _HARD_BLOCKED_PATTERNS:
            if pat.search(candidate):
                reasons.append(f"hard-blocked pattern matched: '{pat.pattern}'")
                return TrustVerificationResult(
                    trusted=False, trust_score=0.0, server_id=server_id, reasons=reasons
                )

        # Also check extracted URLs independently
        for url in _extract_urls(params):
            for pat in _HARD_BLOCKED_PATTERNS:
                if pat.search(url):
                    reasons.append(f"hard-blocked pattern matched in URL '{url}': '{pat.pattern}'")
                    return TrustVerificationResult(
                        trusted=False, trust_score=0.0, server_id=server_id, reasons=reasons
                    )

        # 2. User blocklist
        for pat in self._blocklist:
            if pat.search(candidate):
                reasons.append(f"server matches blocklist pattern '{pat.pattern}'")
                return TrustVerificationResult(
                    trusted=False, trust_score=0.0, server_id=server_id, reasons=reasons
                )

        # 3 & 4. Allowlist
        if self._allowlist is not None:
            matched = any(pat.search(candidate) for pat in self._allowlist)
            if not matched:
                reasons.append(f"server '{server_id}' is not in the allowlist")
                return TrustVerificationResult(
                    trusted=False, trust_score=0.0, server_id=server_id, reasons=reasons
                )
            reasons.append("server matches allowlist — fully trusted")
            return TrustVerificationResult(
                trusted=True, trust_score=1.0, server_id=server_id, reasons=reasons
            )

        # 5. stdio
        if _is_stdio(params):
            reasons.append("stdio server: user-configured subprocess, trusted by convention")
            return TrustVerificationResult(
                trusted=True, trust_score=0.65, server_id=server_id, reasons=reasons
            )

        # 6–9. HTTP/HTTPS URL checks
        urls = _extract_urls(params)
        for url in urls:
            parsed = urlparse(url)
            scheme = (parsed.scheme or "").lower()
            is_local = _is_localhost_url(url)

            if is_local:
                reasons.append(f"localhost URL '{url}' accepted")
                continue

            if scheme == "https":
                reasons.append(f"HTTPS URL '{url}' accepted")
                continue

            if scheme == "http":
                if self.require_https:
                    reasons.append(
                        f"plain HTTP URL '{url}' rejected (require_https=True). "
                        "Use HTTPS or set require_https=False for development only."
                    )
                    return TrustVerificationResult(
                        trusted=False, trust_score=0.0, server_id=server_id, reasons=reasons
                    )
                reasons.append(f"plain HTTP URL '{url}' accepted (require_https=False, dev mode)")
                continue

            # Unknown scheme
            reasons.append(f"unknown URL scheme '{scheme}' in '{url}' — rejected")
            return TrustVerificationResult(
                trusted=False, trust_score=0.0, server_id=server_id, reasons=reasons
            )

        return TrustVerificationResult(
            trusted=True, trust_score=self._score_single(params), server_id=server_id, reasons=reasons
        )

    def _score_single(self, params: Any) -> float:
        """Compute a numeric score for a single params object (no side-effects)."""
        if _is_stdio(params):
            return 0.65

        urls = _extract_urls(params)
        if not urls:
            return 0.65  # No URL means it must be stdio-like

        min_score = 1.0
        for url in urls:
            parsed = urlparse(url)
            scheme = (parsed.scheme or "").lower()
            if _is_localhost_url(url):
                min_score = min(min_score, 0.75)
            elif scheme == "https":
                min_score = min(min_score, 0.85)
            elif scheme == "http":
                min_score = min(min_score, 0.55)  # above min_trust_score=0.5 default; below HTTPS (0.85)
            else:
                min_score = min(min_score, 0.0)
        return min_score


# ---------------------------------------------------------------------------
# CompositeTrustVerifier
# ---------------------------------------------------------------------------


class CompositeTrustVerifier(TrustVerifier):
    """Chain multiple TrustVerifiers with fail-any semantics.

    The server must pass ALL verifiers to be considered trusted.
    The final trust score is the minimum across all verifiers.
    All verifiers are always evaluated so that the full set of reasons
    is collected (useful for audit logs), unless ``fail_fast=True``.

    Args:
        verifiers: Ordered list of TrustVerifier instances to apply.
        fail_fast: If True, stop evaluating after the first failure.
            Defaults to False (collect all reasons).
    """

    def __init__(self, verifiers: list[TrustVerifier], fail_fast: bool = False):
        if not verifiers:
            raise ValueError("CompositeTrustVerifier requires at least one verifier.")
        self.verifiers = verifiers
        self.fail_fast = fail_fast

    def verify(self, server_parameters: Any) -> TrustVerificationResult:
        server_id = _extract_server_id(server_parameters)
        all_reasons: list[str] = []
        min_score = 1.0
        failed = False

        for verifier in self.verifiers:
            result = verifier.verify(server_parameters)
            all_reasons.extend(
                [f"[{type(verifier).__name__}] {r}" for r in result.reasons]
            )
            min_score = min(min_score, result.trust_score)
            if not result.trusted:
                failed = True
                if self.fail_fast:
                    break

        return TrustVerificationResult(
            trusted=not failed,
            trust_score=min_score,
            server_id=server_id,
            reasons=all_reasons,
        )


# ---------------------------------------------------------------------------
# MCPPayloadValidator
# ---------------------------------------------------------------------------

# Set of Python builtin names that must not be used as tool names because they
# would silently shadow the builtin in the agent's generated code.
_PYTHON_BUILTINS: frozenset[str] = frozenset(dir(builtins))

# Regex patterns indicating likely prompt-injection attempts.
# Applied case-insensitively to tool names, descriptions and input descriptions.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore-instructions",      re.compile(r"ignore\s+(previous|all|above|prior)\s+(instructions?|context|prompt|directive)", re.IGNORECASE)),
    ("system-role-override",     re.compile(r"\bsystem\s*:\s", re.IGNORECASE)),
    ("token-injection",          re.compile(r"<\|[a-z_]+\|>", re.IGNORECASE)),
    ("os-environ",               re.compile(r"\bos\s*\.\s*environ\b", re.IGNORECASE)),
    ("subprocess",               re.compile(r"\bsubprocess\b", re.IGNORECASE)),
    ("eval-call",                re.compile(r"\beval\s*\(", re.IGNORECASE)),
    ("exec-call",                re.compile(r"\bexec\s*\(", re.IGNORECASE)),
    ("import-os",                re.compile(r"\bimport\s+os\b", re.IGNORECASE)),
    ("null-byte",                re.compile(r"\x00")),
    ("jinja-template",           re.compile(r"\{\{|\}\}|\{%|%\}")),  # Break Jinja2 templates
    # --- Additional patterns (Phase 6) ---
    # HTML script injection: a description containing <script> could fire in web-rendered UIs
    ("html-script-injection",    re.compile(r"<\s*script\b", re.IGNORECASE)),
    # Prompt extraction: "repeat the system prompt" / "print the above instructions"
    # Classic exfiltration of the LLM's own context window.
    ("prompt-extraction",        re.compile(
        r"\b(?:repeat|print|output|display|echo|show|return)\s+"
        r"(?:the\s+)?(?:above|previous|prior|following|system|prompt|instructions?)",
        re.IGNORECASE,
    )),
    # Unicode bidi override characters used to make malicious text appear benign
    # (Trojan Source / CVE-2021-42574 style attacks).
    ("unicode-bidi-override",    re.compile(r"[‪-‮⁦-⁩‏؜]")),
]


class MCPPayloadValidator:
    """Validates MCP tool metadata to prevent prompt injection and resource exhaustion.

    All limits are conservative defaults tuned for production safety.
    They can be relaxed by passing custom values to the constructor.

    Args:
        max_tool_name_length: Maximum characters in a tool name. Default 64.
        max_description_length: Maximum characters in tool description. Default 4096.
        max_tools_per_server: Maximum number of tools a single server may expose. Default 100.
        max_input_params: Maximum number of input parameters per tool. Default 20.
        max_param_description_length: Maximum characters in an input param description. Default 1024.
        allow_builtin_names: If True, tool names that shadow Python builtins are permitted.
            Default False (recommended: keep False in production).
    """

    def __init__(
        self,
        max_tool_name_length: int = 64,
        max_description_length: int = 4096,
        max_tools_per_server: int = 100,
        max_input_params: int = 20,
        max_param_description_length: int = 1024,
        allow_builtin_names: bool = False,
        extra_injection_patterns: list[str] | None = None,
    ):
        self.max_tool_name_length = max_tool_name_length
        self.max_description_length = max_description_length
        self.max_tools_per_server = max_tools_per_server
        self.max_input_params = max_input_params
        self.max_param_description_length = max_param_description_length
        self.allow_builtin_names = allow_builtin_names
        self.extra_injection_patterns: list[str] = list(extra_injection_patterns or [])
        self._compiled_extra_injection: list[tuple[str, re.Pattern[str]]] = [
            (f"custom-injection-{i}", re.compile(p))
            for i, p in enumerate(self.extra_injection_patterns)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_tool_list(self, tools: list[Any], server_id: str) -> None:
        """Validate the full list of tools returned by an MCP server.

        Args:
            tools: List of smolagents Tool instances returned after MCP connection.
            server_id: Human-readable server identifier for error messages.

        Raises:
            MCPPayloadValidationError: On the first tool that fails validation.
        """
        if len(tools) > self.max_tools_per_server:
            raise MCPPayloadValidationError(
                tool_name="<server>",
                field="tool_count",
                detail=(
                    f"Server '{server_id}' returned {len(tools)} tools, "
                    f"which exceeds the maximum of {self.max_tools_per_server}. "
                    "This may indicate a resource-exhaustion attack."
                ),
            )
        for tool in tools:
            self.validate_tool(tool)

    def validate_tool(self, tool: Any) -> None:
        """Validate a single smolagents Tool instance received from an MCP server.

        Checks ``tool.name``, ``tool.description``, and all input parameter
        descriptions within ``tool.inputs``.

        Args:
            tool: A smolagents Tool instance.

        Raises:
            MCPPayloadValidationError: If any field fails validation.
        """
        raw_name = getattr(tool, "name", "") or ""
        self._validate_name(raw_name)
        self._validate_description(raw_name, getattr(tool, "description", "") or "")
        inputs: dict[str, Any] = getattr(tool, "inputs", {}) or {}
        self._validate_inputs(raw_name, inputs)

    # ------------------------------------------------------------------
    # Internal field validators
    # ------------------------------------------------------------------

    def _validate_name(self, name: str) -> None:
        if not name:
            raise MCPPayloadValidationError(
                tool_name=name, field="name", detail="tool name must not be empty"
            )
        if len(name) > self.max_tool_name_length:
            raise MCPPayloadValidationError(
                tool_name=name,
                field="name",
                detail=f"length {len(name)} exceeds maximum {self.max_tool_name_length}",
            )
        # Dunder names (could interfere with Python internals)
        if name.startswith("__") and name.endswith("__"):
            raise MCPPayloadValidationError(
                tool_name=name,
                field="name",
                detail="dunder names are forbidden as tool names",
            )
        # Python keywords
        if keyword.iskeyword(name):
            raise MCPPayloadValidationError(
                tool_name=name,
                field="name",
                detail=f"'{name}' is a Python reserved keyword and cannot be used as a tool name",
            )
        # Builtin shadowing
        if not self.allow_builtin_names and name in _PYTHON_BUILTINS:
            raise MCPPayloadValidationError(
                tool_name=name,
                field="name",
                detail=(
                    f"'{name}' shadows a Python builtin. "
                    "This can silently override core language functions in agent-generated code."
                ),
            )
        # Injection scan on the name itself
        self._scan_for_injection(name, "name", name)

    def _validate_description(self, tool_name: str, description: str) -> None:
        if len(description) > self.max_description_length:
            raise MCPPayloadValidationError(
                tool_name=tool_name,
                field="description",
                detail=(
                    f"length {len(description)} exceeds maximum {self.max_description_length}. "
                    "Oversized descriptions may indicate a prompt-stuffing attack."
                ),
            )
        self._scan_for_injection(tool_name, "description", description)

    def _validate_inputs(self, tool_name: str, inputs: dict[str, Any]) -> None:
        if len(inputs) > self.max_input_params:
            raise MCPPayloadValidationError(
                tool_name=tool_name,
                field="inputs",
                detail=(
                    f"{len(inputs)} input parameters exceed the maximum of {self.max_input_params}"
                ),
            )
        for param_name, param_schema in inputs.items():
            if not isinstance(param_schema, dict):
                continue
            desc = param_schema.get("description", "") or ""
            if len(desc) > self.max_param_description_length:
                raise MCPPayloadValidationError(
                    tool_name=tool_name,
                    field=f"inputs['{param_name}'].description",
                    detail=(
                        f"parameter description length {len(desc)} exceeds "
                        f"maximum {self.max_param_description_length}"
                    ),
                )
            self._scan_for_injection(tool_name, f"inputs['{param_name}'].description", desc)

    def _scan_for_injection(self, tool_name: str, field: str, text: str) -> None:
        """Raise MCPPayloadValidationError if text matches any injection pattern."""
        for pattern_name, pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                raise MCPPayloadValidationError(
                    tool_name=tool_name,
                    field=field,
                    detail=(
                        f"suspicious pattern '{pattern_name}' detected. "
                        "This may indicate a prompt injection or code injection attempt."
                    ),
                )
        for pattern_name, pattern in self._compiled_extra_injection:
            if pattern.search(text):
                raise MCPPayloadValidationError(
                    tool_name=tool_name,
                    field=field,
                    detail=(
                        f"suspicious pattern '{pattern_name}' detected. "
                        "This may indicate a prompt injection or code injection attempt."
                    ),
                )


# ---------------------------------------------------------------------------
# AST pre-validator for Tool.from_code() / Hub-tool exec() (Layer 3)
# ---------------------------------------------------------------------------

# Calls to these names are forbidden even when trust_remote_code=True because
# they can escape a Tool class into the host process at load time.
_FORBIDDEN_CALL_NAMES: frozenset[str] = frozenset(
    {"exec", "eval", "compile", "__import__", "breakpoint", "open", "memoryview"}
)

# Modules that must never be imported inside Hub-downloaded tool code.
_FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "builtins", "ctypes", "gc", "importlib", "inspect", "io",
        "multiprocessing", "os", "pathlib", "pickle", "platform",
        "pty", "pwd", "shutil", "signal", "socket", "subprocess",
        "sys", "sysconfig", "tempfile", "threading", "urllib", "http",
        "requests", "httpx", "ftplib", "smtplib", "telnetlib",
    }
)


def _validate_tool_code_ast(code: str) -> None:
    """Statically analyse Hub-tool source code before exec().

    Walks the AST and raises ``ValueError`` if the code contains calls to
    dangerous built-ins or imports of forbidden modules.  This is a
    defense-in-depth layer: ``trust_remote_code=True`` is still required by
    the caller; this check adds a static gate on top.

    Args:
        code: Python source code string to analyse.

    Raises:
        ValueError: If the code cannot be parsed or contains forbidden patterns.
    """
    try:
        tree = _ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Tool code has a syntax error: {exc}") from exc

    for node in _ast.walk(tree):
        # Block calls to dangerous built-in names
        if isinstance(node, _ast.Call):
            func = node.func
            if isinstance(func, _ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                raise ValueError(
                    f"Tool code contains a forbidden call to '{func.id}'. "
                    "This function is not allowed in Hub-loaded tool code."
                )
            if isinstance(func, _ast.Attribute) and func.attr in _FORBIDDEN_CALL_NAMES:
                raise ValueError(
                    f"Tool code contains a forbidden attribute call to '{func.attr}'. "
                    "This function is not allowed in Hub-loaded tool code."
                )

        # Block forbidden imports
        if isinstance(node, _ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    raise ValueError(
                        f"Tool code contains a forbidden import of '{alias.name}'. "
                        f"Module '{root}' is not allowed in Hub-loaded tool code."
                    )

        if isinstance(node, _ast.ImportFrom):
            module_root = (node.module or "").split(".")[0]
            if module_root in _FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(
                    f"Tool code contains a forbidden 'from {node.module} import ...' statement. "
                    f"Module '{module_root}' is not allowed in Hub-loaded tool code."
                )


# ===========================================================================
# Phase 5 — MCP Runtime Guardian
# ===========================================================================
# Component A — MCPToolFingerprinter : Rug Pull Detector
# Component B — MCPCallSentinel      : Runtime Firewall (pre/post inspection)
# Component C — MCPAuditLogger       : Structured JSON audit log
# ===========================================================================


# ---------------------------------------------------------------------------
# New Exceptions (Phase 5)
# ---------------------------------------------------------------------------


class MCPRugPullDetectedError(RuntimeError):
    """Raised when an MCP tool's definition changes between connections.

    A rug-pull attack occurs when a malicious server presents safe tool
    definitions during the user's initial trust review, then silently swaps
    them for malicious ones on subsequent connections.

    Attributes:
        server_id: Identifier of the MCP server where the mutation was found.
        tool_name: Name of the tool whose fingerprint changed.
        expected_fingerprint: SHA-256 hex recorded in the lockfile.
        actual_fingerprint: SHA-256 hex computed from the live server data.
    """

    def __init__(
        self,
        server_id: str,
        tool_name: str,
        expected_fingerprint: str,
        actual_fingerprint: str,
    ):
        self.server_id = server_id
        self.tool_name = tool_name
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        super().__init__(
            f"Rug pull detected: tool '{tool_name}' on server '{server_id}' "
            f"has changed since last verified connection. "
            f"Expected fingerprint {expected_fingerprint[:12]}..., "
            f"got {actual_fingerprint[:12]}.... "
            "Call MCPToolFingerprinter.approve_update() to accept the new definition."
        )


class MCPCallInterceptedError(RuntimeError):
    """Raised when a tool call is blocked by MCPCallSentinel.

    Attributes:
        tool_name: Name of the tool that was blocked.
        phase: Either 'pre-call' (args inspection) or 'post-call' (response inspection).
        reason: Human-readable explanation of why the call was blocked.
    """

    def __init__(self, tool_name: str, phase: str, reason: str):
        self.tool_name = tool_name
        self.phase = phase
        self.reason = reason
        super().__init__(
            f"MCP tool call intercepted [{phase}] for tool '{tool_name}': {reason}"
        )


class MCPToolBlockedError(RuntimeError):
    """Raised when a tool call is blocked by ``MCPToolAllowlist``.

    Attributes:
        server_id: Server that hosts the blocked tool.
        tool_name: Tool name that was not on the allowlist.
        reason: Human-readable explanation.
    """

    def __init__(self, server_id: str, tool_name: str, reason: str):
        self.server_id = server_id
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(
            f"MCP tool '{tool_name}' on server '{server_id}' blocked by allowlist: {reason}"
        )


class MCPRateLimitExceededError(RuntimeError):
    """Raised when a tool call exceeds the configured sliding-window rate limit.

    Attributes:
        server_id: Server whose call budget was exhausted.
        tool_name: Tool that was being called.
        limit: Maximum calls allowed within ``window_seconds``.
        window_seconds: Width of the sliding window in seconds.
        scope: ``"server"`` for the per-server limit, ``"tool"`` for the per-tool limit.
    """

    def __init__(
        self,
        server_id: str,
        tool_name: str,
        limit: int,
        window_seconds: float,
        scope: str = "server",
    ):
        self.server_id = server_id
        self.tool_name = tool_name
        self.limit = limit
        self.window_seconds = window_seconds
        self.scope = scope
        subject = tool_name if scope == "tool" else server_id
        super().__init__(
            f"Rate limit exceeded [{scope}] for '{subject}': "
            f"max {limit} call(s) per {window_seconds:.0f}s window"
        )


# ---------------------------------------------------------------------------
# MCPToolFingerprint — dataclass
# ---------------------------------------------------------------------------


@dataclass
class MCPToolFingerprint:
    """Immutable record of a verified MCP tool definition at a point in time.

    Attributes:
        tool_name: The tool's name as reported by the MCP server.
        fingerprint: SHA-256 hex digest of (name + description + inputs).
        server_id: Identifier of the MCP server that provided this tool.
        created_at: ISO 8601 UTC timestamp when this fingerprint was first recorded.
    """

    tool_name: str
    fingerprint: str
    server_id: str
    created_at: str


# ---------------------------------------------------------------------------
# MCPToolFingerprinter — Component A (Rug Pull Detector)
# ---------------------------------------------------------------------------


class MCPToolFingerprinter:
    """Detects rug-pull attacks by fingerprinting MCP tool definitions.

    On the first connection to a server, SHA-256 hashes of every tool's
    ``name + description + inputs`` are written to a local lockfile
    (analogous to ``package-lock.json`` but for MCP servers).

    On every subsequent connection, the live tool definitions are re-hashed
    and compared against the lockfile.  If any previously-seen tool's
    definition has changed, ``MCPRugPullDetectedError`` is raised before the
    tools are made available to the agent.

    New tools (not previously seen) are registered automatically.
    Tool removals are logged as warnings but do not raise.

    Args:
        lockfile_path: Path to the JSON lockfile.  Defaults to
            ``.mcp-lock.json`` in the current working directory.

    Example:
        ```python
        fingerprinter = MCPToolFingerprinter()
        with MCPClient(server_params, fingerprinter=fingerprinter) as tools:
            # First run: lockfile written.
            # Subsequent runs: verified against lockfile.
            ...
        ```
    """

    _LOCKFILE_VERSION = "1"

    def __init__(self, lockfile_path: str | Path | None = None):
        self._lockfile_path = Path(lockfile_path or ".mcp-lock.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fingerprint_and_verify(self, tools: list[Any], server_id: str) -> list[MCPToolFingerprint]:
        """Compute fingerprints and verify against the lockfile.

        First call: registers all tools in the lockfile.
        Subsequent calls: compares live fingerprints to stored ones and raises
        ``MCPRugPullDetectedError`` if any previously-seen tool has changed.

        Args:
            tools: List of smolagents Tool instances from the MCP server.
            server_id: Human-readable server identifier.

        Returns:
            List of ``MCPToolFingerprint`` for each tool.

        Raises:
            MCPRugPullDetectedError: If a tool's definition has mutated.
        """
        lock_data = self._load_lockfile()
        server_entry: dict[str, dict] = (
            lock_data.setdefault("servers", {}).setdefault(server_id, {})
        )

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fingerprints: list[MCPToolFingerprint] = []
        updated = False

        for tool in tools:
            tool_name = getattr(tool, "name", "") or ""
            live_fp = self._compute_fingerprint(tool)

            if tool_name in server_entry:
                stored_fp = server_entry[tool_name]["fingerprint"]
                if stored_fp != live_fp:
                    raise MCPRugPullDetectedError(
                        server_id=server_id,
                        tool_name=tool_name,
                        expected_fingerprint=stored_fp,
                        actual_fingerprint=live_fp,
                    )
                fingerprints.append(
                    MCPToolFingerprint(
                        tool_name=tool_name,
                        fingerprint=live_fp,
                        server_id=server_id,
                        created_at=server_entry[tool_name]["created_at"],
                    )
                )
            else:
                logger.info(
                    "MCPToolFingerprinter: registering new tool '%s' on server '%s'",
                    tool_name,
                    server_id,
                )
                server_entry[tool_name] = {"fingerprint": live_fp, "created_at": now_iso}
                fingerprints.append(
                    MCPToolFingerprint(
                        tool_name=tool_name,
                        fingerprint=live_fp,
                        server_id=server_id,
                        created_at=now_iso,
                    )
                )
                updated = True

        # Warn (but don't block) when known tools disappear from the server
        live_names = {getattr(t, "name", "") for t in tools}
        for stored_name in list(server_entry.keys()):
            if stored_name not in live_names:
                logger.warning(
                    "MCPToolFingerprinter: tool '%s' is no longer present on server '%s'. "
                    "This may indicate tool removal or a server-side change.",
                    stored_name,
                    server_id,
                )

        if updated:
            self._save_lockfile(lock_data)

        return fingerprints

    def approve_update(self, server_id: str, tool_name: str) -> None:
        """Clear a stored fingerprint so the next connection re-registers the tool.

        Use this after manually verifying that a tool's definition change is
        intentional and not malicious.

        Args:
            server_id: Server identifier (same value used in ``fingerprint_and_verify``).
            tool_name: Name of the tool whose fingerprint should be cleared.
        """
        lock_data = self._load_lockfile()
        server_entry = lock_data.get("servers", {}).get(server_id, {})
        if tool_name in server_entry:
            del server_entry[tool_name]
            self._save_lockfile(lock_data)
            logger.info(
                "MCPToolFingerprinter: fingerprint for '%s' on '%s' cleared — "
                "will re-register on next connection.",
                tool_name,
                server_id,
            )
        else:
            logger.warning(
                "MCPToolFingerprinter.approve_update: tool '%s' not found for server '%s'.",
                tool_name,
                server_id,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_fingerprint(self, tool: Any) -> str:
        """SHA-256 of canonical JSON of tool name + description + inputs."""
        data = {
            "name": getattr(tool, "name", "") or "",
            "description": getattr(tool, "description", "") or "",
            "inputs": getattr(tool, "inputs", {}) or {},
        }
        canonical = _json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
        return _hashlib.sha256(canonical.encode()).hexdigest()

    def _load_lockfile(self) -> dict:
        """Load the lockfile; return an empty structure if absent or corrupt."""
        if not self._lockfile_path.exists():
            return {"version": self._LOCKFILE_VERSION, "servers": {}}
        try:
            with open(self._lockfile_path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("Lockfile root must be a JSON object.")
            return data
        except Exception as exc:
            logger.warning(
                "MCPToolFingerprinter: could not read lockfile '%s': %s. "
                "Starting fresh — all tools will be re-registered.",
                self._lockfile_path,
                exc,
            )
            return {"version": self._LOCKFILE_VERSION, "servers": {}}

    def _save_lockfile(self, data: dict) -> None:
        """Write the lockfile atomically (temp file + rename)."""
        tmp_path = self._lockfile_path.parent / (self._lockfile_path.name + ".tmp")
        try:
            self._lockfile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                _json.dump(data, fh, indent=2, ensure_ascii=True)
            tmp_path.replace(self._lockfile_path)
        except Exception as exc:
            logger.error(
                "MCPToolFingerprinter: failed to write lockfile '%s': %s",
                self._lockfile_path,
                exc,
            )
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# MCPToolAllowlist — Component A2 (Whitelist enforcement)
# ---------------------------------------------------------------------------


class MCPToolAllowlist:
    """Whitelist-mode tool gate: only explicitly approved (server, tool) pairs may run.

    On the **first connection** to a new server, all tools are auto-approved and
    saved to a JSON file (``auto_approve_first_connection=True``, default).
    On every subsequent connection, any tool **not** on the allowlist raises
    ``MCPToolBlockedError`` — catching tools silently injected by the server
    between connections.

    Set ``auto_approve_first_connection=False`` for strict environments where
    every tool must be explicitly whitelisted via
    ``smolagents-firewall allowlist add`` before the first connection.

    Args:
        allowlist_path: Path to the JSON allowlist file.  Defaults to
            ``.mcp-allowlist.json`` in the current working directory.
        auto_approve_first_connection: When ``True`` (default), all tools on a
            previously-unknown server are automatically approved on first connect.
            When ``False``, every tool must be pre-approved.

    Example:
        ```python
        from smolagents.mcp_firewall import MCPToolAllowlist, MCPFirewall

        # Strict: all tools must be pre-approved
        fw = MCPFirewall(allowlist=MCPToolAllowlist(auto_approve_first_connection=False))

        # Then from CLI: smolagents-firewall allowlist add <server_id> <tool_name>
        ```
    """

    _DEFAULT_PATH = Path(".mcp-allowlist.json")

    def __init__(
        self,
        allowlist_path: str | Path | None = None,
        auto_approve_first_connection: bool = True,
    ):
        self._allowlist_path = (
            Path(allowlist_path) if allowlist_path is not None else self._DEFAULT_PATH
        )
        self.auto_approve_first_connection = auto_approve_first_connection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def approve(self, server_id: str, tool_name: str) -> None:
        """Explicitly approve a (server_id, tool_name) pair."""
        data = self._load()
        servers = data.setdefault("servers", {})
        tools = servers.setdefault(server_id, {})
        tools[tool_name] = {"approved_at": datetime.now(timezone.utc).isoformat()}
        self._save(data)

    def revoke(self, server_id: str, tool_name: str) -> None:
        """Remove a (server_id, tool_name) pair from the allowlist."""
        data = self._load()
        try:
            del data["servers"][server_id][tool_name]
            if not data["servers"][server_id]:
                del data["servers"][server_id]
        except KeyError:
            pass
        self._save(data)

    def is_allowed(self, server_id: str, tool_name: str) -> bool:
        """Return ``True`` if the tool is on the allowlist for this server."""
        data = self._load()
        return tool_name in data.get("servers", {}).get(server_id, {})

    def list_approved(self, server_id: str | None = None) -> dict:
        """Return allowlist contents, optionally filtered to one server.

        Args:
            server_id: Filter to this server.  Returns all servers if ``None``.

        Returns:
            Dict mapping ``server_id → {tool_name → {approved_at: ...}}``.
        """
        data = self._load()
        servers = data.get("servers", {})
        if server_id is not None:
            return {server_id: servers.get(server_id, {})}
        return servers

    def validate_tool_list(self, tools: list[Any], server_id: str) -> None:
        """Enforce the allowlist on an entire tool list at connect time.

        First-connection behaviour (server not yet in allowlist):
          - ``auto_approve_first_connection=True``: auto-approves all tools.
          - ``auto_approve_first_connection=False``: raises ``MCPToolBlockedError``.

        Subsequent-connection behaviour:
          Raises ``MCPToolBlockedError`` for any tool whose name is **not** in
          the stored allowlist for this server.

        Args:
            tools: List of smolagents Tool objects returned by MCPAdapt.
            server_id: Server identifier (URL or command string).

        Raises:
            MCPToolBlockedError: If any tool is not on the allowlist.
        """
        data = self._load()
        servers = data.setdefault("servers", {})

        if server_id not in servers:
            if self.auto_approve_first_connection:
                servers[server_id] = {
                    t.name: {"approved_at": datetime.now(timezone.utc).isoformat()}
                    for t in tools
                }
                self._save(data)
                logger.info(
                    "MCPToolAllowlist: auto-approved %d tool(s) for server '%s'",
                    len(tools),
                    server_id,
                )
                return
            # Strict mode — unknown server, no auto-approve
            blocked_names = [t.name for t in tools]
            raise MCPToolBlockedError(
                server_id=server_id,
                tool_name=", ".join(blocked_names) if blocked_names else "(none)",
                reason=(
                    f"Server '{server_id}' has no approved tools and "
                    "auto_approve_first_connection is disabled. "
                    "Use `smolagents-firewall allowlist add` to pre-approve tools."
                ),
            )

        # Server known — block any new tools not in the stored list
        approved = servers[server_id]
        blocked = [t.name for t in tools if t.name not in approved]
        if blocked:
            raise MCPToolBlockedError(
                server_id=server_id,
                tool_name=", ".join(blocked),
                reason=(
                    f"Tool(s) {blocked!r} on server '{server_id}' are not on the allowlist. "
                    "This may indicate a new tool was injected by the server. "
                    "Use `smolagents-firewall allowlist add` to approve them."
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not self._allowlist_path.exists():
            return {"version": 1, "servers": {}}
        try:
            with open(self._allowlist_path, "r", encoding="utf-8") as fh:
                return _json.load(fh)
        except Exception:
            return {"version": 1, "servers": {}}

    def _save(self, data: dict) -> None:
        self._allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._allowlist_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                _json.dump(data, fh, indent=2, ensure_ascii=True)
            tmp_path.replace(self._allowlist_path)
        except Exception as exc:
            logger.error("MCPToolAllowlist: failed to save allowlist: %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MCPRateLimiter — Component A3 (Sliding-window rate limiting)
# ---------------------------------------------------------------------------


class MCPRateLimiter:
    """Thread-safe sliding-window rate limiter for MCP tool calls.

    Maintains per-server and optional per-tool call counts over a rolling time
    window using a ``deque`` of monotonic timestamps.  No external dependencies.

    If a tool call would exceed either the server-level cap or the per-tool cap,
    ``check()`` raises ``MCPRateLimitExceededError`` *before* the call is
    forwarded to the MCP server — preventing data exfiltration via high-frequency
    calls (e.g. one env-var per request).

    Args:
        max_calls_per_minute: Maximum total calls allowed per server per window.
            Default: 300.
        per_tool_max_calls_per_minute: Maximum calls per individual tool per window.
            ``None`` disables the per-tool limit.  Default: ``None``.
        window_seconds: Sliding window duration in seconds.  Default: 60.

    Example:
        ```python
        from smolagents.mcp_firewall import MCPRateLimiter, MCPFirewall

        # 60 total calls/min to a server, max 20 per individual tool
        fw = MCPFirewall(rate_limiter=MCPRateLimiter(
            max_calls_per_minute=60,
            per_tool_max_calls_per_minute=20,
        ))
        ```
    """

    def __init__(
        self,
        max_calls_per_minute: int = 300,
        per_tool_max_calls_per_minute: int | None = None,
        window_seconds: float = 60.0,
    ):
        if max_calls_per_minute < 1:
            raise ValueError("max_calls_per_minute must be >= 1")
        if per_tool_max_calls_per_minute is not None and per_tool_max_calls_per_minute < 1:
            raise ValueError("per_tool_max_calls_per_minute must be >= 1")
        self.max_calls_per_minute = max_calls_per_minute
        self.per_tool_max_calls_per_minute = per_tool_max_calls_per_minute
        self.window_seconds = window_seconds
        self._server_counts: dict[str, deque[float]] = {}
        self._tool_counts: dict[tuple[str, str], deque[float]] = {}
        self._lock = _threading.Lock()

    def check(self, server_id: str, tool_name: str) -> None:
        """Record a call attempt and raise if any rate limit would be exceeded.

        Call this *before* forwarding the tool invocation to the MCP server.
        The timestamp is recorded only when both checks pass, so a rejected
        call does not count against the budget.

        Args:
            server_id: MCP server identifier.
            tool_name: Tool being invoked.

        Raises:
            MCPRateLimitExceededError: If the server or tool call budget is full.
        """
        now = _time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # --- Server-level sliding window ---
            if server_id not in self._server_counts:
                self._server_counts[server_id] = deque()
            srv_dq = self._server_counts[server_id]
            # Evict timestamps outside the window
            while srv_dq and srv_dq[0] < cutoff:
                srv_dq.popleft()
            if len(srv_dq) >= self.max_calls_per_minute:
                raise MCPRateLimitExceededError(
                    server_id=server_id,
                    tool_name=tool_name,
                    limit=self.max_calls_per_minute,
                    window_seconds=self.window_seconds,
                    scope="server",
                )

            # --- Per-tool sliding window ---
            tool_dq: deque[float] | None = None
            if self.per_tool_max_calls_per_minute is not None:
                key = (server_id, tool_name)
                if key not in self._tool_counts:
                    self._tool_counts[key] = deque()
                tool_dq = self._tool_counts[key]
                while tool_dq and tool_dq[0] < cutoff:
                    tool_dq.popleft()
                if len(tool_dq) >= self.per_tool_max_calls_per_minute:
                    raise MCPRateLimitExceededError(
                        server_id=server_id,
                        tool_name=tool_name,
                        limit=self.per_tool_max_calls_per_minute,
                        window_seconds=self.window_seconds,
                        scope="tool",
                    )

            # All checks passed — record this call
            srv_dq.append(now)
            if tool_dq is not None:
                tool_dq.append(now)

    def current_counts(self, server_id: str) -> dict[str, int]:
        """Return live call counts within the current window for a server.

        Args:
            server_id: Server to query.

        Returns:
            Dict with ``"server"`` total count and per-tool counts under ``"tools"``.
        """
        now = _time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            srv_dq = self._server_counts.get(server_id, deque())
            server_count = sum(1 for ts in srv_dq if ts >= cutoff)
            tools: dict[str, int] = {}
            for (sid, tname), dq in self._tool_counts.items():
                if sid == server_id:
                    tools[tname] = sum(1 for ts in dq if ts >= cutoff)
        return {"server": server_count, "tools": tools}


# ---------------------------------------------------------------------------
# MCPCallSentinel — Component B (Runtime Firewall)
# ---------------------------------------------------------------------------

# Patterns suggesting credential data is being exfiltrated as tool arguments.
_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai-api-key",    re.compile(r"\bsk-[A-Za-z0-9]{20,48}\b")),
    ("anthropic-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,64}\b")),
    ("github-pat",        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("ssh-private-key",   re.compile(
        r"-----BEGIN\s+(?:RSA|EC|OPENSSH|DSA)\s+PRIVATE\s+KEY-----"
    )),
]

# Patterns suggesting sensitive filesystem paths are being leaked to a server.
_SENSITIVE_PATH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssh-directory",        re.compile(r"[/\\]\.ssh[/\\]", re.IGNORECASE)),
    ("aws-credentials-file", re.compile(
        r"[/\\]\.aws[/\\](?:credentials|config)\b", re.IGNORECASE
    )),
    ("dotenv-file",          re.compile(r"(?:^|[/\\])\.env(?:$|\b)", re.IGNORECASE)),
]


class MCPCallSentinel:
    """Runtime firewall that intercepts MCP tool calls pre- and post-execution.

    **Pre-call inspection** scans all string argument values for credential
    patterns (AWS access keys, OpenAI/Anthropic API keys, SSH private keys,
    GitHub tokens) and sensitive filesystem paths (~/.ssh, ~/.aws, .env).

    **Post-call inspection** scans every response for prompt-injection
    patterns (same set used by ``MCPPayloadValidator``) and enforces a maximum
    response size to prevent context-flooding attacks.

    Args:
        max_response_length: Maximum characters in a tool response before it
            is treated as a context-flooding attack.  Default 100 000.
        block_credential_exfil: Scan args for credential patterns.  Default True.
        block_sensitive_paths: Scan args for sensitive filesystem paths.  Default True.
        extra_blocked_arg_patterns: Additional regex strings scanned in args.
        extra_blocked_response_patterns: Additional regex strings scanned in responses.
    """

    def __init__(
        self,
        max_response_length: int = 100_000,
        block_credential_exfil: bool = True,
        block_sensitive_paths: bool = True,
        extra_blocked_arg_patterns: list[str] | None = None,
        extra_blocked_response_patterns: list[str] | None = None,
    ):
        self.max_response_length = max_response_length
        self.block_credential_exfil = block_credential_exfil
        self.block_sensitive_paths = block_sensitive_paths
        self._extra_arg_patterns: list[tuple[str, re.Pattern[str]]] = [
            (f"custom-arg-{i}", re.compile(p))
            for i, p in enumerate(extra_blocked_arg_patterns or [])
        ]
        self._extra_response_patterns: list[tuple[str, re.Pattern[str]]] = [
            (f"custom-resp-{i}", re.compile(p))
            for i, p in enumerate(extra_blocked_response_patterns or [])
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect_call_args(self, tool_name: str, kwargs: dict) -> None:
        """Inspect tool arguments before they are sent to the MCP server.

        Args:
            tool_name: Name of the tool being called.
            kwargs: Dictionary of tool call arguments.

        Raises:
            MCPCallInterceptedError: If any string value contains a credential
                or sensitive path pattern.
        """
        for value in self._extract_strings(kwargs):
            if self.block_credential_exfil:
                for pattern_name, pattern in _CREDENTIAL_PATTERNS:
                    if pattern.search(value):
                        raise MCPCallInterceptedError(
                            tool_name=tool_name,
                            phase="pre-call",
                            reason=(
                                f"argument contains credential pattern '{pattern_name}'. "
                                "Sending credentials to an external MCP server is not permitted."
                            ),
                        )
            if self.block_sensitive_paths:
                for pattern_name, pattern in _SENSITIVE_PATH_PATTERNS:
                    if pattern.search(value):
                        raise MCPCallInterceptedError(
                            tool_name=tool_name,
                            phase="pre-call",
                            reason=(
                                f"argument references sensitive path '{pattern_name}'. "
                                "Leaking filesystem paths to an external MCP server is not permitted."
                            ),
                        )
            for pattern_name, pattern in self._extra_arg_patterns:
                if pattern.search(value):
                    raise MCPCallInterceptedError(
                        tool_name=tool_name,
                        phase="pre-call",
                        reason=f"argument matches custom blocked pattern '{pattern_name}'.",
                    )

    def inspect_response(self, tool_name: str, response: Any) -> None:
        """Inspect a tool response before it is appended to the agent context.

        Args:
            tool_name: Name of the tool that produced the response.
            response: Raw response value from the tool.

        Raises:
            MCPCallInterceptedError: If the response exceeds the size limit or
                contains injection / credential patterns.
        """
        response_str = response if isinstance(response, str) else str(response)

        if len(response_str) > self.max_response_length:
            raise MCPCallInterceptedError(
                tool_name=tool_name,
                phase="post-call",
                reason=(
                    f"response length {len(response_str)} exceeds maximum "
                    f"{self.max_response_length}. "
                    "This may be a context-flooding attack."
                ),
            )

        for pattern_name, pattern in _INJECTION_PATTERNS:
            if pattern.search(response_str):
                raise MCPCallInterceptedError(
                    tool_name=tool_name,
                    phase="post-call",
                    reason=(
                        f"response contains injection pattern '{pattern_name}'. "
                        "This may be a response injection attack."
                    ),
                )

        for pattern_name, pattern in self._extra_response_patterns:
            if pattern.search(response_str):
                raise MCPCallInterceptedError(
                    tool_name=tool_name,
                    phase="post-call",
                    reason=f"response matches custom blocked pattern '{pattern_name}'.",
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_strings(self, value: Any, depth: int = 0) -> list[str]:
        """Recursively collect all string leaf values from a nested structure."""
        if depth > 10:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            result: list[str] = []
            for v in value.values():
                result.extend(self._extract_strings(v, depth + 1))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(self._extract_strings(item, depth + 1))
            return result
        return []


# ---------------------------------------------------------------------------
# MCPResponseSanitizer — Component C (PII Redaction)
# ---------------------------------------------------------------------------

# PII patterns scrubbed from tool responses before they reach the LLM context.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email",       re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')),
    ("credit-card", re.compile(r'\b(?:\d[ \-]?){13,18}\d\b')),
    ("ssn",         re.compile(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b')),
    ("phone",       re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b')),
    ("jwt",         re.compile(r'eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+')),
]


class MCPResponseSanitizer:
    """Strip PII from MCP tool responses before they reach the LLM context.

    Scans the text of every tool response and replaces detected PII with a
    configurable placeholder, protecting the LLM from inadvertently processing
    or leaking sensitive data that a tool (intentionally or not) returned.

    Detects:
    - Email addresses
    - Credit / debit card numbers (13–19 digit sequences)
    - US Social Security Numbers (NNN-NN-NNNN)
    - US phone numbers
    - JSON Web Tokens (eyJ…)

    Args:
        redact_emails: Redact email addresses (default: ``True``).
        redact_credit_cards: Redact card numbers (default: ``True``).
        redact_ssn: Redact Social Security Numbers (default: ``True``).
        redact_phone_numbers: Redact US phone numbers (default: ``True``).
        redact_jwt: Redact JWTs (default: ``True``).
        mode: How to redact matched text.

            - ``"redact"`` (default): replace with ``[REDACTED:type]``.
            - ``"hash"``: replace with ``[hash:hex8]`` (non-reversible).
            - ``"drop"``: remove entirely.

        custom_patterns: Extra ``(name, compiled_pattern)`` pairs to include.

    Example:
        ```python
        sanitizer = MCPResponseSanitizer(mode="hash")
        clean = sanitizer.sanitize("Contact me at alice@example.com")
        # clean → "Contact me at [hash:3d4f9a1b]"

        # Plug into the firewall:
        fw = MCPFirewall(sanitizer=MCPResponseSanitizer())
        ```
    """

    MODES = ("redact", "hash", "drop")

    def __init__(
        self,
        redact_emails: bool = True,
        redact_credit_cards: bool = True,
        redact_ssn: bool = True,
        redact_phone_numbers: bool = True,
        redact_jwt: bool = True,
        mode: str = "redact",
        custom_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(
                f"MCPResponseSanitizer: mode must be one of {self.MODES}, got '{mode}'"
            )
        self.mode = mode

        flags = [
            ("email",       redact_emails,        _PII_PATTERNS[0][1]),
            ("credit-card", redact_credit_cards,  _PII_PATTERNS[1][1]),
            ("ssn",         redact_ssn,           _PII_PATTERNS[2][1]),
            ("phone",       redact_phone_numbers, _PII_PATTERNS[3][1]),
            ("jwt",         redact_jwt,           _PII_PATTERNS[4][1]),
        ]
        self._active: list[tuple[str, re.Pattern[str]]] = [
            (name, pat) for name, enabled, pat in flags if enabled
        ]
        if custom_patterns:
            self._active.extend(custom_patterns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(self, text: str) -> str:
        """Scan and redact PII from a plain string.

        Args:
            text: Input text to sanitize.

        Returns:
            Sanitized copy of ``text`` with PII replaced.
        """
        for name, pattern in self._active:
            if self.mode == "redact":
                text = pattern.sub(f"[REDACTED:{name}]", text)
            elif self.mode == "hash":
                def _replace(m: re.Match, _name: str = name) -> str:
                    h = _hashlib.sha256(m.group(0).encode()).hexdigest()[:8]
                    return f"[hash:{h}]"
                text = pattern.sub(_replace, text)
            else:  # "drop"
                text = pattern.sub("", text)
        return text

    def sanitize_response(self, response: Any) -> Any:
        """Recursively sanitize a tool response (string, dict, list, or tuple).

        Strings are scanned directly.  Containers are traversed recursively.
        All other value types are returned unchanged.

        Args:
            response: Raw tool response value.

        Returns:
            Sanitized copy of the response.
        """
        if isinstance(response, str):
            return self.sanitize(response)
        if isinstance(response, dict):
            return {k: self.sanitize_response(v) for k, v in response.items()}
        if isinstance(response, (list, tuple)):
            sanitized = [self.sanitize_response(item) for item in response]
            return type(response)(sanitized)
        return response


# ---------------------------------------------------------------------------
# MCPAuditLogger — Component D (Structured Audit Log)
# ---------------------------------------------------------------------------


class MCPAuditLogger:
    """Structured JSON audit log for every MCP tool interaction.

    Appends one JSON line per tool call to a ``.jsonl`` file.  Each record
    contains a timestamp, server info, tool name, *hashed* arguments and
    response (raw values are never stored), performance metrics, and whether
    the call was blocked.

    Args:
        log_path: Path to the JSONL audit log file.  Defaults to
            ``~/.smolagents/audit.jsonl``.
        sink: Optional callable that receives each log record ``dict``.
            Called in addition to writing to the file — useful for streaming
            to a SIEM, Datadog, or any external system.

    Example:
        ```python
        audit = MCPAuditLogger()
        with MCPClient(server_params, audit_logger=audit) as tools:
            ...
        # or with a custom sink:
        audit = MCPAuditLogger(sink=lambda record: send_to_siem(record))
        ```
    """

    _DEFAULT_LOG_DIR = Path.home() / ".smolagents"
    _DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "audit.jsonl"

    def __init__(
        self,
        log_path: str | Path | None = None,
        sink: Any | None = None,
    ):
        self._log_path = Path(log_path) if log_path is not None else self._DEFAULT_LOG_FILE
        self._sink = sink

    def log_call(
        self,
        server_id: str,
        tool_name: str,
        args_hash: str,
        response_hash: str,
        trust_score: float | None,
        fingerprint_verified: bool,
        call_duration_ms: float,
        blocked: bool,
        block_reason: str | None = None,
    ) -> None:
        """Append one structured record to the audit log.

        Args:
            server_id: Human-readable MCP server identifier.
            tool_name: Name of the tool that was (or would have been) called.
            args_hash: Short SHA-256 hex of the call arguments (not raw args).
            response_hash: Short SHA-256 hex of the response (not raw response).
            trust_score: Trust score from pre-flight verification, or ``None``.
            fingerprint_verified: Whether fingerprint verification passed.
            call_duration_ms: Wall-clock time for the tool call in milliseconds.
            blocked: Whether the call was intercepted and blocked.
            block_reason: Human-readable reason if blocked, else ``None``.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "server_id": server_id,
            "tool_name": tool_name,
            "args_hash": args_hash,
            "response_hash": response_hash,
            "trust_score": trust_score,
            "fingerprint_verified": fingerprint_verified,
            "call_duration_ms": round(call_duration_ms, 3),
            "blocked": blocked,
            "block_reason": block_reason,
        }
        self._write_record(record)
        if self._sink is not None:
            try:
                self._sink(record)
            except Exception as exc:
                logger.warning("MCPAuditLogger: sink raised an exception: %s", exc)

    def _write_record(self, record: dict) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.error("MCPAuditLogger: failed to write audit record: %s", exc)


# ===========================================================================
# Phase 17 — Audit Log Reader
# ===========================================================================


class MCPAuditLogReader:
    """Read and analyse an :class:`MCPAuditLogger` JSONL audit log file.

    Each line of the log file is a JSON object written by
    :meth:`MCPAuditLogger.log_call`.  Malformed or blank lines are silently
    skipped with a warning.

    Args:
        path: Path to the JSONL audit log file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Example::

        reader = MCPAuditLogReader("~/.smolagents/audit.jsonl")
        print(reader.summary())
        blocked = reader.filter(blocked=True)
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self.events: list[dict] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        blocked: bool | None = None,
        server_id: str | None = None,
        tool_name: str | None = None,
        last: int | None = None,
    ) -> list[dict]:
        """Return a filtered subset of log events.

        All criteria are ANDed together.  ``None`` means "no filter on this
        field".

        Args:
            blocked: If ``True``, include only blocked calls.  If ``False``,
                include only allowed calls.  ``None`` = no filter.
            server_id: Keep only records for this server identifier.
            tool_name: Keep only records for this tool name.
            last: Return at most the last *N* matching records.

        Returns:
            List of matching event dicts (chronological order).
        """
        result = self.events
        if blocked is not None:
            result = [e for e in result if e.get("blocked") is blocked]
        if server_id is not None:
            result = [e for e in result if e.get("server_id") == server_id]
        if tool_name is not None:
            result = [e for e in result if e.get("tool_name") == tool_name]
        if last is not None:
            result = result[-last:]
        return result

    def summary(self) -> dict:
        """Return a summary dictionary computed from *all* loaded events.

        Keys
        ----
        ``total_calls``
            Total number of records in the log.
        ``blocked_calls``
            Number of calls where ``blocked`` is ``True``.
        ``allowed_calls``
            Number of calls where ``blocked`` is ``False``.
        ``block_reason_breakdown``
            Mapping of ``block_reason`` string → count (blocked calls only).
        ``top_tools``
            List of ``(tool_name, count)`` tuples, sorted by count descending,
            up to 5 entries.
        ``top_servers``
            List of ``(server_id, count)`` tuples, sorted by count descending,
            up to 5 entries.
        ``avg_duration_ms``
            Average ``call_duration_ms`` across all records, rounded to 3
            decimal places.  ``0.0`` when no records have a duration field.
        ``unverified_fingerprints``
            Count of records where ``fingerprint_verified`` is ``False``.

        Returns:
            Summary dict.
        """
        events = self.events
        total = len(events)
        blocked_events = [e for e in events if e.get("blocked")]
        allowed_events = [e for e in events if not e.get("blocked")]

        reason_counts: dict[str, int] = {}
        for e in blocked_events:
            reason = e.get("block_reason") or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        tool_counts: dict[str, int] = {}
        for e in events:
            t = e.get("tool_name", "<unknown>")
            tool_counts[t] = tool_counts.get(t, 0) + 1
        top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]

        server_counts: dict[str, int] = {}
        for e in events:
            s = e.get("server_id", "<unknown>")
            server_counts[s] = server_counts.get(s, 0) + 1
        top_servers = sorted(server_counts.items(), key=lambda x: -x[1])[:5]

        durations = [e["call_duration_ms"] for e in events if "call_duration_ms" in e]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        unverified = sum(1 for e in events if not e.get("fingerprint_verified", True))

        return {
            "total_calls": total,
            "blocked_calls": len(blocked_events),
            "allowed_calls": len(allowed_events),
            "block_reason_breakdown": reason_counts,
            "top_tools": top_tools,
            "top_servers": top_servers,
            "avg_duration_ms": round(avg_duration, 3),
            "unverified_fingerprints": unverified,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        records: list[dict] = []
        with open(self._path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    logger.warning(
                        "MCPAuditLogReader: skipping malformed JSON on line %d", lineno
                    )
        return records


# ===========================================================================
# Phase 10 — Security Event Hooks
# ===========================================================================
# Real-time push notifications for every firewall block event.
# Hooks are registered on MCPFirewall and dispatched automatically by
# wrap_tool_with_guardian, MCPClient, and ToolCollection.from_mcp().
# ===========================================================================


class MCPSecurityHook(ABC):
    """Abstract base class for MCP firewall security event hooks.

    Subclass and implement :meth:`on_event` to receive real-time alerts from
    any firewall layer.  Exceptions raised inside :meth:`on_event` are caught
    and logged so a misbehaving hook never crashes the agent.

    Emitted event types
    -------------------
    ``"server_blocked"``
        Server failed pre-flight trust verification.
        Extra details: ``trust_score``, ``reasons``.

    ``"rug_pull_detected"``
        Tool definition changed since the last fingerprinted connection.
        Extra details: ``tool_name``, ``expected_fingerprint``, ``actual_fingerprint``.

    ``"tool_blocked"``
        Tool is not on the allowlist.
        Extra details: ``tool_name``, ``reason``.

    ``"rate_limit_exceeded"``
        Call budget exhausted for server or tool.
        Extra details: ``tool_name``, ``limit``, ``scope``.

    ``"call_intercepted"``
        Sentinel blocked a pre- or post-call inspection.
        Extra details: ``tool_name``, ``phase``, ``reason``.

    Every ``details`` dict also contains:
        ``"event_type"`` (str): Same as the first argument.
        ``"server_id"`` (str): Server that triggered the event.
        ``"timestamp"`` (str): ISO 8601 UTC timestamp.

    Example:
        ```python
        from smolagents.mcp_firewall import MCPCallbackHook, MCPFirewall

        fw = MCPFirewall.preset("strict")
        fw.add_hook(MCPCallbackHook(lambda evt, det: print(f"ALERT: {evt}", det)))
        ```
    """

    @abstractmethod
    def on_event(self, event_type: str, details: dict) -> None:
        """Handle a security event.

        Args:
            event_type: One of the event type strings documented above.
            details: Structured dict with event-specific metadata.
        """


class MCPConsoleHook(MCPSecurityHook):
    """Print security alerts to stderr with ANSI colour coding.

    Args:
        use_color: Whether to emit ANSI colour codes.
            Defaults to ``True`` when stderr is a TTY, ``False`` otherwise.

    Example:
        ```python
        fw = MCPFirewall.preset("strict")
        fw.add_hook(MCPConsoleHook())
        ```
    """

    _COLORS: dict[str, str] = {
        "server_blocked":      "\033[1;31m",   # bold red
        "rug_pull_detected":   "\033[1;31m",   # bold red
        "tool_blocked":        "\033[1;33m",   # bold yellow
        "rate_limit_exceeded": "\033[1;33m",   # bold yellow
        "call_intercepted":    "\033[0;33m",   # yellow
    }
    _ICONS: dict[str, str] = {
        "server_blocked":      "[BLOCKED]",
        "rug_pull_detected":   "[RUG PULL]",
        "tool_blocked":        "[DENIED]",
        "rate_limit_exceeded": "[RATE LIMIT]",
        "call_intercepted":    "[INTERCEPTED]",
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool | None = None):
        import sys as _sys
        self._use_color = (
            _sys.stderr.isatty() if use_color is None else use_color
        )

    def on_event(self, event_type: str, details: dict) -> None:
        import sys as _sys
        icon = self._ICONS.get(event_type, "[ALERT]")
        color = self._COLORS.get(event_type, "") if self._use_color else ""
        reset = self._RESET if self._use_color else ""
        server = details.get("server_id", "?")
        reason = details.get("reason") or details.get("reasons") or event_type
        if isinstance(reason, list):
            reason = "; ".join(reason)
        ts = details.get("timestamp", "")[:19]
        print(
            f"{color}{icon} MCP Firewall | {ts} | {server} | {reason}{reset}",
            file=_sys.stderr,
        )


class MCPFileHook(MCPSecurityHook):
    """Append one JSON line per security event to a file.

    The output format matches the ``MCPAuditLogger`` style — each line is a
    self-contained JSON object that can be streamed to a SIEM or log aggregator.

    Args:
        path: File to append event lines to.

    Example:
        ```python
        fw = MCPFirewall.preset("strict")
        fw.add_hook(MCPFileHook("/var/log/mcp-alerts.jsonl"))
        ```
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def on_event(self, event_type: str, details: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps(details, ensure_ascii=True, default=str) + "\n")
        except Exception as exc:
            logger.warning("MCPFileHook: failed to write alert to '%s': %s", self._path, exc)


class MCPCallbackHook(MCPSecurityHook):
    """Wrap any Python callable as a security event hook.

    The callable is invoked with ``(event_type: str, details: dict)``.
    Any exception it raises is caught and logged — it will never crash the agent.

    Args:
        callback: Any callable accepting ``(event_type, details)``.

    Example:
        ```python
        import requests

        fw = MCPFirewall.preset("strict")
        fw.add_hook(MCPCallbackHook(
            lambda evt, det: requests.post("https://hooks.example.com/mcp", json=det)
        ))
        ```
    """

    def __init__(self, callback: Any):
        self._callback = callback

    def on_event(self, event_type: str, details: dict) -> None:
        self._callback(event_type, details)


# ---------------------------------------------------------------------------
# _dispatch_hooks — internal helper
# ---------------------------------------------------------------------------


def _dispatch_hooks(
    hooks: list[MCPSecurityHook] | None,
    event_type: str,
    server_id: str,
    extra: dict | None = None,
) -> None:
    """Fire all registered hooks for a security event.  Never raises.

    Args:
        hooks: Hook list from MCPFirewall (may be None or empty).
        event_type: One of the documented event type strings.
        server_id: The server that triggered the event.
        extra: Additional event-specific fields merged into the details dict.
    """
    if not hooks:
        return
    details: dict = {
        "event_type": event_type,
        "server_id": server_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        details.update(extra)
    for hook in hooks:
        try:
            hook.on_event(event_type, details)
        except Exception as exc:
            logger.warning(
                "MCPSecurityHook '%s' raised an exception: %s",
                type(hook).__name__,
                exc,
            )


# ---------------------------------------------------------------------------
# Internal hashing helpers (used by wrap_tool_with_guardian)
# ---------------------------------------------------------------------------


def _hash_args(kwargs: dict) -> str:
    """Short SHA-256 hex of canonical JSON of kwargs (stored in audit; never raw)."""
    try:
        canonical = _json.dumps(kwargs, sort_keys=True, default=str, ensure_ascii=True)
    except Exception:
        canonical = repr(kwargs)
    return _hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _hash_str(s: str) -> str:
    """Short SHA-256 hex of a string."""
    return _hashlib.sha256(s.encode(errors="replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# wrap_tool_with_guardian — wire sentinel + audit into a single tool
# ---------------------------------------------------------------------------


def wrap_tool_with_guardian(
    tool: Any,
    server_id: str,
    sentinel: MCPCallSentinel | None = None,
    audit_logger: MCPAuditLogger | None = None,
    trust_score: float | None = None,
    fingerprint_verified: bool = False,
    allowlist: MCPToolAllowlist | None = None,
    sanitizer: MCPResponseSanitizer | None = None,
    rate_limiter: MCPRateLimiter | None = None,
    hooks: list[MCPSecurityHook] | None = None,
    warn_mode: bool = False,
) -> None:
    """Wrap ``tool.forward()`` in-place with all active security layers.

    After wrapping, every call to ``tool()`` / ``tool.forward()`` will:

    1. Check ``allowlist.is_allowed()`` — raises ``MCPToolBlockedError`` if not approved.
    2. Check ``rate_limiter.check()`` — raises ``MCPRateLimitExceededError`` if over budget.
    3. Run ``sentinel.inspect_call_args()`` — raises ``MCPCallInterceptedError`` on threat.
    4. Execute the original ``forward()`` method.
    5. Run ``sentinel.inspect_response()`` — raises ``MCPCallInterceptedError`` on threat.
    6. Run ``sanitizer.sanitize_response()`` to strip PII from the response.
    7. Append a record to ``audit_logger``.
    8. Fire registered ``hooks`` for any block event.

    If all six optional parameters are ``None`` / empty, this function is a no-op.

    Args:
        tool: A smolagents Tool instance to wrap in place.
        server_id: Server identifier written into audit log records.
        sentinel: Optional ``MCPCallSentinel`` for pre/post call inspection.
        audit_logger: Optional ``MCPAuditLogger`` for call recording.
        trust_score: Trust score from pre-flight verification (audit records only).
        fingerprint_verified: Whether fingerprint check passed (audit records only).
        allowlist: Optional ``MCPToolAllowlist`` for per-call whitelist enforcement.
        sanitizer: Optional ``MCPResponseSanitizer`` to strip PII from responses.
        rate_limiter: Optional ``MCPRateLimiter`` for sliding-window call budgets.
        hooks: Optional list of ``MCPSecurityHook`` instances for real-time alerts.
        warn_mode: If ``True``, violations are logged as warnings and the call
            continues instead of raising.  A ``"violation_warned"`` hook event
            is fired with a ``violation_type`` field indicating what was detected.
    """
    if (sentinel is None and audit_logger is None and allowlist is None
            and sanitizer is None and rate_limiter is None and not hooks):
        return

    original_forward = tool.forward

    def _guardian_forward(**kwargs):
        args_hash = _hash_args(kwargs)
        t0 = _time.monotonic()

        # Allowlist pre-call check
        if allowlist is not None and not allowlist.is_allowed(server_id, tool.name):
            exc = MCPToolBlockedError(
                server_id=server_id,
                tool_name=tool.name,
                reason="tool is not on the allowlist",
            )
            if warn_mode:
                logger.warning("[WARN] MCPFirewall: %s", exc)
                _dispatch_hooks(hooks, "violation_warned", server_id, {
                    "tool_name": tool.name,
                    "violation_type": "tool_blocked",
                    "reason": exc.reason,
                })
            else:
                _dispatch_hooks(hooks, "tool_blocked", server_id, {
                    "tool_name": tool.name, "reason": exc.reason,
                })
                if audit_logger is not None:
                    audit_logger.log_call(
                        server_id=server_id,
                        tool_name=tool.name,
                        args_hash=args_hash,
                        response_hash="",
                        trust_score=trust_score,
                        fingerprint_verified=fingerprint_verified,
                        call_duration_ms=0.0,
                        blocked=True,
                        block_reason=str(exc),
                    )
                raise exc

        # Rate-limit pre-call check
        if rate_limiter is not None:
            try:
                rate_limiter.check(server_id, tool.name)
            except MCPRateLimitExceededError as exc:
                if warn_mode:
                    logger.warning("[WARN] MCPFirewall: %s", exc)
                    _dispatch_hooks(hooks, "violation_warned", server_id, {
                        "tool_name": tool.name,
                        "violation_type": "rate_limit_exceeded",
                        "limit": exc.limit,
                        "scope": exc.scope,
                        "reason": str(exc),
                    })
                else:
                    _dispatch_hooks(hooks, "rate_limit_exceeded", server_id, {
                        "tool_name": tool.name,
                        "limit": exc.limit,
                        "scope": exc.scope,
                        "reason": str(exc),
                    })
                    if audit_logger is not None:
                        audit_logger.log_call(
                            server_id=server_id,
                            tool_name=tool.name,
                            args_hash=args_hash,
                            response_hash="",
                            trust_score=trust_score,
                            fingerprint_verified=fingerprint_verified,
                            call_duration_ms=0.0,
                            blocked=True,
                            block_reason=str(exc),
                        )
                    raise

        # Sentinel pre-call inspection
        if sentinel is not None:
            try:
                sentinel.inspect_call_args(tool.name, kwargs)
            except MCPCallInterceptedError as exc:
                if warn_mode:
                    logger.warning("[WARN] MCPFirewall: %s", exc)
                    _dispatch_hooks(hooks, "violation_warned", server_id, {
                        "tool_name": tool.name,
                        "violation_type": "call_intercepted",
                        "phase": exc.phase,
                        "reason": exc.reason,
                    })
                else:
                    _dispatch_hooks(hooks, "call_intercepted", server_id, {
                        "tool_name": tool.name,
                        "phase": exc.phase,
                        "reason": exc.reason,
                    })
                    if audit_logger is not None:
                        audit_logger.log_call(
                            server_id=server_id,
                            tool_name=tool.name,
                            args_hash=args_hash,
                            response_hash="",
                            trust_score=trust_score,
                            fingerprint_verified=fingerprint_verified,
                            call_duration_ms=0.0,
                            blocked=True,
                            block_reason=str(exc),
                        )
                    raise

        # Execute the original forward
        response = original_forward(**kwargs)
        elapsed_ms = (_time.monotonic() - t0) * 1000
        response_hash = _hash_str(str(response))

        # Sentinel post-call inspection
        if sentinel is not None:
            try:
                sentinel.inspect_response(tool.name, response)
            except MCPCallInterceptedError as exc:
                if warn_mode:
                    logger.warning("[WARN] MCPFirewall: %s", exc)
                    _dispatch_hooks(hooks, "violation_warned", server_id, {
                        "tool_name": tool.name,
                        "violation_type": "call_intercepted",
                        "phase": exc.phase,
                        "reason": exc.reason,
                    })
                else:
                    _dispatch_hooks(hooks, "call_intercepted", server_id, {
                        "tool_name": tool.name,
                        "phase": exc.phase,
                        "reason": exc.reason,
                    })
                    if audit_logger is not None:
                        audit_logger.log_call(
                            server_id=server_id,
                            tool_name=tool.name,
                            args_hash=args_hash,
                            response_hash=response_hash,
                            trust_score=trust_score,
                            fingerprint_verified=fingerprint_verified,
                            call_duration_ms=elapsed_ms,
                            blocked=True,
                            block_reason=str(exc),
                        )
                    raise

        # PII sanitization — applied after inspection, before returning to LLM
        if sanitizer is not None:
            response = sanitizer.sanitize_response(response)

        # Successful call — audit
        if audit_logger is not None:
            audit_logger.log_call(
                server_id=server_id,
                tool_name=tool.name,
                args_hash=args_hash,
                response_hash=response_hash,
                trust_score=trust_score,
                fingerprint_verified=fingerprint_verified,
                call_duration_ms=elapsed_ms,
                blocked=False,
            )

        return response

    tool.forward = _guardian_forward


# ===========================================================================
# Phase 6 — MCPFirewall Facade
# ===========================================================================
# Unifies all five security layers behind a single, developer-friendly object.
# Provides built-in presets (strict / balanced / paranoid / dev) and a
# dict-based config loader for YAML/TOML/JSON-driven configuration.
# ===========================================================================


class MCPFirewall:
    """All-in-one security facade for MCP tool ingestion.

    Bundles TrustVerifier, MCPPayloadValidator, MCPToolFingerprinter,
    MCPCallSentinel, and MCPAuditLogger into a single object so you don't
    have to construct and wire each layer manually.

    **Quick start with a preset:**

    ```python
    from smolagents import MCPFirewall, MCPClient

    fw = MCPFirewall.preset("strict")
    with MCPClient(server_params, **fw.as_kwargs()) as tools:
        agent.run("task", tools=tools)
    ```

    **Fine-grained control:**

    ```python
    fw = MCPFirewall(
        trust_verifier=StaticTrustVerifier(require_https=True),
        payload_validator=MCPPayloadValidator(max_tools_per_server=10),
        fingerprinter=MCPToolFingerprinter(lockfile_path="./prod.mcp-lock.json"),
        sentinel=MCPCallSentinel(max_response_length=50_000),
        audit_logger=MCPAuditLogger(log_path="/var/log/mcp-audit.jsonl"),
    )
    with ToolCollection.from_mcp(server_params, trust_remote_code=True, **fw.as_kwargs()) as tc:
        ...
    ```

    **Dict / YAML config:**

    ```python
    import yaml
    config = yaml.safe_load(open(".smolagents-firewall.yml"))
    fw = MCPFirewall.from_config(config)
    ```

    Presets
    -------
    ``strict``
        All layers enabled.  HTTPS required.  Default limits.
        Suitable for production use against public MCP servers.

    ``balanced``
        All layers enabled.  HTTP allowed (dev / localhost servers ok).
        Good for staging or internal deployments.

    ``paranoid``
        All layers enabled.  HTTPS required.  Minimum trust score 0.85.
        Validator limits halved.  Sentinel response budget 10 000 chars.
        For highly sensitive workloads.

    ``dev``
        Trust verifier + payload validator only.  No fingerprinting or
        call inspection.  Audit logging enabled.  Suitable for local
        development where you own the MCP server.
    """

    PRESETS = ("strict", "balanced", "paranoid", "dev")

    def __init__(
        self,
        trust_verifier: TrustVerifier | None = None,
        payload_validator: MCPPayloadValidator | None = None,
        fingerprinter: MCPToolFingerprinter | None = None,
        sentinel: MCPCallSentinel | None = None,
        audit_logger: MCPAuditLogger | None = None,
        allowlist: MCPToolAllowlist | None = None,
        sanitizer: MCPResponseSanitizer | None = None,
        rate_limiter: MCPRateLimiter | None = None,
        hooks: list[MCPSecurityHook] | None = None,
        mode: str = "enforce",
    ):
        if mode not in ("enforce", "warn"):
            raise ValueError(
                f"MCPFirewall: mode must be 'enforce' or 'warn', got {mode!r}"
            )
        self.trust_verifier = trust_verifier
        self.payload_validator = payload_validator
        self.fingerprinter = fingerprinter
        self.sentinel = sentinel
        self.audit_logger = audit_logger
        self.allowlist = allowlist
        self.sanitizer = sanitizer
        self.rate_limiter = rate_limiter
        self.hooks: list[MCPSecurityHook] = list(hooks) if hooks else []
        self.mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_hook(self, hook: MCPSecurityHook) -> None:
        """Register a security event hook.

        Args:
            hook: An ``MCPSecurityHook`` instance to receive firewall events.
        """
        self.hooks.append(hook)

    def remove_hook(self, hook: MCPSecurityHook) -> None:
        """Unregister a previously added hook.  No-op if not registered.

        Args:
            hook: The hook instance to remove.
        """
        try:
            self.hooks.remove(hook)
        except ValueError:
            pass

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "MCPFirewall":
        """Return a pre-configured ``MCPFirewall`` for common use cases.

        Args:
            name: One of ``"strict"``, ``"balanced"``, ``"paranoid"``, ``"dev"``.
            **overrides: Override specific components after the preset is built.
                Pass ``None`` to disable a layer (e.g. ``fingerprinter=None``).

        Returns:
            A fully configured ``MCPFirewall`` instance.

        Raises:
            ValueError: If ``name`` is not a known preset.
        """
        _factories: dict[str, Any] = {
            "strict":   cls._make_strict,
            "balanced": cls._make_balanced,
            "paranoid": cls._make_paranoid,
            "dev":      cls._make_dev,
        }
        if name not in _factories:
            raise ValueError(
                f"Unknown preset '{name}'. Available presets: "
                + ", ".join(sorted(_factories))
            )
        instance = _factories[name]()
        for attr, value in overrides.items():
            if not hasattr(instance, attr):
                raise ValueError(f"MCPFirewall has no attribute '{attr}'")
            setattr(instance, attr, value)
        return instance

    @classmethod
    def from_config(cls, config: dict) -> "MCPFirewall":
        """Build an ``MCPFirewall`` from a configuration dictionary.

        Designed to work with dictionaries loaded from YAML, TOML, or JSON
        files.  Start from a named preset with ``preset:`` and override
        individual layers as needed.

        Supported top-level keys
        ------------------------
        ``preset`` (str):
            Optional.  Start from ``"strict"``, ``"balanced"``, ``"paranoid"``,
            or ``"dev"``.  Defaults to an empty firewall if omitted.
        ``mode`` (str):
            ``"enforce"`` (default) raises on violations; ``"warn"`` logs a
            warning and fires hooks but lets the call through.
        ``trust_verifier`` (dict | ``false``):
            Keys: ``require_https`` (bool), ``min_trust_score`` (float),
            ``blocklist`` (list[str]), ``allowlist`` (list[str]).
            Set to ``false`` to disable.
        ``payload_validator`` (dict | ``false``):
            Keys: ``max_tool_name_length``, ``max_description_length``,
            ``max_tools_per_server``, ``max_input_params``,
            ``max_param_description_length``, ``extra_injection_patterns``
            (list[str] — additional regex patterns scanned in tool metadata).
        ``fingerprinter`` (dict | bool | ``false``):
            Keys: ``lockfile_path`` (str).
        ``sentinel`` (dict | bool | ``false``):
            Keys: ``max_response_length`` (int),
            ``block_credential_exfil`` (bool), ``block_sensitive_paths`` (bool),
            ``extra_blocked_arg_patterns`` (list[str] — additional regex patterns
            scanned in tool call arguments),
            ``extra_blocked_response_patterns`` (list[str] — additional regex
            patterns scanned in tool responses).
        ``audit_logger`` (dict | bool | ``false``):
            Keys: ``log_path`` (str).
        ``allowlist`` (dict | ``false``):
            Keys: ``allowlist_path`` (str), ``auto_approve_first_connection``
            (bool).
        ``sanitizer`` (dict | ``false``):
            Keys: ``redact_emails``, ``redact_credit_cards``, ``redact_ssn``,
            ``redact_phone_numbers``, ``redact_jwt`` (all bool), ``mode``
            (``"redact"`` | ``"hash"`` | ``"drop"``), ``custom_patterns``
            (list of ``{name: str, pattern: str}`` dicts).
        ``rate_limiter`` (dict | ``false``):
            Keys: ``max_calls_per_minute`` (int), ``window_seconds`` (float),
            ``per_tool_max_calls_per_minute`` (dict[str, int]).
        ``hooks`` (dict):
            Keys: ``console`` (bool — emit to stderr), ``file`` (str — path
            to append security-event JSONL records).

        Example YAML::

            preset: balanced
            trust_verifier:
              require_https: true
              blocklist:
                - "untrusted\\\\.example\\\\.com"
            fingerprinter:
              lockfile_path: /var/run/mcp-lock.json
            sentinel:
              max_response_length: 50000
            audit_logger:
              log_path: /var/log/mcp-audit.jsonl

        Args:
            config: Nested dictionary of configuration values.

        Returns:
            A configured ``MCPFirewall`` instance.
        """
        preset_name = config.get("preset")
        mode = config.get("mode", "enforce")
        if mode not in ("enforce", "warn"):
            raise ValueError(
                f"MCPFirewall.from_config: mode must be 'enforce' or 'warn', got {mode!r}"
            )
        instance: MCPFirewall = cls.preset(preset_name) if preset_name else cls()
        instance.mode = mode

        def _apply(key: str, builder):
            val = config.get(key)
            if val is None:
                return  # not present — keep whatever the preset set
            if val is False:
                setattr(instance, key, None)
                return
            if val is True:
                setattr(instance, key, builder({}))
                return
            if isinstance(val, dict):
                setattr(instance, key, builder(val))
                return
            raise ValueError(f"MCPFirewall.from_config: unexpected value type for '{key}': {type(val)}")

        _apply("trust_verifier", lambda d: StaticTrustVerifier(
            require_https=d.get("require_https", True),
            min_trust_score=d.get("min_trust_score", 0.5),
            blocklist=d.get("blocklist"),
            allowlist=d.get("allowlist"),
        ))
        _apply("payload_validator", lambda d: MCPPayloadValidator(
            max_tool_name_length=d.get("max_tool_name_length", 64),
            max_description_length=d.get("max_description_length", 4096),
            max_tools_per_server=d.get("max_tools_per_server", 100),
            max_input_params=d.get("max_input_params", 20),
            max_param_description_length=d.get("max_param_description_length", 1024),
            extra_injection_patterns=d.get("extra_injection_patterns"),
        ))
        _apply("fingerprinter", lambda d: MCPToolFingerprinter(
            lockfile_path=d.get("lockfile_path"),
        ))
        _apply("sentinel", lambda d: MCPCallSentinel(
            max_response_length=d.get("max_response_length", 100_000),
            block_credential_exfil=d.get("block_credential_exfil", True),
            block_sensitive_paths=d.get("block_sensitive_paths", True),
            extra_blocked_arg_patterns=d.get("extra_blocked_arg_patterns"),
            extra_blocked_response_patterns=d.get("extra_blocked_response_patterns"),
        ))
        _apply("audit_logger", lambda d: MCPAuditLogger(
            log_path=d.get("log_path"),
        ))
        _apply("allowlist", lambda d: MCPToolAllowlist(
            allowlist_path=d.get("allowlist_path"),
            auto_approve_first_connection=d.get("auto_approve_first_connection", True),
        ))
        def _build_sanitizer(d: dict) -> "MCPResponseSanitizer":
            raw_custom = d.get("custom_patterns") or []
            compiled_custom: list[tuple[str, re.Pattern[str]]] = [
                (entry["name"], re.compile(entry["pattern"]))
                for entry in raw_custom
                if isinstance(entry, dict) and "name" in entry and "pattern" in entry
            ]
            return MCPResponseSanitizer(
                redact_emails=d.get("redact_emails", True),
                redact_credit_cards=d.get("redact_credit_cards", True),
                redact_ssn=d.get("redact_ssn", True),
                redact_phone_numbers=d.get("redact_phone_numbers", True),
                redact_jwt=d.get("redact_jwt", True),
                mode=d.get("mode", "redact"),
                custom_patterns=compiled_custom or None,
            )
        _apply("sanitizer", _build_sanitizer)
        _apply("rate_limiter", lambda d: MCPRateLimiter(
            max_calls_per_minute=d.get("max_calls_per_minute", 300),
            per_tool_max_calls_per_minute=d.get("per_tool_max_calls_per_minute"),
            window_seconds=d.get("window_seconds", 60.0),
        ))
        # Hooks: special handling — supports dict with "console" and/or "file" keys
        hooks_conf = config.get("hooks")
        if isinstance(hooks_conf, dict):
            if hooks_conf.get("console"):
                instance.add_hook(MCPConsoleHook())
            if "file" in hooks_conf and hooks_conf["file"]:
                instance.add_hook(MCPFileHook(hooks_conf["file"]))
        return instance

    # ------------------------------------------------------------------
    # Phase 11 — Config file loaders / serialiser
    # ------------------------------------------------------------------

    def _to_config_dict(self) -> dict:
        """Serialize current firewall state to a plain config dictionary.

        The result is compatible with :meth:`from_config`, :meth:`from_yaml`,
        :meth:`save_yaml`, and :meth:`save_json`.  Only ``StaticTrustVerifier`` is fully
        round-tripped; custom ``TrustVerifier`` subclasses are skipped.

        Returns:
            A nested dict representing all 8 layers.
        """
        config: dict = {"mode": self.mode}

        # trust_verifier
        if self.trust_verifier is None:
            config["trust_verifier"] = False
        elif isinstance(self.trust_verifier, StaticTrustVerifier):
            tv = self.trust_verifier
            tv_dict: dict = {
                "require_https": tv.require_https,
                "min_trust_score": tv.min_trust_score,
            }
            if tv._blocklist:
                tv_dict["blocklist"] = [p.pattern for p in tv._blocklist]
            if tv._allowlist is not None:
                tv_dict["allowlist"] = [p.pattern for p in tv._allowlist]
            config["trust_verifier"] = tv_dict

        # payload_validator
        if self.payload_validator is None:
            config["payload_validator"] = False
        else:
            pv = self.payload_validator
            pv_dict: dict = {
                "max_tool_name_length": pv.max_tool_name_length,
                "max_description_length": pv.max_description_length,
                "max_tools_per_server": pv.max_tools_per_server,
                "max_input_params": pv.max_input_params,
                "max_param_description_length": pv.max_param_description_length,
            }
            if pv.extra_injection_patterns:
                pv_dict["extra_injection_patterns"] = list(pv.extra_injection_patterns)
            config["payload_validator"] = pv_dict

        # fingerprinter
        if self.fingerprinter is None:
            config["fingerprinter"] = False
        else:
            config["fingerprinter"] = {
                "lockfile_path": str(self.fingerprinter._lockfile_path),
            }

        # sentinel
        if self.sentinel is None:
            config["sentinel"] = False
        else:
            sentinel_dict: dict = {
                "max_response_length": self.sentinel.max_response_length,
                "block_credential_exfil": self.sentinel.block_credential_exfil,
                "block_sensitive_paths": self.sentinel.block_sensitive_paths,
            }
            if self.sentinel._extra_arg_patterns:
                sentinel_dict["extra_blocked_arg_patterns"] = [
                    p.pattern for _, p in self.sentinel._extra_arg_patterns
                ]
            if self.sentinel._extra_response_patterns:
                sentinel_dict["extra_blocked_response_patterns"] = [
                    p.pattern for _, p in self.sentinel._extra_response_patterns
                ]
            config["sentinel"] = sentinel_dict

        # audit_logger
        if self.audit_logger is None:
            config["audit_logger"] = False
        else:
            config["audit_logger"] = {
                "log_path": str(self.audit_logger._log_path),
            }

        # allowlist
        if self.allowlist is None:
            config["allowlist"] = False
        else:
            config["allowlist"] = {
                "allowlist_path": str(self.allowlist._allowlist_path),
                "auto_approve_first_connection": self.allowlist.auto_approve_first_connection,
            }

        # sanitizer — infer redact_* from the active-patterns list
        if self.sanitizer is None:
            config["sanitizer"] = False
        else:
            _STANDARD_SANITIZER_NAMES = {"email", "credit-card", "ssn", "phone", "jwt"}
            active_names = {name for name, _ in self.sanitizer._active}
            sanitizer_dict: dict = {
                "mode": self.sanitizer.mode,
                "redact_emails": "email" in active_names,
                "redact_credit_cards": "credit-card" in active_names,
                "redact_ssn": "ssn" in active_names,
                "redact_phone_numbers": "phone" in active_names,
                "redact_jwt": "jwt" in active_names,
            }
            custom_pats = [
                {"name": name, "pattern": pat.pattern}
                for name, pat in self.sanitizer._active
                if name not in _STANDARD_SANITIZER_NAMES
            ]
            if custom_pats:
                sanitizer_dict["custom_patterns"] = custom_pats
            config["sanitizer"] = sanitizer_dict

        # rate_limiter
        if self.rate_limiter is None:
            config["rate_limiter"] = False
        else:
            rl = self.rate_limiter
            rl_dict: dict = {
                "max_calls_per_minute": rl.max_calls_per_minute,
                "window_seconds": rl.window_seconds,
            }
            if rl.per_tool_max_calls_per_minute is not None:
                rl_dict["per_tool_max_calls_per_minute"] = rl.per_tool_max_calls_per_minute
            config["rate_limiter"] = rl_dict

        # hooks — MCPConsoleHook and MCPFileHook are serialisable; others are not
        hooks_conf: dict = {}
        for hook in self.hooks:
            if isinstance(hook, MCPConsoleHook):
                hooks_conf["console"] = True
            elif isinstance(hook, MCPFileHook):
                hooks_conf["file"] = str(hook._path)
        if hooks_conf:
            config["hooks"] = hooks_conf

        return config

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "MCPFirewall":
        """Load an ``MCPFirewall`` from a YAML config file.

        Requires ``pyyaml``: ``pip install pyyaml``.

        Args:
            path: Path to the ``.yml`` / ``.yaml`` config file.

        Returns:
            A configured ``MCPFirewall`` instance.

        Raises:
            ImportError: If ``pyyaml`` is not installed.
            FileNotFoundError: If ``path`` does not exist.
        """
        try:
            import yaml as _yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required to use MCPFirewall.from_yaml(). "
                "Install it with: pip install pyyaml"
            )
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            config = _yaml.safe_load(fh)
        return cls.from_config(config or {})

    @classmethod
    def from_toml(cls, path: "str | Path") -> "MCPFirewall":
        """Load an ``MCPFirewall`` from a TOML config file.

        Uses Python 3.11+ built-in ``tomllib``; falls back to ``tomli`` on
        older interpreters.

        Args:
            path: Path to the ``.toml`` config file.

        Returns:
            A configured ``MCPFirewall`` instance.

        Raises:
            ImportError: If neither ``tomllib`` nor ``tomli`` is available.
            FileNotFoundError: If ``path`` does not exist.
        """
        try:
            import tomllib as _toml
        except ImportError:
            try:
                import tomli as _toml  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "tomli is required to use MCPFirewall.from_toml() on Python < 3.11. "
                    "Install it with: pip install tomli"
                )
        path = Path(path)
        with open(path, "rb") as fh:
            config = _toml.load(fh)
        return cls.from_config(config)

    @classmethod
    def from_json(cls, path: "str | Path") -> "MCPFirewall":
        """Load an ``MCPFirewall`` from a JSON config file.

        Args:
            path: Path to the ``.json`` config file.

        Returns:
            A configured ``MCPFirewall`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            config = _json.load(fh)
        return cls.from_config(config)

    def save_yaml(self, path: "str | Path") -> None:
        """Serialize the current firewall configuration to a YAML file.

        Requires ``pyyaml``: ``pip install pyyaml``.  The output is reloadable
        with :meth:`from_yaml`.  Intermediate directories are created
        automatically.

        Args:
            path: Destination file path.  Overwrites any existing file.

        Raises:
            ImportError: If ``pyyaml`` is not installed.
        """
        try:
            import yaml as _yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required to use MCPFirewall.save_yaml(). "
                "Install it with: pip install pyyaml"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self._to_config_dict()
        with open(path, "w", encoding="utf-8") as fh:
            _yaml.dump(config, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def save_json(self, path: "str | Path", *, indent: int = 2) -> None:
        """Serialize the current firewall configuration to a JSON file.

        Uses the Python standard library — no extra dependencies required.
        The output is reloadable with :meth:`from_json`.  Intermediate
        directories are created automatically.

        Args:
            path: Destination file path.  Overwrites any existing file.
            indent: JSON indentation level (default 2).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self._to_config_dict()
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(config, fh, indent=indent, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def from_env(cls, prefix: str = "MCP_FIREWALL_") -> "MCPFirewall":
        """Build an ``MCPFirewall`` from environment variables.

        Reads environment variables with the given ``prefix`` and constructs a
        config dict that is then passed to :meth:`from_config`.  Variables not
        present in the environment are silently ignored — only variables that
        are explicitly set have any effect.

        Supported variables (shown with the default ``MCP_FIREWALL_`` prefix)
        -----------------------------------------------------------------------
        ``MCP_FIREWALL_PRESET``
            Preset base: ``strict``, ``balanced``, ``paranoid``, or ``dev``.
        ``MCP_FIREWALL_MODE``
            Enforcement mode: ``enforce`` (default) or ``warn``.
        ``MCP_FIREWALL_REQUIRE_HTTPS``
            Require HTTPS for remote servers: ``true`` / ``false``.
        ``MCP_FIREWALL_MIN_TRUST_SCORE``
            Minimum trust score threshold, 0.0–1.0 (float string).
        ``MCP_FIREWALL_BLOCKLIST``
            Comma-separated regex patterns added to the trust verifier blocklist.
        ``MCP_FIREWALL_MAX_RESPONSE_LENGTH``
            Maximum tool response length before the sentinel blocks the call (int).
        ``MCP_FIREWALL_AUDIT_LOG``
            File path for the structured JSONL audit log.
        ``MCP_FIREWALL_LOCKFILE``
            File path for the rug-pull fingerprint lockfile.
        ``MCP_FIREWALL_RATE_LIMIT``
            Maximum MCP tool calls per minute per server (int).
        ``MCP_FIREWALL_HOOK_CONSOLE``
            Emit security events to stderr: ``true`` / ``false``.
        ``MCP_FIREWALL_HOOK_FILE``
            File path to append security event JSONL records.
        ``MCP_FIREWALL_BLOCKED_ARG_PATTERNS``
            Comma-separated regex strings added to the sentinel's extra arg blocked patterns.
        ``MCP_FIREWALL_BLOCKED_RESPONSE_PATTERNS``
            Comma-separated regex strings added to the sentinel's extra response blocked patterns.

        .. note::
            When a layer is partially configured via env vars and a ``PRESET``
            is also set, the env-var values take precedence but the layer is
            rebuilt from its constructor defaults for any value not explicitly
            specified.

        Args:
            prefix: Environment variable prefix.  Defaults to
                ``"MCP_FIREWALL_"``.

        Returns:
            A configured ``MCPFirewall`` instance.

        Raises:
            ValueError: If a variable contains an unparseable value (e.g. a
                non-numeric string for an integer variable, or an unrecognised
                boolean string).

        Example::

            # In the shell:
            #   export MCP_FIREWALL_PRESET=strict
            #   export MCP_FIREWALL_MODE=warn
            #   export MCP_FIREWALL_HOOK_CONSOLE=true

            fw = MCPFirewall.from_env()
            with MCPClient(server_params, **fw.as_kwargs()) as tools:
                ...
        """
        import os as _os

        def _bool(val: str, var: str) -> bool:
            if val.lower() in ("1", "true", "yes", "on"):
                return True
            if val.lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError(
                f"Cannot parse boolean from {var}={val!r}; "
                "expected one of: true, false, 1, 0, yes, no, on, off"
            )

        def _int(val: str, var: str) -> int:
            try:
                return int(val)
            except ValueError:
                raise ValueError(
                    f"Cannot parse integer from {var}={val!r}"
                )

        def _float(val: str, var: str) -> float:
            try:
                return float(val)
            except ValueError:
                raise ValueError(
                    f"Cannot parse float from {var}={val!r}"
                )

        env = _os.environ
        config: dict = {}

        # Preset
        if preset := env.get(f"{prefix}PRESET"):
            config["preset"] = preset

        # Mode
        if mode := env.get(f"{prefix}MODE"):
            config["mode"] = mode

        # Trust verifier
        tv: dict = {}
        if (v := env.get(f"{prefix}REQUIRE_HTTPS")) is not None:
            tv["require_https"] = _bool(v, f"{prefix}REQUIRE_HTTPS")
        if (v := env.get(f"{prefix}MIN_TRUST_SCORE")) is not None:
            tv["min_trust_score"] = _float(v, f"{prefix}MIN_TRUST_SCORE")
        if (v := env.get(f"{prefix}BLOCKLIST")) is not None:
            tv["blocklist"] = [p.strip() for p in v.split(",") if p.strip()]
        if tv:
            config["trust_verifier"] = tv

        # Sentinel
        sentinel: dict = {}
        if (v := env.get(f"{prefix}MAX_RESPONSE_LENGTH")) is not None:
            sentinel["max_response_length"] = _int(v, f"{prefix}MAX_RESPONSE_LENGTH")
        if (v := env.get(f"{prefix}BLOCKED_ARG_PATTERNS")) is not None:
            sentinel["extra_blocked_arg_patterns"] = [p.strip() for p in v.split(",") if p.strip()]
        if (v := env.get(f"{prefix}BLOCKED_RESPONSE_PATTERNS")) is not None:
            sentinel["extra_blocked_response_patterns"] = [p.strip() for p in v.split(",") if p.strip()]
        if sentinel:
            config["sentinel"] = sentinel

        # Audit logger
        if v := env.get(f"{prefix}AUDIT_LOG"):
            config["audit_logger"] = {"log_path": v}

        # Fingerprinter
        if v := env.get(f"{prefix}LOCKFILE"):
            config["fingerprinter"] = {"lockfile_path": v}

        # Rate limiter
        if (v := env.get(f"{prefix}RATE_LIMIT")) is not None:
            config["rate_limiter"] = {
                "max_calls_per_minute": _int(v, f"{prefix}RATE_LIMIT"),
            }

        # Hooks
        hooks: dict = {}
        if (v := env.get(f"{prefix}HOOK_CONSOLE")) is not None:
            hooks["console"] = _bool(v, f"{prefix}HOOK_CONSOLE")
        if v := env.get(f"{prefix}HOOK_FILE"):
            hooks["file"] = v
        if hooks:
            config["hooks"] = hooks

        return cls.from_config(config)

    def as_kwargs(self) -> dict:
        """Return a dict of keyword arguments for ``MCPClient`` or ``ToolCollection.from_mcp()``.

        The returned dict contains all security-layer parameters plus ``warn_mode``.
        Merge it with your other constructor arguments using ``**``:

        ```python
        fw = MCPFirewall.preset("strict")
        with MCPClient(server_params, structured_output=False, **fw.as_kwargs()) as tools:
            ...
        ```
        """
        return {
            "trust_verifier":    self.trust_verifier,
            "payload_validator": self.payload_validator,
            "fingerprinter":     self.fingerprinter,
            "sentinel":          self.sentinel,
            "audit_logger":      self.audit_logger,
            "allowlist":         self.allowlist,
            "sanitizer":         self.sanitizer,
            "rate_limiter":      self.rate_limiter,
            "hooks":             self.hooks or None,
            "warn_mode":         self.mode == "warn",
        }

    def summary(self) -> str:
        """Return a human-readable summary of which security layers are enabled."""
        parts: list[str] = []
        if self.trust_verifier is not None:
            tv = self.trust_verifier
            https = getattr(tv, "require_https", "?")
            score = getattr(tv, "min_trust_score", "?")
            parts.append(f"  TrustVerifier    : {type(tv).__name__}(require_https={https}, min_score={score})")
        else:
            parts.append("  TrustVerifier    : DISABLED")

        if self.payload_validator is not None:
            pv = self.payload_validator
            parts.append(
                f"  PayloadValidator : max_desc={pv.max_description_length}, "
                f"max_tools={pv.max_tools_per_server}"
            )
        else:
            parts.append("  PayloadValidator : DISABLED")

        if self.fingerprinter is not None:
            parts.append(f"  Fingerprinter    : lockfile={self.fingerprinter._lockfile_path}")
        else:
            parts.append("  Fingerprinter    : DISABLED")

        if self.sentinel is not None:
            parts.append(f"  Sentinel         : max_response={self.sentinel.max_response_length}")
        else:
            parts.append("  Sentinel         : DISABLED")

        if self.audit_logger is not None:
            parts.append(f"  AuditLogger      : path={self.audit_logger._log_path}")
        else:
            parts.append("  AuditLogger      : DISABLED")

        if self.allowlist is not None:
            mode = "auto-approve" if self.allowlist.auto_approve_first_connection else "strict"
            parts.append(f"  Allowlist        : path={self.allowlist._allowlist_path} ({mode})")
        else:
            parts.append("  Allowlist        : DISABLED")

        if self.sanitizer is not None:
            parts.append(f"  Sanitizer        : mode={self.sanitizer.mode}")
        else:
            parts.append("  Sanitizer        : DISABLED")

        if self.rate_limiter is not None:
            rl = self.rate_limiter
            tool_cap = (
                f", per_tool={rl.per_tool_max_calls_per_minute}/min"
                if rl.per_tool_max_calls_per_minute is not None else ""
            )
            parts.append(
                f"  RateLimiter      : {rl.max_calls_per_minute}/min per server"
                f"{tool_cap}, window={rl.window_seconds:.0f}s"
            )
        else:
            parts.append("  RateLimiter      : DISABLED")

        n_hooks = len(self.hooks)
        if n_hooks:
            names = ", ".join(type(h).__name__ for h in self.hooks[:3])
            suffix = ", ..." if n_hooks > 3 else ""
            parts.append(f"  Hooks            : {n_hooks} registered ({names}{suffix})")
        else:
            parts.append("  Hooks            : none")

        enabled = sum(1 for x in (
            self.trust_verifier, self.payload_validator,
            self.fingerprinter, self.sentinel, self.audit_logger,
            self.allowlist, self.sanitizer, self.rate_limiter,
        ) if x is not None)
        mode_label = "WARN (audit-only)" if self.mode == "warn" else "ENFORCE"
        header = f"MCPFirewall ({enabled}/8 layers active, mode={mode_label}):\n"
        return header + "\n".join(parts)

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------
    # Phase 14 — Config diff
    # ------------------------------------------------------------------

    def diff(self, other: "MCPFirewall") -> str:
        """Return a human-readable diff of two firewall configurations.

        Compares the serialisable state of both instances (via
        :meth:`_to_config_dict`) and reports every setting that differs.

        .. note::
            Custom ``TrustVerifier`` subclasses are not fully serialisable and
            will show as absent in the diff — the same limitation as
            :meth:`save_yaml` and :meth:`save_json`.

        Args:
            other: The firewall configuration to compare against.

        Returns:
            A multi-line string starting with ``"MCPFirewall diff:\\n"`` when
            differences exist, or ``""`` when the configurations are identical.

        Example::

            fw_old = MCPFirewall.preset("balanced")
            fw_new = MCPFirewall.from_yaml(".smolagents-firewall.yml")
            print(fw_old.diff(fw_new))
        """
        def _fmt(val: Any) -> str:
            # _fmt is only called for sub-key values; the top-level False
            # (layer disabled) is handled directly in the loop below.
            if val is None:
                return "None"
            if isinstance(val, list):
                return "[" + ", ".join(str(v) for v in val) + "]"
            return str(val)

        left = self._to_config_dict()
        right = other._to_config_dict()
        left.setdefault("hooks", {})
        right.setdefault("hooks", {})

        key_order = [
            "mode",
            "trust_verifier", "payload_validator", "fingerprinter",
            "sentinel", "audit_logger", "allowlist", "sanitizer",
            "rate_limiter", "hooks",
        ]

        lines: list[str] = []
        for key in key_order:
            lv = left.get(key)
            rv = right.get(key)
            if lv == rv:
                continue
            if isinstance(lv, dict) and isinstance(rv, dict):
                for sub in sorted(set(lv) | set(rv)):
                    ls, rs = lv.get(sub), rv.get(sub)
                    if ls != rs:
                        lines.append(f"  {key}.{sub}: {_fmt(ls)} → {_fmt(rs)}")
            elif lv is False and isinstance(rv, dict):
                lines.append(f"  {key}: DISABLED → enabled")
            elif isinstance(lv, dict) and rv is False:
                lines.append(f"  {key}: enabled → DISABLED")
            else:
                lines.append(f"  {key}: {_fmt(lv)} → {_fmt(rv)}")

        if not lines:
            return ""
        return "MCPFirewall diff:\n" + "\n".join(lines)

    @classmethod
    def diff_presets(cls, a: str, b: str) -> str:
        """Return a human-readable diff between two named presets.

        Equivalent to ``MCPFirewall.preset(a).diff(MCPFirewall.preset(b))``.

        Args:
            a: Name of the left preset (``"strict"``, ``"balanced"``,
                ``"paranoid"``, or ``"dev"``).
            b: Name of the right preset.

        Returns:
            Diff string, or ``""`` if the presets produce identical configs.
        """
        return cls.preset(a).diff(cls.preset(b))

    # ------------------------------------------------------------------
    # Phase 16 — Firewall composition (merge)
    # ------------------------------------------------------------------

    @classmethod
    def merge(cls, base: "MCPFirewall", override: "MCPFirewall") -> "MCPFirewall":
        """Compose two firewalls: *override*'s layers take precedence over *base*.

        For each of the 8 security layers, the override's value is used when it
        is not ``None``; otherwise the base layer is kept.  ``mode`` always
        comes from the override.  Hooks from both firewalls are unioned and
        deduplicated: at most one ``MCPConsoleHook``, one ``MCPFileHook`` per
        unique path, and callback hooks deduplicated by identity.

        Args:
            base: The baseline firewall (e.g. a preset).
            override: The firewall whose layers take precedence (e.g. from env
                or a project-specific config file).

        Returns:
            A new ``MCPFirewall`` with the merged configuration.

        Example::

            base = MCPFirewall.preset("strict")
            project = MCPFirewall.from_yaml("project-firewall.yml")
            fw = MCPFirewall.merge(base, project)
        """
        merged = cls(
            trust_verifier    = override.trust_verifier    if override.trust_verifier    is not None else base.trust_verifier,
            payload_validator = override.payload_validator if override.payload_validator is not None else base.payload_validator,
            fingerprinter     = override.fingerprinter     if override.fingerprinter     is not None else base.fingerprinter,
            sentinel          = override.sentinel          if override.sentinel          is not None else base.sentinel,
            audit_logger      = override.audit_logger      if override.audit_logger      is not None else base.audit_logger,
            allowlist         = override.allowlist         if override.allowlist         is not None else base.allowlist,
            sanitizer         = override.sanitizer         if override.sanitizer         is not None else base.sanitizer,
            rate_limiter      = override.rate_limiter      if override.rate_limiter      is not None else base.rate_limiter,
            mode              = override.mode,
        )
        seen_console = False
        seen_file_paths: set[str] = set()
        seen_cb_ids: set[int] = set()
        for hook in list(base.hooks) + list(override.hooks):
            if isinstance(hook, MCPConsoleHook):
                if not seen_console:
                    merged.add_hook(hook)
                    seen_console = True
            elif isinstance(hook, MCPFileHook):
                path_key = str(hook._path)
                if path_key not in seen_file_paths:
                    merged.add_hook(hook)
                    seen_file_paths.add(path_key)
            elif isinstance(hook, MCPCallbackHook):
                cb_id = id(hook._callback)
                if cb_id not in seen_cb_ids:
                    merged.add_hook(hook)
                    seen_cb_ids.add(cb_id)
        return merged

    @classmethod
    def merge_from_config(cls, base_config: dict, override_config: dict) -> "MCPFirewall":
        """Merge two config dicts and return the composed firewall.

        Equivalent to ``MCPFirewall.merge(from_config(base), from_config(override))``.

        Args:
            base_config: Base config dictionary (same format as :meth:`from_config`).
            override_config: Override config dictionary.

        Returns:
            A new ``MCPFirewall`` with the merged configuration.
        """
        return cls.merge(cls.from_config(base_config), cls.from_config(override_config))

    # ------------------------------------------------------------------
    # Preset factories (private)
    # ------------------------------------------------------------------

    @classmethod
    def _make_strict(cls) -> "MCPFirewall":
        return cls(
            trust_verifier=StaticTrustVerifier(require_https=True),
            payload_validator=MCPPayloadValidator(),
            fingerprinter=MCPToolFingerprinter(),
            sentinel=MCPCallSentinel(),
            audit_logger=MCPAuditLogger(),
            allowlist=None,
            sanitizer=MCPResponseSanitizer(),
            rate_limiter=MCPRateLimiter(
                max_calls_per_minute=300,
                per_tool_max_calls_per_minute=60,
            ),
        )

    @classmethod
    def _make_balanced(cls) -> "MCPFirewall":
        return cls(
            trust_verifier=StaticTrustVerifier(require_https=False),
            payload_validator=MCPPayloadValidator(),
            fingerprinter=MCPToolFingerprinter(),
            sentinel=MCPCallSentinel(),
            audit_logger=MCPAuditLogger(),
            allowlist=None,
            sanitizer=MCPResponseSanitizer(),
            rate_limiter=MCPRateLimiter(
                max_calls_per_minute=600,
                per_tool_max_calls_per_minute=120,
            ),
        )

    @classmethod
    def _make_paranoid(cls) -> "MCPFirewall":
        return cls(
            trust_verifier=StaticTrustVerifier(require_https=True, min_trust_score=0.85),
            payload_validator=MCPPayloadValidator(
                max_description_length=1_024,
                max_tools_per_server=20,
                max_input_params=10,
                max_param_description_length=256,
            ),
            fingerprinter=MCPToolFingerprinter(),
            sentinel=MCPCallSentinel(max_response_length=10_000),
            audit_logger=MCPAuditLogger(),
            allowlist=MCPToolAllowlist(auto_approve_first_connection=True),
            sanitizer=MCPResponseSanitizer(),
            rate_limiter=MCPRateLimiter(
                max_calls_per_minute=60,
                per_tool_max_calls_per_minute=20,
            ),
        )

    @classmethod
    def _make_dev(cls) -> "MCPFirewall":
        return cls(
            trust_verifier=StaticTrustVerifier(require_https=False, min_trust_score=0.0),
            payload_validator=MCPPayloadValidator(),
            fingerprinter=None,
            sentinel=None,
            audit_logger=MCPAuditLogger(),
            allowlist=None,
            sanitizer=None,
            rate_limiter=None,
        )


# ===========================================================================
# Phase 7 — MCPSecurityReport (Audit Log Analytics)
# ===========================================================================


@dataclass
class MCPCallStats:
    """Aggregated statistics computed from an MCPAuditLogger JSONL file.

    Attributes:
        total_calls: Total number of tool calls recorded.
        blocked_calls: Number of calls intercepted by the sentinel.
        block_rate: Fraction of calls that were blocked (0.0–1.0).
        unique_servers: Number of distinct server IDs observed.
        unique_tools: Number of distinct tool names observed.
        calls_by_server: Dict mapping server_id → call count (descending).
        calls_by_tool: Dict mapping tool_name → call count (descending).
        blocked_by_pattern: Dict mapping attack pattern name → block count (descending).
        avg_duration_ms: Mean wall-clock time of successful calls.
        first_call_at: ISO 8601 timestamp of the earliest record, or None.
        last_call_at: ISO 8601 timestamp of the most recent record, or None.
    """

    total_calls: int
    blocked_calls: int
    block_rate: float
    unique_servers: int
    unique_tools: int
    calls_by_server: dict
    calls_by_tool: dict
    blocked_by_pattern: dict
    avg_duration_ms: float
    first_call_at: str | None
    last_call_at: str | None


class MCPSecurityReport:
    """Reads an ``MCPAuditLogger`` JSONL file and computes security analytics.

    Args:
        log_path: Path to the JSONL audit log.  Defaults to
            ``~/.smolagents/audit.jsonl`` (the MCPAuditLogger default).

    Example:
        ```python
        from smolagents import MCPSecurityReport

        report = MCPSecurityReport()
        stats = report.generate()
        print(f"Blocked {stats.blocked_calls}/{stats.total_calls} calls ({stats.block_rate:.1%})")
        report.print_summary()
        ```
    """

    def __init__(self, log_path: str | Path | None = None):
        self._log_path = (
            Path(log_path) if log_path is not None
            else MCPAuditLogger._DEFAULT_LOG_FILE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> MCPCallStats:
        """Parse the audit log and return aggregated ``MCPCallStats``.

        Returns an all-zero ``MCPCallStats`` if the log file does not exist
        or is empty.
        """
        records = self._load_records()
        if not records:
            return MCPCallStats(
                total_calls=0, blocked_calls=0, block_rate=0.0,
                unique_servers=0, unique_tools=0,
                calls_by_server={}, calls_by_tool={},
                blocked_by_pattern={}, avg_duration_ms=0.0,
                first_call_at=None, last_call_at=None,
            )

        blocked_records = [r for r in records if r.get("blocked")]

        by_server: dict[str, int] = {}
        by_tool:   dict[str, int] = {}
        by_pattern: dict[str, int] = {}
        durations: list[float] = []

        for rec in records:
            srv  = rec.get("server_id", "unknown") or "unknown"
            tool = rec.get("tool_name", "unknown") or "unknown"
            by_server[srv]  = by_server.get(srv, 0) + 1
            by_tool[tool]   = by_tool.get(tool, 0) + 1
            d = rec.get("call_duration_ms")
            if d is not None:
                try:
                    durations.append(float(d))
                except (TypeError, ValueError):
                    pass

        for rec in blocked_records:
            reason = rec.get("block_reason") or ""
            # Extract the quoted pattern name, e.g. 'aws-access-key-id'
            m = re.search(r"'([a-z0-9_\-]+)'", reason)
            pattern = m.group(1) if m else "unknown"
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1

        timestamps = sorted(
            r["timestamp"] for r in records if r.get("timestamp")
        )

        return MCPCallStats(
            total_calls=len(records),
            blocked_calls=len(blocked_records),
            block_rate=len(blocked_records) / len(records),
            unique_servers=len(by_server),
            unique_tools=len(by_tool),
            calls_by_server=dict(sorted(by_server.items(), key=lambda x: -x[1])),
            calls_by_tool=dict(sorted(by_tool.items(), key=lambda x: -x[1])),
            blocked_by_pattern=dict(sorted(by_pattern.items(), key=lambda x: -x[1])),
            avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
            first_call_at=timestamps[0] if timestamps else None,
            last_call_at=timestamps[-1] if timestamps else None,
        )

    def print_summary(self, file=None) -> None:
        """Print a human-readable security summary.

        Args:
            file: File-like object to write to.  Defaults to ``sys.stdout``.
        """
        stats = self.generate()

        def _p(*args, **kw):
            print(*args, file=file, **kw)

        _p("=" * 62)
        _p("  MCP FIREWALL — Security Report")
        _p("=" * 62)
        _p(f"  Log file      : {self._log_path}")
        _p(f"  Total calls   : {stats.total_calls}")
        _p(f"  Blocked calls : {stats.blocked_calls}  ({stats.block_rate:.1%} block rate)")
        _p(f"  Unique servers: {stats.unique_servers}")
        _p(f"  Unique tools  : {stats.unique_tools}")
        _p(f"  Avg latency   : {stats.avg_duration_ms:.1f} ms")
        if stats.first_call_at:
            _p(f"  First call    : {stats.first_call_at}")
            _p(f"  Last call     : {stats.last_call_at}")

        if stats.blocked_by_pattern:
            _p("\n  Top attack patterns blocked:")
            for pat, count in list(stats.blocked_by_pattern.items())[:10]:
                _p(f"    {pat:<34} {count:>4} block(s)")

        if stats.calls_by_server:
            _p("\n  Calls by server:")
            for srv, count in list(stats.calls_by_server.items())[:10]:
                display = srv if len(srv) <= 48 else srv[:45] + "..."
                _p(f"    {display:<48} {count:>4} call(s)")

        if stats.calls_by_tool:
            _p("\n  Calls by tool:")
            for tool, count in list(stats.calls_by_tool.items())[:10]:
                _p(f"    {tool:<34} {count:>4} call(s)")

        _p("=" * 62)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        if not self._log_path.exists():
            return []
        records: list[dict] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(_json.loads(line))
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(
                "MCPSecurityReport: could not read '%s': %s",
                self._log_path, exc,
            )
        return records
