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
Three security layers for smolagents MCP tool ingestion:

  Layer 1 — TrustVerifier   : pre-flight check on server URL/command before any
                               TCP connection is made.
  Layer 2 — MCPPayloadValidator : post-connection check on every tool's metadata
                               (name, description, inputSchema) to block prompt
                               injection and resource-exhaustion attacks.
  Layer 3 — _validate_tool_code_ast (in tools.py) : AST static analysis of
                               Hub-tool source code before exec().

Public API
----------
  Exceptions:     MCPServerUntrustedError, MCPPayloadValidationError
  Data:           TrustVerificationResult
  Verifiers:      TrustVerifier (ABC), StaticTrustVerifier, CompositeTrustVerifier
  Validator:      MCPPayloadValidator
"""

from __future__ import annotations

import builtins
import keyword
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


import ast as _ast

logger = logging.getLogger(__name__)

__all__ = [
    "MCPServerUntrustedError",
    "MCPPayloadValidationError",
    "TrustVerificationResult",
    "TrustVerifier",
    "StaticTrustVerifier",
    "CompositeTrustVerifier",
    "MCPPayloadValidator",
    "_validate_tool_code_ast",
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
    ("ignore-instructions",   re.compile(r"ignore\s+(previous|all|above|prior)\s+(instructions?|context|prompt|directive)", re.IGNORECASE)),
    ("system-role-override",  re.compile(r"\bsystem\s*:\s", re.IGNORECASE)),
    ("token-injection",       re.compile(r"<\|[a-z_]+\|>", re.IGNORECASE)),
    ("os-environ",            re.compile(r"\bos\s*\.\s*environ\b", re.IGNORECASE)),
    ("subprocess",            re.compile(r"\bsubprocess\b", re.IGNORECASE)),
    ("eval-call",             re.compile(r"\beval\s*\(", re.IGNORECASE)),
    ("exec-call",             re.compile(r"\bexec\s*\(", re.IGNORECASE)),
    ("import-os",             re.compile(r"\bimport\s+os\b", re.IGNORECASE)),
    ("null-byte",             re.compile(r"\x00")),
    ("jinja-template",        re.compile(r"\{\{|\}\}|\{%|%\}")),  # Break Jinja2 templates
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
    ):
        self.max_tool_name_length = max_tool_name_length
        self.max_description_length = max_description_length
        self.max_tools_per_server = max_tools_per_server
        self.max_input_params = max_input_params
        self.max_param_description_length = max_param_description_length
        self.allow_builtin_names = allow_builtin_names

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
