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

"""Pre-tool-call authorization layer for smolagents.

Provides a ``GuardrailProvider`` protocol that is checked before every tool
call, allowing users to control which tools an agent is authorized to invoke.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class GuardrailDecision:
    """Result of a guardrail check.

    Attributes:
        allowed: Whether the tool call is authorized.
        reason: Human-readable explanation when a call is denied.
    """

    allowed: bool
    reason: str = ""


@runtime_checkable
class GuardrailProvider(Protocol):
    """Protocol for pre-tool-call authorization.

    Implementations are called before every tool invocation. Returning a
    ``GuardrailDecision`` with ``allowed=False`` prevents execution and
    surfaces the ``reason`` as an observation so the agent can adapt.
    """

    def before_tool_call(self, tool_name: str, arguments: dict[str, Any] | str) -> GuardrailDecision:
        """Decide whether a tool call should proceed.

        Args:
            tool_name: Name of the tool about to be called.
            arguments: Arguments that will be passed to the tool.

        Returns:
            A ``GuardrailDecision`` indicating whether the call is allowed.
        """
        ...


class AllowlistGuardrail:
    """Simple guardrail that permits only tools whose names appear in an allowlist.

    ``final_answer`` is always permitted regardless of the allowlist contents.

    Args:
        allowed_tools: Collection of tool names that are authorized.
    """

    def __init__(self, allowed_tools: list[str] | set[str]):
        self.allowed_tools: set[str] = set(allowed_tools) | {"final_answer"}

    def before_tool_call(self, tool_name: str, arguments: dict[str, Any] | str) -> GuardrailDecision:
        if tool_name in self.allowed_tools:
            return GuardrailDecision(allowed=True)
        return GuardrailDecision(
            allowed=False,
            reason=f"Tool '{tool_name}' is not in the allowed tools list: {sorted(self.allowed_tools)}",
        )


class BlocklistGuardrail:
    """Guardrail that denies specific tools and permits everything else.

    Args:
        blocked_tools: Collection of tool names that are denied.
    """

    def __init__(self, blocked_tools: list[str] | set[str]):
        self.blocked_tools: set[str] = set(blocked_tools)

    def before_tool_call(self, tool_name: str, arguments: dict[str, Any] | str) -> GuardrailDecision:
        if tool_name in self.blocked_tools:
            return GuardrailDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is blocked.",
            )
        return GuardrailDecision(allowed=True)


class CompositeGuardrail:
    """Chains multiple guardrail providers; all must allow the call.

    Args:
        providers: Guardrail providers to evaluate in order. The first denial wins.
    """

    def __init__(self, providers: list[GuardrailProvider]):
        self.providers = providers

    def before_tool_call(self, tool_name: str, arguments: dict[str, Any] | str) -> GuardrailDecision:
        for provider in self.providers:
            decision = provider.before_tool_call(tool_name, arguments)
            if not decision.allowed:
                return decision
        return GuardrailDecision(allowed=True)
