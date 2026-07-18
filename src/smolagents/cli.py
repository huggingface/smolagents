#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from .agents import CodeAgent, ToolCallingAgent
from .default_tools import TOOL_MAPPING
from .memory import ActionStep, FinalAnswerStep, MemoryStep, PlanningStep
from .models import (
    STRUCTURED_GENERATION_PROVIDERS,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    OpenAIModel,
    TransformersModel,
)
from .monitoring import LogLevel
from .tools import Tool
from .utils import make_json_serializable, truncate_content


console = Console()

leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


@dataclass(frozen=True)
class AutonomyProfileDefaults:
    max_steps: int
    planning_interval: int | None
    use_structured_internal_output: bool
    tool_retry_limit: int
    stagnation_window: int


AUTONOMY_PROFILES: dict[str, AutonomyProfileDefaults] = {
    "legacy": AutonomyProfileDefaults(
        max_steps=20,
        planning_interval=None,
        use_structured_internal_output=False,
        tool_retry_limit=0,
        stagnation_window=0,
    ),
    "balanced": AutonomyProfileDefaults(
        max_steps=32,
        planning_interval=6,
        use_structured_internal_output=True,
        tool_retry_limit=1,
        stagnation_window=2,
    ),
    "long-horizon": AutonomyProfileDefaults(
        max_steps=64,
        planning_interval=4,
        use_structured_internal_output=True,
        tool_retry_limit=2,
        stagnation_window=3,
    ),
}


@dataclass(frozen=True)
class RunConfig:
    autonomy_profile: Literal["legacy", "balanced", "long-horizon"]
    max_steps: int
    planning_interval: int | None
    use_structured_internal_output: bool
    checkpoint_path: str
    resume: bool
    memory_path: str
    approval_policy: Literal["auto", "on-failure", "always"]
    tool_retry_limit: int
    stagnation_window: int
    json_events: bool
    final_schema_path: str | None
    instructions_files: list[str]
    append_instructions: list[str]
    verbosity_level: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunConfig":
        profile = AUTONOMY_PROFILES[args.autonomy_profile]

        max_steps = args.max_steps if args.max_steps is not None else profile.max_steps
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}")

        planning_interval = args.planning_interval if args.planning_interval is not None else profile.planning_interval
        if planning_interval is not None and planning_interval <= 0:
            planning_interval = None

        use_structured_internal_output = (
            args.use_structured_internal_output
            if args.use_structured_internal_output is not None
            else profile.use_structured_internal_output
        )

        tool_retry_limit = args.tool_retry_limit if args.tool_retry_limit is not None else profile.tool_retry_limit
        if tool_retry_limit < 0:
            raise ValueError(f"tool_retry_limit must be >= 0, got {tool_retry_limit}")

        stagnation_window = args.stagnation_window if args.stagnation_window is not None else profile.stagnation_window
        if stagnation_window < 0:
            raise ValueError(f"stagnation_window must be >= 0, got {stagnation_window}")

        return cls(
            autonomy_profile=args.autonomy_profile,
            max_steps=max_steps,
            planning_interval=planning_interval,
            use_structured_internal_output=use_structured_internal_output,
            checkpoint_path=args.checkpoint_path,
            resume=args.resume,
            memory_path=args.memory_path,
            approval_policy=args.approval_policy,
            tool_retry_limit=tool_retry_limit,
            stagnation_window=stagnation_window,
            json_events=args.json_events,
            final_schema_path=args.final_schema,
            instructions_files=args.instructions_file,
            append_instructions=args.append_instruction,
            verbosity_level=args.verbosity_level,
        )

    @classmethod
    def default(cls) -> "RunConfig":
        profile = AUTONOMY_PROFILES["long-horizon"]
        return cls(
            autonomy_profile="long-horizon",
            max_steps=profile.max_steps,
            planning_interval=profile.planning_interval,
            use_structured_internal_output=profile.use_structured_internal_output,
            checkpoint_path=".smolagent/runs/latest.json",
            resume=False,
            memory_path=".smolagent/memory.md",
            approval_policy="auto",
            tool_retry_limit=profile.tool_retry_limit,
            stagnation_window=profile.stagnation_window,
            json_events=False,
            final_schema_path=None,
            instructions_files=[],
            append_instructions=[],
            verbosity_level=1,
        )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smolagent with configurable autonomy harness settings")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help=(
            "The prompt to run with the agent. If no prompt is provided and --resume is not set, "
            "interactive mode will be launched."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        choices=["InferenceClientModel", "OpenAIModel", "OpenAIServerModel", "LiteLLMModel", "TransformersModel"],
        help="The model type to use.",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="code",
        choices=["code", "tool_calling"],
        help="The action type to use.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Next-80B-A3B-Thinking",
        help="The model ID to use for the specified model type.",
    )
    parser.add_argument(
        "--imports",
        nargs="*",
        default=[],
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas').",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools the agent can use.",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="Verbosity level as int in [-1, 0, 1, 2].",
    )
    parser.add_argument(
        "--autonomy-profile",
        type=str,
        default="long-horizon",
        choices=["legacy", "balanced", "long-horizon"],
        help="Autonomy profile. Defaults to long-horizon.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of action steps.")
    parser.add_argument(
        "--planning-interval",
        type=int,
        default=None,
        help="Plan every N steps. Pass 0 or negative to disable periodic planning.",
    )
    parser.add_argument(
        "--use-structured-internal-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use structured generation for CodeAgent internal action steps when backend supports it.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=".smolagent/runs/latest.json",
        help="Path used to persist checkpoint state after each run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume using checkpoint context from --checkpoint-path.",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default=".smolagent/memory.md",
        help="Path used for persistent memory notes across runs.",
    )
    parser.add_argument(
        "--approval-policy",
        type=str,
        default="auto",
        choices=["auto", "on-failure", "always"],
        help="Runtime approval policy included in harness instructions.",
    )
    parser.add_argument(
        "--tool-retry-limit",
        type=int,
        default=None,
        help="Number of retries for recoverable step failures.",
    )
    parser.add_argument(
        "--stagnation-window",
        type=int,
        default=None,
        help="Trigger re-plan after this many repeated non-progress steps.",
    )
    parser.add_argument(
        "--json-events",
        action="store_true",
        help="Emit machine-readable JSON step events.",
    )
    parser.add_argument(
        "--final-schema",
        type=str,
        default=None,
        help="Path to a JSON schema used to validate final answers.",
    )
    parser.add_argument(
        "--instructions-file",
        action="append",
        default=[],
        help="Path to an additional instruction file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--append-instruction",
        action="append",
        default=[],
        help="Inline instruction snippet to append. Can be passed multiple times.",
    )

    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model.",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model.",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model.",
    )
    return parser.parse_args(argv)


def interactive_mode():
    """Run the CLI in interactive mode."""
    console.print(
        Panel.fit(
            "[bold magenta]Smolagents CLI[/]\n[dim]Intelligent agents at your service[/]", border_style="magenta"
        )
    )

    console.print("\n[bold yellow]Welcome to smolagents![/] Let's set up your agent step by step.\n")

    console.print(Rule("[bold yellow]Configuration", style="bold yellow"))

    action_type = Prompt.ask(
        "[bold white]What action type would you like to use? 'code' or 'tool_calling'?[/]",
        default="code",
        choices=["code", "tool_calling"],
    )

    tools_table = Table(title="[bold yellow]Available Tools", show_header=True, header_style="bold yellow")
    tools_table.add_column("Tool Name", style="bold yellow")
    tools_table.add_column("Description", style="white")

    for tool_name, tool_class in TOOL_MAPPING.items():
        try:
            tool_instance = tool_class()
            description = getattr(tool_instance, "description", "No description available")
        except Exception:
            description = "Built-in tool"
        tools_table.add_row(tool_name, description)

    console.print(tools_table)
    console.print("\n[dim]You can also use Hugging Face Spaces by providing the full path (e.g., 'username/spacename').[/]")

    console.print("[dim]Enter tool names separated by spaces (e.g., 'web_search python_interpreter').[/]")
    tools_input = Prompt.ask("[bold white]Select tools for your agent[/]", default="web_search")
    tools = tools_input.split()

    console.print("\n[bold yellow]Model Configuration:[/]")
    model_type = Prompt.ask(
        "[bold]Model type[/]",
        default="InferenceClientModel",
        choices=["InferenceClientModel", "OpenAIModel", "LiteLLMModel", "TransformersModel"],
    )

    model_id = Prompt.ask("[bold white]Model ID[/]", default="Qwen/Qwen2.5-Coder-32B-Instruct")

    provider = None
    api_base = None
    api_key = None
    imports = []

    if Confirm.ask("\n[bold white]Configure advanced options?[/]", default=False):
        if model_type in ["InferenceClientModel", "OpenAIModel", "LiteLLMModel"]:
            provider = Prompt.ask("[bold white]Provider[/]", default="")
            api_base = Prompt.ask("[bold white]API Base URL[/]", default="")
            api_key = Prompt.ask("[bold white]API Key[/]", default="", password=True)

        imports_input = Prompt.ask("[bold white]Additional imports (space-separated)[/]", default="")
        if imports_input:
            imports = imports_input.split()

    prompt = Prompt.ask(
        "[bold white]Now the final step; what task would you like the agent to perform?[/]", default=leopard_prompt
    )

    return prompt, tools, model_type, model_id, provider, api_base, api_key, imports, action_type


def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> Model:
    if model_type in {"OpenAIModel", "OpenAIServerModel"}:
        return OpenAIModel(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            api_base=api_base or "https://api.fireworks.ai/inference/v1",
            model_id=model_id,
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    elif model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_TOKEN"),
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _resolve_log_level(level: int) -> LogLevel:
    try:
        return LogLevel(level)
    except ValueError as e:
        raise ValueError(f"Unsupported verbosity level: {level}. Expected one of -1, 0, 1, 2.") from e


def _resolve_instruction_path(instruction_path: str, workspace_root: Path) -> Path:
    path = Path(instruction_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path


def _load_file_text(path: Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content if content else None


def _build_runtime_contract(run_config: RunConfig) -> str:
    planning_interval = run_config.planning_interval if run_config.planning_interval is not None else "disabled"
    return "\n".join(
        [
            "Runtime autonomy contract:",
            f"- Autonomy profile: {run_config.autonomy_profile}",
            f"- Max steps budget: {run_config.max_steps}",
            f"- Planning interval: {planning_interval}",
            f"- Approval policy: {run_config.approval_policy}",
            f"- Tool retry limit: {run_config.tool_retry_limit}",
            f"- Stagnation window: {run_config.stagnation_window}",
            "- At every cycle provide: Objective, Next Action, Validation Check, Completion Signal.",
            "- If no progress repeats, trigger a recovery re-plan before continuing.",
            "- If max steps are reached, provide a handoff summary with: attempted actions, blockers, and best next step.",
        ]
    )


def _build_resume_context(checkpoint_payload: dict[str, Any]) -> str | None:
    run_state = checkpoint_payload.get("run_state")
    if not isinstance(run_state, dict):
        return None

    lines = ["Resume context from previous run:"]
    if run_state.get("pending_task"):
        lines.append(f"- Pending task: {run_state['pending_task']}")
    if run_state.get("completed_action_steps") is not None:
        lines.append(f"- Completed action steps: {run_state['completed_action_steps']}")
    if run_state.get("last_plan"):
        lines.append("- Last plan:")
        lines.append(str(run_state["last_plan"]))

    memory_summary = run_state.get("memory_summary") or []
    if memory_summary:
        lines.append("- Recent memory summary:")
        for item in memory_summary[:8]:
            role = item.get("role", "unknown") if isinstance(item, dict) else "unknown"
            content = item.get("content", "") if isinstance(item, dict) else str(item)
            lines.append(f"  - {role}: {truncate_content(str(content), max_length=400)}")

    return "\n".join(lines)


def _load_instruction_stack(run_config: RunConfig, workspace_root: Path, checkpoint_payload: dict[str, Any] | None) -> str:
    sections: list[str] = [_build_runtime_contract(run_config)]

    project_agents = workspace_root / "AGENTS.md"
    project_instructions = _load_file_text(project_agents)
    if project_instructions:
        sections.append(f"Project instructions from AGENTS.md:\n{project_instructions}")

    for instruction_path in run_config.instructions_files:
        path = _resolve_instruction_path(instruction_path, workspace_root)
        content = _load_file_text(path)
        if content is None:
            raise FileNotFoundError(f"Instruction file not found or empty: {path}")
        sections.append(f"Instructions from {path}:\n{content}")

    memory_path = _resolve_instruction_path(run_config.memory_path, workspace_root)
    memory_content = _load_file_text(memory_path)
    if memory_content:
        sections.append(f"Persistent memory notes:\n{memory_content}")

    if checkpoint_payload:
        resume_context = _build_resume_context(checkpoint_payload)
        if resume_context:
            sections.append(resume_context)

    if run_config.append_instructions:
        sections.append("Appended runtime instructions:\n" + "\n".join(run_config.append_instructions))

    return "\n\n".join(sections)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} must contain a JSON object.")
    return payload


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_memory_entry(path: Path, checkpoint_payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_state = checkpoint_payload.get("run_state", {}) if isinstance(checkpoint_payload.get("run_state"), dict) else {}
    timestamp = datetime.now(timezone.utc).isoformat()
    final_output = truncate_content(str(checkpoint_payload.get("output")), max_length=1200)
    last_plan = truncate_content(str(run_state.get("last_plan", "")), max_length=1200)
    entry = "\n".join(
        [
            f"## {timestamp}",
            f"Task: {checkpoint_payload.get('task', '')}",
            f"Status: {checkpoint_payload.get('status', 'unknown')}",
            f"Completed action steps: {run_state.get('completed_action_steps', 'n/a')}",
            "Final output:",
            final_output,
            "Last plan:",
            last_plan,
            "",
        ]
    )

    if path.exists():
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + entry)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Smolagent Memory\n\n" + entry)


def _matches_schema_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_with_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type is not None:
        if isinstance(expected_type, list):
            if not any(_matches_schema_type(value, one_type) for one_type in expected_type):
                errors.append(f"{path}: expected one of {expected_type}, got {type(value).__name__}")
                return errors
        elif not _matches_schema_type(value, expected_type):
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return errors

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"{path}: value {value!r} not in enum {enum_values!r}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        for required_key in required:
            if required_key not in value:
                errors.append(f"{path}.{required_key}: required property is missing")

        properties = schema.get("properties", {})
        additional_properties = schema.get("additionalProperties", True)

        for key, sub_value in value.items():
            if key in properties:
                errors.extend(_validate_with_schema(sub_value, properties[key], f"{path}.{key}"))
            elif additional_properties is False:
                errors.append(f"{path}.{key}: additional properties are not allowed")

    if isinstance(value, list) and "items" in schema and isinstance(schema["items"], dict):
        for index, item in enumerate(value):
            errors.extend(_validate_with_schema(item, schema["items"], f"{path}[{index}]"))

    return errors


def _schema_expects_json_value(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "object" in schema_type or "array" in schema_type
    return schema_type in {"object", "array"}


def _load_final_schema(final_schema_path: str) -> dict[str, Any]:
    path = Path(final_schema_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Final schema file not found: {path}")
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)
    if not isinstance(schema, dict):
        raise ValueError("Final schema must be a JSON object")
    return schema


def _build_final_schema_check(schema: dict[str, Any]) -> Callable[..., bool]:
    def check(final_answer, memory, agent) -> bool:
        answer_for_validation = final_answer
        if _schema_expects_json_value(schema) and isinstance(final_answer, str):
            try:
                answer_for_validation = json.loads(final_answer)
            except json.JSONDecodeError as e:
                raise ValueError(f"Final answer must be JSON for schema validation: {e}") from e

        errors = _validate_with_schema(answer_for_validation, schema)
        if errors:
            raise ValueError("; ".join(errors[:3]))
        return True

    return check


def _create_json_event_callback() -> Callable[..., None]:
    def callback(memory_step: MemoryStep, **kwargs):
        event: dict[str, Any] = {
            "event": "step",
            "step_type": memory_step.__class__.__name__,
        }

        if isinstance(memory_step, ActionStep):
            event["step_number"] = memory_step.step_number
            event["is_final_answer"] = memory_step.is_final_answer
            event["error"] = memory_step.error.dict() if memory_step.error else None
            event["observations"] = memory_step.observations
        elif isinstance(memory_step, PlanningStep):
            event["plan"] = memory_step.plan
        elif isinstance(memory_step, FinalAnswerStep):
            event["output"] = memory_step.output

        print(json.dumps(make_json_serializable(event)), flush=True)

    return callback


def _model_supports_structured_outputs(model: Model) -> bool:
    if not isinstance(model, Model):
        return False
    if isinstance(model, InferenceClientModel):
        provider = model.client_kwargs.get("provider")
        return provider in STRUCTURED_GENERATION_PROVIDERS
    return True


def run_smolagent(
    prompt: str | None,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
    provider: str | None = None,
    action_type: str = "code",
    run_config: RunConfig | None = None,
) -> Any:
    load_dotenv()
    run_config = run_config or RunConfig.default()
    workspace_root = Path.cwd()
    checkpoint_path = _resolve_instruction_path(run_config.checkpoint_path, workspace_root)

    checkpoint_payload = None
    if run_config.resume:
        checkpoint_payload = _load_checkpoint(checkpoint_path)
        if prompt is None:
            prompt = checkpoint_payload.get("task")

    if prompt is None:
        raise ValueError("A prompt is required unless --resume provides one from checkpoint state.")

    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key, provider=provider)

    available_tools = []
    for tool_name in tools:
        if "/" in tool_name:
            space_name = tool_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            description = f"Tool loaded from Hugging Face Space: {tool_name}"
            available_tools.append(Tool.from_space(space_id=tool_name, name=space_name, description=description))
        elif tool_name in TOOL_MAPPING:
            available_tools.append(TOOL_MAPPING[tool_name]())
        else:
            raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    instructions = _load_instruction_stack(run_config, workspace_root, checkpoint_payload)

    final_answer_checks = []
    if run_config.final_schema_path:
        schema = _load_final_schema(run_config.final_schema_path)
        final_answer_checks.append(_build_final_schema_check(schema))

    step_callbacks = None
    if run_config.json_events:
        step_callbacks = {MemoryStep: [_create_json_event_callback()]}

    structured_outputs_requested = run_config.use_structured_internal_output
    structured_outputs_enabled = structured_outputs_requested and _model_supports_structured_outputs(model)

    common_agent_kwargs = {
        "tools": available_tools,
        "model": model,
        "stream_outputs": True,
        "instructions": instructions,
        "max_steps": run_config.max_steps,
        "planning_interval": run_config.planning_interval,
        "tool_retry_limit": run_config.tool_retry_limit,
        "stagnation_window": run_config.stagnation_window,
        "verbosity_level": _resolve_log_level(run_config.verbosity_level),
        "step_callbacks": step_callbacks,
        "final_answer_checks": final_answer_checks or None,
    }

    if action_type == "code":
        agent = CodeAgent(
            additional_authorized_imports=imports,
            use_structured_outputs_internally=structured_outputs_enabled,
            **common_agent_kwargs,
        )
    elif action_type == "tool_calling":
        agent = ToolCallingAgent(**common_agent_kwargs)
    else:
        raise ValueError(f"Unsupported action type: {action_type}")

    output = None
    run_error: str | None = None
    try:
        output = agent.run(prompt)
        return output
    except Exception as e:
        run_error = str(e)
        raise
    finally:
        run_state = agent.get_run_state_snapshot()
        checkpoint_data = {
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if run_error else "success",
            "error": run_error,
            "task": prompt,
            "action_type": action_type,
            "model_type": model_type,
            "model_id": model_id,
            "tools": tools,
            "imports": imports or [],
            "provider": provider,
            "api_base": api_base,
            "autonomy_profile": run_config.autonomy_profile,
            "run_config": {
                "max_steps": run_config.max_steps,
                "planning_interval": run_config.planning_interval,
                "approval_policy": run_config.approval_policy,
                "tool_retry_limit": run_config.tool_retry_limit,
                "stagnation_window": run_config.stagnation_window,
            },
            "run_state": run_state,
            "output": make_json_serializable(output),
        }
        _save_checkpoint(checkpoint_path, checkpoint_data)
        memory_path = _resolve_instruction_path(run_config.memory_path, workspace_root)
        _append_memory_entry(memory_path, checkpoint_data)

        if run_config.json_events:
            completion_event = {
                "event": "run_complete",
                "status": checkpoint_data["status"],
                "checkpoint_path": str(checkpoint_path),
            }
            print(json.dumps(completion_event), flush=True)


def main() -> None:
    args = parse_arguments()
    run_config = RunConfig.from_args(args)

    if args.prompt is None and not run_config.resume:
        prompt, tools, model_type, model_id, provider, api_base, api_key, imports, action_type = interactive_mode()
    else:
        prompt = args.prompt
        tools = args.tools
        model_type = args.model_type
        model_id = args.model_id
        provider = args.provider
        api_base = args.api_base
        api_key = args.api_key
        imports = args.imports
        action_type = args.action_type

    run_smolagent(
        prompt,
        tools,
        model_type,
        model_id,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        imports=imports,
        action_type=action_type,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
