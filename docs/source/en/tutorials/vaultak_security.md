# Runtime security with Vaultak

[[open-in-colab]]

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

## Why add runtime security to agents?

smolagents with real tools can cause real damage: deleted records, leaked PII, unauthorized API calls, accidental data exfiltration. Even well-prompted agents occasionally take unexpected paths.

[Vaultak](https://vaultak.com) is a runtime security platform purpose-built for agentic workloads. It intercepts every tool call, scores risk on a 0–10 scale, enforces your policy rules, and masks PII in tool outputs — before results reach the LLM or your production systems. Every action is recorded in the Vaultak dashboard for later inspection.

## How it works

The cleanest integration point in smolagents is the tool's `forward()` method. Vaultak wraps each tool with a `secure_tool()` helper that:

1. **Risk-scores** the call before execution — blocks if the score meets or exceeds your threshold
2. **Checks policy rules** — enforces any custom rules you've configured in the dashboard
3. **Masks PII** in string outputs before the result is returned to the LLM

smolagents also supports `step_callbacks` — functions called after each agent step. These work well for error monitoring and alerting.

## Install

```shell
pip install vaultak smolagents
```

Sign up at [vaultak.com](https://vaultak.com) to get your API key (starts with `vtk_`).

## Quick start

The `secure_tool()` helper patches a tool's `forward()` method in-place and returns it, so you can wrap tools inline when constructing an agent.

```python
import os

from smolagents import InferenceClientModel, ToolCallingAgent, WebSearchTool
from smolagents.tools import Tool

from vaultak import Vaultak

_api_key = os.environ.get("VAULTAK_API_KEY")
if not _api_key:
    raise ValueError(
        "VAULTAK_API_KEY environment variable is not set. "
        "Sign up at https://vaultak.com to get your API key."
    )

RISK_THRESHOLD = 7.0
vt = Vaultak(api_key=_api_key, agent_name="smolagents-agent")


def secure_tool(tool: Tool) -> Tool:
    """Wrap a smolagents Tool with Vaultak pre-execution security checks."""
    original_forward = tool.forward
    tool_name = tool.name

    def secured_forward(**kwargs):
        # Risk-score the call before it executes
        result = vt.score_action(action=tool_name, context=kwargs)
        if result.score >= RISK_THRESHOLD:
            # Raising an exception here surfaces the block message to the LLM,
            # which can then decide how to proceed without the blocked tool.
            raise RuntimeError(
                f"[Vaultak] Tool '{tool_name}' blocked — risk score "
                f"{result.score:.1f}/10 meets or exceeds threshold {RISK_THRESHOLD}. "
                "Review at app.vaultak.com"
            )
        # Check against your policy rules configured in the dashboard
        vt.check_policy(tool_name=tool_name, input_data=str(kwargs))
        # Execute the original tool
        output = original_forward(**kwargs)
        # Mask PII in string outputs before they reach the model
        if isinstance(output, str):
            output = vt.mask_pii(output)
        return output

    tool.forward = secured_forward
    return tool


# Wrap each tool — works for both ToolCallingAgent and CodeAgent
agent = ToolCallingAgent(
    tools=[secure_tool(WebSearchTool())],
    model=InferenceClientModel(),
)

result = agent.run("Find the latest AI safety research papers.")
print(result)
```

`secure_tool()` works identically for `CodeAgent` since it executes tools via the same `forward()` path.

## Adding step-level monitoring

Use `step_callbacks` to receive an `ActionStep` object after every agent step. This is the right place to alert Vaultak when a tool raises an error.

```python
from smolagents.memory import ActionStep


def vaultak_step_monitor(step, agent):
    """Alert Vaultak dashboard when a tool raises an error during a step."""
    if isinstance(step, ActionStep) and step.error is not None:
        vt.alert(
            level="error",
            message=(
                f"Agent step {step.step_number} failed: {step.error}"
            ),
        )


agent = ToolCallingAgent(
    tools=[secure_tool(WebSearchTool())],
    model=InferenceClientModel(),
    step_callbacks=[vaultak_step_monitor],
)
```

## Multi-agent setup

Wrap tools in every sub-agent independently. Each agent can be given a different `agent_name` so you can distinguish their traffic in the Vaultak dashboard.

```python
from smolagents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool, InferenceClientModel

model = InferenceClientModel()

vt_search = Vaultak(api_key=_api_key, agent_name="search-agent")
vt_manager = Vaultak(api_key=_api_key, agent_name="manager-agent")


def make_secure(tool, client):
    original = tool.forward
    name = tool.name

    def secured(**kwargs):
        r = client.score_action(action=name, context=kwargs)
        if r.score >= RISK_THRESHOLD:
            raise RuntimeError(
                f"[Vaultak] Tool '{name}' blocked — risk {r.score:.1f}/10 "
                f"meets or exceeds threshold {RISK_THRESHOLD}. Review at app.vaultak.com"
            )
        client.check_policy(tool_name=name, input_data=str(kwargs))
        out = original(**kwargs)
        return client.mask_pii(out) if isinstance(out, str) else out

    tool.forward = secured
    return tool


search_agent = ToolCallingAgent(
    tools=[make_secure(WebSearchTool(), vt_search), make_secure(VisitWebpageTool(), vt_search)],
    model=model,
    name="search_agent",
    description="Searches the web and reads pages.",
)

manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)

manager.run("Summarize the latest news on AI regulation in the EU.")
```

## Stricter threshold for sensitive workloads

Lower `RISK_THRESHOLD` for agents with access to databases, payment systems, or external write APIs to block anything above medium risk:

```python
RISK_THRESHOLD = 5.0  # Block medium-risk calls and above

vt = Vaultak(api_key=_api_key, agent_name="prod-db-agent")
```

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Your Vaultak API key — required |
| `agent_name` | `str` | `"smolagents-agent"` | Label for this agent in the Vaultak dashboard |
| `RISK_THRESHOLD` | `float` | `7.0` | Score (0–10) at or above which tool calls are blocked |

## What gets monitored

| smolagents event | Vaultak action |
|---|---|
| Tool call selected (any agent type) | Risk-scores inputs; blocks via `RuntimeError` if score ≥ threshold |
| Policy check | Validates call against your dashboard-configured rules |
| Tool output returned | Scans for PII and masks before result reaches the model |
| Tool or step error (`step_callbacks`) | Sends an error alert to the Vaultak dashboard |

## Links

- [Vaultak documentation](https://docs.vaultak.com)
- [PyPI: `vaultak`](https://pypi.org/project/vaultak)
- [GitHub](https://github.com/vaultak/vaultak-python)
- [Dashboard](https://app.vaultak.com)
- [smolagents Tools guide](tools)
- [smolagents step_callbacks reference](../reference/agents)
