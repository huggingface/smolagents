# Governance and Audit Trails with asqav

[[open-in-colab]]

> [!TIP]
> If you're new to building agents, make sure to first read the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour of smolagents](../guided_tour).

## Why add governance to your agents?

AI agents that operate autonomously - calling tools, executing code, and making decisions - need accountability. In regulated industries and production deployments, you need a verifiable record of what your agent did, when it did it, and what data it used.

[asqav](https://github.com/jagmarques/asqav-sdk) provides cryptographically signed audit trails for AI agents. Each agent action is signed using ML-DSA-65 (quantum-safe cryptography), creating tamper-evident governance records that you can verify independently.

By using smolagents' `step_callbacks`, you can sign every agent step as a governance event without modifying your agent logic.

## Setup

Install both packages:

```shell
pip install smolagents asqav
```

Sign up at [asqav.com](https://asqav.com) to get your API key, then set it as an environment variable:

```shell
export ASQAV_API_KEY="sk_..."
```

## Creating a step callback for governance

The `step_callbacks` parameter on `MultiStepAgent` (and its subclasses `CodeAgent` and `ToolCallingAgent`) accepts a list of callables. Each callback receives the `MemoryStep` and the `agent` instance after every step completes.

Here is a callback that signs each `ActionStep` as a governance event using asqav:

```python
import asqav
from smolagents import ActionStep

asqav.init()  # reads ASQAV_API_KEY from environment
governance_agent = asqav.Agent.create("smolagent-prod")

def sign_step(memory_step: ActionStep, agent) -> None:
    """Sign each agent step as a governance event."""
    event_data = {
        "step_number": memory_step.step_number,
        "model_output": memory_step.model_output,
        "observations": memory_step.observations,
        "error": str(memory_step.error) if memory_step.error else None,
        "tool_calls": [tc.dict() for tc in memory_step.tool_calls] if memory_step.tool_calls else [],
    }
    governance_agent.sign("agent:step", event_data)
```

## Running an agent with governance

Pass the callback in the `step_callbacks` list when creating your agent:

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()

agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    step_callbacks=[sign_step],
    max_steps=10,
)

result = agent.run("What is the current population of France?")
```

Every step the agent takes is now signed and recorded. You can view the full audit trail in the [asqav dashboard](https://asqav.com).

## Signing specific step types

smolagents supports registering callbacks for specific step types using a dictionary. You can sign different step types with different event categories:

```python
from smolagents import ActionStep, PlanningStep

def sign_action(memory_step: ActionStep, agent) -> None:
    governance_agent.sign("agent:action", {
        "step_number": memory_step.step_number,
        "tool_calls": [tc.dict() for tc in memory_step.tool_calls] if memory_step.tool_calls else [],
        "observations": memory_step.observations,
    })

def sign_plan(memory_step: PlanningStep, agent) -> None:
    governance_agent.sign("agent:plan", {
        "plan": memory_step.plan,
    })

agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    step_callbacks={
        ActionStep: sign_action,
        PlanningStep: sign_plan,
    },
    planning_interval=5,
    max_steps=10,
)
```

## Combining governance with telemetry

You can use asqav alongside OpenTelemetry-based monitoring. Step callbacks and OpenTelemetry instrumentation work independently, so you get both observability and governance:

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

# Set up telemetry for observability
SmolagentsInstrumentor().instrument()

# Set up asqav for governance
agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    step_callbacks=[sign_step],
)
```

See the [telemetry guide](../tutorials/inspect_runs) for full setup details on OpenTelemetry backends.

## Learn more

- [asqav documentation](https://asqav.com/docs/sdk)
- [asqav GitHub repository](https://github.com/jagmarques/asqav-sdk)
- [Step callbacks in smolagents](../tutorials/memory)
