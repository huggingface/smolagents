# PersistentCodeAgent

The `PersistentCodeAgent` is an extension of the `CodeAgent` class that adds state persistence capabilities, allowing you to pause and resume agent execution.

## Features

- **Pause Execution**: Pause the agent at any point during execution and save its state
- **Resume Execution**: Restore a previously paused agent and continue execution from where it left off
- **Stop Execution**: Gracefully stop agent execution
- **OpenTelemetry Context Preservation**: Save and restore OTEL context for tracing

## Use Cases

- **Human-in-the-loop interactions**: Pause agent execution to get human input, then resume
- **Post-processing of agent workflows**: Save intermediate states for later analysis
- **Recovery from failures or interruptions**: Restore agent state after system failures
- **Long-running agent sessions**: Support sessions that span multiple interactions

## Usage

### Basic Usage

```python
from smolagents import PersistentCodeAgent, Tool
from smolagents.utils import AgentPausedException, AgentStoppedException

# Create a PersistentCodeAgent
agent = PersistentCodeAgent(
    tools=[...],
    model=your_model,
    storage_path="/path/to/save/agent_state.pkl"  # Optional, defaults to ./agent_state.pkl
)

# Run the agent
result = agent.run("Your task here")
```

### Pausing and Resuming

You can pause the agent by raising an `AgentPausedException` from a callback or tool:

```python
class PauseTool(Tool):
    name = "pause_agent"
    description = "Pauses the agent execution for later resumption"
    
    def __call__(self, agent=None):
        if agent:
            # Optionally capture OTEL context
            otel_context = get_current_otel_context()
            raise AgentPausedException("Agent paused", agent.logger, None, otel_context)
        return "Agent would be paused if running in an agent context"

# Later, to resume:
resumed_agent = PersistentCodeAgent.resume_execution(
    "/path/to/save/agent_state.pkl",
    otel_context_handler=restore_otel_context  # Optional function to restore OTEL context
)

# Continue execution
result = resumed_agent.run(resumed_agent.task, reset=False)
```

### Stopping

To gracefully stop the agent:

```python
class StopTool(Tool):
    name = "stop_agent"
    description = "Stops the agent execution"
    
    def __call__(self, agent=None):
        if agent:
            raise AgentStoppedException("Agent stopped", agent.logger)
        return "Agent would be stopped if running in an agent context"
```

## Implementation Details

The `PersistentCodeAgent` uses [cloudpickle](https://github.com/cloudpipe/cloudpickle) for serialization, which provides better support for pickling Python constructs compared to the standard `pickle` module.

When an agent is paused:
1. The agent state is captured, including memory, tools, and execution context
2. OpenTelemetry context is saved (if provided)
3. The state is serialized to disk using cloudpickle
4. A message is returned indicating the agent was paused

When resuming:
1. The serialized state is loaded from disk
2. The agent is reconstructed with all its previous state
3. OpenTelemetry context is restored (if a handler is provided)
4. Execution can continue from where it left off

## Example

See the [persistent_agent_example.py](../examples/persistent_agent_example.py) file for a complete example of using the `PersistentCodeAgent` with pause and resume functionality.