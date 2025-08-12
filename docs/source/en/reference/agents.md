# Agents

<Tip warning={true}>

Smolagents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

To learn more about agents and tools make sure to read the [introductory guide](../index). This page
contains the API docs for the underlying classes.

## Agents

Our agents inherit from [`MultiStepAgent`], which means they can act in multiple steps, each step consisting of one thought, then one tool call and execution. Read more in [this conceptual guide](../conceptual_guides/react).

We provide two types of agents, based on the main [`Agent`] class.
  - [`CodeAgent`] writes its tool calls in Python code (this is the default).
  - [`ToolCallingAgent`] writes its tool calls in JSON.

Both require arguments `model` and list of tools `tools` at initialization.

### Classes of agents

[[autodoc]] MultiStepAgent

[[autodoc]] CodeAgent

[[autodoc]] ToolCallingAgent

### stream_to_gradio

[[autodoc]] stream_to_gradio

### GradioUI

> [!TIP]
> You must have `gradio` installed to use the UI. Please run `pip install smolagents[gradio]` if it's not the case.

[[autodoc]] GradioUI

## Prompts

[[autodoc]] smolagents.agents.PromptTemplates

[[autodoc]] smolagents.agents.PlanningPromptTemplate

[[autodoc]] smolagents.agents.ManagedAgentPromptTemplate

[[autodoc]] smolagents.agents.FinalAnswerPromptTemplate

## Memory

Smolagents use memory to store information across multiple steps.

[[autodoc]] smolagents.memory.AgentMemory

## Python code executors

[[autodoc]] smolagents.local_python_executor.PythonExecutor

### Local Python executor

[[autodoc]] smolagents.local_python_executor.LocalPythonExecutor

### Remote Python executors

[[autodoc]] smolagents.remote_executors.RemotePythonExecutor

#### E2BExecutor

[[autodoc]] smolagents.remote_executors.E2BExecutor

#### DockerExecutor

[[autodoc]] smolagents.remote_executors.DockerExecutor

#### WasmExecutor

[[autodoc]] smolagents.remote_executors.WasmExecutor
