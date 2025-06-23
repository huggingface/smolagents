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

### ManagedAgent

_This class is deprecated since 1.8.0: now you simply need to pass attributes `name` and `description` to a normal agent to make it callable by a manager agent._

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
