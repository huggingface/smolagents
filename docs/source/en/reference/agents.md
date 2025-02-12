<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
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
  - [`CodeAgent`] is the default agent, it writes its tool calls in Python code.
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
