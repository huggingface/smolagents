
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
# Agents（智能体）

<Tip warning={true}>

Smolagents 是一个实验性的 API，可能会随时发生变化。由于 API 或底层模型可能发生变化，代理返回的结果也可能有所不同。

</Tip>

要了解有关智能体和工具的更多信息，请务必阅读[入门指南](../index)。本页面包含基础类的 API 文档。

## 智能体（Agents）

我们的智能体继承自 [`MultiStepAgent`]，这意味着它们可以执行多步操作，每一步包含一个思考（thought），然后是一个工具调用和执行。请阅读[概念指南](../conceptual_guides/react)以了解更多信息。

我们提供两种类型的代理，它们基于主要的 [`Agent`] 类：
  - [`CodeAgent`] 是默认代理，它以 Python 代码编写工具调用。
  - [`ToolCallingAgent`] 以 JSON 编写工具调用。

两者在初始化时都需要提供参数 `model` 和工具列表 `tools`。

### 智能体类

[[autodoc]] MultiStepAgent

[[autodoc]] CodeAgent

[[autodoc]] ToolCallingAgent

### ManagedAgent

_此类自 1.8.0 起已被弃用：现在您只需向普通代理传递 `name` 和 `description` 属性即可使其可被管理代理调用。_

### stream_to_gradio

[[autodoc]] stream_to_gradio

### GradioUI

> [!TIP]
> 您必须安装 `gradio` 才能使用 UI。如果尚未安装，请运行 `pip install smolagents[gradio]`。

[[autodoc]] GradioUI

## 提示（Prompts）

[[autodoc]] smolagents.agents.PromptTemplates

[[autodoc]] smolagents.agents.PlanningPromptTemplate

[[autodoc]] smolagents.agents.ManagedAgentPromptTemplate

[[autodoc]] smolagents.agents.FinalAnswerPromptTemplate
