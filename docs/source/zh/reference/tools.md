# 工具

<Tip warning={true}>

Smolagents 是一个实验性 API，可能会随时更改。由于 API 或底层模型可能发生变化，代理返回的结果可能会有所不同。

</Tip>

要了解更多关于智能体和工具的信息，请务必阅读[入门指南](../index)。本页面包含底层类的 API 文档。

## 工具

### load_tool

[[autodoc]] load_tool

### tool

[[autodoc]] tool

### Tool

[[autodoc]] Tool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## 默认工具

### PythonInterpreterTool

[[autodoc]] PythonInterpreterTool

### FinalAnswerTool

[[autodoc]] FinalAnswerTool

### UserInputTool

[[autodoc]] UserInputTool

### DuckDuckGoSearchTool

[[autodoc]] DuckDuckGoSearchTool

### GoogleSearchTool

[[autodoc]] GoogleSearchTool

### VisitWebpageTool

[[autodoc]] VisitWebpageTool

### SpeechToTextTool

[[autodoc]] SpeechToTextTool

## 工具集合

[[autodoc]] ToolCollection

## 智能体类型

智能体可以处理工具之间的任何类型的对象；工具是完全多模态的，可以接受和返回文本、图像、音频、视频以及其他类型的对象。为了增加工具之间的兼容性，以及正确呈现在 ipython（jupyter、colab、ipython notebooks 等）中的返回结果，我们为这些类型实现了包装类。

被包装的对象应该继续保持其初始行为；例如，一个文本对象应继续表现为字符串，一个图像对象应继续表现为 `PIL.Image`。

这些类型有三个特定的用途：

- 调用 `to_raw` 方法时，应返回底层对象
- 调用 `to_string` 方法时，应将对象转换为字符串：对于 `AgentText` 类型，可以直接返回字符串；对于其他实例，则返回对象序列化版本的路径
- 在 ipython 内核中显示时，应正确显示对象

### AgentText

[[autodoc]] smolagents.agent_types.AgentText

### AgentImage

[[autodoc]] smolagents.agent_types.AgentImage

### AgentAudio

[[autodoc]] smolagents.agent_types.AgentAudio
