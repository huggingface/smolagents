# Built-in Tools

Ready-to-use tool implementations provided by the `smolagents` library.

These built-in tools are concrete implementations of the [`Tool`] base class, each designed for specific tasks such as web searching, Python code execution, webpage retrieval, and user interaction.
You can use these tools directly in your agents without having to implement the underlying functionality yourself.
Each tool handles a particular capability and follows a consistent interface, making it easy to compose them into powerful agent workflows.

The built-in tools can be categorized by their primary functions:
- **Information Retrieval**: Search and retrieve information from the web and specific knowledge sources.
  - [`ApiWebSearchTool`]
  - [`DuckDuckGoSearchTool`]
  - [`GoogleSearchTool`]
  - [`WebSearchTool`]
  - [`WikipediaSearchTool`]
- **Web Interaction**: Fetch and process content from specific web pages.
  - [`VisitWebpageTool`]
- **Code Execution**: Dynamic execution of Python code for computational tasks.
  - [`PythonInterpreterTool`]
- **User Interaction**: Enable Human-in-the-Loop collaboration between agents and users.
  - [`UserInputTool`]: Collect input from users.
- **Speech Processing**: Convert audio to textual data.
  - [`SpeechToTextTool`]
- **Workflow Control**: Manage and direct the flow of agent operations.
  - [`FinalAnswerTool`]: Conclude agent workflow with final response.

## ApiWebSearchTool

[[autodoc]] smolagents.default_tools.ApiWebSearchTool

## DuckDuckGoSearchTool

[[autodoc]] smolagents.default_tools.DuckDuckGoSearchTool

## FinalAnswerTool

[[autodoc]] smolagents.default_tools.FinalAnswerTool

## GoogleSearchTool

[[autodoc]] smolagents.default_tools.GoogleSearchTool

## PythonInterpreterTool

[[autodoc]] smolagents.default_tools.PythonInterpreterTool

## SpeechToTextTool

[[autodoc]] smolagents.default_tools.SpeechToTextTool

## UserInputTool

[[autodoc]] smolagents.default_tools.UserInputTool

## VisitWebpageTool

[[autodoc]] smolagents.default_tools.VisitWebpageTool

## WebSearchTool

[[autodoc]] smolagents.default_tools.WebSearchTool

## WikipediaSearchTool

[[autodoc]] smolagents.default_tools.WikipediaSearchTool
