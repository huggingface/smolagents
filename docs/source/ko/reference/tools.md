# Tools

<Tip warning={true}>

Smolagents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

To learn more about agents and tools make sure to read the [introductory guide](../index). This page
contains the API docs for the underlying classes.

## Tool Base Classes

### load_tool

[[autodoc]] load_tool

### tool

[[autodoc]] tool

### Tool

[[autodoc]] Tool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## ToolCollection

[[autodoc]] ToolCollection

## MCP Client

[[autodoc]] smolagents.mcp_client.MCPClient

## Agent Types

Agents can handle any type of object in-between tools; tools, being completely multimodal, can accept and return
text, image, audio, video, among other types. In order to increase compatibility between tools, as well as to 
correctly render these returns in ipython (jupyter, colab, ipython notebooks, ...), we implement wrapper classes
around these types.

The wrapped objects should continue behaving as initially; a text object should still behave as a string, an image
object should still behave as a `PIL.Image`.

These types have three specific purposes:

- Calling `to_raw` on the type should return the underlying object
- Calling `to_string` on the type should return the object as a string: that can be the string in case of an `AgentText`
  but will be the path of the serialized version of the object in other instances
- Displaying it in an ipython kernel should display the object correctly

### AgentText

[[autodoc]] smolagents.agent_types.AgentText

### AgentImage

[[autodoc]] smolagents.agent_types.AgentImage

### AgentAudio

[[autodoc]] smolagents.agent_types.AgentAudio
