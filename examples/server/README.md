# Smolagents Chat Server Demo

This is a simple web server that provides a chat interface for interacting with an AI code agent powered by `smolagents` and the Qwen2.5-Coder-32B-Instruct model, enhanced with MCP (Model Control Protocol) tools.

## Features

- Web-based chat interface
- AI code agent powered by Qwen2.5-Coder
- Integration with MCP tools through MCPClient
- Asynchronous request handling
- Clean, responsive UI
- Graceful shutdown handling

## Requirements

- Python 3.8+
- Starlette
- AnyIO
- Smolagents with MCP support

## Installation

1. Install the required packages:

```bash
pip install starlette anyio smolagents[mcp] uvicorn
```

2. Optional: If you want to use a specific model, you may need additional dependencies.

## Usage

1. Run the server:

```bash
uvicorn examples.server.main:app --reload
```

2. Open your browser and navigate to `http://localhost:8000`

3. Interact with the AI code agent through the chat interface

## How It Works

The server consists of two main routes:
- `/` - Serves the HTML page with the chat interface
- `/chat` - API endpoint that processes messages and returns responses

The server integrates with MCP tools through the following components:

1. MCPClient Configuration:
```python
mcp_server_parameters = {
    "url": "https://evalstate-hf-mcp-server.hf.space/mcp",
    "transport": "streamable-http",
}
mcp_client = MCPClient(server_parameters=mcp_server_parameters)
```

2. CodeAgent with MCP Tools:
```python
agent = CodeAgent(
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
    tools=mcp_client.get_tools(),
)
```

When a user sends a message:
1. The message is sent to the `/chat` endpoint
2. The server runs the AI code agent in a separate thread
3. The agent processes the message using MCP tools
4. The agent's response is returned to the client and displayed in the chat

The server also includes a shutdown handler that properly disconnects the MCP client when the server stops:
```python
async def shutdown():
    mcp_client.disconnect()
```

## Customization

You can modify the `CodeAgent` configuration by changing the model or MCP server parameters. For example:

```python
# Custom MCP server
mcp_server_parameters = {
    "url": "your-mcp-server-url",
    "transport": "your-transport-method",
}

# Custom agent configuration
agent = CodeAgent(
    model=InferenceClientModel(model_id="your-preferred-model"),
    tools=mcp_client.get_tools(),
)
```
