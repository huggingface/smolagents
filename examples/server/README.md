# Smolagents Chat Server Demo

This is a simple web server that provides a chat interface for interacting with an AI code agent powered by `smolagents` and the Qwen2.5-Coder-32B-Instruct model.

## Features

- Web-based chat interface
- AI code agent powered by Qwen2.5-Coder
- Asynchronous request handling
- Clean, responsive UI

## Requirements

- Python 3.8+
- Starlette
- AnyIO
- Smolagents

## Installation

1. Install the required packages:

```bash
pip install starlette anyio smolagents uvicorn
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

When a user sends a message:
1. The message is sent to the `/chat` endpoint
2. The server runs the AI code agent in a separate thread
3. The agent's response is returned to the client and displayed in the chat

## Customization

You can modify the `CodeAgent` configuration by adding tools or changing the model. For example:

```python
agent = CodeAgent(
    model=InferenceClientModel(model_id="your-preferred-model"),
    tools=[your_custom_tools],
)
```
