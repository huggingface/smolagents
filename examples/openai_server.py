"""
Thise example script implements an OpenAI-compatible API endpoint for your smolagent.
This way, you can easily use your smolagent in any OpenAI-compatible chat interface, such as OpenWebUI.

Usage:
    1. Start the server:
        python examples/openai_server.py

    2. Make API calls using your favorite OpenAI client:

    ```python
    from openai import OpenAI

    client = OpenAI(
        api_key="dummy",  # any string works
        base_url="http://localhost:8000/v1"  # point to local server
    )

    # Make a chat completion request
    response = client.chat.completions.create(
        model="default",  # model name is ignored
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        stream=True  # streaming is supported
    )
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")
    ```

    Or any other method you like, for instance curl:
    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        }'
    ```

The server implements the models and chat completions endpoints:
- POST /v1/chat/completions - Create chat completion
- GET /v1/models - List available models (returns dummy data)

It also reduces the agent's output to clean messages and shows the agents thoughts in xml tags.

Example: Using with OpenWebUI
1. Make sure you have OpenWebUI running (typically at http://localhost:3000)
2. In OpenWebUI, navigate to Admin Settings > Connections
3. Add a new OpenAI connection with http://host.docker.internal:8000/v1 (if using docker) as base URL and any random string as API key.
4. Select your agent (default if you haven't given a name) from the model dropdown.
"""

from smolagents import CodeAgent, HfApiModel
from smolagents.server import AgentServer


agent = CodeAgent(tools=[], model=HfApiModel(), max_steps=4, verbosity_level=1, add_base_tools=True)

# wrap agent in uvicorn server
server = AgentServer(agent)
server.run()
