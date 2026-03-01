# Tools in smolagents — TL;DR

> Full details: [tutorials/tools.md](./tutorials/tools.md)

## Two Ways to Create Tools

### 1. @tool decorator (recommended for simple tools)

```python
from smolagents import tool

@tool
def get_weather(location: str, date: str) -> str:
    """Returns the weather report for a location and date.

    Args:
        location: Place name, e.g. "Oslo, Norway".
        date: Date in format 'YYYY-MM-DD'.
    """
    # your implementation
    return f"Weather for {location} on {date}: sunny, 18C"
```

The decorator extracts the name, description, and argument types from the function signature and docstring automatically.

### 2. Tool class (for complex tools with multiple methods or class state)

```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = "Returns the most downloaded HuggingFace model for a given task."
    inputs = {
        "task": {
            "type": "string",
            "description": "Task category, e.g. 'text-classification'"
        }
    }
    output_type = "string"

    def forward(self, task: str) -> str:
        from huggingface_hub import list_models
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

## Required Elements for Any Tool

- **name** — short identifier, describes what the tool does
- **description** — used in the agent's system prompt; be specific and clear
- **type annotations** — input and output types must use Pydantic-compatible types: `string`, `boolean`, `integer`, `number`, `image`, `audio`, `array`, `object`, `any`, `null`
- **forward() / decorated function** — the actual logic

## Default / Built-in Tools

smolagents includes ready-to-use tools (install with `pip install 'smolagents[toolkit]'`):

- `DuckDuckGoSearchTool` / `WebSearchTool` — web search
- `VisitWebpageTool` — fetch and convert a webpage to markdown
- `PythonInterpreterTool` — run Python code (for ToolCallingAgent)
- `SpeechToTextTool` — transcribe audio
- `WikipediaSearchTool` — Wikipedia lookup

Enable built-in tools on an agent with `add_base_tools=True`.

## Loading Tools from Hub

```python
from smolagents import load_tool, CodeAgent

tool = load_tool("username/my-tool", trust_remote_code=True)
agent = CodeAgent(tools=[tool], model=model)
```

## MCP Server Tools

```python
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters

params = StdioServerParameters(command="uvx", args=["pubmedmcp@0.1.3"])
with MCPClient(params) as tools:
    agent = CodeAgent(tools=tools, model=model)
    agent.run("Find recent COVID-19 treatment research.")
```

> For tool sharing to Hub, LangChain tool integration, and advanced patterns: see [tutorials/tools.md](./tutorials/tools.md)
