# Smolagents Library API Reference

**Version:** 1.20.0.dev0  
**Repository:** https://github.com/huggingface/smolagents  
**License:** Apache 2.0

## Overview

Smolagents is a library for building powerful AI agents that think in code. It provides:
- Simple agent framework (~1,000 lines of core code)
- First-class support for Code Agents that write actions in code
- Secure execution in sandboxed environments (E2B, Docker, WebAssembly)
- Hub integrations for sharing tools and agents
- Model-agnostic support (local, cloud, any LLM provider)
- Multi-modal support (text, vision, video, audio)
- Tool-agnostic (MCP servers, LangChain, Hub Spaces)

## Quick Start

```python
from smolagents import CodeAgent, HfApiModel, PythonInterpreterTool

# Create model and agent
model = HfApiModel("meta-llama/Llama-3.1-70B-Instruct")
agent = CodeAgent(
    tools=[PythonInterpreterTool()],
    model=model
)

# Run agent
result = agent.run("Calculate the square root of 144")
print(result)
```

## Core Components

### 1. Agents (`smolagents.agents`)

#### MultiStepAgent
Base agent using ReAct framework for step-by-step reasoning.

```python
from smolagents import MultiStepAgent

MultiStepAgent(
    tools: list[Tool],
    model: Model,
    prompt_templates: PromptTemplates | None = None,
    instructions: str | None = None,
    max_steps: int = 20
)
```

**Key Methods:**
- `run(task, stream=False, reset=True, images=None)` - Execute agent task
- `step(memory_step)` - Perform one ReAct step
- `interrupt()` - Stop agent execution
- `save(output_dir)` - Save agent configuration
- `from_hub(repo_id)` - Load agent from Hub
- `push_to_hub(repo_id)` - Upload agent to Hub

#### ToolCallingAgent
Agent using JSON-like tool calls with LLM's native tool calling.

```python
from smolagents import ToolCallingAgent

ToolCallingAgent(
    tools: list[Tool],
    model: Model,
    planning_interval: int | None = None,
    stream_outputs: bool = False,
    max_tool_threads: int | None = None
)
```

#### CodeAgent
Agent that formulates actions as executable code.

```python
from smolagents import CodeAgent

CodeAgent(
    tools: list[Tool],
    model: Model,
    additional_authorized_imports: list[str] | None = None,
    executor_type: Literal["local", "e2b", "docker", "wasm"] = "local"
)
```

### 2. Models (`smolagents.models`)

#### Base Model Class
```python
from smolagents.models import Model

class Model:
    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs
    ) -> ChatMessage
```

#### Local Models
```python
from smolagents.models import VLLMModel, MLXModel, TransformersModel

# vLLM for fast inference
model = VLLMModel("meta-llama/Llama-3.1-8B-Instruct")

# MLX for Apple Silicon
model = MLXModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Transformers
model = TransformersModel("microsoft/DialoGPT-medium")
```

#### API Models
```python
from smolagents.models import (
    LiteLLMModel, InferenceClientModel, OpenAIServerModel,
    AzureOpenAIServerModel, AmazonBedrockServerModel
)

# LiteLLM (supports 100+ providers)
model = LiteLLMModel("gpt-4o")

# Hugging Face Inference
model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")

# OpenAI
model = OpenAIServerModel("gpt-4o", api_key="your-key")

# Azure OpenAI
model = AzureOpenAIServerModel(
    model_id="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key"
)
```

#### Message Types
```python
from smolagents.models import ChatMessage, MessageRole

message = ChatMessage(
    role=MessageRole.USER,
    content="Hello, world!"
)
```

### 3. Tools (`smolagents.tools`)

#### Base Tool Class
```python
from smolagents.tools import Tool

class CustomTool(Tool):
    name = "custom_tool"
    description = "Tool description"
    inputs = {"input_name": {"type": "string", "description": "Input description"}}
    output_type = "string"
    
    def forward(self, input_name: str) -> str:
        return f"Processed: {input_name}"
```

#### Tool Decorators
```python
from smolagents.tools import tool

@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b
```

#### Loading Tools
```python
from smolagents.tools import load_tool, ToolCollection

# Load from Hub
tool = load_tool("huggingface-tools/text-classification")

# Load collection
collection = ToolCollection.from_hub("huggingface-tools/default-tools")

# From MCP server
collection = ToolCollection.from_mcp({
    "command": "npx",
    "args": ["@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
})

# From LangChain
from smolagents.tools import Tool
tool = Tool.from_langchain(langchain_tool)

# From Gradio Space
tool = Tool.from_space("huggingface-tools/text-classification")
```

### 4. Default Tools (`smolagents.default_tools`)

```python
from smolagents.default_tools import (
    PythonInterpreterTool, FinalAnswerTool, UserInputTool,
    DuckDuckGoSearchTool, GoogleSearchTool, WebSearchTool,
    VisitWebpageTool, WikipediaSearchTool, SpeechToTextTool
)

# Python execution
python_tool = PythonInterpreterTool()

# Web search
search_tool = DuckDuckGoSearchTool(max_results=5)
web_tool = VisitWebpageTool(max_output_length=5000)

# Wikipedia
wiki_tool = WikipediaSearchTool()

# User interaction
user_input_tool = UserInputTool()
final_answer_tool = FinalAnswerTool()
```

### 5. Memory & Monitoring (`smolagents.memory`, `smolagents.monitoring`)

```python
from smolagents.memory import AgentMemory, ActionStep, PlanningStep
from smolagents.monitoring import AgentLogger, Monitor, TokenUsage

# Access agent memory
memory = agent.memory
steps = memory.get_succinct_steps()
code = memory.return_full_code()

# Monitoring
logger = AgentLogger()
monitor = Monitor()
token_usage = monitor.get_total_token_counts()
```

### 6. Code Execution (`smolagents.local_python_executor`, `smolagents.remote_executors`)

```python
from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.remote_executors import E2BExecutor, DockerExecutor, WasmExecutor

# Local execution
executor = LocalPythonExecutor()

# Remote execution
e2b_executor = E2BExecutor()  # Requires E2B API key
docker_executor = DockerExecutor()  # Requires Docker
wasm_executor = WasmExecutor()  # WebAssembly sandbox

# Execute code
result = executor.run_code_raise_errors("print('Hello, world!')")
```

### 7. Utilities (`smolagents.utils`)

```python
from smolagents.utils import (
    AgentError, AgentParsingError, AgentExecutionError,
    parse_json_blob, extract_code_from_text, encode_image_base64
)

# Error handling
try:
    result = agent.run(task)
except AgentError as e:
    print(f"Agent error: {e}")

# Utility functions
json_data, rest = parse_json_blob(text)
code = extract_code_from_text(text, ("```python", "```"))
base64_img = encode_image_base64(image)
```

## Advanced Usage

### Multi-Modal Agents
```python
from smolagents import CodeAgent, HfApiModel
from PIL import Image

model = HfApiModel("meta-llama/Llama-3.2-11B-Vision-Instruct")
agent = CodeAgent(tools=[PythonInterpreterTool()], model=model)

image = Image.open("chart.png")
result = agent.run(
    "Analyze this chart and extract the key insights",
    images=[image]
)
```

### Custom Prompt Templates
```python
from smolagents.agents import PromptTemplates

templates = PromptTemplates(
    system_prompt="You are a helpful assistant specialized in data analysis.",
    user_prompt_template="Task: {task}\nPlease solve this step by step."
)

agent = CodeAgent(tools=tools, model=model, prompt_templates=templates)
```

### Streaming Responses
```python
for chunk in agent.run("Analyze this data", stream=True):
    print(chunk, end="")
```

### Agent Persistence
```python
# Save agent
agent.save("./my_agent")

# Load agent
agent = CodeAgent.from_dict(saved_dict)

# Hub operations
agent.push_to_hub("username/my-agent")
loaded_agent = CodeAgent.from_hub("username/my-agent")
```

## Error Handling

```python
from smolagents.utils import (
    AgentError, AgentParsingError, AgentExecutionError,
    AgentMaxStepsError, AgentToolCallError
)

try:
    result = agent.run(task)
except AgentMaxStepsError:
    print("Agent exceeded maximum steps")
except AgentToolCallError as e:
    print(f"Tool call error: {e}")
except AgentExecutionError as e:
    print(f"Execution error: {e}")
except AgentParsingError as e:
    print(f"Parsing error: {e}")
```

## Best Practices

1. **Choose the right agent type:**
   - `CodeAgent`: For complex reasoning and code generation
   - `ToolCallingAgent`: For structured tool usage
   - `MultiStepAgent`: For custom step-by-step workflows

2. **Secure execution:**
   - Use remote executors (`e2b`, `docker`, `wasm`) for untrusted code
   - Limit authorized imports for local execution

3. **Tool selection:**
   - Start with default tools for common tasks
   - Create custom tools for domain-specific needs
   - Use tool collections for related functionality

4. **Model selection:**
   - Local models for privacy and control
   - API models for performance and capabilities
   - Consider model-specific features (vision, tool calling)

5. **Memory management:**
   - Monitor token usage with `Monitor`
   - Use `agent.memory.reset()` for fresh starts
   - Access conversation history via `memory.get_full_steps()`

