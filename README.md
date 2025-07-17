<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<p align="center">
    <!-- Uncomment when CircleCI is set up
    <a href="https://circleci.com/gh/huggingface/accelerate"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master"></a>
    -->
    <a href="https://github.com/huggingface/smolagents/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/smolagents.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/smolagents"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/smolagents/index.html.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/smolagents/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/smolagents.svg"></a>
    <a href="https://github.com/huggingface/smolagents/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
</p>

<h3 align="center">
  <div style="display:flex;flex-direction:row;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/smolagents.png" alt="Hugging Face mascot as James Bond" width=400px>
    <p>Agents that think in code!</p>
  </div>
</h3>

`smolagents` is a lightweight library for building autonomous agents that solve tasks using code and tools. Agents run as independent processes and communicate directly via message queues using the `SendMessageTool` and `ReceiveMessagesTool`. Each agent maintains its own queue in a shared dictionary for decentralized task processing.

✨ **Simplicity**: The core logic fits in ~1,000 lines of code (see [agents.py](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py)), keeping abstractions minimal.

🧑‍💻 **First-class support for Code Agents**: Our [`CodeAgent`](https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent) writes actions as Python code, executed securely in sandboxed environments via [E2B](https://e2b.dev/) or Docker.

🤗 **Hub integrations**: Share or pull tools and agents to/from the Hub for instant collaboration (see [Tool.from_hub](https://huggingface.co/docs/smolagents/reference/tools#smolagents.Tool.from_hub)).

🌐 **Model-agnostic**: Supports any LLM, including local `transformers` or `ollama` models, [HF inference providers](https://huggingface.co/blog/inference-providers), or models from OpenAI, Anthropic, and others via [LiteLLM](https://www.litellm.ai/).

👁️ **Modality-agnostic**: Agents handle text, vision, video, and audio inputs (see [vision tutorial](https://huggingface.co/docs/smolagents/examples/web_browser)).

🛠️ **Tool-agnostic**: Use tools from [MCP servers](https://huggingface.co/docs/smolagents/reference/tools#smolagents.ToolCollection.from_mcp), [LangChain](https://huggingface.co/docs/smolagents/reference/tools#smolagents.Tool.from_langchain), or [Hub Spaces](https://huggingface.co/docs/smolagents/reference/tools#smolagents.Tool.from_space).

Full documentation is available [here](https://huggingface.co/docs/smolagents/index).

> [!NOTE]
> Check our [launch blog post](https://huggingface.co/blog/smolagents) to learn more about `smolagents`!

## Quick Demo

Install the package with default tools:

```bash
pip install smolagents[toolkit]
```

Set your Hugging Face API key:

```bash
export HF_TOKEN=your_huggingface_api_key_here
```

Run multiple agents to solve a task collaboratively:

```python
from multiprocessing import Manager, Process
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    WebSearchTool,
    SendMessageTool,
    ReceiveMessagesTool,
)

def start_agent(agent_id, queue_dict, task=None):
    model = InferenceClientModel()
    tools = [
        WebSearchTool(),
        SendMessageTool(queue_dict, agent_id),
        ReceiveMessagesTool(queue_dict, agent_id),
    ]
    agent = CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=["numpy", "pandas"],
    )
    agent.run(task=task)

if __name__ == "__main__":
    task = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
    num_agents = 2
    with Manager() as manager:
        queue_dict = manager.dict()
        for i in range(num_agents):
            queue_dict[i] = manager.Queue()
        processes = [
            Process(target=start_agent, args=(i, queue_dict, task if i == 0 else None))
            for i in range(num_agents)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
```

This launches two `CodeAgent` instances. Agent 0 processes the task and may send subtasks (e.g., code or search results) to Agent 1 via message queues using `SendMessageTool` and `ReceiveMessagesTool`. Agents use tools like `web_search` or `python_interpreter` and return results with `final_answer()`.

You can share your agent to the Hub as a Space repository:

```python
agent.push_to_hub("m-ric/my_agent")
# agent.from_hub("m-ric/my_agent") to load an agent from Hub
```

`smolagents` is LLM-agnostic. Switch the model as needed:

<details>
<summary> <b>InferenceClientModel (HF inference providers)</b></summary>

```python
from smolagents import InferenceClientModel

model = InferenceClientModel(
    model_id="deepseek-ai/DeepSeek-R1",
    provider="together",
)
```
</details>
<details>
<summary> <b>LiteLLM (100+ LLMs)</b></summary>

```python
from smolagents import LiteLLMModel

model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    temperature=0.2,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```
</details>
<details>
<summary> <b>OpenAI-compatible servers: Together AI</b></summary>

```python
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1/",
    api_key=os.environ["TOGETHER_API_KEY"],
)
```
</details>
<details>
<summary> <b>OpenAI-compatible servers: OpenRouter</b></summary>

```python
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="openai/gpt-4o",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```
</details>
<details>
<summary> <b>Local `transformers` model</b></summary>

```python
from smolagents import TransformersModel

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="auto"
)
```
</details>
<details>
<summary> <b>Azure models</b></summary>

```python
import os
from smolagents import AzureOpenAIServerModel

model = AzureOpenAIServerModel(
    model_id=os.environ.get("AZURE_OPENAI_MODEL"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("OPENAI_API_VERSION")
)
```
</details>
<details>
<summary> <b>Amazon Bedrock models</b></summary>

```python
import os
from smolagents import AmazonBedrockServerModel

model = AmazonBedrockServerModel(
    model_id=os.environ.get("AMAZON_BEDROCK_MODEL_ID")
)
```
</details>

## CLI

Run agents from the CLI using `smolagent` or `webagent`.

`smolagent` runs multiple `CodeAgent` or `ToolCallingAgent` instances that collaborate via message queues:

```bash
smolagent "Plan a trip to Tokyo, Kyoto, and Osaka between Mar 28 and Apr 7." --num-agents 2 --model-type "InferenceClientModel" --model-id "Qwen/Qwen2.5-Coder-32B-Instruct" --imports "pandas numpy" --tools "web_search"
```

`webagent` is a specific web-browsing agent using [helium](https://github.com/mherrmann/helium) (read more [here](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py)):

```bash
webagent "Go to xyz.com/men, get to sale section, click the first clothing item. Get the product details and price, return them. Note that I'm shopping from France" --model-type "LiteLLMModel" --model-id "gpt-4o"
```

## How Do Agents Work?

Agents in `smolagents` run as independent processes. Each agent has a queue inside a shared dictionary, and they communicate by sending messages with `SendMessageTool` and retrieving them with `ReceiveMessagesTool`. This decentralized approach eliminates the need for a centralized ReAct loop.

[`CodeAgent`](https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent) writes actions as Python code snippets, executed securely in sandboxed environments (e.g., [E2B](https://e2b.dev/) or Docker). Code-based actions [use 30% fewer steps](https://huggingface.co/papers/2402.01030) and [achieve higher performance](https://huggingface.co/papers/2411.01747) compared to traditional tool-calling methods.

[`ToolCallingAgent`](https://huggingface.co/docs/smolagents/reference/agents#smolagents.ToolCallingAgent) writes actions as JSON blobs, suitable for tasks requiring structured tool calls. Both agent types support collaborative workflows via message queues.

Example workflow for two agents solving "Compute 5 + 3":

- **Agent 0**: Generates code (`result = 5 + 3; print(result)`) and sends it to Agent 1.
- **Agent 1**: Receives the code, executes it using `python_interpreter`, and returns the result with `final_answer()`.

See [our intro to agents](https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents) for more details.

## How Smol Is This Library?

The core logic in `agents.py` is <1,000 lines, minimizing abstractions. We support `CodeAgent` (Python code actions), `ToolCallingAgent` (JSON actions), multi-agent collaboration, tool collections, remote code execution, and vision models. The framework handles complex tasks like consistent code formatting, parsing, and secure execution, but you can hack into the source code to use only what you need.

## How Strong Are Open Models for Agentic Workflows?

We’ve benchmarked [`CodeAgent`](https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent) with leading models on [this benchmark](https://huggingface.co/datasets/m-ric/agents_medium_benchmark_2), combining varied challenges. [See the benchmarking code](https://github.com/huggingface/smolagents/blob/main/examples/smolagents_benchmark/run.py) for details. Open-source models like DeepSeek-R1 often outperform closed-source models in agentic tasks.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/benchmark_code_agents.jpeg" alt="benchmark of different models on agentic workflows. Open model DeepSeek-R1 beats closed-source models." width=60% max-width=500px>
</p>

## Security

Security is critical for code-executing agents. We provide:
- Sandboxed execution via [E2B](https://e2b.dev/) or Docker.
- Best practices for secure agent execution.

See our [Security Policy](SECURITY.md) for vulnerability reporting and secure execution guidelines.

## Contribute

Everyone is welcome to contribute. See our [contribution guide](https://github.com/huggingface/smolagents/blob/main/CONTRIBUTING.md).

## Cite smolagents

If you use `smolagents` in your publication, please cite it:

```bibtex
@Misc{smolagents,
  title =        {`smolagents`: a smol library to build great agentic systems.},
  author =       {Aymeric Roucher and Albert Villanova del Moral and Thomas Wolf and Leandro von Werra and Erik Kaunismäki},
  howpublished = {\url{https://github.com/huggingface/smolagents}},
  year =         {2025}
}
```