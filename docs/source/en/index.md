# `smolagents`

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png" style="max-width:700px"/>
</div>

## What is smolagents?

`smolagents` is an open-source Python library designed to make it extremely easy to build and run agents using just a few lines of code.

Key features of `smolagents` include:

✨ **Simplicity**: The logic for agents fits in ~thousand lines of code. We kept abstractions to their minimal shape above raw code!

🧑‍💻 **First-class support for Code Agents**: [`CodeAgent`](reference/agents#smolagents.CodeAgent) writes its actions in code (as opposed to "agents being used to write code") to invoke tools or perform computations, enabling natural composability (function nesting, loops, conditionals). To make it secure, we support [executing in sandboxed environment](tutorials/secure_code_execution) via [E2B](https://e2b.dev/) or via Docker.

📡 **Common Tool-Calling Agent Support**: In addition to CodeAgents, [`ToolCallingAgent`](reference/agents#smolagents.ToolCallingAgent) supports usual JSON/text-based tool-calling for scenarios where that paradigm is preferred.

🤗 **Hub integrations**: Seamlessly share and load agents and tools to/from the Hub as Gradio Spaces.

🌐 **Model-agnostic**: Easily integrate any large language model (LLM), whether it's hosted on the Hub via [Inference providers](https://huggingface.co/docs/inference-providers/index), accessed via APIs such as OpenAI, Anthropic, or many others via LiteLLM integration, or run locally using Transformers or Ollama. Powering an agent with your preferred LLM is straightforward and flexible.

👁️ **Modality-agnostic**: Beyond text, agents can handle vision, video, and audio inputs, broadening the range of possible applications. Check out [this tutorial](examples/web_browser) for vision.

🛠️ **Tool-agnostic**: You can use tools from any [MCP server](reference/tools#smolagents.ToolCollection.from_mcp), from [LangChain](reference/tools#smolagents.Tool.from_langchain), you can even use a [Hub Space](reference/tools#smolagents.Tool.from_space) as a tool.

💻 **CLI Tools**: Comes with command-line utilities (smolagent, webagent) for quickly running agents without writing boilerplate code.

## Quickstart

[[open-in-colab]]

Get started with smolagents in just a few minutes! This guide will show you how to create and run your first agent.

### Installation

Install smolagents with pip:

```bash
pip install smolagents[toolkit]  # Includes default tools like web search
```

### Create Your First Agent

Here's a minimal example to create and run an agent:

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize a model (using Hugging Face Inference API)
model = InferenceClientModel()  # Uses a default model

# Create an agent with no tools
agent = CodeAgent(tools=[], model=model)

# Run the agent with a task
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)
```

That's it! Your agent will use Python code to solve the task and return the result.

### Adding Tools

Let's make our agent more capable by adding some tools:

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

# Now the agent can search the web!
result = agent.run("What is the current weather in Paris?")
print(result)
```

### Using Different Models

You can use various models with your agent:

```python
# Using a specific model from Hugging Face
model = InferenceClientModel(model_id="meta-llama/Llama-2-70b-chat-hf")

# Using OpenAI/Anthropic (requires smolagents[litellm])
from smolagents import LiteLLMModel
model = LiteLLMModel(model_id="gpt-4")

# Using local models (requires smolagents[transformers])
from smolagents import TransformersModel
model = TransformersModel(model_id="meta-llama/Llama-2-7b-chat-hf")
```

## Next Steps

- Learn how to set up smolagents with various models and tools in the [Installation Guide](installation)
- Check out the [Guided Tour](guided_tour) for more advanced features
- Learn about [building custom tools](tutorials/tools)
- Explore [secure code execution](tutorials/secure_code_execution)
- See how to create [multi-agent systems](tutorials/building_good_agents)

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guided tour</div>
      <p class="text-gray-700">Learn the basics and become familiar with using Agents. Start here if you are using Agents for the first time!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">How-to guides</div>
      <p class="text-gray-700">Practical guides to help you achieve a specific goal: create an agent to generate and test SQL queries!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Conceptual guides</div>
      <p class="text-gray-700">High-level explanations for building a better understanding of important topics.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutorials</div>
      <p class="text-gray-700">Horizontal tutorials that cover important aspects of building agents.</p>
    </a>
  </div>
</div>
