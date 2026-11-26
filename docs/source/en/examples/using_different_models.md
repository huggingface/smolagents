# Using different models

[[open-in-colab]]

`smolagents` provides a flexible framework that allows you to use various language models from different providers.
This guide will show you how to use different model types with your agents.

## Available model types

`smolagents` supports several model types out of the box:
1. [`InferenceClientModel`]: Uses Hugging Face's Inference API to access models
2. [`TransformersModel`]: Runs models locally using the Transformers library
3. [`LlamaCppModel`]: Connects to models served by an already-running Llama App
4. [`NvidiaModel`]: Connects to NVIDIA NIM, hosted or self-hosted
5. [`VLLMModel`]: Uses vLLM for fast inference with optimized serving
6. [`MLXModel`]: Optimized for Apple Silicon devices using MLX
7. [`LiteLLMModel`]: Provides access to hundreds of LLMs through LiteLLM
8. [`LiteLLMRouterModel`]: Distributes requests among multiple models
9. [`OpenAIModel`]: Provides access to any provider that implements an OpenAI-compatible API
10. [`AzureOpenAIModel`]: Uses Azure's OpenAI service
11. [`AmazonBedrockModel`]: Connects to AWS Bedrock's API

All model classes support passing additional keyword arguments (like `temperature`, `max_tokens`, `top_p`, etc.) directly at instantiation time.
These parameters are automatically forwarded to the underlying model's completion calls, allowing you to configure model behavior such as creativity, response length, and sampling strategies.

## Using llama.cpp models with Llama App

[`LlamaCppModel`] connects smolagents to [Llama App](https://llama.app/docs/serve), with the app managing the server, model weights, and runtime. Start with Llama App already serving on your machine; see the [official serving guide](https://llama.app/docs/serve) if you need setup instructions.

Install the OpenAI-compatible client dependency:

```bash
pip install 'smolagents[openai]'
```

List the model IDs in your own Llama App catalog:

```bash
curl -s http://127.0.0.1:9931/v1/models | jq -r '.data[].id'
```

Copy the full ID of a tool-capable model from the output, including any organization prefix and quantization suffix. The app can load a catalog model on its first request, so it does not need to be loaded in advance. Then replace only the `model_id` below with that ID:

```python
from smolagents import LlamaCppModel, ToolCallingAgent, tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of an order.

    Args:
        order_id: The order identifier to look up.
    """
    return f"Order {order_id} is ready for pickup."


model = LlamaCppModel(model_id="ggml-org/gpt-oss-20b-GGUF:MXFP4")
agent = ToolCallingAgent(tools=[lookup_order], model=model)
result = agent.run("Look up order A-10042 and tell me its status.")
print(result)
```

The model ID is sent unchanged to Llama App. The default endpoint is `http://127.0.0.1:9931/v1`; pass `api_base` for a different host or port, and `api_key` if your instance requires authentication.

## Using NVIDIA NIM models

[`NvidiaModel`] connects smolagents to [NVIDIA NIM](https://build.nvidia.com/) through its OpenAI-compatible API, either the hosted NVIDIA API catalog or a self-hosted NIM.

Install the OpenAI-compatible client dependency:

```bash
pip install 'smolagents[openai]'
```

Create an API key on [build.nvidia.com](https://build.nvidia.com/) and export it as `NVIDIA_API_KEY`. List the model IDs available to you:

```bash
curl -s https://integrate.api.nvidia.com/v1/models | jq -r '.data[].id'
```

Then pass the full ID of a tool-capable model as `model_id`:

```python
from smolagents import NvidiaModel, ToolCallingAgent, tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of an order.

    Args:
        order_id: The order identifier to look up.
    """
    return f"Order {order_id} is ready for pickup."


model = NvidiaModel(model_id="nvidia/nemotron-3.5-super-vl-120b-a12b")
agent = ToolCallingAgent(tools=[lookup_order], model=model)
result = agent.run("Look up order A-10042 and tell me its status.")
print(result)
```

Other models you can use include `nvidia/nemotron-3.5-lightning-30b-a3b` and `nvidia/nemotron-3-ultra-550b-a55b`. The default endpoint is `https://integrate.api.nvidia.com/v1`; pass `api_base` to use a self-hosted NIM, and `api_key` to set the key explicitly instead of reading `NVIDIA_API_KEY`.

## Using Google Gemini Models

As explained in the Google Gemini API documentation (https://ai.google.dev/gemini-api/docs/openai),
Google provides an OpenAI-compatible API for Gemini models, allowing you to use the [`OpenAIModel`]
with Gemini models by setting the appropriate base URL.

First, install the required dependencies:
```bash
pip install 'smolagents[openai]'
```

Then, [get a Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) and set it in your code:
```python
GEMINI_API_KEY = <YOUR-GEMINI-API-KEY>
```

Now, you can initialize the Gemini model using the `OpenAIModel` class
and setting the `api_base` parameter to the Gemini API base URL:
```python
from smolagents import OpenAIModel

model = OpenAIModel(
    model_id="gemini-2.0-flash",
    # Google Gemini OpenAI-compatible API base URL
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)
```

## Using OpenRouter Models

OpenRouter provides access to a wide variety of language models through a unified OpenAI-compatible API.
You can use the [`OpenAIModel`] to connect to OpenRouter by setting the appropriate base URL.

First, install the required dependencies:
```bash
pip install 'smolagents[openai]'
```

Then, [get an OpenRouter API key](https://openrouter.ai/keys) and set it in your code:
```python
OPENROUTER_API_KEY = <YOUR-OPENROUTER-API-KEY>
```

Now, you can initialize any model available on OpenRouter using the `OpenAIModel` class:
```python
from smolagents import OpenAIModel

model = OpenAIModel(
    # You can use any model ID available on OpenRouter
    model_id="openai/gpt-4o",
    # OpenRouter API base URL
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
```

## Using xAI's Grok Models

xAI's Grok models can be accessed through [`LiteLLMModel`].

Some models (such as "grok-4" and "grok-3-mini") don't support the `stop` parameter, so you'll need to use
`REMOVE_PARAMETER` to exclude it from API calls.

First, install the required dependencies:
```bash
pip install smolagents[litellm]
```

Then, [get an xAI API key](https://console.x.ai/) and set it in your code:
```python
XAI_API_KEY = <YOUR-XAI-API-KEY>
```

Now, you can initialize Grok models using the `LiteLLMModel` class and remove the `stop` parameter if applicable:
```python
from smolagents import LiteLLMModel, REMOVE_PARAMETER

# Using Grok-4
model = LiteLLMModel(
    model_id="xai/grok-4",
    api_key=XAI_API_KEY,
    stop=REMOVE_PARAMETER,  # Remove stop parameter as grok-4 model doesn't support it
    temperature=0.7
)

# Or using Grok-3-mini
model_mini = LiteLLMModel(
    model_id="xai/grok-3-mini",
    api_key=XAI_API_KEY,
    stop=REMOVE_PARAMETER,  # Remove stop parameter as grok-3-mini model doesn't support it
    max_tokens=1000
)
```

## Using a local Apple Silicon model with Rapid-MLX

[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) is an OpenAI-compatible
inference server for Apple Silicon, built on Apple's MLX framework. It serves MLX models through the OpenAI API with tool calling.

First, install the required dependencies:
```bash
pip install 'smolagents[openai]'
pip install rapid-mlx
```

Start a local Rapid-MLX server:
```bash
rapid-mlx serve qwen3.5-4b-4bit --port 8000
# OpenAI-compatible API: http://localhost:8000/v1
```

Then connect to it from smolagents using the [`OpenAIModel`] class:
```python
from smolagents import CodeAgent, OpenAIModel

model = OpenAIModel(
    model_id="qwen3.5-4b-4bit",
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
)

agent = CodeAgent(tools=[], model=model)
agent.run("What is the 12th Fibonacci number?")
```

Rapid-MLX supports tool calling with compatible models.
