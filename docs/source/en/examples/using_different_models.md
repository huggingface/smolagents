# Using different models

[[open-in-colab]]

`smolagents` provides a flexible framework that allows you to use various language models from different providers.
This guide will show you how to use different model types with your agents.

## Available model types

`smolagents` supports several model types out of the box:
1. [`InferenceClientModel`]: Uses Hugging Face's Inference API to access models
2. [`TransformersModel`]: Runs models locally using the Transformers library
3. [`VLLMModel`]: Uses vLLM for fast inference with optimized serving
4. [`MLXModel`]: Optimized for Apple Silicon devices using MLX
5. [`LiteLLMModel`]: Provides access to hundreds of LLMs through LiteLLM
6. [`LiteLLMRouterModel`]: Distributes requests among multiple models
7. [`OpenAIModel`]: Provides access to any provider that implements an OpenAI-compatible API
8. [`AzureOpenAIModel`]: Uses Azure's OpenAI service
9. [`AmazonBedrockModel`]: Connects to AWS Bedrock's API

All model classes support passing additional keyword arguments (like `temperature`, `max_tokens`, `top_p`, etc.) directly at instantiation time.
These parameters are automatically forwarded to the underlying model's completion calls, allowing you to configure model behavior such as creativity, response length, and sampling strategies.

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
inference server for Apple Silicon, built on Apple's MLX framework. It runs
MLX-quantized models with streaming and tool calling, and ships day-0 support
for recent open models such as Gemma 4 and Qwen3.5.

First, install the required dependencies:
```bash
pip install 'smolagents[openai]'
pip install rapid-mlx
```

Start a local Rapid-MLX server:
```bash
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit
# Server listens on http://localhost:8000/v1
```

Then connect to it from smolagents using the [`OpenAIModel`] class:
```python
from smolagents import CodeAgent, OpenAIModel

model = OpenAIModel(
    model_id="mlx-community/Qwen3.5-4B-MLX-4bit",
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
)

agent = CodeAgent(tools=[], model=model)
agent.run("What is the 12th Fibonacci number?")
```

Rapid-MLX includes 18 built-in tool-call parsers (hermes, llama3, qwen,
glm47, minimax, gemma4, ...), so smolagents' tool/function calling works
with the supported open-weight models.
