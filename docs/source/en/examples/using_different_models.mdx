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
7. [`OpenAIServerModel`]: Provides access to any provider that implements an OpenAI-compatible API
8. [`AzureOpenAIServerModel`]: Uses Azure's OpenAI service
9. [`AmazonBedrockServerModel`]: Connects to AWS Bedrock's API

## Using Google Gemini Models

As explained in the Google Gemini API documentation (https://ai.google.dev/gemini-api/docs/openai),
Google provides an OpenAI-compatible API for Gemini models, allowing you to use the [`OpenAIServerModel`]
with Gemini models by setting the appropriate base URL.

First, install the required dependencies:
```bash
pip install smolagents[openai]
```

Then, [get a Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) and set it in your code:
```python
GEMINI_API_KEY = <YOUR-GEMINI-API-KEY>
```

Now, you can initialize the Gemini model using the `OpenAIServerModel` class
and setting the `api_base` parameter to the Gemini API base URL:
```python
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    # Google Gemini OpenAI-compatible API base URL
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)
```

## Using OpenRouter Models

OpenRouter provides access to a wide variety of language models through a unified OpenAI-compatible API.
You can use the [`OpenAIServerModel`] to connect to OpenRouter by setting the appropriate base URL.

First, install the required dependencies:
```bash
pip install smolagents[openai]
```

Then, [get an OpenRouter API key](https://openrouter.ai/keys) and set it in your code:
```python
OPENROUTER_API_KEY = <YOUR-OPENROUTER-API-KEY>
```

Now, you can initialize any model available on OpenRouter using the `OpenAIServerModel` class:
```python
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    # You can use any model ID available on OpenRouter
    model_id="openai/gpt-4o",
    # OpenRouter API base URL
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
```
