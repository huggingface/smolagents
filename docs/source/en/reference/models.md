# Models

<Tip warning={true}>

Smolagents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

To learn more about agents and tools make sure to read the [introductory guide](../index). This page
contains the API docs for the underlying classes.

## Models

### Your custom Model

You're free to create and use your own models to power your agent.

You could subclass the base `Model` class to create a model for your agent.
The main criteria is to subclass the `generate` method, with these two criteria:
1. It follows the [messages format](./chat_templating) (`List[Dict[str, str]]`) for its input `messages`, and it returns an object with a `.content` attribute.
2. It stops generating outputs at the sequences passed in the argument `stop_sequences`.

For defining your LLM, you can make a `CustomModel` class that inherits from the base `Model` class.
It should have a generate method that takes a list of [messages](./chat_templating) and returns an object with a .content attribute containing the text. The `generate` method also needs to accept a `stop_sequences` argument that indicates when to stop generating.

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

model_id = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(model=model_id)

class CustomModel(Model):
    def generate(messages, stop_sequences=["Task"]):
        response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1024)
        answer = response.choices[0].message
        return answer

custom_model = CustomModel()
```

Additionally, `generate` can also take a `grammar` argument to allow [constrained generation](https://huggingface.co/docs/text-generation-inference/conceptual/guidance) in order to force properly-formatted agent outputs.

### TransformersModel

For convenience, we have added a `TransformersModel` that implements the points above by building a local `transformers` pipeline for the model_id given at initialization.

```python
from smolagents import TransformersModel

model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}], stop_sequences=["great"]))
```
```text
>>> What a
```

> [!TIP]
> You must have `transformers` and `torch` installed on your machine. Please run `pip install smolagents[transformers]` if it's not the case.

[[autodoc]] TransformersModel

### InferenceClientModel

The `InferenceClientModel` wraps huggingface_hub's [InferenceClient](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference) for the execution of the LLM. It supports all [Inference Providers](https://huggingface.co/docs/inference-providers/index) available on the Hub: Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, and more.

You can also set a rate limit in requests per minute by using the `requests_per_minute` argument.

```python
from smolagents import InferenceClientModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = InferenceClientModel(provider="novita", requests_per_minute=60)
print(model(messages))
```
```text
>>> Of course! If you change your mind, feel free to reach out. Take care!
```
[[autodoc]] InferenceClientModel

### LiteLLMModel

The `LiteLLMModel` leverages [LiteLLM](https://www.litellm.ai/) to support 100+ LLMs from various providers.
You can pass kwargs upon model initialization that will then be used whenever using the model, for instance below we pass `temperature`. You can also set a rate limit in requests per minute by using the `requests_per_minute` argument.

```python
from smolagents import LiteLLMModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", temperature=0.2, max_tokens=10, requests_per_minute=60)
print(model(messages))
```

[[autodoc]] LiteLLMModel

### LiteLLMRouterModel

The `LiteLLMRouterModel` is a wrapper around the [LiteLLM Router](https://docs.litellm.ai/docs/routing) that leverages
advanced routing strategies: load-balancing across multiple deployments, prioritizing critical requests via queueing,
and implementing basic reliability measures such as cooldowns, fallbacks, and exponential backoff retries.

```python
from smolagents import LiteLLMRouterModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMRouterModel(
    model_id="llama-3.3-70b",
    model_list=[
        {
            "model_name": "llama-3.3-70b",
            "litellm_params": {"model": "groq/llama-3.3-70b", "api_key": os.getenv("GROQ_API_KEY")},
        },
        {
            "model_name": "llama-3.3-70b",
            "litellm_params": {"model": "cerebras/llama-3.3-70b", "api_key": os.getenv("CEREBRAS_API_KEY")},
        },
    ],
    client_kwargs={
        "routing_strategy": "simple-shuffle",
    },
)
print(model(messages))
```

[[autodoc]] LiteLLMRouterModel

### OpenAIServerModel

This class lets you call any OpenAIServer compatible model.
Here's how you can set it (you can customise the `api_base` url to point to another server):
```py
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

[[autodoc]] OpenAIServerModel

### AzureOpenAIServerModel

`AzureOpenAIServerModel` allows you to connect to any Azure OpenAI deployment. 

Below you can find an example of how to set it up, note that you can omit the `azure_endpoint`, `api_key`, and `api_version` arguments, provided you've set the corresponding environment variables -- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `OPENAI_API_VERSION`.

Pay attention to the lack of an `AZURE_` prefix for `OPENAI_API_VERSION`, this is due to the way the underlying [openai](https://github.com/openai/openai-python) package is designed. 

```py
import os

from smolagents import AzureOpenAIServerModel

model = AzureOpenAIServerModel(
    model_id = os.environ.get("AZURE_OPENAI_MODEL"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("OPENAI_API_VERSION")    
)
```

[[autodoc]] AzureOpenAIServerModel

### AmazonBedrockServerModel

`AmazonBedrockServerModel` helps you connect to Amazon Bedrock and run your agent with any available models.

Below is an example setup. This class also offers additional options for customization.

```py
import os

from smolagents import AmazonBedrockServerModel

model = AmazonBedrockServerModel(
    model_id = os.environ.get("AMAZON_BEDROCK_MODEL_ID"),
)
```

[[autodoc]] AmazonBedrockServerModel

### MLXModel


```python
from smolagents import MLXModel

model = MLXModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": "Ok!"}], stop_sequences=["great"]))
```
```text
>>> What a
```

> [!TIP]
> You must have `mlx-lm` installed on your machine. Please run `pip install smolagents[mlx-lm]` if it's not the case.

[[autodoc]] MLXModel

### VLLMModel

Model to use [vLLM](https://docs.vllm.ai/) for fast LLM inference and serving.

```python
from smolagents import VLLMModel

model = VLLMModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": "Ok!"}], stop_sequences=["great"]))
```

> [!TIP]
> You must have `vllm` installed on your machine. Please run `pip install smolagents[vllm]` if it's not the case.

[[autodoc]] VLLMModel
