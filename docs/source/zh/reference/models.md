# 模型

<Tip warning={true}>

Smolagents 是一个实验性 API，其可能会随时发生更改。由于 API 或底层模型可能会变化，智能体返回的结果可能会有所不同。

</Tip>

要了解有关智能体和工具的更多信息，请务必阅读[入门指南](../index)。此页面包含底层类的 API 文档。

## 模型

您可以自由创建和使用自己的模型为智能体提供支持。

您可以使用任何 `model` 可调用对象作为智能体的模型，只要满足以下条件：
1. 它遵循[消息格式](./chat_templating)（`List[Dict[str, str]]`），将其作为输入 `messages`，并返回一个 `str`。
2. 它在生成的序列到达 `stop_sequences` 参数中指定的内容之前停止生成输出。

要定义您的 LLM，可以创建一个 `custom_model` 方法，该方法接受一个 [messages](./chat_templating) 列表，并返回一个包含 `.content` 属性的对象，其中包含生成的文本。此可调用对象还需要接受一个 `stop_sequences` 参数，用于指示何时停止生成。

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

model_id = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(model=model_id)

def custom_model(messages, stop_sequences=["Task"]):
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message
    return answer
```

此外，`custom_model` 还可以接受一个 `grammar` 参数。如果在智能体初始化时指定了 `grammar`，则此参数将在调用模型时传递，以便进行[约束生成](https://huggingface.co/docs/text-generation-inference/conceptual/guidance)，从而强制生成格式正确的智能体输出。

### TransformersModel

为了方便起见，我们添加了一个 `TransformersModel`，该模型通过为初始化时指定的 `model_id` 构建一个本地 `transformers` pipeline 来实现上述功能。

```python
from smolagents import TransformersModel

model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}], stop_sequences=["great"]))
```
```text
>>> What a
```

> [!TIP]
> 您必须在机器上安装 `transformers` 和 `torch`。如果尚未安装，请运行 `pip install smolagents[transformers]`。

[[autodoc]] TransformersModel

### InferenceClientModel

`InferenceClientModel` 封装了 huggingface_hub 的 [InferenceClient](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference)，用于执行 LLM。它支持 HF 的 [Inference API](https://huggingface.co/docs/api-inference/index) 以及 Hub 上所有可用的[Inference Providers](https://huggingface.co/blog/inference-providers)。

```python
from smolagents import InferenceClientModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = InferenceClientModel()
print(model(messages))
```
```text
>>> Of course! If you change your mind, feel free to reach out. Take care!
```
[[autodoc]] InferenceClientModel

### LiteLLMModel

`LiteLLMModel` 利用 [LiteLLM](https://www.litellm.ai/) 支持来自不同提供商的 100+ 个 LLM。您可以在模型初始化时传递 `kwargs`，这些参数将在每次使用模型时被使用，例如下面的示例中传递了 `temperature`。

```python
from smolagents import LiteLLMModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", temperature=0.2, max_tokens=10)
print(model(messages))
```

[[autodoc]] LiteLLMModel

### OpenAIServerModel

此类允许您调用任何 OpenAIServer 兼容模型。
以下是设置方法（您可以自定义 `api_base` URL 指向其他服务器）：
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

`AzureOpenAIServerModel` 允许您连接到任何 Azure OpenAI 部署。

下面是设置示例，请注意，如果已经设置了相应的环境变量，您可以省略 `azure_endpoint`、`api_key` 和 `api_version` 参数——环境变量包括 `AZURE_OPENAI_ENDPOINT`、`AZURE_OPENAI_API_KEY` 和 `OPENAI_API_VERSION`。

请注意，`OPENAI_API_VERSION` 没有 `AZURE_` 前缀，这是由于底层 [openai](https://github.com/openai/openai-python) 包的设计所致。

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
> 您必须在机器上安装 `mlx-lm`。如果尚未安装，请运行 `pip install smolagents[mlx-lm]`。

[[autodoc]] MLXModel
