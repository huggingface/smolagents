# Agents - 导览

[[open-in-colab]]

在本导览中，您将学习如何构建一个 agent（智能体），如何运行它，以及如何自定义它以使其更好地适应您的使用场景。

> [!TIP]
> 译者注：Agent 的业内术语是“智能体”。本译文将保留 agent，不作翻译，以带来更高效的阅读体验。(在中文为主的文章中，It's easier to 注意到英文。Attention Is All You Need!)

> [!TIP]
> 中文社区发布了关于 smolagents 的介绍和实践讲解视频(来源：[Issue#80](https://github.com/huggingface/smolagents/issues/80))，你可以访问[这里](https://www.youtube.com/watch?v=wwN3oAugc4c)进行观看！

### 构建您的 agent

要初始化一个最小化的 agent，您至少需要以下两个参数：

- `model`，一个为您的 agent 提供动力的文本生成模型 - 因为 agent 与简单的 LLM 不同，它是一个使用 LLM 作为引擎的系统。您可以使用以下任一选项：
    - [`TransformersModel`] 使用预初始化的 `transformers` 管道在本地机器上运行推理
    - [`InferenceClientModel`] 在底层使用 `huggingface_hub.InferenceClient`
    - [`LiteLLMModel`] 让您通过 [LiteLLM](https://docs.litellm.ai/) 调用 100+ 不同的模型！
    - [`AzureOpenAIServerModel`] 允许您使用部署在 [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service) 中的 OpenAI 模型。
    - [`MLXModel`] 可创建 [mlx-lm](https://pypi.org/project/mlx-lm/) 流水线，以便在本地机器上运行推理。

- `tools`，agent 可以用来解决任务的 `Tools` 列表。它可以是一个空列表。您还可以通过定义可选参数 `add_base_tools=True` 在您的 `tools` 列表之上添加默认工具箱。

一旦有了这两个参数 `tools` 和 `model`，您就可以创建一个 agent 并运行它。您可以使用任何您喜欢的 LLM，无论是通过 [Hugging Face API](https://huggingface.co/docs/api-inference/en/index)、[transformers](https://github.com/huggingface/transformers/)、[ollama](https://ollama.com/)、[LiteLLM](https://www.litellm.ai/)、[Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)，还是[mlx-lm](https://pypi.org/project/mlx-lm/).。

<hfoptions id="选择一个LLM">
<hfoption id="Hugging Face API">

Hugging Face API 可以免费使用而无需 token，但会有速率限制。

要访问受限模型或使用 PRO 账户提高速率限制，您需要设置环境变量 `HF_TOKEN` 或在初始化 `InferenceClientModel` 时传递 `token` 变量。

```python
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

model = InferenceClientModel(model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="本地Transformers模型">

```python
# !pip install smolagents[transformers]
from smolagents import CodeAgent, TransformersModel

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = TransformersModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="OpenAI或Anthropic API">

要使用 `LiteLLMModel`，您需要设置环境变量 `ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY`，或者在初始化时传递 `api_key` 变量。

```python
# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY") # 也可以使用 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Ollama">

```python
# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2", # 这个模型对于 agent 行为来说有点弱
    api_base="http://localhost:11434", # 如果需要可以替换为远程 open-ai 兼容服务器
    api_key="YOUR_API_KEY" # 如果需要可以替换为 API key
    num_ctx=8192 # https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Azure OpenAI">

要连接到 Azure OpenAI，您可以直接使用 `AzureOpenAIServerModel`，或使用 `LiteLLMModel` 并进行相应配置。

初始化 `AzureOpenAIServerModel` 实例时，需要传递模型部署名称，可选择以下任一种方式：1.传递 `azure_endpoint`、`api_key` 和 `api_version` 参数；2.设置环境变量 `AZURE_OPENAI_ENDPOINT`、`AZURE_OPENAI_API_KEY` 和 `OPENAI_API_VERSION`

```python
# !pip install smolagents[openai]
from smolagents import CodeAgent, AzureOpenAIServerModel

model = AzureOpenAIServerModel(model_id="gpt-4o-mini")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

也可按如下方式配置 `LiteLLMModel` 连接 Azure OpenAI：

- 将模型部署名称作为 `model_id` 参数传递，并确保其前缀为 `azure/`
- 确保设置环境变量 `AZURE_API_VERSION`
- 任选其一：1.传递 `api_base` 和 `api_key` 参数；2.设置环境变量 `AZURE_API_KEY` 和 `AZURE_API_BASE`

```python
import os
from smolagents import CodeAgent, LiteLLMModel

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-35-turbo-16k-deployment" # example of deployment name

os.environ["AZURE_API_KEY"] = "" # api_key
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2024-10-01-preview"

model = LiteLLMModel(model_id="azure/" + AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
   "Could you give me the 118th number in the Fibonacci sequence?",
)
```

</hfoption>
<hfoption id="mlx-lm">

```python
# !pip install smolagents[mlx-lm]
from smolagents import CodeAgent, MLXModel

mlx_model = MLXModel("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
agent = CodeAgent(model=mlx_model, tools=[], add_base_tools=True)

agent.run("Could you give me the 118th number in the Fibonacci sequence?")
```

</hfoption>
</hfoptions>

#### CodeAgent 和 ToolCallingAgent

[`CodeAgent`] 是我们的默认 agent。它将在每一步编写并执行 Python 代码片段。

默认情况下，执行是在您的本地环境中完成的。
这应该是安全的，因为唯一可以调用的函数是您提供的工具（特别是如果只有 Hugging Face 的工具）和一组预定义的安全函数，如 `print` 或 `math` 模块中的函数，所以您已经限制了可以执行的内容。

Python 解释器默认也不允许在安全列表之外导入，所以所有最明显的攻击都不应该成为问题。
您可以通过在初始化 [`CodeAgent`] 时将授权模块作为字符串列表传递给参数 `additional_authorized_imports` 来授权额外的导入：

```py
from smolagents import CodeAgent

model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

> [!WARNING]
> LLM 可以生成任意代码然后执行：不要添加任何不安全的导入！

如果生成的代码尝试执行非法操作或出现常规 Python 错误，执行将停止。

您也可以使用 [E2B 代码执行器](https://e2b.dev/docs#what-is-e2-b) 或 Docker 而不是本地 Python 解释器。对于 E2B，首先 [设置 `E2B_API_KEY` 环境变量](https://e2b.dev/dashboard?tab=keys)，然后在初始化 agent 时传递 `executor_type="e2b"`。对于 Docker，在初始化时传递 `executor_type="docker"`。

> [!TIP]
> 在 [该教程中](tutorials/secure_code_execution) 了解更多关于代码执行的内容。

我们还支持广泛使用的将动作编写为 JSON-like 块的方式：[`ToolCallingAgent`]，它的工作方式与 [`CodeAgent`] 非常相似，当然没有 `additional_authorized_imports`，因为它不执行代码：

```py
from smolagents import ToolCallingAgent

agent = ToolCallingAgent(tools=[], model=model)
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

### 检查 agent 运行

以下是一些有用的属性，用于检查运行后发生了什么：
- `agent.logs` 存储 agent 的细粒度日志。在 agent 运行的每一步，所有内容都会存储在一个字典中，然后附加到 `agent.logs` 中。
- 运行 `agent.write_memory_to_messages()` 会为 LLM 创建一个 agent 日志的内部内存，作为聊天消息列表。此方法会遍历日志的每一步，并仅存储它感兴趣的内容作为消息：例如，它会将系统提示和任务存储为单独的消息，然后对于每一步，它会将 LLM 输出存储为一条消息，工具调用输出存储为另一条消息。如果您想要更高级别的视图 - 但不是每个日志都会被此方法转录。

## 工具

工具是 agent 使用的原子函数。为了被 LLM 使用，它还需要一些构成其 API 的属性，这些属性将用于向 LLM 描述如何调用此工具：
- 名称
- 描述
- 输入类型和描述
- 输出类型

例如，您可以查看 [`PythonInterpreterTool`]：它有一个名称、描述、输入描述、输出类型和一个执行操作的 `forward` 方法。

当 agent 初始化时，工具属性用于生成工具描述，该描述被嵌入到 agent 的系统提示中。这让 agent 知道它可以使用哪些工具以及为什么。

### 默认工具箱

`smolagents` 附带了一个用于增强 agent 的默认工具箱，您可以在初始化时通过参数 `add_base_tools=True` 将其添加到您的 agent 中：

- **DuckDuckGo 网页搜索**：使用 DuckDuckGo 浏览器执行网页搜索。
- **Python 代码解释器**：在安全环境中运行 LLM 生成的 Python 代码。只有在使用 `add_base_tools=True` 初始化 [`ToolCallingAgent`] 时才会添加此工具，因为基于代码的 agent 已经可以原生执行 Python 代码
- **转录器**：基于 Whisper-Turbo 构建的语音转文本管道，将音频转录为文本。

您可以通过调用 [`load_tool`] 函数和要执行的任务手动使用工具。

```python
from smolagents import WebSearchTool

search_tool = WebSearchTool()
print(search_tool("Who's the current president of Russia?"))
```

### 创建一个新工具

您可以创建自己的工具，用于 Hugging Face 默认工具未涵盖的用例。
例如，让我们创建一个工具，返回 Hub 上给定任务下载量最多的模型。

您将从以下代码开始。

```python
from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(most_downloaded_model.id)
```

这段代码可以通过将其包装在一个函数中并添加 `tool` 装饰器快速转换为工具：
这不是构建工具的唯一方法：您可以直接将其定义为 [`Tool`] 的子类，这为您提供了更多的灵活性，例如初始化重型类属性的可能性。

让我们看看这两种选项的工作原理：

<hfoptions id="构建工具">
<hfoption id="使用@tool装饰一个函数">

```py
from smolagents import tool

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id
```

该函数需要：
- 一个清晰的名称。名称应该足够描述此工具的功能，以帮助为 agent 提供动力的 LLM。由于此工具返回任务下载量最多的模型，我们将其命名为 `model_download_tool`。
- 输入和输出的类型提示
- 一个描述，其中包括一个 'Args:' 部分，其中每个参数都被描述（这次没有类型指示，它将从类型提示中提取）。与工具名称一样，此描述是为您的 agent 提供动力的 LLM 的说明书，所以不要忽视它。
所有这些元素将在初始化时自动嵌入到 agent 的系统提示中：因此要努力使它们尽可能清晰！

> [!TIP]
> 此定义格式与 `apply_chat_template` 中使用的工具模式相同，唯一的区别是添加了 `tool` 装饰器：[这里](https://huggingface.co/blog/unified-tool-use#passing-tools-to-a-chat-template) 了解更多关于我们的工具使用 API。
</hfoption>
<hfoption id="子类化Tool">

```py
from smolagents import Tool

class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {"task": {"type": "string", "description": "The task for which to get the download count."}}
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return most_downloaded_model.id
```

子类需要以下属性：
- 一个清晰的 `name`。名称应该足够描述此工具的功能，以帮助为 agent 提供动力的 LLM。由于此工具返回任务下载量最多的模型，我们将其命名为 `model_download_tool`。
- 一个 `description`。与 `name` 一样，此描述是为您的 agent 提供动力的 LLM 的说明书，所以不要忽视它。
- 输入类型和描述
- 输出类型
所有这些属性将在初始化时自动嵌入到 agent 的系统提示中：因此要努力使它们尽可能清晰！
</hfoption>
</hfoptions>


然后您可以直接初始化您的 agent：
```py
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[model_download_tool], model=InferenceClientModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

您将获得以下日志：
```text
╭──────────────────────────────────────── New run ─────────────────────────────────────────╮
│                                                                                          │
│ Can you give me the name of the model that has the most downloads in the 'text-to-video' │
│ task on the Hugging Face Hub?                                                            │
│                                                                                          │
╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ───────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 model_name = model_download_tool(task="text-to-video")                               │
│   2 print(model_name)                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Execution logs:
ByteDance/AnimateDiff-Lightning

Out: None
[Step 0: Duration 0.27 seconds| Input tokens: 2,069 | Output tokens: 60]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 final_answer("ByteDance/AnimateDiff-Lightning")                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Out - Final answer: ByteDance/AnimateDiff-Lightning
[Step 1: Duration 0.10 seconds| Input tokens: 4,288 | Output tokens: 148]
Out[20]: 'ByteDance/AnimateDiff-Lightning'
```

> [!TIP]
> 在 [专用教程](./tutorials/tools#what-is-a-tool-and-how-to-build-one) 中了解更多关于工具的内容。

## 多 agent

多 agent 系统是随着微软的框架 [Autogen](https://huggingface.co/papers/2308.08155) 引入的。

在这种类型的框架中，您有多个 agent 一起工作来解决您的任务，而不是只有一个。
经验表明，这在大多数基准测试中表现更好。这种更好表现的原因在概念上很简单：对于许多任务，与其使用一个全能系统，您更愿意将单元专门用于子任务。在这里，拥有具有单独工具集和内存的 agent 可以实现高效的专业化。例如，为什么要用网页搜索 agent 访问的所有网页内容填充代码生成 agent 的内存？最好将它们分开。

您可以使用 `smolagents` 轻松构建分层多 agent 系统。

为此，将 agent 封装在 [`ManagedAgent`] 对象中。此对象需要参数 `agent`、`name` 和 `description`，这些参数将嵌入到管理 agent 的系统提示中，以让它知道如何调用此托管 agent，就像我们对工具所做的那样。

以下是一个使用我们的 [`WebSearchTool`] 制作一个管理特定网页搜索 agent 的 agent 的示例：

```py
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, ManagedAgent

model = InferenceClientModel()

web_agent = CodeAgent(tools=[WebSearchTool()], model=model)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="Runs web searches for you. Give it your query as an argument."
)

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[managed_web_agent]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

> [!TIP]
> 有关高效多 agent 实现的深入示例，请参阅 [我们如何将多 agent 系统推向 GAIA 排行榜的顶部](https://huggingface.co/blog/beating-gaia)。


## 与您的 agent 交谈并在酷炫的 Gradio 界面中可视化其思考过程

您可以使用 `GradioUI` 交互式地向您的 agent 提交任务并观察其思考和执行过程，以下是一个示例：

```py
from smolagents import (
    load_tool,
    CodeAgent,
    InferenceClientModel,
    GradioUI
)

# 从 Hub 导入工具
image_generation_tool = load_tool("m-ric/text-to-image")

model = InferenceClientModel(model_id=model_id)

# 使用图像生成工具初始化 agent
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()
```

在底层，当用户输入新答案时，agent 会以 `agent.run(user_request, reset=False)` 启动。
`reset=False` 标志意味着在启动此新任务之前不会刷新 agent 的内存，这使得对话可以继续。

您也可以在其他 agent 化应用程序中使用此 `reset=False` 参数来保持对话继续。

## 下一步

最后，当您按需配置好agent后，即可将其分享至 Hub！

```py
agent.push_to_hub("m-ric/my_agent")
```

类似地，若要加载已推送至 Hub 的agent，在信任其工具代码的前提下，可使用：

```py
agent.from_hub("m-ric/my_agent", trust_remote_code=True)
```

要更深入地使用，您将需要查看我们的教程：
- [我们的代码 agent 如何工作的解释](./tutorials/secure_code_execution)
- [本指南关于如何构建好的 agent](./tutorials/building_good_agents)。
- [工具使用的深入指南](./tutorials/tools)。
