# 工具

[[open-in-colab]]

在这里，我们将学习高级工具的使用。

> [!TIP]
> 如果你是构建 agent 的新手，请确保先阅读 [agent 介绍](../conceptual_guides/intro_agents) 和 [smolagents 导览](../guided_tour)。

- [工具](#工具)
    - [什么是工具，如何构建一个工具？](#什么是工具如何构建一个工具)
    - [将你的工具分享到 Hub](#将你的工具分享到-hub)
    - [将 Space 导入为工具](#将-space-导入为工具)
    - [使用 LangChain 工具](#使用-langchain-工具)
    - [管理你的 agent 工具箱](#管理你的-agent-工具箱)
    - [使用工具集合](#使用工具集合)

### 什么是工具，如何构建一个工具？

工具主要是 LLM 可以在 agent 系统中使用的函数。

但要使用它，LLM 需要被提供一个 API：名称、工具描述、输入类型和描述、输出类型。

所以它不能仅仅是一个函数。它应该是一个类。

因此，核心上，工具是一个类，它包装了一个函数，并带有帮助 LLM 理解如何使用它的元数据。

以下是它的结构：

```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

model_downloads_tool = HFModelDownloadsTool()
```

自定义工具继承 [`Tool`] 以继承有用的方法。子类还定义了：
- 一个属性 `name`，对应于工具本身的名称。名称通常描述工具的功能。由于代码返回任务中下载量最多的模型，我们将其命名为 `model_download_counter`。
- 一个属性 `description`，用于填充 agent 的系统提示。
- 一个 `inputs` 属性，它是一个带有键 `"type"` 和 `"description"` 的字典。它包含帮助 Python 解释器对输入做出明智选择的信息。
- 一个 `output_type` 属性，指定输出类型。`inputs` 和 `output_type` 的类型应为 [Pydantic 格式](https://docs.pydantic.dev/latest/concepts/json_schema/#generating-json-schema)，它们可以是以下之一：[`~AUTHORIZED_TYPES`]。
- 一个 `forward` 方法，包含要执行的推理代码。

这就是它在 agent 中使用所需的全部内容！

还有另一种构建工具的方法。在 [guided_tour](../guided_tour) 中，我们使用 `@tool` 装饰器实现了一个工具。[`tool`] 装饰器是定义简单工具的推荐方式，但有时你需要更多：在类中使用多个方法以获得更清晰的代码，或使用额外的类属性。

在这种情况下，你可以通过如上所述继承 [`Tool`] 来构建你的工具。

### 将你的工具分享到 Hub

你可以通过调用 [`~Tool.push_to_hub`] 将你的自定义工具分享到 Hub。确保你已经在 Hub 上为其创建了一个仓库，并且使用的是具有读取权限的 token。

```python
model_downloads_tool.push_to_hub("{your_username}/hf-model-downloads", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

为了使推送到 Hub 正常工作，你的工具需要遵守一些规则：
- 所有方法都是自包含的，例如使用来自其参数中的变量。
- 根据上述要点，**所有导入应直接在工具的函数中定义**，否则在尝试使用 [`~Tool.save`] 或 [`~Tool.push_to_hub`] 调用你的自定义工具时会出现错误。
- 如果你继承了 `__init__` 方法，除了 `self` 之外，你不能给它任何其他参数。这是因为在特定工具实例初始化期间设置的参数很难跟踪，这阻碍了将它们正确分享到 Hub。无论如何，创建特定类的想法是你已经可以为任何需要硬编码的内容设置类属性（只需在 `class YourTool(Tool):` 行下直接设置 `your_variable=(...)`）。当然，你仍然可以通过将内容分配给 `self.your_variable` 在代码中的任何地方创建类属性。

一旦你的工具被推送到 Hub，你就可以查看它。[这里](https://huggingface.co/spaces/m-ric/hf-model-downloads) 是我推送的 `model_downloads_tool`。它有一个漂亮的 gradio 界面。

在深入工具文件时，你可以发现所有工具的逻辑都在 [tool.py](https://huggingface.co/spaces/m-ric/hf-model-downloads/blob/main/tool.py) 下。这是你可以检查其他人分享的工具的地方。

然后你可以使用 [`load_tool`] 加载工具或使用 [`~Tool.from_hub`] 创建它，并将其传递给 agent 中的 `tools` 参数。
由于运行工具意味着运行自定义代码，你需要确保你信任该仓库，因此我们需要传递 `trust_remote_code=True` 来从 Hub 加载工具。

```python
from smolagents import load_tool, CodeAgent

model_download_tool = load_tool(
    "{your_username}/hf-model-downloads",
    trust_remote_code=True
)
```

### 将 Space 导入为工具

你可以使用 [`Tool.from_space`] 方法直接从 Hub 导入一个 Space 作为工具！

你只需要提供 Hub 上 Space 的 id、它的名称和一个帮助你的 agent 理解工具功能的描述。在底层，这将使用 [`gradio-client`](https://pypi.org/project/gradio-client/) 库来调用 Space。

例如，让我们从 Hub 导入 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) Space 并使用它生成一张图片。

```python
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

image_generation_tool("A sunny beach")
```
瞧，这是你的图片！🏖️

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sunny_beach.webp">

然后你可以像使用任何其他工具一样使用这个工具。例如，让我们改进提示 `A rabbit wearing a space suit` 并生成它的图片。

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.", additional_args={'user_prompt': 'A rabbit wearing a space suit'}
)
```

```text
=== Agent thoughts:
improved_prompt could be "A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background"

Now that I have improved the prompt, I can use the image generator tool to generate an image based on this prompt.
>>> Agent is executing the code below:
image = image_generator(prompt="A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background")
final_answer(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit_spacesuit_flux.webp">

这得有多酷？🤩

### 使用 LangChain 工具

我们喜欢 Langchain，并认为它有一套非常吸引人的工具。
要从 LangChain 导入工具，请使用 `from_langchain()` 方法。

以下是如何使用它来重现介绍中的搜索结果，使用 LangChain 的 web 搜索工具。
这个工具需要 `pip install langchain google-search-results -q` 才能正常工作。
```python
from langchain.agents import load_tools

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("How many more blocks (also denoted as layers) are in BERT base encoder compared to the encoder from the architecture proposed in Attention is All You Need?")
```

### 管理你的 agent 工具箱

你可以通过添加或替换工具来管理 agent 的工具箱。

让我们将 `model_download_tool` 添加到一个仅使用默认工具箱初始化的现有 agent 中。

```python
from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.tools[model_download_tool.name] = model_download_tool
```
现在我们可以利用新工具：

```python
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub but reverse the letters?"
)
```


> [!TIP]
> 注意不要向 agent 添加太多工具：这可能会让较弱的 LLM 引擎不堪重负。


### 使用工具集合

你可以通过使用 ToolCollection 对象来利用工具集合，使用你想要使用的集合的 slug。
然后将它们作为列表传递给 agent 初始化，并开始使用它们！

```py
from smolagents import ToolCollection, CodeAgent

image_tool_collection = ToolCollection.from_hub(
    collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f",
    token="<YOUR_HUGGINGFACEHUB_API_TOKEN>"
)
agent = CodeAgent(tools=[*image_tool_collection.tools], model=model, add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes.")
```

为了加快启动速度，工具仅在 agent 调用时加载。
