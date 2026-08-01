# 编排 multi-agent 系统 🤖🤝🤖

[[open-in-colab]]

此notebook将构建一个 **multi-agent 网络浏览器：一个有多个代理协作，使用网络进行搜索解决问题的代理系统**

`ManagedAgent` 对象将封装这些管理网络搜索的agent，形成一个简单的层次结构：

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
  Code interpreter   +--------------------------------+
       tool          |         Managed agent          |
                     |      +------------------+      |
                     |      | Web Search agent |      |
                     |      +------------------+      |
                     |         |            |         |
                     |  Web Search tool     |         |
                     |             Visit webpage tool |
                     +--------------------------------+
```
我们来一起构建这个系统。运行下列代码以安装依赖包：

```
!pip install 'smolagents[toolkit]' --upgrade -q
```

我们需要登录Hugging Face Hub以调用HF的Inference API：

```
from huggingface_hub import login

login()
```

⚡️ HF的Inference API 可以快速轻松地运行任何开源模型，因此我们的agent将使用HF的Inference API
中的`InferenceClientModel`类来调用
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)模型。

_Note:_ 基于多参数和部署模型的 Inference API 可能在没有预先通知的情况下更新或替换模型。了解更多信息，请参阅[这里](https://huggingface.co/docs/api-inference/supported-models)。

```py
model_id = "Qwen/Qwen3-4B-Instruct-2507"
```

## 🔍 创建网络搜索工具

虽然我们可以使用已经存在的
[`WebSearchTool`]
工具作为谷歌搜索的平替进行网页浏览，然后我们也需要能够查看`WebSearchTool`找到的页面。为此，我
们可以直接导入库的内置
`VisitWebpageTool`。但是我们将重新构建它以了解其工作原理。

我们将使用`markdownify` 来从头构建我们的`VisitWebpageTool`工具。

```py
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

现在我们初始化这个工具并测试它！

```py
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

## 构建我们的 multi-agent 系统 🤖🤝🤖

现在我们有了所有工具`search`和`visit_webpage`，我们可以使用它们来创建web agent。

我们该选取什么样的配置来构建这个agent呢？
- 网页浏览是一个单线程任务，不需要并行工具调用，因此JSON工具调用对于这个任务非常有效。因此我们选择`ToolCallingAgent`。
- 有时候网页搜索需要探索许多页面才能找到正确答案，所以我们更喜欢将 `max_steps` 增加到10。

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    ManagedAgent,
    WebSearchTool,
)

model = InferenceClientModel(model_id=model_id)

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)
```

请注意，我们为这个代理赋予了 name（名称）和 description（描述）属性，这些是必需属性，以便让管理代理能够调用此代理。

然后，我们创建一个管理代理，在初始化时，将受管代理作为 managed_agents 参数传递给它。

由于这个代理的任务是进行规划和思考，高级推理能力会很有帮助，因此 CodeAgent（代码代理）将是最佳选择。

此外，我们要提出一个涉及当前年份并需要进行额外数据计算的问题：所以让我们添加 additional_authorized_imports=["time", "numpy", "pandas"]，以防代理需要用到这些包。

```py
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

可以了！现在让我们运行我们的系统！我们选择一个需要一些计算和研究的问题：

```py
answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
```

我们用这个report 来回答这个问题：
```
Based on current growth projections and energy consumption estimates, if LLM trainings continue to scale up at the
current rhythm until 2030:

1. The electric power required to power the biggest training runs by 2030 would be approximately 303.74 GW, which
translates to about 2,660,762 GWh/year.

1. Comparing this to countries' electricity consumption:
   - It would be equivalent to about 34% of China's total electricity consumption.
   - It would exceed the total electricity consumption of India (184%), Russia (267%), and Japan (291%).
   - It would be nearly 9 times the electricity consumption of countries like Italy or Mexico.

2. Source of numbers:
   - The initial estimate of 5 GW for future LLM training comes from AWS CEO Matt Garman.
   - The growth projection used a CAGR of 79.80% from market research by Springs.
   - Country electricity consumption data is from the U.S. Energy Information Administration, primarily for the year
2021.
```

如果[scaling hypothesis](https://gwern.net/scaling-hypothesis)持续成立的话，我们需要一些庞大的动力配置。我们的agent成功地协作解决了这个任务！✅

💡 你可以轻松地将这个编排扩展到更多的agent：一个执行代码，一个进行网页搜索，一个处理文件加载⋯⋯
