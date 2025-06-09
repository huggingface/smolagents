# Orchestrate a multi-agent system 🤖🤝🤖

[[open-in-colab]]

In this notebook we will make a **multi-agent web browser: an agentic system with several agents collaborating to solve problems using the web!**

It will be a simple hierarchy:

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
Code Interpreter            +------------------+
    tool                    | Web Search agent |
                            +------------------+
                               |            |
                        Web Search tool     |
                                   Visit webpage tool
```
Let's set up this system. 

Run the line below to install the required dependencies:

```py
!pip install smolagents[toolkit] --upgrade -q
```

Let's login to HF in order to call Inference Providers:

```py
from huggingface_hub import login

login()
```

⚡️ Our agent will be powered by [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) using `InferenceClientModel` class that uses HF's Inference API: the Inference API allows to quickly and easily run any OS model.

> [!TIP]
> Inference Providers give access to hundreds of models, powered by serverless inference partners. A list of supported providers can be found [here](https://huggingface.co/docs/inference-providers/index).

```py
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
```

## 🔍 Create a web search tool

For web browsing, we can already use our native [`WebSearchTool`] tool to provide a Google search equivalent.

But then we will also need to be able to peak into the page found by the `WebSearchTool`.
To do so, we could import the library's built-in `VisitWebpageTool`, but we will build it again to see how it's done.

So let's create our `VisitWebpageTool` tool from scratch using `markdownify`.

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

Ok, now let's initialize and test our tool!

```py
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

## Build our multi-agent system 🤖🤝🤖

Now that we have all the tools `search` and `visit_webpage`, we can use them to create the web agent.

Which configuration to choose for this agent?
- Web browsing is a single-timeline task that does not require parallel tool calls, so JSON tool calling works well for that. We thus choose a `ToolCallingAgent`.
- Also, since sometimes web search requires exploring many pages before finding the correct answer, we prefer to increase the number of `max_steps` to 10.

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    LiteLLMModel,
)

model = InferenceClientModel(model_id=model_id)

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Runs web searches for you.",
)
```

Note that we gave this agent attributes `name` and `description`, mandatory attributes to make this agent callable by its manager agent.

Then we create a manager agent, and upon initialization we pass our managed agent to it in its `managed_agents` argument.

Since this agent is the one tasked with the planning and thinking, advanced reasoning will be beneficial, so a `CodeAgent` will work well.

Also, we want to ask a question that involves the current year and does additional data calculations: so let us add `additional_authorized_imports=["time", "numpy", "pandas"]`, just in case the agent needs these packages.

```py
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

That's all! Now let's run our system! We select a question that requires both some calculation and research:

```py
answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
```

We get this report as the answer:
```
Based on current growth projections and energy consumption estimates, if LLM trainings continue to scale up at the 
current rhythm until 2030:

1. The electric power required to power the biggest training runs by 2030 would be approximately 303.74 GW, which 
translates to about 2,660,762 GWh/year.

2. Comparing this to countries' electricity consumption:
   - It would be equivalent to about 34% of China's total electricity consumption.
   - It would exceed the total electricity consumption of India (184%), Russia (267%), and Japan (291%).
   - It would be nearly 9 times the electricity consumption of countries like Italy or Mexico.

3. Source of numbers:
   - The initial estimate of 5 GW for future LLM training comes from AWS CEO Matt Garman.
   - The growth projection used a CAGR of 79.80% from market research by Springs.
   - Country electricity consumption data is from the U.S. Energy Information Administration, primarily for the year 
2021.
```

Seems like we'll need some sizeable powerplants if the [scaling hypothesis](https://gwern.net/scaling-hypothesis) continues to hold true.

Our agents managed to efficiently collaborate towards solving the task! ✅

💡 You can easily extend this orchestration to more agents: one does the code execution, one the web search, one handles file loadings...
