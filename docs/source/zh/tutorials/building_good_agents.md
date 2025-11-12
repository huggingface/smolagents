# 构建好用的 agent

[[open-in-colab]]

能良好工作的 agent 和不能工作的 agent 之间，有天壤之别。
我们怎么样才能构建出属于前者的 agent 呢？
在本指南中，我们将看到构建 agent 的最佳实践。

> [!TIP]
> 如果你是 agent 构建的新手，请确保首先阅读 [agent 介绍](../conceptual_guides/intro_agents) 和 [smolagents 导览](../guided_tour)。

### 最好的 agent 系统是最简单的：尽可能简化工作流

在你的工作流中赋予 LLM 一些自主权，会引入一些错误风险。

经过良好编程的 agent 系统，通常具有良好的错误日志记录和重试机制，因此 LLM 引擎有机会自我纠错。但为了最大限度地降低 LLM 错误的风险，你应该简化你的工作流！

让我们回顾一下 [agent 介绍](../conceptual_guides/intro_agents) 中的例子：一个为冲浪旅行公司回答用户咨询的机器人。
与其让 agent 每次被问及新的冲浪地点时，都分别调用 "旅行距离 API" 和 "天气 API"，你可以只创建一个统一的工具 "return_spot_information"，一个同时调用这两个 API，并返回它们连接输出的函数。

这可以降低成本、延迟和错误风险！

主要的指导原则是：尽可能减少 LLM 调用的次数。

这可以带来一些启发：
- 尽可能把两个工具合并为一个，就像我们两个 API 的例子。
- 尽可能基于确定性函数，而不是 agent 决策，来实现逻辑。

### 改善流向 LLM 引擎的信息流

记住，你的 LLM 引擎就像一个 ~智能~ 机器人，被关在一个房间里，与外界唯一的交流方式是通过门缝传递的纸条。

如果你没有明确地将信息放入其提示中，它将不知道发生的任何事情。

所以首先要让你的任务非常清晰！
由于 agent 由 LLM 驱动，任务表述的微小变化可能会产生完全不同的结果。

然后，改善工具使用中流向 agent 的信息流。

需要遵循的具体指南：
- 每个工具都应该记录（只需在工具的 `forward` 方法中使用 `print` 语句）对 LLM 引擎可能有用的所有信息。
  - 特别是，记录工具执行错误的详细信息会很有帮助！

例如，这里有一个根据位置和日期时间检索天气数据的工具：

首先，这是一个糟糕的版本：
```python
import datetime
from smolagents import tool

def get_weather_report_at_coordinates(coordinates, date_time):
    # 虚拟函数，返回 [温度（°C），降雨风险（0-1），浪高（m）]
    return [28.0, 0.35, 0.85]

def get_coordinates_from_location(location):
    # 返回虚拟坐标
    return [3.3, -42.0]

@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for.
        date_time: the date and time for which you want the report.
    """
    lon, lat = convert_location_to_coordinates(location)
    date_time = datetime.strptime(date_time)
    return str(get_weather_report_at_coordinates((lon, lat), date_time))
```

为什么它不好？
- 没有说明 `date_time` 应该使用的格式
- 没有说明位置应该如何指定
- 没有记录机制来处理明确的报错情况，如位置格式不正确或 date_time 格式不正确
- 输出格式难以理解

如果工具调用失败，内存中记录的错误跟踪，可以帮助 LLM 逆向工程工具来修复错误。但为什么要让它做这么多繁重的工作呢？

构建这个工具的更好方式如下：
```python
@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for. Should be a place name, followed by possibly a city name, then a country, like "Anchor Point, Taghazout, Morocco".
        date_time: the date and time for which you want the report, formatted as '%m/%d/%y %H:%M:%S'.
    """
    lon, lat = convert_location_to_coordinates(location)
    try:
        date_time = datetime.strptime(date_time)
    except Exception as e:
        raise ValueError("Conversion of `date_time` to datetime format failed, make sure to provide a string in format '%m/%d/%y %H:%M:%S'. Full trace:" + str(e))
    temperature_celsius, risk_of_rain, wave_height = get_weather_report_at_coordinates((lon, lat), date_time)
    return f"Weather report for {location}, {date_time}: Temperature will be {temperature_celsius}°C, risk of rain is {risk_of_rain*100:.0f}%, wave height is {wave_height}m."
```

一般来说，为了减轻 LLM 的负担，要问自己的好问题是："如果我是一个第一次使用这个工具的傻瓜，使用这个工具编程并纠正自己的错误有多容易？"。

### 给 agent 更多参数

除了简单的任务描述字符串外，你还可以使用 `additional_args` 参数传递任何类型的对象：

```py
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

agent = CodeAgent(tools=[], model=InferenceClientModel(model_id=model_id), add_base_tools=True)

agent.run(
    "Why does Mike not know many people in New York?",
    additional_args={"mp3_sound_file_url":'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3'}
)
```
例如，你可以使用这个 `additional_args` 参数传递你希望 agent 利用的图像或字符串。


## 如何调试你的 agent

### 1. 使用更强大的 LLM

在 agent 工作流中，有些错误是实际错误，有些则是你的 LLM 引擎没有正确推理的结果。
例如，参考这个我要求创建一个汽车图片的 `CodeAgent` 的运行记录：
```text
==================================================================================================== New task ====================================================================================================
Make me a cool car picture
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ─────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
image_generator(prompt="A cool, futuristic sports car with LED headlights, aerodynamic design, and vibrant color, high-res, photorealistic")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Step 1:

- Time taken: 16.35 seconds
- Input tokens: 1,383
- Output tokens: 77
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ─────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
final_answer("/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Print outputs:

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Final answer:
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
```
用户看到的是返回了一个路径，而不是图像。
这看起来像是系统的错误，但实际上 agent 系统并没有导致错误：只是 LLM 大脑犯了一个错误，没有把图像输出，保存到变量中。
因此，它无法再次访问图像，只能利用保存图像时记录的路径，所以它返回的是路径，而不是图像。

调试 agent 的第一步是"使用更强大的 LLM"。像 `Qwen2.5-72B-Instruct` 这样的替代方案不会犯这种错误。

### 2. 提供更多指导/更多信息

你也可以使用不太强大的模型，只要你更有效地指导它们。

站在模型的角度思考：如果你是模型在解决任务，你会因为系统提示+任务表述+工具描述中提供的信息而挣扎吗？

你需要一些额外的说明吗？

为了提供额外信息，我们不建议立即更改系统提示：默认系统提示有许多调整，除非你非常了解提示，否则你很容易翻车。
更好的指导 LLM 引擎的方法是：
- 如果是关于要解决的任务：把所有细节添加到任务中。任务可以有几百页长。
- 如果是关于如何使用工具：你的工具的 description 属性。


### 3. 更改系统提示（通常不建议）

如果上述说明不够，你可以更改系统提示。

让我们看看它是如何工作的。例如，让我们检查 [`CodeAgent`] 的默认系统提示（下面的版本通过跳过零样本示例进行了缩短）。

```python
print(agent.prompt_templates["system_prompt"])
```
你会得到：
```text
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```<end_code>
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person.
Code:
```py
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
```<end_code>

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
Code:
```py
result = 5 + 3 + 1294.678
final_answer(result)
```<end_code>

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {answer}")
```<end_code>

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
```<end_code>
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
Code:
```py
pages = search(query="1979 interview Stanislaus Ulam")
print(pages)
```<end_code>
Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I will read the first 2 pages to know more.
Code:
```py
for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
    whole_page = visit_webpage(url)
    print(whole_page)
    print("\n" + "="*80 + "\n")  # Print separator between pages
```<end_code>
Observation:
Manhattan Project Locations:
Los Alamos, NM
Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
Code:
```py
final_answer("diminished")
```<end_code>

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
Code:
```py
for city in ["Guangzhou", "Shanghai"]:
    print(f"Population {city}:", search(f"{city} population")
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `wiki` to get the age of the pope, and confirm that with a web search.
Code:
```py
pope_age_wiki = wiki(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
```<end_code>
Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 88 ** 0.36
final_answer(pope_current_age)
```<end_code>

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:
{%- for tool in tools.values() %}
- {{ tool.to_tool_calling_prompt() }}
{%- endfor %}

{%- if managed_agents and managed_agents.values() | list %}
You can also give tasks to team members.
Calling a team member works similarly to calling a tool: provide the task description as the 'task' argument. Since this team member is a real human, be as detailed and verbose as necessary in your task description.
You can also include any relevant variables or context using the 'additional_args' argument.
Here is a list of the team members that you can call:
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
{%- endif %}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
```

如你所见，有一些占位符，如 `"{{ tool.description }}"`：这些将在 agent 初始化时用于插入某些自动生成的工具或管理 agent 的描述。

因此，虽然你可以通过将自定义提示作为参数传递给 `system_prompt` 参数来覆盖此系统提示模板，但你的新系统提示必须包含以下占位符：
- 用于插入工具描述。
  ```
  {%- for tool in tools.values() %}
  - {{ tool.to_tool_calling_prompt() }}
  {%- endfor %}
  ```
- 用于插入 managed agent 的描述（如果有）。
  ```
  {%- if managed_agents and managed_agents.values() | list %}
  You can also give tasks to team members.
  Calling a team member works similarly to calling a tool: provide the task description as the 'task' argument. Since this team member is a real human, be as detailed and verbose as necessary in your task description.
  You can also include any relevant variables or context using the 'additional_args' argument.
  Here is a list of the team members that you can call:
  {%- for agent in managed_agents.values() %}
  - {{ agent.name }}: {{ agent.description }}
  {%- endfor %}
  {%- endif %}
  ```
- 仅限 `CodeAgent`：`"{{authorized_imports}}"` 用于插入授权导入列表。

然后你可以根据如下，更改系统提示：

```py
agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + "\nHere you go!"
```

这也适用于 [`ToolCallingAgent`]。


### 4. 额外规划

我们提供了一个用于补充规划步骤的模型，agent 可以在正常操作步骤之间定期运行。在此步骤中，没有工具调用，LLM 只是被要求更新它知道的事实列表，并根据这些事实反推它应该采取的下一步。

```py
from smolagents import load_tool, CodeAgent, InferenceClientModel, WebSearchTool
from dotenv import load_dotenv

load_dotenv()

# 从 Hub 导入工具
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = WebSearchTool()

agent = CodeAgent(
    tools=[search_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct"),
    planning_interval=3 # 这是你激活规划的地方！
)

# 运行它！
result = agent.run(
    "How long would a cheetah at full speed take to run the length of Pont Alexandre III?",
)
```