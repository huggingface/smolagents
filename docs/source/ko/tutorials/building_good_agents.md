# 좋은 에이전트 구축하기[[building-good-agents]]

[[open-in-colab]]

성공하는 에이전트와 실패하는 에이전트 사이에는 큰 차이가 있습니다.
성공하는 에이전트는 어떻게 만들 수 있을까요?
이 가이드에서 에이전트 구축의 핵심 원칙들을 소개하겠습니다.

> [!TIP]
> 에이전트 구축이 처음이라면 먼저 [에이전트 소개](../conceptual_guides/intro_agents)와 [안내서](../guided_tour)를 읽어보세요.

### 최고의 에이전트 시스템은 가장 단순합니다: 워크플로우를 최대한 단순하게 만드세요[[the-best-agentic-systems-are-the-simplest:-simplify-the-workflow-as-much-as-you-can]]

워크플로우에 LLM에게 어느 정도의 자율성을 부여하는 것은 오류가 발생할 위험이 있습니다.

잘 설계된 에이전트 시스템은 오류를 기록하고 다시 시도하는 기능을 통해 LLM이 자신의 실수를 교정할 수 있게 해줍니다. 그렇다고 해도 처음부터 LLM이 실수하지 않도록 워크플로우를 간단하게 만드는 것이 훨씬 효과적입니다.

[에이전트 소개](../conceptual_guides/intro_agents)의 예시를 다시 살펴보겠습니다: 서핑 여행사 이용자들의 문의에 대응하는 봇입니다.
새로운 서핑 스팟에 대해 질문을 받을 때마다 에이전트가 "여행 거리 API"와 "날씨 API"에 각각 2번의 서로 다른 호출을 하도록 하는 대신, 두 API를 한 번에 호출하고 연결된 출력을 사용자에게 반환하는 함수인 "return_spot_information"이라는 하나의 통합된 도구를 만들 수 있습니다.

이렇게 하면 비용, 지연 시간, 오류 위험을 줄일 수 있습니다!

주요 지침은 다음과 같습니다: LLM 호출 횟수를 최대한 줄이세요.

이것은 몇 가지 결론으로 이어집니다:
- 가능하면 언제든지 두 개의 API 예시처럼 2개의 도구를 하나로 그룹화하세요.
- 가능하면 언제든지 로직은 에이전트의 결정보다는 결정론적 함수로 처리해야 합니다.

### LLM 엔진으로의 정보 흐름을 개선하세요[[improve-the-information-flow-to-the-llm-engine]]

LLM은 쪽지를 통해서만 소통할 수 있는 밀폐된 방 안의 *똑똑한* 로봇이라고 생각하면 됩니다.

프롬프트에 명시하지 않으면 무슨 일이 일어났는지 전혀 알 수 없습니다.

그러니까 일단 작업을 아주 명확하게 정의하는 것부터 시작하세요!
에이전트는 LLM으로 작동하기 때문에, 작업을 설명하는 방식이 조금만 달라져도 결과가 완전히 바뀔 수 있습니다.

그 다음엔 도구에서 에이전트로 정보가 잘 전달되도록 개선해야 합니다.

구체적으로는 이렇게 하세요:
- 각 도구는 LLM에게 도움이 될 만한 정보를 모두 기록해야 합니다.(도구의 `forward` 메서드 안에서 `print`문을 쓰기만 하면 됩니다.)
  - 특히 도구 실행 오류에 대한 자세한 정보를 기록하면 큰 도움이 됩니다!

예를 들어 위치와 날짜-시간을 받아서 날씨 데이터를 가져오는 도구를 보겠습니다:

먼저 좋지 않은 버전입니다:
```python
import datetime
from smolagents import tool

def get_weather_report_at_coordinates(coordinates, date_time):
    # 더미 함수, [섭씨 온도, 0-1 척도의 비 올 확률, 미터 단위 파도 높이] 리스트를 반환
    return [28.0, 0.35, 0.85]

def convert_location_to_coordinates(location):
    # 더미 좌표를 반환
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

문제점은 무엇일까요?
- `date_time`에 사용해야 하는 형식에 대한 정확한 설명이 없습니다.
- 위치를 어떻게 지정해야 하는지에 대한 세부 정보가 없습니다.
- 위치가 적절한 형식이 아니거나 `date_time`이 제대로 형식화되지 않은 경우와 같은 실패 사례를 명시적으로 기록할 수 있는 로깅 메커니즘이 없습니다.
- 출력 형식을 이해하기 어렵습니다.

도구 호출이 실패하면 메모리에 로깅된 오류 추적이 LLM이 도구를 역설계하여 오류를 수정하는 데 도움이 될 수 있습니다. 하지만 왜 그렇게 많은 무거운 작업을 맡겨야 할까요?

이 도구를 구축하는 더 나은 방법은 다음과 같습니다:
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

LLM의 부담을 덜어주려면 이런 질문을 해보세요: "만약 내가 아무것도 모르는 상태에서 이 도구를 처음 사용한다면, 실수했을 때 스스로 고치기가 얼마나 쉬울까?"

### 에이전트에 더 많은 매개변수 제공[[give-more-arguments-to-the-agent]]

작업을 설명하는 단순한 문자열 외에 에이전트에 추가 객체를 전달하려면 `additional_args` 매개변수를 사용하여 모든 유형의 객체를 전달할 수 있습니다:

```py
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

agent = CodeAgent(tools=[], model=InferenceClientModel(model_id=model_id), add_base_tools=True)

agent.run(
    "Why does Mike not know many people in New York?",
    additional_args={"mp3_sound_file_url":'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3'}
)
```
예를 들어, `additional_args` 매개변수를 통해 에이전트가 활용할 수 있도록 원하는 이미지나 문자열을 전달할 수 있습니다.

## 에이전트 디버깅 방법[[how-to-debug-your-agent]]

### 1. 더 강력한 LLM 사용[[use-a-stronger-llm]]

에이전트 워크플로우에서 발생하는 오류 중 일부는 실제 오류이고, 다른 일부는 LLM 엔진이 제대로 추론하지 못한 탓입니다.
예를 들어, 자동차 그림을 만들어 달라고 요청한 `CodeAgent`에 대한 다음 추적을 고려해보세요:
```
==================================================================================================== New task ====================================================================================================
Make me a cool car picture
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
image_generator(prompt="A cool, futuristic sports car with LED headlights, aerodynamic design, and vibrant color, high-res, photorealistic")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Step 1:

- Time taken: 16.35 seconds
- Input tokens: 1,383
- Output tokens: 77
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
final_answer("/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Print outputs:

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Final answer:
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
```
사용자는 이미지가 반환되는 대신 경로가 반환되는 것을 보게 됩니다.
시스템의 버그처럼 보일 수 있지만, 실제로는 에이전트 시스템이 오류를 일으킨 것이 아닙니다: 단지 LLM이 이미지 출력을 변수에 저장하지 않는 실수를 했을 뿐입니다.
따라서 이미지를 저장하면서 로깅된 경로를 활용하는 것 외에는 이미지에 다시 접근할 수 없으므로 이미지 대신 경로를 반환합니다.

따라서 에이전트를 디버깅하는 첫 번째 단계는 "더 강력한 LLM을 사용하는 것"입니다. `Qwen2/5-72B-Instruct`와 같은 대안은 그런 실수를 하지 않았을 것입니다.

### 2. 더 많은 정보나 구체적인 지침 제공[[provide-more-information-or-specific-instructions]]

더 자세하게 안내해준다면 성능이 낮은 모델도 충분히 사용할 수 있습니다.

모델의 관점에서 생각해보세요: 내가 모델이 되어서 이 작업을 해결해야 한다면, 지금 주어진 정보(시스템 프롬프트 + 작업 설명 + 도구 설명)만으로도 충분할까요?

더 구체적인 안내가 필요할까요?

- 지침이 항상 에이전트에게 주어져야 하는 경우(일반적으로 시스템 프롬프트가 작동한다고 이해하는 것처럼): 에이전트 초기화 시 `instructions` 매개변수 아래에 문자열로 전달할 수 있습니다.
- 해결할 특정 작업에 관한 것이라면: 이 모든 세부 사항을 작업에 추가하세요. 작업은 수십 페이지처럼 매우 길 수 있습니다.
- 특정 도구 사용 방법에 관한 것이라면: 해당 도구의 `description` 속성에 포함시키세요.

### 3. 프롬프트 템플릿 변경 (일반적으로 권장되지 않음)[[change-the-prompt-templates-(generally-not-advised)]]

위의 방법들로도 부족하다면 에이전트의 프롬프트 템플릿을 직접 수정할 수 있습니다.

작동 원리를 살펴보겠습니다. [CodeAgent]의 기본 프롬프트 템플릿을 예로 들어보겠습니다(제로샷 예제는 생략하고 간단히 정리했습니다).

```python
print(agent.prompt_templates["system_prompt"])
```
Here is what you get:
```text
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of Thought, Code, and Observation sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the Code sequence you should write the code in simple Python. The code sequence must be opened with '{{code_block_opening_tag}}', and closed with '{{code_block_closing_tag}}'.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
{{code_block_opening_tag}}
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
{{code_block_closing_tag}}
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person.
{{code_block_opening_tag}}
image = image_generator("A portrait of John Doe, a 55-year-old man living in Canada.")
final_answer(image)
{{code_block_closing_tag}}

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool
{{code_block_opening_tag}}
result = 5 + 3 + 1294.678
final_answer(result)
{{code_block_closing_tag}}

---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French.
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
{{code_block_opening_tag}}
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {answer}")
{{code_block_closing_tag}}

---
Task:
In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.
What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin.
{{code_block_opening_tag}}
pages = web_search(query="1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein")
print(pages)
{{code_block_closing_tag}}
Observation:
No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query.
{{code_block_opening_tag}}
pages = web_search(query="1979 interview Stanislaus Ulam")
print(pages)
{{code_block_closing_tag}}
Observation:
Found 6 pages:
[Stanislaus Ulam 1979 interview](https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/)

[Ulam discusses Manhattan Project](https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/)

(truncated)

Thought: I will read the first 2 pages to know more.
{{code_block_opening_tag}}
for url in ["https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/", "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/"]:
    whole_page = visit_webpage(url)
    print(whole_page)
    print("\n" + "="*80 + "\n")  # Print separator between pages
{{code_block_closing_tag}}
Observation:
Manhattan Project Locations:
Los Alamos, NM
Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at
(truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word.
{{code_block_opening_tag}}
final_answer("diminished")
{{code_block_closing_tag}}

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the tool `web_search` to get the population of both cities.
{{code_block_opening_tag}}
for city in ["Guangzhou", "Shanghai"]:
    print(f"Population {city}:", web_search(f"{city} population")
{{code_block_closing_tag}}
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Now I know that Shanghai has the highest population.
{{code_block_opening_tag}}
final_answer("Shanghai")
{{code_block_closing_tag}}

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `wikipedia_search` to get the age of the pope, and confirm that with a web search.
{{code_block_opening_tag}}
pope_age_wiki = wikipedia_search(query="current pope age")
print("Pope age as per wikipedia:", pope_age_wiki)
pope_age_search = web_search(query="current pope age")
print("Pope age as per google search:", pope_age_search)
{{code_block_closing_tag}}
Observation:
Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using python code.
{{code_block_opening_tag}}
pope_current_age = 88 ** 0.36
final_answer(pope_current_age)
{{code_block_closing_tag}}

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
{{code_block_opening_tag}}
{%- for tool in tools.values() %}
{{ tool.to_code_prompt() }}
{% endfor %}
{{code_block_closing_tag}}

{%- if managed_agents and managed_agents.values() | list %}
You can also give tasks to team members.
Calling a team member works similarly to calling a tool: provide the task description as the 'task' argument. Since this team member is a real human, be as detailed and verbose as necessary in your task description.
You can also include any relevant variables or context using the 'additional_args' argument.
Here is a list of the team members that you can call:
{{code_block_opening_tag}}
{%- for agent in managed_agents.values() %}
def {{ agent.name }}(task: str, additional_args: dict[str, Any]) -> str:
    """{{ agent.description }}

    Args:
        task: Long detailed description of the task.
        additional_args: Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.
    """
{% endfor %}
{{code_block_closing_tag}}
{%- endif %}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a '{{code_block_opening_tag}}' sequence ending with '{{code_block_closing_tag}}', else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wikipedia_search({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wikipedia_search(query="What is the place where James Bond lives?")'.
4. For tools WITHOUT JSON output schema: Take care to not chain too many sequential tool calls in the same code block, as their output format is unpredictable. For instance, a call to wikipedia_search without a JSON output schema has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. For tools WITH JSON output schema: You can confidently chain multiple tool calls and directly access structured output fields in the same code block! When a tool has a JSON output schema, you know exactly what fields and data types to expect, allowing you to write robust code that directly accesses the structured response (e.g., result['field_name']) without needing intermediate print() statements.
6. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
7. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
8. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
9. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
10. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
11. Don't give up! You're in charge of solving the task, not providing directions to solve it.

{%- if custom_instructions %}
{{custom_instructions}}
{%- endif %}

Now Begin!
```

보시다시피 `"{{ tool.description }}"`와 같은 플레이스홀더들이 있습니다. 이것들은 에이전트를 초기화할 때 도구나 관리 에이전트에 대한 설명을 자동으로 넣어주는 역할을 합니다.

따라서 `system_prompt` 매개변수에 커스텀 프롬프트를 넣어서 기본 시스템 프롬프트 템플릿을 덮어쓸 수 있습니다. 새로운 시스템 프롬프트에는 이런 플레이스홀더들을 포함할 수 있습니다:
- 도구 설명을 삽입하려면:
  ```
  {%- for tool in tools.values() %}
  - {{ tool.to_tool_calling_prompt() }}
  {%- endfor %}
  ```
- 관리되는 에이전트가 있는 경우 해당 설명을 삽입하려면:
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
- `CodeAgent`에만 해당하며, 승인된 import 목록을 삽입하려면: `"{{authorized_imports}}"`

그런 다음 다음과 같이 시스템 프롬프트를 변경할 수 있습니다:

```py
agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + "\nHere you go!"
```

이는 [`ToolCallingAgent`]에서도 작동합니다.

하지만 일반적으로 다음과 같이 에이전트 초기화 시 `instructions` 매개변수를 전달하는 것이 더 간단합니다:
```py
agent = CodeAgent(tools=[], model=InferenceClientModel(model_id=model_id), instructions="Always talk like a 5 year old.")
```

### 4. 추가 계획[[extra-planning]]

일반적인 작업 단계들 중간중간에 에이전트가 추가로 계획을 세우는 단계를 넣을 수 있습니다. 이때는 도구를 사용하지 않고, LLM이 현재까지 파악한 정보를 정리하고 그 정보를 토대로 앞으로의 계획을 다시 점검하게 됩니다.

```py
from smolagents import load_tool, CodeAgent, InferenceClientModel, WebSearchTool
from dotenv import load_dotenv

load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = WebSearchTool()

agent = CodeAgent(
    tools=[search_tool, image_generation_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct"),
    planning_interval=3 # This is where you activate planning!
)

# Run it!
result = agent.run(
    "How long would a cheetah at full speed take to run the length of Pont Alexandre III?",
)
```