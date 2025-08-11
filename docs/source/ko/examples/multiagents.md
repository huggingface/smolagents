# 멀티 에이전트 시스템 오케스트레이션 🤖🤝🤖

[[Colab에서 열기]]

이 노트북에서는 **멀티 에이전트 웹 브라우저**를 만들어보겠습니다. 이는 웹을 사용하여 문제를 해결하기 위해 여러 에이전트가 협력하는 에이전트 시스템입니다!

멀티 에이전트는 간단한 계층 구조로 구성됩니다.

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
이 시스템을 설정해보겠습니다. 

다음 명령어를 실행하여 필요한 종속성을 설치합니다.

```py
!pip install smolagents[toolkit] --upgrade -q
```

Inference Providers를 사용하기 위해 Hugging Face에 로그인합니다:

```py
from huggingface_hub import login

login()
```

⚡️ 저희 에이전트는 HF의 Inference API를 사용하는 `InferenceClientModel` 클래스를 통해 [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)로 구동됩니다. Inference API를 사용하면 모든 OS 모델을 빠르고 쉽게 실행할 수 있습니다.

> [!TIP]
> Inference Providers는 서버리스 추론 파트너가 지원하는 수백 개의 모델에 대한 액세스를 제공합니다. 지원되는 프로바이더 목록은 [여기](https://huggingface.co/docs/inference-providers/index)에서 확인할 수 있습니다.

```py
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
```

## 🔍 웹 검색 도구 생성

웹 브라우징을 위해 Google 검색과 동등한 기능을 제공하는 기본 [`WebSearchTool`] 도구를 이미 사용할 수 있습니다.

하지만 `WebSearchTool`에서 찾은 페이지를 확인할 수 있는 기능도 필요합니다.
이를 위해 라이브러리에 내장된 `VisitWebpageTool`을 사용할 수도 있지만, 작동 원리를 이해하기 위해 직접 구현해보겠습니다.

그래서 `markdownify`를 사용하여 `VisitWebpageTool` 도구를 처음부터 만들어보겠습니다.

```py
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """주어진 URL의 웹페이지에 접속하여 그 내용을 마크다운 형식의 반환합니다.

    매개변수:
        url: 방문할 웹페이지의 URL.

    반환값:
        마크다운으로 변환된 웹페이지 내용, 또는 요청이 실패할 경우 오류 메시지.
    """
    try:
        # URL에 GET 요청 전송
        response = requests.get(url)
        response.raise_for_status()  # 잘못된 상태 코드에 대해 예외 발생

        # HTML 내용을 마크다운으로 변환
        markdown_content = markdownify(response.text).strip()

        # 여러 줄 바꿈 제거
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

이제 도구를 초기화하고 테스트해보겠습니다!

```py
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

## 멀티 에이전트 시스템 구축 🤖🤝🤖

이제 `search`와 `visit_webpage` 도구가 모두 준비되었으므로, 이를 사용하여 웹 에이전트를 생성할 수 있습니다.

이 에이전트에 어떤 구성을 선택할까요?
- 웹 브라우징은 병렬 도구 호출이 필요없는 단일 타임라인 작업이므로, JSON 도구 호출 방식이 적합합니다. 따라서 `ToolCallingAgent`를 선택합니다.
- 또한 웹 검색은 올바른 답을 찾기 전에 많은 페이지를 탐색해야 하는 경우가 있으므로, `max_steps`를 10으로 늘리는 것이 좋습니다.

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

이 에이전트에 `name`과 `description` 속성을 부여했습니다. 이는 이 에이전트가 매니저 에이전트에 의해 호출될 수 있도록 하는 필수 속성입니다.

그 다음 매니저 에이전트를 생성하고, 초기화 시 `managed_agents` 인수에 관리되는 에이전트를 전달합니다.

이 에이전트는 계획과 사고를 담당하므로, 고급 추론이 유용할 것입니다. 따라서 `CodeAgent`가 잘 작동할 것입니다.

또한 현재 연도를 포함하고 추가 데이터 계산을 수행하는 질문을 하고 싶으므로, 에이전트가 이러한 패키지를 필요로 할 경우에 대비해 `additional_authorized_imports=["time", "numpy", "pandas"]`를 추가해보겠습니다.

```py
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

이게 전부입니다! 이제 시스템을 실행해보겠습니다! 계산과 연구가 모두 필요한 질문을 선택합니다.

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

[스케일링 가설](https://gwern.net/scaling-hypothesis)이 계속 참이라면 상당히 큰 발전소가 필요할 것 같습니다.

에이전트들이 작업을 해결하기 위해 효율적으로 협력했습니다! ✅

💡 이 오케스트레이션을 더 많은 에이전트로 쉽게 확장할 수 있습니다: 하나는 코드 실행을, 다른 하나는 웹 검색을,  또 다른 하나는 파일 처리를 담당하는 식으로...
