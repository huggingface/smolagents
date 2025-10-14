# 도구[[tools]]

[[open-in-colab]]

여기서는 고급 도구 사용법을 살펴보겠습니다.

> [!TIP]
> 에이전트 구축이 처음이라면, 먼저 [에이전트 소개](../conceptual_guides/intro_agents)와 [smolagents 가이드 투어](../guided_tour)를 읽어보세요.


### 도구란 무엇이며 어떻게 만드나요?[[what-is-a-tool-and-how-to-build-one]]

도구는 기본적으로 LLM이 에이전트 시스템에서 사용할 수 있는 함수입니다.

하지만 이를 사용하려면 LLM에 이름, 도구 설명, 입력 유형 및 설명, 출력 유형과 같은 API가 제공되어야 합니다.

따라서 단순한 함수가 아니라 클래스여야 합니다.

핵심적으로 도구는 LLM이 사용법을 이해하는 데 도움이 되는 메타데이터로 함수를 감싸는 클래스입니다.

다음은 그 예시입니다:

```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    이 도구는 Hugging Face Hub에서 주어진 태스크에 대해 가장 많이 다운로드된 모델을 반환합니다.
    체크포인트의 이름을 반환합니다."""
    inputs = {
        "task": {
            "type": "string",
            "description": "태스크 카테고리 (예: text-classification, depth-estimation 등)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

model_downloads_tool = HFModelDownloadsTool()
```

사용자 정의 도구는 [`Tool`]을 서브클래싱하여 유용한 메서드를 상속받습니다. 자식 클래스는 다음을 정의합니다:
- 도구 자체의 이름에 해당하는 `name` 속성입니다. 이름은 보통 도구가 하는 일을 설명합니다. 이 코드는 특정 태스크에서 가장 많이 다운로드된 모델을 반환하므로, `model_download_counter`라고 이름 짓겠습니다.
- 에이전트의 시스템 프롬프트를 채우는 데 사용되는 `description` 속성입니다.
- `"type"`과 `"description"`을 키로 갖는 딕셔너리인 `inputs` 속성입니다. 이 속성은 파이썬 인터프리터가 입력에 대해 정보에 기반한 선택을 하는 데 도움이 되는 정보를 담고 있습니다.
- 출력 유형을 지정하는 `output_type` 속성입니다. `inputs`와 `output_type`의 유형은 모두 [Pydantic 형식](https://docs.pydantic.dev/latest/concepts/json_schema/#generating-json-schema)이어야 하며, [`~AUTHORIZED_TYPES`] 중 하나일 수 있습니다.
- 실행될 추론 코드를 포함하는 `forward` 메서드입니다.

이것이 에이전트에서 사용되기 위해 필요한 전부입니다!

도구를 만드는 또 다른 방법이 있습니다. [가이드 투어](../guided_tour)에서는 `@tool` 데코레이터를 사용하여 도구를 구현했습니다. [`tool`] 데코레이터는 간단한 도구를 정의하는 권장 방법이지만, 때로는 명확성을 위해 클래스에서 여러 메서드를 사용하거나 추가 클래스 속성을 사용하는 등 더 많은 기능이 필요할 수 있습니다.

이 경우, 위에서 설명한 대로 [`Tool`]을 서브클래싱하여 도구를 만들 수 있습니다.

### Hub에 도구 공유하기[[share-your-tool-to-the-hub]]

도구에서 [`~Tool.push_to_hub`]를 호출하여 사용자 정의 도구를 Space 리포지토리로 Hub에 공유할 수 있습니다. Hub에 해당 리포지토리를 생성하고 읽기 접근 권한이 있는 토큰을 사용하고 있는지 확인하세요.

```python
model_downloads_tool.push_to_hub("{your_username}/hf-model-downloads", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

Hub에 푸시가 작동하려면 도구가 몇 가지 규칙을 준수해야 합니다:
- 모든 메서드는 독립적이어야 합니다. 즉, 인자(args)에서 온 변수만 사용해야 합니다.
- 위 사항에 따라, **모든 import는 도구의 함수 내에서 직접 정의되어야 합니다.** 그렇지 않으면 사용자 정의 도구로 [`~Tool.save`] 또는 [`~Tool.push_to_hub`]를 호출할 때 오류가 발생합니다.
- `__init__` 메서드를 서브클래싱하는 경우, `self` 외에 다른 인자를 전달할 수 없습니다. 특정 도구 인스턴스를 초기화할 때 설정된 인자는 추적하기 어려워 Hub에 제대로 공유할 수 없기 때문입니다. 어쨌든, 특정 클래스를 만드는 이유는 하드코딩해야 할 모든 것에 대해 이미 클래스 속성을 설정할 수 있다는 것입니다(`class YourTool(Tool):` 바로 아래에 `your_variable=(...)`을 설정하면 됩니다). 물론 코드 어디에서든 `self.your_variable`에 값을 할당하여 클래스 속성을 생성할 수도 있습니다.


도구가 Hub에 푸시되면 시각화할 수 있습니다. [여기](https://huggingface.co/spaces/m-ric/hf-model-downloads)에 제가 푸시한 `model_downloads_tool`이 있습니다. 멋진 gradio 인터페이스를 갖추고 있습니다.

도구 파일을 살펴보면 모든 도구의 로직이 [tool.py](https://huggingface.co/spaces/m-ric/hf-model-downloads/blob/main/tool.py)에 있음을 알 수 있습니다. 여기서 다른 사람이 공유한 도구를 검사할 수 있습니다.

그런 다음 [`load_tool`]로 도구를 로드하거나 [`~Tool.from_hub`]로 생성하여 에이전트의 `tools` 매개변수에 전달할 수 있습니다.
도구를 실행하는 것은 사용자 정의 코드를 실행하는 것을 의미하므로, 리포지토리를 신뢰할 수 있는지 확인해야 합니다. 따라서 Hub에서 도구를 로드하려면 `trust_remote_code=True`를 전달해야 합니다.

```python
from smolagents import load_tool, CodeAgent

model_download_tool = load_tool(
    "{your_username}/hf-model-downloads",
    trust_remote_code=True
)
```

### MCP 서버의 도구 사용하기[[use-tools-from-an-mcp-server]]

저희 `MCPClient`를 사용하면 MCP 서버에서 도구를 로드하고 연결 및 도구 관리를 완벽하게 제어할 수 있습니다:

stdio 기반 MCP 서버의 경우:
```python
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
import os

server_parameters = StdioServerParameters(
    command="uvx",  # uvx를 사용하면 의존성을 사용할 수 있습니다
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with MCPClient(server_parameters) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("COVID-19 치료에 대한 최신 연구를 찾아주세요.")
```

스트리밍 HTTP 기반 MCP 서버의 경우:
```python
from smolagents import MCPClient, CodeAgent

with MCPClient({"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("숙취 해소법을 찾아주세요.")
```

try...finally 패턴으로 연결 수명 주기를 수동으로 관리할 수도 있습니다:

```python
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
import os

# 서버 매개변수 초기화
server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

# 연결 수동 관리
try:
    mcp_client = MCPClient(server_parameters)
    tools = mcp_client.get_tools()

    # 에이전트와 함께 도구 사용
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    result = agent.run("알츠하이머병에 대한 최근 치료 접근법은 무엇인가요?")

    # 필요에 따라 결과 처리
    print(f"Agent response: {result}")
finally:
    # 항상 연결이 제대로 닫혔는지 확인
    mcp_client.disconnect()
```

서버 매개변수 목록을 전달하여 한 번에 여러 MCP 서버에 연결할 수도 있습니다:
```python
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
import os

server_params1 = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

server_params2 = {"url": "http://127.0.0.1:8000/sse"}

with MCPClient([server_params1, server_params2]) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("최신 연구를 분석하고 두통 치료법을 제안해주세요.")
```

> [!WARNING]
> **보안 경고:** MCP 서버에 연결하기 전, 특히 프로덕션 환경에서는 항상 해당 서버의 소스와 무결성을 확인하세요.
> MCP 서버 사용에는 보안 위험이 따릅니다:
> - **신뢰가 중요합니다:** 신뢰할 수 있는 소스의 MCP 서버만 사용하세요. 악의적인 서버는 사용자의 컴퓨터에서 유해한 코드를 실행할 수 있습니다.
> - **stdio 기반 MCP 서버**는 항상 사용자의 컴퓨터에서 코드를 실행합니다 (이것이 의도된 기능입니다).
> - **스트리밍 HTTP 기반 MCP 서버:** 원격 MCP 서버는 사용자의 컴퓨터에서 코드를 실행하지 않지만, 여전히 주의해서 사용해야 합니다.

#### 구조화된 출력 및 출력 스키마 지원[[structured-output-and-output-schema-support]]

최신 [MCP 사양 (2025-06-18 이상)](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#structured-content)에는 `outputSchema` 지원이 포함되어, 도구가 정의된 스키마를 가진 구조화된 데이터를 반환할 수 있게 합니다. `smolagents`는 이러한 구조화된 출력 기능을 활용하여 에이전트가 복잡한 데이터 구조, JSON 객체 및 기타 구조화된 형식을 반환하는 도구와 함께 작동할 수 있도록 합니다. 이 기능을 통해 에이전트의 LLM은 도구를 호출하기 전에 도구 출력의 구조를 "볼" 수 있어 더 지능적이고 문맥을 인식하는 상호작용이 가능해집니다.

구조화된 출력 지원을 활성화하려면 `MCPClient`를 초기화할 때 `structured_output=True`를 전달하세요:

```python
from smolagents import MCPClient, CodeAgent

# 구조화된 출력 지원 활성화
with MCPClient(server_parameters, structured_output=True) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("파리의 날씨 정보를 가져오세요")
```

`structured_output=True`일 때 다음 기능이 활성화됩니다:
- **출력 스키마 지원**: 도구는 출력에 대한 JSON 스키마를 정의할 수 있습니다
- **구조화된 콘텐츠 처리**: MCP 응답에서 `structuredContent` 지원
- **JSON 파싱**: 도구 응답에서 구조화된 데이터를 자동으로 파싱

다음은 구조화된 출력을 사용하는 날씨 MCP 서버 예제입니다:

```python
# demo/weather.py - 구조화된 출력을 사용하는 MCP 서버 예제
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Service")

class WeatherInfo(BaseModel):
    location: str = Field(description="위치 이름")
    temperature: float = Field(description="섭씨 온도")
    conditions: str = Field(description="날씨 상태")
    humidity: int = Field(description="습도(%)", ge=0, le=100)

@mcp.tool(
    name="get_weather_info",
    description="위치에 대한 날씨 정보를 구조화된 데이터로 가져옵니다.",
    # structured_output=True는 FastMCP에서 기본적으로 활성화됩니다
)
def get_weather_info(city: str) -> WeatherInfo:
    """도시의 날씨 정보를 가져옵니다."""
    return WeatherInfo(
        location=city,
        temperature=22.5,
        conditions="partly cloudy",
        humidity=65
    )
```

출력 스키마와 구조화된 출력을 사용하는 에이전트:

```python
from smolagents import MCPClient, CodeAgent

# 구조화된 출력을 사용하는 날씨 서버 사용
from mcp import StdioServerParameters

server_parameters = StdioServerParameters(
    command="python",
    args=["demo/weather.py"]
)

with MCPClient(server_parameters, structured_output=True) as tools:
    agent = CodeAgent(tools=tools, model=model)
    result = agent.run("도쿄의 온도는 화씨로 몇 도입니까?")
    print(result)
```

구조화된 출력이 활성화되면, `CodeAgent` 시스템 프롬프트가 도구에 대한 JSON 스키마 정보를 포함하도록 향상되어 에이전트가 도구 출력의 예상 구조를 이해하고 데이터에 적절하게 접근하는 데 도움이 됩니다.

**하위 호환성**: `structured_output` 매개변수는 현재 하위 호환성을 유지하기 위해 기본적으로 `False`입니다. 기존 코드는 변경 없이 계속 작동하며 이전처럼 간단한 텍스트 출력을 받게 됩니다.

**향후 변경 사항**: 향후 릴리스에서는 `structured_output`의 기본값이 `False`에서 `True`로 변경될 예정입니다. 향상된 기능을 사용하려면 `structured_output=True`를 명시적으로 설정하는 것이 좋습니다. 이를 통해 더 나은 도구 출력 처리와 향상된 에이전트 성능을 제공받을 수 있습니다. 현재의 텍스트 전용 동작을 특별히 유지해야 하는 경우에만 `structured_output=False`를 사용하세요.

### Space를 도구로 가져오기[[import-a-space-as-a-tool]]

[`Tool.from_space`] 메서드를 사용하여 Hub의 Gradio Space를 도구로 직접 가져올 수 있습니다!

Hub에 있는 Space의 ID, 이름, 그리고 에이전트가 도구의 역할을 이해하는 데 도움이 될 설명을 제공하기만 하면 됩니다. 내부적으로는 [`gradio-client`](https://pypi.org/project/gradio-client/) 라이브러리를 사용하여 Space를 호출합니다.

예를 들어, Hub에서 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) Space를 가져와 이미지를 생성해 보겠습니다.

```python
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="프롬프트로부터 이미지 생성"
)

image_generation_tool("화창한 해변")
```
자, 여기 이미지가 있습니다! 🏖️

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sunny_beach.webp">

그런 다음 이 도구를 다른 도구와 똑같이 사용할 수 있습니다. 예를 들어, `a rabbit wearing a space suit`라는 프롬프트를 개선하고 그 이미지를 생성해 보겠습니다. 이 예제는 에이전트에 추가 인자를 전달하는 방법도 보여줍니다.

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "이 프롬프트를 개선한 다음, 그것으로 이미지를 생성해 주세요.", additional_args={'user_prompt': '우주복을 입은 토끼'}
)
```

```text
=== 에이전트 생각:
개선된 프롬프트는 "밝은 주황색 일몰 아래 달 표면에서, 지구가 배경으로 보이는 곳에 있는 밝은 파란색 우주복을 입은 토끼"가 될 수 있습니다.

이제 프롬프트를 개선했으니, 이미지 생성기 도구를 사용하여 이 프롬프트를 기반으로 이미지를 생성할 수 있습니다.
>>> 에이전트가 아래 코드를 실행 중입니다:
image = image_generator(prompt="밝은 주황색 일몰 아래 달 표면에서, 지구가 배경으로 보이는 곳에 있는 밝은 파란색 우주복을 입은 토끼")
final_answer(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit_spacesuit_flux.webp">

정말 멋지지 않나요? 🤩

### LangChain 도구 사용하기[[use-langchain-tools]]

저희는 Langchain을 좋아하며 매우 매력적인 도구 모음을 갖추고 있다고 생각합니다.
LangChain에서 도구를 가져오려면 `from_langchain()` 메서드를 사용하세요.

다음은 LangChain 웹 검색 도구를 사용하여 소개의 검색 결과를 재현하는 방법입니다.
이 도구가 제대로 작동하려면 `pip install langchain google-search-results -q`가 필요합니다.
```python
from langchain.agents import load_tools

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("Attention is All You Need에서 제안된 아키텍처의 인코더와 비교했을 때 BERT base 인코더에는 몇 개의 블록(레이어라고도 함)이 더 있나요?")
```

### 에이전트의 도구 상자 관리하기[[manage-your-agents-toolbox]]

에이전트의 도구 상자는 표준 딕셔너리이므로 `agent.tools` 속성에서 도구를 추가하거나 교체하여 관리할 수 있습니다.

기본 도구 상자만으로 초기화된 기존 에이전트에 `model_download_tool`을 추가해 보겠습니다.

```python
from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.tools[model_download_tool.name] = model_download_tool
```
이제 새로운 도구를 활용할 수 있습니다:

```python
agent.run(
    "Hugging Face Hub의 'text-to-video' 태스크에서 가장 많이 다운로드된 모델의 이름을 알려주되, 글자를 거꾸로 뒤집어 주세요."
)
```


> [!TIP]
> 에이전트에 너무 많은 도구를 추가하지 않도록 주의하세요. 성능이 낮은 LLM 엔진에 과부하를 줄 수 있습니다.


### 도구 컬렉션 사용하기[[use-a-collection-of-tools]]

[`ToolCollection`]을 사용하여 도구 컬렉션을 활용할 수 있습니다. Hub의 컬렉션이나 MCP 서버 도구를 로드하는 것을 지원합니다.


#### 모든 MCP 서버의 도구 컬렉션[[tool-collection-from-any-mcp-server]]

[glama.ai](https://glama.ai/mcp/servers) 또는 [smithery.ai](https://smithery.ai/)에서 사용 가능한 수백 개의 MCP 서버 도구를 활용하세요.

MCP 서버 도구는 [`ToolCollection.from_mcp`]로 로드할 수 있습니다.

> [!WARNING]
> **보안 경고:** MCP 서버에 연결하기 전, 특히 프로덕션 환경에서는 항상 해당 서버의 소스와 무결성을 확인하세요.
> MCP 서버 사용에는 보안 위험이 따릅니다:
> - **신뢰가 중요합니다:** 신뢰할 수 있는 소스의 MCP 서버만 사용하세요. 악의적인 서버는 사용자의 컴퓨터에서 유해한 코드를 실행할 수 있습니다.
> - **stdio 기반 MCP 서버**는 항상 사용자의 컴퓨터에서 코드를 실행합니다 (이것이 의도된 기능입니다).
> - **스트리밍 HTTP 기반 MCP 서버:** 원격 MCP 서버는 사용자의 컴퓨터에서 코드를 실행하지 않지만, 여전히 주의해서 사용해야 합니다.

stdio 기반 MCP 서버의 경우, 서버 매개변수를 `mcp.StdioServerParameters`의 인스턴스로 전달하세요:
```py
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("숙취 해소법을 찾아주세요.")
```

ToolCollection에서 구조화된 출력 지원을 활성화하려면 `structured_output=True` 매개변수를 추가하세요:
```py
with ToolCollection.from_mcp(server_parameters, trust_remote_code=True, structured_output=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("숙취 해소법을 찾아주세요.")
```

스트리밍 HTTP 기반 MCP 서버의 경우, `mcp.client.streamable_http.streamablehttp_client`에 매개변수가 있는 딕셔너리를 전달하고 `transport` 키에 `"streamable-http"` 값을 추가하기만 하면 됩니다:
```py
from smolagents import ToolCollection, CodeAgent

with ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True)
    agent.run("숙취 해소법을 찾아주세요.")
```

#### Hub의 컬렉션에서 도구 컬렉션 가져오기[[tool-collection-from-a-collection-in-the-hub]]

사용하려는 컬렉션의 슬러그(slug)를 사용하여 활용할 수 있습니다.
그런 다음 리스트로 전달하여 에이전트를 초기화하고 사용을 시작하세요!

```py
from smolagents import ToolCollection, CodeAgent

image_tool_collection = ToolCollection.from_hub(
    collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f",
    token="<YOUR_HUGGINGFACEHUB_API_TOKEN>"
)
agent = CodeAgent(tools=[*image_tool_collection.tools], model=model, add_base_tools=True)

agent.run("강과 호수가 있는 그림을 그려주세요.")
```

시작 속도를 높이기 위해, 도구는 에이전트에 의해 호출될 때만 로드됩니다.