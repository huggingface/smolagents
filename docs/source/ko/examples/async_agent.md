# 에이전트를 활용한 비동기 애플리케이션[[async-applications-with-agents]]

이 가이드는 smolagents 라이브러리의 동기 에이전트를 Starlette 기반의 비동기 Python 웹 애플리케이션에 통합하는 방법을 설명합니다.
비동기 Python과 에이전트 통합을 처음 접하는 사용자들이 동기 에이전트 로직과 비동기 웹 서버를 효과적으로 결합하는 모범 사례를 익힐 수 있도록 구성했습니다.

## 개요[[overview]]

- **Starlette**: Python에서 비동기 웹 애플리케이션을 구축하기 위한 경량 ASGI 프레임워크입니다.
- **anyio.to_thread.run_sync**: 블로킹(동기) 코드를 백그라운드 스레드에서 실행하여 비동기 이벤트 루프를 차단하지 않도록 하는 유틸리티입니다.
- **CodeAgent**: 프로그래밍 방식으로 작업을 해결할 수 있는 `smolagents` 라이브러리의 에이전트입니다.

## 백그라운드 스레드를 사용하는 이유는?[[why-use-a-background-thread?]]

`CodeAgent.run()`은 Python 코드를 동기적으로 실행합니다. 비동기 엔드포인트에서 직접 호출하면 Starlette의 이벤트 루프를 차단하여 성능과 확장성이 저하됩니다. `anyio.to_thread.run_sync`로 이 작업을 백그라운드 스레드로 위임하면 높은 동시성에서도 앱의 응답성과 효율성을 유지할 수 있습니다.

## 예시 워크플로우[[example-workflow]]

- Starlette 앱은 `task` 문자열이 포함된 JSON 페이로드를 받는 `/run-agent` 엔드포인트를 노출합니다.
- 요청이 수신되면 `anyio.to_thread.run_sync`를 사용하여 백그라운드 스레드에서 에이전트가 실행됩니다.
- 결과는 JSON 응답으로 반환됩니다.

## CodeAgent를 활용한 Starlette 앱 구축[[building-a-starlette-app-with-a-codeagent]]

### 1. 의존성 설치[[1.-install-dependencies]]

```bash
pip install smolagents starlette anyio uvicorn
```

### 2. 애플리케이션 코드 (`main.py`)[[2.-application-code-(`main.py`)]]

```python
import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
    tools=[],
)

async def run_agent(request: Request):
    data = await request.json()
    task = data.get("task", "")
    # Run the agent synchronously in a background thread
    result = await anyio.to_thread.run_sync(agent.run, task)
    return JSONResponse({"result": result})

app = Starlette(routes=[
    Route("/run-agent", run_agent, methods=["POST"]),
])
```

### 3. 앱 실행[[3.-run-the-app]]

```bash
uvicorn async_agent.main:app --reload
```

### 4. 엔드포인트 테스트[[4.-test-the-endpoint]]

```bash
curl -X POST http://localhost:8000/run-agent -H 'Content-Type: application/json' -d '{"task": "What is 2+2?"}'
```

**예상 응답:**

```json
{"result": "4"}
```

## 추가 자료[[further-reading]]

- [Starlette 문서](https://www.starlette.io/)
- [anyio 문서](https://anyio.readthedocs.io/)

---

전체 코드는 [`examples/async_agent`](https://github.com/huggingface/smolagents/tree/main/examples/async_agent)를 참고하세요.
