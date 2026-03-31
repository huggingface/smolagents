# 使用Agent的异步应用

本指南演示了如何将 `smolagents` 库中的同步agent集成到使用 Starlette 构建的异步 Python Web 应用程序中。此示例旨在帮助初次接触异步 Python 和agent集成的用户理解将同步智能体逻辑与异步 Web 服务器结合的最佳实践。

## 概述

- **Starlette**：一个轻量级的 ASGI 框架，用于在 Python 中构建异步 Web 应用程序。
- **anyio.to_thread.run_sync**：一个实用工具，用于在后台线程中运行阻塞（同步）代码，防止其阻塞异步事件循环。
- **CodeAgent**：来自 `smolagents` 库的一个agent，能够通过编程方式解决任务。

## 为何使用后台线程？

`CodeAgent.run()` 是同步执行 Python 代码的。如果在异步端点中直接调用，它会阻塞 Starlette 的事件循环，从而降低性能和可扩展性。通过使用 `anyio.to_thread.run_sync` 将此操作卸载到后台线程，即使在高并发下，也能保持应用程序的响应性和高效性。

## 工作流程示例

- Starlette 应用暴露一个 `/run-agent` 端点，该端点接收包含 `task` 字符串的 JSON 负载。
- 收到请求时，使用 `anyio.to_thread.run_sync` 在后台线程中运行agent。
- 结果以 JSON 响应的形式返回。

## 使用 CodeAgent 构建 Starlette 应用

### 1. 安装依赖

```bash
pip install smolagents starlette anyio uvicorn
```

### 2. 应用程序代码 (`main.py`)

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
    # 在后台线程中同步运行智能体
    result = await anyio.to_thread.run_sync(agent.run, task)
    return JSONResponse({"result": result})

app = Starlette(routes=[
    Route("/run-agent", run_agent, methods=["POST"]),
])
```

### 3. 运行应用

```bash
uvicorn async_agent.main:app --reload
```

### 4. 测试端点

```bash
curl -X POST http://localhost:8000/run-agent -H 'Content-Type: application/json' -d '{"task": "What is 2+2?"}'
```

**预期响应：**

```json
{"result": "4"}
```

## 扩展阅读

- [Starlette 文档](https://www.starlette.io/)
- [anyio 文档](https://anyio.readthedocs.io/)

---

完整代码请参见 [`examples/async_agent`](https://github.com/huggingface/smolagents/tree/main/examples/async_agent)。