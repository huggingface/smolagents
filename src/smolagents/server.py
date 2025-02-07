import json
import re
import time
import uuid
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from smolagents.agents import ActionStep, CodeAgent, MultiStepAgent


class Message(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: bool = False
    reasoning_format: str = "raw"  # Options: "parsed", "raw", "hidden"
    json_mode: bool = False
    reset: bool = True


class ModelPermission(BaseModel):
    id: str = str(uuid.uuid4())
    created: int = int(time.time())
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = False
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    is_blocking: bool = False


class Model(BaseModel):
    id: str
    created: int = int(time.time())
    owned_by: str = "smolagents"
    permission: List[ModelPermission] = [ModelPermission()]


class ModelList(BaseModel):
    data: List[Model]
    object: str = "list"


app = FastAPI(title="SmolAgentsServer", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Global registry for agents
_agent_registry: Dict[str, Union[CodeAgent, MultiStepAgent]] = {}


def create_stream_chunk(
    completion_id: str, created: int, model: str, content: str = None, role: str = None, finish_reason: str = None
) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"delta": {}}],
    }
    if role:
        chunk["choices"][0]["delta"]["role"] = role
    if content:
        chunk["choices"][0]["delta"]["content"] = content
    if finish_reason:
        chunk["choices"][0]["finish_reason"] = finish_reason
    return f"data: {json.dumps(chunk)}\n\n"


def clean_output(text: str) -> str:
    """Clean output text by removing standard prefixes."""
    if not text:
        return text
    text = text.strip()
    # Apply cleanings in sequence to ensure all prefixes are removed
    text = re.sub(r"^Execution logs:\s*", "", text)
    text = re.sub(r"Last output from code snippet:\s*", "", text)
    return text


async def stream_agent_response(
    agent, task: str, model: str, reasoning_format: str = "raw", messages: List[Message] = None, reset: bool = True
):
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    yield create_stream_chunk(completion_id, created, model, role="assistant")

    if reset and messages and len(messages) > 1:
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[:-1]])
        task = f"Previous conversation:\n{context}\n\nCurrent request: {task}"

    for step_log in agent.run(task, stream=True, reset=reset):
        if not isinstance(step_log, ActionStep):
            continue

        if step_log.model_output:
            thought = step_log.model_output.split("Code:", 1)[0].replace("Thought:", "", 1).strip()
            if thought and reasoning_format != "hidden":
                duration = round(float(step_log.duration), 2) if hasattr(step_log, "duration") else None
                yield create_stream_chunk(completion_id, created, model, f'<think duration="{duration}">')
                yield create_stream_chunk(completion_id, created, model, thought)
                yield create_stream_chunk(completion_id, created, model, "</think>\n")

        if step_log.tool_calls:
            for tool_call in step_log.tool_calls:
                if tool_call.name == "python_interpreter":
                    content = tool_call.arguments
                    if isinstance(content, dict):
                        content = str(content.get("answer", str(content)))
                    content = re.sub(r"(```.*?\n|<end_code>|^python\n)", "", content.strip())
                    if content:
                        yield create_stream_chunk(completion_id, created, model, '<think type="code">')
                        yield create_stream_chunk(completion_id, created, model, "```python\n")
                        yield create_stream_chunk(completion_id, created, model, content)
                        yield create_stream_chunk(completion_id, created, model, "\n```")
                        yield create_stream_chunk(completion_id, created, model, "</think>\n")

                if step_log.observations and step_log.observations.strip():
                    print(tool_call.arguments)
                    log_content = step_log.observations.strip()
                    log_content = clean_output(log_content)
                    if log_content and log_content != "None":
                        if "final_answer" in tool_call.arguments:
                            yield create_stream_chunk(completion_id, created, model, f"{log_content}")
                        else:
                            yield create_stream_chunk(completion_id, created, model, '<think type="result">')
                            yield create_stream_chunk(completion_id, created, model, f"{log_content}")
                            yield create_stream_chunk(completion_id, created, model, "</think>\n")

                if step_log.error:
                    duration = round(float(step_log.duration), 2) if hasattr(step_log, "duration") else None
                    yield create_stream_chunk(
                        completion_id,
                        created,
                        model,
                        f'<think type="error" duration="{duration}">Error: {step_log.error}</think>\n',
                    )

        if not hasattr(step_log, "step_number") and step_log.observations:
            final_answer = step_log.observations.strip()
            final_answer = clean_output(final_answer)
            if final_answer and final_answer != "None":
                yield create_stream_chunk(completion_id, created, model, f"{final_answer}\n")

    yield create_stream_chunk(completion_id, created, model, finish_reason="stop")
    yield "data: [DONE]\n\n"


class AgentServer:
    def __init__(self, agents: Dict[str, Union[CodeAgent, MultiStepAgent]] | Union[CodeAgent, MultiStepAgent] = None):
        self.agents = {"default": agents} if isinstance(agents, (CodeAgent, MultiStepAgent)) else agents or {}
        _agent_registry.update(self.agents)
        self.setup_routes()

    def setup_routes(self):
        @app.get("/v1/models")
        def list_models() -> ModelList:
            return ModelList(data=[Model(id=model_id) for model_id in _agent_registry])

        @app.get("/v1/models/{model_id}")
        def get_model(model_id: str) -> Model:
            if model_id not in _agent_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            return Model(id=model_id)

        @app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest) -> Response:
            if request.model not in _agent_registry:
                raise HTTPException(status_code=404, detail="Model not found")
            if request.reasoning_format not in ["raw", "parsed", "hidden"]:
                raise HTTPException(
                    status_code=400, detail="Invalid reasoning_format. Must be 'raw', 'parsed', or 'hidden'"
                )
            if request.json_mode and request.reasoning_format == "raw":
                raise HTTPException(status_code=400, detail="Raw reasoning format is not supported with JSON mode")

            agent = _agent_registry[request.model]
            user_request = request.messages[-1].content

            if request.stream:
                return StreamingResponse(
                    stream_agent_response(
                        agent=agent,
                        task=user_request,
                        model=request.model,
                        reasoning_format=request.reasoning_format,
                        messages=request.messages,
                        reset=request.reset,
                    ),
                    media_type="text/event-stream",
                )

            if request.reset and len(request.messages) > 1:
                context = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[:-1]])
                user_request = f"Previous conversation:\n{context}\n\nCurrent request: {user_request}"

            response = agent.run(user_request, reset=request.reset, stream=False)
            response = clean_output(str(response))

            if request.reasoning_format == "hidden":
                content = response
                reasoning = None
            elif request.reasoning_format == "parsed":
                content = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL).strip()
                reasoning = "\n".join(re.findall(r"<think>(.*?)</think>", response, re.DOTALL)) or None
            else:
                content = response
                reasoning = None

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=content),
                        reasoning=reasoning,
                        finish_reason="stop",
                    )
                ],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    from smolagents import CodeAgent, HfApiModel

    agent = CodeAgent(tools=[], model=HfApiModel(), max_steps=4, verbosity_level=1, add_base_tools=True)
    server = AgentServer(agent)
    server.run()
