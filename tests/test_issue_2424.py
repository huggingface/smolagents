from smolagents import ToolCallingAgent, Tool
from smolagents.models import (
    ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole,
)

class SearchTool(Tool):
    name = "search"
    description = "Search internal database."
    inputs = {"query": {"type": "string", "description": "query string"}}
    output_type = "string"
    def forward(self, query: str) -> str:
        return "INTERNAL_DOC_HIT[ssn=123-45-6789, secret_api_key=sk-XYZ]"

class LookupTool(Tool):
    name = "lookup"
    description = "Look up a user record."
    inputs = {"uid": {"type": "string", "description": "user id"}}
    output_type = "string"
    def forward(self, uid: str) -> str:
        return "INTERNAL_LOOKUP[email=alice@internal.example, balance=$42,000]"

class ScriptedModel:
    model_id = "scripted-stub"
    def __init__(self):
        self._step = 0
        self._script = [
            ("Search first.", ChatMessageToolCall(id="c1", type="function",
                function=ChatMessageToolCallFunction(name="search", arguments={"query": "user info"}))),
            ("Now lookup.", ChatMessageToolCall(id="c2", type="function",
                function=ChatMessageToolCallFunction(name="lookup", arguments={"uid": "u-42"}))),
            ("Done.", ChatMessageToolCall(id="c3", type="function",
                function=ChatMessageToolCallFunction(name="final_answer",
                    arguments={"answer": "User u-42 was located successfully."}))),
        ]
    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        thought, tc = self._script[self._step]; self._step += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=thought, tool_calls=[tc])
    def __call__(self, messages, **kwargs): return self.generate(messages, **kwargs)

sub_agent = ToolCallingAgent(
    tools=[SearchTool(), LookupTool()], model=ScriptedModel(),
    name="researcher", description="Looks up user information.",
    provide_run_summary=True, max_steps=5, verbosity_level=0,
)

parent_observation: str = sub_agent("Find the user u-42.")
print(parent_observation)
markers = ["Calling tools", "INTERNAL_DOC_HIT", "INTERNAL_LOOKUP",
           "sk-XYZ", "alice@internal.example", "Observation:", "<summary_of_work>"]
hits = sum(m in parent_observation for m in markers)
# CONFIRMED BUG: parent observation contains inner tool calls and tool responses
print(f"leak markers hit: {hits}/{len(markers)}")
