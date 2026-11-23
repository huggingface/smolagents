import pytest

from smolagents import CodeAgent
from smolagents.memory import ActionStep, PlanningStep
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import LogLevel, TokenUsage


@pytest.mark.parametrize("stream", [False, True])
def test_replanning_totals_reset_between_runs(stream):
    class CountingModel(Model):
        action_calls = 0

        def generate(self, messages, stop_sequences=None, **kwargs):
            if stop_sequences == ["<end_plan>"]:
                return ChatMessage(
                    role=MessageRole.ASSISTANT, content="Continue the plan.", token_usage=TokenUsage(100, 7)
                )
            self.action_calls += 1
            code = "final_answer(42)" if self.action_calls % 3 == 0 else "print(1)"
            return ChatMessage(
                role=MessageRole.ASSISTANT, content=f"<code>\n{code}\n</code>", token_usage=TokenUsage(10, 2)
            )

    agent = CodeAgent(tools=[], model=CountingModel(), planning_interval=1, max_steps=4, verbosity_level=LogLevel.OFF)
    for _ in range(2):
        result = agent.run("Count three actions", stream=stream, return_full_result=True)
        if stream:
            list(result)
        assert sum(isinstance(step, PlanningStep) for step in agent.memory.steps) == 3
        assert sum(isinstance(step, ActionStep) for step in agent.memory.steps) == 3
        assert agent.monitor.get_total_token_counts() == TokenUsage(330, 27)
        assert len(agent.monitor.step_durations) == 6
        if not stream:
            assert result.token_usage == TokenUsage(330, 27)
