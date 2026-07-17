from unittest.mock import MagicMock, patch

import pytest

from smolagents.agents import CodeAgent
from smolagents.models import Model
from smolagents.utils import AgentGenerationError


class FailingModel(Model):
    def generate(self, messages, stop_sequences=None):
        raise RuntimeError("LLM API quota exceeded")


def test_issue_2050():
    docker_executor = MagicMock()

    with patch("smolagents.agents.DockerExecutor", return_value=docker_executor):
        agent = CodeAgent(tools=[], model=FailingModel(), executor_type="docker")

        with pytest.raises(AgentGenerationError, match="LLM API quota exceeded"):
            agent.run("What was Abraham Lincoln's preferred pet?")

    docker_executor.cleanup.assert_called_once()
