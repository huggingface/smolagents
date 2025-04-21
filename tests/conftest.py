from unittest.mock import patch

import pytest

from smolagents.agents import MultiStepAgent
from smolagents.monitoring import LogLevel


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.agents", "tests.fixtures.tools"]

original_multi_step_agent_init = MultiStepAgent.__init__


@pytest.fixture(autouse=True)
def patch_multi_step_agent_with_suppressed_logging():
    with patch.object(MultiStepAgent, "__init__", autospec=True) as mock_init:

        def init_with_suppressed_logging(self, *args, verbosity_level=LogLevel.OFF, **kwargs):
            original_multi_step_agent_init(self, *args, verbosity_level=verbosity_level, **kwargs)

        mock_init.side_effect = init_with_suppressed_logging
        yield
