from unittest.mock import patch

import pytest

from smolagents.agents import MultiStepAgent


original_init = MultiStepAgent.__init__


@pytest.fixture(autouse=True)
def patch_multi_step_agent():
    with patch.object(MultiStepAgent, "__init__", autospec=True) as mock_init:

        def init_with_verbosity(self, *args, verbosity_level=-1, **kwargs):
            # kwargs['verbosity_level'] = -1
            original_init(self, *args, verbosity_level=verbosity_level, **kwargs)

        mock_init.side_effect = init_with_verbosity
        yield
