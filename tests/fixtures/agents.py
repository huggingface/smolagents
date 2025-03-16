import pytest


AGENT_DICTS = {
    "v1.9": {
        "tools": [],
        "model": {
            "class": "HfApiModel",
            "data": {
                "last_input_token_count": None,
                "last_output_token_count": None,
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": None,
        "max_steps": 10,
        "verbosity_level": 2,
        "grammar": None,
        "planning_interval": 2,
        "name": "test_agent",
        "description": "dummy description",
        "requirements": ["smolagents"],
        "authorized_imports": ["pandas"],
    },
    # Added: executor_type, executor_kwargs, max_print_outputs_length
    "v1.10": {
        "tools": [],
        "model": {
            "class": "HfApiModel",
            "data": {
                "last_input_token_count": None,
                "last_output_token_count": None,
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": None,
        "max_steps": 10,
        "verbosity_level": 2,
        "grammar": None,
        "planning_interval": 2,
        "name": "test_agent",
        "description": "dummy description",
        "requirements": ["smolagents"],
        "authorized_imports": ["pandas"],
        "executor_type": "local",
        "executor_kwargs": {},
        "max_print_outputs_length": None,
    },
}


@pytest.fixture
def get_agent_dict():
    def _get_agent_dict(agent_dict_key):
        return AGENT_DICTS[agent_dict_key]

    return _get_agent_dict
