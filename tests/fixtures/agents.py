import pytest


AGENT_DICTS = {
    "v1.9": {
        "tools": [],
        "model": {
            "class": "InferenceClientModel",
            "data": {
                "last_input_token_count": None,
                "last_output_token_count": None,
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": {
            "system_prompt": "dummy system prompt",
            "planning": {
                "initial_facts": "dummy planning initial facts",
                "initial_plan": "dummy planning initial plan",
                "update_facts_pre_messages": "dummy planning update facts pre messages",
                "update_facts_post_messages": "dummy planning update facts post messages",
                "update_plan_pre_messages": "dummy planning update plan pre messages",
                "update_plan_post_messages": "dummy planning update plan post messages",
            },
            "managed_agent": {
                "task": "dummy managed agent task",
                "report": "dummy managed agent report",
            },
            "final_answer": {
                "pre_messages": "dummy final answer pre messages",
                "post_messages": "dummy final answer post messages",
            },
        },
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
            "class": "InferenceClientModel",
            "data": {
                "last_input_token_count": None,
                "last_output_token_count": None,
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": {
            "system_prompt": "dummy system prompt",
            "planning": {
                "initial_facts": "dummy planning initial facts",
                "initial_plan": "dummy planning initial plan",
                "update_facts_pre_messages": "dummy planning update facts pre messages",
                "update_facts_post_messages": "dummy planning update facts post messages",
                "update_plan_pre_messages": "dummy planning update plan pre messages",
                "update_plan_post_messages": "dummy planning update plan post messages",
            },
            "managed_agent": {
                "task": "dummy managed agent task",
                "report": "dummy managed agent report",
            },
            "final_answer": {
                "pre_messages": "dummy final answer pre messages",
                "post_messages": "dummy final answer post messages",
            },
        },
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
    # Removed: grammar, last_input_token_count, last_output_token_count
    "v1.20": {
        "tools": [],
        "model": {
            "class": "InferenceClientModel",
            "data": {
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": {
            "system_prompt": "dummy system prompt",
            "planning": {
                "initial_facts": "dummy planning initial facts",
                "initial_plan": "dummy planning initial plan",
                "update_facts_pre_messages": "dummy planning update facts pre messages",
                "update_facts_post_messages": "dummy planning update facts post messages",
                "update_plan_pre_messages": "dummy planning update plan pre messages",
                "update_plan_post_messages": "dummy planning update plan post messages",
            },
            "managed_agent": {
                "task": "dummy managed agent task",
                "report": "dummy managed agent report",
            },
            "final_answer": {
                "pre_messages": "dummy final answer pre messages",
                "post_messages": "dummy final answer post messages",
            },
        },
        "max_steps": 10,
        "verbosity_level": 2,
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
