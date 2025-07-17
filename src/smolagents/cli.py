#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
from multiprocessing import Manager, Process

from dotenv import load_dotenv

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, Model, OpenAIServerModel, Tool, TransformersModel
from smolagents.default_tools import TOOL_MAPPING


leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple CodeAgents with direct communication")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=leopard_prompt,
        help="The prompt to run with the first agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        help="The model type to use (e.g., InferenceClientModel, OpenAIServerModel, LiteLLMModel, TransformersModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--imports",
        nargs="*",  # accepts zero or more arguments
        default=[],
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas')",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools that the agents can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents to run",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    return parser.parse_args()


def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> Model:
    if model_type == "OpenAIServerModel":
        return OpenAIServerModel(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            api_base=api_base or "https://api.fireworks.ai/inference/v1",
            model_id=model_id,
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    elif model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_TOKEN"),
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def start_agent(
    agent_id: int,
    queue_dict: dict,
    prompt: str | None,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None,
    api_key: str | None,
    imports: list[str] | None,
    provider: str | None,
    verbosity_level: int,
) -> None:
    load_dotenv()
    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key, provider=provider)
    available_tools = []
    for tool_name in tools:
        if "/" in tool_name:
            available_tools.append(Tool.from_space(tool_name))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")
    print(f"Agent {agent_id} running with tools: {tools}")
    agent = CodeAgent(
        tools=available_tools,
        model=model,
        agent_id=agent_id,
        queue_dict=queue_dict,
        additional_authorized_imports=imports,
        verbosity_level=verbosity_level,
    )
    agent.run(prompt)


def run_smolagent(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    num_agents: int,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
    provider: str | None = None,
    verbosity_level: int = 1,
) -> None:
    with Manager() as manager:
        queue_dict = manager.dict()
        for i in range(num_agents):
            queue_dict[i] = manager.Queue()
        processes = [
            Process(
                target=start_agent,
                args=(
                    i,
                    queue_dict,
                    prompt if i == 0 else None,  # Initial task for Agent 0
                    tools,
                    model_type,
                    model_id,
                    api_base,
                    api_key,
                    imports,
                    provider,
                    verbosity_level,
                ),
            )
            for i in range(num_agents)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def main() -> None:
    args = parse_arguments()
    run_smolagent(
        args.prompt,
        args.tools,
        args.model_type,
        args.model_id,
        args.num_agents,
        provider=args.provider,
        api_base=args.api_base,
        api_key=args.api_key,
        imports=args.imports,
        verbosity_level=args.verbosity_level,
    )


if __name__ == "__main__":
    main()
