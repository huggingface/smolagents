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

from dotenv import load_dotenv

from smolagents import CodeAgent, HfApiModel, LiteLLMModel, Model, OpenAIServerModel, Tool, TransformersModel
from smolagents.default_tools import TOOL_MAPPING


leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def parse_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=leopard_prompt,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="HfApiModel",
        help="The model type to use (e.g., HfApiModel, OpenAIServerModel, LiteLLMModel, TransformersModel)",
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
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    return parser.parse_args()


def load_model(model_type: str, model_id: str) -> Model:
    if model_type == "OpenAIServerModel":
        return OpenAIServerModel(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            api_base="https://api.fireworks.ai/inference/v1",
            model_id=model_id,
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto", flatten_messages_as_text=False)
    elif model_type == "HfApiModel":
        return HfApiModel(
            token=os.getenv("HF_API_KEY"),
            model_id=model_id,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    load_dotenv()

    args = parse_arguments(description="Run a CodeAgent with all specified parameters")

    model = load_model(args.model_type, args.model_id)

    available_tools = []
    for tool_name in args.tools:
        if "/" in tool_name:
            available_tools.append(Tool.from_space(tool_name))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    print(f"Running agent with these tools: {args.tools}")
    agent = CodeAgent(tools=available_tools, model=model, additional_authorized_imports=args.imports)

    agent.run(args.prompt)


if __name__ == "__main__":
    main()
