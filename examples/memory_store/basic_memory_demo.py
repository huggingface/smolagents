#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Example demonstrating basic memory persistence with a free HuggingFace model."""

import json
import os
import sys
from pathlib import Path


# Add the project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from smolagents import CodeAgent, HfApiModel
from smolagents.memory_store import AgentMemoryEncoder, load_agent_state, save_agent_state


def get_model():
    """Get the model instance"""
    model = HfApiModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    return model


def run_first_agent():
    """Run the first agent and save its memory"""
    print("\n=== Starting First Agent Session ===")

    model = get_model()
    agent1 = CodeAgent(tools=[], model=model)

    print("\nAgent 1: Starting conversation...")
    response1 = agent1.run("Hello! My name is Alice. What's 2+2?", reset=True)
    print(f"Agent 1 response: {response1}")

    response2 = agent1.run("Now add 2 to the result", reset=False)
    print(f"Agent 1 response: {response2}")

    print("\nSaving agent state...")
    memory_state = save_agent_state(agent1)
    with open("examples/memory_store/basic_memory.json", "w") as f:
        json.dump(memory_state, f, indent=2, cls=AgentMemoryEncoder)
    print("Memory saved to basic_memory.json")


def run_second_agent():
    """Run the second agent with restored memory"""
    print("\n=== Starting Second Agent Session ===")

    model = get_model()
    agent2 = CodeAgent(tools=[], model=model)

    print("\nLoading previous agent state...")
    with open("examples/memory_store/basic_memory.json", "r") as f:
        stored_state = json.load(f)
    load_agent_state(agent2, stored_state)

    print("\nAgent 2: Testing memory...")
    response = agent2.run("What's my name? And what was the previous calculation?", reset=False)
    print(f"Agent 2 response: {response}")


def main():
    """Main function to run the demo"""
    print("=== Basic Memory Persistence Demo ===")

    # Create output directory if it doesn't exist
    os.makedirs("examples/memory_store", exist_ok=True)

    # Run first agent
    run_first_agent()

    # Run second agent
    run_second_agent()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
