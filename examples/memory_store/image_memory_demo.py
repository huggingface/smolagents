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

"""Example demonstrating image analysis with memory persistence."""

# Standard library imports
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


# Add the project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Local imports
from smolagents import CodeAgent
from smolagents.memory_store import AgentMemoryEncoder, load_agent_state, save_agent_state
from smolagents.models import LiteLLMModel


def get_model():
    """Get the model instance"""
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20241022", api_key=os.environ["ANTHROPIC_API_KEY"])
    return model


def run_first_agent():
    """Run the first agent to analyze the image and save its memory"""
    print("\n=== Starting First Agent Session ===")

    model = get_model()
    agent1 = CodeAgent(tools=[], model=model)

    # Load the image from URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/08.12BarinN23-r2.jpg"
    response = requests.get(image_url)
    if response.status_code != 200:
        print(f"Error: Unable to download image from {image_url}")
        sys.exit(1)

    image = Image.open(BytesIO(response.content))

    print("\nAgent 1: Analyzing the image...")
    response1 = agent1.run(
        "Please analyze this image in detail. Describe what you see, including colors, "
        "objects, composition, and any notable features. Store this information carefully "
        "as you will be asked about it later.",
        images=[image],
        reset=True,
    )
    print(f"\nAgent 1 analysis:\n{response1}")

    print("\nSaving agent state...")
    memory_state = save_agent_state(agent1)
    with open("examples/memory_store/image_memory.json", "w") as f:
        json.dump(memory_state, f, indent=2, cls=AgentMemoryEncoder)
    print("Memory saved to examples/memory_store/image_memory.json")


def run_second_agent():
    """Run the second agent with restored memory to recall image details"""
    print("\n=== Starting Second Agent Session ===")

    model = get_model()
    agent2 = CodeAgent(tools=[], model=model)

    print("\nLoading previous agent state...")
    with open("examples/memory_store/image_memory.json", "r") as f:
        stored_state = json.load(f)
    load_agent_state(agent2, stored_state)

    print("\nAgent 2: Testing image memory...")
    questions = [
        "What was in the image that you analyzed earlier? Please describe it in detail.",
        "What colors were prominent in the image?",
        "Was there anything unusual or particularly interesting about the image?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response2 = agent2.run(question, reset=False)
        print(f"Response: {response2}")


def main():
    """Main function to run the demo"""
    print("=== Image Analysis Memory Demo using a Vision Model ===")

    # Create output directory if it doesn't exist
    os.makedirs("examples/memory_store", exist_ok=True)

    # Run first agent to analyze image
    run_first_agent()

    # Run second agent to test memory
    run_second_agent()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
