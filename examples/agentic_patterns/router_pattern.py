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

"""
Router Pattern Example: Directing tasks to specialized agents based on task type.
"""

from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

# Create specialized agents
# - math_agent
math_agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    name="math_agent",
    description="Specialized in solving mathematical problems and calculations.",
)
# - research_agent
research_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(),
    name="research_agent",
    description="Specialized in researching information and providing factual answers.",
)
# - coding_agent
coding_agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    name="coding_agent",
    description="Specialized in writing and debugging code.",
    additional_authorized_imports=["numpy", "pandas", "matplotlib"],
)
# Create a list of the specialized agents
specialized_agents = [math_agent, research_agent, coding_agent]
# Create a string with the names and descriptions of the specialized agents
specialized_agents_names_and_descriptions = "\n".join(
    [f"- {agent.name}: {agent.description}" for agent in specialized_agents]
)

# Create the router agent
router_agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
)
# Define the task routing prompt
router_task = f"""\
Choose the name of the most appropriate specialized agent for this task:
{{task}}

The specialized agents are:
{specialized_agents_names_and_descriptions}"""


# Example tasks for different specialized agents
tasks = [
    "Calculate the compound interest on $10,000 invested for 5 years at an annual rate of 8% compounded quarterly.",
    "What are the latest developments in quantum computing as of 2023?",
    "Write a Python function to find the longest common subsequence of two strings.",
]
for task in tasks:
    print(f"\n{'=' * 80}\nProcessing task: {task}\n{'=' * 80}")
    response = router_agent.run(router_task.format(task=task))
    print(f"\nFinal response: {response}")
