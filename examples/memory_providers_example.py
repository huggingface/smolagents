#!/usr/bin/env python
# coding=utf-8

"""
Example demonstrating the use of different memory providers with smolagents.

This example shows how to use the default AgentMemory and alternative memory providers
like DictMemory and FilteredMemory.
"""

from smolagents import CodeAgent
from smolagents.models import ChatMessage, MessageRole


# Create a mock model that doesn't require API calls
class MockModel:
    def __init__(self):
        self.model_id = "MockModel"

    def __call__(self, messages, **kwargs):
        # Return a predefined response
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nresult = fibonacci(20)\nresult\n```",
        )

    def to_dict(self):
        return {"class": "MockModel", "data": {}}


from smolagents.memory import ActionStep
from smolagents.memory_providers import DictMemory, FilteredMemory


def run_with_default_memory():
    """Run an agent with the default AgentMemory."""
    print("\n=== Running with Default AgentMemory ===\n")

    # Create an agent with the default memory provider
    agent = CodeAgent(tools=[], model=MockModel(), verbosity_level=1)

    # Run the agent
    result = agent.run("What's the 20th Fibonacci number?")

    # Access the memory
    print("\nAccessing memory steps:")
    for i, step in enumerate(agent.memory.steps):
        if hasattr(step, "step_number"):
            print(f"Step {step.step_number}: {type(step).__name__}")
        else:
            print(f"Step {i}: {type(step).__name__}")

    # Skip replay for now
    print("\nSkipping replay for simplicity")
    # agent.replay()

    return result


def run_with_dict_memory():
    """Run an agent with the DictMemory provider."""
    print("\n=== Running with DictMemory ===\n")

    # Create a simple recursive Fibonacci function to use as a tool
    def fibonacci(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)

    # Create an agent with the Fibonacci tool
    agent = CodeAgent(tools=[], model=MockModel(), verbosity_level=1)

    # Add the Fibonacci function to the agent's state
    agent.state["fibonacci"] = fibonacci

    # Replace the default memory with DictMemory
    system_prompt = agent.system_prompt
    dict_memory = DictMemory(system_prompt)

    # Store the original memory
    original_memory = agent.memory

    # Replace with our custom memory
    agent.memory = dict_memory

    # Run the agent
    task = "What's the 20th Fibonacci number?"

    # We need to manually add the task step since we're using a custom memory provider
    from smolagents.memory import TaskStep

    dict_memory.add_step(TaskStep(task=task))

    # Run the agent step by step
    final_answer = None
    step_number = 1
    max_steps = 10

    while final_answer is None and step_number <= max_steps:
        memory_step = ActionStep(
            step_number=step_number,
            observations_images=[],
        )

        # Run one step
        final_answer = agent.step(memory_step)

        # Add the step to our custom memory
        dict_memory.add_step(memory_step)

        step_number += 1

    # Access the memory by step ID
    print("\nAccessing memory steps by ID:")
    for step_id, step in dict_memory.steps.items():
        print(f"{step_id}: {type(step).__name__}")

    # Get a specific step
    first_step_id = next(iter(dict_memory.steps.keys()))
    first_step = dict_memory.get_step(first_step_id)
    print(f"\nFirst step ({first_step_id}): {type(first_step).__name__}")

    # Skip replay for now
    print("\nSkipping replay for simplicity")
    # dict_memory.replay(agent.logger)

    # Restore the original memory
    agent.memory = original_memory

    return final_answer


def run_with_filtered_memory():
    """Run an agent with the FilteredMemory provider."""
    print("\n=== Running with FilteredMemory ===\n")

    # Create an agent with the default memory provider
    agent = CodeAgent(tools=[], model=MockModel(), verbosity_level=1)

    # Run the agent
    result = agent.run("What's the 20th Fibonacci number?")

    # Create a filtered memory that only includes steps without errors
    def no_error_filter(step):
        if isinstance(step, dict) and "error" in step:
            return step["error"] is None
        return True

    filtered_memory = FilteredMemory(agent.memory, no_error_filter)

    # Get steps from the filtered memory
    print("\nSteps from filtered memory (no errors):")
    for i, step in enumerate(filtered_memory.get_succinct_steps()):
        if "step" in step:
            print(f"Step {step['step']}")
        else:
            print(f"Step {i}")

    return result


def main():
    """Run the example."""
    # Run with default memory
    result1 = run_with_default_memory()
    print(f"\nResult with default memory: {result1}")

    # Run with dict memory
    result2 = run_with_dict_memory()
    print(f"\nResult with dict memory: {result2}")

    # Run with filtered memory
    result3 = run_with_filtered_memory()
    print(f"\nResult with filtered memory: {result3}")


if __name__ == "__main__":
    main()
