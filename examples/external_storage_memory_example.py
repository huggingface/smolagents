#!/usr/bin/env python
# coding=utf-8

"""
Example demonstrating the use of ExternalStorageMemory with smolagents.

This example shows how to use the ExternalStorageMemory provider to store memory in
external storage, create checkpoints, and resume conversations from checkpoints.
"""

from smolagents import CodeAgent
from smolagents.memory import ActionStep, TaskStep
from smolagents.memory_providers import ExternalStorageMemory
from smolagents.models import ChatMessage, MessageRole


# Create a mock model that doesn't require API calls
class MockModel:
    def __init__(self):
        self.model_id = "MockModel"
        self.step_count = 0

    def __call__(self, messages, **kwargs):
        # Return different responses based on step count to simulate a conversation
        self.step_count += 1

        if self.step_count == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# Calculate the 10th Fibonacci number\nresult = fibonacci(10)\nresult\n```",
            )
        elif self.step_count == 2:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="```python\n# Let's optimize the Fibonacci function using memoization\ndef fibonacci_memo(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)\n        return memo[n]\n\n# Calculate the 20th Fibonacci number\nresult = fibonacci_memo(20)\nresult\n```",
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="```python\n# Let's use an iterative approach for even better performance\ndef fibonacci_iter(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    \n    a, b = 0, 1\n    i = 2\n    while i <= n:\n        a, b = b, a + b\n        i += 1\n    return b\n\n# Calculate the 30th Fibonacci number\nresult = fibonacci_iter(30)\nresult\n```",
            )

    def to_dict(self):
        return {"class": "MockModel", "data": {}}


def run_with_external_storage():
    """Run an agent with the ExternalStorageMemory provider."""
    print("\n=== Running with ExternalStorageMemory ===\n")

    # Create an agent
    agent = CodeAgent(tools=[], model=MockModel(), verbosity_level=1)

    # Replace the default memory with ExternalStorageMemory
    system_prompt = agent.system_prompt
    external_memory = ExternalStorageMemory(
        system_prompt=system_prompt, storage_dir="memory_storage", conversation_id="fibonacci_example"
    )

    # Store the original memory
    original_memory = agent.memory

    # Replace with our custom memory
    agent.memory = external_memory

    # Run the agent
    task = "What's the Fibonacci sequence and how can we calculate it efficiently?"

    # We need to manually add the task step since we're using a custom memory provider
    external_memory.add_step(TaskStep(task=task))

    # Run the agent step by step
    final_answer = None
    step_number = 1
    max_steps = 3

    while final_answer is None and step_number <= max_steps:
        print(f"\nRunning step {step_number}...")
        memory_step = ActionStep(
            step_number=step_number,
            observations_images=[],
        )

        # Run one step
        final_answer = agent.step(memory_step)

        # Add the step to our custom memory
        external_memory.add_step(memory_step)

        # Save a checkpoint after each step
        checkpoint_name = external_memory.save_checkpoint(f"step_{step_number}")
        print(f"Saved checkpoint: {checkpoint_name}")

        step_number += 1

    # List all checkpoints
    print("\nListing checkpoint files:")
    for checkpoint_file in external_memory.conversation_dir.glob("*.json"):
        print(f"  - {checkpoint_file.name}")

    checkpoints = external_memory.list_checkpoints()
    print(f"\nAvailable checkpoints: {len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint['name']} (steps: {checkpoint['step_count']}, timestamp: {checkpoint['timestamp']})")

    # Restore the original memory
    agent.memory = original_memory

    return final_answer


def resume_from_checkpoint():
    """Resume a conversation from a checkpoint."""
    print("\n=== Resuming from Checkpoint ===\n")

    # Load the conversation
    try:
        external_memory = ExternalStorageMemory.load_conversation("fibonacci_example")
        print(f"Loaded conversation: {external_memory.conversation_id}")
    except FileNotFoundError:
        print("No previous conversation found. Please run the example first.")
        return None

    # List all checkpoints
    print("\nListing checkpoint files:")
    for checkpoint_file in external_memory.conversation_dir.glob("*.json"):
        print(f"  - {checkpoint_file.name}")

    checkpoints = external_memory.list_checkpoints()
    print(f"\nAvailable checkpoints: {len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint['name']} (steps: {checkpoint['step_count']}, timestamp: {checkpoint['timestamp']})")

    if not checkpoints:
        print("No checkpoints found. Please run the example first.")
        return None

    # Load the first checkpoint (step_1)
    checkpoint_name = "step_1"
    success = external_memory.load_checkpoint(checkpoint_name)
    if success:
        print(f"\nLoaded checkpoint: {checkpoint_name}")
    else:
        print(f"\nFailed to load checkpoint: {checkpoint_name}")
        return None

    # Create an agent with the loaded memory
    agent = CodeAgent(tools=[], model=MockModel(), verbosity_level=1)

    # Store the original memory
    original_memory = agent.memory

    # Replace with our loaded memory
    agent.memory = external_memory

    # Continue the conversation from the checkpoint
    final_answer = None
    step_number = 2  # Start from step 2 since we loaded checkpoint from step 1
    max_steps = 3

    while final_answer is None and step_number <= max_steps:
        print(f"\nRunning step {step_number} (continuing from checkpoint)...")
        memory_step = ActionStep(
            step_number=step_number,
            observations_images=[],
        )

        # Run one step
        final_answer = agent.step(memory_step)

        # Add the step to our custom memory
        external_memory.add_step(memory_step)

        step_number += 1

    # Replay the full conversation
    print("\nReplaying the full conversation:")
    external_memory.replay(agent.logger)

    # Restore the original memory
    agent.memory = original_memory

    return final_answer


def replay_full_conversation():
    """Replay a full conversation."""
    print("\n=== Replaying Full Conversation ===\n")

    # Load the conversation
    try:
        external_memory = ExternalStorageMemory.load_conversation("fibonacci_example")
        print(f"Loaded conversation: {external_memory.conversation_id}")
    except FileNotFoundError:
        print("No previous conversation found. Please run the example first.")
        return

    # Replay the full conversation
    print("\nReplaying the full conversation:")

    # Create a temporary agent just for its logger
    agent = CodeAgent(tools=[], model=MockModel(), verbosity_level=1)

    external_memory.replay(agent.logger)


def list_all_conversations():
    """List all available conversations."""
    print("\n=== Available Conversations ===\n")

    # Create a temporary memory instance to access the list_conversations method
    temp_memory = ExternalStorageMemory(system_prompt="temp")

    conversations = temp_memory.list_conversations()

    if conversations:
        print("Available conversations:")
        for conversation_id in conversations:
            print(f"  - {conversation_id}")
    else:
        print("No conversations found.")


def main():
    """Run the example."""
    # Run with external storage memory
    result1 = run_with_external_storage()
    print(f"\nResult with external storage memory: {result1}")

    # Resume from checkpoint
    result2 = resume_from_checkpoint()
    if result2:
        print(f"\nResult after resuming from checkpoint: {result2}")

    # Replay full conversation
    replay_full_conversation()

    # List all conversations
    list_all_conversations()


if __name__ == "__main__":
    main()
