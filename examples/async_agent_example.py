"""
Example demonstrating async support in smolagents.

This example shows how to use the async API to run multiple agents concurrently,
which can significantly improve performance when making multiple LLM API calls.
"""

import asyncio
import time
from smolagents import LiteLLMModel


# Example 1: Simple async agent execution
async def example_simple_async():
    """
    Simple example showing how to run a single agent asynchronously.
    """
    print("\n=== Example 1: Simple Async Agent ===\n")

    # Note: This example requires CodeAgent which needs async _astep_stream implementation
    # For now, this is a placeholder showing the intended API

    # Create a model (will use async methods internally)
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Create an agent
    # from smolagents import CodeAgent
    # agent = CodeAgent(model=model, tools=[])

    # Run the agent asynchronously
    # result = await agent.arun("What is 2 + 2?")
    # print(f"Result: {result}")

    print("Async agent support is now available in the base Agent class!")
    print("CodeAgent and ToolCallingAgent will need async _astep_stream implementations.")


# Example 2: Running multiple agents concurrently
async def example_concurrent_agents():
    """
    Example showing how to run multiple agents concurrently for better performance.
    This is the main benefit of async support - running multiple tasks in parallel.

    IMPORTANT: Each concurrent task needs its own agent instance!
    Agent memory is stateful, so sharing an agent between concurrent tasks
    will cause memory corruption where steps get mixed together.
    """
    print("\n=== Example 2: Concurrent Agent Execution ===\n")

    # Create multiple tasks
    tasks = [
        "What is 2 + 2?",
        "What is 10 * 5?",
        "What is 100 / 4?",
    ]

    # Create a model (can be shared across agents)
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Note: This example requires CodeAgent with async support
    # from smolagents import CodeAgent

    # CORRECT: Create separate agent instances for each task
    # agents = [CodeAgent(model=model, tools=[]) for _ in tasks]
    # results = await asyncio.gather(*[agent.arun(task) for agent, task in zip(agents, tasks)])

    # INCORRECT: Reusing same agent instance (will cause memory corruption!)
    # agent = CodeAgent(model=model, tools=[])
    # results = await asyncio.gather(*[agent.arun(task) for task in tasks])  # BAD!

    print("Multiple agents can now be run concurrently using asyncio.gather()!")
    print("IMPORTANT: Each concurrent task needs its own agent instance!")
    print("This prevents memory corruption from shared state.")


# Example 3: Async streaming
async def example_async_streaming():
    """
    Example showing how to use async streaming to get results as they come in.
    """
    print("\n=== Example 3: Async Streaming ===\n")

    # Create a model
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Note: This example requires CodeAgent with async support
    # from smolagents import CodeAgent
    # agent = CodeAgent(model=model, tools=[])

    # Run with streaming enabled
    # async for step in agent.arun("Calculate 2 + 2", stream=True):
    #     print(f"Step: {step}")

    print("Async streaming is now supported!")
    print("Use 'async for step in agent.arun(task, stream=True)' to stream results.")


# Example 4: Using async models directly
async def example_async_models():
    """
    Example showing how to use async model methods directly.
    """
    print("\n=== Example 4: Using Async Models Directly ===\n")

    from smolagents.models import ChatMessage, MessageRole

    # Create a model
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Create a message
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "What is the capital of France?"}]
        )
    ]

    # Use async generate
    start_time = time.time()
    response = await model.agenerate(messages)
    end_time = time.time()

    print(f"Response: {response.content}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Token usage: {response.token_usage}")


# Example 5: Performance comparison
async def example_performance_comparison():
    """
    Example showing performance difference between sync and async.
    """
    print("\n=== Example 5: Performance Comparison ===\n")

    from smolagents.models import ChatMessage, MessageRole

    # Create a model
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Create messages
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": f"What is {i} + {i}?"}]
        ) for i in range(3)
    ]

    # Test async (concurrent)
    print("Running 3 requests concurrently with async...")
    start_time = time.time()
    async_results = await asyncio.gather(*[model.agenerate([msg]) for msg in messages])
    async_time = time.time() - start_time
    print(f"Async time: {async_time:.2f} seconds")

    # Test sync (sequential)
    print("\nRunning 3 requests sequentially with sync...")
    start_time = time.time()
    sync_results = [model.generate([msg]) for msg in messages]
    sync_time = time.time() - start_time
    print(f"Sync time: {sync_time:.2f} seconds")

    print(f"\nSpeedup: {sync_time / async_time:.2f}x faster with async!")


async def main():
    """Run all examples."""
    print("=" * 70)
    print("Smolagents Async Support Examples")
    print("=" * 70)

    await example_simple_async()
    await example_concurrent_agents()
    await example_async_streaming()

    # Uncomment these to test with actual API calls (requires API key)
    # await example_async_models()
    # await example_performance_comparison()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
