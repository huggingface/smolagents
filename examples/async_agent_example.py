"""
Example demonstrating async support in smolagents.

This example shows how to use the async API with async tools, run multiple agents
concurrently, and leverage non-blocking I/O for human-in-the-loop workflows.
"""

import asyncio
import time
from smolagents import CodeAgent, Tool, LiteLLMModel


# Example async tool: Human approval (simulated)
class HumanApprovalTool(Tool):
    name = "human_approval"
    description = "Request human approval for an action. Use this before performing sensitive operations."
    inputs = {"action": {"type": "string", "description": "The action requiring approval"}}
    output_type = "string"

    async def forward(self, action: str):
        """Async tool that simulates waiting for human approval."""
        print(f"  [Tool] Requesting approval for: {action}")
        # Simulate network delay for approval system
        await asyncio.sleep(0.5)
        print(f"  [Tool] Approval received for: {action}")
        return f"approved: {action}"


# Example async tool: External API call (simulated)
class ExternalAPITool(Tool):
    name = "fetch_data"
    description = "Fetch data from an external API"
    inputs = {"query": {"type": "string", "description": "What data to fetch"}}
    output_type = "string"

    async def forward(self, query: str):
        """Async tool that simulates an API call with network latency."""
        print(f"  [Tool] Fetching data for: {query}")
        # Simulate API latency
        await asyncio.sleep(0.3)
        print(f"  [Tool] Data fetched for: {query}")
        return f"data for {query}: [sample result]"


# Example 1: Simple async agent execution with async tools
async def example_simple_async():
    """
    Simple example showing how to run an agent asynchronously with async tools.
    """
    print("\n=== Example 1: Simple Async Agent with Async Tools ===\n")

    # Create a model
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Create an agent with async tools
    agent = CodeAgent(model=model, tools=[HumanApprovalTool(), ExternalAPITool()])

    # Run the agent asynchronously - it will transparently handle async tools!
    print("Running agent with async tools...")
    result = await agent.arun("Use fetch_data to get user info, then get approval to process it")
    print(f"Result: {result}")


# Example 2: Running multiple agents concurrently
async def example_concurrent_agents():
    """
    Example showing how to run multiple agents concurrently with non-blocking I/O.

    The key benefit: While one agent waits for an async tool (like human_approval),
    the event loop can switch to other agents instead of blocking. This is far more
    efficient than threading where each blocked thread consumes 1-8MB of memory.

    IMPORTANT: Each concurrent task needs its own agent instance!
    Agent memory is stateful, so sharing an agent between concurrent tasks
    will cause memory corruption where steps get mixed together.
    """
    print("\n=== Example 2: Concurrent Agents with Non-Blocking I/O ===\n")

    # Create multiple tasks that require approval
    tasks = [
        "Request approval to delete user account",
        "Request approval to transfer funds",
        "Request approval to modify database",
    ]

    # Create a model (can be shared across agents)
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # CORRECT: Create separate agent instances for each task
    print("Creating 3 agents with async approval tool...")
    agents = [CodeAgent(model=model, tools=[HumanApprovalTool()]) for _ in tasks]

    # Run all agents concurrently - they'll efficiently share the event loop
    print("Running agents concurrently (watch how they interleave)...\n")
    start_time = time.time()
    results = await asyncio.gather(*[agent.arun(task) for agent, task in zip(agents, tasks)])
    elapsed = time.time() - start_time

    print(f"\n✓ All 3 agents completed in {elapsed:.2f}s (would take ~1.5s sequentially)")
    print("  Thanks to non-blocking I/O, they ran concurrently!")

    # INCORRECT example (commented out to show what NOT to do):
    # agent = CodeAgent(model=model, tools=[])
    # results = await asyncio.gather(*[agent.arun(task) for task in tasks])  # BAD!
    # This would cause memory corruption - don't reuse agent instances!


# Example 3: Async streaming
async def example_async_streaming():
    """
    Example showing how to use async streaming to get results as they come in.
    """
    print("\n=== Example 3: Async Streaming ===\n")

    # Create a model
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    # Create an agent with async tools
    agent = CodeAgent(model=model, tools=[HumanApprovalTool()])

    # Run with streaming enabled to see steps as they happen
    print("Running agent with streaming enabled...\n")
    async for step in agent.arun("Get approval to send email", stream=True):
        print(f"  [Step] {step}")

    print("\n✓ Streaming allows you to see and process results incrementally!")


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
    print("\nThese examples demonstrate:")
    print("  • Async tools for human-in-the-loop workflows")
    print("  • Concurrent agent execution with non-blocking I/O")
    print("  • Transparent async handling (no 'await' needed in generated code)")
    print("  • Async streaming for incremental results")
    print("=" * 70)

    # Uncomment to run examples with actual API calls (requires API key)
    # await example_simple_async()
    # await example_concurrent_agents()
    # await example_async_streaming()
    # await example_async_models()
    # await example_performance_comparison()

    print("\nTo run these examples:")
    print("  1. Set your API key: export ANTHROPIC_API_KEY=your_key")
    print("  2. Uncomment the example calls in main()")
    print("  3. Run: python async_agent_example.py")

    print("\n" + "=" * 70)
    print("Examples ready to run!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
