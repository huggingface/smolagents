"""
Example: Using a reasoning model from Ollama with smolagents.

Reasoning models (e.g. qwq) produce chain-of-thought in a separate
``reasoning`` field alongside the final answer. smolagents captures this via
the ``preserve_reasoning`` flag on the agent.

QwQ requires its reasoning to be included in the conversation history sent to
each subsequent turn. Omitting it degrades answer quality. This contrasts with
DeepSeek-R1 (via the DeepSeek API), which returns a 400 error if reasoning is
re-sent. Always check your model's documentation.

Prerequisites
-------------
1. Install and start Ollama: https://ollama.com/
2. Pull the QwQ reasoning model::

       ollama pull qwq   # ~20 GB; use qwq:32b-preview-q4_K_M for a smaller quant

3. Install smolagents::

       pip install smolagents

Usage
-----
Run the script directly::

    python examples/ollama_reasoning_model.py

The agent will print each reasoning step and the final answer.
``preserve_reasoning=True`` ensures the chain-of-thought is kept in the
conversation history, which QwQ needs to stay coherent across turns.
"""

from smolagents import CodeAgent, LiteLLMModel, tool


# ---------------------------------------------------------------------------
# Model — Ollama exposes an OpenAI-compatible endpoint on port 11434.
# LiteLLMModel routes to it via the "ollama_chat/" prefix.
# ---------------------------------------------------------------------------
model = LiteLLMModel(
    model_id="ollama_chat/qwen3.5:4b",
    api_base="http://localhost:11434",
    api_key="ollama",
    # QwQ generates very long chains-of-thought;
    num_ctx=32768,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The sum of a and b.
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The product of a and b.
    """
    return a * b


@tool
def is_prime(n: int) -> bool:
    """Check whether a positive integer is a prime number.

    Args:
        n: The integer to test (must be >= 2).

    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
# preserve_reasoning=True: qwen3:8b expects its own reasoning to be present in the
# conversation history on every subsequent turn. Without it the model loses
# context and answer quality drops noticeably.
# Use preserve_reasoning=False for providers like DeepSeek that return a 400
# error when reasoning is re-sent.
agent = CodeAgent(
    tools=[add, multiply, is_prime],
    model=model,
    preserve_reasoning=True,
    verbosity_level=2,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = agent.run("Find the smallest prime number greater than 100, then multiply it by 7 and add 42.")
    print("\n=== Final answer ===")
    print(result)
