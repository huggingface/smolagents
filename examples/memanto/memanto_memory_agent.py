"""
Demo: smolagents CodeAgent with Memanto long-term memory.

Prerequisites:
  1. pip install memanto httpx python-dotenv
  2. memanto agent create smolagents-demo   # or set MEMANTO_AGENT_ID
  3. memanto serve                          # default http://localhost:8000

Run from repo root:
  python examples/memanto/memanto_memory_agent.py
  python examples/memanto/memanto_memory_agent.py --seed   # force demo seed memory
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow local imports when run from repo root or this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memanto_client import MemantoClient
from memanto_tools import create_memanto_tools

from smolagents import CodeAgent, InferenceClientModel

DEMO_PREFERENCE = (
    "The user prefers concise, well-structured responses with markdown formatting."
)
DEFAULT_TASK = (
    "What response format do I prefer? Check long-term memory before answering."
)


def create_model():
    """Pick an LLM backend from environment variables.

    Priority:
    1. OpenAI if OPENAI_API_KEY is set
    2. LiteLLM if LITELLM_MODEL_ID or GROQ_API_KEY is set
    3. Hugging Face InferenceClientModel (SMOLAGENTS_MODEL_ID / SMOLAGENTS_MODEL_PROVIDER)
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        from smolagents import OpenAIModel

        return OpenAIModel(
            model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini"),
            api_key=openai_api_key,
        )

    litellm_model_id = os.getenv("LITELLM_MODEL_ID")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if litellm_model_id or groq_api_key:
        from smolagents import LiteLLMModel

        return LiteLLMModel(
            model_id=litellm_model_id or "groq/openai/gpt-oss-120b",
            api_key=groq_api_key or os.getenv("LITELLM_API_KEY"),
        )

    model_id = os.getenv(
        "SMOLAGENTS_MODEL_ID",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    provider = os.getenv("SMOLAGENTS_MODEL_PROVIDER")
    if provider:
        return InferenceClientModel(model_id=model_id, provider=provider)
    return InferenceClientModel(model_id=model_id)


def seed_demo_memory(client: MemantoClient, *, force: bool = False) -> bool:
    """Seed the demo preference once. Returns True if a memory was stored."""
    if not force and client.has_memories():
        print("Skipping demo seed — agent already has memories. Use --seed to force.")
        return False

    client.remember(
        DEMO_PREFERENCE,
        memory_type="preference",
        title="Response style preference",
    )
    print("Seeded demo preference memory.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CodeAgent with Memanto long-term memory.")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Force storing the demo preference memory (default: only seed an empty namespace).",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task for the agent to run.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    agent_id = os.getenv("MEMANTO_AGENT_ID", "smolagents-demo")
    memanto_url = os.getenv("MEMANTO_URL", "http://localhost:8000")

    with MemantoClient(base_url=memanto_url, agent_id=agent_id) as client:
        seed_demo_memory(client, force=args.seed)

        model = create_model()
        print(f"Using model: {getattr(model, 'model_id', model)}")

        tools = create_memanto_tools(client)
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=6,
            verbosity_level=2,
        )

        # reset=True clears in-session AgentMemory; Memanto persists across runs.
        result = agent.run(args.task, reset=True)

    print("\nFinal output:")
    print(result)


if __name__ == "__main__":
    main()
