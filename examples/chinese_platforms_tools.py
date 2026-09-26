"""
Chinese platforms tools example: use BilibiliHotTool to find trending Chinese videos.

This example demonstrates how to wire `BilibiliHotTool` (one of smolagents' built-in
web-style tools) into a `CodeAgent` and let the LLM call it autonomously.

Prerequisites
-------------
1. Install smolagents from source (this repo):

       pip install -e .

2. Set your Hugging Face token so the default `InferenceClientModel` can reach the
   Inference API. See https://huggingface.co/settings/tokens for instructions.

       export HF_TOKEN="hf_xxx"

3. (Optional) Swap in a different model by replacing `InferenceClientModel()` with
   any other smolagents-supported model class, e.g.:

       from smolagents import LiteLLMModel
       model = LiteLLMModel(model_id="gpt-4o-mini")

Run
---
    python examples/chinese_platforms_tools.py
"""

from smolagents import CodeAgent, InferenceClientModel, WebSearchTool
from smolagents.default_tools import BilibiliHotTool


model = InferenceClientModel()

agent = CodeAgent(
    tools=[BilibiliHotTool(), WebSearchTool()],
    model=model,
    stream_outputs=True,
    return_full_result=True,
)

# Ask the agent to find trending AI-related videos on Bilibili. The agent should:
#   1. call `bilibili_hot(limit=10)` to fetch the raw markdown list,
#   2. read the titles to spot AI-related entries,
#   3. (optionally) call `web_search` to gather context for the most interesting one,
#   4. summarize the findings.
result = agent.run(
    "Find the top 10 trending videos on Bilibili right now and tell me which one looks "
    "most interesting to an AI researcher. Briefly explain why."
)

print("\n--- Agent's final answer ---")
print(result)
