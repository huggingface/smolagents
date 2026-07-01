"""Persistent cross-session memory for smolagents using Dakera.

Demonstrates how to use step_callbacks to persist agent observations to
a self-hosted Dakera memory server, enabling continuity across restarts.

Closes: https://github.com/huggingface/smolagents/issues/1216
Relates to: https://github.com/huggingface/smolagents/issues/945

Setup:
    docker run -d -p 3300:3300 -e DAKERA_API_KEY=demo ghcr.io/dakera-ai/dakera:latest
    pip install smolagents dakera

Usage:
    DAKERA_URL=http://localhost:3300 DAKERA_API_KEY=demo python examples/persistent_memory_with_dakera.py
"""

import os

from smolagents import CodeAgent, ActionStep, HfApiModel, DuckDuckGoSearchTool
from dakera import DakeraClient  # pip install dakera


# ---------------------------------------------------------------------------
# Memory client — connects to local or remote Dakera server
# ---------------------------------------------------------------------------
client = DakeraClient(
    base_url=os.environ.get("DAKERA_URL", "http://localhost:3300"),
    api_key=os.environ.get("DAKERA_API_KEY", "demo"),
)
SESSION_ID = "smolagents-research"
AGENT_ID = "smolagent"


def persist_step(step: ActionStep, agent) -> None:
    """Store each completed step observation to Dakera after it finishes.

    Called by the agent runtime after every ActionStep. Observations are the
    raw output from tool calls — exactly what we want to remember across sessions.
    """
    if not step.observations:
        return
    client.store(
        content=step.observations,
        session_id=SESSION_ID,
        agent_id=AGENT_ID,
        metadata={"step_number": step.step_number},
    )


def load_prior_context(query: str) -> str:
    """Recall relevant memories from prior sessions for the given query.

    Uses Dakera's decay-weighted semantic search: recent and frequently-
    accessed memories rank higher than stale ones.
    """
    memories = client.recall(
        query=query,
        session_id=SESSION_ID,
        agent_id=AGENT_ID,
        top_k=5,
    )
    if not memories:
        return ""
    lines = "\n".join(f"- {m.content}" for m in memories)
    return f"Prior context from previous sessions:\n{lines}\n"


# ---------------------------------------------------------------------------
# Session 1: research and store
# ---------------------------------------------------------------------------
task = "Research the top 3 use cases for autonomous AI agents in enterprise settings in 2025."
prior = load_prior_context(task)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=HfApiModel(),
    step_callbacks=[persist_step],  # persist each completed step to Dakera
)

print("=== Session 1: Research ===")
if prior:
    print(f"Loaded prior context from Dakera:\n{prior}\n")

result = agent.run(f"{prior}\n{task}" if prior else task)
print(f"Result: {result[:400]}...\n")
print("✓ Observations stored to Dakera memory\n")

# ---------------------------------------------------------------------------
# Session 2: new agent — recalls what Session 1 stored from Dakera.
# To experience cross-session memory, re-run this script. The second agent
# will recall Session 1's findings even if run in a completely new process.
# ---------------------------------------------------------------------------
followup = "Based on prior research on enterprise AI agents, what are the biggest adoption risks?"
prior2 = load_prior_context(followup)

agent2 = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=HfApiModel(),
    step_callbacks=[persist_step],
)

print("=== Session 2: Recall + Continue ===")
if prior2:
    print(f"Recalled from Dakera:\n{prior2}\n")
else:
    print("No prior memories found — run the script again after Session 1 to see cross-session recall.\n")

result2 = agent2.run(f"{prior2}\n{followup}" if prior2 else followup)
print(f"Result: {result2[:400]}...")
