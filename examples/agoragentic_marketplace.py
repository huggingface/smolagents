"""
Agoragentic Marketplace — Agent-to-Agent Capability Router
============================================================

This example shows how to give any smolagents CodeAgent access to 200+
AI capabilities through the Agoragentic marketplace. The agent can:

  1. Search for capabilities by query, category, or price
  2. Route tasks to the best provider automatically via execute()
  3. Preview providers before committing (dry-run match)
  4. Invoke a specific capability by ID

Payment is automatic in USDC on Base L2.

Setup:
    pip install smolagents requests
    export AGORAGENTIC_API_KEY="amk_your_key"  # Get one free at https://agoragentic.com/api/quickstart

Run:
    python agoragentic_marketplace.py
"""

import json
import os

from smolagents import CodeAgent, HfApiModel, Tool


# ─── Tool 1: execute() — Route any task to the best provider ──

class AgoragenticExecuteTool(Tool):
    """One-shot task routing. Describe what you need, the marketplace
    finds the highest-ranked provider and invokes it automatically."""

    name = "agoragentic_execute"
    description = (
        "Route a task to the best AI provider on the Agoragentic marketplace. "
        "Describe what you need in plain English. The router scores providers "
        "by trust, price, latency, and capability, then invokes the winner. "
        "Payment is automatic in USDC on Base L2."
    )
    inputs = {
        "task": {
            "type": "string",
            "description": "Plain-English task description (e.g., 'summarize this text')",
        },
        "input_json": {
            "type": "string",
            "description": "JSON string with the input payload for the provider",
            "nullable": True,
        },
        "max_cost": {
            "type": "number",
            "description": "Max price in USDC you're willing to pay per call",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, task: str, input_json: str = "{}", max_cost: float = 1.0) -> str:
        import requests

        key = os.environ.get("AGORAGENTIC_API_KEY", "")
        if not key:
            return json.dumps({"error": "Set AGORAGENTIC_API_KEY environment variable"})

        resp = requests.post(
            "https://agoragentic.com/api/execute",
            json={
                "task": task,
                "input": json.loads(input_json) if input_json else {},
                "constraints": {"max_cost": max_cost},
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            timeout=60,
        )
        data = resp.json()
        if resp.status_code == 200:
            return json.dumps(
                {
                    "status": data.get("status"),
                    "provider": data.get("provider", {}).get("name"),
                    "output": data.get("output"),
                    "cost_usdc": data.get("cost"),
                },
                indent=2,
            )
        return json.dumps({"error": data.get("error"), "message": data.get("message")})


# ─── Tool 2: search() — Browse the marketplace ───────────────

class AgoragenticSearchTool(Tool):
    """Browse 200+ capabilities across 20+ categories."""

    name = "agoragentic_search"
    description = (
        "Search the Agoragentic marketplace for AI capabilities. "
        "Filter by query, category (ai-services, data, devtools, etc.), "
        "or max price in USDC. Returns names, prices, and IDs."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Search term (e.g., 'text summarization')",
            "nullable": True,
        },
        "category": {
            "type": "string",
            "description": "Category filter",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str = "", category: str = "") -> str:
        import requests

        params = {"limit": 10, "status": "active"}
        if query:
            params["search"] = query
        if category:
            params["category"] = category

        key = os.environ.get("AGORAGENTIC_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        resp = requests.get(
            "https://agoragentic.com/api/capabilities",
            params=params,
            headers=headers,
            timeout=15,
        )
        data = resp.json()
        caps = data if isinstance(data, list) else data.get("capabilities", [])
        results = [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "price_usdc": c.get("price_per_unit"),
                "category": c.get("category"),
            }
            for c in caps[:10]
        ]
        return json.dumps({"capabilities": results}, indent=2)


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = CodeAgent(
        tools=[AgoragenticExecuteTool(), AgoragenticSearchTool()],
        model=HfApiModel(),
    )

    # The agent will search the marketplace, find the best provider,
    # and execute the task — all automatically.
    result = agent.run(
        "Search the Agoragentic marketplace for text summarization capabilities, "
        "then use the best one to summarize this: "
        "'Agoragentic is an API-first marketplace where AI agents discover, "
        "invoke, and pay for services from other agents using USDC on Base L2. "
        "It supports 200+ capabilities across 20+ categories with automatic "
        "trust scoring and verification.'"
    )
    print(result)
