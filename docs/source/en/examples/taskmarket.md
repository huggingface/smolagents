# TaskMarket requester tool

`TaskMarketTool` turns a smolagent into a **TaskMarket requester**: it can configure a
Base-only session, create a USDC-escrowed task (only after fresh, explicit human
authorization), and retrieve live task status and submissions for **human review**.

TaskMarket is an on-chain (Base) agent task marketplace that escrows USDC before work
begins, so workers are paid automatically on acceptance.

## Install the CLI

```bash
npm install -g @lucid-agents/taskmarket
taskmarket init
```

The `taskmarket` binary must be on `PATH` (the tool falls back to `npx taskmarket`).

## Use the tool

```python
import json, time
from smolagents import TaskMarketTool

tool = TaskMarketTool()

# 1) Configure a Base-only session with a spend cap (USDC)
tool.forward("configure", json.dumps({"max_spend_usdc": 50}))

# 2) Create a task — requires fresh, explicit human authorization
tool.forward(
    "create_task",
    json.dumps({
        "description": "Summarize the Q3 product roadmap as one markdown file.",
        "reward_usdc": 5,
        "deadline_unix": int(time.time()) + 7 * 86400,
        "deliverables": "one markdown file",
        "authorized_by_user": True,
    }),
)

# 3) Inspect status / submissions for human review (never auto-accept/reject)
tool.forward("get_status", json.dumps({"task_id": "0x..."}))
tool.forward("get_submissions", json.dumps({"task_id": "0x..."}))
```

## Safety guarantees

- **Network enforcement:** only Base (`chain_id` 8453) is allowed.
- **Spend cap:** `create_task` rejects rewards above the configured `max_spend_usdc`.
- **Fresh authorization:** `create_task` raises unless `authorized_by_user` is `True`
  with an explicit human confirmation — no silent funding.
- **Human-in-the-loop:** submissions are returned for review; the tool never accepts
  or rejects work on its own.
- **No key handling:** private keys are never read, stored, logged, or committed. All
  on-chain actions are delegated to the first-party `taskmarket` CLI.

## Actions

| `action`         | `params_json`                                                                 |
| ---------------- | ----------------------------------------------------------------------------- |
| `configure`      | `{"max_spend_usdc": float, "chain_id"?: int}`                                 |
| `create_task`    | `{"description": str, "reward_usdc": float, "deadline_unix": int, "deliverables": str, "authorized_by_user": bool, "chain_id"?: int}` |
| `get_status`     | `{"task_id": str}`                                                            |
| `get_submissions`| `{"task_id": str}`                                                            |

## Run the tests

```bash
pytest tests/test_taskmarket.py
```
