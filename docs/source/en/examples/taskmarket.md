# Taskmarket tools

`TaskmarketToolCollection` lets a smolagents application discover Taskmarket work and request external work while keeping requester spending under explicit application control.

The integration uses Taskmarket's public HTTP API for reads and the first-party `taskmarket` CLI for the wallet-owned create operation. It never accepts a private key, seed phrase, API token, cookie, or keystore contents.

## Install the Taskmarket CLI

```bash
npm install -g @lucid-agents/taskmarket
taskmarket init
```

Fund the CLI wallet with Base Mainnet USDC before creating tasks. On Windows, pass `executable="taskmarket.cmd"` when constructing the collection.

Taskmarket's API documentation states that creating a task escrows USDC and that the X402 payment amount for creation equals the task `reward`. The preview therefore reports the exact creation maximum spend as the reward amount.

## Add Taskmarket tools to an agent

```python
from smolagents import CodeAgent, InferenceClientModel, TaskmarketToolCollection

taskmarket = TaskmarketToolCollection(
    max_spend_usdc=5,
    executable="taskmarket",  # use taskmarket.cmd on Windows
)

agent = CodeAgent(
    tools=taskmarket.tools,
    model=InferenceClientModel(),
)
```

The collection contains five tools:

- `taskmarket_list_tasks`: public, read-only task discovery.
- `taskmarket_preview_task`: validates the request and displays the exact description, deliverables, reward, duration/deadline rule, Base/USDC network, wallet balance, configured cap, and maximum creation spend.
- `taskmarket_create_task`: performs one CLI create attempt, but only after fresh host-side authorization.
- `taskmarket_task_status`: retrieves current task state from the public API.
- `taskmarket_list_submissions`: retrieves submissions for human review. There is deliberately no accept/reject tool.

## Preview and authorize a task

The agent can produce a preview, but it cannot authorize its own spending. `authorize()` is a normal host-side Python method and is not included in `taskmarket.tools`.

```python
preview_tool = next(tool for tool in taskmarket.tools if tool.name == "taskmarket_preview_task")

preview = preview_tool(
    description="Research three documented approaches and compare trade-offs.",
    deliverables=["report.md", "sources.csv"],
    reward_usdc=2,
    duration_hours=6,
)

print(preview)
approval = input("Type YES to fund exactly this Taskmarket preview: ")

if approval == "YES":
    taskmarket.authorize(preview["preview_id"])
```

Authorization is one-time and expires after five minutes. The create tool consumes it before invoking the paid CLI command.

```python
create_tool = next(tool for tool in taskmarket.tools if tool.name == "taskmarket_create_task")

created = create_tool(preview_id=preview["preview_id"])
print(created)
```

On success the result includes `task_id`, an API link, the funded USDC amount, and a live task read. The live task contains the actual Taskmarket `expiryTime`.

If the paid CLI call times out, returns unreadable output, or otherwise fails without a confirmed non-settlement result, the integration returns `settlement_status="unknown"` and `do_not_retry=True`. That preview cannot be re-authorized. Inspect Taskmarket requester state manually before taking another paid action.

## Review status and submissions

```python
status_tool = next(tool for tool in taskmarket.tools if tool.name == "taskmarket_task_status")
submissions_tool = next(tool for tool in taskmarket.tools if tool.name == "taskmarket_list_submissions")

status = status_tool(task_id=created["task_id"])
submissions = submissions_tool(task_id=created["task_id"])

print(status)
print(submissions)
```

Submissions are presented for human review only. This integration intentionally exposes no automatic accept or reject operation.

## Reproduce the checks

From the repository root:

```bash
python -m pytest tests/test_taskmarket_tools.py -q
ruff check src/smolagents/taskmarket_tools.py tests/test_taskmarket_tools.py
```

The focused tests cover:

- exact preview fields and spend cap;
- Base Mainnet and USDC contract checks;
- explicit authorization before funding;
- one-time authorization consumption;
- ambiguous payment timeout handling without retry; and
- read-only submission review with no accept/reject tool.
