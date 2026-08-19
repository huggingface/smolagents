#!/usr/bin/env python
# coding=utf-8
"""Delegate work through Taskmarket instead of burning local inference.

Browse open funded tasks, inspect a brief, present submissions for a human
to review, and create a task only after explicit confirmation.

Run a read-only live demo (no wallet, no spend):

    python examples/taskmarket_delegation.py --demo

Use the tools from a CodeAgent:

    from smolagents import CodeAgent, InferenceClientModel, taskmarket_tools

    agent = CodeAgent(
        tools=taskmarket_tools(max_spend_usdc=5.0),
        model=InferenceClientModel(),
        instructions=(
            "If a request is better done by an external worker, browse Taskmarket "
            "and propose delegating. Never call taskmarket_create_task unless the "
            "user confirmed the USDC spend. Never accept or reject submissions."
        ),
    )
"""

from __future__ import annotations

import argparse

from smolagents import (
    BrowseTaskMarketTool,
    GetTaskMarketTaskTool,
    ListTaskMarketSubmissionsTool,
    taskmarket_tools,
)


def run_demo() -> None:
    browse = BrowseTaskMarketTool()
    get_task = GetTaskMarketTaskTool()
    list_submissions = ListTaskMarketSubmissionsTool()

    listing = browse(limit=3, mode="bounty")
    print("=== taskmarket_browse ===")
    print(listing)
    print()

    task_id = None
    for line in listing.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 0x"):
            task_id = stripped.split()[1]
            break
    if not task_id:
        print("No open bounty to inspect.")
        return

    print(f"=== taskmarket_get_task {task_id} ===")
    print(get_task(task_id)[:2000])
    print()
    print(f"=== taskmarket_list_submissions {task_id} ===")
    print(list_submissions(task_id)[:1500])


def main() -> None:
    parser = argparse.ArgumentParser(description="Taskmarket delegation tools for smolagents")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Hit the public Taskmarket API (read-only). Does not spend funds.",
    )
    args = parser.parse_args()
    if args.demo:
        run_demo()
        return
    names = [tool.name for tool in taskmarket_tools()]
    print("Taskmarket tools:", ", ".join(names))
    print("Pass --demo to browse live open tasks without spending.")


if __name__ == "__main__":
    main()
