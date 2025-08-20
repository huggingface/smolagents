#!/usr/bin/env python
"""Main entry point for decentralized agent team execution."""

import argparse
import json
import logging
import sys
import threading
import uuid
from pathlib import Path

from scripts.agents import create_team
from scripts.message_store import MessageStore


def setup_logging(run_dir: Path) -> None:
    """Setup JSON logging to file."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"timestamp":"%(asctime)s", "level":"%(levelname)s", "message":%(message)s}',
        handlers=[
            logging.FileHandler(run_dir / "run.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    # Create run directory
    run_id = str(uuid.uuid4())
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True)

    # Setup logging
    setup_logging(run_dir)
    logging.info(json.dumps({
        "event": "run_started",
        "run_id": run_id,
        "args": vars(args)
    }))

    # Create message store and team
    message_store = MessageStore(run_id)
    agents = create_team(
        message_store=message_store,
        model_type=args.model_type,
        model_id=args.model_id,
        provider=args.provider
    )

    # Post initial task
    message_store.post_message({
        "sender": "system",
        "content": args.question,
        "thread_id": "main"
    })

    # Start agent threads
    threads = []
    for agent in agents:
        thread = threading.Thread(target=agent.run)
        thread.start()
        threads.append(thread)

    # Wait for all agents
    for thread in threads:
        thread.join()

    # Check for final answer consensus
    messages = message_store.get_messages(agent_id="system")
    final_answer = None

    for msg in messages:
        if msg.get("type") == "final_answer_proposal":
            votes = message_store.count_votes(msg["id"])
            if votes["yes"] >= 3:
                final_answer = msg["content"]
                break

    if final_answer:
        print(f"FinalAnswer: {final_answer}")
        logging.info(json.dumps({
            "event": "final_answer",
            "answer": final_answer
        }))
        return 0
    else:
        print("No consensus reached on final answer", file=sys.stderr)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, help="Model type to use")
    parser.add_argument("--model-id", required=True, help="Model ID to use")
    parser.add_argument("--provider", help="Model provider")
    parser.add_argument("question", help="Question to answer")

    args = parser.parse_args()
    sys.exit(main(args))

