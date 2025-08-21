#!/usr/bin/env python
# Example run: python examples/decentralized_smolagents_benchmark/decentralized_agent.py   --model-type LiteLLMModel   --model-id gpt-4o   --provider openai   "What is the half of the speed of a Leopard?"
"""Main entry point for decentralized agent team execution."""

import argparse
import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path

from scripts.agents import create_team
from scripts.message_store import MessageStore


def setup_logging(run_dir: Path) -> None:
    """Setup JSON logging to file."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"timestamp":"%(asctime)s", "level":"%(levelname)s", "message":%(message)s}',
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler(sys.stdout)],
    )


def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    # Create run directory
    run_id = str(uuid.uuid4())
    script_dir = Path(__file__).parent
    run_dir = script_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    # Setup logging
    setup_logging(run_dir)
    logging.info(json.dumps({"event": "run_started", "run_id": run_id, "args": vars(args)}))

    # Create message store and team
    message_store = MessageStore(run_id)
    agents = create_team(
        message_store=message_store, model_type=args.model_type, model_id=args.model_id, provider=args.provider
    )

    # Post initial task
    message_store.post_message({"sender": "system", "content": args.question, "thread_id": "main"})

    # Start agent threads
    threads = []
    consensus_event = threading.Event()
    final_answer: list[str | None] = [None]  # Use list to allow modification in thread
    timeout = 30  # Timeout in seconds for simple questions

    def check_consensus():
        """Monitor for emergent consensus through decentralized pattern detection."""
        from scripts.consensus_protocol import ConsensusProtocol

        protocol = ConsensusProtocol()

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Get all relevant messages
            messages = message_store.get_messages(agent_id="system")

            # Analyze for emerging consensus
            has_consensus, consensus_value, confidence = protocol.analyze_conversation(messages)

            if has_consensus:
                final_answer[0] = consensus_value
                consensus_event.set()
                logging.info(
                    json.dumps({"event": "consensus_reached", "confidence": confidence, "value": consensus_value})
                )
                return

            time.sleep(0.1)  # Avoid tight polling

        # No strong consensus emerged within timeout
        # Do one final check with a lower confidence threshold
        protocol.confidence_threshold = 0.5  # Lower threshold for final check
        has_consensus, consensus_value, confidence = protocol.analyze_conversation(messages)

        if has_consensus:
            final_answer[0] = consensus_value
            logging.info(
                json.dumps({"event": "weak_consensus_reached", "confidence": confidence, "value": consensus_value})
            )
        consensus_event.set()

    # Start consensus monitor thread
    consensus_thread = threading.Thread(target=check_consensus)
    consensus_thread.start()

    # Start agent threads
    for agent in agents:

        def run_agent():
            try:
                while not consensus_event.is_set():
                    agent.run_step()
            except Exception as e:
                logging.error(f"Agent error: {e}")

        thread = threading.Thread(target=run_agent)
        thread.start()
        threads.append(thread)

    # Wait for consensus or timeout
    consensus_thread.join()
    consensus_event.set()  # Signal all agents to stop

    # Brief wait for agents to finish current step
    for thread in threads:
        thread.join(timeout=0.5)

    if final_answer[0]:  # Check the actual answer inside the list
        answer = final_answer[0]
        print(json.dumps({"answer": answer}))  # Clean JSON output
        logging.info(json.dumps({"event": "final_answer", "answer": answer}))
        return 0
    else:
        print(json.dumps({"error": "No consensus reached on final answer"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, help="Model type to use")
    parser.add_argument("--model-id", required=True, help="Model ID to use")
    parser.add_argument("--provider", help="Model provider")
    parser.add_argument("question", help="Question to answer")

    args = parser.parse_args()
    sys.exit(main(args))
