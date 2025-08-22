#!/usr/bin/env python
# Example run: python examples/decentralized_smolagents_benchmark/decentralized_agent.py   --model-type LiteLLMModel   --model-id gpt-4o   --provider openai   "What is the half of the speed of a Leopard?"
"""Main entry point for decentralized agent team execution."""

import argparse
import json
import logging
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from scripts.agents import create_team
from scripts.message_store import MessageStore


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_key_facts(text: str) -> set:
    """Extract key facts from text for comparison."""
    # Remove common words and extract meaningful terms
    words = normalize_text(text).split()
    # Filter out very common words but keep important ones
    stop_words = {
        "the",
        "is",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
    }
    key_facts = set(word for word in words if len(word) > 2 and word not in stop_words)
    return key_facts


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts based on common key facts."""
    facts1 = extract_key_facts(text1)
    facts2 = extract_key_facts(text2)

    if not facts1 and not facts2:
        return 1.0  # Both empty
    if not facts1 or not facts2:
        return 0.0  # One empty, one not

    intersection = facts1.intersection(facts2)
    union = facts1.union(facts2)

    return len(intersection) / len(union) if union else 0.0


def is_similar_answer(answer1: str, answer2: str, threshold: float = 0.5) -> bool:
    """Check if two answers are similar enough to be considered the same."""
    return calculate_similarity(answer1, answer2) >= threshold


def check_semantic_consensus(answers: List[str], min_agreement: int = 3) -> Optional[str]:
    """Check if there's semantic consensus among answers."""
    if len(answers) < min_agreement:
        return None

    # Group similar answers
    answer_groups = []
    for answer in answers:
        # Find which group this answer belongs to
        placed = False
        for group in answer_groups:
            if is_similar_answer(answer, group[0], threshold=0.4):  # Lower threshold for grouping
                group.append(answer)
                placed = True
                break

        if not placed:
            answer_groups.append([answer])

    # Find the largest group that meets the minimum requirement
    largest_group = max(answer_groups, key=len) if answer_groups else []

    if len(largest_group) >= min_agreement:
        # Return the most detailed answer from the consensus group
        return max(largest_group, key=len)

    return None


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
    agent_names = ["CodeAgent", "WebSearchAgent", "DeepResearchAgent", "DocumentReaderAgent"]
    message_store = MessageStore(run_id, agent_names=agent_names)
    agents = create_team(
        message_store=message_store, model_type=args.model_type, model_id=args.model_id, provider=args.provider
    )

    # Post initial task
    initial_msg = message_store.append_message(
        sender="system", content=args.question, thread_id="main", msg_type="task"
    )
    logging.info(json.dumps({"event": "task_posted", "message_id": initial_msg.get("id")}))

    # Start agent threads
    threads = []
    consensus_event = threading.Event()
    final_answer = [None]  # Use list to allow modification in thread
    timeout = 30  # Extended timeout for complex questions

    def check_consensus():
        """Monitor for consensus and stop other agents when reached."""
        N = len(agents)  # Should be 4 agents
        required_yes = (N // 2) + 1  # Require majority agreement

        logging.info(
            json.dumps(
                {
                    "event": "consensus_monitor_started",
                    "total_agents": N,
                    "required_votes": required_yes,
                    "timeout": timeout,
                }
            )
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.5)  # Check every 500ms

            # Check if monitor_and_finalize has found a final answer
            if final_answer[0] is not None:
                logging.info(json.dumps({"event": "consensus_reached", "elapsed_time": time.time() - start_time}))
                consensus_event.set()
                return

        # Timeout reached
        logging.info(json.dumps({"event": "consensus_timeout_reached", "elapsed_time": timeout}))
        consensus_event.set()  # Start consensus monitor thread

    consensus_thread = threading.Thread(target=check_consensus)
    consensus_thread.start()

    # Start background monitoring (only one monitor thread needed)
    monitor_thread = threading.Thread(target=lambda: monitor_and_finalize(message_store, final_answer))
    monitor_thread.daemon = True
    monitor_thread.start()

    # Start agent threads
    for agent in agents:

        def run_agent(agent_instance):
            """Run agent with error handling."""
            logging.info(json.dumps({"event": "agent_started", "agent": agent_instance.config.name}))
            try:
                step_count = 0
                while not consensus_event.is_set():
                    step_count += 1
                    if step_count == 1:  # Log first step
                        logging.info(json.dumps({"event": "agent_first_step", "agent": agent_instance.config.name}))
                    if not agent_instance.run_step():
                        logging.info(
                            json.dumps(
                                {"event": "agent_stopped", "agent": agent_instance.config.name, "steps": step_count}
                            )
                        )
                        break  # Agent decided to stop
                    time.sleep(0.1)  # Brief pause between steps
            except Exception as e:
                logging.error(
                    json.dumps({"event": "agent_error", "agent": agent_instance.config.name, "error": str(e)})
                )

        thread = threading.Thread(target=run_agent, args=(agent,))
        thread.daemon = True  # Allow main program to exit
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


def monitor_and_finalize(message_store: MessageStore, final_answer_holder: list):
    """Watch all open polls and detect consensus through final answers."""
    seen_polls: set[str] = set()
    last_ts = None
    agent_final_answers = {}  # Track final answers by agent

    logging.info(json.dumps({"event": "monitor_started"}))

    while True:
        try:
            # Get new messages since last check
            all_messages = list(message_store._iter_messages())

            for msg in all_messages:
                # Ensure msg is a dictionary before processing
                if not isinstance(msg, dict):
                    continue  # Skip non-dict messages silently

                if last_ts and msg.get("timestamp", "") <= last_ts:
                    continue

                sender = msg.get("sender", "")
                msg_type = msg.get("type", "message")
                content = msg.get("content", {})

                # Ensure content is processed safely
                if isinstance(content, str):
                    # Skip string-only messages for poll processing
                    continue

                # Track final answer tool usage by agents
                if msg_type == "tool_call" and isinstance(content, dict):
                    tool_name = content.get("tool_name", "")
                    if tool_name == "final_answer":
                        tool_args = content.get("arguments", {})
                        if isinstance(tool_args, dict):
                            answer = tool_args.get("answer", "")
                            if answer and sender:
                                agent_final_answers[sender] = answer.strip()
                                logging.info(
                                    json.dumps(
                                        {
                                            "event": "final_answer_recorded",
                                            "agent": sender,
                                            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
                                        }
                                    )
                                )

                # Check for consensus through final answers after each message
                if len(agent_final_answers) >= 2:  # Lower threshold for debugging
                    # Advanced consensus detection
                    answers = list(agent_final_answers.values())

                    logging.info(
                        json.dumps(
                            {
                                "event": "checking_consensus",
                                "agents_with_answers": len(agent_final_answers),
                                "agents": list(agent_final_answers.keys()),
                                "answers_preview": [a[:50] for a in answers],
                            }
                        )
                    )

                    # Method 1: Check for semantic similarity
                    consensus_answer = check_semantic_consensus(answers, min_agreement=2)  # Lower for debugging

                    logging.info(
                        json.dumps(
                            {
                                "event": "consensus_check_result",
                                "consensus_found": consensus_answer is not None,
                                "consensus_answer_preview": consensus_answer[:50] if consensus_answer else None,
                            }
                        )
                    )

                    if consensus_answer:
                        logging.info(
                            json.dumps(
                                {
                                    "event": "consensus_through_final_answers",
                                    "agreeing_agents": len(
                                        [a for a in answers if is_similar_answer(a, consensus_answer)]
                                    ),
                                    "total_agents": len(agent_final_answers),
                                    "answer": consensus_answer,
                                }
                            )
                        )

                        # Set the final answer for the main thread
                        final_answer_holder[0] = consensus_answer
                        print(json.dumps({"answer": consensus_answer}, ensure_ascii=False))
                        return

                # Check for new polls and handle poll finalization
                if isinstance(content, dict) and content.get("type") == "poll" and content.get("status") == "open":
                    poll_id = content.get("poll_id")
                    if poll_id and poll_id not in seen_polls:
                        seen_polls.add(poll_id)
                        logging.info(
                            json.dumps(
                                {
                                    "event": "new_poll_detected",
                                    "poll_id": poll_id,
                                    "threshold": content.get("threshold", "unknown"),
                                }
                            )
                        )

                        # Check if this poll is ready to finalize (has enough YES votes or should be deleted)
                        final_answer_msg = message_store.finalize_poll_if_ready(poll_id)
                        if final_answer_msg:
                            # Check if poll was deleted
                            if final_answer_msg.get("deleted"):
                                logging.info(
                                    json.dumps(
                                        {
                                            "event": "poll_deleted",
                                            "poll_id": poll_id,
                                            "reason": final_answer_msg.get("reason", "Unknown"),
                                        }
                                    )
                                )
                                continue

                            # Poll passed - extract final answer
                            answer_content = final_answer_msg.get("content", {})
                            answer = answer_content.get("answer", "")
                            tally = answer_content.get("tally", {})

                            logging.info(
                                json.dumps(
                                    {
                                        "event": "consensus_achieved_via_poll",
                                        "poll_id": poll_id,
                                        "final_tally": tally,
                                        "answer": answer,
                                    }
                                )
                            )

                            # Set the final answer for the main thread
                            final_answer_holder[0] = answer
                            print(json.dumps({"answer": answer}, ensure_ascii=False))
                            return

                # Check for already finalized answers
                elif content.get("type") == "final_answer":
                    answer = content.get("answer", "")
                    tally = content.get("tally", {})

                    logging.info(json.dumps({"event": "final_answer_found", "tally": tally, "answer": answer}))

                    final_answer_holder[0] = answer
                    print(json.dumps({"answer": answer}, ensure_ascii=False))
                    return

            if all_messages:
                last_ts = all_messages[-1].get("timestamp")

            # Check all active polls periodically to see if they should be finalized or deleted
            active_polls = message_store.get_active_polls()
            for poll in active_polls:
                poll_id = poll.get("poll_id")
                if poll_id:
                    final_answer_msg = message_store.finalize_poll_if_ready(poll_id)
                    if final_answer_msg:
                        # Check if poll was deleted
                        if final_answer_msg.get("deleted"):
                            logging.info(
                                json.dumps(
                                    {
                                        "event": "poll_deleted_on_check",
                                        "poll_id": poll_id,
                                        "reason": final_answer_msg.get("reason", "Unknown"),
                                    }
                                )
                            )
                            continue

                        # Poll passed - extract final answer
                        answer_content = final_answer_msg.get("content", {})
                        answer = answer_content.get("answer", "")
                        tally = answer_content.get("tally", {})

                        logging.info(
                            json.dumps(
                                {
                                    "event": "consensus_achieved_via_periodic_check",
                                    "poll_id": poll_id,
                                    "final_tally": tally,
                                    "answer": answer,
                                }
                            )
                        )

                        # Set the final answer for the main thread
                        final_answer_holder[0] = answer
                        print(json.dumps({"answer": answer}, ensure_ascii=False))
                        return

            time.sleep(0.5)  # Check every 500ms

        except Exception as e:
            logging.error(json.dumps({"event": "monitor_error", "error": str(e)}))
            time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, help="Model type to use")
    parser.add_argument("--model-id", required=True, help="Model ID to use")
    parser.add_argument("--provider", help="Model provider")
    parser.add_argument("question", help="Question to answer")

    args = parser.parse_args()
    sys.exit(main(args))
