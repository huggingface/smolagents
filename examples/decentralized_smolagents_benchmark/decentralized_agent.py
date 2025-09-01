#!/usr/bin/env python
# Example run: python examples/decentralized_smolagents_benchmark/decentralized_agent.py --model-type LiteLLMModel --model-id gpt-4o --provider openai "What is the half of the speed of a Leopard?"
"""Entry point for decentralized agent team execution."""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from scripts.agents import DecentralizedAgents
from scripts.message_store import MessageStore


QUESTION_ADDON= """It is critical to respect the format of the answer when it is asked. """


# Langfuse instrumentation setup
try:
    from dotenv import load_dotenv

    load_dotenv()

    from langfuse import Langfuse
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor

    # Initialize Langfuse client
    langfuse_client = Langfuse()
    if langfuse_client.auth_check():
        print("✅ Langfuse client authenticated successfully")
        SmolagentsInstrumentor().instrument()
        print("✅ SmolagentsInstrumentor enabled")
    else:
        print("⚠️ Langfuse authentication failed - tracing disabled")
        langfuse_client = None
except ImportError as e:
    print(f"⚠️ Langfuse not available: {e}")
    langfuse_client = None
except Exception as e:
    print(f"⚠️ Langfuse setup error: {e}")
    langfuse_client = None


def setup_logging(run_dir: Path) -> None:
    """Setup JSON logging to file."""
    log_file = run_dir / "run.log"

    # Clear existing handlers to avoid duplication
    logger = logging.getLogger()
    logger.handlers.clear()

    # Create formatters
    json_formatter = logging.Formatter('{"timestamp":"%(asctime)s", "level":"%(levelname)s", "message":%(message)s}')
    #console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler with JSON format
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(json_formatter)

    # Console handler with readable format (optional, for debugging)
    # Uncomment the next 4 lines if you want console logging too
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(console_formatter)
    # logger.addHandler(console_handler)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> int:
    """Main entry point - simplified execution."""
    print(f"🚀 Starting decentralized agent team for: {args.question}")

    # Create message store with proper agent names for correct voting thresholds
    run_id = str(uuid.uuid4())[:8]  # Short run ID
    agent_names = ["CodeAgent", "WebSearchAgent", "DeepResearchAgent", "DocumentReaderAgent"]
    message_store = MessageStore(run_id, agent_names=agent_names)

    # Handle the case where __file__ might not be defined
    try:
        script_dir = Path(__file__).parent
    except NameError:
        # Fallback if __file__ is not defined
        script_dir = Path(sys.argv[0]).parent.absolute() if sys.argv[0] else Path.cwd()

    run_dir = script_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(run_dir)
    logging.info(json.dumps({"event": "run_started", "run_id": run_id, "args": vars(args)}))

    try:
        # Create the decentralized agent team
        logging.info(json.dumps({"event": "creating_team", "run_id": run_id}))
        decentralized_team = DecentralizedAgents(
            message_store=message_store,
            model_type=args.model_type,
            model_id=args.model_id,
            provider=args.provider,
            run_id=run_id
        )

        # Run the team on the task with enhanced collaboration instructions
        enhanced_task = f"{args.question}"
        logging.info(json.dumps({"event": "starting_execution", "run_id": run_id, "question": args.question}))
        result = decentralized_team.run(enhanced_task)
        
        logging.info(json.dumps({
            "event": "execution_completed", 
            "run_id": run_id, 
            "status": result.get("status", "unknown"),
            "has_answer": "answer" in result
        }))

        # Output the result
        if result["status"] in ["success", "success_early", "success_fallback"]:
            print(json.dumps({"answer": result["answer"]}))
            return 0
        else:
            error_msg = result.get('error', 'No valid results')
            logging.error(json.dumps({
                "event": "execution_failed", 
                "run_id": run_id, 
                "error": error_msg,
                "result": result
            }))
            print(f"\n❌ Team execution failed: {error_msg}")
            return 1
            
    except Exception as e:
        # Catch any unexpected errors and log them with full context
        logging.error(json.dumps({
            "event": "unexpected_error",
            "run_id": run_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "question": args.question
        }))
        
        # Also log the full stack trace for debugging
        import traceback
        logging.error(json.dumps({
            "event": "error_traceback",
            "run_id": run_id,
            "traceback": traceback.format_exc()
        }))
        
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run decentralized agent team")
    parser.add_argument("--model-type", required=True, help="Model type to use")
    parser.add_argument("--model-id", required=True, help="Model ID to use")
    parser.add_argument("--provider", help="Model provider")
    parser.add_argument("question", help="Question to answer")

    args = parser.parse_args()
    sys.exit(main(args))
