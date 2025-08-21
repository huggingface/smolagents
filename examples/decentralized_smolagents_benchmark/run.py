#!/usr/bin/env python
# Example usage: python run.py --model-type LiteLLMModel --model-id gpt-4o --provider openai
"""Benchmarking script for decentralized agent implementation."""

import argparse
import datetime
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()
os.makedirs("output", exist_ok=True)

APPEND_ANSWER_LOCK = threading.Lock()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Runs decentralized agent team on smolagent benchmark.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="The date for the evaluation.",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="smolagents/benchmark-v1",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The model type to use for the decentralized agents",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="hf-inference",
        help="The provider for the model",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="The number of concurrent benchmark runs",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit the number of examples per task (useful for testing)",
    )
    parser.add_argument(
        "--push-answers-to-hub",
        action="store_true",
        help="Push the answers to the hub",
    )
    parser.add_argument(
        "--answers-dataset",
        type=str,
        default="smolagents/answers",
    )
    return parser.parse_args()


def load_eval_dataset(eval_dataset, num_examples=None):
    """Load the evaluation dataset."""
    # Get all available tasks
    tasks = datasets.get_dataset_config_names(eval_dataset)
    print(f"Available tasks: {tasks}")

    # Load each task's dataset
    all_data = []
    for task in tasks:
        dataset = datasets.load_dataset(eval_dataset, task, split="test")
        data_list = list(dataset)
        print(f"Loaded {len(data_list)} examples for {task}")

        # Add task information to each item
        for item in data_list:
            item["task"] = task
            item["source"] = task  # Add source field to match original format

        all_data.extend(data_list)

    df = pd.DataFrame(all_data)

    if num_examples is not None:
        # Sample num_examples from each task
        df = (
            df.groupby("task")
            .apply(lambda x: x.sample(n=min(num_examples, len(x)), random_state=42))
            .reset_index(drop=True)
        )

    return df


def run_decentralized_agent(row, args):
    """Run decentralized agent on a single benchmark example."""
    start_time = time.time()

    try:
        # Prepare question from the dataset row
        question = row["question"]
        if row.get("context"):
            question = f"Context: {row['context']}\n\nQuestion: {question}"

        # Run the decentralized agent process
        cmd = [
            "python",
            str(Path(__file__).parent / "decentralized_agent.py"),
            "--model-type",
            args.model_type,
            "--model-id",
            args.model_id,
            "--provider",
            args.provider,
            question,
        ]

        # Run the process and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the final answer from output
        final_answer = None
        for line in result.stdout.split("\n"):
            if line.startswith("FinalAnswer: "):
                final_answer = line.replace("FinalAnswer: ", "").strip()
                break

        if not final_answer:
            raise Exception("No final answer reached")

        # Calculate metrics
        duration = time.time() - start_time
        success = True
        error = None

    except Exception as e:
        duration = time.time() - start_time
        success = False
        error = str(e)
        final_answer = None

    # Prepare result dictionary
    result = {
        "task": row["task"],
        "question_id": row["id"],
        "success": success,
        "error": error,
        "duration": duration,
        "answer": final_answer,
        "model_type": args.model_type,
        "model_id": args.model_id,
        "provider": args.provider,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Save result to output file
    output_file = Path("output") / f"results_{args.date}.jsonl"
    with APPEND_ANSWER_LOCK:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(result, f)
            f.write("\n")

    return result


def main():
    """Main benchmarking function."""
    args = parse_arguments()

    # Set date if not provided
    if args.date is None:
        args.date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Load dataset
    print(f"Loading dataset {args.eval_dataset}...")
    df = load_eval_dataset(args.eval_dataset, args.num_examples)

    # If num_examples is specified, sample from each task
    if args.num_examples is not None:
        df = (
            df.groupby("task")
            .apply(lambda x: x.sample(n=min(args.num_examples, len(x)), random_state=42))
            .reset_index(drop=True)
        )

    print(f"\nLoaded {len(df)} examples total:")
    for task in df["task"].unique():
        task_count = len(df[df["task"] == task])
        print(f"- {task}: {task_count} examples")

    # Run benchmark
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        futures = []

        # Group examples by task for better progress tracking
        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            print(f"\n🚀 Starting benchmark for {task} with {len(task_df)} examples...")

            for _, row in task_df.iterrows():
                future = executor.submit(run_decentralized_agent, row, args)
                futures.append(future)

        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing examples"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing example: {str(e)}")

    # Calculate and print summary statistics per task
    print("\n📊 Results Summary:")
    for task in df["task"].unique():
        task_results = [r for r in results if r["task"] == task]
        task_success = sum(1 for r in task_results if r["success"])
        task_duration = sum(r["duration"] for r in task_results) / len(task_results) if task_results else 0

        print(f"\n{task}:")
        print(f"  Total examples: {len(task_results)}")
        print(f"  Success rate: {task_success / len(task_results):.2%}")
        print(f"  Average duration: {task_duration:.2f}s")

    # Overall statistics
    total_success = sum(1 for r in results if r["success"])
    avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0

    print("\n📈 Overall Statistics:")
    print(f"  Total examples: {len(results)}")
    print(f"  Success rate: {total_success / len(results):.2%}")
    print(f"  Average duration: {avg_duration:.2f}s")

    # Push results to hub if requested
    if args.push_answers_to_hub:
        print("\n🚀 Pushing results to hub not yet implemented")


if __name__ == "__main__":
    main()
