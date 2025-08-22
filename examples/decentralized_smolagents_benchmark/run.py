#!/usr/bin/env python
# Example usage: python run.py --model-type LiteLLMModel --model-id gpt-4o --provider openai
"""Benchmarking script for decentralized agent implementation."""

import argparse
import datetime
import json
import os
import re
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

from langfuse import get_client


langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

from openinference.instrumentation.smolagents import SmolagentsInstrumentor


with langfuse.start_as_current_span(name="another-operation"):
    # Add to the current trace
    langfuse.update_current_trace(session_id="zero_shot_1", user_id="cvt8")


SmolagentsInstrumentor().instrument()


script_dir = Path(__file__).parent
output_path = script_dir / "output"
os.makedirs(output_path, exist_ok=True)


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

        # Add task information to each item and create unique IDs
        for i, item in enumerate(data_list):
            item["task"] = task
            item["source"] = task  # Add source field to match original format
            item["id"] = f"{task}_{i}"  # Create unique ID for each item

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

    # Get the date for file naming
    date = args.date if hasattr(args, "date") and args.date else datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        # Prepare question from the dataset row
        question = row["question"]
        if row.get("context"):
            question = f"Context: {row['context']}\n\nQuestion: {question}"

        # Run the decentralized agent process
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decentralized_agent.py")
        cmd = [
            "python",
            script_path,
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

        # Parse the final answer from JSON output
        final_answer = None
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line:
                try:
                    parsed = json.loads(line)
                    if "answer" in parsed:
                        final_answer = parsed["answer"]
                        break
                except json.JSONDecodeError:
                    continue

        if not final_answer:
            # Also check stderr for error messages
            error_msg = result.stderr.strip() if result.stderr else "No final answer reached"
            raise Exception(error_msg)

        # Calculate metrics
        duration = time.time() - start_time
        success = True
        error = None

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        success = False
        error = f"Process timed out after duration {duration}, error: {str(e)}"
        final_answer = None
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        success = False
        error_output = e.stderr.strip() if e.stderr else "No stderr output"
        stdout_output = e.stdout.strip() if e.stdout else "No stdout output"
        error = f"Process failed with exit code {e.returncode}. Stderr: {error_output}. Stdout: {stdout_output}"
        final_answer = None
    except Exception as e:
        duration = time.time() - start_time
        success = False
        error = str(e)
        final_answer = None

    # Prepare result dictionary matching the expected format
    model_id = f"decentralized-{args.model_type}-{args.model_id}"
    action_type = "decentralized-consensus"

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

    # Also save in the format expected by the scoring system
    scoring_result = {
        "model_id": model_id,
        "agent_action_type": action_type,
        "question": row["question"],
        "original_question": row["question"],
        "answer": final_answer,
        "true_answer": row.get("true_answer", ""),
        "source": row["task"],
        "start_time": start_time,
        "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
    }

    # Save result to general output file
    output_file = Path(output_path) / f"results_{date}.jsonl"
    with APPEND_ANSWER_LOCK:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(result, f)
            f.write("\n")

    # Save result to task-specific file for scoring
    task_output_file = Path(output_path) / f"{model_id.replace('/', '__')}__{action_type}__{row['task']}__{date}.jsonl"
    with APPEND_ANSWER_LOCK:
        task_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(task_output_file, "a") as f:
            json.dump(scoring_result, f)
            f.write("\n")

    return result


def normalize_answer(answer):
    """
    Normalize answer for comparison.

    Removes extra whitespace, converts to lowercase, and strips punctuation
    to enable more flexible answer matching.
    """
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # Remove extra whitespace
    answer = re.sub(r"\s+", " ", answer)
    # Remove common punctuation at the end
    answer = re.sub(r"[.!?;,]+$", "", answer)
    return answer


def calculate_exact_match_score(predicted_answer, true_answer):
    """
    Calculate exact match score (1.0 for perfect match, 0.0 otherwise).

    This is the strictest scoring metric.
    """
    return 1.0 if normalize_answer(predicted_answer) == normalize_answer(true_answer) else 0.0


def calculate_contains_score(predicted_answer, true_answer):
    """Calculate score based on whether the predicted answer contains the true answer."""
    normalized_pred = normalize_answer(predicted_answer)
    normalized_true = normalize_answer(true_answer)

    if not normalized_true:
        return 0.0

    return 1.0 if normalized_true in normalized_pred else 0.0


def calculate_benchmark_scores(jsonl_file_path):
    """Calculate scores for a benchmark result file."""
    if not os.path.exists(jsonl_file_path):
        return {"error": "File not found"}

    total_questions = 0
    exact_matches = 0
    contains_matches = 0

    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if not data:
                    continue

                predicted = data.get("answer", "")
                true_answer = data.get("true_answer", "")

                total_questions += 1
                exact_matches += calculate_exact_match_score(predicted, true_answer)
                contains_matches += calculate_contains_score(predicted, true_answer)

            except json.JSONDecodeError:
                continue

    if total_questions == 0:
        return {"error": "No valid questions found"}

    return {
        "total_questions": total_questions,
        "exact_match_score": exact_matches / total_questions,
        "contains_score": contains_matches / total_questions,
        "exact_matches": exact_matches,
        "contains_matches": contains_matches,
    }


def save_benchmark_scores(output_dir, model_id, action_type, date, eval_ds):
    """Calculate and save scores for all benchmarks."""
    scores_file = f"{output_dir}/benchmark_scores_{model_id.replace('/', '__')}__{action_type}__{date}.json"

    all_scores = {
        "model_id": model_id,
        "action_type": action_type,
        "date": date,
        "timestamp": datetime.datetime.now().isoformat(),
        "benchmarks": {},
    }

    total_questions_all = 0
    total_exact_matches_all = 0
    total_contains_matches_all = 0

    print("\n📊 Calculating benchmark scores...")

    for task in eval_ds:
        jsonl_file = f"{output_dir}/{model_id.replace('/', '__')}__{action_type}__{task}__{date}.jsonl"
        scores = calculate_benchmark_scores(jsonl_file)

        if "error" not in scores:
            all_scores["benchmarks"][task] = scores
            total_questions_all += scores["total_questions"]
            total_exact_matches_all += scores["exact_matches"]
            total_contains_matches_all += scores["contains_matches"]

            print(f"  📈 {task.upper()}:")
            print(f"     Questions: {scores['total_questions']}")
            print(
                f"     Exact Match: {scores['exact_match_score']:.1%} ({scores['exact_matches']}/{scores['total_questions']})"
            )
            print(
                f"     Contains: {scores['contains_score']:.1%} ({scores['contains_matches']}/{scores['total_questions']})"
            )
        else:
            print(f"  ❌ {task.upper()}: {scores['error']}")

    # Overall scores
    if total_questions_all > 0:
        all_scores["overall"] = {
            "total_questions": total_questions_all,
            "exact_match_score": total_exact_matches_all / total_questions_all,
            "contains_score": total_contains_matches_all / total_questions_all,
            "exact_matches": total_exact_matches_all,
            "contains_matches": total_contains_matches_all,
        }

        print("\n🎯 OVERALL SCORES:")
        print(f"  Questions: {total_questions_all}")
        print(
            f"  Exact Match: {all_scores['overall']['exact_match_score']:.1%} ({total_exact_matches_all}/{total_questions_all})"
        )
        print(
            f"  Contains: {all_scores['overall']['contains_score']:.1%} ({total_contains_matches_all}/{total_questions_all})"
        )

    # Save scores to file
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Scores saved to: {scores_file}")
    return all_scores


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

    # Calculate and save benchmark scores with proper variables
    output_dir = output_path
    model_id = f"decentralized-{args.model_type}-{args.model_id}"
    action_type = "decentralized-consensus"
    date = args.date
    eval_ds = df["task"].unique()

    save_benchmark_scores(output_dir, model_id, action_type, date, eval_ds)


if __name__ == "__main__":
    main()
