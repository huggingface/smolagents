# Smolagents Benchmark Runner
# ===========================
# 
# Example usage: 
# python run.py --model-type LiteLLMModel --model-id gpt-4o --provider openai --agent-action-type tool-calling
# python run.py --model-type InferenceClientModel --model-id gpt-4o --provider openai --agent-action-type code

import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator
import re

import datasets
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from smolagents import (
    AgentError,
    CodeAgent,
    GoogleSearchTool,
    InferenceClientModel,
    LiteLLMModel,
    PythonInterpreterTool,
    ToolCallingAgent,
    VisitWebpageTool,
)
from smolagents.memory import FinalAnswerStep, ActionStep, PlanningStep
from smolagents.models import ChatMessageStreamDelta


load_dotenv()
os.makedirs("output", exist_ok=True)

APPEND_ANSWER_LOCK = threading.Lock()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Runs an agent powered by the given model on smolagent benchmark.")
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
    # The eval dataset is gated, so you must first visit its page to request access: https://huggingface.co/datasets/smolagents-benchmark/benchmark-v1
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        choices=["LiteLLMModel", "InferenceClientModel"],
        help="The model type to use (LiteLLMModel or InferenceClientModel)",
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
        help="The provider for InferenceClientModel - will not be used for LiteLLMModel",
    )
    parser.add_argument(
        "--agent-action-type",
        type=str,
        default="code",
        choices=["code", "tool-calling", "vanilla"],
        help="The agent action type: 'code', 'tool-calling', or 'vanilla' to use the vanilla llm",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="The number of processes to run in parallel",
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


def load_eval_dataset(eval_dataset):
    # Choose the tasks to evaluate on:
    # tasks = ["gaia"]
    # or evaluate on all tasks: ["gaia", "math", "simpleqa"]
    tasks = datasets.get_dataset_config_names(eval_dataset)
    print(tasks)

    eval_ds = {task: datasets.load_dataset(eval_dataset, task, split="test") for task in tasks}
    print(pd.DataFrame(eval_ds["simpleqa"]).head())
    return eval_ds


def serialize_agent_error(obj):
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with APPEND_ANSWER_LOCK, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, default=convert_to_serializable) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"


def answer_single_question(example, model, answers_file, action_type):
    if action_type == "vanilla":
        agent = model
    elif action_type == "code":
        agent = CodeAgent(
            tools=[GoogleSearchTool(provider="serper"), VisitWebpageTool()],
            model=model,
            max_steps=10,
        )
    elif action_type == "tool-calling":
        agent = ToolCallingAgent(
            tools=[GoogleSearchTool(provider="serper"), VisitWebpageTool(), PythonInterpreterTool()],
            model=model,
            max_steps=10,
        )

    augmented_question = example["question"]
    if example["source"] == "SimpleQA":
        augmented_question += " Answer with only the final number."
    if example["source"] == "MATH":
        augmented_question += " Write code, not latex."

    start_time = time.time()

    try:
        if action_type == "vanilla":
            answer = agent([{"role": "user", "content": augmented_question}]).content
            # For vanilla agents, the agent is just the model and doesn't have a monitor
            token_counts = {"input": 0, "output": 0}
            intermediate_steps = answer
        else:
            # Run agent 🚀
            result = agent.run(augmented_question)
            try:
                if isinstance(result, Generator):
                    steps = list(result)
                    final_step = steps[-1] if steps else None
                    if isinstance(final_step, FinalAnswerStep):
                        answer = final_step.output
                    elif isinstance(final_step, ActionStep):
                        answer = final_step.action_output
                    elif isinstance(final_step, PlanningStep):
                        answer = final_step.plan
                    elif isinstance(final_step, ChatMessageStreamDelta):
                        answer = final_step.content
                    else:
                        answer = str(final_step)
                else:
                    answer = result if isinstance(result, str) else str(result)
            except Exception as e:
                answer = f"Error extracting answer: {str(e)}"

            token_counts = agent.monitor.get_total_token_counts()
            
            # CRITICAL FIX: Changed dict(message) to message.dict()
            # ISSUE: Was getting "'ChatMessage' object is not iterable" error in JSON output
            # CAUSE: ChatMessage objects are not iterable with dict() constructor
            # SOLUTION: Use the .dict() method provided by ChatMessage class
            intermediate_steps = [message.dict() for message in agent.write_memory_to_messages()]

        end_time = time.time()
        duration = end_time - start_time
        answer_preview = str(answer)[:100] + ('...' if len(str(answer)) > 100 else '') if answer else "No answer"
        print(f"✅ Question processed in {duration:.2f}s - Answer: {answer_preview}")
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        question_preview = str(augmented_question)[:50] + ('...' if len(str(augmented_question)) > 50 else '')
        print(f"❌ Error after {duration:.2f}s on question: {question_preview}")
        print(f"   Error details: {str(e)}")
        intermediate_steps = []
        token_counts = {"input": 0, "output": 0}
        answer = str(e)
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "model_id": model.model_id,
        "agent_action_type": action_type,
        "question": augmented_question,
        "original_question": example["question"],
        "answer": answer,
        "true_answer": example["true_answer"],
        "source": example["source"],
        "intermediate_steps": intermediate_steps,
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": token_counts,
    }
    append_answer(annotated_example, answers_file)


# ==============================
# SCORING SYSTEM (ADDED FEATURE)
# ==============================
# This section was added to provide comprehensive benchmark evaluation
# with multiple scoring metrics and detailed performance analysis.

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
    answer = re.sub(r'\s+', ' ', answer)
    # Remove common punctuation at the end
    answer = re.sub(r'[.!?;,]+$', '', answer)
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
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if not data:
                    continue
                    
                predicted = data.get('answer', '')
                true_answer = data.get('true_answer', '')
                
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


def answer_questions(
    eval_ds,
    model,
    date,
    action_type: str = "code",
    output_dir: str = "output",
    answers_dataset: str | None = None,
    push_answers_to_hub: bool = False,
    parallel_workers: int = 32,
):
    date = date or datetime.date.today().isoformat()
    model_id = model.model_id

    for task in eval_ds:
        file_name = f"{output_dir}/{model_id.replace('/', '__')}__{action_type}__{task}__{date}.jsonl"
        print(f"\n🚀 Starting benchmark: {task}")
        print(f"📄 Writing output to: {file_name}")
        answered_questions = []
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                for line in f:
                    answered_questions.append(json.loads(line)["original_question"])

        examples_todo = [example for example in eval_ds[task] if example["question"] not in answered_questions]
        total_questions = len(eval_ds[task])
        remaining_questions = len(examples_todo)
        completed_questions = total_questions - remaining_questions

        # IMPROVED LOGGING: Added detailed progress tracking with emojis for better readability
        print(
            f"📊 Progress: {completed_questions}/{total_questions} questions completed ({remaining_questions} remaining)"
        )
        if remaining_questions == 0:
            print(f"✅ All questions for {task} already completed!")
            continue

        print(f"👥 Launching {parallel_workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=parallel_workers) as exe:
            futures = [
                exe.submit(answer_single_question, example, model, file_name, action_type) for example in examples_todo
            ]
            for f in tqdm(as_completed(futures), total=len(examples_todo), desc="Processing tasks"):
                f.result()

        print(f"✅ All tasks for {task} processed.")

        if push_answers_to_hub and answers_dataset:
            print("Pushing answers to hub...")
            ds = datasets.Dataset.from_pandas(pd.read_json(file_name, lines=True), split="test", preserve_index=False)
            config = f"{model_id.replace('/', '__')}__{action_type}__{task}"
            data_dir = f"{model_id}/{action_type}/{task}/{date}"
            ds.push_to_hub(
                answers_dataset,
                config_name=config,
                data_dir=data_dir,
                split="test",
                commit_message=f"Upload {config}",
            )

    # Calculate and save benchmark scores
    save_benchmark_scores(output_dir, model_id, action_type, date, eval_ds)


if __name__ == "__main__":
    args = parse_arguments()

    eval_ds = load_eval_dataset(args.eval_dataset)

    if args.model_type == "LiteLLMModel":
        model = LiteLLMModel(
            model_id=args.model_id,
            max_completion_tokens=8192,
        )
    else:
        model = InferenceClientModel(model_id=args.model_id, provider=args.provider, max_tokens=8192)

    answer_questions(
        eval_ds,
        model,
        args.date,
        action_type=args.agent_action_type,
        answers_dataset=args.answers_dataset,
        push_answers_to_hub=args.push_answers_to_hub,
        parallel_workers=args.parallel_workers,
    )
