import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
            additional_authorized_imports=["numpy", "sympy"],
            max_steps=10,
        )
    elif action_type == "tool-calling":
        agent = ToolCallingAgent(
            tools=[GoogleSearchTool(provider="serper"), VisitWebpageTool(), PythonInterpreterTool()],
            model=model,
            additional_authorized_imports=["numpy", "sympy"],
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
            token_counts = agent.monitor.get_total_token_counts()
            intermediate_steps = answer
        else:
            # Run agent 🚀
            answer = str(agent.run(augmented_question))
            token_counts = agent.monitor.get_total_token_counts()
            intermediate_steps = [dict(message) for message in agent.write_memory_to_messages()]

        end_time = time.time()
    except Exception as e:
        print("Error on ", augmented_question, e)
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


def answer_questions(
    eval_ds,
    model,
    date,
    action_type: str = "code",
    output_dir: str = "output",
    answers_dataset: str = None,
    push_answers_to_hub: bool = False,
    parallel_workers: int = 32,
):
    date = date or datetime.date.today().isoformat()
    model_id = model.model_id

    for task in eval_ds:
        file_name = f"{output_dir}/{model_id.replace('/', '__')}__{action_type}__{task}__{date}.jsonl"
        print(f"Starting processing and writing output to '{file_name}'")
        answered_questions = []
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                for line in f:
                    answered_questions.append(json.loads(line)["original_question"])

        examples_todo = [example for example in eval_ds[task] if example["question"] not in answered_questions]
        print(f"Launching {parallel_workers} parallel workers.")

        with ThreadPoolExecutor(max_workers=parallel_workers) as exe:
            futures = [
                exe.submit(answer_single_question, example, model, file_name, action_type) for example in examples_todo
            ]
            for f in tqdm(as_completed(futures), total=len(examples_todo), desc="Processing tasks"):
                f.result()

        print("All tasks processed.")

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
