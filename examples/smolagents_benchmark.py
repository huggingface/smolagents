import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import datasets
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from smolagents import (
    AgentError,
    CodeAgent,
    GoogleSearchTool,
    HfApiModel,
    LiteLLMModel,
    PythonInterpreterTool,
    ToolCallingAgent,
    VisitWebpageTool,
)
from smolagents.agents import ActionStep


load_dotenv()
os.makedirs("output", exist_ok=True)

APPEND_ANSWER_LOCK = threading.Lock()



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "--date",
        type=str,
        default="2024-03-04",
        help="The date",
    )
    parser.add_argument(
        "--evals-dataset",
        type=str,
        default="smolagents/benchmark-v1",
    )
    # The evals dataset is gated, so you must first visit its page to request access: https://huggingface.co/datasets/smolagents-benchmark/benchmark-v1
    parser.add_argument(
        "--answers-dataset",
        type=str,
        default="smolagents/answers",
    )
    parser.add_argument(
        "--push-answers-dataset-to-hub",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="HfApiModel",
        choices=["LiteLLMModel", "HfApiModel"],
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, HfApiModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--agent-action-type",
        type=str,
        default="code",
        help="The agent action type: 'code', 'tool-calling', or 'vanilla' to use the vanilla llm",
    )
    return parser.parse_args()


def load_eval_dataset(args):
    # Choose the tasks to evaluate on:
    # tasks = ["gaia"]
    # or evaluate on all tasks: ["gaia", "math", "simpleqa"]
    tasks = datasets.get_dataset_config_names(args.evals_dataset)
    print(tasks)

    eval_ds = {task: datasets.load_dataset(args.evals_dataset, task, split="test") for task in tasks}
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
    with APPEND_ANSWER_LOCK, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print("Answer exported to file:", jsonl_file.resolve())


def answer_single_question(example, model, answers_file, action_type="code"):
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
        if action_type=="vanilla":
            answer= agent([{"role": "user", "text": augmented_question}]).content
            token_count = agent.last_output_token_count
            intermediate_steps = answer
        else:
            # Run agent 🚀
            answer = str(agent.run(augmented_question))
            token_count = agent.monitor.get_total_token_counts()
            # Remove memory from logs to make them more compact.
            for step in agent.memory.steps:
                if isinstance(step, ActionStep):
                    step.agent_memory = None
            intermediate_steps = str(agent.memory.steps)

        end_time = time.time()
    except Exception as e:
        print("Error on ", augmented_question, e)
        intermediate_steps = []
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        "token_counts": token_count,
    }
    append_answer(annotated_example, answers_file)


def answer_questions(
    eval_ds,
    model,
    date,
    action_type="code",
    output_dir="output",
    push_to_hub_dataset=None,
):
    date = date or datetime.date.today().isoformat()
    model_id = model.model_id

    for task in eval_ds:
        file_name = f"output/{model_id.replace('/', '__')}__{action_type}__{task}__{date}.jsonl"
        answered_questions = []
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                for line in f:
                    answered_questions.append(json.loads(line)["original_question"])

        examples_todo = [example for example in eval_ds[task] if example["question"] not in answered_questions]

        with ThreadPoolExecutor(max_workers=32) as exe:
            futures = [exe.submit(answer_single_question, example, model, file_name, action_type) for example in examples_todo]
            for f in tqdm(as_completed(futures), total=len(examples_todo), desc="Processing tasks"):
                f.result()

        print("All tasks processed.")

        if push_to_hub_dataset:
            ds = datasets.Dataset.from_pandas(pd.read_json(file_name, lines=True), split="test", preserve_index=False)
            config = f"{model_id.replace('/', '__')}__{action_type}__{task}"
            data_dir = f"{model_id}/{action_type}/{task}/{date}"
            ds.push_to_hub(
                push_to_hub_dataset,
                config_name=config,
                data_dir=data_dir,
                split="test",
                commit_message=f"Upload {config}",
            )

if __name__ == "__main__":
    args = parse_arguments()

    eval_ds = load_eval_dataset(args)

    if args.model_type == "LiteLLMModel":
        model = LiteLLMModel(
            args.model_id,
            max_completion_tokens=8192,
        )
    elif args.model_type == "HfApiModel":
        model = HfApiModel(args.model_id, provider="together", max_tokens=8192)

    answer_questions(eval_ds, model, action_type="code", date=args.date, push_to_hub_dataset=args.answers_dataset)
