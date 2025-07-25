# GAIA Benchmark Runner for Open Deep Research
# =============================================
#
#
# EXAMPLE COMMAND: from folder examples/open_deep_research, run:
# python run_gaia.py --concurrency 32 --run-name generate-traces-03-apr-noplanning --model-id gpt-4o

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from scripts.gaia_scorer import check_close_call, question_scorer
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
)


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--use-open-models", type=bool, default=False)
    parser.add_argument("--use-raw-dataset", action="store_true")
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated any VPN like Tailscale, else some URLs will be blocked!")

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent_team(model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
        visualizer,  # Correcting error: "prediction": "Critical Error: Cannot use inspect_file_as_text tool with images: use visualizer instead!",
    ]

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    data_dir = BASE_DIR / "data" / "gaia"
    if not data_dir.exists():
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir=str(data_dir),
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir=str(data_dir),
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = str(data_dir / "2023" / set_to_run / row["file_name"])
        return row

    eval_ds = datasets.load_dataset(
        path=str(data_dir),
        name="default",
        split=set_to_run,
        data_files={"validation": "2023/validation/metadata.jsonl", "test": "2023/test/metadata.jsonl"},
    )

    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds


def append_answer(entry: dict, jsonl_file: Path) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        """Convert objects to JSON serializable format"""
        if hasattr(obj, "dict"):
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)

    try:
        with append_answer_lock, jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, default=convert_to_serializable, ensure_ascii=False) + "\n")
            fp.flush()  # Ensure the buffer is flushed immediately
            os.fsync(fp.fileno())  # Force the file system to write the data to disk
    except Exception as e:
        print(f"Error writing to answers file {jsonl_path}: {e}")


def answer_single_question(
    example: dict, model_id: str, answers_file: str, errors_file: str, visual_inspection_tool
) -> None:
    print(f"Processing question: {example['question']}")

    # Initialize variables to avoid unbound errors
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_time = None
    intermediate_steps = []
    parsing_error = False
    iteration_limit_exceeded = False
    raised_exception = False
    exception = None
    output = None
    augmented_question = example["question"]  # Default value

    try:
        model_params: dict[str, Any] = {
            "model_id": model_id,
            "custom_role_conversions": custom_role_conversions,
        }
        if model_id == "o1":
            model_params["reasoning_effort"] = "high"
            model_params["max_completion_tokens"] = 8192
        else:
            model_params["max_tokens"] = 4096
        model = LiteLLMModel(**model_params)
        document_inspection_tool = TextInspectorTool(model, 100000)

        agent = create_agent_team(model)

        augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

        if example.get("file_name"):
            if ".zip" in example["file_name"]:
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
                prompt_use_files += get_zip_description(
                    example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
                )
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
                prompt_use_files += get_single_file_description(
                    example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
                )
            augmented_question += prompt_use_files

        try:
            final_result = agent.run(augmented_question)
            agent_memory = agent.write_memory_to_messages()
            final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
            output = str(final_result)
            # Fix: Use .dict() method for ChatMessage objects instead of dict() constructor
            intermediate_steps = [msg.dict() if hasattr(msg, "dict") else str(msg) for msg in agent_memory]
            parsing_error = any("AgentParsingError" in str(step) for step in intermediate_steps)
            iteration_limit_exceeded = "Agent stopped due to iteration limit or time limit." in output
        except Exception as e:
            print(f"Error processing question '{example['question']}': {e}")
            output = f"Error: {str(e)}"
            exception = e
            raised_exception = True

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Safely get token counts with fallback
        try:
            token_counts_manager = agent.monitor.get_total_token_counts()
            token_counts_web = list(agent.managed_agents.values())[0].monitor.get_total_token_counts()
            total_token_counts = {
                "input": getattr(token_counts_manager, "input", 0) + getattr(token_counts_web, "input", 0),
                "output": getattr(token_counts_manager, "output", 0) + getattr(token_counts_web, "output", 0),
            }
        except Exception as e:
            print(f"Error getting token counts: {e}")
            total_token_counts = {"input": 0, "output": 0}

    except Exception as e:
        print(f"Critical error in answer_single_question for '{example['question']}': {e}")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = f"Critical Error: {str(e)}"
        exception = e
        raised_exception = True
        total_token_counts = {"input": 0, "output": 0}
        intermediate_steps = []

    # Always compute scores and append answer, regardless of errors
    try:
        is_correct = question_scorer(str(output), str(example["true_answer"]))
        is_near_correct = check_close_call(str(output), str(example["true_answer"]), is_correct)
    except Exception as e:
        print(f"Error computing scores: {e}")
        is_correct = False
        is_near_correct = False

    print(f"Question: {example['question'][:50]}{'...' if len(example['question']) > 50 else ''}")
    print(f"Prediction: {str(output)[:100]}{'...' if len(str(output)) > 100 else ''}")
    print(f"Correct: {is_correct}")

    annotated_example = {
        "agent_name": model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
        "is_correct": is_correct,
        "is_near_correct": is_near_correct,
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
    }

    # Always append to main answers file
    append_answer(annotated_example, Path(answers_file))

    # If there was an error, also append to errors file
    if raised_exception or parsing_error or iteration_limit_exceeded:
        append_answer(annotated_example, Path(errors_file))


def get_examples_to_answer(answers_file: Path, eval_ds: datasets.Dataset) -> list[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in eval_ds.to_list() if line["question"] not in done_questions]


def compute_score(answers_file: Path) -> None:
    if not answers_file.exists():
        print(f"Error: The file {answers_file} does not exist.")
        return

    if answers_file.stat().st_size == 0:
        print(f"Error: The file {answers_file} is empty.")
        return

    try:
        df = pd.read_json(answers_file, lines=True)
    except ValueError as e:
        print(f"Error reading JSON from {answers_file}: {e}")
        return

    if "is_correct" not in df.columns:
        df["is_correct"] = df.apply(lambda x: question_scorer(str(x["prediction"]), str(x["true_answer"])), axis=1)

    # Calculate comprehensive scores
    total_questions = len(df)
    correct_answers = df["is_correct"].sum()
    accuracy = df["is_correct"].mean()

    # Calculate additional metrics
    error_count = df["agent_error"].notna().sum()
    parsing_error_count = df["parsing_error"].sum()
    iteration_limit_count = df["iteration_limit_exceeded"].sum()

    # Group by task level for detailed analysis
    task_scores = None
    if "task" in df.columns:
        task_scores = (
            df.groupby("task")
            .agg(
                {
                    "is_correct": ["count", "sum", "mean"],
                    "agent_error": lambda x: x.notna().sum(),
                    "parsing_error": "sum",
                    "iteration_limit_exceeded": "sum",
                }
            )
            .round(3)
        )
        task_scores.columns = ["total", "correct", "accuracy", "errors", "parsing_errors", "iteration_limits"]

    # Save detailed score analysis
    score_file = answers_file.parent / "detailed_scores.txt"
    with score_file.open("w", encoding="utf-8") as f:
        f.write("GAIA Benchmark Detailed Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Overall Performance:\n")
        f.write(f"  Total Questions: {total_questions}\n")
        f.write(f"  Correct Answers: {correct_answers}\n")
        f.write(f"  Accuracy: {accuracy * 100:.2f}%\n\n")

        f.write("Error Analysis:\n")
        f.write(f"  Agent Errors: {error_count} ({error_count / total_questions * 100:.1f}%)\n")
        f.write(f"  Parsing Errors: {parsing_error_count} ({parsing_error_count / total_questions * 100:.1f}%)\n")
        f.write(
            f"  Iteration Limits: {iteration_limit_count} ({iteration_limit_count / total_questions * 100:.1f}%)\n\n"
        )

        if "task" in df.columns:
            f.write("Performance by Task Level:\n")
            if task_scores is not None:
                f.write(str(task_scores) + "\n\n")

        f.write("Individual Results:\n")
        f.write("-" * 50 + "\n")
        for idx, row in df.iterrows():
            status = "✅ CORRECT" if row["is_correct"] else "❌ INCORRECT"
            f.write(f"{status} | Task {row.get('task', 'N/A')} | {row['question'][:80]}...\n")
            f.write(f"  Predicted: {str(row['prediction'])[:100]}...\n")
            f.write(f"  Expected: {str(row['true_answer'])[:100]}...\n")
            if row.get("agent_error"):
                f.write(f"  Error: {str(row['agent_error'])[:100]}...\n")
            f.write("\n")

    # Save JSON summary for programmatic access
    summary_file = answers_file.parent / "score_summary.json"
    summary_data = {
        "total_questions": int(total_questions),
        "correct_answers": int(correct_answers),
        "accuracy": float(accuracy),
        "error_rate": float(error_count / total_questions),
        "parsing_error_rate": float(parsing_error_count / total_questions),
        "iteration_limit_rate": float(iteration_limit_count / total_questions),
        "timestamp": datetime.now().isoformat(),
        "answers_file": str(answers_file.name),
    }

    if "task" in df.columns and task_scores is not None:
        summary_data["task_performance"] = task_scores.to_dict()

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("📊 GAIA BENCHMARK RESULTS")
    print("=" * 60)
    print("📈 Overall Performance:")
    print(f"   Total Questions: {total_questions}")
    print(f"   Correct Answers: {correct_answers}")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print()
    print("⚠️  Error Analysis:")
    print(f"   Agent Errors: {error_count} ({error_count / total_questions * 100:.1f}%)")
    print(f"   Parsing Errors: {parsing_error_count} ({parsing_error_count / total_questions * 100:.1f}%)")
    print(f"   Iteration Limits: {iteration_limit_count} ({iteration_limit_count / total_questions * 100:.1f}%)")

    if "task" in df.columns:
        print()
        print("📋 Performance by Task Level:")
        for task_level in sorted(df["task"].unique()):
            task_data = df[df["task"] == task_level]
            task_acc = task_data["is_correct"].mean()
            task_count = len(task_data)
            task_correct = task_data["is_correct"].sum()
            print(f"   Level {task_level}: {task_acc * 100:.1f}% ({task_correct}/{task_count})")

    print()
    print(f"💾 Detailed results saved to: {score_file}")
    print(f"💾 Summary data saved to: {summary_file}")
    print("=" * 60)


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    def create_output_folders(set_to_run):
        """Create output folders if they don't exist."""
        output_folder = Path(f"output/{set_to_run}")
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output folder exists at: {output_folder}")

    create_output_folders(args.set_to_run)

    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    eval_df = pd.DataFrame(list(eval_ds))
    print(eval_df["task"].value_counts())

    answers_file = BASE_DIR / "output" / args.set_to_run / f"{args.run_name}.jsonl"
    errors_file = BASE_DIR / "output" / args.set_to_run / f"{args.run_name}_errors.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)

    print(f"Tasks to run: {len(tasks_to_run)}")
    if len(tasks_to_run) == 0:
        print("No new tasks to process. All questions may have been completed already.")
        print("To rerun all tasks, delete the existing output file.")
        return

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(
                answer_single_question,
                example,
                args.model_id,
                answers_file,
                errors_file,
                visualizer,  # Fix: Use visualizer for images instead of TextInspectorTool
            )
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            try:
                f.result()
            except Exception as e:
                print(f"Error in task: {e}")
                continue

    if not answers_file.exists():
        print(f"Error: The answers file {answers_file} was not created. Check the append_answer function.")
    else:
        print(f"✅ Main answers file created successfully: {answers_file}")

    if errors_file.exists():
        print(f"⚠️  Errors file created: {errors_file}")
    else:
        print("✅ No errors file created (no errors encountered)")

    print("All tasks processed.")
    compute_score(answers_file)


if __name__ == "__main__":
    main()
