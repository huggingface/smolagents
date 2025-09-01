#!/usr/bin/env python
# Example usage: python run_centralized.py --model-type LiteLLMModel --model-id gpt-4o --provider openai
"""Benchmarking script for centralized agent implementation."""

import argparse
import datetime
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import Tool

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


import datasets
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.text_inspector_tool import TextInspectorTool, FileReaderTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    DownloadTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.text_inspector_tool import FileReaderTool
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
)



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


script_dir = Path(__file__).parent
output_path = script_dir / "output"
os.makedirs(output_path, exist_ok=True)

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

# Base prompt for team charter
TEAM_CHARTER = """You are part of an expert AI agent team working collaboratively to solve complex problems.

TEAM COORDINATION PRINCIPLES:
- Work systematically and build on each other's findings
- Communicate clearly about your role, findings, and next steps
- Verify important information through multiple approaches when possible
- Focus on accuracy and provide evidence for your conclusions
- Coordinate efforts to avoid duplication and ensure comprehensive coverage

RESPONSE FORMAT:
Always structure your responses with clear sections and evidence-based conclusions."""

APPEND_ANSWER_LOCK = threading.Lock()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Runs centralized agent team on smolagent benchmark.")
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
        help="The model type to use for centralized multi-agent system",
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
            df.groupby("task", group_keys=False)
            .apply(lambda x: x.sample(n=min(num_examples, len(x)), random_state=42))
            .reset_index(drop=True)
        )

    return df


def run_centralized_agents(row, args, model):
    """Run centralized agent on a single benchmark example."""
    start_time = time.time()

    # Get the date for file naming
    date = args.date if hasattr(args, "date") and args.date else datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        # Create the centralized agent
        agent = create_agent_team(model)
        
        # Prepare question from the dataset row
        question = row["question"]
        if row.get("context"):
            question = f"Context: {row['context']}\n\nQuestion: {question}"

        # Run the agent
        print(f"Running agent on question: {question[:100]}...")
        result = agent.run(question)
        print(f"Agent result type: {type(result)}")
        print(f"Agent result: {str(result)[:200]}")
        
        # Extract final answer from result
        if hasattr(result, 'content'):
            final_answer = result.content
        elif hasattr(result, 'final_answer'):
            final_answer = result.final_answer
        elif isinstance(result, str):
            final_answer = result
        else:
            final_answer = str(result)
            
        print(f"Final answer: {final_answer}")

        # Calculate metrics
        duration = time.time() - start_time
        success = True
        error = None
        run_id = f"centralized_{int(time.time() * 1000)}"  # Generate a simple run ID

    except Exception as e:
        duration = time.time() - start_time
        success = False
        error = str(e)
        final_answer = None
        run_id = None

    # Prepare result dictionary matching the expected format
    model_id = f"centralized-{args.model_type}-{args.model_id}"
    action_type = "centralized-agents"

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
        "run_id": run_id,
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
        "run_id": run_id,
    }

    # Save result to general output file
    output_file = Path(output_path) / f"centralized_results_{date}.jsonl"
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

def create_agent_team(model: Model):
    """Create a centralized multi-agent system with a manager and 4 specialized agents."""
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    # Create base tools that are shared between agents
    shared_tools = [visualizer, ti_tool]

    # Define tool sets for different agent types
    web_tools: List[Tool] = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        DownloadTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
    ]

    code_tools: List[Tool] = [PythonInterpreterTool()]
    reader_tools: List[Tool] = [FileReaderTool()]

    # Add shared tools to each tool set
    web_tools.extend(shared_tools)
    code_tools.extend(shared_tools)
    reader_tools.extend(shared_tools)

    # Create 4 specialized agents with detailed descriptions
    
    # Agent 1: Python Code Execution and Algorithm Implementation Specialist
    code_agent = CodeAgent(
        model=model,
        tools=code_tools,
        max_steps=15,
        verbosity_level=1,
        additional_authorized_imports=["*"],
        planning_interval=3,
        name="CodeAgent",
        description="""Python Code Execution and Algorithm Implementation Specialist.
        
Responsibilities:
- Write, test, and execute Python code with proper error handling and validation
- Create modular, testable functions with comprehensive docstrings and documentation
- Implement mathematical algorithms, computational solutions, and numerical analysis
- Run validation tests and smoke tests on code changes to ensure reliability
- Optimize code performance, debug complex issues, and handle edge cases
- Handle numerical computations, data processing tasks, and statistical analysis
- Extract precise numerical results for mathematical problems (e.g., "7", "12.5", "3/4")
- Validate results through multiple calculation methods when possible

Special focus on mathematical problem solving:
- Use Python to calculate exact results with proper precision
- Show all calculations, code, and reasoning processes
- Follow specific format requirements (decimal, fraction, etc.)
- Provide executable demonstrations of solutions"""
    )

    # Agent 2: Fast Web Research and Information Gathering Specialist
    web_research_agent = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=15,
        verbosity_level=1,
        name="WebResearchAgent",
        description="""Fast Web Research and Information Gathering Specialist.
        
Responsibilities:
- Conduct rapid, targeted web searches for relevant and up-to-date information
- Evaluate source credibility, authority, and reliability of information sources
- Perform thorough fact-checking and information verification from multiple sources
- Gather real-time data, current information, and recent developments
- Identify trending topics, breaking news, and emerging information patterns
- Extract key facts, summarize findings concisely, and present structured results
- Cross-reference findings against multiple independent sources for accuracy
- Provide source URLs, publication dates, and credibility assessments

Research methodology:
- Always verify information from multiple independent, authoritative sources
- Prioritize recent, peer-reviewed, and official sources over outdated information
- Include comprehensive source attribution with URLs and publication dates
- Flag conflicting information and present different perspectives objectively
- Focus on factual accuracy over speed when sources present contradictory claims"""
    )

    # Agent 3: Document Analysis and Technical Specification Specialist
    document_agent = ToolCallingAgent(
        model=model,
        tools=reader_tools,
        max_steps=10,
        verbosity_level=1,
        name="DocumentAgent",
        description="""Document Analysis and Technical Specification Specialist.
        
Responsibilities:
- Analyze and extract critical information from technical documents, PDFs, and specifications
- Maintain precise citations with page numbers, section references, and source attribution
- Structure complex documentation into digestible, actionable summaries and insights
- Validate technical specifications against implementation requirements and standards
- Cross-reference multiple documents for consistency, completeness, and accuracy
- Identify critical details, potential implementation considerations, and technical constraints
- Extract both explicit information and implied requirements from documentation
- Highlight contradictions, ambiguities, and gaps found in documents

Documentation standards:
- Always include precise citations with page numbers or section references
- Maintain clear separation between documented facts and personal interpretations
- Focus on actionable information that directly impacts problem-solving and decision-making
- Cross-reference claims against other available documentation and external sources
- Present structured analysis with clear organization and logical flow"""
    )

    # Agent 4: Comprehensive Analysis and Advanced Research Specialist
    analysis_tools = list(set(web_tools + code_tools))
    analysis_agent = CodeAgent(
        model=model,
        tools=analysis_tools,
        max_steps=20,
        verbosity_level=1,
        additional_authorized_imports=["*"],
        planning_interval=4,
        name="AnalysisAgent",
        description="""Comprehensive Analysis and Advanced Research Specialist.
        
Responsibilities:
- Conduct thorough, multi-layered investigations and comprehensive analysis of complex problems
- Develop, test, and rigorously validate complex hypotheses and theoretical frameworks
- Synthesize information from diverse sources into coherent, actionable insights and conclusions
- Perform advanced reasoning, logical validation, and critical evaluation of findings
- Design and execute comprehensive research methodologies with systematic approaches
- Validate findings through multiple analytical approaches and cross-verification methods
- Integrate quantitative computational analysis with qualitative research insights
- Challenge assumptions, test edge cases, and explore alternative explanations

Advanced analytical framework:
- Apply systematic analytical frameworks and methodologies to complex, multi-faceted problems
- Present multiple perspectives and consider alternative explanations and interpretations
- Use both web research and code execution capabilities to validate hypotheses comprehensively
- Document reasoning processes, analytical methodologies, and decision-making criteria clearly
- Integrate diverse analytical approaches including statistical, computational, and theoretical methods
- Maintain objectivity while exploring creative and innovative solution approaches"""
    )

    # Create the manager agent with comprehensive coordination capabilities
    manager_agent = CodeAgent(
        model=model,
        tools=shared_tools,  # Manager has minimal tools, delegates to agents
        max_steps=25,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=5,
        managed_agents=[code_agent, web_research_agent, document_agent, analysis_agent],
        name="ManagerAgent",
        description="""Centralized Manager and Team Coordination Specialist.
        
Primary Role:
- Orchestrate and coordinate the work of 4 specialized agents to solve complex, multi-faceted problems
- Strategically delegate tasks to the most appropriate agent based on their expertise and capabilities
- Synthesize results from multiple agents into comprehensive, well-reasoned final answers
- Ensure quality control and consistency across all agent contributions and outputs
- Manage workflow optimization, resource allocation, and inter-agent communication

Coordination Responsibilities:
- CodeAgent: Delegate mathematical computations, algorithm implementation, and code execution tasks
- WebResearchAgent: Assign web research, fact-checking, and real-time information gathering tasks  
- DocumentAgent: Direct document analysis, technical specification review, and citation extraction
- AnalysisAgent: Coordinate comprehensive analysis, hypothesis testing, and multi-source synthesis

Management Strategy:
- Break down complex problems into specialized subtasks aligned with agent expertise
- Ensure comprehensive coverage of all problem aspects through strategic task distribution
- Validate and cross-check results from multiple agents for accuracy and consistency
- Integrate diverse perspectives and methodologies into cohesive, evidence-based solutions
- Maintain high standards for accuracy, completeness, and methodological rigor"""
    )

    return manager_agent



def main():
    """Main benchmarking function."""
    args = parse_arguments()

    # Set date if not provided
    if args.date is None:
        args.date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Initialize the model
    if args.model_type == "LiteLLMModel":
        if args.provider == "openai":
            # For OpenAI, use the model_id directly
            model = LiteLLMModel(model_id=args.model_id)
        else:
            # For other providers, include provider in model_id
            model_id = f"{args.provider}/{args.model_id}" if args.provider != "hf-inference" else args.model_id
            model = LiteLLMModel(model_id=model_id)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Create downloads folder for browser
    os.makedirs(BROWSER_CONFIG["downloads_folder"], exist_ok=True)

    # Load dataset
    print(f"Loading dataset {args.eval_dataset}...")
    df = load_eval_dataset(args.eval_dataset, args.num_examples)

    # If num_examples is specified, sample from each task
    if args.num_examples is not None:
        df = (
            df.groupby("task", group_keys=False)
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
                future = executor.submit(run_centralized_agents, row, args, model)
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
    model_id = f"centralized-{args.model_type}-{args.model_id}"
    action_type = "centralized-agents"
    date = args.date
    eval_ds = df["task"].unique()

    save_benchmark_scores(output_dir, model_id, action_type, date, eval_ds)


if __name__ == "__main__":
    main()
