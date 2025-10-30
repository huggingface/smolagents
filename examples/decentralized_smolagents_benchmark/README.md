# Decentralized smolagents Benchmark

This folder contains a decentralized multi-agent system implementation for benchmarking against the smolagents benchmark dataset. The system coordinates multiple specialized agents working collaboratively to solve complex problems.

## Overview

The decentralized approach distributes problem-solving across multiple specialized agents that communicate and coordinate through a message-passing system with consensus mechanisms. This contrasts with the centralized approach where a single agent has access to all tools.

### Architecture

The system consists of:

- **4 Specialized Agents**:
  - **CodeAgent**: Handles code execution and computational tasks
  - **WebSearchAgent**: Performs web searches and information retrieval
  - **DeepResearchAgent**: Conducts in-depth research using web browsing
  - **DocumentReaderAgent**: Reads and analyzes various document formats

- **Message Store**: Central communication hub for agent coordination
- **Consensus Protocol**: Voting mechanism for final answer agreement

## Files

- **`decentralized_agent.py`**: Main entry point for running a single question through the decentralized agent team
- **`run.py`**: Benchmark runner that evaluates the decentralized system across the entire benchmark dataset
- **`run_centralized.py`**: Comparison implementation using a centralized agent approach
- **`requirements.txt`**: Python dependencies required for the project
- **`scripts/`**: Supporting modules for agents, tools, communication, and utilities

### Key Scripts

- `scripts/agents.py`: Agent definitions and team coordination logic
- `scripts/message_store.py`: Message-passing infrastructure for agent communication
- `scripts/consensus_protocol.py`: Voting mechanism for reaching consensus on answers
- `scripts/decentralized_tools.py`: Custom tools for decentralized agent communication
- `scripts/text_web_browser.py`: Text-based web browsing tools
- `scripts/text_inspector_tool.py`: Document reading and analysis tools
- `scripts/visual_qa.py`: Visual question answering capabilities
- `scripts/html_renderer.py`: HTML visualization of agent runs
- `scripts/convert_messages_to_html.py`: Convert message logs to HTML format
- `scripts/gaia_scorer.py`: Scoring utilities for GAIA benchmark format

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:
```bash
# API Keys
OPENAI_API_KEY=your_openai_key #You can replace it by whatever model you want to use
ANTHROPIC_API_KEY=your_anthropic_key #You can replace it by whatever model you want to use
SERPAPI_API_KEY=your_serpapi_key  # For web search functionality
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key  # Optional: for tracing
LANGFUSE_SECRET_KEY=your_langfuse_secret_key  # Optional: for tracing
LANGFUSE_HOST=your_langfuse_host              # Optional: for tracing
```

## Usage

### Running a Single Question

Use `decentralized_agent.py` to run a single question through the decentralized team:

```bash
python decentralized_agent.py \
  --model-type LiteLLMModel \
  --model-id gpt-4o \ #or another model
  --provider openai \ #or another provider
  "What is the half of the speed of a Leopard?"
```

**Arguments:**
- `--model-type`: Model type to use (e.g., `LiteLLMModel`)
- `--model-id`: Specific model identifier (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`)
- `--provider`: Model provider (e.g., `openai`, `anthropic`, `hf-inference`)
- `question`: The question to answer (positional argument)

**Output:**
- Creates a `runs/{run_id}/` directory with:
  - `run.log`: JSON-formatted execution logs
  - Agent interaction traces and message history

### Running the Full Benchmark

Use `run.py` to evaluate across the entire benchmark dataset:

```bash
python run.py \
  --model-type LiteLLMModel \
  --model-id gpt-4o \ #or another model
  --provider openai \ #or another provider
  --parallel-workers 4 
```

**Arguments:**
- `--date`: Date string for the evaluation (default: current date)
- `--eval-dataset`: Dataset to evaluate on (default: `smolagents/benchmark-v1`)
- `--model-type`: Model type to use
- `--model-id`: Specific model identifier
- `--provider`: Model provider
- `--parallel-workers`: Number of concurrent benchmark runs (default: 4)
- `--num-examples`: Limit examples per task for testing (optional)
- `--push-answers-to-hub`: Push results to HuggingFace Hub
- `--answers-dataset`: Dataset name for answers (default: `smolagents/answers`)

**Output:**
- `output/results_{date}_{model_id}.csv`: Benchmark results
- `output/answers_{date}_{model_id}.json`: Generated answers
- Individual run directories under `runs/`

### Running the Centralized Baseline

For comparison, run the centralized agent:

```bash
python run_centralized.py \
  --model-type LiteLLMModel \
  --model-id gpt-4o \ #or another model
  --provider openai \ #or another provider
  --parallel-workers 4
```

Uses the same arguments as `run.py`.

## Features

### Decentralized Coordination

- **Message-Based Communication**: Agents communicate through a shared message store
- **Consensus Protocol**: Multiple agents must agree on the final answer through voting
- **Specialized Roles**: Each agent has specific capabilities and responsibilities
- **Parallel Execution**: Agents can work concurrently on different aspects of the problem

### Monitoring & Observability

- **Langfuse Integration**: Optional tracing and monitoring of agent interactions
- **JSON Logging**: Structured logs for debugging and analysis
- **HTML Visualization**: Convert message logs to interactive HTML reports
- **Run Tracking**: Unique run IDs for tracking individual executions

### Tool Capabilities

The agents have access to various tools:
- Python code execution
- Google search
- Web browsing (text-based)
- Document reading (PDF, DOCX, PPTX, etc.)
- Visual question answering
- File downloads
- Archive searching

## Project Structure

```
decentralized_smolagents_benchmark/
├── decentralized_agent.py       # Single question entry point
├── run.py                       # Benchmark runner (decentralized)
├── run_centralized.py          # Benchmark runner (centralized baseline)
├── requirements.txt            # Dependencies
├── scripts/                    # Supporting modules
│   ├── agents.py              # Agent definitions
│   ├── message_store.py       # Communication infrastructure
│   ├── consensus_protocol.py  # Voting mechanism
│   ├── decentralized_tools.py # Communication tools
│   ├── text_web_browser.py    # Web browsing tools
│   ├── text_inspector_tool.py # Document tools
│   ├── visual_qa.py           # Visual QA
│   ├── html_renderer.py       # HTML visualization
│   └── ...                    # Other utilities
├── runs/                      # Individual run outputs (created at runtime)
└── output/                    # Benchmark results (created at runtime)
```

## Contributing

When contributing to this project, please follow the guidelines in the root-level `AGENTS.md`:
- Follow OOP principles
- Be Pythonic: follow Python best practices and idiomatic patterns
- Write unit tests for new functionality

## License

This project is part of the smolagents repository. Please refer to the root LICENSE file for licensing information.
