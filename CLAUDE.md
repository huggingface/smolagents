# Atla Insights SDK Integration Guide

## Overview

Atla Insights is a platform for monitoring and improving AI agents. This guide provides step-by-step instructions for integrating the Atla Insights SDK into your Python codebase.

## Guide for Claude Code

**What Claude Code should do:**
- Automatically detect Python environment and package manager
- Install the atla-insights package using the appropriate command
- Detect AI frameworks and LLM providers in the codebase
- Find the application entry point
- Add configuration and instrumentation code
- Update existing code with decorators and instrumentation
- Run tests and verify integration

**What the user must do:**
- Get their authentication token from https://app.atla-ai.com
- Confirm framework and LLM provider choices when asked
- Provide the token when prompted for .env file creation

## Prerequisites

- Python 3.8+
- Active Atla Insights account
- Authentication token from [Atla Insights platform](https://app.atla-ai.com)

## Step 1: Detect Your Python Environment

**Claude Code Action:** Automatically detect the package manager by checking for these files:

| File(s) Present | Package Manager | Installation Method |
|---|---|---|
| `pyproject.toml` with `[tool.poetry]` | Poetry | `poetry add` |
| `pyproject.toml` with `[tool.uv]` | uv | `uv add` |
| `Pipfile` | Pipenv | `pipenv install` |
| `environment.yml` | Conda | `conda install` or `pip install` |
| `requirements.txt` only | pip/venv | `pip install` |

**If detection fails, ask:** "Which Python package manager are you using for this project?"

## Step 2: Identify Your AI Framework

**Claude Code Action:** Automatically detect frameworks by searching for these import patterns:

### AI Agent Framework Detection

| Framework | Common Usage | Import Patterns |
|---|---|---|
| **LangChain** | Most popular, includes LangGraph | `import langchain`, `from langchain import *`, `from langchain_* import *` |
| **CrewAI** | Multi-agent collaboration | `import crewai`, `from crewai import *` |
| **Agno** | Modern agent framework | `import agno`, `from agno import *` |
| **OpenAI Agents** | OpenAI's agent framework | `from openai_agents import *` |
| **Smolagents** | HuggingFace agents | `import smolagents`, `from smolagents import *` |
| **MCP** | Model Context Protocol | MCP-related imports |
| **Custom/None** | Building your own agent | Direct LLM provider calls only |

**If detection is unclear, ask:** "Which AI agent framework are you using for this project? (LangChain, CrewAI, Agno, OpenAI Agents, Smolagents, MCP, or Custom/None)"

### LLM Provider Detection

**Claude Code Action:** Automatically detect LLM providers by searching for these import patterns:

| Provider | Common Usage | Import Patterns |
|---|---|---|
| **OpenAI** | GPT models, most popular | `import openai`, `from openai import OpenAI` |
| **Anthropic** | Claude models | `import anthropic`, `from anthropic import Anthropic` |
| **Google GenAI** | Gemini models | `import google.generativeai`, `from google.generativeai import *` |
| **LiteLLM** | Multi-provider wrapper | `import litellm`, `from litellm import completion` |
| **Multiple** | Using several providers | Multiple imports from above |

**If detection is unclear, ask:** "Which LLM provider(s) are you using? (OpenAI, Anthropic, Google GenAI, LiteLLM, or Multiple)"

## Step 3: Installation

**Claude Code Action:** Run the appropriate installation command based on detected package manager and framework.

### Using uv (Recommended for new projects)
```bash
# Basic installation
uv add atla-insights

# For LiteLLM support (if LiteLLM detected)
uv add "atla-insights[litellm]"
```

### Using Poetry
```bash
# Basic installation
poetry add atla-insights

# For LiteLLM support (if LiteLLM detected)
poetry add "atla-insights[litellm]"
```

### Using Pipenv
```bash
# Basic installation
pipenv install atla-insights

# For LiteLLM support (if LiteLLM detected)
pipenv install "atla-insights[litellm]"
```

### Using Conda
```bash
# Basic installation
pip install atla-insights

# For LiteLLM support (if LiteLLM detected)
pip install "atla-insights[litellm]"
```

### Using pip (with virtual environment)
```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Basic installation
pip install atla-insights

# For LiteLLM support (if LiteLLM detected)
pip install "atla-insights[litellm]"
```

## Step 4: Environment Configuration

**User Action Required:** Get your authentication token from https://app.atla-ai.com

**Claude Code Action:** Create a `.env` file in your project root:

```env
ATLA_INSIGHTS_TOKEN=your_actual_token_here
```

**Claude Code will ask:** "Please provide your ATLA_INSIGHTS_TOKEN from https://app.atla-ai.com"


## Step 5: Identify Your Application Entry Point

**Claude Code Action:** Automatically find the main entry point by searching for:

### Entry Point Detection

| File Name | Common Usage | What to Look For |
|---|---|---|
| **main.py** | Standard entry point | Contains `if __name__ == "__main__":` |
| **app.py** | Web applications | Flask/FastAPI app creation |
| **run.py** | Script runners | Application startup logic |
| **server.py** | Server applications | Server startup code |
| **manage.py** | Django projects | Django management commands |
| **__init__.py** | Package initialization | Package-level imports |
| **Custom file** | Your specific setup | Your main application file |

**Claude Code searches for these patterns:**
- `if __name__ == "__main__":` - This is usually your main file
- Agent initialization code
- LLM client setup
- Server startup (Flask, FastAPI, etc.)

**If detection is unclear, ask:** "What's the name of your main application file where your app starts? (main.py, app.py, run.py, server.py, or something else)"

## Step 6: Configure Atla Insights in Your Entry Point

**Claude Code Action:** Add this configuration **at the very beginning** of your main application file (identified in Step 5):

```python
import os
from dotenv import load_dotenv
from atla_insights import configure

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))
```

### Entry Point Examples

**For main.py:**
```python
import os
from dotenv import load_dotenv
from atla_insights import configure

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# Your existing imports and code
# ... rest of your main.py
```

**For app.py (Flask/FastAPI):**
```python
import os
from dotenv import load_dotenv
from atla_insights import configure

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# Then your Flask/FastAPI imports
from flask import Flask
# or
from fastapi import FastAPI

# ... rest of your app.py
```

**For server.py:**
```python
import os
from dotenv import load_dotenv
from atla_insights import configure

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST  
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# Then your server setup
# ... rest of your server.py
```

### Optional: Add Metadata

You can attach metadata to provide additional context about your application:

```python
from atla_insights import configure

metadata = {
    "environment": "dev",
    "prompt-version": "v1.4", 
    "model": "gpt-4o-2024-08-06",
    "run-id": "my-test",
}

configure(
    token=os.getenv("ATLA_INSIGHTS_TOKEN"),
    metadata=metadata,
)
```

## Step 7: Add Framework-Specific Instrumentation

**Claude Code Action:** Based on detected framework from Step 2, add the appropriate instrumentation:

#### If you chose LangChain:
```python
from atla_insights import instrument_langchain
instrument_langchain()
```

#### If you chose CrewAI:
```python
from atla_insights import instrument_crewai
instrument_crewai()
```

#### If you chose Agno:
```python
from atla_insights import instrument_agno

# If using a single LLM provider
instrument_agno("openai")  # or "anthropic", "google-genai", "litellm"

# If using multiple LLM providers
instrument_agno(["openai", "anthropic"])  # adjust based on your LLM choices
```

#### If you chose OpenAI Agents:
```python
from atla_insights import instrument_openai_agents
instrument_openai_agents()
```

#### If you chose Smolagents:
```python
from atla_insights import instrument_smolagents
instrument_smolagents()
```

#### If you chose MCP:
```python
from atla_insights import instrument_mcp
instrument_mcp()
```

#### If you chose Custom/None:
Skip framework instrumentation, just add LLM provider instrumentation below.

## Step 8: Add LLM Provider Instrumentation

**Claude Code Action:** Based on detected LLM provider from Step 2, add the appropriate instrumentation:

#### If you use OpenAI:
```python
from atla_insights import instrument_openai
instrument_openai()
```

#### If you use Anthropic:
```python
from atla_insights import instrument_anthropic
instrument_anthropic()
```

#### If you use Google GenAI:
```python
from atla_insights import instrument_google_genai
instrument_google_genai()
```

#### If you use LiteLLM:
```python
from atla_insights import instrument_litellm
instrument_litellm()
```

#### If you use Multiple providers:
Add all the instrumentation calls for each provider you use.

## Step 9: Instrument Your Functions

**Claude Code Action:** Add instrumentation decorators to important functions:

### Agent Functions

For important agent functions, add the `@instrument` decorator:

```python
from atla_insights import instrument, mark_success, mark_failure

@instrument("My agent doing its thing")
def run_my_agent() -> None:
    try:
        result = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user", 
                    "content": "What is 1 + 2? Reply with only the answer, nothing else.",
                }
            ]
        )
        response = result.choices[0].message.content
        
        # Add success/failure marking based on your criteria
        if response == "3":
            mark_success()
        else:
            mark_failure()
            
        return response
    except Exception as e:
        mark_failure()
        raise
```

### Custom Tools

For custom tools, add the `@tool` decorator:

```python
from atla_insights import tool

@tool
def my_tool(my_arg: str) -> str:
    """Process the input and return a result"""
    return f"Processed: {my_arg}"
```

## Step 10: Complete Integration Examples

**Reference:** Here are complete examples based on common framework choices:

Here are complete examples based on common framework choices:

### Example 1: LangChain + OpenAI

```python
# main.py
import os
from dotenv import load_dotenv
from atla_insights import configure, instrument, instrument_langchain, instrument_openai, mark_success, mark_failure, tool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# Instrument based on your choices from Step 2
instrument_langchain()  # Framework choice
instrument_openai()     # LLM provider choice

# Initialize your components
llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["query"],
    template="Answer this question: {query}"
)
chain = LLMChain(llm=llm, prompt=prompt)

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

@instrument("Main agent workflow")
def run_agent(user_input: str) -> str:
    try:
        # Use your instrumented tool
        search_results = search_web(user_input)
        
        # Use LangChain (automatically instrumented)
        response = chain.run(query=f"Based on: {search_results}, respond to: {user_input}")
        
        # Mark success/failure based on your criteria
        if response and len(response) > 0:
            mark_success()
        else:
            mark_failure()
            
        return response
        
    except Exception as e:
        mark_failure()
        raise

if __name__ == "__main__":
    response = run_agent("What's the weather like?")
    print(response)
```

### Example 2: CrewAI + Anthropic

```python
# app.py
import os
from dotenv import load_dotenv
from atla_insights import configure, instrument, instrument_crewai, instrument_anthropic, mark_success, mark_failure, tool
from crewai import Agent, Task, Crew
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# Instrument based on your choices from Step 2
instrument_crewai()     # Framework choice
instrument_anthropic()  # LLM provider choice

# Initialize your components
client = Anthropic()

@tool
def analyze_data(data: str) -> str:
    """Analyze the provided data"""
    return f"Analysis of: {data}"

@instrument("CrewAI workflow")
def run_crew(task_description: str) -> str:
    try:
        # Define your CrewAI setup
        agent = Agent(
            role="Data Analyst",
            goal="Analyze data and provide insights",
            backstory="Expert in data analysis"
        )
        
        task = Task(
            description=task_description,
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task]
        )
        
        # Run the crew (automatically instrumented)
        result = crew.kickoff()
        
        # Mark success/failure
        if result:
            mark_success()
        else:
            mark_failure()
            
        return result
        
    except Exception as e:
        mark_failure()
        raise

if __name__ == "__main__":
    response = run_crew("Analyze the sales data trends")
    print(response)
```

### Example 3: Custom Agent + Multiple LLM Providers

```python
# server.py
import os
from dotenv import load_dotenv
from atla_insights import configure, instrument, instrument_openai, instrument_anthropic, mark_success, mark_failure, tool
from openai import OpenAI
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# Instrument multiple LLM providers
instrument_openai()     # First provider
instrument_anthropic()  # Second provider

# Initialize clients
openai_client = OpenAI()
anthropic_client = Anthropic()

@tool
def choose_model(task_type: str) -> str:
    """Choose the best model for the task"""
    if task_type == "creative":
        return "anthropic"
    else:
        return "openai"

@instrument("Multi-provider agent")
def run_multi_agent(user_input: str, task_type: str) -> str:
    try:
        # Choose provider based on task
        provider = choose_model(task_type)
        
        if provider == "openai":
            response = openai_client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": user_input}]
            )
            result = response.choices[0].message.content
        else:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": user_input}]
            )
            result = response.content[0].text
        
        # Mark success/failure
        if result and len(result) > 0:
            mark_success()
        else:
            mark_failure()
            
        return result
        
    except Exception as e:
        mark_failure()
        raise

if __name__ == "__main__":
    response = run_multi_agent("Write a creative story", "creative")
    print(response)
```

## Step 11: Alternative Usage Patterns

**Reference:** Advanced patterns for specific use cases:

### Session-wide Instrumentation

You can manually enable/disable instrumentation:

```python
from atla_insights import configure, instrument_openai, uninstrument_openai

configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

instrument_openai()
# All OpenAI calls from this point will be instrumented

uninstrument_openai()
# All OpenAI calls from this point will NOT be instrumented
```

### Context-based Instrumentation

Use instrumentation only within specific contexts:

```python
from atla_insights import configure, instrument_openai

configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

with instrument_openai():
    # Only OpenAI calls within this context will be instrumented
    result = client.chat.completions.create(...)

# OpenAI calls outside the context are not instrumented
```

## Integration Checklist

Based on your setup choices:

### Environment Setup
- [ ] ✅ Detected Python package manager (Step 1)
- [ ] ✅ Identified AI framework and LLM provider (Step 2)
- [ ] ✅ Installed `atla-insights` with appropriate extras (Step 3)
- [ ] ✅ Created `.env` file with `ATLA_INSIGHTS_TOKEN` (Step 4)

### Framework Detection
- [ ] ✅ Identified AI agent framework (Step 2)
- [ ] ✅ Identified LLM provider(s) (Step 2)
- [ ] ✅ Identified application entry point (Step 5)

### Code Integration
- [ ] ✅ Added `configure()` call at the beginning of entry point file (Step 6)
- [ ] ✅ Added framework instrumentation (Step 7)
- [ ] ✅ Added LLM provider instrumentation (Step 8)
- [ ] ✅ Added `@instrument` decorators to key functions (Step 9)
- [ ] ✅ Added `@tool` decorators to custom tools (Step 9)
- [ ] ✅ Added `mark_success()`/`mark_failure()` calls (Step 9)

### Testing
- [ ] ✅ Tested that application runs without errors
- [ ] ✅ Verified traces appear in Atla Insights dashboard

## Troubleshooting

| Problem | Solution |
|---|---|
| **No traces appearing** | Verify `configure()` is called before any AI framework code in your entry point |
| **Token errors** | Check that `ATLA_INSIGHTS_TOKEN` is set in `.env` file |
| **Import errors** | Ensure packages are installed in correct virtual environment |
| **Wrong framework instrumentation** | Review Step 2 and ensure you're using the correct framework |
| **Multiple entry points** | Add `configure()` to each entry point file |

## Common Questions

### Q: I use multiple frameworks, what should I do?
A: Add instrumentation for all frameworks you use. For example:
```python
instrument_langchain()
instrument_crewai()
instrument_openai()
```

### Q: My entry point is a custom file name, will this work?
A: Yes! The key is to add the `configure()` call at the beginning of whichever file starts your application.

### Q: I'm using a web framework like Flask/FastAPI, where do I put the configuration?
A: Add it at the very beginning of your app file, before creating the Flask/FastAPI app instance.

### Q: Can I use this with existing observability tools?
A: Yes! Atla Insights is built on OpenTelemetry and is compatible with existing setups.

## OpenTelemetry Compatibility

Atla Insights is built on OpenTelemetry and is compatible with existing observability setups:

```python
from atla_insights import configure
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Add custom span processors
my_span_exporter = OTLPSpanExporter(endpoint="https://my-otel-provider/v1/traces")
my_span_processor = SimpleSpanProcessor(my_span_exporter)

configure(
    token=os.getenv("ATLA_INSIGHTS_TOKEN"),
    additional_span_processors=[my_span_processor],
)
```

## Next Steps

1. **Review your traces** in the [Atla Insights dashboard](https://app.atla-ai.com)
2. **Explore the examples** in the [SDK repository](https://github.com/atla-ai/atla-insights-sdk/tree/main/examples)
3. **Add more instrumentation** to additional functions as needed
4. **Configure metadata** to track different versions or environments

For more advanced usage patterns, visit the [official documentation](https://github.com/atla-ai/atla-insights-sdk).
