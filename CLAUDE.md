# Atla Insights SDK Manual Instrumentation Guide for Claude

## Overview
This document provides specific instructions for manually instrumenting the Atla Insights SDK into Python codebases for AI agent monitoring. All instrumentation must be done manually using the actual SDK functions.

## Prerequisites
- Python 3.8+
- Access to Atla Insights platform (https://app.atla-ai.com)
- Environment variable `ATLA_INSIGHTS_TOKEN` with your authentication token

## Step-by-Step Manual Instrumentation

### 1. Installation
```bash
pip install atla-insights
# For LiteLLM support
pip install "atla-insights[litellm]"
2. Environment Setup
Create/update .env file:
ATLA_INSIGHTS_TOKEN=your_actual_token_here
3. Basic Configuration (Required First Step)
Add this to the very beginning of your main application file:
pythonimport os
from atla_insights import configure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))
4. Provider-Specific Manual Instrumentation
When you see OpenAI imports, add this AFTER the configure() call:
python# If you see: import openai OR from openai import OpenAI
from atla_insights import instrument_openai
instrument_openai()
When you see Anthropic imports, add this AFTER the configure() call:
python# If you see: import anthropic OR from anthropic import Anthropic
from atla_insights import instrument_anthropic
instrument_anthropic()
When you see Google GenAI imports, add this AFTER the configure() call:
python# If you see: import google.generativeai OR from google.generativeai import *
from atla_insights import instrument_google_genai
instrument_google_genai()
When you see LiteLLM imports, add this AFTER the configure() call:
python# If you see: import litellm OR from litellm import completion
from atla_insights import instrument_litellm
instrument_litellm()
5. Framework-Specific Manual Instrumentation
When you see LangChain imports, add this AFTER the configure() call:
python# If you see: import langchain OR from langchain import * OR from langchain_* import *
from atla_insights import instrument_langchain
instrument_langchain()
When you see CrewAI imports, add this AFTER the configure() call:
python# If you see: import crewai OR from crewai import *
from atla_insights import instrument_crewai
instrument_crewai()
When you see Agno imports, add this AFTER the configure() call:
python# If you see: import agno OR from agno import *
from atla_insights import instrument_agno

# If they're using OpenAI with Agno:
instrument_agno("openai")

# If they're using multiple providers with Agno:
instrument_agno(["anthropic", "openai"])  # Adjust based on what providers you see imported
When you see OpenAI Agents imports, add this AFTER the configure() call:
python# If you see: import openai_agents OR from openai_agents import *
from atla_insights import instrument_openai_agents
instrument_openai_agents()
When you see Smolagents imports, add this AFTER the configure() call:
python# If you see: import smolagents OR from smolagents import *
from atla_insights import instrument_smolagents
instrument_smolagents()
When you see MCP imports, add this AFTER the configure() call:
python# If you see MCP-related imports
from atla_insights import instrument_mcp
instrument_mcp()
6. Function-Level Manual Instrumentation
For important agent functions, wrap them with @instrument:
pythonfrom atla_insights import instrument, mark_success, mark_failure

@instrument("Description of what this function does")
def your_agent_function():
    # Your existing function code here
    try:
        result = your_existing_code()
        
        # Add success/failure marking based on your logic
        if result_meets_your_criteria:
            mark_success()
        else:
            mark_failure()
            
        return result
    except Exception as e:
        mark_failure()
        raise
For custom tools, add @tool decorator:
pythonfrom atla_insights import tool

@tool
def your_custom_tool(input_param: str) -> str:
    """Your tool description"""
    # Your existing tool code here
    return your_result
7. Complete Integration Template
Here's the complete pattern to add to the TOP of your main file:
pythonimport os
from dotenv import load_dotenv
from atla_insights import configure

# Load environment variables
load_dotenv()

# Configure Atla Insights - REQUIRED FIRST
configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))

# ONLY add the instrument calls for frameworks you actually see imported:

# If you see OpenAI imports anywhere:
from atla_insights import instrument_openai
instrument_openai()

# If you see Anthropic imports anywhere:
from atla_insights import instrument_anthropic
instrument_anthropic()

# If you see Google GenAI imports anywhere:
from atla_insights import instrument_google_genai
instrument_google_genai()

# If you see LiteLLM imports anywhere:
from atla_insights import instrument_litellm
instrument_litellm()

# If you see LangChain imports anywhere:
from atla_insights import instrument_langchain
instrument_langchain()

# If you see CrewAI imports anywhere:
from atla_insights import instrument_crewai
instrument_crewai()

# If you see Agno imports anywhere:
from atla_insights import instrument_agno
instrument_agno("openai")  # Or whatever providers they're using

# If you see OpenAI Agents imports anywhere:
from atla_insights import instrument_openai_agents
instrument_openai_agents()

# If you see Smolagents imports anywhere:
from atla_insights import instrument_smolagents
instrument_smolagents()

# If you see MCP imports anywhere:
from atla_insights import instrument_mcp
instrument_mcp()

# Rest of their existing code...
Manual Instrumentation Rules
What to Look For:

Import statements - Scan for AI framework imports
Function definitions - Look for agent/tool functions to wrap
Main execution - Find where to add the configure() call

What to Add:

Always first: configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"))
For each framework found: Add the specific instrument_*() call
For key functions: Add @instrument("description") decorator
For tools: Add @tool decorator
For results: Add mark_success() or mark_failure() calls

Where to Add:

Configuration: Very top of main file, after imports
Instrumentation: After configure(), before any AI code runs
Function decorators: Directly above function definitions
Success/failure marking: Inside functions, after determining outcome

Important Notes

No auto-detection exists - You must manually identify what frameworks are being used
Order matters - Always call configure() first, then instrument_*() calls
One-time setup - These calls should be made once at application startup
Context managers - You can also use instrumentation as context managers if needed:
pythonwith instrument_openai():
    # OpenAI calls here will be instrumented


Troubleshooting

Missing token error: Ensure ATLA_INSIGHTS_TOKEN is set in .env
No traces appearing: Verify you called configure() first
Import errors: Make sure you installed the correct packages
Multiple providers: Call instrument_*() for each provider you find

This is purely manual instrumentation - there is no automatic detection or setup.