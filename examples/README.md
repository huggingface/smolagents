# smolagents Examples

This directory contains example scripts demonstrating how to use smolagents. These examples show different capabilities of the library, from basic agent usage to specialized applications.

 ## Setup

 First, install smolagents with the required dependencies:

 ```bash
 pip install -r requirements.txt
 ```

## Key Examples

### Basic Agent Usage
- `agent_from_any_llm.py` - Shows how to use different LLM backends (HuggingFace, Transformers, Ollama, LiteLLM)
- `rag.py` - Basic retrieval-augmented generation using BM25 with transformers documentation
- `rag_using_chromadb.py` - Advanced RAG implementation using ChromaDB and embeddings
- `multiple_tools.py` - Comprehensive example showing how to create and use multiple API-based tools

### Code and Database Interaction
- `e2b_example.py` - Shows how to use secure code execution with E2B sandbox environment
  - Requires: `pip install "smolagents[e2b]"`
- `text_to_sql.py` - Demonstrates natural language to SQL query generation and database interaction

### Web and Document Processing
- `open_deep_research/` - A comprehensive example showing:
  - Document conversion and processing
  - Web browsing capabilities
  - Multi-step research tasks

### User Interfaces and Monitoring
- `gradio_upload.py` - Simple example of creating a Gradio UI with file upload functionality
- `inspect_multiagent_run.py` - Shows how to instrument and monitor multi-agent systems with OpenTelemetry

 ## Running Examples

 Run any example script from this `examples/` folder:

 ```bash
 python agent_from_any_llm.py
 ```
