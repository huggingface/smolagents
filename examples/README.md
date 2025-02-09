# smolagents Examples

This directory contains example scripts demonstrating how to use smolagents. These examples show different capabilities of the library, from basic agent usage to specialized applications.

 ## Setup

 First, install smolagents with the required dependencies:

 ```bash
 pip install -r requirements.txt
 ```

 ## Key Examples

 ### Basic Agent Usage
 - `rag.py` - Shows how to use retrieval-augmented generation (RAG) with a basic agent
 - `rag_using_chromadb.py` - Demonstrates RAG using ChromaDB as the vector store

 ### Code Execution
 - `e2b_example.py` - Shows how to use secure code execution with E2B sandbox environment
   - Requires: `pip install "smolagents[e2b]"`

 ### Web and Document Processing
 - `open_deep_research/` - A comprehensive example showing:
   - Document conversion and processing
   - Web browsing capabilities
   - Multi-step research tasks

 ## Running Examples

 Run any example script from this `examples/` folder:

 ```bash
 python rag_using_chromadb.py
 ```
