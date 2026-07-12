# RAG Example with BM25 Retrieval

This example demonstrates how to build a RAG (Retrieval-Augmented Generation) agent using lexical (BM25) search on the HuggingFace Transformers documentation.

## How it works

- **`ingest.py`**: Loads and processes the HuggingFace documentation dataset, splits documents into chunks, and saves them as a pickle file.
- **`app.py`**: Loads the pre-processed documents, creates a BM25 retriever tool, and runs a `CodeAgent` to answer queries.

> **Note**: Indexing and agent execution are kept in separate files to avoid multiprocessing errors that occur when both run in the same script.

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run ingestion** (only needed once):
   ```bash
   python ingest.py
   ```

3. **Run the agent**:
   ```bash
   python app.py
   ```

## Files

- `ingest.py`: Data loading, text splitting, and serialization.
- `app.py`: Agent setup and execution with BM25 retriever tool.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.
