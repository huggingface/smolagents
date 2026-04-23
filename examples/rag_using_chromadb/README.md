# RAG Example with ChromaDB

This example demonstrates how to build a RAG (Retrieval-Augmented Generation) agent using semantic search with ChromaDB on the HuggingFace documentation.

## How it works

- **`ingest.py`**: Loads and processes the HuggingFace documentation dataset, splits documents into chunks with deduplication, embeds them, and stores them in a ChromaDB vector store.
- **`app.py`**: Loads the persisted ChromaDB vector store, creates a semantic search retriever tool, and runs a `CodeAgent` to answer queries.

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

- `ingest.py`: Data loading, text splitting, embedding, and ChromaDB storage.
- `app.py`: Agent setup and execution with ChromaDB retriever tool.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.
