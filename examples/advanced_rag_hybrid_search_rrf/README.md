## Overview:

Build a reliable RAG pipeline for Finance document with combine Dense and Sparse Vectors on the knowledge base
and then reorder the retrieved documents using Reciprocal Rank Fusion using Qdrant

### Tech Stack: 

- Smol Agents, HuggingFace, Qdrant, LangChain, LiteLLM
- Document Loading: PyPDFium2Loader. As per this research:https://arxiv.org/pdf/2410.09871. pypdfium2 works decent on Finance documents
- For Embedding: FastEmbed. Runs HF models in ONNX runtime. 

### Usage

1. Installation

```bash
pip install qdrant-client fastembed pypdfium2 langchain-community
pip install 'smolagents[litellm]'
pip install "smolagents[toolkit]"
```

2. Set up environment variables:

Check .env.example_custom for the required environment variables
Create a .env file with the necessary configuration

```bash
cp .env.example .env 
```

3. Ingest data: ingest.py
4. Execute Agent: app.py

5. Results Snapshots:

