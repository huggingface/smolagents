"""
Ingestion script for RAG example.

This script loads the HuggingFace documentation dataset,
filters for transformers docs, splits them into chunks,
and serializes them to a pickle file for use by the agent.

Run this script before running app.py.
"""

import pickle

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

with open("docs_processed.pkl", "wb") as f:
    pickle.dump(docs_processed, f)

print(f"Successfully processed and saved {len(docs_processed)} document chunks to docs_processed.pkl")
