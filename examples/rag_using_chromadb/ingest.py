"""
Ingestion script for RAG example using ChromaDB.

This script loads the HuggingFace documentation dataset,
splits documents into chunks with deduplication,
and stores them in a ChromaDB vector store.

Run this script before running app.py.
"""

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer


knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]

## For your own PDFs, you can use the following code to load them into source_docs
# from langchain_community.document_loaders import PyPDFLoader
# pdf_directory = "pdfs"
# pdf_files = [
#     os.path.join(pdf_directory, f)
#     for f in os.listdir(pdf_directory)
#     if f.endswith(".pdf")
# ]
# source_docs = []
# for file_path in pdf_files:
#     loader = PyPDFLoader(file_path)
#     source_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split docs and keep only unique ones
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)


print("Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)")
# Initialize embeddings and ChromaDB vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory="./chroma_db")

print(f"Successfully processed and stored {len(docs_processed)} document chunks in ./chroma_db")
