import requests,os
from langchain_community.document_loaders import PyPDFium2Loader
from qdrant_client import QdrantClient,models
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, Modifier
)

from dotenv import load_dotenv
load_dotenv()

pdf_url = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001065280/5c943b10-61c2-4e9a-a175-f61e2391fa78.pdf"
pdf_path = "form_8k.pdf"

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

print("Loading your data...")
loader = PyPDFium2Loader(pdf_path)
docs = loader.load() # already chunked

client = QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY"),
)

collection_name = "hybrid"
dense_vector_name = "dense"
sparse_vector_name = "sparse"
dense_model_name = "jinaai/jina-embeddings-v2-small-en"
sparse_model_name = "Qdrant/BM25"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_vector_name: VectorParams(
                size=512,
                distance=Distance.COSINE,
                on_disk=True),
        },
        sparse_vectors_config={
            sparse_vector_name: SparseVectorParams(
                modifier=Modifier.IDF),
        }
    )
documents = []
metadata = []

for doc in docs:
    content = doc.page_content.strip()

    documents.append({
        dense_vector_name: models.Document(text=content, model=dense_model_name),
        sparse_vector_name: models.Document(text=content, model=sparse_model_name),
    })

    metadata.append({
        "content": content,
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
    })

print("Indexing your data...")
client.upload_collection(
    collection_name = collection_name,
    vectors = documents,
    payload = metadata,
)