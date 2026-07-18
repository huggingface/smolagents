import os
from qdrant_client import QdrantClient,models

from dotenv import load_dotenv
load_dotenv()

collection_name = "hybrid"
dense_vector_name = "dense"
sparse_vector_name = "sparse"
dense_model_name = "jinaai/jina-embeddings-v2-small-en"
sparse_model_name = "Qdrant/BM25"

client = QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY"),
)

query = "Warner bros share price"
prefetch = [
    models.Prefetch(
        query=models.Document(text=query, model=dense_model_name),
        using=dense_vector_name
    ),
    models.Prefetch(
        query=models.Document(text=query, model=sparse_model_name),
        using=sparse_vector_name,
    ),
]   
relevant_docs = client.query_points(
        collection_name = collection_name,
        query= models.FusionQuery(fusion=models.Fusion.RRF), # apply RRF for Hybrid Search results
        prefetch=prefetch,
        with_payload = True,
        limit=3
).points


context = [point.payload["content"] for point in relevant_docs]
print("/n/n".join(context))