import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import QDRANT_CLOUD_URL, QDRANT_API_KEY, COLLECTION_NAME

def initialize_qdrant_collection(qdrant, collection_name, vector_size):
    collections = qdrant.get_collections().collections
    existing = any(c.name == collection_name for c in collections)
    if not existing:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def upload_embeddings(qdrant, collection_name, embeddings, texts, batch_size=100):
    for i in range(0, len(embeddings), batch_size):
        batch_points = [
            PointStruct(
                id=i+j,
                vector=embeddings[i+j].tolist(),
                payload={"text": texts[i+j]}
            )
            for j in range(len(embeddings[i:i+batch_size]))
        ]
        qdrant.upsert(collection_name=collection_name, points=batch_points)

def embed_and_store(text_chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array(model.encode(text_chunks))
    qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
    initialize_qdrant_collection(qdrant, COLLECTION_NAME, embeddings.shape[1])
    upload_embeddings(qdrant, COLLECTION_NAME, embeddings, text_chunks)
    print(f"Uploaded {len(text_chunks)} chunks to Qdrant collection '{COLLECTION_NAME}'.")