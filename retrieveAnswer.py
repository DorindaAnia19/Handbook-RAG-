import requests
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import QDRANT_CLOUD_URL, QDRANT_API_KEY, COLLECTION_NAME

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)

# --- Retrieve Top-k Relevant Chunks from Qdrant ---
def retrieve_chunks(query, top_k=5):
    query_vector = embedding_model.encode([query])[0]
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k
    )
    return [r.payload["text"] for r in results if "text" in r.payload]

# --- Generate Answer using Ollama (gemma 3) ---
def generate_answer(query, retrieved_text):
    joined_text = "\n".join(retrieved_text)
    prompt = f"""You are an AI assistant that answers based on a document.

Document Excerpts:
{joined_text}

Question:
{query}

Instructions:
- Summarize in 2–3 sentences.
- Be factual and concise.
- If unclear, respond with: 'The document does not provide a clear answer to this question.'

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi3:mini", "prompt": prompt},
        stream=True  
    )

    full_answer = ""

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full_answer += data["response"]
            if data.get("done", False):
                break

    return full_answer.strip()
