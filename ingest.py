import uuid
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("mini-rag-project-1file")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def split_text(text, chunk_size=1200, overlap=15):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def ingest(text: str, source: str = "user"):
    chunks = split_text(text)

    embeddings = embed_model.encode(chunks)

    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {
                "source": source,
                "position": i,
                "text": chunks[i]
            }
        })

    index.upsert(vectors)
