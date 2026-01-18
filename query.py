import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import cohere

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("mini-rag-project-1file")  # dimension=384


co = cohere.Client(os.getenv("COHERE_API_KEY"))


embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim


model_name = "google/flan-t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def split_text(text, chunk_size=800, overlap=100):
    """
    Split text into chunks of ~chunk_size words with overlap.
    Adjust chunk_size & overlap as per requirement.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def ingest(text: str, source: str = "user"):
    """
    Ingest text into Pinecone:
    1. Chunk text
    2. Generate embeddings
    3. Upsert vectors with metadata (source, position, text)
    """
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


def retrieve(query, top_k=10):
    """Retrieve top-k chunks from Pinecone"""
    query_vector = embed_model.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    docs = []
    for match in results.get("matches", []):
        text = match.get("metadata", {}).get("text", "")
        docs.append({
            "id": match.get("id"),
            "text": text,
            "metadata": match.get("metadata", {}),
            "score": match.get("score", 0)
        })
    return docs

def rerank(query, docs, top_n=5):
    """Optional: Re-rank retrieved docs using Cohere"""
    if not co or not docs:
        return docs

    documents = [d["text"] for d in docs]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n
    )

    reranked_docs = []
    for r in response.results:
        doc = docs[r.index]
        doc["rerank_score"] = r.relevance_score
        reranked_docs.append(doc)

    return reranked_docs[:top_n]



def answer(query, docs):
    """
    Generate answer using top-k retrieved & reranked docs.

    docs: list of dicts with 'text' key OR list of strings
    """
    # Handle both dicts and strings
    context_text = "\n\n".join(
        [d["text"] if isinstance(d, dict) else d for d in docs]
    )
    print(docs)
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{query}
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

