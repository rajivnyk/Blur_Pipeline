import chromadb
from chromadb.config import Settings

CHROMA_PATH = "RAG/chroma_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="image_embeddings",
    metadata={"hnsw:space": "cosine"}
)

def get_collection():
    return collection