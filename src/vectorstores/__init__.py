"""Vector stores. `get_vector_store()` builds one by name (defaults from Settings)."""
from src.vectorstores.base import SearchResult, VectorStore

AVAILABLE_STORES = ("chroma", "faiss", "qdrant", "pgvector")


def get_vector_store(name: str | None = None, dimension: int = 384, collection: str | None = None) -> VectorStore:
    from src.config import get_settings

    s = get_settings()
    name = name or s.vector_store
    collection = collection or s.collection_name
    if name == "chroma":
        from src.vectorstores.chroma_store import ChromaStore

        return ChromaStore(collection, dimension, data_dir=s.data_dir)
    if name == "faiss":
        from src.vectorstores.faiss_store import FaissStore

        return FaissStore(collection, dimension, data_dir=s.data_dir)
    if name == "qdrant":
        from src.vectorstores.qdrant_store import QdrantStore

        return QdrantStore(collection, dimension, url=s.qdrant_url, api_key=s.qdrant_api_key.get_secret_value())
    if name == "pgvector":
        from src.vectorstores.pgvector_store import PgVectorStore

        return PgVectorStore(collection, dimension, dsn=s.pg_dsn)
    raise ValueError(f"Unknown vector store: {name!r}. Choose from {AVAILABLE_STORES}")


__all__ = ["AVAILABLE_STORES", "SearchResult", "VectorStore", "get_vector_store"]
