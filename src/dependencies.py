from fastapi import Depends
from src.services.vector_store import VectorStore
from src.services.rag_pipeline import RAGPipeline
from src.core.config import settings


def get_vector_store() -> VectorStore:
    """
    Creates and returns a VectorStore instance configured with settings.
    """
    return VectorStore(
        index_name=settings.index_name,
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_rag_pipeline(
    vector_store: VectorStore = Depends(get_vector_store),
) -> RAGPipeline:
    """
    Creates and returns a RAGPipeline instance using the injected VectorStore.
    """
    return RAGPipeline(vector_store, settings.openai_api_key)
