from fastapi import APIRouter, Query, HTTPException, Depends
from src.dependencies import get_vector_store
from src.services.vector_store import VectorStore
from src.core.logger import logger

router = APIRouter()


@router.get("/query/", summary="Retrieve relevant documents")
async def query_rag(
    query: str = Query(..., description="Query string"),
    top_k: int = 3,
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Retrieve the most relevant documents."""
    try:
        retrieved_docs = vector_store.retrieve_documents(query, top_k)
        logger.info("Successfully retrieved documents for query: %s", query)
        return {"query": query, "retrieved_docs": retrieved_docs}
    except Exception as e:
        logger.error("Error retrieving documents for query: %s", query, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )
