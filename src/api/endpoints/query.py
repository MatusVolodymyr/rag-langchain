from fastapi import APIRouter, Query, HTTPException
from services.vector_store import VectorStore

router = APIRouter()


@router.get("/query/", summary="Retrieve relevant documents")
def query_rag(query: str = Query(..., description="Query string"), top_k: int = 3):
    """Retrieve the most relevant documents."""
    try:
        retrieved_docs = VectorStore.retrieve_documents(query, top_k)
        return {"query": query, "retrieved_docs": retrieved_docs}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )
