from fastapi import APIRouter, Query, HTTPException
from services.rag_pipeline import RAGPipeline
from services.vector_store import VectorStore
from core.config import settings

router = APIRouter()

rag_pipeline = RAGPipeline(VectorStore(), settings.openai_api_key)


@router.get("/generate_response/", summary="Generate response using RAG")
def generate_response(query: str = Query(..., description="User query")):
    """Generate an LLM-based answer using retrieved documents."""
    try:
        response = rag_pipeline.generate_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )
