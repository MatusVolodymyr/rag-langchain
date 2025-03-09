from fastapi import APIRouter, Query, HTTPException, Depends
from src.dependencies import get_rag_pipeline
from src.services.rag_pipeline import RAGPipeline
from src.core.logger import logger

router = APIRouter()


@router.get("/generate_response/", summary="Generate response using RAG")
async def generate_response(
    query: str = Query(..., description="User query"),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Generate an LLM-based answer using retrieved documents."""
    try:
        response = rag_pipeline.generate_response(query)
        logger.info(f"Succesfuly generated response for query: {query}")
        return {"query": query, "response": response}
    except Exception as e:
        logger.error(f"Error generating response for query: {query}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )
