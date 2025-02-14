from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from typing import List
import os
import shutil
from dotenv import load_dotenv
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Set up API keys and configurations
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
UPLOAD_DIR = "uploads"

# Ensure necessary environment variables are set
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY, INDEX_NAME]):
    raise ValueError("Missing required environment variables")

# Initialize components
vector_store = VectorStore(INDEX_NAME, PINECONE_API_KEY, PINECONE_ENVIRONMENT)
rag_pipeline = RAGPipeline(vector_store, OPENAI_API_KEY)

# Create FastAPI app
app = FastAPI(title="RAG System API", version="1.0")


@app.post("/upload/", summary="Upload and process documents")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files (.txt or .pdf), process them, and store chunks in Pinecone.
    """
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)  # Ensure the upload directory exists

    saved_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            vector_store.add_documents_from_file(file_path)
            saved_files.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
            )

    return {"message": "Uploaded and stored successfully", "files": saved_files}


@app.post("/add_documents/", summary="Add documents to vector store")
def add_documents(documents: List[dict]):
    """
    Add a list of documents to the vector store.
    """
    try:
        vector_store.add_documents(documents)
        return {"message": "Documents added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


@app.get("/query/", summary="Retrieve relevant documents")
def query_rag(
    query: str = Query(
        ..., description="Query string to search for relevant documents"
    ),
    top_k: int = 3,
):
    """
    Retrieve the most relevant documents based on a user query.
    """
    try:
        retrieved_docs = vector_store.retrieve_documents(query, top_k)
        return {"query": query, "retrieved_docs": retrieved_docs}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )


@app.get("/generate_response/", summary="Generate response using RAG")
def generate_response(
    query: str = Query(..., description="User query for response generation")
):
    """
    Generate an answer using LLM and retrieved documents.
    """
    try:
        response = rag_pipeline.generate_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
