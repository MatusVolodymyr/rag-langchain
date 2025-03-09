from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
import os
import shutil
from typing import List
from src.dependencies import get_vector_store
from src.services.vector_store import VectorStore
from src.models.document import Document
from src.core.logger import logger

router = APIRouter()

UPLOAD_DIR = "uploads"


def process_and_store_file(file_path: str, vector_store):
    """
    Function to process a file and store its contents using the vector store.
    This will be run as a background task.
    """
    vector_store.add_documents_from_file(file_path)


@router.post("/upload/", summary="Upload and process documents")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    vector_store=Depends(get_vector_store),
):
    """Upload multiple files and process them asynchronously."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            # Save the file to disk
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            # Schedule the processing of the file as a background task
            background_tasks.add_task(process_and_store_file, file_path, vector_store)
            logger.info(f"Stored {len(saved_files)} files in Pinecone.")
        except Exception as e:
            logger.error("Error processing file {file}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
            )

    return {
        "message": "Files received; processing in background.",
        "files": saved_files,
    }


@router.post("/add_documents/", summary="Add documents to vector store")
def add_documents(
    documents: List[Document],
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Adds a list of validated documents to the vector store."""
    try:
        vector_store.add_documents(
            [doc.model_dump() for doc in documents]
        )  # Convert Pydantic objects to dicts
        logger.info(f"Stored {len(documents)} documents in Pinecone.")
        return {"message": "Documents added successfully"}
    except Exception as e:
        logger.error("Error reading documents", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")
