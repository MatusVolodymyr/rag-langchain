from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from services.vector_store import VectorStore
from core.config import settings

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.post("/upload/", summary="Upload and process documents")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload multiple files and store chunks in Pinecone."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            VectorStore.add_documents_from_file(file_path)
            saved_files.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
            )

    return {"message": "Uploaded and stored successfully", "files": saved_files}
