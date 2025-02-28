from fastapi import FastAPI
from api.endpoints import documents, query, response
from core.config import settings

app = FastAPI(title="RAG System API", version="1.0")

# Include Routers
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(response.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
