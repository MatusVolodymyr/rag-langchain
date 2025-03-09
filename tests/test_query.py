# tests/test_query.py
import pytest
from fastapi.testclient import TestClient
from fastapi import Depends, FastAPI
from src.api.endpoints import query
from src.dependencies import get_vector_store
from src.services.vector_store import VectorStore


class DummyVectorStore:
    def retrieve_documents(self, query: str, top_k: int = 3):
        # Return a fixed response for testing purposes
        return ["Document 1", "Document 2", "Document 3"]


def override_get_vector_store():
    return DummyVectorStore()


app = FastAPI()
app.include_router(query.router, prefix="/api")

app.dependency_overrides[get_vector_store] = override_get_vector_store

client = TestClient(app)


def test_query_endpoint():
    response = client.get("/api/query/", params={"query": "test query", "top_k": 3})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["query"] == "test query"
    # Check if we got our dummy response
    assert json_response["retrieved_docs"] == ["Document 1", "Document 2", "Document 3"]


if __name__ == "__main__":
    pytest.main()
