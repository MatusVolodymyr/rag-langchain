from pydantic import BaseModel
from typing import List, Optional


class Document(BaseModel):
    id: str  # Every document must have a unique ID
    text: str  # The main text content


class QueryRequest(BaseModel):
    query: str  # The search query
    top_k: Optional[int] = 3  # Number of results (default is 3)
