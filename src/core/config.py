from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_environment: str
    openai_api_key: str
    index_name: str
    debug: bool = False
    chunk_size: int = 256
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
