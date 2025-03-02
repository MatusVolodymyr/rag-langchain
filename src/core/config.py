from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_environment: str
    openai_api_key: str
    index_name: str

    class Config:
        env_file = ".env"


settings = Settings()
