from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # Vector Store
    faiss_index_path: str = "./data/vectorstore/faiss_index"

    # Database
    database_url: str = "sqlite:///./data/rag_research.db"

    # File Storage
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50

    # RAG Pipeline
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = ["http://localhost:3000"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()