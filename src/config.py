from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "set-your-api-key"
    ELASTICSEARCH_HOST: str = "http://localhost:9200"
    INDEX_NAME: str = "utrains-qa"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    VECTOR_DIMENSION: int = 1536
    STREAMLIT_PORT: int = 8501

    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env")

settings = Settings()
