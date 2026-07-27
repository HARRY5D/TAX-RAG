"""
Pydantic Settings for FinAssist AI.
All configuration is loaded from .env file.
"""
import os
import sys
from pathlib import Path

# Allow HF_HUB_OFFLINE to be controlled via .env (RERANKER_OFFLINE=true/false)
# Only force offline if explicitly set. Don't hard-code it so internet downloads work.
_reranker_offline = os.environ.get("RERANKER_OFFLINE", "false").lower() == "true"
if _reranker_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    # Ensure these are NOT set so downloads work when internet is available
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

# Prevent TensorFlow import crash due to Protobuf version conflict on Windows
sys.modules['tensorflow'] = None

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = Field(default="FinAssist AI")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)

    # Google / Gemini (Section Verifier — Google AI Studio key)
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-1.5-flash")          # primary: 1500 req/day free tier
    gemini_fallback_model: str = Field(default="gemini-1.5-flash-8b")  # fallback: high free-tier quota
    gemini_verify_enabled: bool = Field(default=True)

    # LangSmith
    langsmith_tracing: str = Field(default="true")
    langsmith_endpoint: str = Field(default="https://eu.api.smith.langchain.com")
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="TAX_RAG")

    # Model config
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    reranker_offline: bool = Field(default=False)  # set True to skip all downloads
    llm_temperature: float = Field(default=0.1)

    # Ollama (local offline LLM)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5-coder:7b")
    ollama_fallback_model: str = Field(default="deepseek-coder:6.7b")

    # FAISS
    faiss_index_path: str = Field(default="vectordb/faiss_index")
    top_k_retrieval: int = Field(default=20)
    top_k_rerank: int = Field(default=5)

    # Data paths
    raw_pdf_dir: str = Field(default="data/raw")
    curated_dir: str = Field(default="data/curated")
    processed_dir: str = Field(default="processed")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def abs_raw_pdf_dir(self) -> Path:
        return self.base_dir / self.raw_pdf_dir

    @property
    def abs_curated_dir(self) -> Path:
        return self.base_dir / self.curated_dir

    @property
    def abs_processed_dir(self) -> Path:
        return self.base_dir / self.processed_dir

    @property
    def abs_faiss_index_path(self) -> Path:
        return self.base_dir / self.faiss_index_path


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
