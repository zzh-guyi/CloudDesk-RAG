"""
应用配置管理
使用 Pydantic Settings 从 .env / Docker 环境变量加载配置
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============ LLM ============
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="LLM_MODEL",
    )

    # ============ Embedding ============
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_device: str = Field(
        default="cpu",
        validation_alias="EMBEDDING_DEVICE",
    )
    embedding_dim: int = Field(
        default=1024,
        validation_alias="EMBEDDING_DIM",
    )
    embedding_api_key: str = Field(
        default="",
        validation_alias="EMBEDDING_API_KEY",
    )
    embedding_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        validation_alias="EMBEDDING_BASE_URL",
    )

    # ============ Milvus ============
    milvus_host: str = Field(
        default="localhost",
        validation_alias="MILVUS_HOST",
    )
    milvus_port: int = Field(
        default=19530,
        validation_alias="MILVUS_PORT",
    )
    milvus_collection: str = Field(
        default="cloudDesk_documents",
        validation_alias="MILVUS_COLLECTION",
    )
    milvus_weight: float = Field(
        default=0.5,
        validation_alias="MILVUS_WEIGHT",
    )

    # ============ MySQL ============
    mysql_host: str = Field(
        default="localhost",
        validation_alias="MYSQL_HOST",
    )
    mysql_port: int = Field(
        default=3306,
        validation_alias="MYSQL_PORT",
    )
    mysql_user: str = Field(
        default="root",
        validation_alias="MYSQL_USER",
    )
    mysql_password: str = Field(
        default="",
        validation_alias="MYSQL_PASSWORD",
    )
    mysql_database: str = Field(
        default="rag_knowledge_base",
        validation_alias="MYSQL_DATABASE",
    )

    # ============ Redis ============
    redis_host: str = Field(
        default="localhost",
        validation_alias="REDIS_HOST",
    )
    redis_port: int = Field(
        default=6379,
        validation_alias="REDIS_PORT",
    )
    redis_db: int = Field(
        default=0,
        validation_alias="REDIS_DB",
    )
    redis_password: Optional[str] = Field(
        default=None,
        validation_alias="REDIS_PASSWORD",
    )

    # ============ RAG ============
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    rag_rrf_k: int = Field(default=60, validation_alias="RAG_RRF_K")
    rag_chunk_size: int = Field(default=500, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, validation_alias="RAG_CHUNK_OVERLAP")
    rag_rerank_top_k: int = Field(default=5, validation_alias="RAG_RERANK_TOP_K")
    rag_rerank_limit: int = Field(default=20, validation_alias="RAG_RERANK_LIMIT")

    # ============ Session ============
    session_ttl: int = Field(default=3600, validation_alias="SESSION_TTL")
    session_max_messages: int = Field(
        default=10,
        validation_alias="SESSION_MAX_MESSAGES",
    )

    # ============ Knowledge Base ============
    knowledge_base_path: str = Field(
        default="data/knowledge_base",
        validation_alias="KNOWLEDGE_BASE_PATH",
    )


settings = Settings()