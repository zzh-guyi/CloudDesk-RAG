from app.logging_config import setup_logging

"""
FastAPI 应用入口
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
logger = logging.getLogger(__name__)

from app.routers import chat, knowledge, health, metrics as metrics_router
from app.services.vector_store import get_vector_store
from app.services.keyword_store import get_keyword_store
from app.services.redis_service import get_redis_service
from app.services.embedding_service import get_embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info("Starting up RAG service...")
    
    # 初始化各服务
    embedding = get_embedding_service()
    vector_store = get_vector_store()
    keyword_store = get_keyword_store()
    redis_service = get_redis_service()
    
    vector_store.connect()
    vector_store.ensure_collection()
    keyword_store.connect()
    redis_service.connect()
    
    logger.info("RAG service started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down RAG service...")
    keyword_store.close()


app = FastAPI(
    title="CloudDesk RAG Customer Service",
    description="企业 SaaS 智能客服系统 - 完整 RAG Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(metrics_router.router, prefix="/api/v1", tags=["metrics"])


@app.get("/")
async def root():
    return {
        "service": "CloudDesk RAG Customer Service",
        "version": "1.0.0",
        "docs": "/docs"
    }



