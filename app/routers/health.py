"""
健康检查路由
"""
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.vector_store import get_vector_store
from app.services.keyword_store import get_keyword_store
from app.services.redis_service import get_redis_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    services = {}
    vector_store = get_vector_store()
    keyword_store = get_keyword_store()
    redis_service = get_redis_service()
    
    services["milvus"] = "ok" if vector_store.is_available else "unavailable"
    services["mysql"] = "ok" if keyword_store.is_available else "unavailable"
    services["redis"] = "ok" if redis_service.is_available else "unavailable"
    
    status = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(status=status, services=services)


