"""
指标查询路由
"""
import ast

from fastapi import APIRouter
from app.models.schemas import MetricsResponse
from app.services.metrics import metrics
from app.services.redis_service import get_redis_service

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """查询运行指标"""
    summary = metrics.get_summary()

    # 尝试从 Redis 获取历史数据
    redis = get_redis_service()
    if redis.is_available:
        redis_metrics = redis.get_metrics()
        if redis_metrics:
            summary.update(redis_metrics)

    # Redis 可能将 category_distribution 以字符串形式返回，
    # 这里统一转换为 dict，避免 Pydantic 校验失败。
    category_distribution = summary.get("category_distribution", {})

    if isinstance(category_distribution, str):
        try:
            category_distribution = ast.literal_eval(category_distribution)
        except (ValueError, SyntaxError):
            category_distribution = {}

    if not isinstance(category_distribution, dict):
        category_distribution = {}

    return MetricsResponse(
        total_requests=summary.get("total_requests", 0),
        avg_latency_ms=summary.get("avg_latency_ms", 0.0),
        hit_rate=0.0,  # 待 Evaluation 模块完成后填充
        category_distribution=category_distribution,
    )