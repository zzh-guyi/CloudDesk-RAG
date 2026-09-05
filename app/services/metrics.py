"""
指标采集服务
"""
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import time
import logging
logger = logging.getLogger(__name__)

from app.services.redis_service import get_redis_service


class MetricsCollector:
    """RAG Pipeline 指标采集"""

    def __init__(self):
        self._request_count = 0
        self._total_latency = 0.0
        self._category_counts: Dict[str, int] = {}

    @contextmanager
    def measure_latency(self):
        """测量延迟的上下文管理器"""
        start = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start) * 1000
            self.record_latency(latency_ms)

    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self._request_count += 1
        self._total_latency += latency_ms

    def record_category(self, category: str):
        """记录查询分类"""
        self._category_counts[category] = self._category_counts.get(category, 0) + 1

    def record_hit(self, hit: bool):
        """记录命中"""
        pass  # 可与 eval 结果关联

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        avg_latency = (
            self._total_latency / self._request_count
            if self._request_count > 0 else 0.0
        )
        return {
            "total_requests": self._request_count,
            "avg_latency_ms": round(avg_latency, 2),
            "category_distribution": dict(self._category_counts),
            "timestamp": datetime.now().isoformat(),
        }

    def save_to_redis(self):
        """保存到 Redis"""
        redis = get_redis_service()
        if redis.is_available:
            redis.save_metrics(self.get_summary())


metrics = MetricsCollector()


