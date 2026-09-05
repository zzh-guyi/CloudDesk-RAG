"""
Redis Session 管理
"""
from typing import List, Optional, Any, Dict
from datetime import datetime
import logging
import json

from config.settings import settings

logger = logging.getLogger(__name__)


class RedisService:
    """Redis 会话管理服务"""

    def __init__(self):
        self._client = None

    def connect(self):
        """连接 Redis"""
        try:
            import redis
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self._client.ping()
            logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, session memory will be disabled")
            self._client = None

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史"""
        if self._client is None:
            return []
        try:
            key = f"rag:session:{session_id}"
            messages = self._client.lrange(key, 0, -1)
            return [msg for msg in messages if msg]
        except Exception as e:
            logger.warning(f"Failed to get session: {e}")
            return []

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话历史"""
        if self._client is None:
            return
        try:
            key = f"rag:session:{session_id}"
            ts = datetime.now().isoformat()
            message = json.dumps({"role": role, "content": content, "timestamp": ts})
            self._client.rpush(key, message)
            self._client.expire(key, settings.session_ttl)
            while self._client.llen(key) > settings.session_max_messages * 2:
                self._client.ltrim(key, 0, settings.session_max_messages * 2 - 1)
        except Exception as e:
            logger.warning(f"Failed to add message: {e}")

    def get_context(self, session_id: str) -> str:
        """获取会话上下文"""
        messages = self.get_session(session_id)
        if not messages:
            return ""
        context_parts = []
        for msg in messages[-10:]:
            try:
                data = json.loads(msg)
                role = data.get("role", "")
                text = data.get("content", "")
                prefix = "User: " if role == "user" else "Assistant: "
                context_parts.append(prefix + text)
            except (json.JSONDecodeError, KeyError):
                context_parts.append(msg)
        return "\n".join(context_parts)

    def save_metrics(self, metrics: Dict[str, Any]):
        """保存指标"""
        if self._client is None:
            return

        try:
            key = "rag:metrics:" + datetime.now().strftime("%Y-%m-%d")

            for k, v in metrics.items():
                if isinstance(v, (dict, list)):
                    value = json.dumps(v, ensure_ascii=False)
                else:
                    value = str(v)

                self._client.hset(key, k, value)

            self._client.expire(key, 86400)

        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        if self._client is None:
            return {}

        try:
            key = "rag:metrics:" + datetime.now().strftime("%Y-%m-%d")
            raw_metrics = self._client.hgetall(key)

            metrics = {}

            for k, v in raw_metrics.items():

                if k == "category_distribution":
                    try:
                        metrics[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        metrics[k] = {}

                elif k == "total_requests":
                    try:
                        metrics[k] = int(v)
                    except (ValueError, TypeError):
                        metrics[k] = 0

                elif k == "avg_latency_ms":
                    try:
                        metrics[k] = float(v)
                    except (ValueError, TypeError):
                        metrics[k] = 0.0

                else:
                    metrics[k] = v

            return metrics

        except Exception as e:
            logger.warning(f"Failed to get metrics: {e}")
            return {}

    @property
    def is_available(self) -> bool:
        return self._client is not None


_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service

