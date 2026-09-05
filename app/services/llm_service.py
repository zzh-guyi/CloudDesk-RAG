"""
LLM 服务 - OpenAI-compatible API 封装
"""
from typing import Optional, Dict, Any
import httpx
import logging
logger = logging.getLogger(__name__)

from config.settings import settings


class LLMService:
    """LLM 服务，支持 OpenAI-compatible API"""

    def __init__(self):
        self._client: Optional[httpx.Client] = None
        self._model = settings.llm_model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=settings.openai_base_url,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=60.0
            )
        return self._client

    def generate(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> str:
        """
        调用 LLM 生成回答
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式输出
            
        Returns:
            生成的文本
        """
        try:
            client = self._get_client()
            response = client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(messages)

    def _fallback_response(self, messages: list) -> str:
        """LLM 失败时的降级响应"""
        logger.warning("Using fallback response for LLM")
        last_msg = messages[-1]["content"] if messages else "未知问题"
        return f"抱歉，我暂时无法回答这个问题。您的问题是：{last_msg}\n\n请直接查阅相关文档获取帮助。"

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    @property
    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.get("/models")
            return True
        except Exception:
            return False


# 全局单例
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


