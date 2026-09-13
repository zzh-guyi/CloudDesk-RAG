"""
LLM 服务 - OpenAI-compatible API 封装
"""
from typing import Optional, Dict, Any
import httpx
import logging
logger = logging.getLogger(__name__)

from config.settings import settings


class LLMResponseError(RuntimeError):
    """LLM 响应缺少可用文本或未正常结束。"""

    def __init__(
        self,
        message: str,
        *,
        finish_reason: Optional[str] = None,
        reasoning_content_length: int = 0,
        usage: Optional[Dict[str, Any]] = None,
        content_type: str = "NoneType",
    ):
        self.finish_reason = finish_reason
        self.reasoning_content_length = reasoning_content_length
        self.usage = usage
        self.content_type = content_type
        super().__init__(
            f"{message}; finish_reason={finish_reason}; "
            f"reasoning_content_length={reasoning_content_length}; "
            f"usage={usage}; content_type={content_type}"
        )


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
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用 LLM 生成回答
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式输出
            response_format: 可选的 OpenAI-compatible 响应格式
            
        Returns:
            生成的文本
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        try:
            client = self._get_client()
            response = client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices") or []
            if not choices:
                raise ValueError("LLM response does not contain choices")

            choice = choices[0]
            message_data = choice.get("message") or {}
            content = message_data.get("content")
            reasoning_content = message_data.get("reasoning_content")
            reasoning_content_length = (
                len(reasoning_content)
                if isinstance(reasoning_content, str)
                else 0
            )
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage")

            logger.debug(
                "LLM response received: finish_reason=%s, "
                "content_length=%s, reasoning_content_length=%s, usage=%s",
                finish_reason,
                len(content) if isinstance(content, str) else 0,
                reasoning_content_length,
                usage,
            )

            if finish_reason == "length" and response_format is not None:
                raise LLMResponseError(
                    "LLM response was truncated before valid completion",
                    finish_reason=finish_reason,
                    reasoning_content_length=reasoning_content_length,
                    usage=usage,
                    content_type=type(content).__name__,
                )

            if not isinstance(content, str) or not content.strip():
                raise LLMResponseError(
                    "LLM returned empty or non-text content",
                    finish_reason=finish_reason,
                    reasoning_content_length=reasoning_content_length,
                    usage=usage,
                    content_type=type(content).__name__,
                )

            return content

        except LLMResponseError as exc:
            logger.error("LLM response validation failed: %s", exc)
            raise

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if response_format is not None:
                raise
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


