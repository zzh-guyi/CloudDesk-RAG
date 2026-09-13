"""
LLM-as-a-Judge 基础模块。

该模块复用项目现有的 LLMService，只负责调用、JSON 解析、结果校验和重试。
"""

import hashlib
import json
import logging
import math
from typing import Any, Dict, Optional, Protocol

try:
    import httpx
except ImportError:  # 允许只导入 Judge 模块时进行静态检查
    httpx = None

from app.eval.judge_prompts import (
    build_answer_relevancy_prompt,
    build_citation_accuracy_prompt,
    build_faithfulness_prompt,
)


logger = logging.getLogger(__name__)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True

    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return True

    return exc.__class__.__name__ in {
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }


def _is_length_error(exc: Exception) -> bool:
    return (
        getattr(exc, "finish_reason", None) == "length"
        or "finish_reason=length" in str(exc)
    )


class JudgeCache(Protocol):
    """后续 Cache 实现的轻量接口。"""

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        ...

    def set(self, key: str, value: Dict[str, Any]) -> None:
        ...


class JudgeOutputError(ValueError):
    """Judge 输出无法解析或不符合预期格式。"""


class LLMJudge:
    """使用现有 LLMService 对生成结果进行结构化评审。"""

    _SYSTEM_PROMPT = (
        "你是严格的 RAG 评价器。必须遵守用户消息中的字段、评分标准和 JSON 格式，"
        "只返回一个合法 JSON 对象。"
    )

    def __init__(
        self,
        llm_service: Any = None,
        max_retries: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache: Optional[JudgeCache] = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if llm_service is None:
            from app.services.llm_service import get_llm_service

            llm_service = get_llm_service()

        self._llm_service = llm_service
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache = cache

    def judge_faithfulness(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> Dict[str, Any]:
        """评价回答中的事实是否由 context 支持。"""

        prompt = build_faithfulness_prompt(
            question=question,
            context=context,
            answer=answer,
        )
        return self._judge(
            metric="faithfulness",
            prompt=prompt,
            cache_payload={
                "question": question,
                "context": context,
                "answer": answer,
            },
        )

    def judge_answer_relevancy(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """评价回答是否直接且完整地回应问题。"""

        prompt = build_answer_relevancy_prompt(
            question=question,
            answer=answer,
        )
        return self._judge(
            metric="answer_relevancy",
            prompt=prompt,
            cache_payload={
                "question": question,
                "answer": answer,
                "context": context or "",
            },
        )

    def judge_citation_accuracy(
        self,
        question: str,
        context: str,
        answer: str,
        sources: Any,
    ) -> Dict[str, Any]:
        """评价回答引用的有效性，以及引用对陈述的支持程度。"""

        prompt = build_citation_accuracy_prompt(
            question=question,
            context=context,
            answer=answer,
            sources=sources,
        )
        return self._judge(
            metric="citation_accuracy",
            prompt=prompt,
            cache_payload={
                "question": question,
                "context": context,
                "answer": answer,
                "sources": sources,
            },
        )

    def _judge(
        self,
        metric: str,
        prompt: str,
        cache_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        cache_key = self._make_cache_key(metric, cache_payload)
        cached = self._read_cache(cache_key)

        if cached is not None:
            result = dict(cached)
            result["cache_hit"] = True
            result["attempts"] = 0
            return result

        last_error = "unknown error"
        last_error_type = "unknown"
        last_exception: Optional[Exception] = None
        current_max_tokens = self._max_tokens

        for attempt in range(1, self._max_retries + 2):
            try:
                content = self._call_llm(
                    prompt,
                    max_tokens=current_max_tokens,
                )
                payload = self._parse_json_object(content)
                result = self._validate_result(metric, payload)
                result.update(
                    {
                        "status": "success",
                        "attempts": attempt,
                        "cache_hit": False,
                    }
                )
                self._write_cache(cache_key, result)
                return result

            except JudgeOutputError as exc:
                last_error_type = "invalid_output"
                last_error = str(exc)
                last_exception = exc

            except Exception as exc:
                if _is_timeout_error(exc):
                    last_error_type = "timeout"
                elif _is_length_error(exc):
                    last_error_type = "length"
                else:
                    last_error_type = "llm_error"
                last_error = str(exc) or exc.__class__.__name__
                last_exception = exc

            if attempt <= self._max_retries:
                logger.warning(
                    "LLM judge %s attempt %s failed (%s, max_tokens=%s): "
                    "%s; retrying",
                    metric,
                    attempt,
                    last_error_type,
                    current_max_tokens,
                    last_error,
                )

                if (
                    last_exception is not None
                    and _is_length_error(last_exception)
                ):
                    next_max_tokens = max(
                        current_max_tokens * 2,
                        current_max_tokens + 1024,
                    )
                    logger.warning(
                        "LLM judge %s retrying with larger token budget: "
                        "%s -> %s",
                        metric,
                        current_max_tokens,
                        next_max_tokens,
                    )
                    current_max_tokens = next_max_tokens

        logger.error(
            "LLM judge %s failed after %s attempts (%s): %s",
            metric,
            self._max_retries + 1,
            last_error_type,
            last_error,
        )

        return {
            "metric": metric,
            "status": "error",
            "score": None,
            "reason": "",
            "error_type": last_error_type,
            "error": last_error,
            "attempts": self._max_retries + 1,
            "cache_hit": False,
            "max_tokens": current_max_tokens,
        }

    def _call_llm(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self._SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return self._llm_service.generate(
            messages,
            temperature=self._temperature,
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else self._max_tokens
            ),
            stream=False,
            response_format={"type": "json_object"},
        )

    def _parse_json_object(self, content: Any) -> Dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise JudgeOutputError("LLM returned empty or non-text output")

        text = content.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeOutputError(
                f"LLM output is not valid JSON: {exc.msg}"
            ) from exc

        if not isinstance(payload, dict):
            raise JudgeOutputError("LLM JSON output must be an object")

        return payload

    def _validate_result(
        self,
        metric: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = payload.get("score")

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeOutputError("score must be a number")

        score = float(score)

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise JudgeOutputError("score must be between 0.0 and 1.0")

        reason = payload.get("reason")

        if not isinstance(reason, str):
            raise JudgeOutputError("reason must be a string")

        result: Dict[str, Any] = {
            "metric": metric,
            "score": score,
            "reason": reason.strip(),
        }

        if metric == "faithfulness":
            result["unsupported_claims"] = self._validate_string_list(
                payload.get("unsupported_claims"),
                "unsupported_claims",
            )

        elif metric == "answer_relevancy":
            result["missing_aspects"] = self._validate_string_list(
                payload.get("missing_aspects"),
                "missing_aspects",
            )

        elif metric == "citation_accuracy":
            result["valid_citations"] = self._validate_citation_list(
                payload.get("valid_citations"),
                "valid_citations",
            )
            result["invalid_citations"] = self._validate_citation_list(
                payload.get("invalid_citations"),
                "invalid_citations",
            )

        else:
            raise JudgeOutputError(f"unsupported metric: {metric}")

        return result

    @staticmethod
    def _validate_string_list(value: Any, field_name: str) -> list:
        if not isinstance(value, list):
            raise JudgeOutputError(f"{field_name} must be a list")

        if not all(isinstance(item, str) for item in value):
            raise JudgeOutputError(
                f"{field_name} items must be strings"
            )

        return [item.strip() for item in value if item.strip()]

    @staticmethod
    def _validate_citation_list(value: Any, field_name: str) -> list:
        if not isinstance(value, list):
            raise JudgeOutputError(f"{field_name} must be a list")

        citations = []

        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    citations.append(text)
            elif isinstance(item, dict):
                citations.append(item)
            else:
                raise JudgeOutputError(
                    f"{field_name} items must be strings or objects"
                )

        return citations

    @staticmethod
    def _make_cache_key(
        metric: str,
        payload: Dict[str, Any],
    ) -> str:
        serialized = json.dumps(
            {
                "metric": metric,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()
        return f"llm_judge:{metric}:{digest}"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if self._cache is None:
            return None

        try:
            value = self._cache.get(key)
        except Exception as exc:
            logger.warning("LLM judge cache read failed: %s", exc)
            return None

        if not isinstance(value, dict) or value.get("status") != "success":
            return None

        return value

    def _write_cache(
        self,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        if self._cache is None:
            return

        try:
            self._cache.set(key, dict(value))
        except Exception as exc:
            logger.warning("LLM judge cache write failed: %s", exc)
