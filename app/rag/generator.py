"""
LLM 生成模块 - 生成回答 + Citation
"""
from typing import List, Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)

from app.services.llm_service import get_llm_service
from app.rag.models import RetrievalResult
from config.settings import settings


PROMPT_TEMPLATE = """你是一个 CloudDesk 企业 SaaS 产品的智能客服助手。请根据以下知识库资料回答用户问题。

## 用户问题
{query}

## 相关知识资料
{context}

## 回答要求
1. 基于知识库资料回答问题，不要编造信息
2. 如果知识库中没有相关信息，请如实说明
3. 回答要清晰、简洁、有帮助
4. 在回答末尾列出引用的资料来源

## 输出格式
请按照以下格式回答：

**回答：**
[你的回答内容]

**引用来源：**
- [来源1] 标题
- [来源2] 标题
"""


class Generator:
    """LLM 生成模块"""

    def __init__(self):
        self.llm = get_llm_service()

    def generate(
        self,
        query: str,
        rewritten_query: str,
        context: str,
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 原始查询
            rewritten_query: 重写后的查询
            context: 压缩后的上下文
            results: 检索结果
            
        Returns:
            包含 answer 和 sources 的字典
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是 CloudDesk 企业 SaaS 产品的智能客服助手，擅长根据知识库资料回答用户问题。"
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        query=rewritten_query,
                        context=context
                    )
                }
            ]

            answer = self.llm.generate(messages, max_tokens=1024)
            sources = self._extract_sources(results)

            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Generator failed: {e}")
            return {
                "answer": "抱歉，生成回答时出现错误。",
                "sources": self._extract_sources(results),
            }

    def _extract_sources(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """提取引用来源"""
        sources = []
        for r in results:
            source = {
                "document_id": r.document_id,
                "title": r.title,
                "category": r.category,
            }
            if r.rerank_score is not None:
                source["relevance_score"] = round(r.rerank_score, 4)
            elif r.rrf_score is not None:
                source["relevance_score"] = round(r.rrf_score, 4)
            sources.append(source)
        return sources


