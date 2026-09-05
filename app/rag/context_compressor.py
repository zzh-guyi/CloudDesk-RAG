"""
Context Compression 模块
过滤无关片段，保留最相关信息
"""
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)

from app.rag.models import RetrievalResult
from config.settings import settings


class ContextCompressor:
    """
    Context Compression - 压缩检索结果，保留最相关信息
    
    实现策略：
    1. 按 rerank_score / rrf_score 排序
    2. 截断过长内容
    3. 合并重叠片段
    """

    def __init__(self):
        self.max_context_length = 3000  # 最大上下文字符数

    def compress(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> str:
        """
        压缩检索结果为上下文文本
        
        Args:
            results: 检索结果列表
            query: 原始查询
            
        Returns:
            压缩后的上下文文本
        """
        if not results:
            return ""

        # 按 rerank_score 排序（优先），其次 rrf_score
        sorted_results = sorted(
            results,
            key=lambda x: x.rerank_score or x.rrf_score or 0,
            reverse=True
        )

        chunks = []
        total_length = 0

        for r in sorted_results:
            content = r.content.strip()
            if not content:
                continue

            # 截断过长内容
            if len(content) > settings.rag_chunk_size:
                content = content[:settings.rag_chunk_size] + "..."

            chunk = f"[{r.category}] {r.title}:\n{content}"
            if total_length + len(chunk) > self.max_context_length:
                break

            chunks.append(chunk)
            total_length += len(chunk)

        return "\n\n".join(chunks)


