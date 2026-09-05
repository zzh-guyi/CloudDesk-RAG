"""
Rerank 模块 - 对 RRF 融合结果进行重排序
"""
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)

from app.services.reranker_service import get_reranker_service
from app.rag.models import RetrievalResult
from config.settings import settings


class Reranker:
    """Rerank 模块，使用 Cross-Encoder 重排序"""

    def __init__(self):
        self.service = get_reranker_service()

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: RRF 融合后的结果
            top_k: 返回数量，默认使用配置
            
        Returns:
            重排序后的结果
        """
        if top_k is None:
            top_k = settings.rag_rerank_top_k

        if not results:
            return []

        # 转换为 reranker 需要的格式
        docs = [
            {
                "document_id": r.document_id,
                "title": r.title,
                "category": r.category,
                "source": r.source,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "rrf_score": r.rrf_score,
            }
            for r in results
        ]

        try:
            ranked = self.service.rerank(query, docs, top_k=top_k)
            output = []
            for doc, score in ranked:
                output.append(RetrievalResult(
                    document_id=doc["document_id"],
                    title=doc["title"],
                    category=doc["category"],
                    source=doc["source"],
                    content=doc["content"],
                    chunk_index=doc["chunk_index"],
                    retrieval_source=doc.get("retrieval_source", "rrf"),
                    rrf_score=doc.get("rrf_score"),
                    rerank_score=score,
                ))
            return output
        except Exception as e:
            logger.warning(f"Rerank failed: {e}, using RRF order as fallback")
            return results[:top_k]


