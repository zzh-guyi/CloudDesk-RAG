"""
Hybrid Retrieval - 向量检索 + 关键词检索 + RRF 融合
"""
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)

from app.retrievers.vector_retriever import VectorRetriever
from app.retrievers.keyword_retriever import KeywordRetriever
from app.rag.rrf_fusion import rrf_fusion
from app.rag.models import RetrievalResult
from config.settings import settings


class HybridRetriever:
    """Hybrid Retrieval - 向量检索 + 关键词检索 + RRF 融合"""

    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None
    ) -> RetrievalResult:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 最终返回数量
            category: 分类过滤
            
        Returns:
            RetrievalResult 包含所有检索结果
        """
        # 并行执行向量和关键词检索
        vector_results = self.vector_retriever.search(query, top_k=top_k, category=category)
        keyword_results = self.keyword_retriever.search(query, top_k=top_k, category=category)

        logger.info(
            f"Hybrid retrieval: vector={len(vector_results)}, keyword={len(keyword_results)}"
        )

        # RRF 融合
        fused = rrf_fusion(
            [r.model_dump() for r in vector_results],
            [r.model_dump() for r in keyword_results],
            k=settings.rag_rrf_k
        )

        # 转换为 RetrievalResult
        output = []
        for doc in fused[:top_k]:
            output.append(RetrievalResult(
                document_id=doc["document_id"],
                title=doc["title"],
                category=doc["category"],
                source=doc["source"],
                content=doc["content"],
                chunk_index=doc.get("chunk_index", 0),
                retrieval_source=doc.get("retrieval_source", "hybrid"),
                vector_score=doc.get("vector_score"),
                keyword_score=doc.get("keyword_score"),
                rrf_score=doc.get("rrf_score"),
            ))

        # 返回一个包含计数的特殊对象
        class HybridResult:
            def __init__(self, results, vector_count, keyword_count):
                self.results = results
                self.vector_count = vector_count
                self.keyword_count = keyword_count

        return HybridResult(output, len(vector_results), len(keyword_results))


# 全局单例
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


