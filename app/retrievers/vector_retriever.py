"""
向量检索器 - Milvus Vector Search
"""
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)

from app.services.vector_store import get_vector_store
from app.services.embedding_service import get_embedding_service
from app.rag.models import RetrievalResult


class VectorRetriever:
    """基于 Milvus 的向量检索器"""

    def __init__(self):
        self.store = get_vector_store()
        self.embedding = get_embedding_service()

    def search(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            category: 分类过滤
            
        Returns:
            检索结果列表
        """
        try:
            query_embedding = self.embedding.encode_query(query)
            results = self.store.search(query_embedding, top_k=top_k, category=category)

            output = []
            for r in results:
                output.append(RetrievalResult(
                    document_id=r["document_id"],
                    title=r["title"],
                    category=r["category"],
                    source=r["source"],
                    content=r["content"],
                    chunk_index=r["chunk_index"],
                    retrieval_source=r["retrieval_source"],
                    vector_score=r["score"],
                ))
            return output
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []


