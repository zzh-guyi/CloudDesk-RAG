"""
关键词检索器 - MySQL BM25/Keyword Search
"""
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)

from app.services.keyword_store import get_keyword_store
from app.rag.models import RetrievalResult


class KeywordRetriever:
    """基于 MySQL 的关键词检索器"""

    def __init__(self):
        self.store = get_keyword_store()

    def search(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            category: 分类过滤，传递给 KeywordStore
            
        Returns:
            检索结果列表
        """
        try:
            self.store.connect()
            results = self.store.search(query, top_k=top_k, category=category)

            output = []
            for r in results:
                output.append(RetrievalResult(
                    document_id=r["document_id"],
                    title=r["title"],
                    category=r["category"],
                    source=r["source"],
                    content=r["content"],
                    chunk_index=0,
                    retrieval_source=r["retrieval_source"],
                    keyword_score=r["score"],
                ))
            return output
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []


