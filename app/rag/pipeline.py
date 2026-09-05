"""
RAG Pipeline - 完整 RAG 流程编排
"""
from typing import Dict, Any, Optional
from time import time
import logging

logger = logging.getLogger(__name__)

from config.settings import settings
from app.rag.query_rewrite import get_query_rewriter, get_query_router
from app.rag.hybrid_retrieval import get_hybrid_retriever
from app.rag.reranker import Reranker
from app.rag.context_compressor import ContextCompressor
from app.rag.generator import Generator
from app.services.metrics import metrics
from app.models.schemas import RetrievalInfo, Source


class RAGPipeline:
    """RAG 完整 Pipeline"""

    def __init__(self):
        self.rewriter = get_query_rewriter()
        self.router = get_query_router()
        self.retriever = get_hybrid_retriever()
        self.reranker = Reranker()
        self.compressor = ContextCompressor()
        self.generator = Generator()

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行完整 RAG Pipeline

        Args:
            query: 用户查询
            session_id: 会话 ID
            top_k: 返回数量

        Returns:
            包含 answer, sources, retrieval_info 的字典
        """
        start_time = time()
        top_k = top_k or settings.rag_top_k

        # 保存最终结果，确保退出 measure_latency() 后再持久化 Metrics
        result = None

        with metrics.measure_latency():
            # Step 1: Query Rewrite
            rewrite_result = self.rewriter.rewrite(query)
            rewritten_query = rewrite_result["rewritten"]
            logger.info(f"Query rewrite: {query} -> {rewritten_query}")

            # Step 2: Query Router
            category = self.router.route(rewritten_query)
            metrics.record_category(category)
            logger.info(f"Query routed to category: {category}")

            # Step 3: Hybrid Retrieval
            retrieval_results = self.retriever.retrieve(
                query=rewritten_query,
                top_k=max(top_k * 4, 20),
                category=category
            )

            # Step 4: Rerank
            reranked = self.reranker.rerank(
                query=rewritten_query,
                results=retrieval_results.results,
                top_k=settings.rag_rerank_limit
            )

            # Step 5: Context Compression
            context = self.compressor.compress(
                reranked,
                rewritten_query
            )

            # Step 6: LLM Generation
            generation_result = self.generator.generate(
                query=query,
                rewritten_query=rewritten_query,
                context=context,
                results=reranked[:top_k]
            )

            # Step 7: Redis Memory
            if session_id:
                from app.services.redis_service import get_redis_service

                redis = get_redis_service()

                if redis.is_available:
                    redis.add_message(
                        session_id,
                        "user",
                        query
                    )
                    redis.add_message(
                        session_id,
                        "assistant",
                        generation_result["answer"]
                    )

            # Calculate pipeline latency
            latency_ms = (time() - start_time) * 1000

            # Build response sources
            sources = [
                Source(
                    document_id=s["document_id"],
                    title=s["title"],
                    category=s["category"],
                    relevance_score=s.get("relevance_score")
                )
                for s in generation_result["sources"]
            ]

            # Build retrieval information
            retrieval_info = RetrievalInfo(
                vector_count=retrieval_results.vector_count,
                keyword_count=retrieval_results.keyword_count,
                rrf_top_k=len(reranked),
                latency_ms=latency_ms,
                rewritten_query=rewritten_query,
                query_category=category
            )

            result = {
                "query": query,
                "rewritten_query": rewritten_query,
                "answer": generation_result["answer"],
                "sources": sources,
                "retrieval_info": retrieval_info,
                "session_id": session_id,
            }

        # ============================================================
        # 重要：
        # measure_latency() 已经退出，此时 finally 中的
        # metrics.record_latency() 已经执行完成。
        # 因此这里保存到 Redis 时，total_requests 和
        # avg_latency_ms 已经是最新值。
        # ============================================================
        metrics.save_to_redis()

        return result


# 全局单例
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()

    return _pipeline