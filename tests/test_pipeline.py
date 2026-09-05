"""
RAG Pipeline 集成测试
"""
import pytest
from unittest.mock import MagicMock, patch


def test_pipeline_structure():
    """测试 Pipeline 结构"""
    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    assert pipeline.rewriter is not None
    assert pipeline.router is not None
    assert pipeline.retriever is not None
    assert pipeline.reranker is not None
    assert pipeline.compressor is not None
    assert pipeline.generator is not None


def test_query_rewrite():
    """测试 Query Rewrite"""
    from app.rag.query_rewrite import QueryRewriter

    rewriter = QueryRewriter()
    result = rewriter.rewrite("我创建项目以后咋看不到人了？")

    assert "original" in result
    assert "rewritten" in result
    # 重写结果不应为空
    assert result["rewritten"] is not None


def test_query_router():
    """测试 Query Router"""
    from app.rag.query_rewrite import QueryRouter

    router = QueryRouter()

    # 测试分类
    assert router.route("如何注册账号") in [
        "user_manual", "faq", "troubleshooting",
        "product_rules", "pricing", "api_docs"
    ]
    assert router.route("E1001 是什么错误") == "troubleshooting"
    assert router.route("免费版和团队版区别") in ["pricing", "faq"]


def test_context_compressor():
    """测试 Context Compression"""
    from app.rag.context_compressor import ContextCompressor
    from app.rag.models import RetrievalResult

    compressor = ContextCompressor()
    results = [
        RetrievalResult(
            document_id="doc_1",
            title="测试文档",
            category="faq",
            source="test.md",
            content="这是一段测试内容。" * 50,
            chunk_index=0,
            rrf_score=0.5,
            rerank_score=0.8,
        )
    ]

    context = compressor.compress(results, "测试查询")
    assert isinstance(context, str)
    assert len(context) > 0
