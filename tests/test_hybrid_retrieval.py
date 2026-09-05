"""
Hybrid Retrieval 单元测试
"""
import pytest
from unittest.mock import MagicMock, patch


def test_rrf_fusion():
    """测试 RRF 融合算法"""
    from app.rag.rrf_fusion import rrf_fusion

    vector_results = [
        {"document_id": "doc_1", "score": 0.9, "title": "文档1"},
        {"document_id": "doc_2", "score": 0.8, "title": "文档2"},
    ]
    keyword_results = [
        {"document_id": "doc_2", "score": 5, "title": "文档2"},
        {"document_id": "doc_3", "score": 3, "title": "文档3"},
    ]

    result = rrf_fusion(vector_results, keyword_results, k=60)

    assert len(result) == 3
    assert result[0]["document_id"] == "doc_2"  # doc_2 在两个结果集中都有，RRF 分数最高
    assert "rrf_score" in result[0]


def test_rrf_fusion_empty():
    """测试空结果"""
    from app.rag.rrf_fusion import rrf_fusion
    result = rrf_fusion([], [], k=60)
    assert result == []


def test_hybrid_retriever_structure():
    """测试 HybridRetriever 结构"""
    from app.rag.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever()
    assert retriever.vector_retriever is not None
    assert retriever.keyword_retriever is not None


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_mocks(mock_vector_store, mock_keyword_store):
    """测试混合检索流程（Mock 模式）"""
    from app.rag.hybrid_retrieval import HybridRetriever

    # Mock vector results
    mock_vector_store.search.return_value = [
        {
            "document_id": "doc_001",
            "title": "测试文档",
            "category": "faq",
            "source": "test.md",
            "content": "测试内容",
            "chunk_index": 0,
            "score": 0.95,
            "retrieval_source": "vector",
        }
    ]

    # Mock keyword results
    mock_keyword_store.search.return_value = [
        {
            "document_id": "doc_001",
            "title": "测试文档",
            "category": "faq",
            "source": "test.md",
            "content": "测试内容",
            "score": 3,
            "retrieval_source": "keyword",
        }
    ]

    retriever = HybridRetriever()
    result = retriever.retrieve("测试查询", top_k=5)

    assert result.vector_count == 1
    assert result.keyword_count == 1
    assert len(result.results) >= 1
