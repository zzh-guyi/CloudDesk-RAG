"""
pytest 配置和 fixtures
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_embedding_service():
    """Mock EmbeddingService"""
    with patch("app.services.embedding_service.get_embedding_service") as mock:
        service = MagicMock()
        service.encode_query.return_value = [0.1] * 1024
        service.encode.return_value = [[0.1] * 1024]
        service.dimension = 1024
        mock.return_value = service
        yield service


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore"""
    with patch("app.services.vector_store.get_vector_store") as mock:
        store = MagicMock()
        store.is_available = True
        store.search.return_value = []
        mock.return_value = store
        yield store


@pytest.fixture
def mock_keyword_store():
    """Mock KeywordStore"""
    with patch("app.services.keyword_store.get_keyword_store") as mock:
        store = MagicMock()
        store.is_available = True
        store.search.return_value = []
        mock.return_value = store
        yield store


@pytest.fixture
def mock_redis_service():
    """Mock RedisService"""
    with patch("app.services.redis_service.get_redis_service") as mock:
        service = MagicMock()
        service.is_available = False
        mock.return_value = service
        yield service


@pytest.fixture
def mock_llm_service():
    """Mock LLMService"""
    with patch("app.services.llm_service.get_llm_service") as mock:
        service = MagicMock()
        service.is_available = True
        service.generate.return_value = "这是测试回答"
        mock.return_value = service
        yield service


@pytest.fixture
def sample_chunks():
    """样本分块数据"""
    return [
        {
            "document_id": "doc_0001",
            "chunk_id": "chunk_001",
            "title": "如何注册账号",
            "category": "user_manual",
            "source": "user_manual/register.md",
            "content": "注册 CloudDesk 账号需要访问官网，点击右上角免费注册...",
            "chunk_index": 0,
        },
        {
            "document_id": "doc_0002",
            "chunk_id": "chunk_002",
            "title": "E1001 错误码",
            "category": "troubleshooting",
            "source": "troubleshooting/e1001_error.md",
            "content": "E1001 错误表示认证凭据无效，通常发生在 API 调用时...",
            "chunk_index": 0,
        },
    ]
