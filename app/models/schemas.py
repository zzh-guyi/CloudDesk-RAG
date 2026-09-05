"""
API 请求/响应数据模型
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID，用于多轮对话")
    top_k: Optional[int] = Field(None, description="返回结果数量，默认使用配置")


class Source(BaseModel):
    """引用来源"""
    document_id: str
    title: str
    category: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None


class RetrievalInfo(BaseModel):
    """检索信息"""
    vector_count: int = 0
    keyword_count: int = 0
    rrf_top_k: int = 0
    latency_ms: float = 0.0
    rewritten_query: Optional[str] = None
    query_category: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: List[Source] = []
    retrieval_info: RetrievalInfo = Field(default_factory=RetrievalInfo)
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    """文档入库请求"""
    path: str = Field(..., description="知识库目录路径")


class IngestResponse(BaseModel):
    """文档入库响应"""
    status: str
    documents_loaded: int
    chunks_created: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    services: dict[str, str]


class MetricsResponse(BaseModel):
    """指标响应"""
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    hit_rate: float = 0.0
    category_distribution: dict[str, int] = Field(default_factory=dict)


class KnowledgeCategory(BaseModel):
    """知识库分类"""
    category: str
    document_count: int

