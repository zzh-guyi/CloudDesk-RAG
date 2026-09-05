"""
RAG Pipeline 内部数据模型
"""
from typing import List, Optional, Any
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """文档分块"""
    document_id: str
    chunk_id: str
    title: str
    category: str
    source: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = {}


class RetrievalResult(BaseModel):
    """检索结果"""
    document_id: str
    title: str
    category: str
    source: str
    content: str
    chunk_index: int
    retrieval_source: str  # "vector" | "keyword"
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RAGContext(BaseModel):
    """RAG 上下文"""
    query: str
    rewritten_query: str
    query_category: Optional[str] = None
    chunks: List[DocumentChunk] = []
    retrieved_docs: List[RetrievalResult] = []
    compressed_context: str = ""


