"""
知识库管理路由
"""
from fastapi import APIRouter
import logging
logger = logging.getLogger(__name__)

from app.models.schemas import IngestRequest, IngestResponse
from app.loaders.markdown_loader import KnowledgeBaseLoader
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.keyword_store import get_keyword_store
from config.settings import settings

router = APIRouter()


@router.post("/knowledge/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    触发文档入库
    
    流程：Markdown 加载 → 分块 → Embedding → 写入 Milvus + MySQL
    """
    loader = KnowledgeBaseLoader(
        base_path=request.path or settings.knowledge_base_path,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap
    )
    
    embedding = get_embedding_service()
    vector_store = get_vector_store()
    keyword_store = get_keyword_store()
    
    chunks = loader.load_and_chunk()
    
    if not chunks:
        return IngestResponse(status="no_documents", documents_loaded=0, chunks_created=0)
    
    # 生成 embeddings
    contents = [c.content for c in chunks]
    embeddings = embedding.encode(contents)
    
    # 写入 Milvus
    vector_store.insert(chunks, embeddings)
    
    # 写入 MySQL
    for chunk in chunks:
        keyword_store.insert_document(
            document_id=chunk.document_id,
            title=chunk.title,
            category=chunk.category,
            source=chunk.source,
            content=chunk.content,
            chunk_id=chunk.chunk_id
        )
    
    logger.info(f"Ingested {len(chunks)} chunks")
    return IngestResponse(
        status="ok",
        documents_loaded=len(set(c.document_id for c in chunks)),
        chunks_created=len(chunks)
    )


@router.get("/knowledge/categories")
async def list_categories():
    """查询知识库分类及文档数量"""
    from app.services.keyword_store import get_keyword_store
    store = get_keyword_store()
    if store.is_available:
        cursor = store._conn.cursor()
        cursor.execute("SELECT category, COUNT(*) FROM documents GROUP BY category")
        rows = cursor.fetchall()
        return [{"category": r[0], "document_count": r[1]} for r in rows]
    return []


