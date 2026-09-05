"""
文档入库脚本
用法: python scripts/ingest.py [--path data/knowledge_base]
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logger = logging.getLogger(__name__)

from app.loaders.markdown_loader import KnowledgeBaseLoader
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.keyword_store import get_keyword_store
from config.settings import settings


def ingest(knowledge_base_path: str = None):
    """执行文档入库"""
    kb_path = knowledge_base_path or settings.knowledge_base_path

    logger.info(f"Starting ingestion from: {kb_path}")

    # 加载和分块
    loader = KnowledgeBaseLoader(
        base_path=kb_path,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap
    )
    chunks = loader.load_and_chunk()

    if not chunks:
        logger.warning("No chunks to ingest")
        return

    logger.info(f"Loaded {len(chunks)} chunks")

    # 生成 embeddings
    logger.info("Generating embeddings...")
    embedding_service = get_embedding_service()
    contents = [c.content for c in chunks]
    embeddings = embedding_service.encode(contents)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # 写入 Milvus
    logger.info("Writing to Milvus...")
    vector_store = get_vector_store()
    vector_store.insert(chunks, embeddings)
    logger.info("Milvus write complete")

    # 写入 MySQL
    logger.info("Writing to MySQL...")
    keyword_store = get_keyword_store()
    for chunk in chunks:
        keyword_store.insert_document(
            document_id=chunk.document_id,
            title=chunk.title,
            category=chunk.category,
            source=chunk.source,
            content=chunk.content,
            chunk_id=chunk.chunk_id
        )
    logger.info("MySQL write complete")

    logger.success(f"Ingestion complete: {len(chunks)} chunks from {len(set(c.document_id for c in chunks))} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge base ingestion script")
    parser.add_argument("--path", default=None, help="Knowledge base path")
    args = parser.parse_args()

    ingest(args.path)

