"""
Milvus 向量存储封装
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

from config.settings import settings
from app.rag.models import DocumentChunk


class VectorStore:
    """Milvus 向量存储"""

    def __init__(self):
        self._client = None
        self._collection = None
        self._collection_name = settings.milvus_collection
        self._dim = settings.embedding_dim

    def connect(self):
        """连接到 Milvus"""
        if self._client is not None:
            return

        try:
            from pymilvus import connections

            connections.connect(
                "default",
                host=settings.milvus_host,
                port=str(settings.milvus_port),
            )

            logger.info(
                f"Connected to Milvus at "
                f"{settings.milvus_host}:{settings.milvus_port}"
            )

            self._client = connections

        except Exception as e:
            logger.exception(f"Milvus connection failed: {e}")
            self._client = None
            raise

    def ensure_collection(self):
        """
        确保 Collection 存在并处于可搜索状态。

        Evaluation 是独立 Python 进程，因此不能依赖
        FastAPI 启动时已经建立的 Milvus 连接和 Collection 状态。
        """
        try:
            from pymilvus import (
                Collection,
                FieldSchema,
                CollectionSchema,
                DataType,
                utility,
            )

            # 确保已经连接 Milvus
            if self._client is None:
                self.connect()

            # Collection 已存在
            if utility.has_collection(self._collection_name):
                self._collection = Collection(self._collection_name)

                # 确保 Collection 已加载到内存，可以执行 search
                self._collection.load()

                logger.info(
                    f"Using existing Milvus collection: "
                    f"{self._collection_name}"
                )
                logger.info(
                    f"Milvus collection loaded successfully: "
                    f"{self._collection_name}"
                )
                return

            # Collection 不存在，创建 Collection
            fields = [
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    is_primary=True,
                ),
                FieldSchema(
                    name="title",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                ),
                FieldSchema(
                    name="category",
                    dtype=DataType.VARCHAR,
                    max_length=50,
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="chunk_index",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self._dim,
                ),
            ]

            schema = CollectionSchema(
                fields,
                "CloudDesk Knowledge Base",
            )

            self._collection = Collection(
                self._collection_name,
                schema,
            )

            # 创建向量索引
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {
                    "M": 16,
                    "efConstruction": 256,
                },
            }

            self._collection.create_index(
                "embedding",
                index_params,
            )

            # 新创建的 Collection 同样需要 load
            self._collection.load()

            logger.info(
                f"Created and loaded Milvus collection: "
                f"{self._collection_name}"
            )

        except Exception as e:
            logger.exception(
                f"Milvus collection initialization failed: {e}"
            )
            self._collection = None
            raise

    def insert(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ):
        """
        插入文档分块

        Args:
            chunks: 文档分块列表
            embeddings: 对应的向量列表
        """
        if self._collection is None:
            self.connect()
            self.ensure_collection()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match "
                f"embeddings count ({len(embeddings)})"
            )

        data = [
            [c.document_id for c in chunks],
            [c.title for c in chunks],
            [c.category for c in chunks],
            [c.source for c in chunks],
            [c.content for c in chunks],
            [c.chunk_index for c in chunks],
            [str(c.metadata) for c in chunks],
            embeddings,
        ]

        # 记录插入前的 entity 数量
        pre_count = self._collection.num_entities
        logger.info(f"Before insert: {pre_count} entities in collection")

        insert_result = self._collection.insert(data)

        # 显式 flush，确保数据持久化并对查询可见
        self._collection.flush()

        # 验证写入结果
        post_count = self._collection.num_entities
        logger.info(
            f"Milvus insert result: insert_count={insert_result.insert_count}, "
            f"success_count={insert_result.succ_count}, err_count={insert_result.err_count}"
        )
        logger.info(f"After flush: {post_count} entities")

        if insert_result.err_count > 0:
            raise RuntimeError(
                f"Milvus insert failed: {insert_result.err_count} errors"
            )
        if insert_result.succ_count != len(chunks):
            raise RuntimeError(
                f"Milvus insert partial failure: expected {len(chunks)} successes, "
                f"got {insert_result.succ_count}"
            )

        # 重新 load 确保索引可用
        self._collection.load()

        logger.info(
            f"Successfully inserted {len(chunks)} chunks into Milvus"
        )
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        category: Optional[str] = None,
    ) -> List[dict]:
        """
        向量检索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            category: 可选的分类过滤

        Returns:
            检索结果列表
        """
        # 兼容独立进程，例如 Evaluation
        if self._collection is None:
            self.connect()
            self.ensure_collection()

        try:
            # 确保 Collection 已加载
            self._collection.load()

            # 检查向量维度
            if len(query_embedding) != self._dim:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected={self._dim}, "
                    f"actual={len(query_embedding)}"
                )

            expr = (
                f'category == "{category}"'
                if category
                else ""
            )

            search_params = {
                "metric_type": "IP",
                "params": {
                    "ef": 64,
                },
            }

            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[
                    "document_id",
                    "title",
                    "category",
                    "source",
                    "content",
                    "chunk_index",
                ],
            )

            output = []

            for hits in results:
                for hit in hits:
                    output.append(
                        {
                            "document_id": hit.entity.get(
                                "document_id"
                            ),
                            "title": hit.entity.get("title"),
                            "category": hit.entity.get("category"),
                            "source": hit.entity.get("source"),
                            "content": hit.entity.get("content"),
                            "chunk_index": hit.entity.get(
                                "chunk_index"
                            ),
                            "score": hit.distance,
                            "retrieval_source": "vector",
                        }
                    )

            logger.info(
                f"Vector search returned "
                f"{len(output)} results"
            )

            return output

        except Exception as e:
            # 不再把真实异常静默转换成 []
            logger.exception(
                f"Vector search failed: {e}"
            )
            raise

    def drop_collection(self):
        """删除 collection（用于测试）"""
        try:
            from pymilvus import utility

            if utility.has_collection(self._collection_name):
                utility.drop_collection(
                    self._collection_name
                )

                self._collection = None

                logger.info(
                    f"Dropped collection: "
                    f"{self._collection_name}"
                )

        except Exception as e:
            logger.exception(
                f"Failed to drop collection: {e}"
            )

    @property
    def is_available(self) -> bool:
        return self._collection is not None


# 全局单例
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取 VectorStore 全局单例"""
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store


