"""
Embedding 服务 - BGE-M3 封装
通过 SiliconFlow OpenAI-compatible API 调用，fallback 为随机向量
"""
from typing import List, Optional
import numpy as np
import logging
logger = logging.getLogger(__name__)

from config.settings import settings


class EmbeddingService:
    """Embedding 服务，通过 SiliconFlow API 调用 BGE-M3"""

    def __init__(self):
        self._model: object = None
        self._dim = settings.embedding_dim

    def _get_model(self):
        """懒加载模型：优先 API，失败降级到 Dummy"""
        if self._model is None:
            # 正常路径：调用 Embedding API
            if settings.embedding_api_key and settings.embedding_base_url:
                try:
                    self._model = self._ApiModel(
                        api_key=settings.embedding_api_key,
                        base_url=settings.embedding_base_url,
                        model=settings.embedding_model,
                        dim=settings.embedding_dim,
                    )
                    logger.info(
                        f"Embedding via API: {settings.embedding_base_url}, model={settings.embedding_model}"
                    )
                    return self._model
                except Exception as e:
                    logger.warning(f"Embedding API initialization failed: {e}, using placeholder")
                    self._model = self._DummyModel()
                    return self._model
            else:
                logger.warning(
                    "embedding_api_key or embedding_base_url not configured, using placeholder embeddings"
                )
                self._model = self._DummyModel()
        return self._model

    class _ApiModel:
        """SiliconFlow / OpenAI-compatible Embedding API"""

        def __init__(self, api_key: str, base_url: str, model: str, dim: int):
            import httpx
            self._client = httpx.Client(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            self._model = model
            self._dim = dim

        def encode(self, texts: list, batch_size: int = 32, normalize_embeddings: bool = True):
            all_embeddings = []
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    resp = self._client.post(
                        "/embeddings",
                        json={"model": self._model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data["data"]:
                        emb = item["embedding"]
                        if normalize_embeddings:
                            norm = sum(x * x for x in emb) ** 0.5
                            if norm > 0:
                                emb = [x / norm for x in emb]
                        all_embeddings.append(emb)
                return np.array(all_embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning(
                    f"Embedding API request failed: {e}, returning placeholder vectors"
                )
                return np.random.random((len(texts), self._dim)).astype(np.float32)

    class _DummyModel:
        """占位模型：API 不可用时的降级方案"""

        def encode(self, texts, batch_size=32, normalize_embeddings=True):
            dims = settings.embedding_dim
            return np.random.random((len(texts), dims)).astype(np.float32)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表，每个向量长度为 embedding_dim
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings  # type: ignore

    def encode_query(self, query: str) -> List[float]:
        """编码单个查询"""
        return self.encode([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service