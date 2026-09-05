"""
Rerank service - BGE Reranker wrapper
"""
from typing import List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class RerankerService:
    """Rerank service, wraps BGE Reranker Cross-Encoder."""

    def __init__(self):
        self._model = None
        self._device = os.getenv("RERANKER_DEVICE", "cuda")

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "Loading BGE Reranker: model=BAAI/bge-reranker-v2-m3, "
                    f"device={self._device}"
                )

                self._model = CrossEncoder(
                    "BAAI/bge-reranker-v2-m3",
                    device=self._device,
                )

                logger.info(
                    f"Reranker model loaded successfully on device={self._device}"
                )

            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using score fallback"
                )
                self._model = self._DummyReranker()

            except Exception as e:
                logger.exception(
                    f"Failed to load reranker model on device={self._device}: {e}"
                )
                self._model = self._DummyReranker()

        return self._model

    class _DummyReranker:
        """Deterministic fallback when sentence-transformers is unavailable."""

        def predict(
            self,
            pairs,
            batch_size=32,
            show_progress_bar=False,
        ):
            return [0.0] * len(pairs)

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:

        if not documents:
            return []

        try:
            model = self._get_model()

            pairs = [
                (query, doc.get("content", ""))
                for doc in documents
            ]

            # CrossEncoder 必须使用 predict()，不能使用 encode()
            scores = model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )

            ranked = sorted(
                zip(documents, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )

            return [
                (doc, float(score))
                for doc, score in ranked[:top_k]
            ]

        except Exception as e:
            logger.warning(
                f"Reranker failed: {e}, using original order"
            )

            return [
                (doc, 0.0)
                for doc in documents[:top_k]
            ]

    @property
    def is_available(self) -> bool:
        return self._model is not None


_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    global _reranker_service

    if _reranker_service is None:
        _reranker_service = RerankerService()

    return _reranker_service