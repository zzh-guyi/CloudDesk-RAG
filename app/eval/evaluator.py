"""
Retrieval Evaluator - 评估四种检索策略
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if str(_PROJECT_ROOT / "local_packages") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "local_packages"))

import numpy as np

logger = logging.getLogger(__name__)

from app.eval.metrics import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    compute_query_metrics,
    aggregate_metrics,
)

from app.rag.query_rewrite import get_query_rewriter
from app.retrievers.vector_retriever import VectorRetriever
from app.retrievers.keyword_retriever import KeywordRetriever
from app.rag.hybrid_retrieval import get_hybrid_retriever
from app.rag.reranker import Reranker
from app.services.vector_store import get_vector_store


def load_evaluation_dataset(
    path: str,
) -> List[Dict[str, Any]]:
    """加载 Evaluation 数据集"""

    dataset = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            entry = json.loads(line)
            dataset.append(entry)

    logger.info(
        f"Loaded {len(dataset)} evaluation queries "
        f"from {path}"
    )

    return dataset


class RewriteCache:
    """Query Rewrite 缓存"""

    def __init__(self, cache_path: str):
        self._cache_path = cache_path
        self._rewrites: Dict[str, str] = {}

        self._load()

    def _load(self):
        if os.path.exists(self._cache_path):
            try:
                with open(
                    self._cache_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = json.load(f)

                self._rewrites = data.get(
                    "rewrites",
                    {},
                )

                logger.info(
                    f"Loaded rewrite cache: "
                    f"{len(self._rewrites)} entries"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to load rewrite cache: "
                    f"{e}, starting fresh"
                )

                self._rewrites = {}

    def _save(self):
        data = {
            "last_run": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "rewrites": self._rewrites,
        }

        with open(
            self._cache_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            f"Saved rewrite cache: "
            f"{len(self._rewrites)} entries"
        )

    def get(self, query: str) -> str:
        if query in self._rewrites:
            return self._rewrites[query]

        rewriter = get_query_rewriter()

        result = rewriter.rewrite(query)

        rewritten = result["rewritten"]

        self._rewrites[query] = rewritten

        self._save()

        logger.info(
            f"Rewrote query: "
            f"{query} -> {rewritten}"
        )

        return rewritten


def run_vector_search(
    query: str,
    top_k: int,
) -> List[str]:
    """执行 Vector Search"""

    retriever = VectorRetriever()

    results = retriever.search(
        query,
        top_k=top_k,
    )

    return [
        r.document_id
        for r in results
    ]


def run_keyword_search(
    query: str,
    top_k: int,
) -> List[str]:
    """执行 Keyword Search"""

    retriever = KeywordRetriever()

    results = retriever.search(
        query,
        top_k=top_k,
    )

    return [
        r.document_id
        for r in results
    ]


def run_hybrid_rrf(
    query: str,
    top_k: int,
) -> List[str]:
    """执行 Hybrid Retrieval + RRF"""

    retriever = get_hybrid_retriever()

    hybrid_result = retriever.retrieve(
        query,
        top_k=top_k,
    )

    return [
        r.document_id
        for r in hybrid_result.results
    ]


def run_hybrid_rrf_rerank(
    query: str,
    top_k: int,
) -> List[str]:
    """执行 Hybrid Retrieval + RRF + Rerank"""

    hybrid_retriever = get_hybrid_retriever()

    hybrid_result = hybrid_retriever.retrieve(
        query,
        top_k=top_k,
    )

    reranker = Reranker()

    reranked = reranker.rerank(
        query,
        hybrid_result.results,
        top_k=top_k,
    )

    return [
        r.document_id
        for r in reranked
    ]


STRATEGIES = {
    "vector": run_vector_search,
    "keyword": run_keyword_search,
    "hybrid_rrf": run_hybrid_rrf,
    "hybrid_rrf_rerank": run_hybrid_rrf_rerank,
}

KS = [1, 3, 5, 10]


class RetrievalEvaluator:
    """Retrieval Evaluation 主类"""

    def __init__(
        self,
        dataset_path,
        cache_path,
        top_k=20,
        output_dir="eval_results",
    ):
        self.dataset = load_evaluation_dataset(
            dataset_path
        )

        self.cache = RewriteCache(
            cache_path
        )

        self.top_k = top_k

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==================================================
        # Evaluation 是独立 Python 进程。
        #
        # FastAPI 启动时建立的 Milvus 连接不能被这里复用，
        # 因此 Evaluation 启动时必须显式初始化 VectorStore。
        # ==================================================

        logger.info(
            "Initializing Milvus for Evaluation..."
        )

        vector_store = get_vector_store()

        vector_store.connect()

        vector_store.ensure_collection()

        if not vector_store.is_available:
            raise RuntimeError(
                "Milvus VectorStore initialization failed: "
                "collection is unavailable"
            )

        logger.info(
            "Milvus initialization for Evaluation completed"
        )

    def _run_strategy(
        self,
        strategy_name,
        query,
    ):
        start = time.perf_counter()

        result_fn = STRATEGIES[
            strategy_name
        ]

        doc_ids = result_fn(
            query,
            top_k=self.top_k,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return (
            doc_ids,
            latency_ms,
        )

    def evaluate_query(
        self,
        entry,
    ):
        query = entry["query"]

        relevant_doc_ids = entry[
            "relevant_doc_ids"
        ]

        category = entry.get(
            "category",
            "unknown",
        )

        difficulty = entry.get(
            "difficulty",
            "unknown",
        )

        # Query Rewrite
        rewritten_query = self.cache.get(
            query
        )

        strategy_results = {}

        # 执行四种 Retrieval Strategy
        for strategy_name in STRATEGIES:

            doc_ids, latency_ms = (
                self._run_strategy(
                    strategy_name,
                    rewritten_query,
                )
            )

            metrics = compute_query_metrics(
                doc_ids,
                relevant_doc_ids,
                KS,
            )

            strategy_results[
                strategy_name
            ] = {
                "retrieved_doc_ids": doc_ids,
                "latency_ms": round(
                    latency_ms,
                    2,
                ),
                "metrics": metrics,
            }

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "relevant_doc_ids": relevant_doc_ids,
            "category": category,
            "difficulty": difficulty,
            "strategies": strategy_results,
        }

    def run(self):
        """执行整个 Evaluation"""

        all_results = []

        for i, entry in enumerate(
            self.dataset
        ):
            logger.info(
                f"Evaluating query "
                f"{i + 1}/{len(self.dataset)}: "
                f"{entry['query']}"
            )

            result = self.evaluate_query(
                entry
            )

            all_results.append(
                result
            )

        return all_results

    def _print_summary(
        self,
        results,
    ):
        """打印 Evaluation 汇总"""

        print("\n" + "=" * 70)
        print("Retrieval Evaluation Summary")
        print("=" * 70)

        print(
            f"Dataset: {len(results)} queries"
        )

        print(
            f"Top-K (retrieval): {self.top_k}"
        )

        print(
            f"Metrics K: {KS}"
        )

        print()

        # ==================================================
        # 按策略汇总
        # ==================================================

        for strategy_name in STRATEGIES:

            strategy_results = [
                r["strategies"][
                    strategy_name
                ]
                for r in results
            ]

            print(
                f"--- {strategy_name} ---"
            )

            # 平均延迟
            avg_latency = (
                sum(
                    x["latency_ms"]
                    for x in strategy_results
                )
                / len(strategy_results)
                if strategy_results
                else 0.0
            )

            print(
                f"  Avg Latency: "
                f"{avg_latency:.1f} ms"
            )

            # 各 K 指标
            for k in KS:

                metric_names = [
                    f"hit@{k}",
                    f"recall@{k}",
                    f"precision@{k}",
                    f"mrr@{k}",
                ]

                values = {}

                for metric_name in metric_names:
                    values[metric_name] = (
                        sum(
                            x["metrics"].get(
                                metric_name,
                                0.0,
                            )
                            for x in strategy_results
                        )
                        / len(strategy_results)
                        if strategy_results
                        else 0.0
                    )

                print(
                    f"    Hit@{k}: "
                    f"{values[f'hit@{k}']:.4f} "
                    f"Recall@{k}: "
                    f"{values[f'recall@{k}']:.4f} "
                    f"Precision@{k}: "
                    f"{values[f'precision@{k}']:.4f} "
                    f"MRR@{k}: "
                    f"{values[f'mrr@{k}']:.4f}"
                )

            print()

        # ==================================================
        # 按 category 汇总
        # ==================================================

        categories = sorted(
            set(
                r.get(
                    "category",
                    "unknown",
                )
                for r in results
            )
        )

        print(
            "=" * 70
        )
        print("By Category")
        print("=" * 70)

        for category in categories:

            category_results = [
                r
                for r in results
                if r.get(
                    "category",
                    "unknown",
                )
                == category
            ]

            print(
                f"\n[{category}] "
                f"({len(category_results)} queries)"
            )

            for strategy_name in STRATEGIES:

                strategy_results = [
                    r["strategies"][
                        strategy_name
                    ]
                    for r in category_results
                ]

                if not strategy_results:
                    continue

                metrics = {}

                for k in KS:
                    metrics[
                        f"hit@{k}"
                    ] = (
                        sum(
                            x["metrics"].get(
                                f"hit@{k}",
                                0.0,
                            )
                            for x in strategy_results
                        )
                        / len(strategy_results)
                    )

                    metrics[
                        f"recall@{k}"
                    ] = (
                        sum(
                            x["metrics"].get(
                                f"recall@{k}",
                                0.0,
                            )
                            for x in strategy_results
                        )
                        / len(strategy_results)
                    )

                    metrics[
                        f"mrr@{k}"
                    ] = (
                        sum(
                            x["metrics"].get(
                                f"mrr@{k}",
                                0.0,
                            )
                            for x in strategy_results
                        )
                        / len(strategy_results)
                    )

                print(
                    f"  {strategy_name}: "
                    f"Hit@5={metrics['hit@5']:.4f}, "
                    f"Recall@5={metrics['recall@5']:.4f}, "
                    f"MRR@5={metrics['mrr@5']:.4f}"
                )

        # ==================================================
        # 按 difficulty 汇总
        # ==================================================

        difficulties = sorted(
            set(
                r.get(
                    "difficulty",
                    "unknown",
                )
                for r in results
            )
        )

        print(
            "\n" + "=" * 70
        )
        print("By Difficulty")
        print("=" * 70)

        for difficulty in difficulties:

            difficulty_results = [
                r
                for r in results
                if r.get(
                    "difficulty",
                    "unknown",
                )
                == difficulty
            ]

            print(
                f"\n[{difficulty}] "
                f"({len(difficulty_results)} queries)"
            )

            for strategy_name in STRATEGIES:

                strategy_results = [
                    r["strategies"][
                        strategy_name
                    ]
                    for r in difficulty_results
                ]

                if not strategy_results:
                    continue

                hit5 = (
                    sum(
                        x["metrics"].get(
                            "hit@5",
                            0.0,
                        )
                        for x in strategy_results
                    )
                    / len(strategy_results)
                )

                recall5 = (
                    sum(
                        x["metrics"].get(
                            "recall@5",
                            0.0,
                        )
                        for x in strategy_results
                    )
                    / len(strategy_results)
                )

                mrr5 = (
                    sum(
                        x["metrics"].get(
                            "mrr@5",
                            0.0,
                        )
                        for x in strategy_results
                    )
                    / len(strategy_results)
                )

                print(
                    f"  {strategy_name}: "
                    f"Hit@5={hit5:.4f}, "
                    f"Recall@5={recall5:.4f}, "
                    f"MRR@5={mrr5:.4f}"
                )

    def save_results(
        self,
        results,
    ):
        """保存 Evaluation 结果"""

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        output_path = (
            self.output_dir
            / f"results_{timestamp}.jsonl"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            for result in results:
                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(
            f"Results saved to "
            f"{output_path}"
        )

        return output_path


def main():
    """Evaluation CLI 入口"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieval Evaluation"
    )

    parser.add_argument(
        "--dataset",
        default=str(
            _PROJECT_ROOT
            / "data"
            / "evaluation.jsonl"
        ),
    )

    parser.add_argument(
        "--cache",
        default=str(
            _PROJECT_ROOT
            / "data"
            / "evaluation_cache.json"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            _PROJECT_ROOT
            / "eval_results"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: %(message)s"
        ),
    )

    evaluator = RetrievalEvaluator(
        dataset_path=args.dataset,
        cache_path=args.cache,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )

    results = evaluator.run()

    evaluator._print_summary(
        results
    )

    output_path = (
        evaluator.save_results(
            results
        )
    )

    total_queries = len(results)
    total_strategies = len(
        STRATEGIES
    )

    print(
        f"\nTotal: "
        f"{total_queries} queries x "
        f"{total_strategies} strategies = "
        f"{total_queries * total_strategies} evaluations"
    )

    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()

