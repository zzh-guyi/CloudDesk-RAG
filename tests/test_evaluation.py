"""
Evaluation 单元测试（使用 unittest，无需额外依赖）
测试 metrics.py 纯函数和 evaluator 基础逻辑
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 确保路径正确
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "local_packages"))

from app.eval.metrics import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    compute_query_metrics,
    aggregate_metrics,
)
from app.eval.evaluator import load_evaluation_dataset, RewriteCache


# ─────────────────────────────────────────────
# hit_at_k 测试
# ─────────────────────────────────────────────

class TestHitAtK(unittest.TestCase):
    def test_hit_in_top_k(self):
        self.assertEqual(hit_at_k(["doc_a", "doc_b", "doc_c"], ["doc_a", "doc_d"], 3), 1.0)

    def test_no_hit_in_top_k(self):
        self.assertEqual(hit_at_k(["doc_x", "doc_y", "doc_z"], ["doc_a", "doc_d"], 3), 0.0)

    def test_hit_at_k_1(self):
        self.assertEqual(hit_at_k(["doc_a"], ["doc_a"], 1), 1.0)

    def test_no_hit_at_k_1(self):
        self.assertEqual(hit_at_k(["doc_b"], ["doc_a"], 1), 0.0)

    def test_empty_retrieved(self):
        self.assertEqual(hit_at_k([], ["doc_a"], 1), 0.0)

    def test_k_larger_than_retrieved(self):
        self.assertEqual(hit_at_k(["doc_a"], ["doc_a"], 5), 1.0)


# ─────────────────────────────────────────────
# recall_at_k 测试
# ─────────────────────────────────────────────

class TestRecallAtK(unittest.TestCase):
    def test_partial_recall(self):
        self.assertEqual(recall_at_k(["doc_a", "doc_b"], ["doc_a", "doc_d"], 2), 0.5)

    def test_full_recall(self):
        self.assertEqual(recall_at_k(["doc_a", "doc_d"], ["doc_a", "doc_d"], 2), 1.0)

    def test_no_recall(self):
        self.assertEqual(recall_at_k(["doc_x"], ["doc_a"], 1), 0.0)

    def test_empty_relevant(self):
        self.assertEqual(recall_at_k(["doc_a"], [], 1), 0.0)

    def test_empty_retrieved(self):
        self.assertEqual(recall_at_k([], ["doc_a"], 1), 0.0)

    def test_k_greater_than_retrieved(self):
        self.assertEqual(recall_at_k(["doc_a"], ["doc_a", "doc_b"], 5), 0.5)


# ─────────────────────────────────────────────
# precision_at_k 测试（标准 IR 定义：除以 K）
# ─────────────────────────────────────────────

class TestPrecisionAtK(unittest.TestCase):
    def test_standard_precision(self):
        self.assertAlmostEqual(precision_at_k(["doc_a", "doc_b", "doc_c"], ["doc_a"], 3), 1 / 3)

    def test_precision_k_equals_retrieved(self):
        self.assertEqual(precision_at_k(["doc_a", "doc_b"], ["doc_a"], 2), 0.5)

    def test_precision_k_larger_than_retrieved(self):
        self.assertAlmostEqual(precision_at_k(["doc_a"], ["doc_a"], 5), 0.2)

    def test_precision_empty_retrieved(self):
        self.assertEqual(precision_at_k([], ["doc_a"], 1), 0.0)

    def test_precision_k_zero(self):
        self.assertEqual(precision_at_k(["doc_a"], ["doc_a"], 0), 0.0)

    def test_precision_k_negative(self):
        self.assertEqual(precision_at_k(["doc_a"], ["doc_a"], -1), 0.0)

    def test_full_precision(self):
        self.assertEqual(precision_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b"], 2), 1.0)


# ─────────────────────────────────────────────
# mrr_at_k 测试
# ─────────────────────────────────────────────

class TestMrrAtK(unittest.TestCase):
    def test_first_hit(self):
        self.assertEqual(mrr_at_k(["doc_a", "doc_b"], ["doc_a"], 3), 1.0)

    def test_second_hit(self):
        self.assertEqual(mrr_at_k(["doc_b", "doc_a"], ["doc_a"], 3), 0.5)

    def test_no_hit(self):
        self.assertEqual(mrr_at_k(["doc_x", "doc_y"], ["doc_a"], 3), 0.0)

    def test_hit_outside_k(self):
        self.assertEqual(mrr_at_k(["doc_b", "doc_c", "doc_d", "doc_a"], ["doc_a"], 3), 0.0)

    def test_empty_retrieved(self):
        self.assertEqual(mrr_at_k([], ["doc_a"], 1), 0.0)

    def test_k_zero(self):
        self.assertEqual(mrr_at_k(["doc_a"], ["doc_a"], 0), 0.0)

    def test_multiple_relevant_first_hit(self):
        self.assertEqual(mrr_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b"], 3), 1.0)


# ─────────────────────────────────────────────
# compute_query_metrics 测试
# ─────────────────────────────────────────────

class TestComputeQueryMetrics(unittest.TestCase):
    def test_all_metrics(self):
        retrieved = ["doc_0004", "doc_0034", "other1", "other2", "other3"]
        relevant = ["doc_0004", "doc_0034"]
        metrics = compute_query_metrics(retrieved, relevant, [1, 3, 5])

        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertEqual(metrics["hit@5"], 1.0)

        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)

        self.assertEqual(metrics["precision@1"], 1.0)
        self.assertAlmostEqual(metrics["precision@3"], 2 / 3)
        self.assertAlmostEqual(metrics["precision@5"], 2 / 5)

        self.assertEqual(metrics["mrr@1"], 1.0)
        self.assertEqual(metrics["mrr@3"], 1.0)
        self.assertEqual(metrics["mrr@5"], 1.0)

    def test_no_hit(self):
        retrieved = ["doc_x", "doc_y", "doc_z"]
        relevant = ["doc_a"]
        metrics = compute_query_metrics(retrieved, relevant, [1, 3])

        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["precision@1"], 0.0)
        self.assertEqual(metrics["mrr@1"], 0.0)
        self.assertEqual(metrics["mrr@3"], 0.0)


# ─────────────────────────────────────────────
# aggregate_metrics 测试
# ─────────────────────────────────────────────

class TestAggregateMetrics(unittest.TestCase):
    def test_average(self):
        results = [
            {"hit@1": 1.0, "recall@1": 1.0},
            {"hit@1": 0.0, "recall@1": 0.0},
        ]
        agg = aggregate_metrics(results)
        self.assertAlmostEqual(agg["hit@1"], 0.5)
        self.assertAlmostEqual(agg["recall@1"], 0.5)

    def test_empty(self):
        self.assertEqual(aggregate_metrics([]), {})


# ─────────────────────────────────────────────
# RewriteCache 测试
# ─────────────────────────────────────────────

class TestRewriteCache(unittest.TestCase):
    def test_cache_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = RewriteCache(cache_path)
            self.assertEqual(len(cache._rewrites), 0)

            cache._rewrites["test query"] = "rewritten query"
            cache._save()

            cache2 = RewriteCache(cache_path)
            self.assertEqual(cache2.get("test query"), "rewritten query")

    def test_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache2.json")
            cache = RewriteCache(cache_path)
            cache._rewrites["q1"] = "r1"
            cache._rewrites["q2"] = "r2"
            cache._save()

            cache3 = RewriteCache(cache_path)
            self.assertEqual(cache3.get("q1"), "r1")
            self.assertEqual(cache3.get("q2"), "r2")


# ─────────────────────────────────────────────
# load_evaluation_dataset 测试
# ─────────────────────────────────────────────

class TestLoadDataset(unittest.TestCase):
    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "test_eval.jsonl")
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"query": "test", "relevant_doc_ids": ["d1"], "category": "faq", "difficulty": "easy"}) + "\n")
                f.write(json.dumps({"query": "test2", "relevant_doc_ids": ["d2", "d3"], "category": "pricing", "difficulty": "medium"}) + "\n")

            dataset = load_evaluation_dataset(dataset_path)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["query"], "test")
            self.assertEqual(dataset[1]["relevant_doc_ids"], ["d2", "d3"])

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "empty.jsonl")
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write("")

            dataset = load_evaluation_dataset(dataset_path)
            self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
