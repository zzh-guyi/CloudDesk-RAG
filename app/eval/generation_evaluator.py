"""
Generation Evaluation - 调用 RAG Pipeline 并使用 LLM Judge 评价生成结果。
"""

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.eval.llm_judge import LLMJudge


logger = logging.getLogger(__name__)

DATASET_VERSION = "v1"
PROMPT_VERSION = "v1"
PASS_THRESHOLD = 0.7
EXPECTED_SAMPLES = 115
EXPECTED_WITH_RELEVANT = 109
EXPECTED_WITHOUT_RELEVANT = 6

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "citation_accuracy",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data" / "evaluation.jsonl"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "eval_results"
DEFAULT_CACHE_FILE = _PROJECT_ROOT / "data" / "judge_cache.json"


class DatasetValidationError(ValueError):
    """Dataset 不符合 Generation Evaluation 约定。"""


def _normalize_entry(
    entry: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise DatasetValidationError(
            f"Dataset entry {index} must be a JSON object"
        )

    query = entry.get("query")
    relevant_doc_ids = entry.get("relevant_doc_ids")
    category = entry.get("category")
    difficulty = entry.get("difficulty")

    if not isinstance(query, str) or not query.strip():
        raise DatasetValidationError(
            f"Dataset entry {index} has an invalid query"
        )

    if not isinstance(relevant_doc_ids, list):
        raise DatasetValidationError(
            f"Dataset entry {index} has invalid relevant_doc_ids"
        )

    if not isinstance(category, str) or not category:
        raise DatasetValidationError(
            f"Dataset entry {index} has an invalid category"
        )

    if not isinstance(difficulty, str) or not difficulty:
        raise DatasetValidationError(
            f"Dataset entry {index} has an invalid difficulty"
        )

    sample_id = entry.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        sample_id = f"eval_{index:04d}"

    return {
        "id": sample_id,
        "query": query,
        "relevant_doc_ids": list(relevant_doc_ids),
        "category": category,
        "difficulty": difficulty,
    }


def normalize_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """标准化 Dataset，但不生成 Query 或 reference_answer。"""

    return [
        _normalize_entry(entry, index)
        for index, entry in enumerate(dataset, start=1)
    ]


def validate_dataset(dataset: List[Dict[str, Any]]) -> None:
    """校验当前 v1 Dataset 的固定统计契约。"""

    if len(dataset) != EXPECTED_SAMPLES:
        raise DatasetValidationError(
            f"Expected exactly {EXPECTED_SAMPLES} samples, "
            f"got {len(dataset)}"
        )

    with_relevant = sum(
        1
        for sample in dataset
        if len(sample["relevant_doc_ids"]) > 0
    )
    without_relevant = len(dataset) - with_relevant

    if with_relevant != EXPECTED_WITH_RELEVANT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_WITH_RELEVANT} samples with "
            f"relevant_doc_ids, got {with_relevant}"
        )

    if without_relevant != EXPECTED_WITHOUT_RELEVANT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_WITHOUT_RELEVANT} no-answer queries, "
            f"got {without_relevant}"
        )


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """加载 JSONL Dataset，并根据原始行号补齐缺失的 id。"""

    dataset_path = Path(path)
    samples: List[Dict[str, Any]] = []

    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(
                        f"Invalid JSON at {dataset_path}:{line_number}: "
                        f"{exc.msg}"
                    ) from exc

                samples.append(_normalize_entry(entry, line_number))
    except OSError as exc:
        raise DatasetValidationError(
            f"Failed to load dataset {dataset_path}: {exc}"
        ) from exc

    validate_dataset(samples)
    return samples


class FileJudgeCache:
    """不依赖 Redis 的 JSON 文件 Judge Cache。"""

    def __init__(
        self,
        path: str | Path,
        dataset_version: str,
        model: str,
        prompt_version: str,
    ):
        self._path = Path(path)
        self._dataset_version = dataset_version
        self._model = model
        self._prompt_version = prompt_version
        self._entries: Dict[str, Dict[str, Any]] = {}

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _storage_key(self, judge_key: str) -> str:
        key_payload = {
            "dataset_version": self._dataset_version,
            "model": self._model,
            "prompt_version": self._prompt_version,
            "judge_key": judge_key,
        }
        serialized = json.dumps(
            key_payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _load(self) -> None:
        if not self._path.exists():
            return

        try:
            with self._path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            entries = data.get("entries", {})
            if isinstance(entries, dict):
                self._entries = entries
        except Exception as exc:
            logger.warning(
                "Failed to load judge cache %s: %s",
                self._path,
                exc,
            )
            self._entries = {}

    def _save(self) -> None:
        data = {
            "dataset_version": self._dataset_version,
            "model": self._model,
            "prompt_version": self._prompt_version,
            "entries": self._entries,
        }
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        storage_key = self._storage_key(key)
        record = self._entries.get(storage_key)

        if not isinstance(record, dict):
            return None

        result = record.get("result")
        if not isinstance(result, dict):
            return None

        return dict(result)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        storage_key = self._storage_key(key)
        self._entries[storage_key] = {
            "dataset_version": self._dataset_version,
            "model": self._model,
            "prompt_version": self._prompt_version,
            "judge_key": key,
            "result": dict(value),
        }
        self._save()


def _error_metric(message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "score": None,
        "reason": "",
        "error": message,
    }


def _valid_score(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None

    return score


class GenerationEvaluator:
    """Generation Evaluation 主流程。"""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        cache_file: str | Path = DEFAULT_CACHE_FILE,
        pipeline: Any = None,
        judge: Any = None,
        dataset: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.cache_file = Path(cache_file)
        self.pass_threshold = PASS_THRESHOLD

        if dataset is None:
            self.dataset = load_dataset(self.dataset_path)
        else:
            self.dataset = normalize_dataset(dataset)
            validate_dataset(self.dataset)

        self._pipeline = pipeline
        self._judge = judge
        self._model_name = model_name

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from app.rag.pipeline import get_pipeline

            self._pipeline = get_pipeline()
        return self._pipeline

    def _get_judge(self) -> Any:
        if self._judge is not None:
            return self._judge

        model_name = self._model_name
        if model_name is None:
            from config.settings import settings

            model_name = settings.llm_model

        cache = FileJudgeCache(
            path=self.cache_file,
            dataset_version=DATASET_VERSION,
            model=model_name,
            prompt_version=PROMPT_VERSION,
        )
        self._judge = LLMJudge(cache=cache)
        return self._judge

    def _extract_metadata(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        pipeline = self._get_pipeline()
        result = pipeline.run(
            query=sample["query"],
            include_evaluation_metadata=True,
        )

        if not isinstance(result, dict):
            raise ValueError("Pipeline result must be a dictionary")

        metadata = result.get("evaluation_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Pipeline result is missing evaluation_metadata"
            )

        required_fields = (
            "question",
            "rewritten_query",
            "context",
            "answer",
            "sources",
        )
        missing = [
            field
            for field in required_fields
            if field not in metadata
        ]
        if missing:
            raise ValueError(
                "Evaluation metadata is missing: "
                + ", ".join(missing)
            )

        for field in (
            "question",
            "rewritten_query",
            "context",
            "answer",
        ):
            if not isinstance(metadata[field], str):
                raise ValueError(
                    f"Evaluation metadata field {field} must be a string"
                )

        if not isinstance(metadata["sources"], list):
            raise ValueError(
                "Evaluation metadata field sources must be a list"
            )

        return metadata

    @staticmethod
    def _normalize_judge_result(
        result: Any,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return _error_metric("Judge returned a non-dictionary result")

        if result.get("status") == "error":
            return {
                "status": "error",
                "score": None,
                "reason": "",
                "error": str(
                    result.get("error", "Judge returned error status")
                ),
            }

        if result.get("status") != "success":
            return _error_metric("Judge returned an invalid status")

        score = _valid_score(result.get("score"))
        if score is None:
            return _error_metric("Judge returned an invalid score")

        normalized = dict(result)
        normalized["score"] = score
        normalized["reason"] = str(result.get("reason", ""))
        return normalized

    def _judge_sample(
        self,
        question: str,
        context: str,
        answer: str,
        sources: List[Any],
    ) -> Dict[str, Dict[str, Any]]:
        judge = self._get_judge()
        calls = {
            "faithfulness": lambda: judge.judge_faithfulness(
                question,
                context,
                answer,
            ),
            "answer_relevancy": lambda: judge.judge_answer_relevancy(
                question,
                answer,
                context=context,
            ),
            "citation_accuracy": lambda: judge.judge_citation_accuracy(
                question,
                context,
                answer,
                sources,
            ),
        }

        results: Dict[str, Dict[str, Any]] = {}

        for metric_name, call in calls.items():
            try:
                results[metric_name] = self._normalize_judge_result(
                    call()
                )
            except Exception as exc:
                logger.exception(
                    "Generation judge failed: metric=%s",
                    metric_name,
                )
                results[metric_name] = _error_metric(
                    str(exc) or exc.__class__.__name__
                )

        return results

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            metadata = self._extract_metadata(sample)
        except Exception as exc:
            logger.exception(
                "Generation evaluation pipeline failed: id=%s",
                sample["id"],
            )
            error = str(exc) or exc.__class__.__name__
            return {
                "id": sample["id"],
                "query": sample["query"],
                "category": sample["category"],
                "difficulty": sample["difficulty"],
                "rewritten_query": None,
                "answer": None,
                "retrieved_context": None,
                "sources": [],
                "faithfulness": _error_metric(error),
                "answer_relevancy": _error_metric(error),
                "citation_accuracy": _error_metric(error),
            }

        judge_results = self._judge_sample(
            question=metadata["question"],
            context=metadata["context"],
            answer=metadata["answer"],
            sources=metadata["sources"],
        )

        return {
            "id": sample["id"],
            "query": sample["query"],
            "category": sample["category"],
            "difficulty": sample["difficulty"],
            "rewritten_query": metadata["rewritten_query"],
            "answer": metadata["answer"],
            "retrieved_context": metadata["context"],
            "sources": metadata["sources"],
            **judge_results,
        }

    def _summarize_metric(
        self,
        results: List[Dict[str, Any]],
        metric_name: str,
    ) -> Dict[str, Any]:
        scores = []

        for result in results:
            metric_result = result.get(metric_name)
            if not isinstance(metric_result, dict):
                continue
            if metric_result.get("status") != "success":
                continue

            score = _valid_score(metric_result.get("score"))
            if score is not None:
                scores.append(score)

        evaluated = len(scores)
        if evaluated == 0:
            return {
                "evaluated": 0,
                "mean": None,
                "pass_rate": None,
            }

        passed = sum(
            1
            for score in scores
            if score >= self.pass_threshold
        )

        return {
            "evaluated": evaluated,
            "mean": round(sum(scores) / evaluated, 4),
            "pass_rate": round(passed / evaluated, 4),
        }

    def build_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        successful_samples = 0

        for result in results:
            if all(
                isinstance(result.get(metric_name), dict)
                and result[metric_name].get("status") == "success"
                and _valid_score(
                    result[metric_name].get("score")
                ) is not None
                for metric_name in METRIC_NAMES
            ):
                successful_samples += 1

        summary = {
            "dataset_version": DATASET_VERSION,
            "total_samples": len(results),
            "successful_samples": successful_samples,
            "failed_samples": len(results) - successful_samples,
            "pass_threshold": self.pass_threshold,
        }

        for metric_name in METRIC_NAMES:
            summary[metric_name] = self._summarize_metric(
                results,
                metric_name,
            )

        return summary

    def save_results(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        result_path = (
            self.output_dir
            / f"generation_evaluation_{DATASET_VERSION}.jsonl"
        )
        summary_path = (
            self.output_dir
            / f"generation_summary_{DATASET_VERSION}.json"
        )

        with result_path.open("w", encoding="utf-8") as file:
            for result in results:
                file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return result_path, summary_path

    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")

        samples = (
            self.dataset[:limit]
            if limit is not None
            else self.dataset
        )
        results = []

        for index, sample in enumerate(samples, start=1):
            logger.info(
                "Generation evaluation %s/%s: %s",
                index,
                len(samples),
                sample["id"],
            )
            results.append(self.evaluate_sample(sample))

        summary = self.build_summary(results)
        result_path, summary_path = self.save_results(
            results,
            summary,
        )

        return {
            "results": results,
            "summary": summary,
            "result_path": str(result_path),
            "summary_path": str(summary_path),
        }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generation Evaluation"
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--cache-file",
        default=str(DEFAULT_CACHE_FILE),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
    )

    evaluator = GenerationEvaluator(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
    )
    outcome = evaluator.run(limit=args.limit)
    print(
        json.dumps(
            outcome["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
