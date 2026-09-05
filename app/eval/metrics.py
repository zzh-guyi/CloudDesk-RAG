"""
Evaluation Metrics - 纯函数，无外部依赖
计算 Hit@K, Recall@K, Precision@K, MRR@K
"""
from typing import List, Set


def hit_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Top-K 中是否至少命中一个相关文档。
    
    Args:
        retrieved: 检索结果 document_id 列表（已排序，前 K 个为 Top-K）
        relevant: 相关文档 document_id 列表（ground truth）
        k: 截断位置
    
    Returns:
        1.0 如果 Top-K 中至少有一个相关文档，否则 0.0
    """
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    return 1.0 if any(d in relevant_set for d in top_k) else 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Top-K 中召回的相关文档比例。
    
    Recall@K = |Top-K ∩ Relevant| / |Relevant|
    
    Args:
        retrieved: 检索结果 document_id 列表
        relevant: 相关文档 document_id 列表
        k: 截断位置
    
    Returns:
        召回率，范围 [0, 1]
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hit_count = len(top_k & relevant_set)
    return hit_count / len(relevant_set)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Top-K 中精确命中的比例（标准 IR 定义：除以 K，不除以实际返回数）。
    
    Precision@K = |Top-K ∩ Relevant| / K
    
    即使实际 retrieved 数量 < K，仍然除以 K。
    k <= 0 或 retrieved 为空时返回 0.0。
    
    Args:
        retrieved: 检索结果 document_id 列表
        relevant: 相关文档 document_id 列表
        k: 截断位置
    
    Returns:
        精确率，范围 [0, 1]
    """
    if k <= 0:
        return 0.0
    if not retrieved:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hit_count = len(top_k & relevant_set)
    return hit_count / k


def mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Top-K 内第一个相关文档的倒数排名。
    
    MRR@K = 1/r  如果第一个相关文档排名 r <= K
           = 0   如果 Top-K 内无命中，或首个命中排名 > K
    
    MRR 是 per-query 计算后再取平均，此处返回单条 query 的 MRR 值。
    
    Args:
        retrieved: 检索结果 document_id 列表（完整排序列表）
        relevant: 相关文档 document_id 列表
        k: 截断位置（MRR 计算时使用完整列表，但超过 K 的命中记为 0）
    
    Returns:
        MRR 值，范围 [0, 1]
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank if rank <= k else 0.0
    return 0.0


def compute_query_metrics(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    ks: List[int] = None
) -> dict:
    """
    对单条 query 计算所有 K 值下的所有指标。
    
    Args:
        retrieved_doc_ids: 该策略返回的 document_id 列表（已排序）
        relevant_doc_ids: ground truth 相关文档 ID 列表
        ks: 要计算的 K 值列表，默认 [1, 3, 5, 10]
    
    Returns:
        {
            "hit@1": 1.0, "hit@3": 1.0, ...
            "recall@1": 0.5, ...
            "precision@1": 1.0, ...
            "mrr@1": 1.0, ...
        }
    """
    if ks is None:
        ks = [1, 3, 5, 10]
    metrics = {}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"mrr@{k}"] = mrr_at_k(retrieved_doc_ids, relevant_doc_ids, k)
    return metrics


def aggregate_metrics(all_results: List[dict]) -> dict:
    """
    对多条 query 的指标结果取平均，得到 Overall 指标。
    
    Args:
        all_results: 每条 query 的 compute_query_metrics 结果列表
    
    Returns:
        每种指标的平均值，key 格式 "strategy_metric@k"
    """
    if not all_results:
        return {}
    # 收集所有指标名
    metric_names = set()
    for r in all_results:
        metric_names.update(r.keys())
    aggregated = {}
    for name in sorted(metric_names):
        values = [r[name] for r in all_results if name in r]
        if values:
            aggregated[name] = sum(values) / len(values)
    return aggregated
