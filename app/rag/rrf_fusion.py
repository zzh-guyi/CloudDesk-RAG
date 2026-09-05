"""
RRF (Reciprocal Rank Fusion) 融合算法
"""
from typing import List, Dict, Any
from collections import defaultdict


def rrf_fusion(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    RRF 融合算法
    
    score(doc_id) = 1/(k + rank_vector) + 1/(k + rank_keyword)
    
    Args:
        vector_results: 向量检索结果列表
        keyword_results: 关键词检索结果列表
        k: RRF 常数，默认 60
        
    Returns:
        融合后的结果列表，按 rrf_score 降序排列
    """
    scores = defaultdict(float)
    result_map = {}

    # 计算向量检索的 RRF 分数
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["document_id"]
        scores[doc_id] += 1.0 / (k + rank)
        result_map[doc_id] = doc

    # 计算关键词检索的 RRF 分数
    for rank, doc in enumerate(keyword_results, start=1):
        doc_id = doc["document_id"]
        scores[doc_id] += 1.0 / (k + rank)
        if doc_id not in result_map:
            result_map[doc_id] = doc

    # 排序
    sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    output = []
    for doc_id in sorted_docs:
        doc = result_map[doc_id]
        doc["rrf_score"] = scores[doc_id]
        output.append(doc)

    return output


