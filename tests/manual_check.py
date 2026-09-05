import sys
sys.path.insert(0, r"H:\pythonProject\RAG 企业 SaaS 智能客服系统")
sys.path.insert(0, r"H:\pythonProject\RAG 企业 SaaS 智能客服系统\local_packages")

from app.rag.rrf_fusion import rrf_fusion
from app.rag.context_compressor import ContextCompressor
from app.rag.query_rewrite import QueryRewriter, QueryRouter
from app.rag.models import RetrievalResult

print("=== Running Tests ===")

# Test 1: RRF Fusion
print("\nTest 1: RRF Fusion")
vector_results = [
    {"document_id": "doc_1", "score": 0.9, "title": "文档1"},
    {"document_id": "doc_2", "score": 0.8, "title": "文档2"},
]
keyword_results = [
    {"document_id": "doc_2", "score": 5, "title": "文档2"},
    {"document_id": "doc_3", "score": 3, "title": "文档3"},
]
result = rrf_fusion(vector_results, keyword_results, k=60)
assert len(result) == 3
assert result[0]["document_id"] == "doc_2"
assert "rrf_score" in result[0]
print("  PASS")

# Test 2: RRF empty
print("\nTest 2: RRF Fusion (empty)")
result = rrf_fusion([], [], k=60)
assert result == []
print("  PASS")

# Test 3: Context Compressor
print("\nTest 3: Context Compressor")
compressor = ContextCompressor()
results = [
    RetrievalResult(
        document_id="doc_1", title="测试", category="faq", source="test.md",
        content="这是一段测试内容。" * 100, chunk_index=0,
        retrieval_source="vector", rrf_score=0.5, rerank_score=0.8
    )
]
context = compressor.compress(results, "测试")
assert len(context) > 0
print("  PASS")

# Test 4: Query Router
print("\nTest 4: Query Router")
router = QueryRouter()
assert router.route("如何注册账号") in ["user_manual", "faq"]
assert router.route("E1001 错误码是什么") == "troubleshooting"
assert router.route("免费版和团队版有什么区别") in ["pricing", "faq"]
print("  PASS")

# Test 5: Query Rewrite
print("\nTest 5: Query Rewrite")
rewriter = QueryRewriter()
result = rewriter.rewrite("我创建项目以后咋看不到人了？")
assert "original" in result and "rewritten" in result
print("  PASS")

print("\n=== All Tests Passed ===")
