"""
Query Rewrite 模块
对用户查询进行重写，提升检索效果
"""
from typing import Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)

from app.services.llm_service import get_llm_service
from config.settings import settings


REWRITE_PROMPT = """请对用户的问题进行分析和重写，使其更适合作为知识库检索的查询。

原始问题：{query}

要求：
1. 保留原始问题的核心意图
2. 补充可能的关键词和同义词
3. 如果问题已经足够明确，保持原样
4. 输出格式：只输出重写后的问题，不要解释

重写后的问题："""

CATEGORY_PROMPT = """请对以下问题进行分类，从以下类别中选择一个最合适的：

问题：{query}

类别列表：
- user_manual: 用户操作指南（注册、登录、创建项目、成员管理等）
- faq: 常见问题解答
- troubleshooting: 故障排查（错误码、异常处理）
- product_rules: 产品规则（限制、配额、政策）
- pricing: 价格套餐（免费版、团队版、企业版）
- api_docs: API 文档（接口说明、错误码）

只输出类别名称，不要解释。

类别："""


class QueryRewriter:
    """Query Rewrite 模块"""

    def __init__(self):
        self.llm = get_llm_service()

    def rewrite(self, query: str) -> Dict[str, Any]:
        """
        重写查询
        
        Args:
            query: 原始查询
            
        Returns:
            {"original": str, "rewritten": str}
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个查询优化助手，擅长将用户问题转换为更适合检索的查询语句。"},
                {"role": "user", "content": REWRITE_PROMPT.format(query=query)}
            ]
            rewritten = self.llm.generate(messages, max_tokens=200)
            # 清理输出
            rewritten = rewritten.strip().replace('"', "").replace("'", "").strip()
            if not rewritten or rewritten == query:
                rewritten = query
            return {"original": query, "rewritten": rewritten}
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query")
            return {"original": query, "rewritten": query}


class QueryRouter:
    """Query Router 模块 - 分类查询"""

    def __init__(self):
        self.llm = get_llm_service()
        self.categories = ["user_manual", "faq", "troubleshooting", "product_rules", "pricing", "api_docs"]

    def route(self, query: str) -> str:
        """
        对查询进行分类
        
        Args:
            query: 查询文本
            
        Returns:
            分类名称
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个查询分类助手，擅长将用户问题归类到最合适的类别。"},
                {"role": "user", "content": CATEGORY_PROMPT.format(query=query)}
            ]
            category = self.llm.generate(messages, max_tokens=50).strip().lower()
            # 验证分类
            if category in self.categories:
                return category
            # fallback: 基于关键词的简单分类
            return self._keyword_route(query)
        except Exception as e:
            logger.warning(f"Query routing failed: {e}, using keyword-based routing")
            return self._keyword_route(query)

    def _keyword_route(self, query: str) -> str:
        """基于关键词的简单分类（LLM 失败时的 fallback）"""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["e1", "e2", "e3", "error", "错误", "故障", "排查"]):
            return "troubleshooting"
        if any(kw in query_lower for kw in ["价格", "收费", "套餐", "版", "pricing", "plan", "免费"]):
            return "pricing"
        if any(kw in query_lower for kw in ["api", "接口", "调用", "文档"]):
            return "api_docs"
        if any(kw in query_lower for kw in ["怎么", "如何", "步骤", "操作", "教程", "guide", "how"]):
            return "user_manual"
        if any(kw in query_lower for kw in ["为什么", "为何", "原因", "why"]):
            return "faq"
        return "faq"  # default


# 全局单例
_query_rewriter: Optional[QueryRewriter] = None
_query_router: Optional[QueryRouter] = None


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter


def get_query_router() -> QueryRouter:
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
    return _query_router


