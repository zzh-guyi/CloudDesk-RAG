"""
MySQL 关键词检索封装（简化版 BM25）
"""
from typing import List, Optional, Any
import logging
logger = logging.getLogger(__name__)

import re
import math
from collections import defaultdict
from config.settings import settings


class KeywordStore:
    """MySQL 关键词检索，实现 BM25 评分 + category 过滤"""

    def __init__(self):
        self._conn = None

    def connect(self):
        """连接 MySQL"""
        try:
            import pymysql
            self._conn = pymysql.connect(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
                charset="utf8mb4"
            )
            logger.info(f"Connected to MySQL at {settings.mysql_host}:{settings.mysql_port}")
            self._create_tables()
        except Exception as e:
            logger.warning(f"MySQL connection failed: {e}, using fallback")
            self._conn = None

    def _create_tables(self):
        """创建必要的表"""
        if self._conn is None:
            return
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id VARCHAR(64) PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                category VARCHAR(50) NOT NULL,
                version VARCHAR(20) DEFAULT '\''1.0'\'',
                content LONGTEXT,
                chunk_id VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                keyword_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                keyword VARCHAR(128) NOT NULL,
                document_id VARCHAR(64) NOT NULL,
                term_freq INT DEFAULT 1,
                INDEX idx_keyword (keyword),
                INDEX idx_doc (document_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                query TEXT NOT NULL,
                expected_doc_ids VARCHAR(255),
                retrieved_doc_ids VARCHAR(255),
                hit TINYINT DEFAULT 0,
                mrr DECIMAL(10,6),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        self._conn.commit()
        logger.info("MySQL tables ensured")

    def insert_document(self, document_id: str, title: str, category: str,
                        source: str, content: str, chunk_id: Optional[str] = None):
        """插入文档"""
        if self._conn is None:
            return
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO documents (document_id, title, category, source, content, chunk_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE content = VALUES(content)
        """, (document_id, title, category, source, content, chunk_id))

        # 提取关键词并插入倒排表
        keywords = self._extract_keywords(content)
        for kw in keywords:
            cursor.execute("""
                INSERT INTO keywords (keyword, document_id, term_freq)
                VALUES (%s, %s, 1)
                ON DUPLICATE KEY UPDATE term_freq = term_freq + 1
            """, (kw, document_id))

        self._conn.commit()

    def _extract_keywords(self, text: str) -> List[str]:
        """关键词提取：中文 2-gram + 英文 token，去除停用词"""
        stop_words = {"的", "是", "在", "我", "有", "和", "就", "不", "人", "都",
                      "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
                      "会", "着", "没有", "看", "好", "自己", "这", "那", "但", "还",
                      "the", "is", "at", "which", "on", "and", "a", "an", "for", "to"}
        # 按 token 切分：中文字符串、英文单词、数字
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+", text)
        words = []
        for token in tokens:
            if "\u4e00" <= token[0] <= "\u9fff":
                # 中文字符串：2-gram 滑动窗口
                for i in range(len(token) - 1):
                    bigram = token[i:i + 2]
                    if bigram not in stop_words:
                        words.append(bigram)
            else:
                # 英文/数字：转小写，长度>=2
                low = token.lower()
                if len(low) >= 2 and low not in stop_words:
                    words.append(low)
        return list(set(words))
    def search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[dict]:
        """
        关键词检索（BM25 评分 + category 过滤）

        Args:
            query: 查询文本
            top_k: 返回数量
            category: 可选分类过滤，None 时全表搜索

        Returns:
            检索结果列表
        """
        if self._conn is None:
            return []

        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        cursor = self._conn.cursor()

        # 构建 category 过滤条件
        cat_filter = ""
        cat_params: list = []
        if category:
            cat_filter = " AND d.category = %s"
            cat_params.append(category)

        # 预计算所有候选文档的长度，用于 BM25
        doc_lengths: dict = {}
        for kw in keywords:
            param = tuple([kw] + cat_params) if cat_params else (kw,)
            cursor.execute(
                f"""
                SELECT DISTINCT k.document_id, LENGTH(d.content) as doc_len
                FROM keywords k
                INNER JOIN documents d ON k.document_id = d.document_id
                WHERE k.keyword = %s{cat_filter}
                """,
                param,
            )
            for row in cursor.fetchall():
                doc_lengths[row[0]] = max(row[1], 1)

        if not doc_lengths:
            return []

        avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths)

        k1, b = 1.5, 0.75

        results = {}
        for kw in keywords:
            param = tuple([kw] + cat_params) if cat_params else (kw,)
            cursor.execute(
                f"""
                SELECT k.document_id, k.term_freq
                FROM keywords k
                INNER JOIN documents d ON k.document_id = d.document_id
                WHERE k.keyword = %s{cat_filter}
                """,
                param,
            )
            for row in cursor.fetchall():
                doc_id, tf = row
                if doc_id not in results:
                    results[doc_id] = {"score": 0.0, "keywords": []}
                doc_len = doc_lengths.get(doc_id, 1)
                bm25_score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
                results[doc_id]["score"] += bm25_score
                results[doc_id]["keywords"].append(kw)

        # 获取文档详情
        output = []
        for doc_id, info in results.items():
            cursor.execute(
                "SELECT title, category, source, content FROM documents WHERE document_id = %s",
                (doc_id,)
            )
            row = cursor.fetchone()
            if row:
                output.append({
                    "document_id": doc_id,
                    "title": row[0],
                    "category": row[1],
                    "source": row[2],
                    "content": row[3],
                    "score": round(info["score"], 4),
                    "retrieval_source": "keyword",
                })

        # 按分数排序
        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]

    def get_document(self, document_id: str) -> Optional[dict]:
        """获取单个文档"""
        if self._conn is None:
            return None
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT document_id, title, category, source, content FROM documents WHERE document_id = %s",
            (document_id,)
        )
        row = cursor.fetchone()
        if row:
            return {"document_id": row[0], "title": row[1], "category": row[2],
                    "source": row[3], "content": row[4]}
        return None

    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def is_available(self) -> bool:
        return self._conn is not None


# 全局单例
_keyword_store: Optional[KeywordStore] = None


def get_keyword_store() -> KeywordStore:
    global _keyword_store
    if _keyword_store is None:
        _keyword_store = KeywordStore()
    return _keyword_store


