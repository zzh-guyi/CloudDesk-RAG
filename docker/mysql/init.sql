-- CloudDesk RAG 知识库数据库初始化

CREATE DATABASE IF NOT EXISTS rag_knowledge_base;
USE rag_knowledge_base;

-- 知识库文档表
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(64) PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    version VARCHAR(20) DEFAULT '1.0',
    content LONGTEXT,
    chunk_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 关键词倒排表
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    keyword VARCHAR(128) NOT NULL,
    document_id VARCHAR(64) NOT NULL,
    term_freq INT DEFAULT 1,
    INDEX idx_keyword (keyword),
    INDEX idx_doc (document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 评测结果表
CREATE TABLE IF NOT EXISTS eval_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query TEXT NOT NULL,
    expected_doc_ids VARCHAR(255),
    retrieved_doc_ids VARCHAR(255),
    hit TINYINT DEFAULT 0,
    mrr DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
