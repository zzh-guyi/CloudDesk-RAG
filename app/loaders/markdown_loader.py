"""
Markdown 文档加载器
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)

from app.rag.models import DocumentChunk


class MarkdownLoader:
    """加载 Markdown 文档，提取内容和元数据"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def load(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载知识库文档
        
        Args:
            category: 可选的分类过滤
            
        Returns:
            文档列表，每项包含 content, metadata
        """
        documents = []
        search_path = self.base_path

        if category:
            search_path = search_path / category

        if not search_path.exists():
            logger.warning(f"Knowledge base path not found: {search_path}")
            return documents

        for md_file in search_path.glob("**/*.md"):
            doc = self._load_file(md_file)
            if doc:
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {search_path}")
        return documents

    def _load_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """加载单个 Markdown 文件"""
        try:
            content = filepath.read_text(encoding="utf-8")
            metadata = self._extract_metadata(content, filepath)
            return {
                "content": content,
                "metadata": metadata,
                "source": str(filepath.relative_to(self.base_path.parent.parent)),
            }
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def _extract_metadata(self, content: str, filepath: Path) -> Dict[str, Any]:
        """从 Markdown 文件提取元数据"""
        metadata = {
            "source": str(filepath),
            "category": filepath.parent.name,
            "title": self._extract_title(content),
            "version": "1.0",
        }
        # 尝试从 frontmatter 提取
        if content.startswith("```"):
            lines = content.split("\n")
            for i, line in enumerate(lines[:10]):
                if line.startswith("title:"):
                    metadata["title"] = line.replace("title:", "").strip()
                elif line.startswith("category:"):
                    metadata["category"] = line.replace("category:", "").strip()
                elif line.startswith("version:"):
                    metadata["version"] = line.replace("version:", "").strip()
        return metadata

    def _extract_title(self, content: str) -> str:
        """提取 Markdown 标题"""
        for line in content.split("\n"):
            if line.startswith("# "):
                return line[2:].strip()
        return Path(content).stem


class KnowledgeBaseLoader:
    """知识库加载器 - 加载文档并分块"""

    def __init__(self, base_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.loader = MarkdownLoader(base_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_chunk(self, category: Optional[str] = None) -> List[DocumentChunk]:
        """
        加载文档并分块
        
        Returns:
            文档分块列表
        """
        from app.chunkers.text_splitter import TextChunker
        chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        documents = self.loader.load(category)
        chunks = []
        doc_counter = 0

        for doc in documents:
            doc_counter += 1
            doc_id = f"doc_{doc_counter:04d}"
            chunks.extend(chunker.chunk(doc["content"], doc_id, doc["metadata"]))

        logger.info(f"Loaded {len(documents)} documents, created {len(chunks)} chunks")
        return chunks


