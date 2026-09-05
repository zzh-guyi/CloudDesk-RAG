"""
文本分块器 - RecursiveCharacterTextSplitter 封装
"""
from typing import List, Dict, Any
import hashlib
import logging
logger = logging.getLogger(__name__)

from app.rag.models import DocumentChunk


class TextChunker:
    """文本分块器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        将文本分块
        
        Args:
            text: 原文本
            document_id: 文档 ID
            metadata: 元数据
            
        Returns:
            分块列表
        """
        if not text or not text.strip():
            return []

        # 按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果单个段落超过 chunk_size，按句子分割
            if len(para) > self.chunk_size:
                sentences = self._split_by_sentences(para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                current_chunk, document_id, chunk_index, metadata
                            ))
                            chunk_index += 1
                            # 保留 overlap
                            overlap_text = self._get_overlap(current_chunk, sentence)
                            current_chunk = overlap_text + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += "\n\n" + sentence if current_chunk else sentence
            else:
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk, document_id, chunk_index, metadata
                        ))
                        chunk_index += 1
                        overlap_text = self._get_overlap(current_chunk, para)
                        current_chunk = overlap_text + para
                    else:
                        current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

        # 添加最后一个块
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, document_id, chunk_index, metadata
            ))

        logger.info(f"Split document {document_id} into {len(chunks)} chunks")
        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        import re
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap(self, text1: str, text2: str) -> str:
        """获取两个文本的重叠部分"""
        if len(text2) <= self.chunk_overlap:
            return text2
        return text2[:self.chunk_overlap]

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        chunk_index: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """创建分块对象"""
        chunk_id = hashlib.md5(
            f"{document_id}_{chunk_index}_{content[:50]}".encode()
        ).hexdigest()[:16]
        return DocumentChunk(
            document_id=document_id,
            chunk_id=chunk_id,
            title=metadata.get("title", "Untitled"),
            category=metadata.get("category", "unknown"),
            source=metadata.get("source", ""),
            content=content,
            chunk_index=chunk_index,
            metadata=metadata
        )


