"""
聊天接口路由
"""
from fastapi import APIRouter
import logging
logger = logging.getLogger(__name__)

from app.models.schemas import ChatRequest, ChatResponse
from app.rag.pipeline import get_pipeline

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    主对话接口
    
    - 用户提问
    - 返回有依据的回答 + 引用来源
    """
    logger.info(f"Chat request: {request.query[:50]}...")
    
    pipeline = get_pipeline()
    result = pipeline.run(
        query=request.query,
        session_id=request.session_id,
        top_k=request.top_k
    )
    
    return ChatResponse(
        query=result["query"],
        rewritten_query=result.get("rewritten_query"),
        answer=result["answer"],
        sources=result["sources"],
        retrieval_info=result["retrieval_info"],
        session_id=result.get("session_id")
    )


