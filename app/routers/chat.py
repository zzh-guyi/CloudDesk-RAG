import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import logging
logger = logging.getLogger(__name__)

from app.models.schemas import ChatRequest, ChatResponse
from app.rag.pipeline import get_pipeline

router = APIRouter()


@router.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f'Chat request: {request.query[:50]}...')
    pipeline = get_pipeline()
    result = pipeline.run(query=request.query, session_id=request.session_id, top_k=request.top_k)
    return ChatResponse(
        query=result['query'],
        rewritten_query=result.get('rewritten_query'),
        answer=result['answer'],
        sources=result['sources'],
        retrieval_info=result['retrieval_info'],
        session_id=result.get('session_id')
    )


def _sse_event(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


@router.post('/chat/stream')
async def chat_stream(request: ChatRequest, req: Request):
    logger.info(f'Chat stream request: {request.query[:50]}...')
    pipeline = get_pipeline()

    async def event_stream():
        try:
            for event in pipeline.run_stream(query=request.query, top_k=request.top_k):
                if await req.is_disconnected():
                    break
                if event['type'] == 'sources':
                    yield _sse_event('sources', event['data'])
                elif event['type'] == 'token':
                    yield _sse_event('token', {'content': event['data']})
                elif event['type'] == 'error':
                    yield _sse_event('error', {'message': event['data']})
                    break
                elif event['type'] == 'done':
                    yield _sse_event('done', event['data'])
        except Exception as e:
            logger.error(f'Stream error: {e}')
            yield _sse_event('error', {'message': str(e)})

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'}
    )
