"""Query endpoints for RAG system."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...agents.rag_agent import RAGAgent
from ...api.dependencies import get_rag_agent_dependency, get_request_id, verify_api_key
from ...api.models import DocumentResponse, QueryRequest, QueryResponse
from ...observability.logging import get_logger
from ...utils.exceptions import RAGException

logger = get_logger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["query"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    agent: RAGAgent = Depends(get_rag_agent_dependency),
    request_id: str = Depends(get_request_id),
) -> QueryResponse:
    """
    Query the RAG system.

    Args:
        request: Query request
        agent: RAG agent instance
        request_id: Request ID for tracking

    Returns:
        Query response with answer and documents

    Raises:
        HTTPException: If query fails
    """
    logger.info(f"Query request [{request_id}]: {request.query}")

    try:
        # Execute agent
        result = await agent.run(
            query=request.query,
            top_k=request.top_k,
            use_reflection=request.use_reflection,
        )

        # Format response
        response = QueryResponse(
            query=result["query"],
            answer=result["answer"],
            documents=[
                DocumentResponse(
                    id=doc["id"],
                    score=doc["score"],
                    content=doc["content"],
                )
                for doc in result["documents"]
            ],
            reflection=result.get("reflection"),
            metadata=result.get("metadata", {}),
        )

        logger.info(f"Query completed [{request_id}]")
        return response

    except RAGException as e:
        logger.error(f"RAG error [{request_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail=e.to_dict(),
        )
    except Exception as e:
        logger.error(f"Unexpected error [{request_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)},
        )


@router.post("/stream")
async def query_rag_stream(
    request: QueryRequest,
    agent: RAGAgent = Depends(get_rag_agent_dependency),
    request_id: str = Depends(get_request_id),
):
    """
    Stream RAG query results.

    Args:
        request: Query request
        agent: RAG agent instance
        request_id: Request ID for tracking

    Returns:
        Streaming response with events
    """
    logger.info(f"Stream query request [{request_id}]: {request.query}")

    async def event_generator():
        """Generate server-sent events."""
        try:
            async for event in agent.stream(query=request.query):
                import json

                yield f"data: {json.dumps(event)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error [{request_id}]: {e}")
            import json

            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
