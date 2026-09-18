"""FastAPI dependency injection."""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from ..agents.rag_agent import RAGAgent, create_rag_agent
from ..generation.llm_client import LLMClient, get_llm_client
from ..observability.logging import get_logger
from ..retrieval.vector_store import VectorStoreRetriever, get_retriever
from ..utils.cache import CacheManager, get_cache_manager
from ..utils.config import Settings, get_settings

logger = get_logger(__name__)

# Global instances
_retriever: Optional[VectorStoreRetriever] = None
_rag_agent: Optional[RAGAgent] = None


async def get_retriever_dependency() -> VectorStoreRetriever:
    """Get or create retriever dependency."""
    global _retriever
    if _retriever is None:
        _retriever = await get_retriever()
    return _retriever


async def get_rag_agent_dependency() -> RAGAgent:
    """Get or create RAG agent dependency."""
    global _rag_agent
    if _rag_agent is None:
        retriever = await get_retriever_dependency()
        _rag_agent = await create_rag_agent(retriever=retriever)
    return _rag_agent


def get_settings_dependency() -> Settings:
    """Get settings dependency."""
    return get_settings()


def get_llm_client_dependency() -> LLMClient:
    """Get LLM client dependency."""
    return get_llm_client()


def get_cache_dependency() -> CacheManager:
    """Get cache manager dependency."""
    return get_cache_manager()


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings_dependency),
) -> str:
    """
    Verify API key from header.

    Args:
        x_api_key: API key from header
        settings: Application settings

    Returns:
        Verified API key

    Raises:
        HTTPException: If API key is invalid
    """
    if not settings.api_auth_key:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is not configured",
            )
        return "development"

    import secrets

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_auth_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    logger.debug("API key verified")
    return x_api_key


def get_request_id(x_request_id: Optional[str] = Header(None)) -> str:
    """
    Get or generate request ID.

    Args:
        x_request_id: Request ID from header

    Returns:
        Request ID
    """
    import uuid

    return x_request_id or str(uuid.uuid4())
