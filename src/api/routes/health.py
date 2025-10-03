"""Health check endpoints."""

from fastapi import APIRouter, Depends

from ...api.dependencies import get_cache_dependency, get_settings_dependency
from ...api.models import HealthResponse
from ...database.vector_db import get_vector_store
from ...observability.logging import get_logger
from ...utils.cache import CacheManager
from ...utils.config import Settings

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings_dependency),
    cache: CacheManager = Depends(get_cache_dependency),
) -> HealthResponse:
    """
    Check system health and dependencies.

    Returns:
        Health status of the system and all dependencies
    """
    logger.debug("Health check requested")

    dependencies = {}

    # Check vector database
    try:
        vector_store = get_vector_store()
        dependencies["vector_db"] = "healthy"
    except Exception as e:
        logger.error(f"Vector DB unhealthy: {e}")
        dependencies["vector_db"] = f"unhealthy: {str(e)}"

    # Check Redis cache
    try:
        client = await cache.get_client()
        await client.ping()
        dependencies["redis"] = "healthy"
    except Exception as e:
        logger.warning(f"Redis unhealthy: {e}")
        dependencies["redis"] = f"unhealthy: {str(e)}"

    # Check LLM (just check if API key is configured)
    if settings.anthropic_api_key:
        dependencies["llm"] = "configured"
    else:
        dependencies["llm"] = "not configured"

    # Determine overall status
    unhealthy_deps = [k for k, v in dependencies.items() if "unhealthy" in v.lower()]
    overall_status = "healthy" if not unhealthy_deps else "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        dependencies=dependencies,
    )


@router.get("/readiness")
async def readiness_check() -> dict:
    """
    Kubernetes readiness probe.

    Returns:
        Simple ready status
    """
    return {"ready": True}


@router.get("/liveness")
async def liveness_check() -> dict:
    """
    Kubernetes liveness probe.

    Returns:
        Simple alive status
    """
    return {"alive": True}
