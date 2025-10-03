"""LangSmith tracing integration."""

import os
from typing import Any, Dict, Optional

from ..observability.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


def setup_langsmith() -> None:
    """Configure LangSmith tracing."""
    settings = get_settings()

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

        logger.info(f"LangSmith tracing enabled for project: {settings.langsmith_project}")
    else:
        logger.info("LangSmith tracing disabled")


def trace_event(
    event_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a trace event.

    Args:
        event_name: Event name
        metadata: Event metadata
    """
    logger.debug(f"Trace event: {event_name}", extra=metadata or {})


# Initialize on module import
setup_langsmith()
