"""FastAPI main application."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..api.routes import health, ingest, query
from ..api.rate_limit import SlidingWindowRateLimiter
from ..observability.logging import get_logger, setup_logging
from ..utils.config import get_settings
from ..utils.exceptions import RAGException

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()
rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting RAG application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Vector DB: {settings.vector_db_type}")

    yield

    # Shutdown
    logger.info("Shutting down RAG application...")


# Create FastAPI app
app = FastAPI(
    title="Multi-Modal RAG Agent API",
    description="Production-grade RAG system with agentic workflows",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def enforce_rate_limit(request: Request, call_next):
    """Reject requests that exceed the configured per-client limit."""
    if request.url.path.startswith("/api/v1"):
        client_key = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_key):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )

    return await call_next(request)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    request_id = request.headers.get("X-Request-ID", "unknown")

    logger.info(
        f"Request [{request_id}]: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    response = await call_next(request)

    logger.info(
        f"Response [{request_id}]: {response.status_code}",
        extra={"request_id": request_id},
    )

    return response


# Exception handlers
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle RAG exceptions."""
    logger.error(f"RAG exception: {exc.message}", exc_info=exc.original_error)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    logger.error(f"Value error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": "ValueError",
            "message": str(exc),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred",
        },
    )


# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Modal RAG Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# API info endpoint
@app.get("/api/v1")
async def api_info():
    """API version information."""
    return {
        "version": "v1",
        "endpoints": {
            "query": "/api/v1/query",
            "ingest": "/api/v1/ingest",
            "health": "/api/v1/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
