"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Summary

from ..observability.logging import get_logger

logger = get_logger(__name__)

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# RAG metrics
rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["status"],
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "RAG query processing time in seconds",
)

rag_documents_retrieved = Histogram(
    "rag_documents_retrieved",
    "Number of documents retrieved per query",
)

rag_reflection_score = Histogram(
    "rag_reflection_score",
    "RAG answer reflection scores",
)

# Ingestion metrics
ingestion_documents_total = Counter(
    "ingestion_documents_total",
    "Total documents ingested",
    ["status"],
)

ingestion_chunks_total = Counter(
    "ingestion_chunks_total",
    "Total chunks created",
)

ingestion_duration_seconds = Histogram(
    "ingestion_duration_seconds",
    "Document ingestion time in seconds",
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "status"],
)

llm_tokens_used = Counter(
    "llm_tokens_used",
    "Total tokens used",
    ["model", "type"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["model"],
)

# Vector DB metrics
vector_db_operations_total = Counter(
    "vector_db_operations_total",
    "Total vector DB operations",
    ["operation", "status"],
)

vector_db_operation_duration_seconds = Histogram(
    "vector_db_operation_duration_seconds",
    "Vector DB operation duration in seconds",
    ["operation"],
)

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
)

# System metrics
active_requests = Gauge(
    "active_requests",
    "Number of active requests",
)


logger.info("Prometheus metrics initialized")
