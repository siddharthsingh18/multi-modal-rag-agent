"""Pydantic models for API request/response schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Request models
class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str = Field(..., description="User query", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1)
    use_reflection: Optional[bool] = Field(True, description="Use reflection for quality")
    stream: Optional[bool] = Field(False, description="Stream response")

    model_config = {"json_schema_extra": {"example": {"query": "What is machine learning?"}}}


class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    file_path: Optional[str] = Field(None, description="Path to file to ingest")
    text: Optional[str] = Field(None, description="Raw text to ingest")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Document metadata"
    )
    collection_name: Optional[str] = Field(
        "documents", description="Collection name for storage"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This is a sample document",
                "metadata": {"source": "api", "type": "text"},
            }
        }
    }


class BatchIngestRequest(BaseModel):
    """Request model for batch document ingestion."""

    directory: str = Field(..., description="Directory path to ingest")
    recursive: Optional[bool] = Field(True, description="Recursively ingest subdirectories")
    collection_name: Optional[str] = Field(
        "documents", description="Collection name for storage"
    )

    model_config = {"json_schema_extra": {"example": {"directory": "/path/to/documents"}}}


# Response models
class DocumentResponse(BaseModel):
    """Document information in response."""

    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    content: str = Field(..., description="Document content")


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    documents: List[DocumentResponse] = Field(..., description="Retrieved documents")
    reflection: Optional[Dict[str, Any]] = Field(None, description="Reflection results")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI...",
                "documents": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "content": "Machine learning involves...",
                    }
                ],
                "metadata": {"num_documents": 5, "iterations": 1},
            }
        }
    }


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    success: bool = Field(..., description="Whether ingestion succeeded")
    num_documents: int = Field(..., description="Number of documents ingested")
    num_chunks: int = Field(..., description="Number of chunks created")
    document_ids: List[str] = Field(..., description="IDs of ingested documents")
    message: str = Field(..., description="Status message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "num_documents": 5,
                "num_chunks": 120,
                "document_ids": ["doc1", "doc2"],
                "message": "Successfully ingested 5 documents",
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    dependencies: Dict[str, str] = Field(..., description="Dependency health status")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "dependencies": {
                    "llm": "healthy",
                    "vector_db": "healthy",
                    "redis": "healthy",
                },
            }
        }
    }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_type": "ValidationError",
                "message": "Invalid query parameter",
                "details": {"field": "query", "issue": "must not be empty"},
            }
        }
    }
