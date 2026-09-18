"""Document ingestion endpoints."""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...api.dependencies import get_retriever_dependency, get_request_id
from ...api.models import BatchIngestRequest, IngestRequest, IngestResponse
from ...ingestion.chunking import DocumentChunker
from ...ingestion.document_loaders import (
    DirectoryLoader,
    Document,
    get_loader,
)
from ...observability.logging import get_logger
from ...retrieval.vector_store import VectorStoreRetriever
from ...utils.exceptions import IngestionError
from ...utils.config import get_settings

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])
ALLOWED_EXTENSIONS = {".html", ".htm", ".md", ".pdf", ".txt"}


def _validate_path(path_value: str, *, require_directory: bool = False) -> Path:
    """Resolve an ingestion path and keep it inside the configured data directory."""
    settings = get_settings()
    root = Path(settings.ingest_data_dir).expanduser().resolve()
    path = Path(path_value).expanduser().resolve()

    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Path must be inside the ingestion directory")
    if require_directory and not path.is_dir():
        raise HTTPException(status_code=400, detail="Ingestion directory does not exist")
    if not require_directory and (not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported or missing ingestion file")
    return path


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Save a validated upload without allowing unbounded request bodies."""
    settings = get_settings()
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                tmp.write(chunk)
            return tmp.name, filename
        except Exception:
            os.unlink(tmp.name)
            raise


@router.post("", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    retriever: VectorStoreRetriever = Depends(get_retriever_dependency),
    request_id: str = Depends(get_request_id),
) -> IngestResponse:
    """
    Ingest a single document.

    Args:
        request: Ingestion request
        retriever: Vector store retriever
        request_id: Request ID for tracking

    Returns:
        Ingestion response with status

    Raises:
        HTTPException: If ingestion fails
    """
    logger.info(f"Ingest request [{request_id}]")

    try:
        documents = []

        if request.file_path:
            # Load from file
            path = _validate_path(request.file_path)
            loader = get_loader(str(path))
            documents = await loader.load(str(path))

        elif request.text:
            # Create document from text
            documents = [
                Document(
                    content=request.text,
                    metadata=request.metadata,
                )
            ]

        else:
            raise HTTPException(
                status_code=400,
                detail="Either file_path or text must be provided",
            )

        # Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        # Add to vector store
        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        document_ids = await retriever.add_texts(
            texts=texts,
            metadata=metadata,
            ids=ids,
        )

        response = IngestResponse(
            success=True,
            num_documents=len(documents),
            num_chunks=len(chunks),
            document_ids=document_ids,
            message=f"Successfully ingested {len(documents)} documents",
        )

        logger.info(f"Ingestion completed [{request_id}]: {len(chunks)} chunks")
        return response

    except IngestionError as e:
        logger.error(f"Ingestion error [{request_id}]: {e}")
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


@router.post("/batch", response_model=IngestResponse)
async def ingest_directory(
    request: BatchIngestRequest,
    retriever: VectorStoreRetriever = Depends(get_retriever_dependency),
    request_id: str = Depends(get_request_id),
) -> IngestResponse:
    """
    Ingest all documents from a directory.

    Args:
        request: Batch ingestion request
        retriever: Vector store retriever
        request_id: Request ID for tracking

    Returns:
        Ingestion response with status
    """
    logger.info(f"Batch ingest request [{request_id}]: {request.directory}")

    try:
        # Load directory
        directory = _validate_path(request.directory, require_directory=True)
        loader = DirectoryLoader(recursive=request.recursive)
        documents = await loader.load(str(directory))

        if not documents:
            return IngestResponse(
                success=True,
                num_documents=0,
                num_chunks=0,
                document_ids=[],
                message="No documents found in directory",
            )

        # Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        # Add to vector store
        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        document_ids = await retriever.add_texts(
            texts=texts,
            metadata=metadata,
            ids=ids,
        )

        response = IngestResponse(
            success=True,
            num_documents=len(documents),
            num_chunks=len(chunks),
            document_ids=document_ids,
            message=f"Successfully ingested {len(documents)} documents from directory",
        )

        logger.info(f"Batch ingestion completed [{request_id}]: {len(chunks)} chunks")
        return response

    except Exception as e:
        logger.error(f"Batch ingestion error [{request_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Batch ingestion failed", "message": str(e)},
        )


@router.post("/upload", response_model=IngestResponse)
async def upload_file(
    file: UploadFile = File(...),
    retriever: VectorStoreRetriever = Depends(get_retriever_dependency),
    request_id: str = Depends(get_request_id),
) -> IngestResponse:
    """
    Upload and ingest a file.

    Args:
        file: Uploaded file
        retriever: Vector store retriever
        request_id: Request ID for tracking

    Returns:
        Ingestion response with status
    """
    logger.info(f"File upload [{request_id}]: {file.filename}")

    try:
        # Save uploaded file temporarily
        tmp_path, filename = await _save_upload(file)

        # Load and ingest
        try:
            loader = get_loader(tmp_path)
            documents = await loader.load(tmp_path)

            # Chunk and add
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(documents)

            texts = [chunk.text for chunk in chunks]
            metadata = [
                {**chunk.metadata, "filename": filename} for chunk in chunks
            ]
            ids = [chunk.chunk_id for chunk in chunks]

            document_ids = await retriever.add_texts(
                texts=texts,
                metadata=metadata,
                ids=ids,
            )

            return IngestResponse(
                success=True,
                num_documents=len(documents),
                num_chunks=len(chunks),
                document_ids=document_ids,
                message=f"Successfully uploaded and ingested {filename}",
            )

        finally:
            # Clean up temp file
            import os

            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Upload error [{request_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Upload failed", "message": str(e)},
        )
