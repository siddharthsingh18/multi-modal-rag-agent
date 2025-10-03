"""Document ingestion endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List

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

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


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
            loader = get_loader(request.file_path)
            documents = await loader.load(request.file_path)

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
        loader = DirectoryLoader(recursive=request.recursive)
        documents = await loader.load(request.directory)

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
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load and ingest
        try:
            loader = get_loader(tmp_path)
            documents = await loader.load(tmp_path)

            # Chunk and add
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(documents)

            texts = [chunk.text for chunk in chunks]
            metadata = [
                {**chunk.metadata, "filename": file.filename} for chunk in chunks
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
                message=f"Successfully uploaded and ingested {file.filename}",
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
