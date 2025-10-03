"""High-level vector store wrapper for retrieval operations."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..database.vector_db import VectorStore, get_vector_store
from ..observability.logging import get_logger
from ..retrieval.embeddings import EmbeddingService, get_embedding_service
from ..utils.config import get_settings
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class VectorStoreRetriever:
    """High-level retriever combining embedding and vector store."""

    def __init__(
        self,
        collection_name: str = "documents",
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize retriever.

        Args:
            collection_name: Name of vector store collection
            vector_store: Vector store instance
            embedding_service: Embedding service instance
        """
        self.collection_name = collection_name
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.settings = get_settings()

    async def initialize(self) -> None:
        """Initialize collection."""
        await self.vector_store.create_collection(
            name=self.collection_name,
            dimension=self.settings.embedding_dimension,
        )
        logger.info(f"Initialized collection: {self.collection_name}")

    async def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to vector store.

        Args:
            texts: List of texts to add
            metadata: Optional metadata for each text
            ids: Optional IDs for each text

        Returns:
            List of document IDs

        Raises:
            RetrievalError: If adding texts fails
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = await self.embedding_service.embed_batch_async(texts)

            # Add to vector store
            await self.vector_store.add_documents(
                collection_name=self.collection_name,
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadata=metadata,
            )

            logger.info(f"Added {len(texts)} texts to vector store")
            return ids

        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise RetrievalError(
                "Failed to add texts to vector store",
                original_error=e,
            )

    async def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Perform similarity search.

        Args:
            query: Query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of (id, score, text) tuples

        Raises:
            RetrievalError: If search fails
        """
        try:
            top_k = top_k or self.settings.retrieval_top_k

            # Generate query embedding
            logger.debug(f"Generating query embedding")
            query_embedding = await self.embedding_service.embed_text_async(query)

            # Search vector store
            results = await self.vector_store.search(
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            # Format results
            formatted_results = [
                (doc_id, score, payload.get("text", ""))
                for doc_id, score, payload in results
            ]

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RetrievalError(
                "Similarity search failed",
                original_error=e,
            )

    async def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Raises:
            RetrievalError: If deletion fails
        """
        try:
            await self.vector_store.delete_documents(
                collection_name=self.collection_name,
                ids=ids,
            )
            logger.info(f"Deleted {len(ids)} documents")

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RetrievalError(
                "Failed to delete documents",
                original_error=e,
            )

    async def get_relevant_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Get relevant document texts for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of relevant document texts
        """
        results = await self.similarity_search(query=query, top_k=top_k)
        return [text for _, _, text in results]


# Global retriever instance
_retriever: Optional[VectorStoreRetriever] = None


async def get_retriever(collection_name: str = "documents") -> VectorStoreRetriever:
    """
    Get or create global retriever instance.

    Args:
        collection_name: Collection name

    Returns:
        VectorStoreRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = VectorStoreRetriever(collection_name=collection_name)
        await _retriever.initialize()
    return _retriever
