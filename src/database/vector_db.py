"""Vector database interface supporting Qdrant and ChromaDB."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from qdrant_client import AsyncQdrantClient, models

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import DatabaseError

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    async def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to collection."""
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_documents(
        self,
        collection_name: str,
        ids: List[str],
    ) -> None:
        """Delete documents from collection."""
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant vector database implementation."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
        """
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key

        logger.info(f"Connecting to Qdrant at {self.url}")
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key if self.api_key else None,
        )

    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection in Qdrant."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if name in collection_names:
                logger.info(f"Collection {name} already exists")
                return

            await self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise DatabaseError(
                f"Failed to create collection {name}",
                original_error=e,
            )

    async def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to Qdrant collection."""
        try:
            points = []
            for i, (doc_id, embedding, document) in enumerate(zip(ids, embeddings, documents)):
                payload = {
                    "text": document,
                    "metadata": metadata[i] if metadata else {},
                }
                points.append(
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            await self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            logger.info(f"Added {len(points)} documents to {collection_name}")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise DatabaseError(
                f"Failed to add documents to {collection_name}",
                original_error=e,
            )

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search Qdrant collection."""
        try:
            # Build filter if provided
            search_filter = None
            if filter_dict:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in filter_dict.items()
                    ]
                )

            results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )

            # Format results as (id, score, payload)
            formatted_results = [
                (
                    str(result.id),
                    result.score,
                    result.payload,
                )
                for result in results
            ]

            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise DatabaseError(
                f"Search failed in {collection_name}",
                original_error=e,
            )

    async def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """Delete documents from Qdrant."""
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=ids,
                ),
            )
            logger.info(f"Deleted {len(ids)} documents from {collection_name}")

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise DatabaseError(
                f"Failed to delete documents from {collection_name}",
                original_error=e,
            )


class ChromaVectorStore(VectorStore):
    """ChromaDB vector database implementation."""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory for persistent storage
        """
        settings = get_settings()
        self.persist_directory = persist_directory or settings.chroma_persist_dir

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
        )

    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection in ChromaDB."""
        try:
            # ChromaDB creates collections on first access
            self.client.get_or_create_collection(name=name)
            logger.info(f"Created ChromaDB collection: {name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise DatabaseError(
                f"Failed to create collection {name}",
                original_error=e,
            )

    async def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to ChromaDB collection."""
        try:
            collection = self.client.get_or_create_collection(name=collection_name)

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
            )
            logger.info(f"Added {len(ids)} documents to {collection_name}")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise DatabaseError(
                f"Failed to add documents to {collection_name}",
                original_error=e,
            )

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search ChromaDB collection."""
        try:
            collection = self.client.get_collection(name=collection_name)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
            )

            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        (
                            results["ids"][0][i],
                            results["distances"][0][i],
                            {
                                "text": results["documents"][0][i],
                                "metadata": results["metadatas"][0][i]
                                if results["metadatas"]
                                else {},
                            },
                        )
                    )

            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise DatabaseError(
                f"Search failed in {collection_name}",
                original_error=e,
            )

    async def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from {collection_name}")

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise DatabaseError(
                f"Failed to delete documents from {collection_name}",
                original_error=e,
            )


# Factory function
def get_vector_store(store_type: Optional[str] = None) -> VectorStore:
    """
    Get vector store instance based on configuration.

    Args:
        store_type: Type of vector store ('qdrant' or 'chroma')

    Returns:
        VectorStore instance
    """
    settings = get_settings()
    store_type = store_type or settings.vector_db_type

    if store_type.lower() == "qdrant":
        return QdrantVectorStore()
    elif store_type.lower() == "chroma":
        return ChromaVectorStore()
    else:
        raise DatabaseError(f"Unknown vector store type: {store_type}")
