"""Embedding generation using sentence-transformers."""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize embedding service.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self.dimension = settings.embedding_dimension

        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Embedding model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RetrievalError(
                f"Failed to load embedding model {self.model_name}",
                original_error=e,
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector

        Raises:
            RetrievalError: If embedding generation fails
        """
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RetrievalError(
                "Embedding generation failed",
                details={"text_length": len(text)},
                original_error=e,
            )

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            List of embedding vectors

        Raises:
            RetrievalError: If batch embedding fails
        """
        try:
            logger.debug(f"Embedding batch of {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RetrievalError(
                "Batch embedding generation failed",
                details={"batch_size": len(texts)},
                original_error=e,
            )

    async def embed_text_async(self, text: str) -> List[float]:
        """
        Async wrapper for embed_text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Note: sentence-transformers doesn't have native async support
        # This is a simple wrapper; for production, consider using asyncio.to_thread
        return self.embed_text(text)

    async def embed_batch_async(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Async wrapper for embed_batch.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        return self.embed_batch(texts, batch_size=batch_size)

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
