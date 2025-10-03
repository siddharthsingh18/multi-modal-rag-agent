"""Unit tests for retrieval components."""

import pytest

from src.retrieval.embeddings import EmbeddingService


class TestEmbeddingService:
    """Tests for embedding service."""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance."""
        return EmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

    def test_embed_text(self, embedding_service):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = embedding_service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, embedding_service):
        """Test batch text embedding."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]

        embeddings = embedding_service.embed_batch(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_similarity(self, embedding_service):
        """Test similarity calculation."""
        text1 = "Machine learning is great."
        text2 = "Machine learning is awesome."
        text3 = "The weather is nice today."

        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        emb3 = embedding_service.embed_text(text3)

        # Similar texts should have higher similarity
        sim_similar = embedding_service.similarity(emb1, emb2)
        sim_different = embedding_service.similarity(emb1, emb3)

        assert sim_similar > sim_different
        assert 0 <= sim_similar <= 1
        assert 0 <= sim_different <= 1
