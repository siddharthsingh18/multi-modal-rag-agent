"""Re-ranking module for improving retrieval quality."""

from typing import List, Tuple

from ..observability.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


class ReRanker:
    """Base re-ranker class."""

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Re-rank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of (doc_id, score, text) tuples
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents
        """
        raise NotImplementedError


class SimpleScoreReRanker(ReRanker):
    """Simple re-ranker based on existing scores."""

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Re-rank by sorting existing scores.

        Args:
            query: Query text (not used)
            documents: List of (doc_id, score, text) tuples
            top_k: Number of top documents to return

        Returns:
            Top-k documents sorted by score
        """
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class KeywordReRanker(ReRanker):
    """Re-ranker that boosts documents containing query keywords."""

    def __init__(self, keyword_weight: float = 0.3):
        """
        Initialize keyword re-ranker.

        Args:
            keyword_weight: Weight for keyword matching boost
        """
        self.keyword_weight = keyword_weight

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Re-rank documents with keyword matching boost.

        Args:
            query: Query text
            documents: List of (doc_id, score, text) tuples
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents
        """
        query_terms = set(query.lower().split())

        reranked = []
        for doc_id, score, text in documents:
            # Count keyword matches
            text_lower = text.lower()
            matches = sum(1 for term in query_terms if term in text_lower)

            # Boost score based on keyword matches
            keyword_boost = matches / len(query_terms) if query_terms else 0
            new_score = score + (keyword_boost * self.keyword_weight)

            reranked.append((doc_id, new_score, text))

        # Sort by new score
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(documents)} documents with keyword boosting")
        return reranked[:top_k]


class LengthNormalizedReRanker(ReRanker):
    """Re-ranker that normalizes scores by document length."""

    def __init__(self, length_penalty: float = 0.1):
        """
        Initialize length-normalized re-ranker.

        Args:
            length_penalty: Penalty factor for longer documents
        """
        self.length_penalty = length_penalty

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Re-rank documents with length normalization.

        Args:
            query: Query text
            documents: List of (doc_id, score, text) tuples
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents
        """
        if not documents:
            return []

        # Calculate average document length
        avg_length = sum(len(text) for _, _, text in documents) / len(documents)

        reranked = []
        for doc_id, score, text in documents:
            # Apply length penalty
            length_ratio = len(text) / avg_length if avg_length > 0 else 1
            length_factor = 1 / (1 + self.length_penalty * abs(length_ratio - 1))

            new_score = score * length_factor
            reranked.append((doc_id, new_score, text))

        # Sort by new score
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(documents)} documents with length normalization")
        return reranked[:top_k]


class CombinedReRanker(ReRanker):
    """Combine multiple re-ranking strategies."""

    def __init__(
        self,
        use_keywords: bool = True,
        use_length_norm: bool = True,
        keyword_weight: float = 0.3,
        length_penalty: float = 0.1,
    ):
        """
        Initialize combined re-ranker.

        Args:
            use_keywords: Whether to use keyword boosting
            use_length_norm: Whether to use length normalization
            keyword_weight: Weight for keyword matching
            length_penalty: Penalty for length normalization
        """
        self.rerankers = []

        if use_keywords:
            self.rerankers.append(KeywordReRanker(keyword_weight=keyword_weight))

        if use_length_norm:
            self.rerankers.append(
                LengthNormalizedReRanker(length_penalty=length_penalty)
            )

    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """
        Apply multiple re-ranking strategies.

        Args:
            query: Query text
            documents: List of (doc_id, score, text) tuples
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents
        """
        current_docs = documents

        # Apply each re-ranker sequentially
        for reranker in self.rerankers:
            current_docs = await reranker.rerank(
                query=query,
                documents=current_docs,
                top_k=len(current_docs),  # Keep all for intermediate steps
            )

        logger.debug(f"Applied {len(self.rerankers)} re-ranking strategies")
        return current_docs[:top_k]


# Factory function
def get_reranker(strategy: str = "combined") -> ReRanker:
    """
    Get re-ranker instance based on strategy.

    Args:
        strategy: Re-ranking strategy ('simple', 'keyword', 'length', 'combined')

    Returns:
        ReRanker instance
    """
    settings = get_settings()

    if strategy == "simple":
        return SimpleScoreReRanker()
    elif strategy == "keyword":
        return KeywordReRanker()
    elif strategy == "length":
        return LengthNormalizedReRanker()
    elif strategy == "combined":
        return CombinedReRanker()
    else:
        logger.warning(f"Unknown re-ranker strategy: {strategy}, using simple")
        return SimpleScoreReRanker()
