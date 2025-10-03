"""Custom evaluation metrics for RAG system."""

from typing import List

from ..observability.logging import get_logger

logger = get_logger(__name__)


def answer_relevancy(query: str, answer: str, keywords: List[str]) -> float:
    """
    Calculate answer relevancy score.

    Args:
        query: User query
        answer: Generated answer
        keywords: Expected keywords

    Returns:
        Relevancy score 0-1
    """
    if not keywords:
        return 1.0

    answer_lower = answer.lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in answer_lower)

    return matches / len(keywords)


def context_precision(
    retrieved_docs: List[str],
    relevant_docs: List[str],
) -> float:
    """
    Calculate context precision.

    Args:
        retrieved_docs: Retrieved document IDs
        relevant_docs: Relevant document IDs

    Returns:
        Precision score 0-1
    """
    if not retrieved_docs:
        return 0.0

    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)

    true_positives = len(relevant_set & retrieved_set)

    return true_positives / len(retrieved_set)


def context_recall(
    retrieved_docs: List[str],
    relevant_docs: List[str],
) -> float:
    """
    Calculate context recall.

    Args:
        retrieved_docs: Retrieved document IDs
        relevant_docs: Relevant document IDs

    Returns:
        Recall score 0-1
    """
    if not relevant_docs:
        return 1.0

    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)

    true_positives = len(relevant_set & retrieved_set)

    return true_positives / len(relevant_set)


def answer_similarity(answer1: str, answer2: str) -> float:
    """
    Calculate simple word overlap similarity.

    Args:
        answer1: First answer
        answer2: Second answer

    Returns:
        Similarity score 0-1
    """
    words1 = set(answer1.lower().split())
    words2 = set(answer2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


class RAGMetrics:
    """Container for RAG evaluation metrics."""

    def __init__(self):
        """Initialize metrics."""
        self.total_queries = 0
        self.total_docs_retrieved = 0
        self.avg_reflection_score = 0.0
        self.scores = []

    def add_query_result(
        self,
        num_docs: int,
        reflection_score: float = None,
    ) -> None:
        """
        Add query result to metrics.

        Args:
            num_docs: Number of documents retrieved
            reflection_score: Reflection quality score
        """
        self.total_queries += 1
        self.total_docs_retrieved += num_docs

        if reflection_score is not None:
            self.scores.append(reflection_score)
            self.avg_reflection_score = sum(self.scores) / len(self.scores)

    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            "total_queries": self.total_queries,
            "avg_docs_per_query": (
                self.total_docs_retrieved / self.total_queries
                if self.total_queries > 0
                else 0
            ),
            "avg_reflection_score": self.avg_reflection_score,
        }
