"""Query transformation and rewriting for better retrieval."""

from typing import List, Optional

from ..observability.logging import get_logger
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class QueryTransformer:
    """Transform and rewrite queries for better retrieval."""

    def __init__(self, llm_client=None):
        """
        Initialize query transformer.

        Args:
            llm_client: LLM client for query rewriting
        """
        self.llm_client = llm_client

    async def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query

        Returns:
            Rewritten query
        """
        if self.llm_client is None:
            return query

        from ..generation.prompt_templates import PromptTemplates

        prompt = PromptTemplates.query_rewrite_prompt(query)

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=150,
            )

            # Take the first rewritten query
            rewritten = response.split("\n")[0].strip()
            logger.debug(f"Rewritten query: {query} -> {rewritten}")

            return rewritten if rewritten else query

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query

    async def expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple variations.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        if self.llm_client is None:
            return [query]

        from ..generation.prompt_templates import PromptTemplates

        prompt = PromptTemplates.multi_query_prompt(query)

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )

            # Parse variations
            variations = [q.strip() for q in response.split("\n") if q.strip()]
            all_queries = [query] + variations

            logger.debug(f"Expanded query into {len(all_queries)} variations")
            return all_queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    def add_context(self, query: str, context: str) -> str:
        """
        Add conversation context to query.

        Args:
            query: Current query
            context: Previous context

        Returns:
            Query with context
        """
        if not context:
            return query

        return f"Context: {context}\n\nQuestion: {query}"

    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.

        Args:
            query: Query text

        Returns:
            List of keywords
        """
        # Simple keyword extraction
        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "what",
            "when",
            "where",
            "who",
            "how",
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords


class HyDETransformer:
    """Hypothetical Document Embeddings (HyDE) transformer."""

    def __init__(self, llm_client):
        """
        Initialize HyDE transformer.

        Args:
            llm_client: LLM client for generating hypothetical documents
        """
        self.llm_client = llm_client

    async def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        prompt = f"""Generate a detailed passage that would answer the following question.
Write as if you are an expert providing a comprehensive answer.

Question: {query}

Hypothetical passage:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=300,
            )

            logger.debug(f"Generated hypothetical document for query")
            return response.strip()

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            raise RetrievalError(
                "Failed to generate hypothetical document",
                original_error=e,
            )


class QueryDecomposer:
    """Decompose complex queries into sub-queries."""

    def __init__(self, llm_client):
        """
        Initialize query decomposer.

        Args:
            llm_client: LLM client for decomposition
        """
        self.llm_client = llm_client

    async def decompose(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        prompt = f"""Break down the following complex question into 2-4 simpler sub-questions.
Each sub-question should focus on one specific aspect.

Complex question: {query}

Sub-questions (one per line):"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )

            # Parse sub-queries
            sub_queries = [q.strip() for q in response.split("\n") if q.strip()]

            logger.debug(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries if sub_queries else [query]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]


# Global transformer instance
_query_transformer: Optional[QueryTransformer] = None


def get_query_transformer(llm_client=None) -> QueryTransformer:
    """
    Get global query transformer instance.

    Args:
        llm_client: Optional LLM client

    Returns:
        QueryTransformer instance
    """
    global _query_transformer
    if _query_transformer is None:
        _query_transformer = QueryTransformer(llm_client=llm_client)
    return _query_transformer
