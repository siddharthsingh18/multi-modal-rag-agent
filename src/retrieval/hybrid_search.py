"""Hybrid search combining dense and sparse retrieval."""

from typing import Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Okapi

from ..observability.logging import get_logger
from ..retrieval.embeddings import EmbeddingService, get_embedding_service
from ..retrieval.vector_store import VectorStoreRetriever
from ..utils.config import get_settings
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 sparse retrieval using keyword matching."""

    def __init__(self):
        """Initialize BM25 retriever."""
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_documents(
        self,
        documents: List[Tuple[str, str]],
    ) -> None:
        """
        Index documents for BM25 search.

        Args:
            documents: List of (doc_id, text) tuples
        """
        logger.info(f"Indexing {len(documents)} documents for BM25")

        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.corpus = [text for _, text in documents]

        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 indexing complete")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search documents using BM25.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples

        Raises:
            RetrievalError: If search fails
        """
        if self.bm25 is None:
            raise RetrievalError("BM25 index not initialized. Call index_documents first.")

        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            results = [
                (self.doc_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]

            logger.debug(f"BM25 found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise RetrievalError("BM25 search failed", original_error=e)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval."""

    def __init__(
        self,
        vector_retriever: VectorStoreRetriever,
        embedding_service: Optional[EmbeddingService] = None,
        alpha: float = 0.5,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Dense vector retriever
            embedding_service: Embedding service
            alpha: Weight for dense retrieval (1-alpha for sparse)
        """
        self.vector_retriever = vector_retriever
        self.embedding_service = embedding_service or get_embedding_service()
        self.bm25_retriever = BM25Retriever()
        self.alpha = alpha  # Dense weight
        self.settings = get_settings()

    async def index_documents(
        self,
        documents: List[Tuple[str, str]],
    ) -> None:
        """
        Index documents for both dense and sparse retrieval.

        Args:
            documents: List of (doc_id, text) tuples
        """
        logger.info(f"Indexing {len(documents)} documents for hybrid search")

        # Index for BM25 (sparse)
        self.bm25_retriever.index_documents(documents)

        # Index for vector search (dense)
        texts = [text for _, text in documents]
        ids = [doc_id for doc_id, _ in documents]
        await self.vector_retriever.add_texts(texts=texts, ids=ids)

        logger.info("Hybrid indexing complete")

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Hybrid search combining dense and sparse retrieval.

        Args:
            query: Query text
            top_k: Number of results to return
            alpha: Override default alpha weight

        Returns:
            List of (doc_id, score, text) tuples

        Raises:
            RetrievalError: If search fails
        """
        try:
            top_k = top_k or self.settings.retrieval_top_k
            alpha = alpha if alpha is not None else self.alpha

            logger.debug(f"Hybrid search with alpha={alpha}")

            # Dense retrieval
            dense_results = await self.vector_retriever.similarity_search(
                query=query,
                top_k=top_k * 2,  # Get more for fusion
            )

            # Sparse retrieval
            sparse_results = self.bm25_retriever.search(
                query=query,
                top_k=top_k * 2,
            )

            # Reciprocal rank fusion
            fused_results = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                alpha=alpha,
                top_k=top_k,
            )

            logger.info(f"Hybrid search returned {len(fused_results)} results")
            return fused_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError("Hybrid search failed", original_error=e)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float, str]],
        sparse_results: List[Tuple[str, float]],
        alpha: float,
        top_k: int,
        k: int = 60,
    ) -> List[Tuple[str, float, str]]:
        """
        Combine results using reciprocal rank fusion.

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            alpha: Weight for dense results
            top_k: Number of final results
            k: RRF constant (typically 60)

        Returns:
            Fused and ranked results
        """
        # Build document map from dense results
        doc_map: Dict[str, str] = {
            doc_id: text for doc_id, _, text in dense_results
        }

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}

        # Dense scores (weighted by alpha)
        for rank, (doc_id, score, text) in enumerate(dense_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + alpha / (k + rank)
            doc_map[doc_id] = text

        # Sparse scores (weighted by 1-alpha)
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1 - alpha) / (k + rank)

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # Format results
        results = [
            (doc_id, score, doc_map.get(doc_id, ""))
            for doc_id, score in sorted_docs
            if doc_id in doc_map
        ]

        return results


class MultiQueryRetriever:
    """Retriever that generates multiple queries for better coverage."""

    def __init__(
        self,
        base_retriever: VectorStoreRetriever,
        llm_client=None,
    ):
        """
        Initialize multi-query retriever.

        Args:
            base_retriever: Base retriever to use
            llm_client: LLM client for query generation
        """
        self.base_retriever = base_retriever
        self.llm_client = llm_client

    async def search(
        self,
        query: str,
        num_queries: int = 3,
        top_k: int = 10,
    ) -> List[Tuple[str, float, str]]:
        """
        Search using multiple generated queries.

        Args:
            query: Original query
            num_queries: Number of queries to generate
            top_k: Number of results per query

        Returns:
            Deduplicated and ranked results
        """
        queries = await self._generate_queries(query, num_queries)

        # Search with each query
        all_results: List[Tuple[str, float, str]] = []
        seen_ids: Set[str] = set()

        for q in queries:
            results = await self.base_retriever.similarity_search(
                query=q,
                top_k=top_k,
            )

            for doc_id, score, text in results:
                if doc_id not in seen_ids:
                    all_results.append((doc_id, score, text))
                    seen_ids.add(doc_id)

        # Re-rank by score
        all_results.sort(key=lambda x: x[1], reverse=True)

        return all_results[:top_k]

    async def _generate_queries(
        self,
        query: str,
        num_queries: int,
    ) -> List[str]:
        """
        Generate multiple queries from original.

        Args:
            query: Original query
            num_queries: Number of queries to generate

        Returns:
            List of queries including original
        """
        if self.llm_client is None:
            # Return original query if no LLM
            return [query]

        from ..generation.prompt_templates import PromptTemplates

        prompt = PromptTemplates.multi_query_prompt(query)

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )

            # Parse generated queries
            generated = [q.strip() for q in response.split("\n") if q.strip()]
            queries = [query] + generated[:num_queries - 1]

            logger.debug(f"Generated {len(queries)} queries")
            return queries

        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return [query]
