"""Retriever tool for agent workflows."""

from typing import Any, Dict, List

from ...agents.base_agent import Tool
from ...observability.logging import get_logger
from ...retrieval.vector_store import VectorStoreRetriever

logger = get_logger(__name__)


class RetrieverTool(Tool):
    """Tool for retrieving relevant documents."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        top_k: int = 5,
    ):
        """
        Initialize retriever tool.

        Args:
            retriever: Vector store retriever instance
            top_k: Number of documents to retrieve
        """
        super().__init__(
            name="retriever",
            description="Retrieve relevant documents from the knowledge base",
        )
        self.retriever = retriever
        self.top_k = top_k

    async def execute(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)

        Returns:
            List of retrieved document dictionaries
        """
        k = top_k or self.top_k

        logger.info(f"Retrieving {k} documents for query: {query}")

        results = await self.retriever.similarity_search(query=query, top_k=k)

        # Format as dictionaries
        documents = [
            {
                "id": doc_id,
                "score": score,
                "content": text,
            }
            for doc_id, score, text in results
        ]

        logger.info(f"Retrieved {len(documents)} documents")
        return documents
