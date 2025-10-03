"""Main RAG agent with LangGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.base_agent import AgentState, BaseAgent
from ..agents.tools.calculator import CalculatorTool
from ..agents.tools.code_executor import CodeExecutorTool
from ..agents.tools.retriever import RetrieverTool
from ..agents.tools.web_search import WebSearchTool
from ..agents.workflows.planning import PlanningWorkflow
from ..agents.workflows.reflection import ReflectionWorkflow
from ..generation.llm_client import LLMClient, get_llm_client
from ..generation.prompt_templates import PromptTemplates
from ..observability.logging import get_logger
from ..retrieval.hybrid_search import HybridRetriever
from ..retrieval.reranker import get_reranker
from ..retrieval.vector_store import VectorStoreRetriever
from ..utils.config import get_settings
from ..utils.exceptions import AgentError

logger = get_logger(__name__)


class GraphState(TypedDict):
    """LangGraph state dictionary."""

    query: str
    retrieved_documents: List[Dict[str, Any]]
    answer: str
    plan: List[str]
    current_step: int
    reflection: Dict[str, Any]
    should_continue: bool
    iterations: int


class RAGAgent(BaseAgent):
    """RAG Agent with LangGraph workflow orchestration."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm_client: Optional[LLMClient] = None,
        use_hybrid_search: bool = True,
        use_reflection: bool = True,
    ):
        """
        Initialize RAG agent.

        Args:
            retriever: Vector store retriever
            llm_client: LLM client (defaults to global instance)
            use_hybrid_search: Whether to use hybrid search
            use_reflection: Whether to use reflection loop
        """
        super().__init__(name="rag_agent")

        self.retriever = retriever
        self.llm_client = llm_client or get_llm_client()
        self.settings = get_settings()
        self.use_reflection = use_reflection

        # Initialize tools
        self.tools = {
            "retriever": RetrieverTool(
                retriever=retriever,
                top_k=self.settings.retrieval_top_k,
            ),
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "code_executor": CodeExecutorTool(),
        }

        # Initialize workflows
        self.planning = PlanningWorkflow(llm_client=self.llm_client)
        self.reflection_workflow = ReflectionWorkflow(llm_client=self.llm_client)

        # Initialize re-ranker
        self.reranker = get_reranker(strategy="combined")

        # Build LangGraph workflow
        self.graph = self._build_graph()

        logger.info("RAG Agent initialized with LangGraph workflow")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("generate", self._generate_node)

        if self.use_reflection:
            workflow.add_node("reflect", self._reflect_node)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Add edges
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")

        if self.use_reflection:
            workflow.add_edge("generate", "reflect")
            workflow.add_conditional_edges(
                "reflect",
                self._should_retry,
                {
                    "retry": "generate",
                    "finish": END,
                },
            )
        else:
            workflow.add_edge("generate", END)

        return workflow.compile()

    async def _retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieval node."""
        logger.info("Executing retrieval node")

        query = state["query"]

        # Retrieve documents
        results = await self.retriever.similarity_search(
            query=query,
            top_k=self.settings.retrieval_top_k,
        )

        # Format documents
        documents = [
            {
                "id": doc_id,
                "score": score,
                "content": text,
            }
            for doc_id, score, text in results
        ]

        state["retrieved_documents"] = documents
        logger.info(f"Retrieved {len(documents)} documents")

        return state

    async def _rerank_node(self, state: GraphState) -> GraphState:
        """Re-ranking node."""
        logger.info("Executing re-ranking node")

        query = state["query"]
        documents = state["retrieved_documents"]

        if not documents:
            return state

        # Convert to re-ranker format
        docs_for_rerank = [
            (doc["id"], doc["score"], doc["content"]) for doc in documents
        ]

        # Re-rank
        reranked = await self.reranker.rerank(
            query=query,
            documents=docs_for_rerank,
            top_k=self.settings.rerank_top_k,
        )

        # Convert back
        state["retrieved_documents"] = [
            {
                "id": doc_id,
                "score": score,
                "content": text,
            }
            for doc_id, score, text in reranked
        ]

        logger.info(f"Re-ranked to top {len(state['retrieved_documents'])} documents")

        return state

    async def _generate_node(self, state: GraphState) -> GraphState:
        """Answer generation node."""
        logger.info("Executing generation node")

        query = state["query"]
        documents = state["retrieved_documents"]

        # Extract context
        context = [doc["content"] for doc in documents]

        # Generate answer
        prompt = PromptTemplates.rag_prompt(query=query, context=context)
        system_prompt = PromptTemplates.system_prompt()

        answer = await self.llm_client.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

        state["answer"] = answer
        logger.info("Generated answer")

        return state

    async def _reflect_node(self, state: GraphState) -> GraphState:
        """Reflection node."""
        logger.info("Executing reflection node")

        query = state["query"]
        answer = state["answer"]
        documents = state["retrieved_documents"]

        context = [doc["content"] for doc in documents]

        # Reflect on answer quality
        reflection = await self.reflection_workflow.reflect_on_answer(
            query=query,
            answer=answer,
            context=context,
        )

        state["reflection"] = reflection
        state["iterations"] = state.get("iterations", 0) + 1

        logger.info(f"Reflection score: {reflection['score']}/10")

        return state

    def _should_retry(self, state: GraphState) -> str:
        """Determine if answer should be regenerated."""
        reflection = state.get("reflection", {})
        iterations = state.get("iterations", 0)

        if iterations >= 2:  # Max 2 retries
            return "finish"

        if reflection.get("score", 10) < 6.0:
            logger.info("Low reflection score, retrying generation")
            return "retry"

        return "finish"

    async def run(
        self,
        query: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute RAG agent workflow.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            Agent execution results with answer and metadata
        """
        logger.info(f"Running RAG agent for query: {query}")

        try:
            # Create initial state
            initial_state: GraphState = {
                "query": query,
                "retrieved_documents": [],
                "answer": "",
                "plan": [],
                "current_step": 0,
                "reflection": {},
                "should_continue": True,
                "iterations": 0,
            }

            # Execute graph
            final_state = await self.graph.ainvoke(initial_state)

            # Format results
            result = {
                "query": query,
                "answer": final_state["answer"],
                "documents": final_state["retrieved_documents"],
                "reflection": final_state.get("reflection", {}),
                "metadata": {
                    "iterations": final_state.get("iterations", 0),
                    "num_documents": len(final_state["retrieved_documents"]),
                },
            }

            logger.info("RAG agent execution complete")
            return result

        except Exception as e:
            logger.error(f"RAG agent execution failed: {e}")
            raise AgentError(
                "RAG agent execution failed",
                details={"query": query},
                original_error=e,
            )

    async def stream(
        self,
        query: str,
        **kwargs,
    ):
        """
        Stream RAG agent execution.

        Args:
            query: User query
            **kwargs: Additional parameters

        Yields:
            Execution progress updates
        """
        logger.info(f"Streaming RAG agent for query: {query}")

        # Create initial state
        initial_state: GraphState = {
            "query": query,
            "retrieved_documents": [],
            "answer": "",
            "plan": [],
            "current_step": 0,
            "reflection": {},
            "should_continue": True,
            "iterations": 0,
        }

        # Stream graph execution
        async for event in self.graph.astream(initial_state):
            yield event


# Factory function
async def create_rag_agent(
    retriever: VectorStoreRetriever,
    use_reflection: bool = True,
) -> RAGAgent:
    """
    Create and initialize RAG agent.

    Args:
        retriever: Vector store retriever
        use_reflection: Whether to use reflection

    Returns:
        RAGAgent instance
    """
    agent = RAGAgent(
        retriever=retriever,
        use_reflection=use_reflection,
    )

    logger.info("RAG agent created successfully")
    return agent
