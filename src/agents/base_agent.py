"""Base agent class and agent state management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    """State container for agent execution."""

    # Input
    query: str = ""
    context: str = ""

    # Processing
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    thoughts: List[str] = field(default_factory=list)

    # Retrieved information
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)

    # Output
    answer: str = ""
    reflection: str = ""
    confidence: float = 0.0

    # Metadata
    iterations: int = 0
    max_iterations: int = 10
    should_continue: bool = True

    def add_thought(self, thought: str) -> None:
        """Add a thought to the reasoning chain."""
        self.thoughts.append(thought)
        logger.debug(f"Agent thought: {thought}")

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Store result from a tool execution."""
        self.tool_results[tool_name] = result
        logger.debug(f"Tool result stored: {tool_name}")

    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.iterations += 1
        if self.iterations >= self.max_iterations:
            self.should_continue = False
            logger.warning(f"Max iterations ({self.max_iterations}) reached")


class BaseAgent(ABC):
    """Abstract base class for agents."""

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name
        """
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute agent workflow.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            Agent execution results
        """
        pass

    def create_state(self, query: str, **kwargs) -> AgentState:
        """
        Create initial agent state.

        Args:
            query: User query
            **kwargs: Additional state parameters

        Returns:
            Initial AgentState
        """
        state = AgentState(query=query)

        # Update with any additional parameters
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        return state

    async def should_continue_execution(self, state: AgentState) -> bool:
        """
        Determine if agent should continue execution.

        Args:
            state: Current agent state

        Returns:
            True if should continue
        """
        if not state.should_continue:
            return False

        if state.iterations >= state.max_iterations:
            return False

        return True


class Tool(ABC):
    """Abstract base class for agent tools."""

    def __init__(self, name: str, description: str):
        """
        Initialize tool.

        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        pass

    def __repr__(self) -> str:
        return f"Tool(name={self.name})"
