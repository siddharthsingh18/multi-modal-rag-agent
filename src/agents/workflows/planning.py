"""Planning workflow for agent task decomposition."""

from typing import List

from ...generation.llm_client import LLMClient
from ...generation.prompt_templates import PromptTemplates
from ...observability.logging import get_logger

logger = get_logger(__name__)


class PlanningWorkflow:
    """Agent planning workflow."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize planning workflow.

        Args:
            llm_client: LLM client for plan generation
        """
        self.llm_client = llm_client

    async def create_plan(
        self,
        task: str,
        available_tools: List[str],
    ) -> List[str]:
        """
        Create a step-by-step plan for a task.

        Args:
            task: Task description
            available_tools: List of available tool names

        Returns:
            List of plan steps
        """
        logger.info(f"Creating plan for task: {task}")

        prompt = PromptTemplates.agent_planning_prompt(task, available_tools)

        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
        )

        # Parse plan steps
        steps = []
        for line in response.split("\n"):
            line = line.strip()
            # Remove numbering and bullet points
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                # Clean up step text
                step = line.lstrip("0123456789.-* ").strip()
                if step:
                    steps.append(step)

        logger.info(f"Created plan with {len(steps)} steps")
        return steps

    async def should_use_tool(
        self,
        step: str,
        available_tools: List[str],
    ) -> tuple[bool, str]:
        """
        Determine if a tool should be used for a step.

        Args:
            step: Plan step description
            available_tools: List of available tools

        Returns:
            Tuple of (should_use_tool, tool_name)
        """
        # Simple heuristic-based approach
        step_lower = step.lower()

        if "search" in step_lower or "find" in step_lower or "retrieve" in step_lower:
            if "retriever" in available_tools:
                return True, "retriever"

        if "calculate" in step_lower or "compute" in step_lower or any(
            op in step_lower for op in ["+", "-", "*", "/"]
        ):
            if "calculator" in available_tools:
                return True, "calculator"

        if "web" in step_lower or "internet" in step_lower or "online" in step_lower:
            if "web_search" in available_tools:
                return True, "web_search"

        if "code" in step_lower or "execute" in step_lower or "run" in step_lower:
            if "code_executor" in available_tools:
                return True, "code_executor"

        return False, ""
