"""Multi-step workflow orchestration."""

from typing import Any, Dict, List

from ...agents.base_agent import AgentState
from ...generation.llm_client import LLMClient
from ...observability.logging import get_logger

logger = get_logger(__name__)


class MultiStepWorkflow:
    """Multi-step workflow for complex tasks."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize multi-step workflow.

        Args:
            llm_client: LLM client
        """
        self.llm_client = llm_client

    async def execute_step(
        self,
        step: str,
        state: AgentState,
        tools: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step.

        Args:
            step: Step description
            state: Current agent state
            tools: Available tools

        Returns:
            Step execution results
        """
        logger.info(f"Executing step: {step}")

        state.add_thought(f"Executing: {step}")

        # Determine if tool use is needed
        tool_name = self._identify_tool(step, tools)

        if tool_name and tool_name in tools:
            # Execute tool
            tool = tools[tool_name]
            result = await self._execute_tool(tool, step, state)
            state.add_tool_result(tool_name, result)

            return {
                "success": True,
                "tool_used": tool_name,
                "result": result,
            }
        else:
            # Use LLM to process step
            result = await self._process_with_llm(step, state)

            return {
                "success": True,
                "tool_used": None,
                "result": result,
            }

    def _identify_tool(self, step: str, tools: Dict[str, Any]) -> str:
        """Identify which tool to use for a step."""
        step_lower = step.lower()

        if "retriev" in step_lower or "search" in step_lower or "find" in step_lower:
            return "retriever"

        if "calculat" in step_lower or "compute" in step_lower:
            return "calculator"

        if "web" in step_lower or "internet" in step_lower:
            return "web_search"

        if "code" in step_lower or "execute" in step_lower:
            return "code_executor"

        return ""

    async def _execute_tool(
        self,
        tool: Any,
        step: str,
        state: AgentState,
    ) -> Any:
        """Execute a tool for a step."""
        logger.debug(f"Executing tool: {tool.name}")

        try:
            if tool.name == "retriever":
                return await tool.execute(query=state.query)

            elif tool.name == "calculator":
                # Extract expression from step
                # Simple heuristic: look for mathematical expression
                import re

                expr_match = re.search(r"[\d\+\-\*/\(\)\s]+", step)
                if expr_match:
                    expression = expr_match.group(0).strip()
                    return await tool.execute(expression=expression)

            elif tool.name == "web_search":
                return await tool.execute(query=state.query)

            elif tool.name == "code_executor":
                # This requires code in the step - simplified for now
                return {"success": False, "error": "Code extraction not implemented"}

            return await tool.execute()

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}

    async def _process_with_llm(self, step: str, state: AgentState) -> str:
        """Process step with LLM."""
        context = "\n".join([doc.get("content", "") for doc in state.retrieved_documents])

        prompt = f"""Complete the following step based on the available information.

Step: {step}

Context:
{context}

Query: {state.query}

Response:"""

        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
        )

        return response.strip()
