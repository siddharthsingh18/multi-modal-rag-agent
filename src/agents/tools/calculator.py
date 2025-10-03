"""Calculator tool for mathematical operations."""

import ast
import operator
from typing import Any

from ...agents.base_agent import Tool
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations."""

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def __init__(self):
        """Initialize calculator tool."""
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations (e.g., '2 + 2', '10 * 5')",
        )

    async def execute(self, expression: str) -> float:
        """
        Evaluate mathematical expression.

        Args:
            expression: Mathematical expression as string

        Returns:
            Result of calculation

        Raises:
            AgentError: If expression is invalid or unsafe
        """
        logger.info(f"Calculating: {expression}")

        try:
            # Parse expression
            tree = ast.parse(expression, mode="eval")

            # Evaluate safely
            result = self._eval_node(tree.body)

            logger.info(f"Result: {result}")
            return float(result)

        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            raise AgentError(
                f"Invalid calculation expression: {expression}",
                original_error=e,
            )

    def _eval_node(self, node: ast.AST) -> Any:
        """
        Safely evaluate AST node.

        Args:
            node: AST node

        Returns:
            Evaluation result
        """
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[op](left, right)

        elif isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op}")

            operand = self._eval_node(node.operand)
            return self.OPERATORS[op](operand)

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
