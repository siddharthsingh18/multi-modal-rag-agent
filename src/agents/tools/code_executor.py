"""Code execution tool (sandboxed Python execution)."""

import ast
import sys
from io import StringIO
from typing import Any, Dict

from ...agents.base_agent import Tool
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class CodeExecutorTool(Tool):
    """Tool for executing Python code in a restricted environment."""

    # Allowed built-in functions
    ALLOWED_BUILTINS = {
        "abs",
        "all",
        "any",
        "enumerate",
        "filter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "round",
        "sorted",
        "sum",
        "zip",
    }

    def __init__(self):
        """Initialize code executor tool."""
        super().__init__(
            name="code_executor",
            description="Execute Python code for data processing and analysis",
        )

    async def execute(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute Python code in a restricted environment.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (not implemented in this version)

        Returns:
            Dictionary with execution results

        Raises:
            AgentError: If code execution fails or is unsafe
        """
        logger.info("Executing Python code")

        try:
            # Validate code syntax
            ast.parse(code)

            # Capture stdout
            stdout = StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout

            # Create restricted namespace
            namespace = {"__builtins__": {}}

            # Add allowed builtins
            for name in self.ALLOWED_BUILTINS:
                namespace["__builtins__"][name] = getattr(__builtins__, name)

            try:
                # Execute code
                exec(code, namespace)

                # Get output
                output = stdout.getvalue()

                # Extract variables
                variables = {
                    k: v
                    for k, v in namespace.items()
                    if not k.startswith("_") and k != "__builtins__"
                }

                result = {
                    "success": True,
                    "output": output,
                    "variables": str(variables),
                    "error": None,
                }

                logger.info("Code executed successfully")
                return result

            finally:
                sys.stdout = old_stdout

        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            raise AgentError(
                f"Syntax error in code: {e}",
                original_error=e,
            )

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            raise AgentError(
                f"Code execution failed: {e}",
                original_error=e,
            )
