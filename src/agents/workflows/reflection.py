"""Reflection workflow for answer quality assessment."""

from typing import Dict, List

from ...generation.llm_client import LLMClient
from ...generation.prompt_templates import PromptTemplates
from ...observability.logging import get_logger

logger = get_logger(__name__)


class ReflectionWorkflow:
    """Agent reflection workflow for self-evaluation."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize reflection workflow.

        Args:
            llm_client: LLM client for reflection
        """
        self.llm_client = llm_client

    async def reflect_on_answer(
        self,
        query: str,
        answer: str,
        context: List[str],
    ) -> Dict[str, any]:
        """
        Reflect on answer quality.

        Args:
            query: Original query
            answer: Generated answer
            context: Context used for generation

        Returns:
            Reflection results with score and feedback
        """
        logger.info("Reflecting on answer quality")

        prompt = PromptTemplates.reflection_prompt(query, answer, context)

        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=300,
        )

        # Parse reflection
        score = self._extract_score(response)
        feedback = response

        reflection = {
            "score": score,
            "feedback": feedback,
            "needs_improvement": score < 7.0,
        }

        logger.info(f"Reflection score: {score}/10")
        return reflection

    def _extract_score(self, reflection_text: str) -> float:
        """
        Extract numerical score from reflection.

        Args:
            reflection_text: Reflection text

        Returns:
            Score from 0-10
        """
        # Look for patterns like "8/10", "Score: 8", etc.
        import re

        patterns = [
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"score:\s*(\d+(?:\.\d+)?)",
            r"rating:\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, reflection_text.lower())
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)

        # Default to mid-range if no score found
        return 5.0

    async def should_retry(
        self,
        reflection: Dict[str, any],
        max_retries: int = 2,
        current_retry: int = 0,
    ) -> bool:
        """
        Determine if answer should be regenerated.

        Args:
            reflection: Reflection results
            max_retries: Maximum number of retries
            current_retry: Current retry count

        Returns:
            True if should retry
        """
        if current_retry >= max_retries:
            return False

        if reflection["score"] < 6.0:
            logger.info(f"Low score ({reflection['score']}), will retry")
            return True

        return False
