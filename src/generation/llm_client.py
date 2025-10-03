"""Anthropic Claude LLM client wrapper."""

from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic
from langchain.schema import HumanMessage, SystemMessage

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import GenerationError

logger = get_logger(__name__)


class LLMClient:
    """Async wrapper for Anthropic Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Anthropic API key (defaults to settings)
            model: Model name (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Max tokens to generate (defaults to settings)
        """
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop_sequences: Stop sequences
            **kwargs: Additional API parameters

        Returns:
            Generated text

        Raises:
            GenerationError: If generation fails
        """
        try:
            logger.debug(f"Generating completion with model {self.model}")

            messages = [{"role": "user", "content": prompt}]

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system or "",
                messages=messages,
                stop_sequences=stop_sequences or [],
                **kwargs,
            )

            generated_text = response.content[0].text
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(
                "LLM generation failed",
                details={"model": self.model, "error": str(e)},
                original_error=e,
            )

    async def generate_with_context(
        self,
        query: str,
        context: List[str],
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate response with retrieved context.

        Args:
            query: User query
            context: Retrieved context chunks
            system: System prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        # Build prompt with context
        context_text = "\n\n".join(
            [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context)]
        )

        prompt = f"""Based on the following context, please answer the question.

{context_text}

Question: {query}

Answer:"""

        return await self.generate(prompt=prompt, system=system, **kwargs)

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream text completion.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional API parameters

        Yields:
            Text chunks

        Raises:
            GenerationError: If streaming fails
        """
        try:
            logger.debug(f"Streaming completion with model {self.model}")

            messages = [{"role": "user", "content": prompt}]

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system or "",
                messages=messages,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise GenerationError(
                "LLM streaming failed",
                details={"model": self.model, "error": str(e)},
                original_error=e,
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Multi-turn chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        try:
            logger.debug(f"Chat completion with {len(messages)} messages")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system or "",
                messages=messages,
                **kwargs,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise GenerationError(
                "Chat completion failed",
                details={"model": self.model, "error": str(e)},
                original_error=e,
            )


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
