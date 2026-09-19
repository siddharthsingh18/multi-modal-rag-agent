"""Provider-neutral async LLM client for Anthropic and Gemini."""

from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from anthropic import AsyncAnthropic

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import GenerationError

logger = get_logger(__name__)


class LLMClient:
    """Async wrapper for Anthropic and Gemini generation APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Provider API key (defaults to settings)
            model: Model name (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Max tokens to generate (defaults to settings)
            provider: LLM provider, either ``anthropic`` or ``gemini``
        """
        settings = get_settings()
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        if self.provider == "anthropic":
            self.api_key = api_key or settings.anthropic_api_key
            self.client = AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            self.api_key = api_key or settings.gemini_api_key
            self.client = httpx.AsyncClient(timeout=120)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

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

            if self.provider == "gemini":
                response = await self.client.post(
                    self._gemini_url("generateContent"),
                    json=self._gemini_payload(
                        prompt,
                        system,
                        temperature,
                        max_tokens,
                        stop_sequences,
                    ),
                )
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

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

            if self.provider == "gemini":
                async with self.client.stream(
                    "POST",
                    self._gemini_url("streamGenerateContent") + "&alt=sse",
                    json=self._gemini_payload(prompt, system, temperature, max_tokens),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = __import__("json").loads(line[6:])
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        if parts and parts[0].get("text"):
                            yield parts[0]["text"]
                return

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

    def _gemini_url(self, action: str) -> str:
        """Build a Gemini Generative Language API URL."""
        return (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:{action}?key={self.api_key}"
        )

    def _gemini_payload(
        self,
        prompt: str,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a Gemini generateContent request body."""
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": max_tokens or self.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if stop_sequences:
            payload["generationConfig"]["stopSequences"] = stop_sequences
        return payload

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

            if self.provider == "gemini":
                conversation = "\n\n".join(
                    f"{message['role'].title()}: {message['content']}" for message in messages
                )
                return await self.generate(prompt=conversation, system=system)

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
