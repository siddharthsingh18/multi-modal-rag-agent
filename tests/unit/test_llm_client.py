"""Tests for provider selection and Gemini request construction."""

from src.generation.llm_client import LLMClient


def test_gemini_client_uses_configured_provider():
    client = LLMClient(
        provider="gemini",
        api_key="test-gemini-key",
        model="gemini-2.0-flash",
    )

    payload = client._gemini_payload(
        prompt="Answer this.",
        system="Be concise.",
        temperature=0.2,
        max_tokens=128,
        stop_sequences=["END"],
    )

    assert client.provider == "gemini"
    assert client._gemini_url("generateContent").endswith(
        "models/gemini-2.0-flash:generateContent"
    )
    assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise."
    assert payload["generationConfig"]["maxOutputTokens"] == 128
    assert payload["generationConfig"]["stopSequences"] == ["END"]