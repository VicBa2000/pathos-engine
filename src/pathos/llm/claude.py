"""Claude (Anthropic) LLM Provider.

Usa Ollama para embeddings (Claude API no tiene endpoint de embeddings).
"""

import anthropic
import httpx

from pathos.llm.base import LLMProvider


class ClaudeProvider(LLMProvider):
    """Proveedor LLM usando Anthropic Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
    ):
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._embed_client = httpx.AsyncClient(timeout=60.0)
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._embed_model = embed_model

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        think: bool = True,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        # repetition_penalty, presence_penalty, frequency_penalty not supported by Claude API
        response = await self._client.messages.create(**kwargs)
        if not response.content:
            raise RuntimeError("Claude returned empty response content")
        return response.content[0].text

    async def embed(self, text: str) -> list[float]:
        response = await self._embed_client.post(
            f"{self._ollama_base_url}/api/embed",
            json={"model": self._embed_model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned empty embeddings")
        return embeddings[0]

    async def close(self) -> None:
        await self._client.close()
        await self._embed_client.aclose()
