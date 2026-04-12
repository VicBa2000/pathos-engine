"""OpenAI-compatible LLM Provider.

Works with any service that implements the OpenAI Chat Completions API:
Groq, OpenRouter, Together AI, or any custom endpoint.
Uses Ollama for embeddings (these services don't provide embedding endpoints).
"""

import httpx

from pathos.llm.base import LLMProvider


# Known presets for cloud providers
CLOUD_PRESETS: dict[str, dict[str, str]] = {
    "ollama_cloud": {
        "label": "Ollama Cloud",
        "base_url": "https://ollama.com",
        "key_prefix": "",
        "description": "Run large models in Ollama's cloud (same API as local)",
        "default_model": "llama3.1:8b",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_prefix": "gsk_",
        "description": "Ultra-fast inference (free tier available)",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_prefix": "sk-or-",
        "description": "Multi-provider aggregator (some free models)",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
    },
    "together": {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "key_prefix": "tok_",
        "description": "Open-source models in the cloud (free credits)",
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com",
        "key_prefix": "sk-ant-",
        "description": "Claude models (paid API)",
        "default_model": "claude-sonnet-4-20250514",
    },
    "custom": {
        "label": "Custom OpenAI-Compatible",
        "base_url": "",
        "key_prefix": "",
        "description": "Any OpenAI-compatible API endpoint",
        "default_model": "",
    },
}


class OpenAICompatProvider(LLMProvider):
    """Provider for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        ollama_base_url: str = "http://127.0.0.1:11434",
        embed_model: str = "nomic-embed-text",
        provider_name: str = "custom",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=120.0)
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
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        body: dict = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": 2048,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        # top_k and repetition_penalty not in OpenAI API spec

        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError("OpenAI-compatible API returned empty choices")
        content = choices[0].get("message", {}).get("content")
        if content is None:
            raise RuntimeError("OpenAI-compatible API returned no content")
        return content

    async def embed(self, text: str) -> list[float]:
        """Embeddings via Ollama (cloud providers don't offer compatible embeddings)."""
        resp = await self._client.post(
            f"{self._ollama_base_url}/api/embed",
            json={"model": self._embed_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned empty embeddings")
        return embeddings[0]

    async def close(self) -> None:
        await self._client.aclose()
