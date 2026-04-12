"""Ollama LLM Provider."""

import asyncio
import json as _json
import logging
import re

import httpx

from pathos.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Errores transitorios que ameritan retry
_RETRYABLE = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectTimeout,
    httpx.TimeoutException,
    ConnectionResetError,
    ConnectionAbortedError,
    OSError,
)

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # segundos

class OllamaProvider(LLMProvider):
    """Proveedor LLM usando Ollama local o Ollama Cloud.

    For Ollama Cloud, set base_url="https://ollama.com" and provide api_key.
    The API is identical — cloud just adds Authorization: Bearer header.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        embed_model: str = "nomic-embed-text",
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self._api_key = api_key
        self._client = self._make_client()

    def _make_client(self) -> httpx.AsyncClient:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, read=900.0),
            headers=headers,
        )

    async def _request_with_retry(
        self,
        url: str,
        payload: dict,
    ) -> dict:
        """Hace POST sin streaming con retry (para embed y otros calls cortos)."""
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except _RETRYABLE as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    err_type = type(e).__name__
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s(%s). Retrying in %.1fs...",
                        attempt, MAX_RETRIES, err_type, e, RETRY_DELAY * attempt,
                    )
                    try:
                        await self._client.aclose()
                    except Exception:
                        pass
                    self._client = self._make_client()
                    await asyncio.sleep(RETRY_DELAY * attempt)
                else:
                    logger.error(
                        "Ollama request failed after %d attempts: %s(%s)", MAX_RETRIES, type(e).__name__, e,
                    )
        raise last_error  # type: ignore[misc]

    async def _stream_with_retry(
        self,
        url: str,
        payload: dict,
    ) -> str:
        """Hace POST con streaming y retry. Acumula respuesta completa.

        Ollama con stream=true envía tokens uno a uno como JSON lines.
        Esto mantiene la conexión viva y evita timeouts con modelos lentos
        o con thinking mode (qwen3).
        """
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self._client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    content_parts: list[str] = []
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        # /api/chat format
                        msg = chunk.get("message", {})
                        if msg.get("content"):
                            content_parts.append(msg["content"])
                        if chunk.get("done"):
                            break
                    result = "".join(content_parts)
                    if not result:
                        raise RuntimeError("Ollama returned empty response")
                    return result
            except _RETRYABLE as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Ollama stream failed (attempt %d/%d): %s(%s). Retrying in %.1fs...",
                        attempt, MAX_RETRIES, type(e).__name__, e, RETRY_DELAY * attempt,
                    )
                    try:
                        await self._client.aclose()
                    except Exception:
                        pass
                    self._client = self._make_client()
                    await asyncio.sleep(RETRY_DELAY * attempt)
                else:
                    logger.error(
                        "Ollama stream failed after %d attempts: %s(%s)", MAX_RETRIES, type(e).__name__, e,
                    )
        raise last_error  # type: ignore[misc]

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

        payload: dict = {
            "model": self.model,
            "messages": all_messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {"num_ctx": 4096},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if top_p is not None:
            payload["options"]["top_p"] = top_p
        if top_k is not None:
            payload["options"]["top_k"] = top_k
        if repetition_penalty is not None:
            payload["options"]["repeat_penalty"] = repetition_penalty
        if logit_bias:
            # Ollama accepts logit_bias as string-keyed token_id → bias
            payload["options"]["logit_bias"] = logit_bias
        if presence_penalty is not None:
            payload["options"]["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["options"]["frequency_penalty"] = frequency_penalty

        # Ollama API: think=false disables <think> reasoning in models like qwen3
        if not think:
            payload["think"] = False

        content = await self._stream_with_retry(
            f"{self.base_url}/api/chat",
            payload,
        )

        # Safety net: strip any remaining thinking content.
        # qwen3 sometimes outputs thinking without proper <think> tag,
        # or with only a closing </think> tag.
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "</think>" in content:
            # Thinking leaked without opening tag — everything before </think> is thinking
            idx = content.rfind("</think>")
            content = content[idx + len("</think>"):].strip()

        return content

    async def embed(self, text: str) -> list[float]:
        data = await self._request_with_retry(
            f"{self.base_url}/api/embed",
            {
                "model": self.embed_model,
                "input": text,
                "keep_alive": "30m",
            },
        )
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned empty embeddings")
        return embeddings[0]

    async def close(self) -> None:
        await self._client.aclose()
