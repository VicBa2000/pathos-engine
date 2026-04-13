"""Transformers LLM Provider — Direct model access with steering support.

Loads models via HuggingFace transformers library, providing direct access
to hidden states for emotional steering vector injection. Supports:
- HuggingFace model IDs (auto-download from Hub)
- Local GGUF files (dequantized to float16 via transformers GGUF support)
- Local HuggingFace model directories

Trade-offs vs Ollama:
- Pro: Full steering support (register_forward_hook on any layer)
- Pro: Access to hidden states for vector extraction
- Con: GGUF files are dequantized to fp16 (more VRAM than quantized Ollama)
- Con: No built-in quantization management (Ollama handles this transparently)

Hardware note (GTX 1660 Super 6GB):
- Models up to ~3B params fit in fp16 on GPU
- Larger models use device_map="auto" to split between GPU and CPU RAM
- For qwen3:4b (~8GB fp16), expect ~60% GPU + 40% CPU split
- CPU offload is slower but functional for steering-enabled inference

Requires: torch, transformers (already in project dependencies).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from pathos.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Default generation parameters
_DEFAULT_MAX_NEW_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.9
_DEFAULT_REPETITION_PENALTY = 1.1


def find_ollama_gguf(model_name: str) -> Path | None:
    """Locate a GGUF model file downloaded by Ollama.

    Ollama stores models as content-addressed blobs under:
      ~/.ollama/models/blobs/sha256-<hash>

    The manifest at:
      ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>/
    contains JSON pointing to the blob.

    Args:
        model_name: Ollama model name, e.g. "qwen3:4b" or "llama3.2:3b".

    Returns:
        Path to the GGUF blob file, or None if not found.
    """
    import json as _json

    ollama_home = Path(os.environ.get("OLLAMA_MODELS", ""))
    if not ollama_home.is_dir():
        # Default locations
        candidates = [
            Path.home() / ".ollama" / "models",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "models",
            Path(os.environ.get("APPDATA", "")) / "Ollama" / "models",
        ]
        for c in candidates:
            if c.is_dir():
                ollama_home = c
                break
        else:
            return None

    # Parse model_name into name and tag
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"

    manifest_dir = ollama_home / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_dir.is_file():
        # Try as directory with a single file inside
        manifest_path = manifest_dir
        if manifest_dir.is_dir():
            files = list(manifest_dir.iterdir())
            manifest_path = files[0] if files else manifest_dir
        if not manifest_path.is_file():
            return None
    else:
        manifest_path = manifest_dir

    try:
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Find the model layer (mediaType contains "model")
    blobs_dir = ollama_home / "blobs"
    for layer in manifest.get("layers", []):
        media_type = layer.get("mediaType", "")
        if "model" in media_type:
            digest = layer.get("digest", "")  # e.g. "sha256:abc123..."
            blob_name = digest.replace(":", "-")  # sha256-abc123...
            blob_path = blobs_dir / blob_name
            if blob_path.is_file():
                return blob_path

    return None


class TransformersProvider(LLMProvider):
    """LLM Provider using HuggingFace transformers with steering support.

    This provider loads models directly into memory (GPU/CPU) rather than
    communicating over HTTP. This enables:
    - Steering vector injection via forward hooks
    - Hidden state extraction for vector computation
    - Direct control over sampling parameters

    Args:
        model_id: HuggingFace model ID, local path, or Ollama model name.
            If an Ollama name is given (e.g. "qwen3:4b"), it will attempt
            to find the GGUF file in Ollama's storage.
        device_map: How to distribute model across devices.
            "auto" = split between GPU and CPU as needed.
            "cpu" = force CPU-only (slow but always works).
            "cuda" = force full GPU (may OOM on large models).
        embed_model_url: Ollama URL for embeddings (fallback, since transformers
            embedding is via forward pass — slower than dedicated embed model).
        embed_model: Ollama embed model name for fallback embeddings.
        torch_dtype: Model dtype. "auto" lets transformers decide.
            "float16" for GPU, "float32" for CPU.
    """

    def __init__(
        self,
        model_id: str = "qwen3:4b",
        device_map: str = "auto",
        embed_model_url: str = "http://127.0.0.1:11434",
        embed_model: str = "nomic-embed-text",
        torch_dtype: str = "auto",
        adapter_path: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._device_map = device_map
        self._embed_model_url = embed_model_url.rstrip("/")
        self._embed_model = embed_model
        self._torch_dtype = torch_dtype
        self._adapter_path = adapter_path  # Path to QLoRA adapter (5.2b/5.3b)

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._adapter_loaded = False
        self._embed_client: Any = None  # Lazy httpx client for Ollama embed fallback

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TransformersProvider requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from e

        model_path = self._model_id
        gguf_file: str | None = None

        # Check if it's an Ollama model name (contains ":" like "qwen3:4b")
        if ":" in self._model_id and "/" not in self._model_id:
            gguf_path = find_ollama_gguf(self._model_id)
            if gguf_path is not None:
                logger.info("Found Ollama GGUF for '%s' at %s", self._model_id, gguf_path)
                model_path = str(gguf_path.parent)
                gguf_file = gguf_path.name
            else:
                logger.warning(
                    "Ollama GGUF for '%s' not found. Attempting as HuggingFace model ID.",
                    self._model_id,
                )

        # Determine dtype
        if self._torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif self._torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        logger.info(
            "Loading model '%s' (device_map=%s, dtype=%s)...",
            self._model_id, self._device_map, dtype,
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self._device_map,
            "trust_remote_code": True,
        }
        if gguf_file:
            load_kwargs["gguf_file"] = gguf_file

        # Load tokenizer (try same path, fallback to model_id for GGUF)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
        except Exception:
            # GGUF blobs don't have tokenizer files — try the HF model ID
            hf_id = _ollama_to_hf_id(self._model_id)
            if hf_id:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True,
                )
            else:
                raise

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs,
        )

        # Load QLoRA adapter if specified (5.2b/5.3b)
        if self._adapter_path and Path(self._adapter_path).exists():
            try:
                from peft import PeftModel
                adapter_dir = Path(self._adapter_path)
                # If adapter includes new tokens, load tokenizer from adapter
                adapter_tokenizer_path = adapter_dir / "tokenizer_config.json"
                if adapter_tokenizer_path.exists():
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(adapter_dir), trust_remote_code=True,
                    )
                    self._model.resize_token_embeddings(len(self._tokenizer))
                    logger.info("Loaded adapter tokenizer with %d tokens", len(self._tokenizer))
                self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))
                self._adapter_loaded = True
                logger.info("Loaded QLoRA adapter from %s", adapter_dir)
            except ImportError:
                logger.warning("peft not installed — adapter at %s ignored", self._adapter_path)
            except Exception as e:
                logger.warning("Failed to load adapter from %s: %s", self._adapter_path, e)

        self._model.eval()
        self._loaded = True

        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info("Model loaded. GPU memory: %.2f GB", allocated)
        else:
            logger.info("Model loaded on CPU.")

    @property
    def model(self) -> str:
        """Model identifier string."""
        return self._model_id

    @property
    def supports_steering(self) -> bool:
        return True

    @property
    def steerable_model(self) -> object | None:
        self._ensure_loaded()
        return self._model

    @property
    def has_adapter(self) -> bool:
        """Whether a QLoRA adapter is loaded."""
        return self._adapter_loaded

    @property
    def tokenizer(self) -> Any:
        """Expose tokenizer for steering vector extraction."""
        self._ensure_loaded()
        return self._tokenizer

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
        self._ensure_loaded()
        import torch

        temp = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        tp = top_p if top_p is not None else _DEFAULT_TOP_P
        rp = repetition_penalty if repetition_penalty is not None else _DEFAULT_REPETITION_PENALTY

        # Build prompt: system + conversation
        prompt = self._build_prompt(system_prompt, messages)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        # Move to model's device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": _DEFAULT_MAX_NEW_TOKENS,
            "temperature": max(temp, 0.01),  # Avoid zero
            "top_p": tp,
            "repetition_penalty": rp,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        # Note: presence_penalty and frequency_penalty are not natively supported
        # by HuggingFace generate() — they are Ollama/OpenAI-specific params.
        # Silently ignored here.

        # Apply logit bias via LogitsProcessor if provided
        if logit_bias:
            from transformers import LogitsProcessor, LogitsProcessorList

            class _EmotionalLogitBiasProcessor(LogitsProcessor):
                def __init__(self, bias_dict: dict[str, float]) -> None:
                    self._bias = bias_dict

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                    for token_str, bias_val in self._bias.items():
                        try:
                            tid = int(token_str)
                            if 0 <= tid < scores.shape[-1]:
                                scores[:, tid] += bias_val
                        except (ValueError, IndexError):
                            pass
                    return scores

            gen_kwargs["logits_processor"] = LogitsProcessorList([
                _EmotionalLogitBiasProcessor(logit_bias),
            ])

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_len:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip thinking tags (qwen3 style)
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "</think>" in content:
            idx = content.rfind("</think>")
            content = content[idx + len("</think>"):].strip()

        return content

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings.

        Uses Ollama's embed endpoint as fallback (same as ClaudeProvider),
        since running embeddings through a large causal LM is wasteful.
        """
        import httpx

        if self._embed_client is None:
            self._embed_client = httpx.AsyncClient(timeout=60.0)

        response = await self._embed_client.post(
            f"{self._embed_model_url}/api/embed",
            json={"model": self._embed_model, "input": text, "keep_alive": "30m"},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned empty embeddings")
        return embeddings[0]

    async def close(self) -> None:
        """Release model and client resources."""
        if self._embed_client is not None:
            await self._embed_client.aclose()
            self._embed_client = None
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _build_prompt(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Build a chat prompt using the tokenizer's chat template if available.

        Falls back to a simple format if no template is configured.
        """
        chat_messages = [{"role": "system", "content": system_prompt}] + messages

        # Try using the tokenizer's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass  # Fallback to manual format

        # Simple fallback format
        parts = [f"<|system|>\n{system_prompt}\n"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)


# --- Utility: Ollama model name → HuggingFace model ID ---

_OLLAMA_TO_HF: dict[str, str] = {
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:1.7b": "Qwen/Qwen3-1.7B",
    "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi4:14b": "microsoft/phi-4",
}


def _ollama_to_hf_id(ollama_name: str) -> str | None:
    """Map common Ollama model names to HuggingFace model IDs.

    Used as fallback when loading tokenizer from GGUF blob fails.
    """
    # Exact match
    if ollama_name in _OLLAMA_TO_HF:
        return _OLLAMA_TO_HF[ollama_name]

    # Try without tag
    base = ollama_name.split(":")[0]
    for key, value in _OLLAMA_TO_HF.items():
        if key.startswith(base + ":"):
            return value

    return None
