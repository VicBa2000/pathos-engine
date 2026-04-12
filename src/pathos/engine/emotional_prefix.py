"""Soft Emotional Prefix — synthetic embedding injection at input layer.

Level 5.3a (lightweight): Injects 1-4 "phantom" tokens at the start of the
input sequence whose embeddings are composed from steering vectors. The model
processes these tokens as if they were real words, but their content is purely
emotional.

Difference from steering vectors:
  - Steering vectors: add perturbation to ALL positions at INTERMEDIATE layers
    → "process everything with this emotional bias"
  - Emotional prefix: inject emotional INFORMATION at the INPUT layer
    → "here is my emotional state as context"

Both are complementary: prefix provides context, steering shifts processing.

Requirements:
  - TransformersProvider with steerable model (same as steering)
  - Pre-extracted steering vectors (reuses existing cached vectors)
  - PyTorch model with get_input_embeddings() method

Only works with local models. Cloud APIs degrade gracefully (no prefix).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pathos.engine.steering import (
    CachedVectors,
    EmotionalSteeringEngine,
    EMOTIONAL_DIMENSIONS,
    DIMENSION_WEIGHTS,
    MAX_STEERING_NORM,
)

logger = logging.getLogger(__name__)

# Number of prefix tokens to inject (more tokens = stronger conditioning)
DEFAULT_NUM_PREFIX_TOKENS: int = 2

# Scaling factor for prefix embeddings (controls how "loud" the prefix is)
# Lower = subtle influence, higher = stronger emotional context
PREFIX_SCALE: float = 0.5

# Maximum number of prefix tokens allowed
MAX_PREFIX_TOKENS: int = 4


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


@dataclass
class PrefixResult:
    """Result of computing emotional prefix embeddings."""

    active: bool = False
    num_tokens: int = 0
    embedding_norm: float = 0.0
    dominant_dimension: str = "neutral"
    scale: float = PREFIX_SCALE


def compose_prefix_embedding(
    cached: CachedVectors,
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    scale: float = PREFIX_SCALE,
) -> np.ndarray | None:
    """Compose a prefix embedding vector from cached steering vectors.

    Uses the EARLIEST available layer's vectors as the embedding base,
    since embedding space is closest to early-layer representation space.

    The embedding is a weighted sum of dimensional vectors (same formula as
    compute_composite_vector but without layer-role scaling, and using
    prefix-specific scaling).

    Args:
        cached: Pre-extracted steering vectors.
        valence, arousal, dominance, certainty: Emotional dimensions.
        intensity: Overall emotional intensity [0, 1].
        scale: Scaling factor for the prefix (default: PREFIX_SCALE).

    Returns:
        Embedding vector of shape (hidden_size,), or None if not available.
    """
    # Use earliest available layer (closest to embedding space)
    if not cached.available_layers:
        return None
    early_layer = min(cached.available_layers)

    # Intensity gate: very weak emotions don't need prefix
    if intensity < 0.05:
        return None

    deviations: dict[str, float] = {
        "valence": valence,
        "arousal": (arousal - 0.5) * 2.0,
        "dominance": (dominance - 0.5) * 2.0,
        "certainty": (certainty - 0.5) * 2.0,
    }

    composite = np.zeros(cached.hidden_size, dtype=np.float32)
    any_added = False

    for dim in EMOTIONAL_DIMENSIONS:
        if not cached.has_dimension(dim):
            continue
        layer_vectors = cached.vectors[dim]
        if early_layer not in layer_vectors:
            continue

        direction = layer_vectors[early_layer]
        weight = DIMENSION_WEIGHTS.get(dim, 0.0)
        deviation = deviations[dim]

        composite += (weight * deviation * intensity * scale) * direction.astype(np.float32)
        any_added = True

    if not any_added:
        return None

    # Clamp norm
    norm = float(np.linalg.norm(composite))
    if norm > MAX_STEERING_NORM * scale:
        composite = composite * (MAX_STEERING_NORM * scale / norm)

    return composite


def _get_embedding_layer(model: Any) -> Any | None:
    """Get the embedding layer from a HuggingFace model.

    Supports:
    - model.get_input_embeddings() (standard HF API)
    - model.model.embed_tokens (Llama/Qwen2/Mistral direct access)
    - model.transformer.wte (GPT-2 style)
    """
    # Standard HF API (preferred)
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb

    # Direct access fallback
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens

    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte

    return None


class EmotionalPrefixHook:
    """Hooks the embedding layer to prepend emotional prefix tokens.

    During inference, intercepts the embedding output and prepends N
    synthetic embeddings that encode the emotional state. The model
    sees these as additional context tokens at the start of the sequence.

    Usage:
        hook = EmotionalPrefixHook(model, engine, state, num_tokens=2)
        hook.apply()
        output = model(...)    # Prefix tokens are auto-prepended
        hook.remove()

    Or as context manager:
        with EmotionalPrefixHook(...) as h:
            output = model(...)
    """

    def __init__(
        self,
        model: Any,
        engine: EmotionalSteeringEngine,
        valence: float,
        arousal: float,
        dominance: float,
        certainty: float,
        intensity: float,
        num_tokens: int = DEFAULT_NUM_PREFIX_TOKENS,
        scale: float = PREFIX_SCALE,
    ) -> None:
        self._model = model
        self._engine = engine
        self._valence = valence
        self._arousal = arousal
        self._dominance = dominance
        self._certainty = certainty
        self._intensity = intensity
        self._num_tokens = min(num_tokens, MAX_PREFIX_TOKENS)
        self._scale = scale
        self._handle: Any = None
        self._prefix_embedding: np.ndarray | None = None
        self._result = PrefixResult()

    @property
    def is_applied(self) -> bool:
        return self._handle is not None

    @property
    def result(self) -> PrefixResult:
        return self._result

    def apply(self) -> bool:
        """Register a forward hook on the embedding layer.

        The hook prepends emotional prefix embeddings to the input.

        Returns True if the hook was registered, False otherwise.
        """
        if self.is_applied:
            self.remove()

        if not self._engine.is_ready:
            return False

        cached = self._engine._cached
        if cached is None:
            return False

        # Compose prefix embedding
        embedding = compose_prefix_embedding(
            cached,
            valence=self._valence,
            arousal=self._arousal,
            dominance=self._dominance,
            certainty=self._certainty,
            intensity=self._intensity,
            scale=self._scale,
        )
        if embedding is None:
            return False

        self._prefix_embedding = embedding

        # Find embedding layer
        emb_layer = _get_embedding_layer(self._model)
        if emb_layer is None:
            logger.warning("Cannot find embedding layer on model %s", type(self._model).__name__)
            return False

        # Determine dominant dimension for diagnostics
        deviations = {
            "valence": abs(self._valence),
            "arousal": abs(self._arousal - 0.5) * 2,
            "dominance": abs(self._dominance - 0.5) * 2,
            "certainty": abs(self._certainty - 0.5) * 2,
        }
        dominant = max(deviations, key=deviations.get)  # type: ignore[arg-type]

        emb_norm = float(np.linalg.norm(embedding))

        # Create hook
        num_tokens = self._num_tokens
        hook_fn = _make_prefix_hook(embedding, num_tokens)
        self._handle = emb_layer.register_forward_hook(hook_fn)

        self._result = PrefixResult(
            active=True,
            num_tokens=num_tokens,
            embedding_norm=round(emb_norm, 4),
            dominant_dimension=dominant,
            scale=self._scale,
        )

        logger.debug(
            "Emotional prefix hook applied: %d tokens, norm=%.4f, dominant=%s",
            num_tokens, emb_norm, dominant,
        )
        return True

    def remove(self) -> None:
        """Remove the registered hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "EmotionalPrefixHook":
        self.apply()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    def __del__(self) -> None:
        self.remove()


def _make_prefix_hook(embedding: np.ndarray, num_tokens: int):
    """Create a forward hook that prepends emotional embeddings.

    The embedding layer outputs shape (batch, seq_len, hidden_size).
    We prepend num_tokens copies of the emotional embedding, creating
    (batch, num_tokens + seq_len, hidden_size).

    Each prefix token gets a slightly different version of the embedding
    (scaled by position) to avoid degenerate attention patterns.

    Args:
        embedding: 1D array of shape (hidden_size,).
        num_tokens: How many prefix tokens to prepend.

    Returns:
        Hook function compatible with PyTorch register_forward_hook.
    """
    _cache: dict[str, Any] = {}

    def hook_fn(module: Any, input: Any, output: Any) -> Any:
        try:
            import torch
        except ImportError:
            return output

        # Build prefix tensor (cached)
        if "prefix" not in _cache or _cache.get("device") != str(output.device):
            base = torch.tensor(embedding, dtype=output.dtype, device=output.device)
            # Each token gets slightly different scale to break symmetry
            prefix_tokens = []
            for i in range(num_tokens):
                scale = 1.0 - i * 0.1  # Token 0: 1.0, Token 1: 0.9, etc.
                prefix_tokens.append(base * scale)
            # Shape: (1, num_tokens, hidden_size)
            _cache["prefix"] = torch.stack(prefix_tokens).unsqueeze(0)
            _cache["device"] = str(output.device)

        prefix = _cache["prefix"]

        # Cast if needed
        if prefix.dtype != output.dtype:
            prefix = prefix.to(output.dtype)
            _cache["prefix"] = prefix

        # Expand prefix to batch size
        batch_size = output.shape[0]
        if prefix.shape[0] != batch_size:
            prefix = prefix.expand(batch_size, -1, -1)

        # Prepend: (batch, num_tokens + seq_len, hidden_size)
        return torch.cat([prefix, output], dim=1)

    return hook_fn
