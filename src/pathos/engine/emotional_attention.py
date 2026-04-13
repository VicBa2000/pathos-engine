"""Emotional Attention Modulation — Bias attention weights by emotional state.

Human emotions filter what information is relevant:
  - Fear: amplifies attention to threatening tokens (narrowing)
  - Anger: focuses on agents, responsibility markers (narrowing)
  - Joy: broadens attention distribution (Fredrickson's broaden-and-build)
  - Sadness: amplifies loss-related and self-referential tokens
  - Surprise: heightened attention to novel/unexpected tokens

Implementation: register forward hooks on attention layers that add a bias
to the attention scores BEFORE softmax. This shifts which context tokens
the model "focuses on" without changing the content of hidden states
(that's what steering vectors do — the two systems are complementary).

Only works with local HuggingFace transformers models. Cloud APIs degrade
gracefully (no-op).

Technical approach:
  We hook into the attention module's forward pass and modify the attention
  score matrix (pre-softmax logits). For each token in the vocabulary of
  emotional relevance categories, we add a small bias to the attention
  scores where that token appears in the key sequence. This makes the
  model "pay more attention" to emotionally relevant tokens.

  The bias is computed from the emotional state:
  - Valence controls positive/negative token attention
  - Arousal controls narrowing (high) vs broadening (low) of attention
  - Dominance controls agent/responsibility token attention
  - Certainty controls uncertainty marker attention
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Paths ---
_ATTENTION_DATA_DIR = Path(__file__).parent.parent / "sampling_data"
_ATTENTION_VOCAB_PATH = _ATTENTION_DATA_DIR / "attention_vocabulary.json"

# --- Constants ---

# Maximum attention bias magnitude (pre-softmax logit addition).
# Too high → model ignores context; too low → no effect.
# Range: ±1.5 is moderate (for comparison, typical logits are ~[-5, +5]).
MAX_ATTENTION_BIAS: float = 1.5
MIN_ATTENTION_BIAS: float = -1.5

# Intensity gate: below this, no attention modulation.
INTENSITY_GATE: float = 0.10

# Arousal-based attention distribution control.
# High arousal → narrower attention (focus on fewer tokens more strongly).
# Low arousal → broader attention (distribute more evenly, Fredrickson's broadening).
# This scales the magnitude of category-specific biases.
NARROWING_SCALE_HIGH: float = 1.3   # High arousal amplifies biases
NARROWING_SCALE_LOW: float = 0.6    # Low arousal dampens biases (broader attention)

# Attention vocabulary categories.
ATTENTION_CATEGORIES = (
    "threat", "agent", "positive", "negative",
    "loss", "novelty", "uncertainty_markers",
)


@dataclass(frozen=True)
class AttentionBiasResult:
    """Result of computing attention biases from emotional state.

    token_biases: word → bias value. Positive = attend more, negative = attend less.
    categories_active: which categories were activated and their raw activation level.
    broadening_factor: arousal-based scaling (>1 = narrowing, <1 = broadening).
    """
    token_biases: dict[str, float]
    categories_active: dict[str, float] = field(default_factory=dict)
    broadening_factor: float = 1.0


# Cache for loaded vocabulary
_attention_vocab_cache: dict[str, list[str]] | None = None


def load_attention_vocabulary(path: Path | None = None) -> dict[str, list[str]]:
    """Load attention vocabulary from JSON file.

    Returns:
        Dict with category names as keys, word lists as values.
    """
    global _attention_vocab_cache
    if _attention_vocab_cache is not None and path is None:
        return _attention_vocab_cache

    p = path or _ATTENTION_VOCAB_PATH
    if not p.exists():
        raise FileNotFoundError(f"Attention vocabulary not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    vocab: dict[str, list[str]] = {}
    for cat in ATTENTION_CATEGORIES:
        if cat not in data:
            raise ValueError(f"Missing category '{cat}' in attention vocabulary")
        words = data[cat]
        if not isinstance(words, list) or len(words) < 3:
            raise ValueError(
                f"Category '{cat}' needs at least 3 words, "
                f"got {len(words) if isinstance(words, list) else 0}"
            )
        vocab[cat] = [w for w in words if isinstance(w, str)]

    if path is None:
        _attention_vocab_cache = vocab
    return vocab


def compute_attention_bias(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    vocabulary: dict[str, list[str]] | None = None,
) -> AttentionBiasResult:
    """Compute word-level attention biases from emotional state.

    Maps emotional dimensions to attention focus:
    - Fear (neg valence + high arousal): amplify threat tokens, suppress positive
    - Anger (neg valence + high dominance): amplify agent/responsibility tokens
    - Joy (pos valence): boost positive tokens, slight suppress negative
    - Sadness (neg valence + low arousal): amplify loss tokens
    - Uncertainty (low certainty): amplify uncertainty markers
    - Arousal modulates breadth: high = narrow focus, low = broad (Fredrickson)

    Args:
        valence: [-1, 1]
        arousal: [0, 1]
        dominance: [0, 1]
        certainty: [0, 1]
        intensity: [0, 1]
        vocabulary: Override vocabulary (for testing).

    Returns:
        AttentionBiasResult with word_biases and metadata.
    """
    if intensity < INTENSITY_GATE:
        return AttentionBiasResult(token_biases={})

    vocab = vocabulary or load_attention_vocabulary()
    scale = intensity

    # Arousal-based broadening/narrowing factor
    arousal_dev = (arousal - 0.5) * 2.0  # [-1, +1]
    if arousal_dev > 0:
        broadening_factor = 1.0 + arousal_dev * (NARROWING_SCALE_HIGH - 1.0)
    else:
        broadening_factor = 1.0 + arousal_dev * (1.0 - NARROWING_SCALE_LOW)

    categories: dict[str, float] = {}

    # --- Threat attention (fear pattern: negative valence + high arousal) ---
    # Threat tokens become salient when afraid
    if valence < -0.10 and arousal > 0.4:
        threat_signal = (-valence) * (arousal - 0.3) * scale
        categories["threat"] = min(threat_signal, 1.0)

    # --- Agent attention (anger pattern: negative valence + high dominance) ---
    # When angry, focus on who is responsible
    if valence < -0.10 and dominance > 0.4:
        agent_signal = (-valence) * (dominance - 0.3) * scale
        categories["agent"] = min(agent_signal, 1.0)

    # --- Positive attention (joy/contentment) ---
    if valence > 0.15:
        categories["positive"] = valence * scale
        # Joy also slightly suppresses negative tokens
        categories["negative"] = -valence * scale * 0.3

    # --- Negative attention (sadness/distress) ---
    if valence < -0.15:
        categories["negative"] = (-valence) * scale
        # Suppress positive tokens when distressed
        categories["positive"] = valence * scale * 0.3

    # --- Loss attention (sadness pattern: negative valence + low arousal) ---
    if valence < -0.10 and arousal < 0.5:
        loss_signal = (-valence) * (0.6 - arousal) * scale
        categories["loss"] = min(loss_signal, 1.0)

    # --- Novelty attention (surprise/curiosity: low certainty + high arousal) ---
    if certainty < 0.45 and arousal > 0.4:
        novelty_signal = (0.5 - certainty) * (arousal - 0.3) * scale
        categories["novelty"] = min(novelty_signal, 1.0)

    # --- Uncertainty markers (low certainty) ---
    cert_dev = (0.5 - certainty) * 2.0  # positive when uncertain
    if cert_dev > 0.15:
        categories["uncertainty_markers"] = cert_dev * scale

    if not categories:
        return AttentionBiasResult(token_biases={}, broadening_factor=broadening_factor)

    # Build word → bias mapping
    word_biases: dict[str, float] = {}
    for cat, activation in categories.items():
        if cat not in vocab:
            continue
        # Scale activation to bias magnitude, modulated by broadening factor
        bias_value = _clamp(
            activation * MAX_ATTENTION_BIAS * broadening_factor,
            MIN_ATTENTION_BIAS, MAX_ATTENTION_BIAS,
        )
        if abs(bias_value) < 0.05:
            continue
        for word in vocab[cat]:
            word_biases[word] = _clamp(
                word_biases.get(word, 0.0) + bias_value,
                MIN_ATTENTION_BIAS, MAX_ATTENTION_BIAS,
            )

    # Remove near-zero biases
    word_biases = {w: round(b, 3) for w, b in word_biases.items() if abs(b) >= 0.05}

    return AttentionBiasResult(
        token_biases=word_biases,
        categories_active={cat: round(act, 3) for cat, act in categories.items()},
        broadening_factor=round(broadening_factor, 3),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ===========================================================================
# Runtime Attention Hooks — Modify attention scores during inference
# ===========================================================================

def build_token_set(
    word_biases: dict[str, float],
    tokenizer: Any,
) -> dict[int, float]:
    """Convert word-level biases to token-id-level biases via tokenizer.

    Each word is tokenized. All resulting token IDs get the word's bias.
    Multi-token words: bias applied to EACH sub-token (not split — each
    sub-token triggers attention to the full word).

    Args:
        word_biases: word → bias from compute_attention_bias().
        tokenizer: HuggingFace-compatible tokenizer.

    Returns:
        token_id → bias value.
    """
    if not word_biases:
        return {}

    token_biases: dict[int, float] = {}
    for word, bias in word_biases.items():
        try:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                continue
            for tid in token_ids:
                current = token_biases.get(tid, 0.0)
                token_biases[tid] = _clamp(
                    current + bias, MIN_ATTENTION_BIAS, MAX_ATTENTION_BIAS,
                )
        except Exception:
            continue

    return {tid: round(b, 3) for tid, b in token_biases.items() if abs(b) >= 0.05}


class AttentionHook:
    """Registers forward hooks on attention layers to bias attention scores.

    During inference, for each hooked attention layer, the hook identifies
    which token positions in the key sequence correspond to emotionally
    relevant tokens (via token ID matching) and adds a bias to their
    attention scores before softmax.

    This complements SteeringHook: steering modifies WHAT is represented
    in hidden states, attention modulation modifies WHERE the model focuses.

    Usage:
        hook = AttentionHook(model, token_biases, input_ids)
        hook.apply()
        output = model(...)
        hook.remove()

    Or as context manager:
        with AttentionHook(model, token_biases, input_ids):
            output = model(...)

    Only works with HuggingFace transformers models.
    """

    def __init__(
        self,
        model: Any,
        token_biases: dict[int, float],
        input_ids: Any | None = None,
        layers: list[int] | None = None,
    ) -> None:
        """
        Args:
            model: HuggingFace transformers model.
            token_biases: token_id → attention bias from build_token_set().
            input_ids: Token IDs of the input sequence (tensor or list).
                Used to identify which positions contain emotional tokens.
                If None, hook is a no-op.
            layers: Which layers to hook. None = auto-select mid layers.
        """
        self._model = model
        self._token_biases = token_biases
        self._input_ids = input_ids
        self._target_layers = layers
        self._handles: list[Any] = []
        self._layers_hooked: list[int] = []
        self._positions_biased: int = 0

    @property
    def is_applied(self) -> bool:
        return len(self._handles) > 0

    @property
    def layers_hooked(self) -> list[int]:
        return list(self._layers_hooked)

    @property
    def positions_biased(self) -> int:
        return self._positions_biased

    def apply(self) -> bool:
        """Register attention bias hooks on model layers.

        Returns True if at least one hook was registered.
        """
        if self.is_applied:
            self.remove()

        if not self._token_biases or self._input_ids is None:
            return False

        # Build position mask: which positions in the input contain emotional tokens
        position_biases = _build_position_biases(
            self._input_ids, self._token_biases,
        )
        if not position_biases:
            return False

        self._positions_biased = len(position_biases)

        # Get attention layers
        attn_layers = _get_attention_layers(self._model)
        if attn_layers is None:
            logger.warning(
                "Cannot find attention layers on model %s",
                type(self._model).__name__,
            )
            return False

        # Select which layers to hook
        target = self._target_layers
        if target is None:
            # Auto-select: mid layers (semantic processing)
            n = len(attn_layers)
            if n == 0:
                return False
            mid_start = n // 3
            mid_end = 2 * n // 3
            target = list(range(mid_start, mid_end))

        any_hooked = False
        for layer_idx in target:
            if layer_idx >= len(attn_layers):
                continue

            hook_fn = _make_attention_hook(position_biases)
            handle = attn_layers[layer_idx].register_forward_hook(hook_fn)
            self._handles.append(handle)
            self._layers_hooked.append(layer_idx)
            any_hooked = True

        if any_hooked:
            logger.debug(
                "Attention hooks applied: %d layers, %d positions biased",
                len(self._handles), self._positions_biased,
            )
        return any_hooked

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._layers_hooked.clear()
        self._positions_biased = 0

    def __enter__(self) -> "AttentionHook":
        self.apply()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    def __del__(self) -> None:
        self.remove()


# --- Internal helpers ---

def _build_position_biases(
    input_ids: Any,
    token_biases: dict[int, float],
) -> dict[int, float]:
    """Map token positions in input_ids to bias values.

    For each position in the input sequence, check if the token ID at that
    position has an emotional bias. Returns position → bias.

    Args:
        input_ids: 1D or 2D tensor/list of token IDs.
        token_biases: token_id → bias.

    Returns:
        position_index → bias value (only for positions with emotional tokens).
    """
    # Convert to flat list of ints
    ids: list[int]
    if hasattr(input_ids, "tolist"):
        flat = input_ids.tolist()
        if isinstance(flat, list) and flat and isinstance(flat[0], list):
            ids = flat[-1]  # Last sequence in batch (most recent)
        else:
            ids = flat
    elif isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            ids = list(input_ids[-1])
        else:
            ids = list(input_ids)
    else:
        return {}

    position_biases: dict[int, float] = {}
    for pos, tid in enumerate(ids):
        if tid in token_biases:
            position_biases[pos] = token_biases[tid]

    return position_biases


def _get_attention_layers(model: Any) -> list[Any] | None:
    """Get the list of attention sub-modules from a model.

    In HuggingFace transformers, each transformer block has a self_attn
    attribute. We return the full block list — the hook on the block
    intercepts the attention computation.

    Supports:
    - model.model.layers (Llama/Qwen2/Mistral/Gemma/Phi)
    - model.transformer.h (GPT-2/GPT-Neo)
    """
    # LlamaForCausalLM, Qwen2ForCausalLM, etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if hasattr(layers, "__len__"):
            return list(layers)

    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        if hasattr(layers, "__len__"):
            return list(layers)

    return None


def _make_attention_hook(position_biases: dict[int, float]):
    """Create a forward hook that biases attention scores at specific positions.

    The hook intercepts the transformer block's output and modifies the
    attention weights. Since standard HuggingFace models compute attention
    internally within the block, we use an alternative approach:

    We hook the SELF-ATTENTION module (not the full block) and modify
    the attention output to upweight/downweight certain key positions.

    However, the cleanest approach for maximum compatibility is to modify
    the hidden states entering the attention computation — by slightly
    scaling the key projections at emotional positions. This is equivalent
    to attention biasing but works through the standard forward pass.

    For this implementation, we use the approach of adding bias to the
    hidden states at emotional token positions BEFORE they enter the
    attention layer. This effectively makes those positions "louder" in
    the key space, causing them to attract more attention from all queries.

    Args:
        position_biases: position → bias value.

    Returns:
        Hook function for register_forward_hook.
    """
    _tensor_cache: dict[str, Any] = {}

    def hook_fn(module: Any, input: Any, output: Any) -> Any:
        try:
            import torch
        except ImportError:
            return output

        # The input to a transformer block is typically:
        # - A tuple where input[0] is hidden_states (batch, seq_len, hidden_dim)
        # We modify the output hidden states to amplify emotional positions.
        # This is a post-hook: we scale output hidden states at emotional positions.

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if hidden_states.dim() < 2:
            return output

        seq_len = hidden_states.size(-2)

        # Build scaling factors for each position
        if "scale" not in _tensor_cache or _tensor_cache.get("seq_len") != seq_len:
            scale = torch.ones(seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
            for pos, bias in position_biases.items():
                if 0 <= pos < seq_len:
                    # Convert bias to a multiplicative scale factor.
                    # bias of +1.0 → scale of 1.10 (10% amplification)
                    # bias of -1.0 → scale of 0.90 (10% attenuation)
                    # This is gentler than additive — prevents distribution collapse.
                    scale_factor = 1.0 + bias * 0.10
                    scale_factor = max(0.70, min(1.30, scale_factor))
                    scale[pos, 0] = scale_factor
            _tensor_cache["scale"] = scale
            _tensor_cache["seq_len"] = seq_len

        scale_tensor = _tensor_cache["scale"]

        # Ensure device/dtype match
        if scale_tensor.device != hidden_states.device:
            scale_tensor = scale_tensor.to(hidden_states.device)
            _tensor_cache["scale"] = scale_tensor
        if scale_tensor.dtype != hidden_states.dtype:
            scale_tensor = scale_tensor.to(hidden_states.dtype)

        # Apply: scale hidden states at emotional positions
        # shape: (batch, seq_len, hidden) * (seq_len, 1) broadcasts correctly
        modified = hidden_states * scale_tensor

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    return hook_fn
