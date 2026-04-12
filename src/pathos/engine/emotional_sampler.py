"""Emotional Sampler — Maps emotional state to LLM sampling parameters.

Instead of only modifying temperature, the sampler adjusts multiple degrees
of freedom in the generation process to produce text that is naturally
congruent with the agent's emotional state:

  - temperature: emotional intensity → higher temp, calm → lower temp
  - top_p: certainty → lower top_p (predictable), uncertainty → higher top_p
  - top_k: arousal → lower top_k (focused), low arousal → higher top_k
  - repetition_penalty: rumination (low valence + low arousal) → lower penalty,
    joy/excitement → higher penalty (more varied)
  - presence_penalty: openness (high dominance + positive valence) → higher
    presence penalty (explores new topics)
  - frequency_penalty: high arousal → lower penalty (allows emphasis/repetition)

All parameters are computed as modifiers around base values. The base values
come from the caller (typically the existing temperature logic in main.py).
The sampler returns absolute values, already clamped to safe ranges.

Works with any LLM provider. Ollama supports all params natively.
TransformersProvider passes them to model.generate().
Cloud APIs (Claude) support only temperature — other params are ignored gracefully.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- Paths ---
_SAMPLING_DATA_DIR = Path(__file__).parent.parent / "sampling_data"
_VOCABULARY_PATH = _SAMPLING_DATA_DIR / "emotional_vocabulary.json"

# --- Base values (neutral emotional state → these exact values) ---
BASE_TEMPERATURE: float = 0.7
BASE_TOP_P: float = 0.9
BASE_TOP_K: int = 40
BASE_REPETITION_PENALTY: float = 1.1
BASE_PRESENCE_PENALTY: float = 0.0
BASE_FREQUENCY_PENALTY: float = 0.0

# --- Safe ranges ---
MIN_TEMPERATURE: float = 0.10
MAX_TEMPERATURE: float = 1.50
MIN_TOP_P: float = 0.50
MAX_TOP_P: float = 1.00
MIN_TOP_K: int = 10
MAX_TOP_K: int = 100
MIN_REPETITION_PENALTY: float = 1.00
MAX_REPETITION_PENALTY: float = 1.40
MIN_PRESENCE_PENALTY: float = -0.20
MAX_PRESENCE_PENALTY: float = 0.60
MIN_FREQUENCY_PENALTY: float = -0.20
MAX_FREQUENCY_PENALTY: float = 0.50


@dataclass(frozen=True)
class SamplingParams:
    """Computed sampling parameters from emotional state.

    All values are absolute (not deltas) and already clamped to safe ranges.
    """
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float

    # Metadata for diagnostics
    source: str = "neutral"  # Description of dominant emotional influence


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def compute_sampling_params(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    base_temperature: float = BASE_TEMPERATURE,
) -> SamplingParams:
    """Compute sampling parameters from emotional state.

    The emotional state modulates each parameter around its base value.
    The strength of modulation scales with emotional intensity — neutral
    states (intensity ~0) produce base values unchanged.

    Args:
        valence: Emotional valence [-1, 1]. Positive = happy, negative = sad/angry.
        arousal: Emotional arousal [0, 1]. High = excited/tense, low = calm/lethargic.
        dominance: Emotional dominance [0, 1]. High = in control, low = submissive.
        certainty: Emotional certainty [0, 1]. High = confident, low = confused.
        intensity: Overall emotional intensity [0, 1]. Scales all modulations.
        base_temperature: Starting temperature before emotional modulation.

    Returns:
        SamplingParams with all values clamped to safe ranges.
    """
    # Intensity gates everything: weak emotions don't modify sampling
    if intensity < 0.05:
        return SamplingParams(
            temperature=_clamp(base_temperature, MIN_TEMPERATURE, MAX_TEMPERATURE),
            top_p=BASE_TOP_P,
            top_k=BASE_TOP_K,
            repetition_penalty=BASE_REPETITION_PENALTY,
            presence_penalty=BASE_PRESENCE_PENALTY,
            frequency_penalty=BASE_FREQUENCY_PENALTY,
            source="neutral",
        )

    # Scale factor: how much to shift from base. Linear with intensity.
    scale = intensity

    # --- Temperature ---
    # High arousal → higher temperature (more creative/chaotic)
    # Low arousal → lower temperature (more measured/predictable)
    # Intensity amplifies the effect
    arousal_deviation = (arousal - 0.5) * 2.0  # [-1, +1]
    temp_delta = arousal_deviation * 0.30 * scale  # max ±0.30
    temperature = _clamp(
        base_temperature + temp_delta,
        MIN_TEMPERATURE, MAX_TEMPERATURE,
    )

    # --- Top-p (nucleus sampling) ---
    # High certainty → lower top_p (focused, predictable word choices)
    # Low certainty → higher top_p (explores more options, hesitant)
    certainty_deviation = (certainty - 0.5) * 2.0  # [-1, +1]
    top_p_delta = -certainty_deviation * 0.20 * scale  # inverted: high certainty → lower top_p
    top_p = _clamp(BASE_TOP_P + top_p_delta, MIN_TOP_P, MAX_TOP_P)

    # --- Top-k ---
    # High arousal → lower top_k (narrow focus, intense)
    # Low arousal → higher top_k (broader exploration, relaxed)
    top_k_delta = -arousal_deviation * 25 * scale  # inverted: high arousal → fewer options
    top_k = int(_clamp(BASE_TOP_K + top_k_delta, MIN_TOP_K, MAX_TOP_K))

    # --- Repetition penalty ---
    # Rumination (negative valence + low arousal) → LOWER penalty (allows repetitive patterns)
    # Joy/excitement (positive valence + high arousal) → HIGHER penalty (more varied output)
    valence_arousal = valence * 0.6 + arousal_deviation * 0.4  # combined signal [-1, +1]
    rep_delta = valence_arousal * 0.15 * scale  # max ±0.15
    repetition_penalty = _clamp(
        BASE_REPETITION_PENALTY + rep_delta,
        MIN_REPETITION_PENALTY, MAX_REPETITION_PENALTY,
    )

    # --- Presence penalty ---
    # High dominance + positive valence → higher presence penalty (explores new topics)
    # Low dominance → lower presence penalty (stays on current topic, defers)
    dominance_deviation = (dominance - 0.5) * 2.0  # [-1, +1]
    openness_signal = dominance_deviation * 0.5 + max(valence, 0.0) * 0.5
    pres_delta = openness_signal * 0.30 * scale  # max ±0.30
    presence_penalty = _clamp(
        BASE_PRESENCE_PENALTY + pres_delta,
        MIN_PRESENCE_PENALTY, MAX_PRESENCE_PENALTY,
    )

    # --- Frequency penalty ---
    # High arousal → lower frequency penalty (allows emphasis through repetition)
    # Low arousal → slightly higher (avoids monotone repetition)
    freq_delta = -arousal_deviation * 0.20 * scale  # inverted: high arousal → lower
    frequency_penalty = _clamp(
        BASE_FREQUENCY_PENALTY + freq_delta,
        MIN_FREQUENCY_PENALTY, MAX_FREQUENCY_PENALTY,
    )

    # Determine dominant source for diagnostics
    source = _determine_source(valence, arousal, dominance, certainty, intensity)

    return SamplingParams(
        temperature=round(temperature, 3),
        top_p=round(top_p, 3),
        top_k=top_k,
        repetition_penalty=round(repetition_penalty, 3),
        presence_penalty=round(presence_penalty, 3),
        frequency_penalty=round(frequency_penalty, 3),
        source=source,
    )


def _determine_source(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
) -> str:
    """Determine dominant emotional influence on sampling for diagnostics."""
    if intensity < 0.15:
        return "low_intensity"

    # Find the most deviated dimension
    deviations = {
        "valence": abs(valence),
        "arousal": abs(arousal - 0.5) * 2,
        "dominance": abs(dominance - 0.5) * 2,
        "certainty": abs(certainty - 0.5) * 2,
    }
    dominant = max(deviations, key=lambda k: deviations[k])

    labels = {
        "valence": "positive_valence" if valence > 0 else "negative_valence",
        "arousal": "high_arousal" if arousal > 0.5 else "low_arousal",
        "dominance": "high_dominance" if dominance > 0.5 else "low_dominance",
        "certainty": "high_certainty" if certainty > 0.5 else "low_certainty",
    }
    return labels[dominant]


# ===========================================================================
# Token-Level Logit Bias — Shift vocabulary probabilities by emotional state
# ===========================================================================

# Bias magnitude limits. Ollama uses logit-space bias (additive to log-probs).
# ±2.0 is a moderate shift; ±5.0 is very strong.
MAX_BIAS: float = 2.5
MIN_BIAS: float = -2.5

# Emotional vocabulary categories → which emotional dimensions affect them.
VOCABULARY_CATEGORIES = (
    "positive", "negative", "high_energy", "low_energy",
    "uncertainty", "assertive",
)


@dataclass(frozen=True)
class TokenBiasResult:
    """Result of computing token-level logit biases.

    word_biases: word → bias value (tokenizer-agnostic, for diagnostics).
    token_biases: token_id → bias value (tokenizer-specific, for LLM generation).
    categories_active: which vocabulary categories were activated and their directions.
    """
    word_biases: dict[str, float]
    token_biases: dict[int, float] = field(default_factory=dict)
    categories_active: dict[str, float] = field(default_factory=dict)


# Cache for loaded vocabulary
_vocabulary_cache: dict[str, list[str]] | None = None


def load_emotional_vocabulary(path: Path | None = None) -> dict[str, list[str]]:
    """Load emotional vocabulary from JSON file.

    Returns:
        Dict with category names as keys, word lists as values.
    """
    global _vocabulary_cache
    if _vocabulary_cache is not None and path is None:
        return _vocabulary_cache

    p = path or _VOCABULARY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Emotional vocabulary not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    vocab: dict[str, list[str]] = {}
    for cat in VOCABULARY_CATEGORIES:
        if cat not in data:
            raise ValueError(f"Missing category '{cat}' in emotional vocabulary")
        words = data[cat]
        if not isinstance(words, list) or len(words) < 5:
            raise ValueError(f"Category '{cat}' needs at least 5 words, got {len(words) if isinstance(words, list) else 0}")
        vocab[cat] = [w for w in words if isinstance(w, str)]

    if path is None:
        _vocabulary_cache = vocab
    return vocab


def compute_token_bias(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    vocabulary: dict[str, list[str]] | None = None,
) -> TokenBiasResult:
    """Compute word-level logit biases from emotional state.

    Maps emotional dimensions to vocabulary category activations:
    - valence: positive words boosted when positive, negative boosted when negative
    - arousal: high_energy boosted when high, low_energy boosted when low
    - dominance + certainty: assertive boosted when high, uncertainty boosted when low

    The bias for each word is the sum of all category activations it belongs to.
    Intensity gates everything: low intensity → empty biases.

    Args:
        valence: [-1, 1]
        arousal: [0, 1]
        dominance: [0, 1]
        certainty: [0, 1]
        intensity: [0, 1]
        vocabulary: Override vocabulary (for testing). None = load from file.

    Returns:
        TokenBiasResult with word_biases (tokenizer-agnostic) and categories_active.
    """
    # Intensity gate: weak emotions don't bias vocabulary
    if intensity < 0.10:
        return TokenBiasResult(word_biases={}, categories_active={})

    vocab = vocabulary or load_emotional_vocabulary()
    scale = intensity

    # Compute category activations: how much to boost each category
    # Each activation is in [-1, +1] where positive = boost, negative = suppress
    categories: dict[str, float] = {}

    # Valence → positive/negative vocabulary
    # Strong positive valence: boost positive words, suppress negative
    # Strong negative valence: boost negative words, suppress positive
    if abs(valence) > 0.15:
        categories["positive"] = valence * scale       # +val → boost positive
        categories["negative"] = -valence * scale       # +val → suppress negative

    # Arousal → energy vocabulary
    arousal_dev = (arousal - 0.5) * 2.0  # [-1, +1]
    if abs(arousal_dev) > 0.15:
        categories["high_energy"] = arousal_dev * scale   # high arousal → boost energy
        categories["low_energy"] = -arousal_dev * scale    # high arousal → suppress low energy

    # Dominance + Certainty → assertive/uncertainty vocabulary
    # Combined signal: high dominance + high certainty → assertive
    dom_dev = (dominance - 0.5) * 2.0
    cert_dev = (certainty - 0.5) * 2.0
    assertive_signal = (dom_dev * 0.5 + cert_dev * 0.5)
    if abs(assertive_signal) > 0.15:
        categories["assertive"] = assertive_signal * scale
        categories["uncertainty"] = -assertive_signal * scale

    if not categories:
        return TokenBiasResult(word_biases={}, categories_active={})

    # Build word → bias mapping
    word_biases: dict[str, float] = {}
    for cat, activation in categories.items():
        if cat not in vocab:
            continue
        # Scale activation to bias magnitude
        bias_value = _clamp(activation * MAX_BIAS, MIN_BIAS, MAX_BIAS)
        if abs(bias_value) < 0.05:
            continue
        for word in vocab[cat]:
            # Accumulate biases for words that appear in multiple categories
            word_biases[word] = _clamp(
                word_biases.get(word, 0.0) + bias_value,
                MIN_BIAS, MAX_BIAS,
            )

    # Remove near-zero biases
    word_biases = {w: round(b, 3) for w, b in word_biases.items() if abs(b) >= 0.05}

    return TokenBiasResult(
        word_biases=word_biases,
        categories_active={cat: round(act, 3) for cat, act in categories.items()},
    )


def resolve_token_ids(
    word_biases: dict[str, float],
    tokenizer: Any,
) -> dict[int, float]:
    """Resolve word-level biases to token-level biases using a tokenizer.

    Each word is tokenized, and its bias is applied to all resulting token IDs.
    Multi-token words split the bias across their tokens.

    Args:
        word_biases: word → bias value from compute_token_bias().
        tokenizer: A HuggingFace-compatible tokenizer with encode() method.

    Returns:
        token_id → bias value dictionary, suitable for logit_bias parameter.
    """
    if not word_biases:
        return {}

    token_biases: dict[int, float] = {}

    for word, bias in word_biases.items():
        try:
            # Encode word to token IDs (without special tokens)
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                continue
            # Split bias across tokens for multi-token words
            per_token_bias = bias / len(token_ids)
            for tid in token_ids:
                # Accumulate (same token from different words)
                current = token_biases.get(tid, 0.0)
                token_biases[tid] = _clamp(current + per_token_bias, MIN_BIAS, MAX_BIAS)
        except Exception:
            # Skip words that fail to tokenize
            continue

    return {tid: round(b, 3) for tid, b in token_biases.items() if abs(b) >= 0.05}
