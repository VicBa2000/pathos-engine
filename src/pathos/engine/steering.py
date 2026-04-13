"""Emotional Steering Engine — Representation Engineering for emotional modification.

Extracts emotional direction vectors from transformer hidden states using
contrastive pairs (Zou et al. 2023, Rimsky et al. 2023). At runtime,
these vectors can be added to model activations to steer generation
toward the computed emotional state.

Extraction is OFFLINE (batch). Runtime steering adds ~5% overhead.
Only works with local open-source models (Ollama/llama-cpp/transformers).
Cloud APIs (Claude, GPT) gracefully degrade to prompt injection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Paths ---
_STEERING_DATA_DIR = Path(__file__).parent.parent / "steering_data"
_CONTRASTIVE_PAIRS_PATH = _STEERING_DATA_DIR / "contrastive_pairs.json"
_CACHED_VECTORS_DIR = _STEERING_DATA_DIR / "cached_vectors"

# --- Constants ---
EMOTIONAL_DIMENSIONS = ("valence", "arousal", "dominance", "certainty")

# Intensity scaling is quadratic: weak emotions don't steer.
# steer_weight = intensity ** INTENSITY_EXPONENT
INTENSITY_EXPONENT: float = 2.0

# Dimension weights for composite vector (sum = 1.0).
# Valence dominates because it's the most perceptually salient dimension.
DIMENSION_WEIGHTS: dict[str, float] = {
    "valence": 0.40,
    "arousal": 0.25,
    "dominance": 0.20,
    "certainty": 0.15,
}

# Max steering magnitude (L2 norm) to prevent catastrophic drift.
MAX_STEERING_NORM: float = 10.0

# --- Multi-Layer Steering ---
# Different layers encode different aspects of language processing.
# Early layers handle lexical/tonal choices, mid layers handle semantics,
# and late layers handle behavioral planning.

class LayerRole(str, Enum):
    """Role of a transformer layer in the multi-layer steering scheme."""
    EARLY = "early"    # Lexical: tone, word selection
    MID = "mid"        # Semantic: emotional understanding
    LATE = "late"      # Behavioral: planning, decision-making

# Per-role dimension weights: which emotional dimensions matter most at each depth.
# Early layers: valence dominates (tone/word choice depends on positive/negative).
# Mid layers: balanced (semantic understanding uses all dimensions).
# Late layers: dominance/certainty dominate (behavioral decisions, assertiveness).
LAYER_ROLE_WEIGHTS: dict[LayerRole, dict[str, float]] = {
    LayerRole.EARLY: {
        "valence": 0.50,
        "arousal": 0.30,
        "dominance": 0.12,
        "certainty": 0.08,
    },
    LayerRole.MID: {
        "valence": 0.35,
        "arousal": 0.25,
        "dominance": 0.22,
        "certainty": 0.18,
    },
    LayerRole.LATE: {
        "valence": 0.20,
        "arousal": 0.15,
        "dominance": 0.35,
        "certainty": 0.30,
    },
}

# Per-role scaling: controls overall steering intensity at each depth.
# Early layers get lighter steering (lexical changes are more visible to the user).
# Mid layers get full steering (semantic shift is the core effect).
# Late layers get moderate steering (behavioral changes should be noticeable but stable).
LAYER_ROLE_SCALING: dict[LayerRole, float] = {
    LayerRole.EARLY: 0.70,
    LayerRole.MID: 1.00,
    LayerRole.LATE: 0.85,
}


def classify_layer_role(layer_index: int, num_layers: int) -> LayerRole:
    """Classify a transformer layer as early, mid, or late.

    Uses relative position: first third = early, middle third = mid, last third = late.

    Args:
        layer_index: Zero-based layer index.
        num_layers: Total number of layers in the model.

    Returns:
        LayerRole classification.
    """
    if num_layers <= 0:
        return LayerRole.MID  # Fallback
    ratio = layer_index / num_layers
    if ratio < 0.33:
        return LayerRole.EARLY
    elif ratio < 0.67:
        return LayerRole.MID
    else:
        return LayerRole.LATE


# Known model architectures: model_family -> (num_layers, hidden_size).
# Used for validation; actual values come from the model at extraction time.
KNOWN_ARCHITECTURES: dict[str, dict[str, int]] = {
    "qwen2": {"num_layers": 36, "hidden_size": 2560},      # qwen3:4b uses qwen2 arch
    "qwen2_small": {"num_layers": 28, "hidden_size": 1536}, # qwen3:1.7b
    "llama": {"num_layers": 32, "hidden_size": 4096},       # llama3:8b
    "llama_small": {"num_layers": 32, "hidden_size": 3072}, # llama3.2:3b
    "mistral": {"num_layers": 32, "hidden_size": 4096},     # mistral:7b
}


@dataclass
class SteeringConfig:
    """Per-model steering configuration."""
    model_id: str
    num_layers: int = 0
    hidden_size: int = 0
    # Which layers to extract from (empty = auto-select mid layers).
    extraction_layers: list[int] = field(default_factory=list)

    def auto_select_layers(self) -> list[int]:
        """Select extraction layers if not manually configured.

        Strategy: sample early, mid, and late layers for broad coverage.
        Early layers (lexical), mid layers (semantic), late layers (behavioral).
        """
        if self.extraction_layers:
            return self.extraction_layers
        if self.num_layers == 0:
            return []
        n = self.num_layers
        # ~25%, ~50%, ~75% of layer depth
        early = max(1, n // 4)
        mid = n // 2
        late = min(n - 2, 3 * n // 4)
        return sorted({early, mid, late})


@dataclass
class CachedVectors:
    """Pre-computed steering vectors for one model."""
    model_id: str
    num_layers: int
    hidden_size: int
    # dimension -> layer_index -> vector (shape: hidden_size,)
    vectors: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)

    @property
    def available_dimensions(self) -> list[str]:
        return [d for d in EMOTIONAL_DIMENSIONS if d in self.vectors]

    @property
    def available_layers(self) -> set[int]:
        layers: set[int] = set()
        for layer_dict in self.vectors.values():
            layers.update(layer_dict.keys())
        return layers

    def has_dimension(self, dim: str) -> bool:
        return dim in self.vectors and len(self.vectors[dim]) > 0


def load_contrastive_pairs(path: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Load contrastive pairs from JSON file.

    Returns:
        Dict with keys = dimension names, values = {"positive": [...], "negative": [...]}.
    """
    p = path or _CONTRASTIVE_PAIRS_PATH
    if not p.exists():
        raise FileNotFoundError(f"Contrastive pairs file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # Validate structure
    pairs: dict[str, dict[str, list[str]]] = {}
    for dim in EMOTIONAL_DIMENSIONS:
        if dim not in data:
            raise ValueError(f"Missing dimension '{dim}' in contrastive pairs")
        dim_data = data[dim]
        if "positive" not in dim_data or "negative" not in dim_data:
            raise ValueError(f"Dimension '{dim}' must have 'positive' and 'negative' keys")
        pos = dim_data["positive"]
        neg = dim_data["negative"]
        if len(pos) < 10 or len(neg) < 10:
            raise ValueError(f"Dimension '{dim}' needs at least 10 pairs, got {len(pos)}/{len(neg)}")
        pairs[dim] = {"positive": pos, "negative": neg}
    return pairs


def compute_direction_vector(
    positive_activations: np.ndarray,
    negative_activations: np.ndarray,
) -> np.ndarray:
    """Compute emotional direction vector from contrastive activations.

    direction = mean(positive) - mean(negative)

    Both inputs should be 2D arrays of shape (num_samples, hidden_size).
    The result is a 1D vector of shape (hidden_size,) normalized to unit length.

    Args:
        positive_activations: Hidden states from positive prompts.
        negative_activations: Hidden states from negative prompts.

    Returns:
        Unit direction vector of shape (hidden_size,).
    """
    if positive_activations.ndim != 2 or negative_activations.ndim != 2:
        raise ValueError("Activations must be 2D arrays (num_samples, hidden_size)")
    if positive_activations.shape[1] != negative_activations.shape[1]:
        raise ValueError("Hidden sizes must match between positive and negative activations")

    mean_pos = positive_activations.mean(axis=0)
    mean_neg = negative_activations.mean(axis=0)
    direction = mean_pos - mean_neg

    # Normalize to unit vector
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        logger.warning("Direction vector near zero — contrastive pairs may be too similar")
        return direction
    return direction / norm


def compute_composite_vector(
    cached: CachedVectors,
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    layer: int,
    layer_role: LayerRole | None = None,
) -> np.ndarray | None:
    """Compute composite steering vector from emotional state for a specific layer.

    The composite vector is a weighted sum of dimension vectors, scaled by:
    - Each dimension's deviation from neutral (0.5 for arousal/dominance/certainty, 0.0 for valence)
    - The overall emotional intensity (quadratic scaling)
    - Dimension weights: role-specific (LAYER_ROLE_WEIGHTS) if layer_role given, else DIMENSION_WEIGHTS
    - Layer role scaling (LAYER_ROLE_SCALING) if layer_role given

    Args:
        cached: Pre-computed steering vectors for the model.
        valence: Emotional valence [-1, 1].
        arousal: Emotional arousal [0, 1].
        dominance: Emotional dominance [0, 1].
        certainty: Emotional certainty [0, 1].
        intensity: Overall emotional intensity [0, 1].
        layer: Which layer to compute the vector for.
        layer_role: Optional layer role for multi-layer steering.
            If provided, uses role-specific dimension weights and scaling.
            If None, uses global DIMENSION_WEIGHTS (backward compatible).

    Returns:
        Composite vector of shape (hidden_size,), or None if layer not available.
    """
    # Quadratic intensity scaling: weak emotions → negligible steering
    intensity_weight = intensity ** INTENSITY_EXPONENT
    if intensity_weight < 0.01:
        return None

    # Select dimension weights: role-specific or global
    if layer_role is not None:
        dim_weights = LAYER_ROLE_WEIGHTS[layer_role]
        role_scale = LAYER_ROLE_SCALING[layer_role]
    else:
        dim_weights = DIMENSION_WEIGHTS
        role_scale = 1.0

    # Map each dimension to a signed deviation from neutral.
    # valence: already [-1, 1], neutral = 0 → use directly
    # arousal/dominance/certainty: [0, 1], neutral = 0.5 → map to [-1, 1]
    deviations: dict[str, float] = {
        "valence": valence,                    # -1..+1
        "arousal": (arousal - 0.5) * 2.0,      # 0..1 → -1..+1
        "dominance": (dominance - 0.5) * 2.0,  # 0..1 → -1..+1
        "certainty": (certainty - 0.5) * 2.0,  # 0..1 → -1..+1
    }

    composite = np.zeros(cached.hidden_size, dtype=np.float32)
    any_added = False

    for dim in EMOTIONAL_DIMENSIONS:
        if not cached.has_dimension(dim):
            continue
        layer_vectors = cached.vectors[dim]
        if layer not in layer_vectors:
            continue

        direction = layer_vectors[layer]
        weight = dim_weights.get(dim, 0.0)
        deviation = deviations[dim]

        # Scale: weight * deviation * intensity² * role_scale * direction
        # Positive deviation → add direction (toward positive pole)
        # Negative deviation → subtract direction (toward negative pole)
        composite += (weight * deviation * intensity_weight * role_scale) * direction.astype(np.float32)
        any_added = True

    if not any_added:
        return None

    # Clamp magnitude to prevent catastrophic drift
    norm = np.linalg.norm(composite)
    if norm > MAX_STEERING_NORM:
        composite = composite * (MAX_STEERING_NORM / norm)

    return composite


def save_cached_vectors(cached: CachedVectors, directory: Path | None = None) -> Path:
    """Save pre-computed steering vectors to disk as .npz file.

    Args:
        cached: The vectors to save.
        directory: Override cache directory.

    Returns:
        Path to the saved file.
    """
    d = directory or _CACHED_VECTORS_DIR
    d.mkdir(parents=True, exist_ok=True)

    # Sanitize model_id for filename
    safe_id = cached.model_id.replace("/", "_").replace(":", "_").replace("\\", "_")
    path = d / f"{safe_id}.npz"

    # Build flat dict for npz: "valence_layer_5" -> vector
    arrays: dict[str, np.ndarray] = {}
    for dim, layer_dict in cached.vectors.items():
        for layer_idx, vec in layer_dict.items():
            arrays[f"{dim}_layer_{layer_idx}"] = vec

    # Save with metadata
    np.savez(
        path,
        _model_id=np.array([cached.model_id]),
        _num_layers=np.array([cached.num_layers]),
        _hidden_size=np.array([cached.hidden_size]),
        **arrays,
    )
    logger.info("Saved steering vectors for '%s' to %s", cached.model_id, path)
    return path


def load_cached_vectors(model_id: str, directory: Path | None = None) -> CachedVectors | None:
    """Load pre-computed steering vectors from disk.

    Args:
        model_id: Model identifier (e.g. "qwen3:4b").
        directory: Override cache directory.

    Returns:
        CachedVectors if found, None otherwise.
    """
    d = directory or _CACHED_VECTORS_DIR
    safe_id = model_id.replace("/", "_").replace(":", "_").replace("\\", "_")
    path = d / f"{safe_id}.npz"

    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        logger.error("Failed to load cached vectors from %s: %s", path, e)
        return None

    stored_id = str(data["_model_id"][0])
    num_layers = int(data["_num_layers"][0])
    hidden_size = int(data["_hidden_size"][0])

    vectors: dict[str, dict[int, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("_"):
            continue
        # Parse "valence_layer_5" -> ("valence", 5)
        parts = key.rsplit("_layer_", 1)
        if len(parts) != 2:
            continue
        dim, layer_str = parts
        if dim not in EMOTIONAL_DIMENSIONS:
            continue
        try:
            layer_idx = int(layer_str)
        except ValueError:
            continue
        if dim not in vectors:
            vectors[dim] = {}
        vectors[dim][layer_idx] = data[key]

    cached = CachedVectors(
        model_id=stored_id,
        num_layers=num_layers,
        hidden_size=hidden_size,
        vectors=vectors,
    )
    logger.info(
        "Loaded steering vectors for '%s': %d dimensions, %d layers",
        stored_id, len(cached.available_dimensions), len(cached.available_layers),
    )
    return cached


class EmotionalSteeringEngine:
    """Manages emotional steering vector extraction, caching, and composition.

    This is the main interface for the steering system. It:
    1. Loads/saves pre-computed vectors from cache
    2. Orchestrates offline extraction (requires transformers or llama-cpp-python)
    3. Computes composite steering vectors from emotional state at runtime

    Extraction is always OFFLINE. Runtime composition is lightweight (~0.1ms).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHED_VECTORS_DIR
        self._cached: CachedVectors | None = None
        self._contrastive_pairs: dict[str, dict[str, list[str]]] | None = None

    @property
    def is_ready(self) -> bool:
        """Whether steering vectors are loaded and ready for use."""
        return self._cached is not None and len(self._cached.available_dimensions) > 0

    @property
    def model_id(self) -> str | None:
        return self._cached.model_id if self._cached else None

    @property
    def available_dimensions(self) -> list[str]:
        return self._cached.available_dimensions if self._cached else []

    @property
    def available_layers(self) -> set[int]:
        return self._cached.available_layers if self._cached else set()

    def load_vectors(self, model_id: str) -> bool:
        """Attempt to load cached vectors for a model.

        Returns True if vectors were found and loaded.
        """
        cached = load_cached_vectors(model_id, self._cache_dir)
        if cached is not None:
            self._cached = cached
            return True
        return False

    def save_vectors(self) -> Path | None:
        """Save current vectors to cache. Returns path or None if nothing to save."""
        if self._cached is None:
            return None
        return save_cached_vectors(self._cached, self._cache_dir)

    def get_contrastive_pairs(self) -> dict[str, dict[str, list[str]]]:
        """Load and cache contrastive pairs."""
        if self._contrastive_pairs is None:
            self._contrastive_pairs = load_contrastive_pairs()
        return self._contrastive_pairs

    def extract_vectors(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        layers: list[int] | None = None,
        max_seq_len: int = 64,
        batch_size: int = 8,
    ) -> CachedVectors:
        """Extract emotional direction vectors from a model (OFFLINE operation).

        This requires either a HuggingFace transformers model or a llama-cpp-python
        model with hidden state access. It is designed to run as a batch job,
        NOT during live inference.

        Args:
            model: A model object that supports hidden state extraction.
                   For transformers: AutoModelForCausalLM with output_hidden_states=True.
                   For llama-cpp-python: Llama with embedding=True.
            tokenizer: The tokenizer for the model.
            model_id: Identifier string (e.g. "qwen3:4b").
            layers: Which layers to extract from. None = auto-select.
            max_seq_len: Max tokens per prompt for extraction.
            batch_size: Batch size for processing prompts.

        Returns:
            CachedVectors with extracted direction vectors.
        """
        pairs = self.get_contrastive_pairs()

        # Detect model type and get architecture info
        num_layers, hidden_size = _detect_model_architecture(model)
        config = SteeringConfig(
            model_id=model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            extraction_layers=layers or [],
        )
        target_layers = config.auto_select_layers()
        if not target_layers:
            raise ValueError(
                f"No extraction layers determined for model with {num_layers} layers. "
                "Please specify layers manually."
            )

        logger.info(
            "Extracting steering vectors for '%s' (layers=%s, hidden=%d)",
            model_id, target_layers, hidden_size,
        )

        vectors: dict[str, dict[int, np.ndarray]] = {}
        for dim in EMOTIONAL_DIMENSIONS:
            dim_pairs = pairs[dim]
            pos_prompts = dim_pairs["positive"]
            neg_prompts = dim_pairs["negative"]

            dim_vectors: dict[int, np.ndarray] = {}
            for layer_idx in target_layers:
                pos_acts = _extract_hidden_states(
                    model, tokenizer, pos_prompts, layer_idx,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                )
                neg_acts = _extract_hidden_states(
                    model, tokenizer, neg_prompts, layer_idx,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                )
                direction = compute_direction_vector(pos_acts, neg_acts)
                dim_vectors[layer_idx] = direction
                logger.debug(
                    "  %s layer %d: norm=%.4f", dim, layer_idx, np.linalg.norm(direction),
                )

            vectors[dim] = dim_vectors

        self._cached = CachedVectors(
            model_id=model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            vectors=vectors,
        )
        logger.info("Extraction complete: %d dimensions × %d layers", len(vectors), len(target_layers))
        return self._cached

    def get_steering_vector(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        certainty: float,
        intensity: float,
        layer: int,
        multilayer: bool = True,
    ) -> np.ndarray | None:
        """Compute composite steering vector for a given emotional state and layer.

        This is the RUNTIME method — called during inference to get the activation
        addition vector. Lightweight (~0.1ms).

        Args:
            multilayer: If True (default), uses layer-role-aware dimension weights
                and scaling. If False, uses global DIMENSION_WEIGHTS (legacy behavior).

        Returns None if steering is not ready or intensity too low.
        """
        if not self.is_ready:
            return None
        assert self._cached is not None
        layer_role: LayerRole | None = None
        if multilayer and self._cached.num_layers > 0:
            layer_role = classify_layer_role(layer, self._cached.num_layers)
        return compute_composite_vector(
            self._cached,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            certainty=certainty,
            intensity=intensity,
            layer=layer,
            layer_role=layer_role,
        )

    def get_steering_vectors_all_layers(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        certainty: float,
        intensity: float,
        multilayer: bool = True,
    ) -> dict[int, np.ndarray]:
        """Compute steering vectors for ALL available layers.

        Args:
            multilayer: If True, uses layer-role-aware steering per layer.

        Returns dict of layer_index -> composite_vector (only non-None entries).
        """
        result: dict[int, np.ndarray] = {}
        if not self.is_ready:
            return result
        for layer in self.available_layers:
            vec = self.get_steering_vector(
                valence, arousal, dominance, certainty, intensity, layer,
                multilayer=multilayer,
            )
            if vec is not None:
                result[layer] = vec
        return result

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic info for research endpoint."""
        if not self.is_ready:
            return {"status": "not_loaded", "model_id": None}
        c = self._cached
        assert c is not None
        # Classify each available layer by role
        layer_roles: dict[str, list[int]] = {"early": [], "mid": [], "late": []}
        for layer_idx in sorted(c.available_layers):
            role = classify_layer_role(layer_idx, c.num_layers)
            layer_roles[role.value].append(layer_idx)
        return {
            "status": "ready",
            "model_id": c.model_id,
            "num_layers": c.num_layers,
            "hidden_size": c.hidden_size,
            "dimensions": c.available_dimensions,
            "layers": sorted(c.available_layers),
            "layer_roles": layer_roles,
            "multilayer": True,
            "total_vectors": sum(len(ld) for ld in c.vectors.values()),
        }


# --- Internal helpers ---

def _detect_model_architecture(model: Any) -> tuple[int, int]:
    """Detect num_layers and hidden_size from a model object.

    Supports:
    - HuggingFace transformers models (model.config)
    - llama-cpp-python models (model.n_embd(), model.n_layer() or metadata)

    Returns:
        (num_layers, hidden_size)
    """
    # HuggingFace transformers
    if hasattr(model, "config"):
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        if num_layers > 0 and hidden_size > 0:
            return (num_layers, hidden_size)

    # llama-cpp-python
    if hasattr(model, "n_embd") and callable(model.n_embd):
        hidden_size = model.n_embd()
        num_layers = 0
        if hasattr(model, "metadata"):
            meta = model.metadata
            for key in ("llama.block_count", "block_count"):
                if key in meta:
                    num_layers = int(meta[key])
                    break
        if num_layers == 0 and hasattr(model, "n_layer") and callable(model.n_layer):
            num_layers = model.n_layer()
        if num_layers > 0 and hidden_size > 0:
            return (num_layers, hidden_size)

    raise ValueError(
        "Cannot detect model architecture. Provide a HuggingFace transformers model "
        "or llama-cpp-python model. Got: " + type(model).__name__
    )


def _extract_hidden_states(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layer: int,
    max_seq_len: int = 64,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract hidden states from a specific layer for a list of prompts.

    Returns array of shape (num_prompts, hidden_size) — the last-token
    hidden state at the specified layer for each prompt.

    Supports HuggingFace transformers models.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required for steering vector extraction. "
            "Install with: pip install torch"
        )

    all_states: list[np.ndarray] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (num_layers+1) tensors
        # Each tensor has shape (batch_size, seq_len, hidden_size)
        # Index 0 = embedding layer, so layer N = index N+1
        hidden_states = outputs.hidden_states
        if layer + 1 >= len(hidden_states):
            raise ValueError(
                f"Layer {layer} out of range (model has {len(hidden_states) - 1} layers)"
            )

        layer_output = hidden_states[layer + 1]  # (batch, seq_len, hidden)

        # Get last non-padding token's hidden state for each sample
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            # Find last valid token index per sample
            lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            for j in range(layer_output.size(0)):
                last_idx = lengths[j].item()
                state = layer_output[j, int(last_idx), :].cpu().numpy()
                all_states.append(state)
        else:
            # No padding, use last token
            for j in range(layer_output.size(0)):
                state = layer_output[j, -1, :].cpu().numpy()
                all_states.append(state)

    return np.stack(all_states, axis=0)


# ==========================================================================
# Runtime Steering Hooks — Activation Addition during inference
# ==========================================================================

class SteeringHook:
    """Registers forward hooks on transformer layers to add steering vectors.

    During inference, each hooked layer's output has the steering vector added
    to the hidden states. This shifts the model's internal representations
    toward the computed emotional direction WITHOUT modifying prompts.

    Usage:
        hook = SteeringHook(model, engine, emotional_state)
        hook.apply()           # Register hooks
        output = model(...)    # Generate with steering
        hook.remove()          # Clean up hooks

    Or as context manager:
        with SteeringHook(model, engine, emotional_state):
            output = model(...)

    Only works with HuggingFace transformers models that have a .model.layers
    attribute (LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, etc).
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
    ) -> None:
        self._model = model
        self._engine = engine
        self._valence = valence
        self._arousal = arousal
        self._dominance = dominance
        self._certainty = certainty
        self._intensity = intensity
        self._handles: list[Any] = []
        self._vectors_applied: dict[int, float] = {}  # layer -> norm of applied vector
        self._layer_roles: dict[int, str] = {}  # layer -> role name for diagnostics
        self._raw_vectors: dict[int, np.ndarray] = {}  # raw vectors before momentum

    @property
    def is_applied(self) -> bool:
        return len(self._handles) > 0

    @property
    def vectors_applied(self) -> dict[int, float]:
        """layer_index -> L2 norm of the steering vector that was applied."""
        return dict(self._vectors_applied)

    @property
    def layer_roles(self) -> dict[int, str]:
        """layer_index -> role name ("early", "mid", "late") for applied layers."""
        return dict(self._layer_roles)

    def _compute_raw_vectors(self) -> dict[int, np.ndarray]:
        """Compute raw (pre-momentum) steering vectors for all available layers."""
        return self._engine.get_steering_vectors_all_layers(
            valence=self._valence,
            arousal=self._arousal,
            dominance=self._dominance,
            certainty=self._certainty,
            intensity=self._intensity,
            multilayer=True,
        )

    @property
    def raw_vectors(self) -> dict[int, np.ndarray]:
        """Get the raw vectors computed during the last apply() call.

        Useful for recording into momentum history BEFORE blending.
        """
        return dict(self._raw_vectors)

    def apply(self, momentum: SteeringMomentum | None = None) -> bool:
        """Register forward hooks on model layers.

        Uses multi-layer steering: each layer gets role-specific dimension weights
        and scaling (early=lexical, mid=semantic, late=behavioral).

        Args:
            momentum: Optional SteeringMomentum to blend with history.
                If provided, current vectors are blended with decayed past vectors.

        Returns True if at least one hook was registered, False otherwise.
        """
        if self.is_applied:
            self.remove()

        if not self._engine.is_ready:
            return False

        # Get the model's transformer layers
        layers = _get_model_layers(self._model)
        if layers is None:
            logger.warning("Cannot find transformer layers on model %s", type(self._model).__name__)
            return False

        # Compute raw vectors
        raw = self._compute_raw_vectors()
        self._raw_vectors = raw

        # Apply momentum blend if available
        if momentum is not None and momentum.has_history:
            vectors = momentum.apply_momentum(raw)
        else:
            vectors = raw

        any_hooked = False

        for layer_idx, vec in vectors.items():
            if layer_idx >= len(layers):
                continue

            vec_norm = float(np.linalg.norm(vec))
            if vec_norm < 1e-8:
                continue

            # Create the hook function — captures vec by closure
            hook_fn = _make_steering_hook(vec)
            handle = layers[layer_idx].register_forward_hook(hook_fn)
            self._handles.append(handle)
            self._vectors_applied[layer_idx] = vec_norm

            # Track layer role for diagnostics
            cached = self._engine._cached
            if cached and cached.num_layers > 0:
                role = classify_layer_role(layer_idx, cached.num_layers)
                self._layer_roles[layer_idx] = role.value

            any_hooked = True

        if any_hooked:
            logger.debug(
                "Multi-layer steering hooks applied: %d layers, roles=%s, norms=%s%s",
                len(self._handles),
                self._layer_roles,
                {k: f"{v:.4f}" for k, v in self._vectors_applied.items()},
                f", momentum={momentum.momentum:.2f}" if momentum and momentum.has_history else "",
            )
        return any_hooked

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._vectors_applied.clear()
        self._layer_roles.clear()

    def __enter__(self) -> "SteeringHook":
        self.apply()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    def __del__(self) -> None:
        self.remove()


# ===========================================================================
# Steering Momentum — temporal inertia across turns
# ===========================================================================

# Default momentum factor: how much of the previous vector carries over.
# Range [0, 1). Higher = more inertia (emotions linger longer in hidden states).
DEFAULT_MOMENTUM: float = 0.30

# Maximum number of history entries per layer to keep.
MAX_MOMENTUM_HISTORY: int = 5


@dataclass
class SteeringMomentum:
    """Maintains temporal inertia for steering vectors across turns.

    Instead of computing fresh steering vectors each turn, momentum blends
    the current vector with a decayed history. This creates a "recurrent
    emotional hidden state" — prolonged sadness doesn't reset instantly,
    and emotional shifts are smoother.

    Momentum factor is modulated by neuroticism: high neuroticism = more
    inertia (emotions stick longer). Low neuroticism = faster adaptation.
    """

    # layer_index -> list of past vectors (most recent first)
    _history: dict[int, list[np.ndarray]] = field(default_factory=dict)
    _momentum: float = DEFAULT_MOMENTUM

    @property
    def momentum(self) -> float:
        return self._momentum

    @property
    def has_history(self) -> bool:
        return len(self._history) > 0

    @property
    def turns_stored(self) -> int:
        """Number of turns with stored history (based on first layer)."""
        if not self._history:
            return 0
        first_key = next(iter(self._history))
        return len(self._history[first_key])

    def configure_from_personality(self, neuroticism: float) -> None:
        """Set momentum factor based on neuroticism trait.

        High neuroticism (0.8+): momentum 0.45 — emotions linger longer
        Low neuroticism (0.2-): momentum 0.15 — adapts quickly
        Average (0.5): momentum 0.30 (default)

        Args:
            neuroticism: Big Five neuroticism trait [0, 1].
        """
        # Linear mapping: neuroticism [0,1] → momentum [0.10, 0.50]
        self._momentum = max(0.10, min(0.50, 0.10 + neuroticism * 0.40))

    def apply_momentum(
        self,
        current_vectors: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Blend current steering vectors with decayed history.

        For each layer:
            blended = (1 - momentum) * current + momentum * weighted_history
            weighted_history = sum(history[i] * momentum^i) / sum(momentum^i)

        Args:
            current_vectors: layer_index -> current composite vector.

        Returns:
            layer_index -> blended vector (same keys as input).
        """
        if self._momentum < 0.01:
            return current_vectors

        blended: dict[int, np.ndarray] = {}

        for layer, current in current_vectors.items():
            history = self._history.get(layer, [])
            if not history:
                blended[layer] = current
                continue

            # Compute weighted history (exponential decay)
            weighted_sum = np.zeros_like(current)
            weight_total = 0.0
            for i, past_vec in enumerate(history):
                w = self._momentum ** (i + 1)  # more recent = higher weight
                weighted_sum += w * past_vec
                weight_total += w

            if weight_total > 0:
                avg_history = weighted_sum / weight_total
                blended[layer] = (1.0 - self._momentum) * current + self._momentum * avg_history
            else:
                blended[layer] = current

            # Clamp blended vector norm
            norm = float(np.linalg.norm(blended[layer]))
            if norm > MAX_STEERING_NORM:
                blended[layer] = blended[layer] * (MAX_STEERING_NORM / norm)

        return blended

    def record_turn(self, vectors: dict[int, np.ndarray]) -> None:
        """Store the current turn's vectors in history.

        Called AFTER generation completes. Keeps at most MAX_MOMENTUM_HISTORY
        entries per layer (most recent first).

        Args:
            vectors: layer_index -> composite vector used this turn.
        """
        for layer, vec in vectors.items():
            if layer not in self._history:
                self._history[layer] = []
            self._history[layer].insert(0, vec.copy())
            # Trim to max history
            if len(self._history[layer]) > MAX_MOMENTUM_HISTORY:
                self._history[layer] = self._history[layer][:MAX_MOMENTUM_HISTORY]

    def clear(self) -> None:
        """Clear all history (e.g., session reset)."""
        self._history.clear()

    def get_info(self) -> dict[str, Any]:
        """Diagnostic info for research endpoint."""
        return {
            "momentum_factor": round(self._momentum, 3),
            "has_history": self.has_history,
            "turns_stored": self.turns_stored,
            "layers_tracked": sorted(self._history.keys()),
        }


def _get_model_layers(model: Any) -> list[Any] | None:
    """Get the list of transformer layers from a model.

    Supports common HuggingFace architectures:
    - model.model.layers (Llama, Qwen2, Mistral, Gemma, Phi)
    - model.transformer.h (GPT-2, GPT-Neo)
    """
    # Most common: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
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


def _make_steering_hook(steering_vector: np.ndarray):
    """Create a forward hook function that adds a steering vector to layer output.

    The hook adds the steering vector to the hidden states (first element of output tuple).
    This implements activation addition (Rimsky et al., 2023).

    Args:
        steering_vector: 1D array of shape (hidden_size,).

    Returns:
        Hook function compatible with PyTorch register_forward_hook.
    """
    _tensor_cache: dict[str, Any] = {}

    def hook_fn(module: Any, input: Any, output: Any) -> Any:
        try:
            import torch
        except ImportError:
            return output

        # Convert steering vector to tensor (cached for performance)
        cache_key = "vec"
        if cache_key not in _tensor_cache:
            _tensor_cache[cache_key] = torch.tensor(
                steering_vector, dtype=torch.float32,
            )

        vec_tensor = _tensor_cache[cache_key]

        # Output is typically a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Move vector to same device
            if vec_tensor.device != hidden_states.device:
                vec_tensor = vec_tensor.to(hidden_states.device)
                _tensor_cache[cache_key] = vec_tensor
            # Cast to same dtype
            if vec_tensor.dtype != hidden_states.dtype:
                vec_tensor = vec_tensor.to(hidden_states.dtype)
            # Add steering vector to ALL token positions
            # hidden_states shape: (batch, seq_len, hidden_size)
            modified = hidden_states + vec_tensor.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            # Single tensor output
            if vec_tensor.device != output.device:
                vec_tensor = vec_tensor.to(output.device)
                _tensor_cache[cache_key] = vec_tensor
            if vec_tensor.dtype != output.dtype:
                vec_tensor = vec_tensor.to(output.dtype)
            return output + vec_tensor.unsqueeze(0).unsqueeze(0)

    return hook_fn
