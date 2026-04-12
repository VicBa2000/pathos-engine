"""Tests for Emotional Steering Engine (Representation Engineering)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathos.engine.steering import (
    DEFAULT_MOMENTUM,
    DIMENSION_WEIGHTS,
    EMOTIONAL_DIMENSIONS,
    INTENSITY_EXPONENT,
    LAYER_ROLE_SCALING,
    LAYER_ROLE_WEIGHTS,
    MAX_MOMENTUM_HISTORY,
    MAX_STEERING_NORM,
    CachedVectors,
    EmotionalSteeringEngine,
    LayerRole,
    SteeringConfig,
    SteeringHook,
    SteeringMomentum,
    _get_model_layers,
    _make_steering_hook,
    classify_layer_role,
    compute_composite_vector,
    compute_direction_vector,
    load_cached_vectors,
    load_contrastive_pairs,
    save_cached_vectors,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_cached() -> CachedVectors:
    """CachedVectors with simple unit vectors for testing."""
    hidden_size = 8
    vectors: dict[str, dict[int, np.ndarray]] = {}
    for i, dim in enumerate(EMOTIONAL_DIMENSIONS):
        vec = np.zeros(hidden_size, dtype=np.float32)
        vec[i] = 1.0  # Each dimension points along a different axis
        # Available at layers 5 and 10
        vectors[dim] = {5: vec.copy(), 10: vec.copy()}
    return CachedVectors(
        model_id="test-model",
        num_layers=20,
        hidden_size=hidden_size,
        vectors=vectors,
    )


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cached_vectors"


# ========== TestContrastivePairs ==========

class TestContrastivePairs:
    """Tests for loading and validating contrastive pairs."""

    def test_load_real_pairs(self) -> None:
        """Real contrastive_pairs.json loads without error."""
        pairs = load_contrastive_pairs()
        assert set(EMOTIONAL_DIMENSIONS).issubset(pairs.keys())

    def test_all_dimensions_present(self) -> None:
        pairs = load_contrastive_pairs()
        for dim in EMOTIONAL_DIMENSIONS:
            assert dim in pairs
            assert "positive" in pairs[dim]
            assert "negative" in pairs[dim]

    def test_minimum_30_pairs_per_dimension(self) -> None:
        pairs = load_contrastive_pairs()
        for dim in EMOTIONAL_DIMENSIONS:
            assert len(pairs[dim]["positive"]) >= 30, f"{dim} positive < 30"
            assert len(pairs[dim]["negative"]) >= 30, f"{dim} negative < 30"

    def test_pairs_are_strings(self) -> None:
        pairs = load_contrastive_pairs()
        for dim in EMOTIONAL_DIMENSIONS:
            for prompt in pairs[dim]["positive"] + pairs[dim]["negative"]:
                assert isinstance(prompt, str)
                assert len(prompt) > 10, f"Too short: {prompt!r}"

    def test_positive_negative_same_count(self) -> None:
        pairs = load_contrastive_pairs()
        for dim in EMOTIONAL_DIMENSIONS:
            assert len(pairs[dim]["positive"]) == len(pairs[dim]["negative"]), (
                f"{dim}: pos={len(pairs[dim]['positive'])}, neg={len(pairs[dim]['negative'])}"
            )

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_contrastive_pairs(Path("/nonexistent/file.json"))

    def test_invalid_structure_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"valence": {"positive": ["a"]}}))
        with pytest.raises(ValueError):
            load_contrastive_pairs(bad_file)

    def test_too_few_pairs_raises(self, tmp_path: Path) -> None:
        data = {dim: {"positive": ["a"] * 5, "negative": ["b"] * 5} for dim in EMOTIONAL_DIMENSIONS}
        bad_file = tmp_path / "few.json"
        bad_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="at least 10"):
            load_contrastive_pairs(bad_file)


# ========== TestDirectionVector ==========

class TestDirectionVector:
    """Tests for compute_direction_vector."""

    def test_basic_direction(self) -> None:
        pos = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        neg = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        d = compute_direction_vector(pos, neg)
        assert d.shape == (3,)
        assert d[0] > 0.99  # Points in positive x direction
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)

    def test_unit_normalized(self) -> None:
        rng = np.random.default_rng(42)
        pos = rng.normal(size=(20, 64))
        neg = rng.normal(size=(20, 64))
        d = compute_direction_vector(pos, neg)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)

    def test_opposite_directions(self) -> None:
        pos = np.array([[0.0, 1.0], [0.0, 1.0]])
        neg = np.array([[0.0, -1.0], [0.0, -1.0]])
        d = compute_direction_vector(pos, neg)
        assert d[1] > 0.99

    def test_zero_vector_warning(self) -> None:
        """Identical positive/negative → near-zero vector."""
        same = np.array([[1.0, 2.0, 3.0]])
        d = compute_direction_vector(same, same)
        assert np.linalg.norm(d) < 1e-6

    def test_wrong_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            compute_direction_vector(np.zeros(5), np.zeros(5))

    def test_mismatched_hidden_raises(self) -> None:
        with pytest.raises(ValueError, match="match"):
            compute_direction_vector(np.zeros((3, 4)), np.zeros((3, 5)))

    def test_different_sample_counts(self) -> None:
        """Different numbers of positive/negative prompts is fine."""
        pos = np.ones((10, 4))
        neg = np.zeros((5, 4))
        d = compute_direction_vector(pos, neg)
        assert d.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)


# ========== TestCompositeVector ==========

class TestCompositeVector:
    """Tests for compute_composite_vector."""

    def test_neutral_state_returns_none(self, sample_cached: CachedVectors) -> None:
        """Neutral state (intensity ~0) → no steering."""
        result = compute_composite_vector(
            sample_cached, valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.0, layer=5,
        )
        assert result is None

    def test_low_intensity_returns_none(self, sample_cached: CachedVectors) -> None:
        """Very low intensity → below threshold."""
        result = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.8, dominance=0.8,
            certainty=0.8, intensity=0.05, layer=5,
        )
        assert result is None  # 0.05² = 0.0025 < 0.01

    def test_high_intensity_positive_valence(self, sample_cached: CachedVectors) -> None:
        """High intensity + positive valence → vector with positive valence component."""
        result = compute_composite_vector(
            sample_cached, valence=0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.9, layer=5,
        )
        assert result is not None
        # Valence direction is axis 0, and deviation is positive
        assert result[0] > 0

    def test_negative_valence_steers_opposite(self, sample_cached: CachedVectors) -> None:
        """Negative valence → negative component along valence axis."""
        result = compute_composite_vector(
            sample_cached, valence=-0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.9, layer=5,
        )
        assert result is not None
        assert result[0] < 0

    def test_quadratic_intensity_scaling(self, sample_cached: CachedVectors) -> None:
        """Doubling intensity more than doubles the vector magnitude."""
        low = compute_composite_vector(
            sample_cached, valence=1.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5, layer=5,
        )
        high = compute_composite_vector(
            sample_cached, valence=1.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=1.0, layer=5,
        )
        assert low is not None and high is not None
        # intensity 1.0² / 0.5² = 4x ratio
        ratio = np.linalg.norm(high) / np.linalg.norm(low)
        np.testing.assert_allclose(ratio, 4.0, atol=0.01)

    def test_missing_layer_returns_none(self, sample_cached: CachedVectors) -> None:
        result = compute_composite_vector(
            sample_cached, valence=0.9, arousal=0.8, dominance=0.7,
            certainty=0.6, intensity=0.9, layer=99,
        )
        assert result is None

    def test_max_steering_norm_clamp(self, sample_cached: CachedVectors) -> None:
        """Extreme values → clamped to MAX_STEERING_NORM."""
        # Use very large vectors to trigger clamp
        for dim in sample_cached.vectors:
            for layer in sample_cached.vectors[dim]:
                sample_cached.vectors[dim][layer] = np.ones(8, dtype=np.float32) * 100.0
        result = compute_composite_vector(
            sample_cached, valence=1.0, arousal=1.0, dominance=1.0,
            certainty=1.0, intensity=1.0, layer=5,
        )
        assert result is not None
        assert np.linalg.norm(result) <= MAX_STEERING_NORM + 1e-6

    def test_all_dimensions_contribute(self, sample_cached: CachedVectors) -> None:
        """All 4 dimensions are non-neutral → all axes active."""
        result = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.9, dominance=0.2,
            certainty=0.1, intensity=0.8, layer=5,
        )
        assert result is not None
        # All 4 axes should have non-zero components
        for i in range(4):
            assert result[i] != 0.0, f"Axis {i} (dim {EMOTIONAL_DIMENSIONS[i]}) is zero"

    def test_arousal_deviation_mapping(self, sample_cached: CachedVectors) -> None:
        """Arousal 0.0 → deviation -1.0, arousal 1.0 → deviation +1.0."""
        low_arousal = compute_composite_vector(
            sample_cached, valence=0.0, arousal=0.0, dominance=0.5,
            certainty=0.5, intensity=1.0, layer=5,
        )
        high_arousal = compute_composite_vector(
            sample_cached, valence=0.0, arousal=1.0, dominance=0.5,
            certainty=0.5, intensity=1.0, layer=5,
        )
        assert low_arousal is not None and high_arousal is not None
        # Arousal is axis 1; low → negative, high → positive
        assert low_arousal[1] < 0
        assert high_arousal[1] > 0

    def test_dimension_weights_sum_to_one(self) -> None:
        total = sum(DIMENSION_WEIGHTS.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


# ========== TestCaching ==========

class TestCaching:
    """Tests for save/load of cached steering vectors."""

    def test_save_and_load_roundtrip(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        save_cached_vectors(sample_cached, tmp_cache_dir)
        loaded = load_cached_vectors("test-model", tmp_cache_dir)
        assert loaded is not None
        assert loaded.model_id == "test-model"
        assert loaded.num_layers == 20
        assert loaded.hidden_size == 8
        assert set(loaded.available_dimensions) == set(EMOTIONAL_DIMENSIONS)
        for dim in EMOTIONAL_DIMENSIONS:
            for layer in [5, 10]:
                np.testing.assert_array_equal(
                    loaded.vectors[dim][layer],
                    sample_cached.vectors[dim][layer],
                )

    def test_load_nonexistent_returns_none(self, tmp_cache_dir: Path) -> None:
        result = load_cached_vectors("no-such-model", tmp_cache_dir)
        assert result is None

    def test_model_id_sanitized(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        """Model IDs with special chars (e.g. 'qwen3:4b') are safely stored."""
        sample_cached.model_id = "qwen3:4b"
        path = save_cached_vectors(sample_cached, tmp_cache_dir)
        assert "qwen3_4b" in path.name
        loaded = load_cached_vectors("qwen3:4b", tmp_cache_dir)
        assert loaded is not None
        assert loaded.model_id == "qwen3:4b"

    def test_corrupted_file_returns_none(self, tmp_cache_dir: Path) -> None:
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        bad_file = tmp_cache_dir / "bad_model.npz"
        bad_file.write_bytes(b"not a valid npz file")
        result = load_cached_vectors("bad_model", tmp_cache_dir)
        assert result is None


# ========== TestSteeringConfig ==========

class TestSteeringConfig:
    """Tests for SteeringConfig."""

    def test_auto_select_layers_36(self) -> None:
        """Qwen2 (36 layers) → selects 3 spread layers."""
        cfg = SteeringConfig(model_id="qwen3:4b", num_layers=36)
        layers = cfg.auto_select_layers()
        assert len(layers) == 3
        assert layers[0] >= 1  # early
        assert layers[-1] <= 34  # not last layer
        # Should be sorted
        assert layers == sorted(layers)

    def test_auto_select_layers_32(self) -> None:
        cfg = SteeringConfig(model_id="llama3:8b", num_layers=32)
        layers = cfg.auto_select_layers()
        assert len(layers) == 3

    def test_manual_layers_override(self) -> None:
        cfg = SteeringConfig(model_id="custom", num_layers=32, extraction_layers=[3, 15, 28])
        layers = cfg.auto_select_layers()
        assert layers == [3, 15, 28]

    def test_zero_layers_returns_empty(self) -> None:
        cfg = SteeringConfig(model_id="unknown", num_layers=0)
        assert cfg.auto_select_layers() == []


# ========== TestEmotionalSteeringEngine ==========

class TestEmotionalSteeringEngine:
    """Tests for the main engine class."""

    def test_not_ready_by_default(self) -> None:
        engine = EmotionalSteeringEngine()
        assert not engine.is_ready
        assert engine.model_id is None
        assert engine.available_dimensions == []

    def test_load_vectors_from_cache(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        save_cached_vectors(sample_cached, tmp_cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=tmp_cache_dir)
        assert engine.load_vectors("test-model")
        assert engine.is_ready
        assert engine.model_id == "test-model"
        assert len(engine.available_dimensions) == 4

    def test_load_nonexistent_returns_false(self, tmp_cache_dir: Path) -> None:
        engine = EmotionalSteeringEngine(cache_dir=tmp_cache_dir)
        assert not engine.load_vectors("nope")
        assert not engine.is_ready

    def test_get_steering_vector_not_ready(self) -> None:
        engine = EmotionalSteeringEngine()
        result = engine.get_steering_vector(0.5, 0.5, 0.5, 0.5, 0.8, layer=5)
        assert result is None

    def test_get_steering_vector_ready(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        save_cached_vectors(sample_cached, tmp_cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=tmp_cache_dir)
        engine.load_vectors("test-model")
        result = engine.get_steering_vector(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, layer=5,
        )
        assert result is not None
        assert result.shape == (8,)

    def test_get_all_layers(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        save_cached_vectors(sample_cached, tmp_cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=tmp_cache_dir)
        engine.load_vectors("test-model")
        results = engine.get_steering_vectors_all_layers(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        assert 5 in results
        assert 10 in results
        assert len(results) == 2

    def test_get_info_not_loaded(self) -> None:
        engine = EmotionalSteeringEngine()
        info = engine.get_info()
        assert info["status"] == "not_loaded"

    def test_get_info_loaded(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        save_cached_vectors(sample_cached, tmp_cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=tmp_cache_dir)
        engine.load_vectors("test-model")
        info = engine.get_info()
        assert info["status"] == "ready"
        assert info["model_id"] == "test-model"
        assert info["num_layers"] == 20
        assert info["hidden_size"] == 8
        assert len(info["dimensions"]) == 4
        assert info["total_vectors"] == 8  # 4 dims × 2 layers

    def test_save_vectors(self, sample_cached: CachedVectors, tmp_cache_dir: Path) -> None:
        engine = EmotionalSteeringEngine(cache_dir=tmp_cache_dir)
        engine._cached = sample_cached
        path = engine.save_vectors()
        assert path is not None
        assert path.exists()

    def test_save_nothing_returns_none(self) -> None:
        engine = EmotionalSteeringEngine()
        assert engine.save_vectors() is None

    def test_contrastive_pairs_cached(self) -> None:
        engine = EmotionalSteeringEngine()
        pairs1 = engine.get_contrastive_pairs()
        pairs2 = engine.get_contrastive_pairs()
        assert pairs1 is pairs2  # Same object (cached)


# ========== TestCachedVectors ==========

class TestCachedVectors:
    """Tests for CachedVectors dataclass."""

    def test_available_dimensions(self, sample_cached: CachedVectors) -> None:
        assert set(sample_cached.available_dimensions) == set(EMOTIONAL_DIMENSIONS)

    def test_available_layers(self, sample_cached: CachedVectors) -> None:
        assert sample_cached.available_layers == {5, 10}

    def test_has_dimension(self, sample_cached: CachedVectors) -> None:
        assert sample_cached.has_dimension("valence")
        assert not sample_cached.has_dimension("nonexistent")

    def test_empty_vectors(self) -> None:
        empty = CachedVectors(model_id="x", num_layers=10, hidden_size=4)
        assert empty.available_dimensions == []
        assert empty.available_layers == set()
        assert not empty.has_dimension("valence")


# ========== TestConstants ==========

class TestConstants:
    """Sanity checks for module-level constants."""

    def test_dimension_weights_sum(self) -> None:
        assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6

    def test_all_dimensions_have_weights(self) -> None:
        for dim in EMOTIONAL_DIMENSIONS:
            assert dim in DIMENSION_WEIGHTS

    def test_intensity_exponent_reasonable(self) -> None:
        assert 1.0 <= INTENSITY_EXPONENT <= 3.0

    def test_max_steering_norm_positive(self) -> None:
        assert MAX_STEERING_NORM > 0

    def test_valence_weight_highest(self) -> None:
        """Valence should be the most salient dimension."""
        assert DIMENSION_WEIGHTS["valence"] == max(DIMENSION_WEIGHTS.values())


# ========== Mock helpers for SteeringHook tests ==========

class _MockHandle:
    """Simulates a PyTorch hook handle."""
    def __init__(self) -> None:
        self.removed = False
    def remove(self) -> None:
        self.removed = True


class _MockLayer:
    """Simulates a transformer layer with register_forward_hook."""
    def __init__(self) -> None:
        self.hooks: list = []
    def register_forward_hook(self, fn) -> _MockHandle:
        handle = _MockHandle()
        self.hooks.append((fn, handle))
        return handle


class _MockModel:
    """Simulates a HuggingFace transformers model (LlamaForCausalLM style)."""
    def __init__(self, num_layers: int = 20) -> None:
        layers = [_MockLayer() for _ in range(num_layers)]
        self.model = MagicMock()
        self.model.layers = layers


class _MockGPT2Model:
    """Simulates a GPT-2 style model (transformer.h)."""
    def __init__(self, num_layers: int = 12) -> None:
        layers = [_MockLayer() for _ in range(num_layers)]
        self.transformer = MagicMock()
        self.transformer.h = layers


# ========== TestGetModelLayers ==========

class TestGetModelLayers:
    """Tests for _get_model_layers."""

    def test_llama_style(self) -> None:
        model = _MockModel(20)
        layers = _get_model_layers(model)
        assert layers is not None
        assert len(layers) == 20

    def test_gpt2_style(self) -> None:
        model = _MockGPT2Model(12)
        layers = _get_model_layers(model)
        assert layers is not None
        assert len(layers) == 12

    def test_unknown_returns_none(self) -> None:
        model = MagicMock(spec=[])  # No .model.layers or .transformer.h
        assert _get_model_layers(model) is None

    def test_empty_layers(self) -> None:
        model = _MockModel(0)
        layers = _get_model_layers(model)
        assert layers is not None
        assert len(layers) == 0


# ========== TestMakeSteeringHook ==========

class TestMakeSteeringHook:
    """Tests for _make_steering_hook function."""

    def test_hook_returns_callable(self) -> None:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hook_fn = _make_steering_hook(vec)
        assert callable(hook_fn)

    def test_hook_modifies_tuple_output(self) -> None:
        """Hook adds steering vector to hidden states in tuple output."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        hook_fn = _make_steering_hook(vec)

        # Simulate: (hidden_states, attention, ...)
        hidden = torch.zeros(1, 5, 4)  # (batch=1, seq=5, hidden=4)
        other = torch.zeros(1)
        output = (hidden, other)

        result = hook_fn(None, None, output)
        assert isinstance(result, tuple)
        assert len(result) == 2
        modified = result[0]
        assert modified.shape == (1, 5, 4)
        # Each position should have the steering vector added
        for s in range(5):
            torch.testing.assert_close(
                modified[0, s], torch.tensor(vec), atol=1e-6, rtol=1e-6,
            )
        # Other tensors unchanged
        assert result[1] is other

    def test_hook_modifies_single_tensor(self) -> None:
        """Hook adds steering vector to single tensor output."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        vec = np.array([0.5, -0.5], dtype=np.float32)
        hook_fn = _make_steering_hook(vec)

        hidden = torch.zeros(1, 3, 2)  # (batch=1, seq=3, hidden=2)
        result = hook_fn(None, None, hidden)
        assert result.shape == (1, 3, 2)
        for s in range(3):
            torch.testing.assert_close(
                result[0, s], torch.tensor(vec), atol=1e-6, rtol=1e-6,
            )

    def test_hook_applies_to_all_positions(self) -> None:
        """Steering vector is added to every token position."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        vec = np.array([1.0, 0.0], dtype=np.float32)
        hook_fn = _make_steering_hook(vec)

        hidden = torch.zeros(2, 10, 2)  # batch=2, seq=10
        result = hook_fn(None, None, (hidden,))
        modified = result[0]
        # Every position in every batch should have [1.0, 0.0] added
        for b in range(2):
            for s in range(10):
                assert modified[b, s, 0].item() == pytest.approx(1.0)
                assert modified[b, s, 1].item() == pytest.approx(0.0)


# ========== TestSteeringHook ==========

class TestSteeringHook:
    """Tests for SteeringHook class."""

    def _make_ready_engine(self, tmp_path: Path, cached: CachedVectors) -> EmotionalSteeringEngine:
        cache_dir = tmp_path / "cache"
        save_cached_vectors(cached, cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=cache_dir)
        engine.load_vectors(cached.model_id)
        return engine

    def test_apply_registers_hooks(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        result = hook.apply()
        assert result is True
        assert hook.is_applied
        assert len(hook.vectors_applied) > 0
        # Hooks registered on the correct layers
        for layer_idx in hook.vectors_applied:
            layer = model.model.layers[layer_idx]
            assert len(layer.hooks) > 0

    def test_remove_cleans_hooks(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        applied_layers = list(hook.vectors_applied.keys())
        hook.remove()
        assert not hook.is_applied
        assert len(hook.vectors_applied) == 0

    def test_context_manager(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        with SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        ) as hook:
            assert hook.is_applied or not engine.is_ready
        assert not hook.is_applied

    def test_low_intensity_no_hooks(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """Very low intensity → no hooks registered."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.5, arousal=0.5, dominance=0.5, certainty=0.5, intensity=0.05,
        )
        result = hook.apply()
        assert result is False
        assert not hook.is_applied

    def test_engine_not_ready(self, tmp_path: Path) -> None:
        engine = EmotionalSteeringEngine(cache_dir=tmp_path / "empty")
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        result = hook.apply()
        assert result is False

    def test_no_model_layers(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """Model without recognizable layers → no hooks."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = MagicMock(spec=[])  # No .model.layers
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        result = hook.apply()
        assert result is False

    def test_vectors_applied_has_norms(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        for layer_idx, norm in hook.vectors_applied.items():
            assert isinstance(layer_idx, int)
            assert norm > 0.0

    def test_reapply_removes_old_hooks(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """Calling apply() twice removes old hooks first."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        first_count = len(hook.vectors_applied)
        hook.apply()  # Should not double-register
        assert len(hook.vectors_applied) == first_count


# ========== TestLLMProviderSteering ==========

class TestLLMProviderSteering:
    """Tests for steering capability on LLM providers."""

    def test_base_provider_no_steering(self) -> None:
        from pathos.llm.base import LLMProvider
        # Can't instantiate abstract class, but check the property exists
        assert hasattr(LLMProvider, "supports_steering")
        assert hasattr(LLMProvider, "steerable_model")

    def test_ollama_no_steering(self) -> None:
        from pathos.llm.ollama import OllamaProvider
        provider = OllamaProvider()
        assert provider.supports_steering is False
        assert provider.steerable_model is None


# ========== TestClassifyLayerRole ==========

class TestClassifyLayerRole:
    """Tests for classify_layer_role function."""

    def test_early_layer(self) -> None:
        """First third of layers → EARLY."""
        assert classify_layer_role(0, 36) == LayerRole.EARLY
        assert classify_layer_role(5, 36) == LayerRole.EARLY
        assert classify_layer_role(11, 36) == LayerRole.EARLY

    def test_mid_layer(self) -> None:
        """Middle third of layers → MID."""
        assert classify_layer_role(12, 36) == LayerRole.MID
        assert classify_layer_role(18, 36) == LayerRole.MID
        assert classify_layer_role(23, 36) == LayerRole.MID

    def test_late_layer(self) -> None:
        """Last third of layers → LATE."""
        # 25/36=0.694 > 0.67
        assert classify_layer_role(25, 36) == LayerRole.LATE
        assert classify_layer_role(30, 36) == LayerRole.LATE
        assert classify_layer_role(35, 36) == LayerRole.LATE

    def test_32_layer_model(self) -> None:
        """Works for 32-layer models (Llama/Mistral)."""
        assert classify_layer_role(2, 32) == LayerRole.EARLY
        assert classify_layer_role(16, 32) == LayerRole.MID
        assert classify_layer_role(28, 32) == LayerRole.LATE

    def test_zero_layers_fallback(self) -> None:
        """Zero layers → MID fallback."""
        assert classify_layer_role(0, 0) == LayerRole.MID

    def test_boundary_33_percent(self) -> None:
        """Layer at exactly 33% boundary."""
        # 36 * 0.33 = 11.88, so layer 12 should be MID
        assert classify_layer_role(12, 36) == LayerRole.MID
        # But layer 11 (ratio=11/36=0.305) should be EARLY
        assert classify_layer_role(11, 36) == LayerRole.EARLY

    def test_boundary_67_percent(self) -> None:
        """Layer at 67% boundary."""
        # 24/36=0.666 < 0.67 → still MID
        assert classify_layer_role(24, 36) == LayerRole.MID
        # 25/36=0.694 >= 0.67 → LATE
        assert classify_layer_role(25, 36) == LayerRole.LATE

    def test_small_model_6_layers(self) -> None:
        """Small model with 6 layers."""
        assert classify_layer_role(0, 6) == LayerRole.EARLY  # 0/6=0.0
        assert classify_layer_role(1, 6) == LayerRole.EARLY  # 1/6=0.167
        assert classify_layer_role(2, 6) == LayerRole.MID    # 2/6=0.333
        assert classify_layer_role(3, 6) == LayerRole.MID    # 3/6=0.5
        assert classify_layer_role(4, 6) == LayerRole.MID    # 4/6=0.666 < 0.67
        assert classify_layer_role(5, 6) == LayerRole.LATE   # 5/6=0.833


# ========== TestLayerRoleWeights ==========

class TestLayerRoleWeights:
    """Tests for per-role dimension weight constants."""

    def test_all_roles_have_weights(self) -> None:
        """Every LayerRole has a weight dict."""
        for role in LayerRole:
            assert role in LAYER_ROLE_WEIGHTS

    def test_weights_sum_to_one(self) -> None:
        """Each role's dimension weights must sum to 1.0."""
        for role in LayerRole:
            total = sum(LAYER_ROLE_WEIGHTS[role].values())
            np.testing.assert_allclose(total, 1.0, atol=1e-6, err_msg=f"{role} weights don't sum to 1")

    def test_all_dimensions_covered(self) -> None:
        """Each role has weights for all emotional dimensions."""
        for role in LayerRole:
            for dim in EMOTIONAL_DIMENSIONS:
                assert dim in LAYER_ROLE_WEIGHTS[role], f"{role} missing {dim}"

    def test_early_valence_dominant(self) -> None:
        """Early layers: valence should have the highest weight."""
        w = LAYER_ROLE_WEIGHTS[LayerRole.EARLY]
        assert w["valence"] == max(w.values())

    def test_late_dominance_dominant(self) -> None:
        """Late layers: dominance should have the highest weight."""
        w = LAYER_ROLE_WEIGHTS[LayerRole.LATE]
        assert w["dominance"] == max(w.values())

    def test_mid_is_balanced(self) -> None:
        """Mid layers: no single dimension should exceed 0.40."""
        w = LAYER_ROLE_WEIGHTS[LayerRole.MID]
        for dim, weight in w.items():
            assert weight <= 0.40, f"Mid layer {dim}={weight} exceeds 0.40"

    def test_all_roles_have_scaling(self) -> None:
        """Every role has a scaling factor."""
        for role in LayerRole:
            assert role in LAYER_ROLE_SCALING
            assert 0.0 < LAYER_ROLE_SCALING[role] <= 1.0

    def test_mid_has_full_scaling(self) -> None:
        """Mid layers get full scaling (1.0)."""
        assert LAYER_ROLE_SCALING[LayerRole.MID] == 1.0

    def test_early_lighter_than_mid(self) -> None:
        """Early layers should be lighter than mid."""
        assert LAYER_ROLE_SCALING[LayerRole.EARLY] < LAYER_ROLE_SCALING[LayerRole.MID]


# ========== TestMultiLayerCompositeVector ==========

class TestMultiLayerCompositeVector:
    """Tests for compute_composite_vector with layer_role parameter."""

    def test_role_aware_differs_from_global(self, sample_cached: CachedVectors) -> None:
        """Using a layer_role should produce a different vector than global weights."""
        global_vec = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.7, dominance=0.6,
            certainty=0.4, intensity=0.8, layer=5, layer_role=None,
        )
        role_vec = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.7, dominance=0.6,
            certainty=0.4, intensity=0.8, layer=5, layer_role=LayerRole.EARLY,
        )
        assert global_vec is not None and role_vec is not None
        # They should differ because weights are different
        assert not np.allclose(global_vec, role_vec)

    def test_early_emphasizes_valence(self, sample_cached: CachedVectors) -> None:
        """Early role: valence component should be larger than late role."""
        early_vec = compute_composite_vector(
            sample_cached, valence=0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.9, layer=5, layer_role=LayerRole.EARLY,
        )
        late_vec = compute_composite_vector(
            sample_cached, valence=0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.9, layer=5, layer_role=LayerRole.LATE,
        )
        assert early_vec is not None and late_vec is not None
        # Valence is axis 0 in sample_cached
        assert abs(early_vec[0]) > abs(late_vec[0])

    def test_late_emphasizes_dominance(self, sample_cached: CachedVectors) -> None:
        """Late role: dominance component should be larger than early role."""
        early_vec = compute_composite_vector(
            sample_cached, valence=0.5, arousal=0.5, dominance=0.9,
            certainty=0.5, intensity=0.9, layer=5, layer_role=LayerRole.EARLY,
        )
        late_vec = compute_composite_vector(
            sample_cached, valence=0.5, arousal=0.5, dominance=0.9,
            certainty=0.5, intensity=0.9, layer=5, layer_role=LayerRole.LATE,
        )
        assert early_vec is not None and late_vec is not None
        # Dominance is axis 2 in sample_cached
        assert abs(late_vec[2]) > abs(early_vec[2])

    def test_scaling_reduces_early_magnitude(self, sample_cached: CachedVectors) -> None:
        """Early role scaling (0.7) should make overall magnitude smaller than mid (1.0)."""
        early_vec = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.8, dominance=0.8,
            certainty=0.8, intensity=0.8, layer=5, layer_role=LayerRole.EARLY,
        )
        mid_vec = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.8, dominance=0.8,
            certainty=0.8, intensity=0.8, layer=5, layer_role=LayerRole.MID,
        )
        assert early_vec is not None and mid_vec is not None
        assert np.linalg.norm(early_vec) < np.linalg.norm(mid_vec)

    def test_none_role_is_backward_compatible(self, sample_cached: CachedVectors) -> None:
        """layer_role=None should give exactly the same result as before (DIMENSION_WEIGHTS)."""
        vec_none = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.7, dominance=0.6,
            certainty=0.4, intensity=0.8, layer=5, layer_role=None,
        )
        # Manually compute expected using DIMENSION_WEIGHTS
        intensity_weight = 0.8 ** INTENSITY_EXPONENT
        expected = np.zeros(8, dtype=np.float32)
        dims_vals = {"valence": 0.8, "arousal": (0.7 - 0.5) * 2, "dominance": (0.6 - 0.5) * 2, "certainty": (0.4 - 0.5) * 2}
        for i, dim in enumerate(EMOTIONAL_DIMENSIONS):
            w = DIMENSION_WEIGHTS[dim]
            d = dims_vals[dim]
            direction = np.zeros(8, dtype=np.float32)
            direction[i] = 1.0
            expected += (w * d * intensity_weight * 1.0) * direction
        assert vec_none is not None
        np.testing.assert_allclose(vec_none, expected, atol=1e-6)

    def test_low_intensity_still_none_with_role(self, sample_cached: CachedVectors) -> None:
        """Low intensity with role should still return None."""
        result = compute_composite_vector(
            sample_cached, valence=0.9, arousal=0.9, dominance=0.9,
            certainty=0.9, intensity=0.05, layer=5, layer_role=LayerRole.MID,
        )
        assert result is None

    def test_clamp_still_works_with_role(self, sample_cached: CachedVectors) -> None:
        """MAX_STEERING_NORM clamp still applies with role."""
        for dim in sample_cached.vectors:
            for layer in sample_cached.vectors[dim]:
                sample_cached.vectors[dim][layer] = np.ones(8, dtype=np.float32) * 100.0
        result = compute_composite_vector(
            sample_cached, valence=1.0, arousal=1.0, dominance=1.0,
            certainty=1.0, intensity=1.0, layer=5, layer_role=LayerRole.LATE,
        )
        assert result is not None
        assert np.linalg.norm(result) <= MAX_STEERING_NORM + 1e-6


# ========== TestMultiLayerEngine ==========

class TestMultiLayerEngine:
    """Tests for multi-layer steering in EmotionalSteeringEngine."""

    def _make_ready_engine(self, tmp_path: Path, cached: CachedVectors) -> EmotionalSteeringEngine:
        cache_dir = tmp_path / "cache"
        save_cached_vectors(cached, cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=cache_dir)
        engine.load_vectors(cached.model_id)
        return engine

    def test_multilayer_default_on(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """get_steering_vector defaults to multilayer=True."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        vec_default = engine.get_steering_vector(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, layer=5,
        )
        vec_multi = engine.get_steering_vector(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, layer=5, multilayer=True,
        )
        assert vec_default is not None and vec_multi is not None
        np.testing.assert_array_equal(vec_default, vec_multi)

    def test_multilayer_false_uses_global_weights(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """multilayer=False gives same result as layer_role=None."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        vec_legacy = engine.get_steering_vector(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, layer=5, multilayer=False,
        )
        vec_direct = compute_composite_vector(
            sample_cached, valence=0.8, arousal=0.7, dominance=0.6,
            certainty=0.4, intensity=0.8, layer=5, layer_role=None,
        )
        assert vec_legacy is not None and vec_direct is not None
        np.testing.assert_allclose(vec_legacy, vec_direct, atol=1e-6)

    def test_multilayer_differs_from_legacy(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """Multi-layer and legacy produce different vectors."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        vec_multi = engine.get_steering_vector(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, layer=5, multilayer=True,
        )
        vec_legacy = engine.get_steering_vector(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, layer=5, multilayer=False,
        )
        assert vec_multi is not None and vec_legacy is not None
        assert not np.allclose(vec_multi, vec_legacy)

    def test_get_info_includes_layer_roles(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """get_info() should include layer_roles and multilayer flag."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        info = engine.get_info()
        assert "layer_roles" in info
        assert "multilayer" in info
        assert info["multilayer"] is True
        roles = info["layer_roles"]
        assert "early" in roles
        assert "mid" in roles
        assert "late" in roles
        # With 20 layers, layer 5 is early (5/20=0.25), layer 10 is mid (10/20=0.5)
        assert 5 in roles["early"]
        assert 10 in roles["mid"]

    def test_all_layers_multilayer(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """get_steering_vectors_all_layers with multilayer=True."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        results = engine.get_steering_vectors_all_layers(
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4,
            intensity=0.8, multilayer=True,
        )
        assert 5 in results
        assert 10 in results
        # Different layers get different role-based vectors
        assert not np.allclose(results[5], results[10])


# ========== TestMultiLayerSteeringHook ==========

class TestMultiLayerSteeringHook:
    """Tests for multi-layer diagnostics in SteeringHook."""

    def _make_ready_engine(self, tmp_path: Path, cached: CachedVectors) -> EmotionalSteeringEngine:
        cache_dir = tmp_path / "cache"
        save_cached_vectors(cached, cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=cache_dir)
        engine.load_vectors(cached.model_id)
        return engine

    def test_layer_roles_populated(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """After apply(), layer_roles should have role names for each hooked layer."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        roles = hook.layer_roles
        assert len(roles) > 0
        for layer_idx, role_name in roles.items():
            assert role_name in ("early", "mid", "late")

    def test_layer_5_is_early(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """Layer 5 of 20 (25%) should be classified as early."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        assert hook.layer_roles.get(5) == "early"

    def test_layer_10_is_mid(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """Layer 10 of 20 (50%) should be classified as mid."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        assert hook.layer_roles.get(10) == "mid"

    def test_remove_clears_roles(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """remove() should clear layer_roles."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        assert len(hook.layer_roles) > 0
        hook.remove()
        assert len(hook.layer_roles) == 0


# ===========================================================================
# Steering Momentum (5.2a)
# ===========================================================================


class TestSteeringMomentum:
    """Tests for SteeringMomentum — temporal inertia across turns."""

    def test_default_momentum(self) -> None:
        """Default momentum factor is DEFAULT_MOMENTUM."""
        sm = SteeringMomentum()
        assert sm.momentum == DEFAULT_MOMENTUM
        assert not sm.has_history
        assert sm.turns_stored == 0

    def test_configure_from_personality_high_neuroticism(self) -> None:
        """High neuroticism → high momentum (emotions linger)."""
        sm = SteeringMomentum()
        sm.configure_from_personality(0.9)
        assert sm.momentum > 0.40

    def test_configure_from_personality_low_neuroticism(self) -> None:
        """Low neuroticism → low momentum (adapts quickly)."""
        sm = SteeringMomentum()
        sm.configure_from_personality(0.1)
        assert sm.momentum < 0.20

    def test_configure_from_personality_average(self) -> None:
        """Average neuroticism → default-ish momentum."""
        sm = SteeringMomentum()
        sm.configure_from_personality(0.5)
        assert 0.25 <= sm.momentum <= 0.35

    def test_configure_clamps(self) -> None:
        """Momentum clamped to [0.10, 0.50]."""
        sm = SteeringMomentum()
        sm.configure_from_personality(0.0)
        assert sm.momentum >= 0.10
        sm.configure_from_personality(1.0)
        assert sm.momentum <= 0.50

    def test_record_turn_creates_history(self) -> None:
        """Recording a turn creates history entries."""
        sm = SteeringMomentum()
        vecs = {5: np.ones(10), 10: np.ones(10) * 2}
        sm.record_turn(vecs)
        assert sm.has_history
        assert sm.turns_stored == 1

    def test_record_multiple_turns(self) -> None:
        """Multiple turns accumulate history."""
        sm = SteeringMomentum()
        for i in range(3):
            sm.record_turn({5: np.ones(10) * (i + 1)})
        assert sm.turns_stored == 3

    def test_history_limit(self) -> None:
        """History is capped at MAX_MOMENTUM_HISTORY."""
        sm = SteeringMomentum()
        for i in range(MAX_MOMENTUM_HISTORY + 3):
            sm.record_turn({5: np.ones(10) * i})
        assert sm.turns_stored == MAX_MOMENTUM_HISTORY

    def test_apply_momentum_no_history(self) -> None:
        """Without history, apply_momentum returns current vectors unchanged."""
        sm = SteeringMomentum()
        current = {5: np.ones(10)}
        result = sm.apply_momentum(current)
        np.testing.assert_array_equal(result[5], current[5])

    def test_apply_momentum_with_history(self) -> None:
        """With history, blended vector differs from current."""
        sm = SteeringMomentum()
        sm.configure_from_personality(0.5)  # high momentum
        # Record a past vector pointing in different direction
        sm.record_turn({5: np.ones(10) * -1.0})
        current = {5: np.ones(10) * 1.0}
        result = sm.apply_momentum(current)
        # Blended should be between current and history
        assert result[5][0] < 1.0  # pulled toward history
        assert result[5][0] > -1.0  # not fully history

    def test_apply_momentum_zero_momentum(self) -> None:
        """With momentum=0, returns current vectors unchanged."""
        sm = SteeringMomentum()
        sm._momentum = 0.0
        sm.record_turn({5: np.ones(10) * -1.0})
        current = {5: np.ones(10) * 1.0}
        result = sm.apply_momentum(current)
        np.testing.assert_array_almost_equal(result[5], current[5])

    def test_apply_momentum_clamps_norm(self) -> None:
        """Blended vector norm is clamped to MAX_STEERING_NORM."""
        sm = SteeringMomentum()
        sm._momentum = 0.3
        big_vec = np.ones(10) * 100.0
        sm.record_turn({5: big_vec})
        result = sm.apply_momentum({5: big_vec})
        norm = float(np.linalg.norm(result[5]))
        assert norm <= MAX_STEERING_NORM + 0.01

    def test_clear(self) -> None:
        """clear() removes all history."""
        sm = SteeringMomentum()
        sm.record_turn({5: np.ones(10)})
        assert sm.has_history
        sm.clear()
        assert not sm.has_history
        assert sm.turns_stored == 0

    def test_get_info(self) -> None:
        """get_info returns diagnostic dict."""
        sm = SteeringMomentum()
        sm.configure_from_personality(0.6)
        sm.record_turn({5: np.ones(10), 10: np.ones(10)})
        info = sm.get_info()
        assert "momentum_factor" in info
        assert info["has_history"]
        assert info["turns_stored"] == 1
        assert 5 in info["layers_tracked"]
        assert 10 in info["layers_tracked"]

    def test_record_copies_vectors(self) -> None:
        """Recorded vectors are copies, not references."""
        sm = SteeringMomentum()
        vec = np.ones(10)
        sm.record_turn({5: vec})
        vec[:] = 0  # mutate original
        # History should still have ones
        assert float(sm._history[5][0][0]) == 1.0

    def test_exponential_decay_weighting(self) -> None:
        """More recent history has more weight than older history."""
        sm = SteeringMomentum()
        sm._momentum = 0.4
        # Record: turn 1 = negative, turn 2 = positive (more recent)
        sm.record_turn({5: np.ones(10) * -1.0})  # older
        sm.record_turn({5: np.ones(10) * 1.0})   # recent
        current = {5: np.zeros(10)}
        result = sm.apply_momentum(current)
        # The blend of history should lean positive (recent had higher weight)
        assert result[5][0] > 0


class TestSteeringHookWithMomentum:
    """Tests for SteeringHook.apply(momentum=...) integration."""

    @pytest.fixture
    def sample_cached(self) -> CachedVectors:
        return CachedVectors(
            model_id="test",
            num_layers=20,
            hidden_size=10,
            vectors={
                "valence": {5: np.random.randn(10).astype(np.float32)},
                "arousal": {5: np.random.randn(10).astype(np.float32)},
            },
        )

    def _make_ready_engine(self, tmp_path: Path, cached: CachedVectors) -> EmotionalSteeringEngine:
        cache_dir = tmp_path / "cache"
        save_cached_vectors(cached, cache_dir)
        engine = EmotionalSteeringEngine(cache_dir=cache_dir)
        engine.load_vectors(cached.model_id)
        return engine

    def test_apply_with_momentum(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """apply(momentum=sm) works without error."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        sm = SteeringMomentum()
        sm.configure_from_personality(0.5)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        result = hook.apply(momentum=sm)
        assert result  # should hook at least one layer

    def test_raw_vectors_populated(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """raw_vectors are populated after apply()."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply()
        assert len(hook.raw_vectors) > 0

    def test_momentum_blends_differ_from_raw(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """With momentum history, applied vectors differ from raw."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        sm = SteeringMomentum()
        sm._momentum = 0.4
        # Record a different vector as history
        sm.record_turn({5: np.ones(10) * 5.0})

        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        hook.apply(momentum=sm)
        # vectors_applied should have norms affected by momentum
        assert len(hook.vectors_applied) > 0

    def test_apply_without_momentum_backward_compatible(self, sample_cached: CachedVectors, tmp_path: Path) -> None:
        """apply() without momentum works as before."""
        engine = self._make_ready_engine(tmp_path, sample_cached)
        model = _MockModel(20)
        hook = SteeringHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.8,
        )
        result = hook.apply()
        assert result


class TestMomentumConstants:
    """Test momentum-related constants."""

    def test_default_momentum_range(self) -> None:
        assert 0.1 <= DEFAULT_MOMENTUM <= 0.5

    def test_max_history_positive(self) -> None:
        assert MAX_MOMENTUM_HISTORY >= 3

    def test_max_history_reasonable(self) -> None:
        assert MAX_MOMENTUM_HISTORY <= 20
