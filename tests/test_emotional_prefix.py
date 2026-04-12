"""Tests for Soft Emotional Prefix (5.3a ARK Rework).

Injects synthetic emotional embeddings at the embedding layer.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock
from pathlib import Path

from pathos.engine.emotional_prefix import (
    DEFAULT_NUM_PREFIX_TOKENS,
    MAX_PREFIX_TOKENS,
    PREFIX_SCALE,
    EmotionalPrefixHook,
    PrefixResult,
    compose_prefix_embedding,
    _get_embedding_layer,
    _make_prefix_hook,
)
from pathos.engine.steering import (
    CachedVectors,
    EmotionalSteeringEngine,
    save_cached_vectors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cached(hidden_size: int = 16) -> CachedVectors:
    """Create sample cached vectors for testing."""
    return CachedVectors(
        model_id="test",
        num_layers=20,
        hidden_size=hidden_size,
        vectors={
            "valence": {2: np.random.randn(hidden_size).astype(np.float32)},
            "arousal": {2: np.random.randn(hidden_size).astype(np.float32)},
            "dominance": {2: np.random.randn(hidden_size).astype(np.float32)},
            "certainty": {2: np.random.randn(hidden_size).astype(np.float32)},
        },
    )


def _make_engine(tmp_path: Path, cached: CachedVectors) -> EmotionalSteeringEngine:
    cache_dir = tmp_path / "cache"
    save_cached_vectors(cached, cache_dir)
    engine = EmotionalSteeringEngine(cache_dir=cache_dir)
    engine.load_vectors(cached.model_id)
    return engine


class _MockEmbeddingLayer:
    """Mock embedding layer with register_forward_hook."""

    def __init__(self) -> None:
        self._hooks: list = []

    def register_forward_hook(self, fn):
        handle = MagicMock()
        handle.remove = MagicMock()
        self._hooks.append((fn, handle))
        return handle


class _MockModel:
    """Mock model with embedding layer."""

    def __init__(self, hidden_size: int = 16) -> None:
        self._emb = _MockEmbeddingLayer()
        self._hidden_size = hidden_size

    def get_input_embeddings(self):
        return self._emb


# ===========================================================================
# TestPrefixResult
# ===========================================================================

class TestPrefixResult:
    def test_defaults(self) -> None:
        r = PrefixResult()
        assert not r.active
        assert r.num_tokens == 0
        assert r.dominant_dimension == "neutral"

    def test_active(self) -> None:
        r = PrefixResult(active=True, num_tokens=2, embedding_norm=1.5, dominant_dimension="valence")
        assert r.active
        assert r.num_tokens == 2


# ===========================================================================
# TestComposePrefixEmbedding
# ===========================================================================

class TestComposePrefixEmbedding:
    def test_returns_vector(self) -> None:
        """Returns embedding vector of correct shape."""
        c = _cached(16)
        result = compose_prefix_embedding(c, valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.6)
        assert result is not None
        assert result.shape == (16,)

    def test_low_intensity_returns_none(self) -> None:
        """Very low intensity → no prefix needed."""
        c = _cached()
        result = compose_prefix_embedding(c, valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.01)
        assert result is None

    def test_no_layers_returns_none(self) -> None:
        """No available layers → None."""
        c = CachedVectors(model_id="test", num_layers=20, hidden_size=16, vectors={})
        result = compose_prefix_embedding(c, valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.6)
        assert result is None

    def test_positive_valence_differs_from_negative(self) -> None:
        """Positive and negative valence produce different embeddings."""
        c = _cached()
        pos = compose_prefix_embedding(c, valence=0.8, arousal=0.5, dominance=0.5, certainty=0.5, intensity=0.6)
        neg = compose_prefix_embedding(c, valence=-0.8, arousal=0.5, dominance=0.5, certainty=0.5, intensity=0.6)
        assert pos is not None and neg is not None
        assert not np.allclose(pos, neg)

    def test_scale_affects_magnitude(self) -> None:
        """Higher scale → larger embedding norm."""
        c = _cached()
        low = compose_prefix_embedding(c, valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.6, scale=0.3)
        high = compose_prefix_embedding(c, valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.6, scale=0.8)
        assert low is not None and high is not None
        assert np.linalg.norm(high) >= np.linalg.norm(low)

    def test_uses_earliest_layer(self) -> None:
        """Uses earliest available layer for embedding composition."""
        c = CachedVectors(
            model_id="test", num_layers=20, hidden_size=8,
            vectors={
                "valence": {
                    2: np.ones(8, dtype=np.float32),
                    15: np.ones(8, dtype=np.float32) * 5.0,
                },
            },
        )
        result = compose_prefix_embedding(c, valence=0.8, arousal=0.5, dominance=0.5, certainty=0.5, intensity=0.6)
        assert result is not None
        # Should use layer 2 (earliest), not layer 15

    def test_norm_clamped(self) -> None:
        """Output norm is clamped."""
        c = CachedVectors(
            model_id="test", num_layers=20, hidden_size=8,
            vectors={"valence": {0: np.ones(8, dtype=np.float32) * 100.0}},
        )
        result = compose_prefix_embedding(c, valence=1.0, arousal=0.5, dominance=0.5, certainty=0.5, intensity=1.0, scale=1.0)
        assert result is not None
        # norm should be clamped


# ===========================================================================
# TestGetEmbeddingLayer
# ===========================================================================

class TestGetEmbeddingLayer:
    def test_hf_api(self) -> None:
        """Standard HF get_input_embeddings() works."""
        model = _MockModel()
        emb = _get_embedding_layer(model)
        assert emb is not None

    def test_direct_embed_tokens(self) -> None:
        """Direct model.model.embed_tokens access."""
        model = MagicMock()
        model.get_input_embeddings.return_value = None
        model.model.embed_tokens = "embed_layer"
        assert _get_embedding_layer(model) == "embed_layer"

    def test_gpt2_style(self) -> None:
        """GPT-2 style model.transformer.wte."""
        model = MagicMock()
        model.get_input_embeddings.return_value = None
        del model.model
        model.transformer.wte = "wte_layer"
        assert _get_embedding_layer(model) == "wte_layer"

    def test_no_embedding_layer(self) -> None:
        """No known embedding layer → None."""
        model = MagicMock()
        model.get_input_embeddings.return_value = None
        del model.model
        del model.transformer
        assert _get_embedding_layer(model) is None


# ===========================================================================
# TestMakePrefixHook
# ===========================================================================

class TestMakePrefixHook:
    def test_returns_callable(self) -> None:
        hook = _make_prefix_hook(np.ones(8, dtype=np.float32), 2)
        assert callable(hook)

    def test_prepends_tokens(self) -> None:
        """Hook prepends N tokens to sequence."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        hook = _make_prefix_hook(np.ones(8, dtype=np.float32), 2)
        # Input: (batch=1, seq_len=5, hidden=8)
        output = torch.ones(1, 5, 8)
        result = hook(None, None, output)
        assert result.shape == (1, 7, 8)  # 2 prefix + 5 original

    def test_symmetry_breaking(self) -> None:
        """Each prefix token has slightly different scale."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        hook = _make_prefix_hook(np.ones(8, dtype=np.float32), 3)
        output = torch.zeros(1, 1, 8)
        result = hook(None, None, output)
        # Prefix tokens should differ from each other
        t0 = result[0, 0, 0].item()
        t1 = result[0, 1, 0].item()
        t2 = result[0, 2, 0].item()
        assert t0 != t1  # different scale
        assert t1 != t2

    def test_batch_support(self) -> None:
        """Hook supports batch size > 1."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        hook = _make_prefix_hook(np.ones(8, dtype=np.float32), 1)
        output = torch.ones(3, 4, 8)  # batch=3
        result = hook(None, None, output)
        assert result.shape == (3, 5, 8)  # 1 prefix + 4 original


# ===========================================================================
# TestEmotionalPrefixHook
# ===========================================================================

class TestEmotionalPrefixHook:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> EmotionalSteeringEngine:
        return _make_engine(tmp_path, _cached())

    def test_apply_success(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
        )
        assert hook.apply()
        assert hook.is_applied
        assert hook.result.active
        assert hook.result.num_tokens == DEFAULT_NUM_PREFIX_TOKENS

    def test_apply_engine_not_ready(self) -> None:
        engine = EmotionalSteeringEngine()
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
        )
        assert not hook.apply()

    def test_remove(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
        )
        hook.apply()
        assert hook.is_applied
        hook.remove()
        assert not hook.is_applied

    def test_context_manager(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        with EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
        ) as hook:
            assert hook.is_applied
        assert not hook.is_applied

    def test_custom_num_tokens(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
            num_tokens=4,
        )
        hook.apply()
        assert hook.result.num_tokens == 4

    def test_num_tokens_capped(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
            num_tokens=10,  # exceeds MAX
        )
        hook.apply()
        assert hook.result.num_tokens <= MAX_PREFIX_TOKENS

    def test_low_intensity_no_apply(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.01,
        )
        assert not hook.apply()

    def test_reapply_cleans_previous(self, engine: EmotionalSteeringEngine) -> None:
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.8, arousal=0.7, dominance=0.6, certainty=0.4, intensity=0.7,
        )
        hook.apply()
        hook.apply()  # second apply should clean first
        assert hook.is_applied

    def test_dominant_dimension(self, engine: EmotionalSteeringEngine) -> None:
        """Result includes dominant emotional dimension."""
        model = _MockModel()
        hook = EmotionalPrefixHook(
            model=model, engine=engine,
            valence=0.9, arousal=0.3, dominance=0.5, certainty=0.5, intensity=0.7,
        )
        hook.apply()
        assert hook.result.dominant_dimension == "valence"


# ===========================================================================
# TestConstants
# ===========================================================================

class TestConstants:
    def test_default_num_tokens(self) -> None:
        assert 1 <= DEFAULT_NUM_PREFIX_TOKENS <= 4

    def test_max_prefix_tokens(self) -> None:
        assert MAX_PREFIX_TOKENS >= DEFAULT_NUM_PREFIX_TOKENS

    def test_prefix_scale_range(self) -> None:
        assert 0.1 <= PREFIX_SCALE <= 1.0
