"""Tests for Emotional Attention Modulation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathos.engine.emotional_attention import (
    ATTENTION_CATEGORIES,
    AttentionBiasResult,
    AttentionHook,
    INTENSITY_GATE,
    MAX_ATTENTION_BIAS,
    MIN_ATTENTION_BIAS,
    NARROWING_SCALE_HIGH,
    NARROWING_SCALE_LOW,
    _build_position_biases,
    _get_attention_layers,
    _make_attention_hook,
    build_token_set,
    compute_attention_bias,
    load_attention_vocabulary,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_vocab() -> dict[str, list[str]]:
    """Minimal attention vocabulary for testing."""
    return {
        "threat": ["danger", "kill", "attack"],
        "agent": ["you", "blame", "responsible"],
        "positive": ["love", "happy", "joy", "beautiful"],
        "negative": ["hate", "sad", "ugly", "terrible"],
        "loss": ["lost", "gone", "miss", "never"],
        "novelty": ["new", "surprise", "unexpected"],
        "uncertainty_markers": ["maybe", "perhaps", "possibly"],
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns predictable token IDs."""
    tokenizer = MagicMock()
    token_map = {
        "danger": [100], "kill": [101], "attack": [102],
        "you": [200], "blame": [201], "responsible": [202],
        "love": [300], "happy": [301], "joy": [302], "beautiful": [303],
        "hate": [400], "sad": [401], "ugly": [402], "terrible": [403],
        "lost": [500], "gone": [501], "miss": [502], "never": [503],
        "new": [600], "surprise": [601], "unexpected": [602],
        "maybe": [700], "perhaps": [701], "possibly": [702],
    }
    def encode_fn(word, add_special_tokens=False):
        return token_map.get(word, [999])
    tokenizer.encode = encode_fn
    return tokenizer


# ========== TestLoadAttentionVocabulary ==========

class TestLoadAttentionVocabulary:
    """Tests for load_attention_vocabulary()."""

    def test_load_real_file(self):
        """Real attention_vocabulary.json loads without errors."""
        vocab = load_attention_vocabulary()
        assert isinstance(vocab, dict)
        assert len(vocab) >= len(ATTENTION_CATEGORIES)

    def test_all_categories_present(self):
        """Every required category exists."""
        vocab = load_attention_vocabulary()
        for cat in ATTENTION_CATEGORIES:
            assert cat in vocab, f"Missing category: {cat}"
            assert len(vocab[cat]) >= 3

    def test_all_words_are_strings(self):
        """Every word in every category is a string."""
        vocab = load_attention_vocabulary()
        for cat, words in vocab.items():
            for w in words:
                assert isinstance(w, str), f"Non-string in {cat}: {w}"

    def test_missing_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_attention_vocabulary(Path("/nonexistent/vocab.json"))

    def test_invalid_structure_raises(self):
        """Invalid JSON structure raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"threat": ["a", "b"]}, f)  # too few words + missing categories
            f.flush()
            with pytest.raises(ValueError):
                load_attention_vocabulary(Path(f.name))


# ========== TestAttentionBiasResult ==========

class TestAttentionBiasResult:
    """Tests for AttentionBiasResult dataclass."""

    def test_empty(self):
        r = AttentionBiasResult(token_biases={})
        assert r.token_biases == {}
        assert r.categories_active == {}
        assert r.broadening_factor == 1.0

    def test_with_data(self):
        r = AttentionBiasResult(
            token_biases={"danger": 0.5},
            categories_active={"threat": 0.8},
            broadening_factor=1.2,
        )
        assert r.token_biases["danger"] == 0.5
        assert r.broadening_factor == 1.2


# ========== TestComputeAttentionBias ==========

class TestComputeAttentionBias:
    """Tests for compute_attention_bias()."""

    def test_low_intensity_empty(self, sample_vocab):
        """Below intensity gate → no biases."""
        result = compute_attention_bias(
            valence=-0.8, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.05, vocabulary=sample_vocab,
        )
        assert result.token_biases == {}
        assert result.categories_active == {}

    def test_fear_pattern_threat_active(self, sample_vocab):
        """Negative valence + high arousal → threat tokens biased."""
        result = compute_attention_bias(
            valence=-0.7, arousal=0.8, dominance=0.3,
            certainty=0.5, intensity=0.8, vocabulary=sample_vocab,
        )
        assert "threat" in result.categories_active
        assert result.categories_active["threat"] > 0
        # Threat words should have positive bias (attend more)
        for word in ["danger", "kill", "attack"]:
            assert word in result.token_biases
            assert result.token_biases[word] > 0

    def test_anger_pattern_agent_active(self, sample_vocab):
        """Negative valence + high dominance → agent tokens biased."""
        result = compute_attention_bias(
            valence=-0.6, arousal=0.5, dominance=0.8,
            certainty=0.5, intensity=0.7, vocabulary=sample_vocab,
        )
        assert "agent" in result.categories_active
        for word in ["you", "blame", "responsible"]:
            assert word in result.token_biases
            assert result.token_biases[word] > 0

    def test_joy_pattern_positive_active(self, sample_vocab):
        """Positive valence → positive tokens boosted, negative slightly suppressed."""
        result = compute_attention_bias(
            valence=0.8, arousal=0.6, dominance=0.5,
            certainty=0.5, intensity=0.7, vocabulary=sample_vocab,
        )
        assert "positive" in result.categories_active
        assert result.categories_active["positive"] > 0
        for word in ["love", "happy", "joy"]:
            assert word in result.token_biases
            assert result.token_biases[word] > 0
        # Negative should be suppressed
        assert "negative" in result.categories_active
        assert result.categories_active["negative"] < 0

    def test_sadness_pattern_loss_active(self, sample_vocab):
        """Negative valence + low arousal → loss tokens biased."""
        result = compute_attention_bias(
            valence=-0.6, arousal=0.2, dominance=0.3,
            certainty=0.5, intensity=0.7, vocabulary=sample_vocab,
        )
        assert "loss" in result.categories_active
        for word in ["lost", "gone", "miss"]:
            assert word in result.token_biases
            assert result.token_biases[word] > 0

    def test_uncertainty_pattern(self, sample_vocab):
        """Low certainty → uncertainty markers biased."""
        result = compute_attention_bias(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.2, intensity=0.6, vocabulary=sample_vocab,
        )
        assert "uncertainty_markers" in result.categories_active
        for word in ["maybe", "perhaps", "possibly"]:
            assert word in result.token_biases
            assert result.token_biases[word] > 0

    def test_novelty_pattern(self, sample_vocab):
        """Low certainty + high arousal → novelty tokens biased."""
        result = compute_attention_bias(
            valence=0.0, arousal=0.7, dominance=0.5,
            certainty=0.2, intensity=0.7, vocabulary=sample_vocab,
        )
        assert "novelty" in result.categories_active
        for word in ["new", "surprise", "unexpected"]:
            assert word in result.token_biases
            assert result.token_biases[word] > 0

    def test_neutral_empty(self, sample_vocab):
        """Neutral emotional state → no active categories."""
        result = compute_attention_bias(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.6, vocabulary=sample_vocab,
        )
        # Neutral on all dimensions → no category thresholds exceeded
        assert len(result.categories_active) == 0
        assert len(result.token_biases) == 0

    def test_intensity_scaling(self, sample_vocab):
        """Higher intensity → stronger biases."""
        low = compute_attention_bias(
            valence=0.8, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.3, vocabulary=sample_vocab,
        )
        high = compute_attention_bias(
            valence=0.8, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.9, vocabulary=sample_vocab,
        )
        # Both should have positive tokens, but high intensity → stronger
        if "love" in low.token_biases and "love" in high.token_biases:
            assert abs(high.token_biases["love"]) >= abs(low.token_biases["love"])

    def test_categories_active_populated(self, sample_vocab):
        """Active categories include numeric activation values."""
        result = compute_attention_bias(
            valence=-0.8, arousal=0.9, dominance=0.7,
            certainty=0.3, intensity=0.9, vocabulary=sample_vocab,
        )
        assert len(result.categories_active) > 0
        for cat, val in result.categories_active.items():
            assert isinstance(val, float)


# ========== TestBroadeningFactor ==========

class TestBroadeningFactor:
    """Tests for arousal-based broadening/narrowing."""

    def test_high_arousal_narrowing(self, sample_vocab):
        """High arousal → broadening_factor > 1 (narrowing = stronger biases)."""
        result = compute_attention_bias(
            valence=0.5, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.7, vocabulary=sample_vocab,
        )
        assert result.broadening_factor > 1.0

    def test_low_arousal_broadening(self, sample_vocab):
        """Low arousal → broadening_factor < 1 (broadening = weaker biases)."""
        result = compute_attention_bias(
            valence=0.5, arousal=0.1, dominance=0.5,
            certainty=0.5, intensity=0.7, vocabulary=sample_vocab,
        )
        assert result.broadening_factor < 1.0

    def test_neutral_arousal_factor_one(self, sample_vocab):
        """Neutral arousal (0.5) → broadening_factor ≈ 1.0."""
        result = compute_attention_bias(
            valence=0.5, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.7, vocabulary=sample_vocab,
        )
        assert abs(result.broadening_factor - 1.0) < 0.01

    def test_narrowing_amplifies_biases(self, sample_vocab):
        """High arousal narrowing should produce stronger biases than low arousal."""
        narrow = compute_attention_bias(
            valence=0.8, arousal=0.95, dominance=0.5,
            certainty=0.5, intensity=0.8, vocabulary=sample_vocab,
        )
        broad = compute_attention_bias(
            valence=0.8, arousal=0.1, dominance=0.5,
            certainty=0.5, intensity=0.8, vocabulary=sample_vocab,
        )
        # Same valence but different arousal → narrowing should have stronger biases
        if "love" in narrow.token_biases and "love" in broad.token_biases:
            assert abs(narrow.token_biases["love"]) > abs(broad.token_biases["love"])


# ========== TestBiasRanges ==========

class TestBiasRanges:
    """All biases stay within safe ranges under extreme inputs."""

    @pytest.mark.parametrize("valence,arousal,dominance,certainty,intensity", [
        (-1.0, 1.0, 1.0, 0.0, 1.0),  # extreme fear+anger
        (1.0, 1.0, 1.0, 1.0, 1.0),   # extreme positive
        (-1.0, 0.0, 0.0, 0.0, 1.0),  # extreme sadness
        (0.0, 0.5, 0.5, 0.5, 1.0),   # neutral dimensions, full intensity
        (-1.0, 1.0, 0.0, 0.0, 1.0),  # extreme panic
    ])
    def test_all_biases_in_range(self, valence, arousal, dominance, certainty, intensity, sample_vocab):
        result = compute_attention_bias(
            valence=valence, arousal=arousal, dominance=dominance,
            certainty=certainty, intensity=intensity, vocabulary=sample_vocab,
        )
        for word, bias in result.token_biases.items():
            assert MIN_ATTENTION_BIAS <= bias <= MAX_ATTENTION_BIAS, \
                f"Bias for '{word}' out of range: {bias}"


# ========== TestBuildTokenSet ==========

class TestBuildTokenSet:
    """Tests for build_token_set() — word→token ID conversion."""

    def test_empty_biases(self, mock_tokenizer):
        assert build_token_set({}, mock_tokenizer) == {}

    def test_single_token(self, mock_tokenizer):
        result = build_token_set({"danger": 0.5}, mock_tokenizer)
        assert 100 in result
        assert result[100] == 0.5

    def test_multiple_words(self, mock_tokenizer):
        result = build_token_set(
            {"danger": 0.5, "love": 0.3}, mock_tokenizer,
        )
        assert 100 in result  # danger
        assert 300 in result  # love

    def test_clamped(self, mock_tokenizer):
        result = build_token_set({"danger": 5.0}, mock_tokenizer)
        assert result[100] <= MAX_ATTENTION_BIAS

    def test_near_zero_filtered(self, mock_tokenizer):
        result = build_token_set({"danger": 0.01}, mock_tokenizer)
        assert len(result) == 0

    def test_tokenizer_failure_skipped(self):
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(side_effect=RuntimeError("fail"))
        result = build_token_set({"danger": 0.5}, tokenizer)
        assert result == {}


# ========== TestBuildPositionBiases ==========

class TestBuildPositionBiases:
    """Tests for _build_position_biases()."""

    def test_flat_list(self):
        result = _build_position_biases([100, 200, 300], {100: 0.5, 300: 0.3})
        assert result == {0: 0.5, 2: 0.3}

    def test_2d_list(self):
        """2D input → uses last sequence."""
        result = _build_position_biases(
            [[999, 999], [100, 200, 300]],
            {100: 0.5, 200: 0.2},
        )
        assert 0 in result  # position of 100
        assert 1 in result  # position of 200

    def test_no_matches(self):
        result = _build_position_biases([1, 2, 3], {100: 0.5})
        assert result == {}

    def test_numpy_array(self):
        ids = np.array([100, 200, 300])
        result = _build_position_biases(ids, {200: 0.4})
        assert result == {1: 0.4}

    def test_empty(self):
        result = _build_position_biases([], {100: 0.5})
        assert result == {}


# ========== TestGetAttentionLayers ==========

class TestGetAttentionLayers:
    """Tests for _get_attention_layers()."""

    def test_llama_style(self):
        model = MagicMock()
        layers = [MagicMock() for _ in range(12)]
        model.model.layers = layers
        result = _get_attention_layers(model)
        assert result is not None
        assert len(result) == 12

    def test_gpt2_style(self):
        model = MagicMock()
        del model.model  # Remove default mock attribute
        blocks = [MagicMock() for _ in range(6)]
        model.transformer.h = blocks
        result = _get_attention_layers(model)
        assert result is not None
        assert len(result) == 6

    def test_unknown_architecture(self):
        model = MagicMock()
        del model.model
        del model.transformer
        result = _get_attention_layers(model)
        assert result is None


# ========== TestMakeAttentionHook ==========

class TestMakeAttentionHook:
    """Tests for _make_attention_hook()."""

    def test_callable(self):
        hook = _make_attention_hook({0: 0.5, 2: -0.3})
        assert callable(hook)

    def test_hook_modifies_output(self):
        """Hook should modify output hidden states at biased positions."""
        import torch

        hook = _make_attention_hook({1: 1.0})  # Amplify position 1
        hidden = torch.ones(1, 3, 4)  # (batch=1, seq_len=3, hidden=4)
        output = (hidden,)

        result = hook(None, None, output)
        modified = result[0]

        # Position 0 and 2 should be unchanged
        assert torch.allclose(modified[0, 0, :], hidden[0, 0, :])
        assert torch.allclose(modified[0, 2, :], hidden[0, 2, :])
        # Position 1 should be scaled up
        assert modified[0, 1, 0].item() > hidden[0, 1, 0].item()

    def test_hook_single_tensor(self):
        """Hook works with non-tuple output."""
        import torch

        hook = _make_attention_hook({0: 1.0})
        hidden = torch.ones(1, 2, 4)
        result = hook(None, None, hidden)
        # Position 0 should be scaled
        assert result[0, 0, 0].item() > 1.0

    def test_no_bias_no_change(self):
        """Empty position biases → no modification."""
        import torch

        hook = _make_attention_hook({})
        hidden = torch.ones(1, 3, 4)
        output = (hidden,)
        result = hook(None, None, output)
        assert torch.allclose(result[0], hidden)

    def test_scale_clamped(self):
        """Extreme biases don't produce extreme scales."""
        import torch

        hook = _make_attention_hook({0: 100.0})  # extreme
        hidden = torch.ones(1, 2, 4)
        output = (hidden,)
        result = hook(None, None, output)
        # Scale should be clamped to 1.30 max
        assert result[0][0, 0, 0].item() <= 1.31  # small float tolerance


# ========== TestAttentionHook ==========

class TestAttentionHook:
    """Tests for the AttentionHook class."""

    def _make_model(self, num_layers: int = 12):
        """Create a mock model with registerable layers."""
        model = MagicMock()
        layers = []
        for _ in range(num_layers):
            layer = MagicMock()
            handle = MagicMock()
            layer.register_forward_hook = MagicMock(return_value=handle)
            layers.append(layer)
        model.model.layers = layers
        return model

    def test_apply_registers_hooks(self):
        model = self._make_model(12)
        token_biases = {100: 0.5, 200: 0.3}
        input_ids = [50, 100, 150, 200, 250]

        hook = AttentionHook(model, token_biases, input_ids)
        result = hook.apply()

        assert result is True
        assert hook.is_applied
        assert hook.positions_biased == 2  # positions 1 and 3
        assert len(hook.layers_hooked) > 0

    def test_remove_cleans_up(self):
        model = self._make_model(12)
        hook = AttentionHook(model, {100: 0.5}, [100, 200])
        hook.apply()
        assert hook.is_applied

        hook.remove()
        assert not hook.is_applied
        assert hook.positions_biased == 0
        assert hook.layers_hooked == []

    def test_context_manager(self):
        model = self._make_model(12)
        with AttentionHook(model, {100: 0.5}, [100]) as hook:
            assert hook.is_applied
        assert not hook.is_applied

    def test_no_token_biases_noop(self):
        model = self._make_model(12)
        hook = AttentionHook(model, {}, [100])
        assert hook.apply() is False
        assert not hook.is_applied

    def test_no_input_ids_noop(self):
        model = self._make_model(12)
        hook = AttentionHook(model, {100: 0.5}, None)
        assert hook.apply() is False

    def test_no_matching_positions_noop(self):
        model = self._make_model(12)
        hook = AttentionHook(model, {100: 0.5}, [200, 300, 400])
        assert hook.apply() is False

    def test_specific_layers(self):
        model = self._make_model(12)
        hook = AttentionHook(model, {100: 0.5}, [100], layers=[3, 7])
        hook.apply()
        assert hook.layers_hooked == [3, 7]

    def test_reapply_cleans_previous(self):
        model = self._make_model(12)
        hook = AttentionHook(model, {100: 0.5}, [100])
        hook.apply()
        first_count = len(hook.layers_hooked)

        hook.apply()  # Re-apply
        assert len(hook.layers_hooked) == first_count


# ========== TestConstants ==========

class TestConstants:
    """Sanity checks for module constants."""

    def test_bias_range_symmetric(self):
        assert MAX_ATTENTION_BIAS == -MIN_ATTENTION_BIAS

    def test_bias_range_positive_max(self):
        assert MAX_ATTENTION_BIAS > 0

    def test_intensity_gate_reasonable(self):
        assert 0 < INTENSITY_GATE < 0.5

    def test_narrowing_scale_high_above_one(self):
        assert NARROWING_SCALE_HIGH > 1.0

    def test_narrowing_scale_low_below_one(self):
        assert NARROWING_SCALE_LOW < 1.0

    def test_seven_categories(self):
        assert len(ATTENTION_CATEGORIES) == 7

    def test_all_categories_are_strings(self):
        for cat in ATTENTION_CATEGORIES:
            assert isinstance(cat, str)
