"""Tests for Emotional Sampler — maps emotional state to LLM sampling parameters."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pathos.engine.emotional_sampler import (
    BASE_FREQUENCY_PENALTY,
    BASE_PRESENCE_PENALTY,
    BASE_REPETITION_PENALTY,
    BASE_TEMPERATURE,
    BASE_TOP_K,
    BASE_TOP_P,
    MAX_FREQUENCY_PENALTY,
    MAX_PRESENCE_PENALTY,
    MAX_REPETITION_PENALTY,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MAX_TOP_P,
    MIN_FREQUENCY_PENALTY,
    MIN_PRESENCE_PENALTY,
    MIN_REPETITION_PENALTY,
    MIN_TEMPERATURE,
    MIN_TOP_K,
    MIN_TOP_P,
    MAX_BIAS,
    MIN_BIAS,
    SamplingParams,
    TokenBiasResult,
    VOCABULARY_CATEGORIES,
    compute_sampling_params,
    compute_token_bias,
    load_emotional_vocabulary,
    resolve_token_ids,
)


# ========== TestSamplingParamsDataclass ==========

class TestSamplingParamsDataclass:
    """Tests for SamplingParams dataclass."""

    def test_frozen(self) -> None:
        """SamplingParams should be immutable."""
        params = SamplingParams(
            temperature=0.7, top_p=0.9, top_k=40,
            repetition_penalty=1.1, presence_penalty=0.0, frequency_penalty=0.0,
        )
        with pytest.raises(AttributeError):
            params.temperature = 0.5  # type: ignore[misc]

    def test_default_source(self) -> None:
        params = SamplingParams(
            temperature=0.7, top_p=0.9, top_k=40,
            repetition_penalty=1.1, presence_penalty=0.0, frequency_penalty=0.0,
        )
        assert params.source == "neutral"

    def test_custom_source(self) -> None:
        params = SamplingParams(
            temperature=0.7, top_p=0.9, top_k=40,
            repetition_penalty=1.1, presence_penalty=0.0, frequency_penalty=0.0,
            source="high_arousal",
        )
        assert params.source == "high_arousal"


# ========== TestNeutralState ==========

class TestNeutralState:
    """Neutral emotional state → base values unchanged."""

    def test_zero_intensity_returns_base(self) -> None:
        """Zero intensity → exact base values."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.0,
        )
        assert params.temperature == BASE_TEMPERATURE
        assert params.top_p == BASE_TOP_P
        assert params.top_k == BASE_TOP_K
        assert params.repetition_penalty == BASE_REPETITION_PENALTY
        assert params.presence_penalty == BASE_PRESENCE_PENALTY
        assert params.frequency_penalty == BASE_FREQUENCY_PENALTY
        assert params.source == "neutral"

    def test_very_low_intensity_returns_base(self) -> None:
        """Intensity < 0.05 → base values."""
        params = compute_sampling_params(
            valence=0.9, arousal=0.9, dominance=0.9,
            certainty=0.1, intensity=0.03,
        )
        assert params.temperature == BASE_TEMPERATURE
        assert params.source == "neutral"

    def test_neutral_dimensions_near_base(self) -> None:
        """All dimensions at neutral (0.5) with moderate intensity → near base."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        # Should be very close to base (valence=0 contributes nothing)
        assert abs(params.temperature - BASE_TEMPERATURE) < 0.05
        assert abs(params.top_p - BASE_TOP_P) < 0.05


# ========== TestTemperature ==========

class TestTemperature:
    """Temperature modulation from arousal."""

    def test_high_arousal_increases_temperature(self) -> None:
        """High arousal → higher temperature (more creative/chaotic)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert params.temperature > BASE_TEMPERATURE

    def test_low_arousal_decreases_temperature(self) -> None:
        """Low arousal → lower temperature (more measured/predictable)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.1, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert params.temperature < BASE_TEMPERATURE

    def test_temperature_clamped_high(self) -> None:
        """Extreme values don't exceed MAX_TEMPERATURE."""
        params = compute_sampling_params(
            valence=1.0, arousal=1.0, dominance=1.0,
            certainty=0.0, intensity=1.0,
            base_temperature=1.4,
        )
        assert params.temperature <= MAX_TEMPERATURE

    def test_temperature_clamped_low(self) -> None:
        """Extreme values don't go below MIN_TEMPERATURE."""
        params = compute_sampling_params(
            valence=-1.0, arousal=0.0, dominance=0.0,
            certainty=1.0, intensity=1.0,
            base_temperature=0.15,
        )
        assert params.temperature >= MIN_TEMPERATURE

    def test_custom_base_temperature(self) -> None:
        """Base temperature parameter is respected."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
            base_temperature=1.0,
        )
        assert abs(params.temperature - 1.0) < 0.05


# ========== TestTopP ==========

class TestTopP:
    """Top-p modulation from certainty."""

    def test_high_certainty_lowers_top_p(self) -> None:
        """High certainty → lower top_p (focused, predictable choices)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.9, intensity=0.8,
        )
        assert params.top_p < BASE_TOP_P

    def test_low_certainty_raises_top_p(self) -> None:
        """Low certainty → higher top_p (explores more options)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.1, intensity=0.8,
        )
        assert params.top_p > BASE_TOP_P

    def test_top_p_clamped(self) -> None:
        """Top-p stays in [MIN_TOP_P, MAX_TOP_P]."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.0, intensity=1.0,
        )
        assert MIN_TOP_P <= params.top_p <= MAX_TOP_P


# ========== TestTopK ==========

class TestTopK:
    """Top-k modulation from arousal."""

    def test_high_arousal_lowers_top_k(self) -> None:
        """High arousal → lower top_k (narrow focus)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert params.top_k < BASE_TOP_K

    def test_low_arousal_raises_top_k(self) -> None:
        """Low arousal → higher top_k (broader exploration)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.1, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert params.top_k > BASE_TOP_K

    def test_top_k_is_integer(self) -> None:
        params = compute_sampling_params(
            valence=0.5, arousal=0.7, dominance=0.5,
            certainty=0.5, intensity=0.6,
        )
        assert isinstance(params.top_k, int)

    def test_top_k_clamped(self) -> None:
        params = compute_sampling_params(
            valence=0.0, arousal=1.0, dominance=0.5,
            certainty=0.5, intensity=1.0,
        )
        assert MIN_TOP_K <= params.top_k <= MAX_TOP_K


# ========== TestRepetitionPenalty ==========

class TestRepetitionPenalty:
    """Repetition penalty modulation from valence + arousal."""

    def test_rumination_lowers_penalty(self) -> None:
        """Negative valence + low arousal (rumination) → lower penalty."""
        params = compute_sampling_params(
            valence=-0.8, arousal=0.1, dominance=0.3,
            certainty=0.5, intensity=0.8,
        )
        assert params.repetition_penalty < BASE_REPETITION_PENALTY

    def test_joy_raises_penalty(self) -> None:
        """Positive valence + high arousal (joy/excitement) → higher penalty."""
        params = compute_sampling_params(
            valence=0.8, arousal=0.9, dominance=0.6,
            certainty=0.5, intensity=0.8,
        )
        assert params.repetition_penalty > BASE_REPETITION_PENALTY

    def test_repetition_penalty_clamped(self) -> None:
        params = compute_sampling_params(
            valence=-1.0, arousal=0.0, dominance=0.0,
            certainty=0.0, intensity=1.0,
        )
        assert MIN_REPETITION_PENALTY <= params.repetition_penalty <= MAX_REPETITION_PENALTY


# ========== TestPresencePenalty ==========

class TestPresencePenalty:
    """Presence penalty modulation from dominance + valence."""

    def test_high_dominance_positive_valence_raises(self) -> None:
        """High dominance + positive valence → higher presence penalty (explores new topics)."""
        params = compute_sampling_params(
            valence=0.8, arousal=0.5, dominance=0.9,
            certainty=0.5, intensity=0.8,
        )
        assert params.presence_penalty > BASE_PRESENCE_PENALTY

    def test_low_dominance_lowers(self) -> None:
        """Low dominance → lower presence penalty (stays on topic)."""
        params = compute_sampling_params(
            valence=-0.3, arousal=0.5, dominance=0.1,
            certainty=0.5, intensity=0.8,
        )
        assert params.presence_penalty < BASE_PRESENCE_PENALTY or abs(params.presence_penalty) < 0.1

    def test_presence_penalty_clamped(self) -> None:
        params = compute_sampling_params(
            valence=1.0, arousal=0.5, dominance=1.0,
            certainty=0.5, intensity=1.0,
        )
        assert MIN_PRESENCE_PENALTY <= params.presence_penalty <= MAX_PRESENCE_PENALTY


# ========== TestFrequencyPenalty ==========

class TestFrequencyPenalty:
    """Frequency penalty modulation from arousal."""

    def test_high_arousal_lowers_frequency_penalty(self) -> None:
        """High arousal → lower frequency penalty (allows emphasis)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert params.frequency_penalty < BASE_FREQUENCY_PENALTY

    def test_low_arousal_raises_frequency_penalty(self) -> None:
        """Low arousal → higher frequency penalty (avoids monotone)."""
        params = compute_sampling_params(
            valence=0.0, arousal=0.1, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert params.frequency_penalty > BASE_FREQUENCY_PENALTY

    def test_frequency_penalty_clamped(self) -> None:
        params = compute_sampling_params(
            valence=0.0, arousal=1.0, dominance=0.5,
            certainty=0.5, intensity=1.0,
        )
        assert MIN_FREQUENCY_PENALTY <= params.frequency_penalty <= MAX_FREQUENCY_PENALTY


# ========== TestIntensityScaling ==========

class TestIntensityScaling:
    """Emotional intensity gates all modulations."""

    def test_higher_intensity_larger_delta(self) -> None:
        """Higher intensity → larger deviation from base."""
        low = compute_sampling_params(
            valence=0.0, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.3,
        )
        high = compute_sampling_params(
            valence=0.0, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.9,
        )
        # Both should increase temperature (high arousal), but high intensity more
        assert high.temperature > low.temperature

    def test_intensity_zero_exact_base(self) -> None:
        """Intensity 0 → exact base values regardless of other dimensions."""
        params = compute_sampling_params(
            valence=-1.0, arousal=1.0, dominance=0.0,
            certainty=0.0, intensity=0.0,
        )
        assert params.temperature == BASE_TEMPERATURE
        assert params.top_p == BASE_TOP_P
        assert params.top_k == BASE_TOP_K


# ========== TestEmotionalProfiles ==========

class TestEmotionalProfiles:
    """Tests for recognizable emotional profiles."""

    def test_anxious_profile(self) -> None:
        """Anxious: negative valence, high arousal, low dominance, low certainty."""
        params = compute_sampling_params(
            valence=-0.6, arousal=0.85, dominance=0.2,
            certainty=0.2, intensity=0.75,
        )
        # High arousal → high temperature
        assert params.temperature > BASE_TEMPERATURE
        # Low certainty → high top_p
        assert params.top_p > BASE_TOP_P
        # High arousal → low top_k
        assert params.top_k < BASE_TOP_K

    def test_confident_calm_profile(self) -> None:
        """Confident and calm: positive valence, low arousal, high dominance, high certainty."""
        params = compute_sampling_params(
            valence=0.5, arousal=0.2, dominance=0.8,
            certainty=0.9, intensity=0.6,
        )
        # Low arousal → low temperature
        assert params.temperature < BASE_TEMPERATURE
        # High certainty → low top_p
        assert params.top_p < BASE_TOP_P
        # Low arousal → high top_k
        assert params.top_k > BASE_TOP_K

    def test_depressed_rumination_profile(self) -> None:
        """Depressed rumination: negative valence, low arousal, low dominance."""
        params = compute_sampling_params(
            valence=-0.7, arousal=0.15, dominance=0.2,
            certainty=0.4, intensity=0.6,
        )
        # Negative valence + low arousal → low repetition penalty (allows rumination)
        assert params.repetition_penalty < BASE_REPETITION_PENALTY

    def test_excited_joy_profile(self) -> None:
        """Excited joy: positive valence, high arousal, high dominance."""
        params = compute_sampling_params(
            valence=0.8, arousal=0.9, dominance=0.7,
            certainty=0.6, intensity=0.85,
        )
        # High temperature, high repetition penalty, high presence penalty
        assert params.temperature > BASE_TEMPERATURE
        assert params.repetition_penalty > BASE_REPETITION_PENALTY
        assert params.presence_penalty > BASE_PRESENCE_PENALTY


# ========== TestDetermineSource ==========

class TestDetermineSource:
    """Tests for the diagnostic source label."""

    def test_low_intensity_source(self) -> None:
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.1,
        )
        assert params.source == "low_intensity"

    def test_high_arousal_source(self) -> None:
        params = compute_sampling_params(
            valence=0.0, arousal=0.95, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        assert params.source == "high_arousal"

    def test_negative_valence_source(self) -> None:
        params = compute_sampling_params(
            valence=-0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        assert params.source == "negative_valence"

    def test_positive_valence_source(self) -> None:
        params = compute_sampling_params(
            valence=0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        assert params.source == "positive_valence"

    def test_low_certainty_source(self) -> None:
        params = compute_sampling_params(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.05, intensity=0.5,
        )
        assert params.source == "low_certainty"


# ========== TestSafeRanges ==========

class TestSafeRanges:
    """All outputs stay within safe ranges regardless of extreme inputs."""

    @pytest.mark.parametrize("valence,arousal,dominance,certainty,intensity", [
        (1.0, 1.0, 1.0, 1.0, 1.0),     # All max
        (-1.0, 0.0, 0.0, 0.0, 1.0),     # All min (except intensity)
        (-1.0, 1.0, 0.0, 1.0, 1.0),     # Mixed extremes
        (1.0, 0.0, 1.0, 0.0, 1.0),      # Other mixed extremes
        (0.0, 0.5, 0.5, 0.5, 0.0),      # Perfectly neutral
    ])
    def test_all_params_in_range(
        self, valence: float, arousal: float, dominance: float,
        certainty: float, intensity: float,
    ) -> None:
        params = compute_sampling_params(
            valence=valence, arousal=arousal, dominance=dominance,
            certainty=certainty, intensity=intensity,
        )
        assert MIN_TEMPERATURE <= params.temperature <= MAX_TEMPERATURE
        assert MIN_TOP_P <= params.top_p <= MAX_TOP_P
        assert MIN_TOP_K <= params.top_k <= MAX_TOP_K
        assert MIN_REPETITION_PENALTY <= params.repetition_penalty <= MAX_REPETITION_PENALTY
        assert MIN_PRESENCE_PENALTY <= params.presence_penalty <= MAX_PRESENCE_PENALTY
        assert MIN_FREQUENCY_PENALTY <= params.frequency_penalty <= MAX_FREQUENCY_PENALTY


# ========== TestConstants ==========

class TestConstants:
    """Sanity checks for module-level constants."""

    def test_base_temperature_reasonable(self) -> None:
        assert 0.1 <= BASE_TEMPERATURE <= 1.5

    def test_base_top_p_reasonable(self) -> None:
        assert 0.5 <= BASE_TOP_P <= 1.0

    def test_base_top_k_reasonable(self) -> None:
        assert 10 <= BASE_TOP_K <= 100

    def test_min_less_than_max(self) -> None:
        assert MIN_TEMPERATURE < MAX_TEMPERATURE
        assert MIN_TOP_P < MAX_TOP_P
        assert MIN_TOP_K < MAX_TOP_K
        assert MIN_REPETITION_PENALTY < MAX_REPETITION_PENALTY
        assert MIN_PRESENCE_PENALTY < MAX_PRESENCE_PENALTY
        assert MIN_FREQUENCY_PENALTY < MAX_FREQUENCY_PENALTY

    def test_base_within_range(self) -> None:
        assert MIN_TEMPERATURE <= BASE_TEMPERATURE <= MAX_TEMPERATURE
        assert MIN_TOP_P <= BASE_TOP_P <= MAX_TOP_P
        assert MIN_TOP_K <= BASE_TOP_K <= MAX_TOP_K
        assert MIN_REPETITION_PENALTY <= BASE_REPETITION_PENALTY <= MAX_REPETITION_PENALTY
        assert MIN_PRESENCE_PENALTY <= BASE_PRESENCE_PENALTY <= MAX_PRESENCE_PENALTY
        assert MIN_FREQUENCY_PENALTY <= BASE_FREQUENCY_PENALTY <= MAX_FREQUENCY_PENALTY


# ===========================================================================
# Token-Level Logit Bias Tests
# ===========================================================================


# ========== TestLoadEmotionalVocabulary ==========

class TestLoadEmotionalVocabulary:
    """Tests for loading emotional vocabulary from JSON."""

    def test_load_real_vocabulary(self) -> None:
        """Real vocabulary file loads without error."""
        vocab = load_emotional_vocabulary()
        for cat in VOCABULARY_CATEGORIES:
            assert cat in vocab
            assert len(vocab[cat]) >= 5

    def test_all_categories_present(self) -> None:
        vocab = load_emotional_vocabulary()
        assert set(VOCABULARY_CATEGORIES).issubset(vocab.keys())

    def test_words_are_strings(self) -> None:
        vocab = load_emotional_vocabulary()
        for cat in VOCABULARY_CATEGORIES:
            for word in vocab[cat]:
                assert isinstance(word, str)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_emotional_vocabulary(Path("/nonexistent/vocab.json"))

    def test_invalid_structure_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"positive": ["a"]}))
        with pytest.raises(ValueError):
            load_emotional_vocabulary(bad_file)


# ========== TestTokenBiasResult ==========

class TestTokenBiasResult:
    """Tests for TokenBiasResult dataclass."""

    def test_empty_result(self) -> None:
        result = TokenBiasResult(word_biases={})
        assert result.word_biases == {}
        assert result.token_biases == {}
        assert result.categories_active == {}

    def test_with_data(self) -> None:
        result = TokenBiasResult(
            word_biases={"happy": 1.5, "sad": -1.0},
            categories_active={"positive": 0.8, "negative": -0.8},
        )
        assert len(result.word_biases) == 2
        assert result.word_biases["happy"] == 1.5


# ========== TestComputeTokenBias ==========

class TestComputeTokenBias:
    """Tests for compute_token_bias function."""

    def test_low_intensity_empty(self) -> None:
        """Intensity < 0.10 → empty biases."""
        result = compute_token_bias(
            valence=0.9, arousal=0.9, dominance=0.9,
            certainty=0.1, intensity=0.05,
        )
        assert result.word_biases == {}
        assert result.categories_active == {}

    def test_positive_valence_boosts_positive(self) -> None:
        """Positive valence → positive words boosted."""
        result = compute_token_bias(
            valence=0.8, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        # Some positive words should have positive bias
        positive_words = load_emotional_vocabulary()["positive"]
        boosted = [w for w in positive_words if result.word_biases.get(w, 0) > 0]
        assert len(boosted) > 0

    def test_positive_valence_suppresses_negative(self) -> None:
        """Positive valence → negative words suppressed."""
        result = compute_token_bias(
            valence=0.8, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        negative_words = load_emotional_vocabulary()["negative"]
        suppressed = [w for w in negative_words if result.word_biases.get(w, 0) < 0]
        assert len(suppressed) > 0

    def test_negative_valence_boosts_negative(self) -> None:
        """Negative valence → negative words boosted."""
        result = compute_token_bias(
            valence=-0.8, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        negative_words = load_emotional_vocabulary()["negative"]
        boosted = [w for w in negative_words if result.word_biases.get(w, 0) > 0]
        assert len(boosted) > 0

    def test_high_arousal_boosts_energy(self) -> None:
        """High arousal → high_energy words boosted."""
        result = compute_token_bias(
            valence=0.0, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        energy_words = load_emotional_vocabulary()["high_energy"]
        boosted = [w for w in energy_words if result.word_biases.get(w, 0) > 0]
        assert len(boosted) > 0

    def test_low_arousal_boosts_low_energy(self) -> None:
        """Low arousal → low_energy words boosted."""
        result = compute_token_bias(
            valence=0.0, arousal=0.1, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        low_energy_words = load_emotional_vocabulary()["low_energy"]
        boosted = [w for w in low_energy_words if result.word_biases.get(w, 0) > 0]
        assert len(boosted) > 0

    def test_high_dominance_certainty_boosts_assertive(self) -> None:
        """High dominance + certainty → assertive words boosted."""
        result = compute_token_bias(
            valence=0.0, arousal=0.5, dominance=0.9,
            certainty=0.9, intensity=0.8,
        )
        assertive_words = load_emotional_vocabulary()["assertive"]
        boosted = [w for w in assertive_words if result.word_biases.get(w, 0) > 0]
        assert len(boosted) > 0

    def test_low_dominance_certainty_boosts_uncertainty(self) -> None:
        """Low dominance + certainty → uncertainty words boosted."""
        result = compute_token_bias(
            valence=0.0, arousal=0.5, dominance=0.1,
            certainty=0.1, intensity=0.8,
        )
        uncertainty_words = load_emotional_vocabulary()["uncertainty"]
        boosted = [w for w in uncertainty_words if result.word_biases.get(w, 0) > 0]
        assert len(boosted) > 0

    def test_neutral_state_empty_or_minimal(self) -> None:
        """All dimensions neutral → few or no biases."""
        result = compute_token_bias(
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        # Valence=0 and all dimensions at neutral → no categories pass threshold
        assert len(result.word_biases) == 0

    def test_categories_active_populated(self) -> None:
        """categories_active should list which categories were activated."""
        result = compute_token_bias(
            valence=0.8, arousal=0.9, dominance=0.5,
            certainty=0.5, intensity=0.8,
        )
        assert len(result.categories_active) > 0
        assert "positive" in result.categories_active
        assert "high_energy" in result.categories_active

    def test_custom_vocabulary(self) -> None:
        """Custom vocabulary override works."""
        custom = {
            "positive": ["good", "nice", "happy", "great", "fine"],
            "negative": ["bad", "awful", "terrible", "worst", "hate"],
            "high_energy": ["wow", "yes", "go", "fast", "rush"],
            "low_energy": ["slow", "soft", "calm", "quiet", "rest"],
            "uncertainty": ["maybe", "perhaps", "might", "could", "unsure"],
            "assertive": ["must", "will", "shall", "demand", "insist"],
        }
        result = compute_token_bias(
            valence=0.9, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.9,
            vocabulary=custom,
        )
        # Should use custom vocab
        assert "good" in result.word_biases or "nice" in result.word_biases


# ========== TestTokenBiasRanges ==========

class TestTokenBiasRanges:
    """All bias values stay within safe ranges."""

    @pytest.mark.parametrize("valence,arousal,dominance,certainty,intensity", [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 1.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 0.0, 1.0),
    ])
    def test_all_biases_in_range(
        self, valence: float, arousal: float, dominance: float,
        certainty: float, intensity: float,
    ) -> None:
        result = compute_token_bias(
            valence=valence, arousal=arousal, dominance=dominance,
            certainty=certainty, intensity=intensity,
        )
        for word, bias in result.word_biases.items():
            assert MIN_BIAS <= bias <= MAX_BIAS, f"{word}: {bias} out of range"


# ========== TestResolveTokenIds ==========

class TestResolveTokenIds:
    """Tests for resolve_token_ids function."""

    def test_empty_biases_empty_result(self) -> None:
        tokenizer = MagicMock()
        result = resolve_token_ids({}, tokenizer)
        assert result == {}

    def test_single_token_word(self) -> None:
        """Single-token word gets full bias."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]
        result = resolve_token_ids({"happy": 1.5}, tokenizer)
        assert 42 in result
        assert result[42] == 1.5

    def test_multi_token_word_splits_bias(self) -> None:
        """Multi-token word splits bias across tokens."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 20]  # 2 tokens
        result = resolve_token_ids({"wonderful": 2.0}, tokenizer)
        assert 10 in result
        assert 20 in result
        assert result[10] == pytest.approx(1.0, abs=0.01)
        assert result[20] == pytest.approx(1.0, abs=0.01)

    def test_accumulates_same_token(self) -> None:
        """Same token from different words accumulates bias."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda w, **k: [42] if w == "happy" else [42]
        result = resolve_token_ids({"happy": 1.0, "glad": 0.5}, tokenizer)
        assert 42 in result
        assert result[42] == pytest.approx(1.5, abs=0.01)

    def test_clamped_accumulation(self) -> None:
        """Accumulated bias is clamped to [MIN_BIAS, MAX_BIAS]."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]
        result = resolve_token_ids({"a": 2.0, "b": 2.0}, tokenizer)
        assert result[42] <= MAX_BIAS

    def test_tokenizer_failure_skipped(self) -> None:
        """Words that fail to tokenize are skipped."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = RuntimeError("bad word")
        result = resolve_token_ids({"broken": 1.0}, tokenizer)
        assert result == {}

    def test_empty_token_list_skipped(self) -> None:
        """Words that tokenize to empty list are skipped."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []
        result = resolve_token_ids({"empty": 1.0}, tokenizer)
        assert result == {}

    def test_near_zero_filtered(self) -> None:
        """Very small biases are filtered out."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]
        result = resolve_token_ids({"tiny": 0.01}, tokenizer)
        assert 42 not in result


# ========== TestBiasConstants ==========

class TestBiasConstants:
    """Sanity checks for bias constants."""

    def test_max_bias_positive(self) -> None:
        assert MAX_BIAS > 0

    def test_min_bias_negative(self) -> None:
        assert MIN_BIAS < 0

    def test_symmetric_range(self) -> None:
        assert abs(MAX_BIAS + MIN_BIAS) < 0.01  # Roughly symmetric

    def test_all_categories_defined(self) -> None:
        assert len(VOCABULARY_CATEGORIES) == 6
