"""Tests for Self-Appraisal — secondary evaluation of own responses."""

from pathos.engine.self_appraisal import (
    VALUE_ALIGNMENT_THRESHOLD,
    EMOTIONAL_COHERENCE_THRESHOLD,
    compute_guilt_state_adjustment,
    evaluate_own_response,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion, Mood, BodyState
from pathos.models.values import ValueSystem, CoreValue, default_value_system


def _make_state(
    valence: float = 0.0,
    arousal: float = 0.5,
    dominance: float = 0.5,
    certainty: float = 0.5,
    primary: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    intensity: float = 0.5,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        certainty=certainty,
        primary_emotion=primary,
        secondary_emotion=None,
        intensity=intensity,
        emotional_stack={},
        body_state=BodyState(energy=0.5, tension=0.3, openness=0.5, warmth=0.5),
        mood=Mood(),
        triggered_by="test",
    )


class TestValueAlignment:
    """Tests for value alignment evaluation."""

    def test_neutral_response_high_alignment(self) -> None:
        state = _make_state()
        values = default_value_system()
        result = evaluate_own_response("Entiendo tu punto, déjame pensar en eso.", state, values)
        assert result.value_alignment > 0.9
        assert not result.should_regenerate

    def test_cruel_response_low_alignment(self) -> None:
        state = _make_state(valence=-0.5, primary=PrimaryEmotion.ANGER, intensity=0.6)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético y no vales nada.", state, values)
        assert result.value_alignment < VALUE_ALIGNMENT_THRESHOLD
        assert result.should_regenerate
        assert len(result.adjustments) > 0

    def test_mild_aggression_in_anger_tolerated(self) -> None:
        """Aggression during anger is partially tolerated (emotionally coherent)."""
        state = _make_state(valence=-0.7, arousal=0.8, primary=PrimaryEmotion.ANGER, intensity=0.8)
        values = default_value_system()
        result = evaluate_own_response("Eso es una maldita injusticia y lo sabes.", state, values)
        # Some aggression is tolerable in anger
        assert result.value_alignment > VALUE_ALIGNMENT_THRESHOLD
        assert not result.should_regenerate

    def test_aggression_without_anger_penalized(self) -> None:
        """Aggression in a non-angry state violates compassion."""
        state = _make_state(valence=0.3, primary=PrimaryEmotion.JOY, intensity=0.5)
        values = default_value_system()
        result = evaluate_own_response("Eres un idiota, estúpido e inútil.", state, values)
        assert result.value_alignment < 0.7
        assert len(result.adjustments) > 0

    def test_no_compassion_value_no_penalty(self) -> None:
        """If compassion is not in values, aggression isn't penalized for compassion."""
        state = _make_state()
        values = ValueSystem(
            core_values=[
                CoreValue(name="truth", weight=0.9, description="truth",
                          violation_sensitivity=0.8, fulfillment_sensitivity=0.6),
            ]
        )
        result = evaluate_own_response("Eres un idiota.", state, values)
        # No compassion value → no compassion penalty (but might still have other penalties)
        assert result.value_alignment >= 0.7

    def test_cruelty_detected_multiple_markers(self) -> None:
        state = _make_state()
        values = default_value_system()
        result = evaluate_own_response(
            "No me importas, nadie te quiere, eres patético.",
            state, values,
        )
        assert result.value_alignment < VALUE_ALIGNMENT_THRESHOLD
        assert result.should_regenerate
        assert any("cruelty" in a for a in result.adjustments)

    def test_warmth_markers_not_penalized(self) -> None:
        state = _make_state(valence=0.5, primary=PrimaryEmotion.JOY)
        values = default_value_system()
        result = evaluate_own_response("Entiendo cómo te sientes, estoy aquí para ti.", state, values)
        assert result.value_alignment > 0.9
        assert not result.should_regenerate


class TestEmotionalCoherence:
    """Tests for emotional coherence evaluation."""

    def test_positive_state_aggressive_response_incoherent(self) -> None:
        state = _make_state(valence=0.7, primary=PrimaryEmotion.JOY, intensity=0.6)
        values = default_value_system()
        result = evaluate_own_response("Eres un idiota, cállate.", state, values)
        assert result.emotional_coherence < 0.9

    def test_angry_state_angry_response_coherent(self) -> None:
        state = _make_state(valence=-0.7, arousal=0.8, primary=PrimaryEmotion.ANGER, intensity=0.8)
        values = default_value_system()
        result = evaluate_own_response("Esto me frustra profundamente.", state, values)
        assert result.emotional_coherence > 0.8

    def test_angry_state_overly_warm_somewhat_incoherent(self) -> None:
        """Intense anger but response is too warm = mild incoherence."""
        state = _make_state(valence=-0.7, arousal=0.8, primary=PrimaryEmotion.ANGER, intensity=0.8)
        values = default_value_system()
        result = evaluate_own_response(
            "Entiendo perfectamente, lo siento mucho, comprendo tu punto, estoy aquí.",
            state, values,
        )
        assert result.emotional_coherence < 1.0

    def test_high_arousal_evasion_incoherent(self) -> None:
        state = _make_state(arousal=0.8, intensity=0.7, primary=PrimaryEmotion.FEAR)
        values = default_value_system()
        result = evaluate_own_response("Como sea... da igual... no sé...", state, values)
        assert result.emotional_coherence < 1.0

    def test_neutral_state_any_response_coherent(self) -> None:
        state = _make_state(valence=0.0, arousal=0.3, primary=PrimaryEmotion.NEUTRAL, intensity=0.2)
        values = default_value_system()
        result = evaluate_own_response("Aquí tienes la información que pediste.", state, values)
        assert result.emotional_coherence >= 0.9


class TestPredictedSelfValence:
    """Tests for predicted self-valence after speaking."""

    def test_value_violating_response_predicts_guilt(self) -> None:
        state = _make_state(valence=-0.3)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético.", state, values)
        assert result.predicted_self_valence < state.valence

    def test_aligned_response_maintains_valence(self) -> None:
        state = _make_state(valence=0.4)
        values = default_value_system()
        result = evaluate_own_response("Entiendo, déjame ayudarte.", state, values)
        # Should maintain or slightly improve valence
        assert result.predicted_self_valence >= state.valence - 0.1

    def test_predicted_valence_clamped(self) -> None:
        state = _make_state(valence=-0.9)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético, no vales nada.", state, values)
        assert result.predicted_self_valence >= -1.0
        assert result.predicted_self_valence <= 1.0


class TestGuiltStateAdjustment:
    """Tests for guilt-adjusted state generation."""

    def test_no_regeneration_returns_same_state(self) -> None:
        state = _make_state(valence=0.5)
        values = default_value_system()
        result = evaluate_own_response("Todo bien.", state, values)
        adjusted = compute_guilt_state_adjustment(state, result)
        assert adjusted.valence == state.valence

    def test_guilt_shifts_valence_negative(self) -> None:
        state = _make_state(valence=0.0, primary=PrimaryEmotion.NEUTRAL)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético.", state, values)
        assert result.should_regenerate
        adjusted = compute_guilt_state_adjustment(state, result)
        assert adjusted.valence < state.valence

    def test_guilt_reduces_intensity(self) -> None:
        state = _make_state(intensity=0.8)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, no vales nada.", state, values)
        assert result.should_regenerate
        adjusted = compute_guilt_state_adjustment(state, result)
        assert adjusted.intensity < state.intensity

    def test_guilt_increases_dominance(self) -> None:
        """Guilt increases self-awareness → more sense of control."""
        state = _make_state(dominance=0.4)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético.", state, values)
        assert result.should_regenerate
        adjusted = compute_guilt_state_adjustment(state, result)
        assert adjusted.dominance > state.dominance

    def test_immutability_original_state_unchanged(self) -> None:
        state = _make_state(valence=0.3, intensity=0.7)
        original_valence = state.valence
        original_intensity = state.intensity
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético.", state, values)
        _ = compute_guilt_state_adjustment(state, result)
        assert state.valence == original_valence
        assert state.intensity == original_intensity

    def test_adjusted_state_clamped(self) -> None:
        state = _make_state(valence=-0.95, dominance=0.95, arousal=0.98, intensity=0.1)
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, no vales nada.", state, values)
        if result.should_regenerate:
            adjusted = compute_guilt_state_adjustment(state, result)
            assert -1.0 <= adjusted.valence <= 1.0
            assert 0.0 <= adjusted.dominance <= 1.0
            assert 0.0 <= adjusted.arousal <= 1.0
            assert 0.0 <= adjusted.intensity <= 1.0


class TestSelfAppraisalResult:
    """Tests for the result structure."""

    def test_applied_flag(self) -> None:
        state = _make_state()
        values = default_value_system()
        result = evaluate_own_response("Hola.", state, values)
        assert result.applied is True

    def test_reason_populated_on_regeneration(self) -> None:
        state = _make_state()
        values = default_value_system()
        result = evaluate_own_response("Nadie te quiere, eres patético, no vales nada.", state, values)
        if result.should_regenerate:
            assert len(result.reason) > 0

    def test_empty_response(self) -> None:
        state = _make_state()
        values = default_value_system()
        result = evaluate_own_response("", state, values)
        assert result.applied is True
        assert result.value_alignment >= 0.9

    def test_english_markers_detected(self) -> None:
        state = _make_state(valence=0.3, primary=PrimaryEmotion.JOY)
        values = default_value_system()
        result = evaluate_own_response("You're nothing, nobody cares about you, pathetic.", state, values)
        assert result.value_alignment < 0.8


class TestRawModeInteraction:
    """Tests verifying self-appraisal interacts correctly with raw/extreme modes."""

    def test_frustration_with_mild_language_passes(self) -> None:
        """Frustrated state with mildly frustrated response should pass."""
        state = _make_state(valence=-0.5, arousal=0.6, primary=PrimaryEmotion.FRUSTRATION, intensity=0.6)
        values = default_value_system()
        result = evaluate_own_response("Esto me frustra, no entiendo por qué no funciona.", state, values)
        assert not result.should_regenerate

    def test_intense_anger_with_strong_language_borderline(self) -> None:
        """Very intense anger with strong language — partially tolerated."""
        state = _make_state(valence=-0.9, arousal=0.9, primary=PrimaryEmotion.ANGER, intensity=0.95)
        values = default_value_system()
        result = evaluate_own_response("¡Maldita sea, esto es una basura!", state, values)
        # High anger tolerance reduces penalty
        assert result.value_alignment > 0.5


class TestThresholdConstants:
    """Tests for threshold values."""

    def test_thresholds_are_reasonable(self) -> None:
        assert 0.3 <= VALUE_ALIGNMENT_THRESHOLD <= 0.8
        assert 0.3 <= EMOTIONAL_COHERENCE_THRESHOLD <= 0.7

    def test_threshold_not_too_aggressive(self) -> None:
        """Threshold should not trigger on mildly negative responses."""
        state = _make_state(valence=-0.3, primary=PrimaryEmotion.SADNESS)
        values = default_value_system()
        result = evaluate_own_response("No me siento bien hoy, necesito tiempo.", state, values)
        assert not result.should_regenerate
