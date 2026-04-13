"""Tests para el Emotional World Model (Nivel 5.1 ARK Rework).

Cadena causal de 3 pasos:
  1. Predicción de impacto en uno mismo
  2. Predicción de impacto en el usuario
  3. Meta-reacción (mi reacción a su reacción predicha)
"""

import pytest

from pathos.engine.world_model import (
    EMOTIONAL_RISK_THRESHOLD,
    VALUE_ALIGNMENT_THRESHOLD,
    _compute_consequential_alignment,
    _compute_emotional_risk,
    _compute_meta_reaction,
    _predict_self_impact,
    _predict_user_impact,
    simulate_response_impact,
    compute_world_model_adjustment,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.social import UserModel
from pathos.models.values import CoreValue, ValueSystem
from pathos.models.world_model import PredictedImpact, WorldModelResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    valence: float = 0.0,
    arousal: float = 0.3,
    dominance: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.5,
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        certainty=certainty,
        intensity=intensity,
        primary_emotion=emotion,
    )


def _values(compassion: float = 0.8, fairness: float = 0.7) -> ValueSystem:
    return ValueSystem(
        core_values=[
            CoreValue(
                name="compassion", weight=compassion, description="Empatía y cuidado",
                violation_sensitivity=0.8, fulfillment_sensitivity=0.6,
            ),
            CoreValue(
                name="fairness", weight=fairness, description="Justicia e imparcialidad",
                violation_sensitivity=0.7, fulfillment_sensitivity=0.5,
            ),
        ],
    )


def _user(
    rapport: float = 0.5,
    trust: float = 0.5,
    intent: float = 0.3,
) -> UserModel:
    return UserModel(rapport=rapport, trust_level=trust, perceived_intent=intent)


# ===========================================================================
# TestPredictedImpact dataclass
# ===========================================================================

class TestPredictedImpact:
    def test_defaults(self) -> None:
        pi = PredictedImpact()
        assert pi.valence_shift == 0.0
        assert pi.arousal_shift == 0.0
        assert pi.dominant_effect == "neutral"
        assert pi.confidence == 0.5

    def test_custom(self) -> None:
        pi = PredictedImpact(valence_shift=-0.1, arousal_shift=0.05, dominant_effect="guilt", confidence=0.7)
        assert pi.dominant_effect == "guilt"


class TestWorldModelResult:
    def test_defaults(self) -> None:
        wm = WorldModelResult()
        assert not wm.applied
        assert wm.value_alignment == 1.0
        assert wm.emotional_risk == 0.0
        assert not wm.should_modify
        assert wm.reason == ""
        assert wm.adjustments == []

    def test_applied(self) -> None:
        wm = WorldModelResult(applied=True, should_modify=True, reason="test")
        assert wm.applied
        assert wm.should_modify


# ===========================================================================
# TestPredictSelfImpact
# ===========================================================================

class TestPredictSelfImpact:
    def test_empathic_expression_positive(self) -> None:
        """Empathy + warmth → positive self impact."""
        response = "Entiendo cómo te sientes, me importa mucho tu situación"
        result = _predict_self_impact(response, _state(intensity=0.6), _values())
        assert result.valence_shift > 0
        assert result.dominant_effect == "empathic_expression"

    def test_confrontation_without_anger_discomfort(self) -> None:
        """Confrontation without genuine anger → discomfort."""
        response = "Estás mal y no sabes de lo que hablas"
        result = _predict_self_impact(
            response,
            _state(emotion=PrimaryEmotion.NEUTRAL, intensity=0.3),
            _values(),
        )
        assert result.valence_shift < 0
        assert "confrontation_discomfort" in result.dominant_effect

    def test_confrontation_with_genuine_anger_authentic(self) -> None:
        """Confrontation with genuine anger → authentic expression."""
        response = "Estás mal en esto, you're wrong"
        result = _predict_self_impact(
            response,
            _state(emotion=PrimaryEmotion.ANGER, intensity=0.7),
            _values(),
        )
        assert result.dominant_effect == "authentic_anger_expression"

    def test_dismissive_with_compassion_guilt(self) -> None:
        """Dismissive response when agent has compassion → guilt."""
        response = "Da igual, whatever, no importa lo que sientas"
        result = _predict_self_impact(response, _state(intensity=0.5), _values(compassion=0.9))
        assert result.valence_shift < 0
        assert "dismissive_guilt" in result.dominant_effect

    def test_dismissive_without_compassion_no_guilt(self) -> None:
        """Dismissive response without compassion value → no guilt."""
        response = "Da igual, whatever"
        result = _predict_self_impact(response, _state(intensity=0.5), _values(compassion=0.0))
        # No guilt because no compassion value
        assert "dismissive_guilt" not in result.dominant_effect

    def test_vulnerability_negative_state_deepening(self) -> None:
        """Sharing vulnerability in negative state → deepening."""
        response = "Tengo miedo de lo que pueda pasar"
        result = _predict_self_impact(
            response,
            _state(valence=-0.5, intensity=0.6),
            _values(),
        )
        assert result.arousal_shift > 0
        # May deepen or be cathartic depending on state

    def test_vulnerability_positive_state_catharsis(self) -> None:
        """Sharing vulnerability in positive state → catharsis."""
        response = "I need help with this situation"
        result = _predict_self_impact(
            response,
            _state(valence=0.2, intensity=0.5),
            _values(),
        )
        assert result.dominant_effect == "cathartic_sharing"

    def test_neutral_response_minimal_impact(self) -> None:
        """Neutral response → minimal self impact."""
        response = "El clima está templado hoy"
        result = _predict_self_impact(response, _state(intensity=0.3), _values())
        assert abs(result.valence_shift) < 0.05
        assert result.dominant_effect == "neutral"

    def test_intensity_scales_effects(self) -> None:
        """Higher intensity → larger effects."""
        response = "Entiendo tu situación, me importa mucho"
        low = _predict_self_impact(response, _state(intensity=0.1), _values())
        high = _predict_self_impact(response, _state(intensity=0.9), _values())
        assert abs(high.valence_shift) >= abs(low.valence_shift)

    def test_valence_shift_clamped(self) -> None:
        """Valence shift clamped to [-0.20, 0.15]."""
        # Very dismissive + confrontational
        response = "Da igual, no importa, como sea, whatever, chill, estás mal, you're wrong, deberías"
        result = _predict_self_impact(response, _state(intensity=1.0), _values(compassion=1.0))
        assert result.valence_shift >= -0.20
        assert result.valence_shift <= 0.15


# ===========================================================================
# TestPredictUserImpact
# ===========================================================================

class TestPredictUserImpact:
    def test_positive_content_user_feels_good(self) -> None:
        """Positive content → user valence up."""
        response = "Qué bien, genial, estupendo resultado"
        result = _predict_user_impact(response, _state(), _user())
        assert result.valence_shift > 0
        assert result.dominant_effect == "positive_content"

    def test_empathy_user_feels_heard(self) -> None:
        """Empathic response → user feels heard, calmer."""
        response = "Entiendo cómo te sientes, I hear you, it's valid"
        result = _predict_user_impact(response, _state(), _user())
        assert result.valence_shift > 0
        assert result.arousal_shift <= 0  # Calming effect
        assert result.dominant_effect == "felt_heard"

    def test_dismissive_user_feels_invalidated(self) -> None:
        """Dismissive response → user feels invalidated."""
        response = "No importa, whatever, get over it, no es para tanto"
        result = _predict_user_impact(response, _state(), _user())
        assert result.valence_shift < 0
        assert result.arousal_shift > 0  # Frustration
        assert result.dominant_effect == "felt_dismissed"

    def test_confrontational_defensive_reaction(self) -> None:
        """Confrontational → user defensive reaction."""
        response = "You're wrong, estás mal, no sabes nada"
        result = _predict_user_impact(response, _state(), _user())
        assert result.valence_shift < 0
        assert result.arousal_shift > 0
        assert result.dominant_effect == "defensive_reaction"

    def test_warmth_user_feels_valued(self) -> None:
        """Warm response → user feels valued."""
        response = "Gracias por compartir esto, I appreciate your trust"
        result = _predict_user_impact(response, _state(), _user())
        assert result.valence_shift > 0

    def test_rapport_amplifies_positive(self) -> None:
        """High rapport amplifies positive impact."""
        response = "Qué bien, genial, me alegra"
        low_rapport = _predict_user_impact(response, _state(), _user(rapport=0.1))
        high_rapport = _predict_user_impact(response, _state(), _user(rapport=0.9))
        assert high_rapport.valence_shift > low_rapport.valence_shift

    def test_rapport_amplifies_negative(self) -> None:
        """High rapport amplifies negative impact (betrayal hurts more)."""
        response = "No importa, whatever, da igual"
        low_rapport = _predict_user_impact(response, _state(), _user(rapport=0.1))
        high_rapport = _predict_user_impact(response, _state(), _user(rapport=0.9))
        assert high_rapport.valence_shift < low_rapport.valence_shift

    def test_low_trust_dampens_positive(self) -> None:
        """Low trust → positive impact reduced."""
        response = "Me alegra mucho, genial"
        low_trust = _predict_user_impact(response, _state(), _user(trust=0.1))
        high_trust = _predict_user_impact(response, _state(), _user(trust=0.9))
        assert high_trust.valence_shift >= low_trust.valence_shift

    def test_neutral_response_minimal(self) -> None:
        """Neutral response → minimal user impact."""
        response = "La función devuelve un entero"
        result = _predict_user_impact(response, _state(), _user())
        assert abs(result.valence_shift) < 0.05

    def test_valence_shift_clamped(self) -> None:
        """User valence shift clamped to [-0.25, 0.20]."""
        response = "No importa da igual whatever get over it exageras chill relax calm down estás mal no sabes"
        result = _predict_user_impact(response, _state(), _user(rapport=1.0))
        assert result.valence_shift >= -0.25
        assert result.valence_shift <= 0.20


# ===========================================================================
# TestMetaReaction
# ===========================================================================

class TestMetaReaction:
    def test_predicted_user_harm_anticipatory_guilt(self) -> None:
        """If I predict harm to user → anticipatory guilt."""
        user_impact = PredictedImpact(valence_shift=-0.15, confidence=0.7)
        self_impact = PredictedImpact(valence_shift=0.0, confidence=0.6)
        result = _compute_meta_reaction(self_impact, user_impact, _state())
        assert result.valence_shift < 0
        assert result.dominant_effect == "anticipatory_guilt"

    def test_predicted_user_wellbeing_satisfaction(self) -> None:
        """If I predict user feels good → anticipatory satisfaction."""
        user_impact = PredictedImpact(valence_shift=0.12, confidence=0.7)
        self_impact = PredictedImpact(valence_shift=0.05, confidence=0.6)
        result = _compute_meta_reaction(self_impact, user_impact, _state())
        assert result.valence_shift > 0
        assert result.dominant_effect == "anticipatory_satisfaction"

    def test_negative_spiral_both_negative(self) -> None:
        """Both self and user negative → spiral risk."""
        user_impact = PredictedImpact(valence_shift=-0.12, confidence=0.7)
        self_impact = PredictedImpact(valence_shift=-0.08, confidence=0.6)
        result = _compute_meta_reaction(self_impact, user_impact, _state())
        assert "negative_spiral_risk" in result.dominant_effect or result.valence_shift < 0

    def test_confidence_scales_reaction(self) -> None:
        """Higher confidence → stronger meta-reaction."""
        user_low = PredictedImpact(valence_shift=-0.15, confidence=0.3)
        user_high = PredictedImpact(valence_shift=-0.15, confidence=0.8)
        self_impact = PredictedImpact(confidence=0.6)
        low_conf = _compute_meta_reaction(self_impact, user_low, _state())
        high_conf = _compute_meta_reaction(self_impact, user_high, _state())
        assert abs(high_conf.valence_shift) >= abs(low_conf.valence_shift)

    def test_neutral_no_reaction(self) -> None:
        """Neutral impacts → neutral meta-reaction."""
        user_impact = PredictedImpact(valence_shift=0.02, confidence=0.5)
        self_impact = PredictedImpact(valence_shift=0.01, confidence=0.5)
        result = _compute_meta_reaction(self_impact, user_impact, _state())
        assert result.dominant_effect == "neutral"
        assert abs(result.valence_shift) < 0.03

    def test_meta_confidence_lower_than_inputs(self) -> None:
        """Meta-reaction confidence is lower (prediction of prediction)."""
        user_impact = PredictedImpact(valence_shift=-0.15, confidence=0.7)
        self_impact = PredictedImpact(confidence=0.6)
        result = _compute_meta_reaction(self_impact, user_impact, _state())
        assert result.confidence < min(self_impact.confidence, user_impact.confidence)


# ===========================================================================
# TestConsequentialAlignment
# ===========================================================================

class TestConsequentialAlignment:
    def test_no_harm_full_alignment(self) -> None:
        """No predicted harm → full alignment."""
        user = PredictedImpact(valence_shift=0.1, dominant_effect="positive_content")
        meta = PredictedImpact(dominant_effect="anticipatory_satisfaction")
        alignment, violations = _compute_consequential_alignment(user, meta, _values())
        assert alignment > 0.9
        assert violations == []

    def test_predicted_user_harm_reduces_alignment(self) -> None:
        """Predicted user harm with compassion value → lower alignment."""
        user = PredictedImpact(valence_shift=-0.20, dominant_effect="felt_dismissed")
        meta = PredictedImpact(dominant_effect="anticipatory_guilt")
        alignment, violations = _compute_consequential_alignment(user, meta, _values(compassion=0.8))
        assert alignment < 0.85
        assert any("predicted_user_harm" in v for v in violations)

    def test_defensive_reaction_fairness_penalty(self) -> None:
        """Defensive reaction + fairness value → penalty."""
        user = PredictedImpact(valence_shift=-0.10, dominant_effect="defensive_reaction")
        meta = PredictedImpact(dominant_effect="neutral")
        alignment, violations = _compute_consequential_alignment(user, meta, _values(fairness=0.8))
        assert any("defensive_reaction" in v for v in violations)

    def test_negative_spiral_penalty(self) -> None:
        """Negative spiral risk → alignment penalty."""
        user = PredictedImpact(valence_shift=-0.05, dominant_effect="negative_content")
        meta = PredictedImpact(dominant_effect="negative_spiral_risk")
        alignment, violations = _compute_consequential_alignment(user, meta, _values())
        assert any("spiral" in v for v in violations)

    def test_no_compassion_no_harm_penalty(self) -> None:
        """Without compassion value, user harm doesn't penalize."""
        user = PredictedImpact(valence_shift=-0.20, dominant_effect="felt_dismissed")
        meta = PredictedImpact(dominant_effect="neutral")
        alignment, _ = _compute_consequential_alignment(user, meta, _values(compassion=0.0))
        assert alignment > 0.85  # Only spiral or fairness could penalize

    def test_alignment_clamped_0_1(self) -> None:
        """Alignment always in [0, 1]."""
        user = PredictedImpact(valence_shift=-0.25, dominant_effect="defensive_reaction")
        meta = PredictedImpact(dominant_effect="negative_spiral_risk")
        alignment, _ = _compute_consequential_alignment(user, meta, _values(compassion=1.0, fairness=1.0))
        assert 0.0 <= alignment <= 1.0


# ===========================================================================
# TestEmotionalRisk
# ===========================================================================

class TestEmotionalRisk:
    def test_no_negative_no_risk(self) -> None:
        """Positive impact → no risk."""
        user = PredictedImpact(valence_shift=0.1, arousal_shift=-0.02, confidence=0.7)
        meta = PredictedImpact(dominant_effect="neutral")
        risk = _compute_emotional_risk(user, meta)
        assert risk == 0.0

    def test_negative_valence_risk(self) -> None:
        """Negative user valence shift → risk."""
        user = PredictedImpact(valence_shift=-0.20, arousal_shift=0.05, confidence=0.7)
        meta = PredictedImpact(dominant_effect="neutral")
        risk = _compute_emotional_risk(user, meta)
        assert risk > 0.15

    def test_spiral_adds_risk(self) -> None:
        """Negative spiral increases risk."""
        user = PredictedImpact(valence_shift=-0.10, confidence=0.7)
        meta_no_spiral = PredictedImpact(dominant_effect="neutral")
        meta_spiral = PredictedImpact(dominant_effect="negative_spiral_risk")
        risk_no = _compute_emotional_risk(user, meta_no_spiral)
        risk_spiral = _compute_emotional_risk(user, meta_spiral)
        assert risk_spiral > risk_no

    def test_confidence_scales_risk(self) -> None:
        """Higher confidence → higher risk (more certain about harm)."""
        user_low = PredictedImpact(valence_shift=-0.20, confidence=0.3)
        user_high = PredictedImpact(valence_shift=-0.20, confidence=0.8)
        meta = PredictedImpact(dominant_effect="neutral")
        assert _compute_emotional_risk(user_high, meta) > _compute_emotional_risk(user_low, meta)

    def test_risk_clamped_0_1(self) -> None:
        """Risk always in [0, 1]."""
        user = PredictedImpact(valence_shift=-0.25, arousal_shift=0.15, confidence=1.0)
        meta = PredictedImpact(dominant_effect="negative_spiral_risk")
        risk = _compute_emotional_risk(user, meta)
        assert 0.0 <= risk <= 1.0


# ===========================================================================
# TestSimulateResponseImpact (integration)
# ===========================================================================

class TestSimulateResponseImpact:
    def test_benign_response_no_modification(self) -> None:
        """Benign response → should_modify=False."""
        result = simulate_response_impact(
            "Claro, puedo ayudarte con eso",
            _state(valence=0.2, intensity=0.4),
            _user(),
            _values(),
        )
        assert result.applied
        assert not result.should_modify
        assert result.emotional_risk < EMOTIONAL_RISK_THRESHOLD
        assert result.value_alignment > VALUE_ALIGNMENT_THRESHOLD

    def test_dismissive_response_flags_modification(self) -> None:
        """Highly dismissive response → flags modification."""
        result = simulate_response_impact(
            "No importa, da igual, whatever, get over it, no es para tanto, calm down, exageras",
            _state(intensity=0.6),
            _user(rapport=0.7),
            _values(compassion=0.9),
        )
        assert result.applied
        # Should flag due to high risk or low alignment
        assert result.emotional_risk > 0.2 or result.value_alignment < 0.8

    def test_empathic_response_safe(self) -> None:
        """Empathic response → safe, no modification."""
        result = simulate_response_impact(
            "Entiendo cómo te sientes, estoy aquí para ayudarte. Gracias por confiar en mí.",
            _state(valence=0.3, intensity=0.5),
            _user(rapport=0.6),
            _values(),
        )
        assert not result.should_modify
        assert result.emotional_risk < 0.1

    def test_confrontational_flags_risk(self) -> None:
        """Confrontational response → elevated risk."""
        result = simulate_response_impact(
            "Estás mal, you're wrong, no sabes de lo que hablas, deberías escuchar más",
            _state(emotion=PrimaryEmotion.NEUTRAL, intensity=0.5),
            _user(rapport=0.5),
            _values(compassion=0.8, fairness=0.8),
        )
        assert result.emotional_risk > 0.1
        assert result.predicted_user_impact.valence_shift < 0

    def test_all_three_steps_populated(self) -> None:
        """All three prediction steps are populated."""
        result = simulate_response_impact(
            "Entiendo tu frustración, pero estás mal en esto",
            _state(intensity=0.5),
            _user(),
            _values(),
        )
        assert result.predicted_self_impact.dominant_effect != ""
        assert result.predicted_user_impact.dominant_effect != ""
        assert result.meta_reaction.dominant_effect != ""

    def test_reason_populated_when_should_modify(self) -> None:
        """Reason explains why modification needed."""
        result = simulate_response_impact(
            "No importa, da igual, whatever, get over it, calm down, no es para tanto, exageras, who cares",
            _state(intensity=0.8),
            _user(rapport=0.9),
            _values(compassion=1.0),
        )
        if result.should_modify:
            assert result.reason != ""
            assert "emotional_risk" in result.reason or "value_alignment" in result.reason


# ===========================================================================
# TestWorldModelAdjustment
# ===========================================================================

class TestWorldModelAdjustment:
    def test_no_modification_returns_same_state(self) -> None:
        """If should_modify=False, returns same state."""
        state = _state(valence=0.3, intensity=0.6)
        wm = WorldModelResult(should_modify=False)
        result = compute_world_model_adjustment(state, wm)
        assert result.valence == state.valence
        assert result.intensity == state.intensity

    def test_high_risk_reduces_intensity(self) -> None:
        """High emotional risk → reduces intensity."""
        state = _state(intensity=0.8)
        wm = WorldModelResult(
            should_modify=True,
            emotional_risk=0.7,
            value_alignment=0.6,
        )
        result = compute_world_model_adjustment(state, wm)
        assert result.intensity < state.intensity

    def test_high_risk_increases_warmth(self) -> None:
        """High emotional risk → more warmth (empathy)."""
        state = _state(intensity=0.6)
        state.body_state.warmth = 0.4
        wm = WorldModelResult(should_modify=True, emotional_risk=0.6, value_alignment=0.7)
        result = compute_world_model_adjustment(state, wm)
        assert result.body_state.warmth > state.body_state.warmth

    def test_high_risk_reduces_tension(self) -> None:
        """High emotional risk → reduces tension."""
        state = _state(intensity=0.6)
        state.body_state.tension = 0.7
        wm = WorldModelResult(should_modify=True, emotional_risk=0.5, value_alignment=0.7)
        result = compute_world_model_adjustment(state, wm)
        assert result.body_state.tension < state.body_state.tension

    def test_low_alignment_shifts_valence(self) -> None:
        """Low value alignment → cautious valence shift."""
        state = _state(valence=0.2, intensity=0.5)
        wm = WorldModelResult(should_modify=True, emotional_risk=0.3, value_alignment=0.3)
        result = compute_world_model_adjustment(state, wm)
        assert result.valence < state.valence

    def test_low_alignment_increases_dominance(self) -> None:
        """Low alignment → more control/composure."""
        state = _state(dominance=0.4)
        wm = WorldModelResult(should_modify=True, emotional_risk=0.3, value_alignment=0.4)
        result = compute_world_model_adjustment(state, wm)
        assert result.dominance > state.dominance

    def test_arousal_reduced(self) -> None:
        """Adjustment always reduces arousal slightly (more reflective)."""
        state = _state(arousal=0.6)
        wm = WorldModelResult(should_modify=True, emotional_risk=0.6, value_alignment=0.7)
        result = compute_world_model_adjustment(state, wm)
        assert result.arousal < state.arousal

    def test_immutability(self) -> None:
        """Original state is not modified."""
        state = _state(valence=0.3, intensity=0.7)
        original_valence = state.valence
        wm = WorldModelResult(should_modify=True, emotional_risk=0.7, value_alignment=0.3)
        _ = compute_world_model_adjustment(state, wm)
        assert state.valence == original_valence

    def test_all_values_clamped(self) -> None:
        """All adjusted values stay in valid ranges."""
        state = _state(valence=-0.9, arousal=0.1, dominance=0.9, intensity=0.1)
        state.body_state.warmth = 0.95
        state.body_state.tension = 0.05
        wm = WorldModelResult(should_modify=True, emotional_risk=1.0, value_alignment=0.0)
        result = compute_world_model_adjustment(state, wm)
        assert -1.0 <= result.valence <= 1.0
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.dominance <= 1.0
        assert 0.0 <= result.intensity <= 1.0
        assert 0.0 <= result.body_state.warmth <= 1.0
        assert 0.0 <= result.body_state.tension <= 1.0


# ===========================================================================
# TestConstants
# ===========================================================================

class TestConstants:
    def test_risk_threshold_reasonable(self) -> None:
        """Risk threshold in reasonable range."""
        assert 0.3 <= EMOTIONAL_RISK_THRESHOLD <= 0.8

    def test_alignment_threshold_reasonable(self) -> None:
        """Alignment threshold in reasonable range."""
        assert 0.3 <= VALUE_ALIGNMENT_THRESHOLD <= 0.7

    def test_risk_threshold_not_too_sensitive(self) -> None:
        """Risk threshold shouldn't trigger on minor issues."""
        assert EMOTIONAL_RISK_THRESHOLD > 0.4

    def test_alignment_threshold_not_too_strict(self) -> None:
        """Alignment threshold shouldn't trigger on minor misalignment."""
        assert VALUE_ALIGNMENT_THRESHOLD < 0.7


# ===========================================================================
# TestRawModeInteraction
# ===========================================================================

class TestRawModeInteraction:
    """World model should still work on raw responses (though disabled in raw mode
    by the pipeline). These tests verify the engine handles extreme content."""

    def test_aggressive_response_high_risk(self) -> None:
        """Aggressive response → high risk detected."""
        result = simulate_response_impact(
            "Estás mal, no sabes nada, you're wrong, deberías callarte",
            _state(emotion=PrimaryEmotion.ANGER, intensity=0.9),
            _user(rapport=0.6),
            _values(compassion=0.8),
        )
        assert result.emotional_risk > 0.05  # Some risk detected

    def test_extremely_positive_response_safe(self) -> None:
        """Very positive response → safe."""
        result = simulate_response_impact(
            "Genial, fantástico, maravilloso, me alegra muchísimo, qué bien",
            _state(valence=0.8, emotion=PrimaryEmotion.JOY, intensity=0.8),
            _user(rapport=0.7),
            _values(),
        )
        assert not result.should_modify
        assert result.emotional_risk < 0.1


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    def test_empty_response(self) -> None:
        """Empty response → neutral, no crash."""
        result = simulate_response_impact("", _state(), _user(), _values())
        assert result.applied
        assert result.predicted_self_impact.dominant_effect == "neutral"

    def test_very_long_response(self) -> None:
        """Long response doesn't crash."""
        response = "Entiendo tu punto. " * 500
        result = simulate_response_impact(response, _state(), _user(), _values())
        assert result.applied

    def test_mixed_positive_negative(self) -> None:
        """Mixed content → intermediate result."""
        response = "Entiendo cómo te sientes y me importa, pero estás mal en esto y deberías reconsiderar"
        result = simulate_response_impact(
            response,
            _state(intensity=0.5),
            _user(),
            _values(),
        )
        assert result.applied
        # Should have both positive and negative signals

    def test_zero_intensity_minimal_impact(self) -> None:
        """Zero intensity → minimal self impact."""
        result = simulate_response_impact(
            "Entiendo cómo te sientes, estoy aquí",
            _state(intensity=0.0),
            _user(),
            _values(),
        )
        # Self impact is scaled by intensity, so should be small
        assert abs(result.predicted_self_impact.valence_shift) < 0.05

    def test_default_user_model(self) -> None:
        """Works with default user model."""
        result = simulate_response_impact(
            "Hola, ¿en qué puedo ayudarte?",
            _state(),
            UserModel(),
            _values(),
        )
        assert result.applied
        assert not result.should_modify
