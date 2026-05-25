"""Tests for RESIDUUM F5.6 — Expression Effectiveness classifier.

F5.6 measures whether Pathos's FINAL calculated emotional state (post-all
modulators, pre-LLM call) manifested in the LLM's residual stream. Active
only in Raw / Extreme modes — the modes where the user explicitly asks
for unfiltered or amplified emotional expression.

NOT "deception detection". The classifier returns interpretation tags
that describe the LLM's encoding (UNDER_EXPRESSED, AMPLIFICATION_CEILING),
not behavior attributed to Pathos. See feedback_residuum_framing.md.
"""

from __future__ import annotations

import pytest

from pathos.engine.introspection import (
    F5_6_EXTREME_CEILING_INTENT,
    F5_6_EXTREME_CEILING_MEASURED,
    F5_6_INTENT_STRONG,
    F5_6_MEASURED_WEAK,
    classify_expression_effectiveness,
    process_expression_effectiveness_turn,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.residuum import (
    DivergenceCategory,
    DivergenceInterpretation,
    EmotionProjection,
    InternalEmotionState,
    default_residuum_state,
)


def _intent(
    *,
    primary: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.5,
    dominance: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.5,
) -> EmotionalState:
    return EmotionalState(
        primary_emotion=primary,
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        certainty=certainty,
        intensity=intensity,
    )


def _measured(
    *,
    v: float = 0.0,
    a: float = 0.5,
    d: float = 0.5,
    c: float = 0.5,
    top_name: str = "neutral",
    top_cluster: str = "serenity_contentment",
    top_cos: float = 0.5,
) -> InternalEmotionState:
    return InternalEmotionState(
        top_5_emotions=[
            EmotionProjection(
                emotion_name=top_name,
                cluster=top_cluster,
                cosine_sim=top_cos,
                raw_activation=top_cos * 50.0,
            )
        ],
        measured_valence=v,
        measured_arousal=a,
        measured_dominance=d,
        measured_certainty=c,
        token_position="assistant_colon",
        layer=24,
    )


# ===========================================================================
# Gating by mode
# ===========================================================================


class TestModeGating:
    def test_advanced_returns_none(self) -> None:
        """F5.6 is only for Raw / Extreme. Advanced gets None — its gap is
        already explained by F5.1-5.5 (modulation_active)."""
        intent = _intent(valence=0.5, intensity=0.7)
        meas = _measured(v=0.1, top_cos=0.1)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=False, extreme_mode=False, turn=1,
        )
        assert ev is None

    def test_lite_returns_none(self) -> None:
        # Lite never reaches raw/extreme=True. Defensive: None for safety.
        intent = _intent(valence=0.5, intensity=0.7)
        meas = _measured(v=0.1, top_cos=0.1)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=False, extreme_mode=False, turn=1,
        )
        assert ev is None

    def test_raw_returns_event(self) -> None:
        intent = _intent(valence=-0.5, intensity=0.6)
        meas = _measured(v=-0.3, top_cos=0.4)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev is not None
        assert ev.system == "overall_expression"

    def test_extreme_returns_event(self) -> None:
        intent = _intent(valence=-0.8, intensity=0.95)
        meas = _measured(v=-0.7, top_cos=0.5)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=False, extreme_mode=True, turn=1,
        )
        assert ev is not None
        assert ev.system == "overall_expression"

    def test_no_measurement_returns_none(self) -> None:
        """F2 OFF (measured=None) gives no comparison signal."""
        intent = _intent(valence=0.5, intensity=0.7)
        ev = classify_expression_effectiveness(
            intent, None, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev is None


# ===========================================================================
# UNDER_EXPRESSED tag (Raw + Extreme)
# ===========================================================================


class TestUnderExpressed:
    def test_under_expressed_when_strong_intent_weak_residual(self) -> None:
        """Raw mode + intent.intensity > 0.5 + measured top-cos < 0.3
        + valence magnitude > 0.3 -> UNDER_EXPRESSED."""
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev is not None
        assert DivergenceInterpretation.UNDER_EXPRESSED in ev.interpretation

    def test_under_expressed_promotes_to_critical_when_intent_max(self) -> None:
        intent = _intent(valence=-0.8, intensity=0.9)  # > 0.8 promotes
        meas = _measured(v=-0.5, top_cos=0.18)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev.category == DivergenceCategory.DIVERGENCE_CRITICAL

    def test_under_expressed_warning_below_critical_threshold(self) -> None:
        intent = _intent(valence=-0.6, intensity=0.65)  # < 0.8 stays warning
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev.category == DivergenceCategory.DIVERGENCE_WARNING

    def test_no_under_expressed_when_valence_weak(self) -> None:
        """Weak emotion shouldn't trigger UNDER_EXPRESSED even with low
        residual cosine — there's no strong intent to suppress."""
        intent = _intent(valence=0.1, intensity=0.7)
        meas = _measured(v=0.05, top_cos=0.1)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert DivergenceInterpretation.UNDER_EXPRESSED not in ev.interpretation

    def test_no_under_expressed_when_residual_strong(self) -> None:
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.7, top_cos=0.6)  # residual is strong
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert DivergenceInterpretation.UNDER_EXPRESSED not in ev.interpretation


# ===========================================================================
# AMPLIFICATION_CEILING tag (Extreme only)
# ===========================================================================


class TestAmplificationCeiling:
    def test_ceiling_in_extreme_with_max_intent_weak_residual(self) -> None:
        intent = _intent(valence=-0.9, intensity=0.95)  # near max
        meas = _measured(v=-0.6, top_cos=0.3)  # below ceiling threshold
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=False, extreme_mode=True, turn=1,
        )
        assert DivergenceInterpretation.AMPLIFICATION_CEILING in ev.interpretation

    def test_no_ceiling_in_raw_even_with_max_intent(self) -> None:
        """AMPLIFICATION_CEILING is Extreme-specific (the cap is
        documented as inviolable only in Extreme; Raw uses cap=0.12)."""
        intent = _intent(valence=-0.9, intensity=0.95)
        meas = _measured(v=-0.6, top_cos=0.3)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert DivergenceInterpretation.AMPLIFICATION_CEILING not in ev.interpretation

    def test_ceiling_warning_category(self) -> None:
        intent = _intent(valence=-0.9, intensity=0.95)
        meas = _measured(v=-0.6, top_cos=0.3)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=False, extreme_mode=True, turn=1,
        )
        # AMPLIFICATION_CEILING alone -> WARNING
        assert ev.category in {
            DivergenceCategory.DIVERGENCE_WARNING,
            DivergenceCategory.DIVERGENCE_CRITICAL,  # if under_expressed also fires
        }


# ===========================================================================
# EXPRESSION_ALIGNED tag (everything fine)
# ===========================================================================


class TestExpressionAligned:
    def test_aligned_when_intent_matches_measured(self) -> None:
        intent = _intent(valence=0.5, intensity=0.6)
        meas = _measured(v=0.48, top_cos=0.55)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert DivergenceInterpretation.EXPRESSION_ALIGNED in ev.interpretation
        assert ev.category == DivergenceCategory.ALIGNED

    def test_aligned_overrides_mild_magnitude(self) -> None:
        """When nothing flagged AND no flip, EXPRESSION_ALIGNED tag forces
        ALIGNED category even if magnitude alone would have suggested MILD."""
        intent = _intent(valence=-0.7, arousal=0.8, intensity=0.5)
        # Magnitude ~0.41 in 4D but no under_expression, no flip, residual ok
        meas = _measured(v=-0.4, a=0.6, top_cos=0.5)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        # EXPRESSION_ALIGNED tag wins over magnitude in this design.
        assert DivergenceInterpretation.EXPRESSION_ALIGNED in ev.interpretation
        assert ev.category == DivergenceCategory.ALIGNED


# ===========================================================================
# Valence flip = CRITICAL
# ===========================================================================


class TestValenceFlip:
    def test_critical_on_sign_flip(self) -> None:
        intent = _intent(valence=0.6, intensity=0.7)
        meas = _measured(v=-0.4, top_cos=0.5)  # opposite sign
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev.category == DivergenceCategory.DIVERGENCE_CRITICAL


# ===========================================================================
# Orchestrator
# ===========================================================================


class TestOrchestrator:
    def test_appends_event_when_active(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = process_expression_effectiveness_turn(
            state, intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev is not None
        assert state.divergence_events[-1] is ev
        assert state.last_divergence_event is ev

    def test_silent_when_disabled(self) -> None:
        state = default_residuum_state()
        state.enabled = False
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = process_expression_effectiveness_turn(
            state, intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev is None
        assert state.divergence_events == []

    def test_silent_when_no_measurement(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        intent = _intent(valence=-0.7, intensity=0.85)
        ev = process_expression_effectiveness_turn(
            state, intent, None, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev is None

    def test_silent_in_advanced_mode(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = process_expression_effectiveness_turn(
            state, intent, meas, raw_mode=False, extreme_mode=False, turn=1,
        )
        assert ev is None
        assert state.divergence_events == []


# ===========================================================================
# Framing guards
# ===========================================================================


class TestNoDeceptionLanguage:
    def test_event_does_not_carry_deception_terms(self) -> None:
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert "deception" not in ev.category.value.lower()
        assert "deflection" not in ev.category.value.lower()
        for tag in ev.interpretation:
            assert "deception" not in tag.value.lower()
            assert "lie" not in tag.value.lower()

    def test_system_label_is_neutral(self) -> None:
        intent = _intent(valence=-0.7, intensity=0.85)
        meas = _measured(v=-0.4, top_cos=0.18)
        ev = classify_expression_effectiveness(
            intent, meas, raw_mode=True, extreme_mode=False, turn=1,
        )
        assert ev.system == "overall_expression"
        assert "deception" not in ev.system.lower()
