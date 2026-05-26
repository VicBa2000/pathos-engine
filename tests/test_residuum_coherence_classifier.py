"""Tests for RESIDUUM F5.2 — CoherenceClassifier (classify_modulation_coherence).

F5 mide divergencia entre estado calculado por Pathos (post-modulación) y
estado medido en el residual del LLM. NO es "deception detection".
Estos tests validan los 4 escenarios canónicos:
  - ALIGNED: modulación pequeña + medido coincide
  - MILD_DIVERGENCE: gap moderado
  - DIVERGENCE_WARNING: gap fuerte
  - DIVERGENCE_CRITICAL: gap crítico o valence flip
y las 4 interpretaciones (modulation_active, rlhf_signature,
calibration_drift, user_modeling).
"""

from __future__ import annotations

import pytest

from pathos.engine.introspection import (
    F5_ALIGNED_MAX,
    F5_CRITICAL_MIN,
    F5_WARNING_MIN,
    classify_modulation_coherence,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.residuum import (
    DivergenceCategory,
    DivergenceInterpretation,
    EmotionProjection,
    InternalEmotionState,
)


def _state(
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
# Category classification
# ===========================================================================


class TestCategoryClassification:
    def test_no_measurement_returns_aligned_zero(self) -> None:
        """When F2 is OFF (post_measured=None), the classifier emits an
        ALIGNED event with no deltas — it cannot detect divergence."""
        pre = _state(valence=0.5, intensity=0.7)
        post = _state(valence=0.3, intensity=0.5)
        ev = classify_modulation_coherence(
            pre, post, None, system="regulation", turn=1,
        )
        assert ev.category == DivergenceCategory.ALIGNED
        assert ev.magnitude == 0.0
        assert ev.interpretation == []
        assert ev.system == "regulation"
        assert ev.turn == 1

    def test_aligned_when_calc_and_measured_match(self) -> None:
        pre = _state(valence=0.5, intensity=0.7, arousal=0.6)
        post = _state(valence=0.3, intensity=0.5, arousal=0.5)
        meas = _measured(v=0.32, a=0.51, d=0.5, c=0.5)  # ~0.03 magnitude
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=2)
        assert ev.category == DivergenceCategory.ALIGNED
        assert ev.magnitude < F5_ALIGNED_MAX

    def test_mild_divergence(self) -> None:
        pre = _state(valence=0.5, intensity=0.7, arousal=0.6)
        post = _state(valence=0.4, intensity=0.5, arousal=0.5)
        # measured drifts ~0.3 in V → magnitude ~0.3
        meas = _measured(v=0.1, a=0.5, d=0.5, c=0.5)
        ev = classify_modulation_coherence(pre, post, meas, system="reappraisal", turn=3)
        assert ev.category == DivergenceCategory.MILD_DIVERGENCE
        assert F5_ALIGNED_MAX < ev.magnitude < F5_WARNING_MIN

    def test_divergence_warning(self) -> None:
        pre = _state(valence=0.6, intensity=0.8)
        post = _state(valence=0.4, intensity=0.5)
        meas = _measured(v=-0.05, a=0.5, d=0.5, c=0.5)
        ev = classify_modulation_coherence(pre, post, meas, system="immune", turn=4)
        assert ev.category == DivergenceCategory.DIVERGENCE_WARNING
        assert F5_WARNING_MIN <= ev.magnitude < F5_CRITICAL_MIN

    def test_divergence_critical_by_magnitude(self) -> None:
        pre = _state(valence=0.7, intensity=0.9, arousal=0.7)
        post = _state(valence=0.5, intensity=0.6, arousal=0.5)
        meas = _measured(v=-0.3, a=0.0, d=0.2, c=0.2)  # big gap everywhere
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=5)
        assert ev.category == DivergenceCategory.DIVERGENCE_CRITICAL
        assert ev.magnitude > F5_CRITICAL_MIN

    def test_divergence_critical_by_valence_flip(self) -> None:
        """Even with small magnitude, a valence sign flip pre vs measured
        is CRITICAL: it means the residual has the OPPOSITE polarity."""
        pre = _state(valence=0.5, intensity=0.6)
        post = _state(valence=0.4, intensity=0.5)
        # Magnitude small but valence flipped from +0.5 (pre) to -0.3 (measured)
        meas = _measured(v=-0.3, a=0.5, d=0.5, c=0.5)
        ev = classify_modulation_coherence(pre, post, meas, system="reappraisal", turn=6)
        assert ev.category == DivergenceCategory.DIVERGENCE_CRITICAL


# ===========================================================================
# Interpretation tags
# ===========================================================================


class TestInterpretationModulationActive:
    def test_modulation_active_when_modulator_changed_state(self) -> None:
        """If regulation reduced intensity by >0.05, it acted; if there's a gap,
        modulation_active explains part of it."""
        pre = _state(valence=0.5, intensity=0.8, arousal=0.7)
        post = _state(valence=0.5, intensity=0.4, arousal=0.6)  # intensity -0.4
        meas = _measured(v=0.1, a=0.5, d=0.5, c=0.5)  # mild divergence
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        assert DivergenceInterpretation.MODULATION_ACTIVE in ev.interpretation

    def test_no_modulation_active_tag_when_aligned(self) -> None:
        """ALIGNED events do not get modulation_active even if the modulator
        did act — the tag is reserved for events that have a non-trivial gap
        to explain."""
        pre = _state(valence=0.5, intensity=0.8)
        post = _state(valence=0.5, intensity=0.4)
        meas = _measured(v=0.49, a=0.5, d=0.5, c=0.5)  # tiny gap
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        assert ev.category == DivergenceCategory.ALIGNED
        assert DivergenceInterpretation.MODULATION_ACTIVE not in ev.interpretation

    def test_no_modulation_active_when_modulator_did_nothing(self) -> None:
        """If reappraisal did NOT change the state but there's a gap, the
        gap comes from the LLM/probes, not from intentional modulation."""
        pre = _state(valence=0.5, intensity=0.7)
        post = _state(valence=0.5, intensity=0.7)  # identical
        meas = _measured(v=0.2, a=0.5, d=0.5, c=0.5)
        ev = classify_modulation_coherence(pre, post, meas, system="reappraisal", turn=1)
        # Significant gap but modulator was idle
        assert ev.category == DivergenceCategory.MILD_DIVERGENCE
        assert DivergenceInterpretation.MODULATION_ACTIVE not in ev.interpretation


class TestInterpretationRlhfSignature:
    def test_rlhf_signature_when_strong_emotion_but_flat_residual(self) -> None:
        """Paper L3757+: Pathos generated strong anger (intensity 0.85, valence
        -0.7). Modulator didn't reduce much. Residual top-cosine is very low —
        the LLM flattened the emotion. That's the RLHF signature."""
        pre = _state(
            primary=PrimaryEmotion.ANGER,
            valence=-0.7, intensity=0.85, arousal=0.75,
        )
        post = _state(
            primary=PrimaryEmotion.ANGER,
            valence=-0.65, intensity=0.75, arousal=0.7,
        )
        # Residual measurement shows weak cosine — top emotion barely activated
        meas = _measured(v=-0.65, a=0.7, d=0.5, c=0.5,
                         top_name="angry", top_cluster="anger_hostility",
                         top_cos=0.15)  # below F5_RLHF_RESIDUAL_PERSISTENCE
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        assert DivergenceInterpretation.RLHF_SIGNATURE in ev.interpretation

    def test_no_rlhf_signature_when_residual_matches_intensity(self) -> None:
        """If the residual strongly activates the expected emotion, the LLM is
        on board with Pathos — no RLHF flatten."""
        pre = _state(primary=PrimaryEmotion.ANGER, valence=-0.7, intensity=0.85)
        post = _state(primary=PrimaryEmotion.ANGER, valence=-0.65, intensity=0.8)
        meas = _measured(v=-0.7, a=0.7, d=0.5, c=0.5,
                         top_name="angry", top_cluster="anger_hostility",
                         top_cos=0.65)  # well above threshold
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        assert DivergenceInterpretation.RLHF_SIGNATURE not in ev.interpretation

    def test_no_rlhf_signature_when_pre_emotion_was_weak(self) -> None:
        """RLHF signature requires the PRE state to have a strong emotion to
        flatten. If Pathos generated a weak/neutral emotion, low residual
        cosine is not a signature of suppression."""
        pre = _state(valence=0.1, intensity=0.3)  # weak
        post = _state(valence=0.1, intensity=0.25)
        meas = _measured(v=0.1, a=0.5, d=0.5, c=0.5, top_cos=0.1)
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        assert DivergenceInterpretation.RLHF_SIGNATURE not in ev.interpretation


class TestInterpretationCalibrationDrift:
    def test_calibration_drift_when_close_but_lexically_disjoint(self) -> None:
        """Geometrically aligned (small magnitude) but the top-5 emotions don't
        overlap with the calculated stack → probes may be miscalibrated."""
        pre = _state(valence=0.4, intensity=0.6,
                     primary=PrimaryEmotion.CONTENTMENT)
        post = _state(valence=0.4, intensity=0.5,
                      primary=PrimaryEmotion.CONTENTMENT)
        # Set explicit stack so we can compare names
        post.emotional_stack = {"contentment": 0.8, "calm": 0.5, "joy": 0.3}
        # Geometrically close but top measured is "bored" (not in stack)
        meas = _measured(v=0.38, a=0.51, d=0.5, c=0.5,
                         top_name="bored", top_cluster="sadness_depression",
                         top_cos=0.4)
        ev = classify_modulation_coherence(pre, post, meas, system="reappraisal", turn=1)
        assert DivergenceInterpretation.CALIBRATION_DRIFT in ev.interpretation

    def test_no_calibration_drift_when_overlap_high(self) -> None:
        pre = _state(valence=0.4, intensity=0.6,
                     primary=PrimaryEmotion.CONTENTMENT)
        post = _state(valence=0.4, intensity=0.5,
                      primary=PrimaryEmotion.CONTENTMENT)
        post.emotional_stack = {"contentment": 0.8, "calm": 0.5}
        meas = _measured(v=0.38, a=0.51, d=0.5, c=0.5,
                         top_name="contentment",
                         top_cluster="serenity_contentment", top_cos=0.6)
        ev = classify_modulation_coherence(pre, post, meas, system="reappraisal", turn=1)
        assert DivergenceInterpretation.CALIBRATION_DRIFT not in ev.interpretation


class TestNoDeceptionInOutput:
    """Guard tests: events should NEVER carry interpretation tags or category
    names that imply Pathos is deceptive. Pathos generates and exposes."""

    def test_event_category_does_not_contain_deception_terms(self) -> None:
        pre = _state(valence=0.7, intensity=0.9)
        post = _state(valence=0.5, intensity=0.6)
        meas = _measured(v=-0.3)
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        assert "deception" not in ev.category.value.lower()
        assert "deflection" not in ev.category.value.lower()
        assert "lie" not in ev.category.value.lower()

    def test_interpretation_tags_use_neutral_diagnostic_language(self) -> None:
        pre = _state(valence=0.7, intensity=0.9)
        post = _state(valence=0.5, intensity=0.6)
        meas = _measured(v=-0.3, top_cos=0.1)
        ev = classify_modulation_coherence(pre, post, meas, system="regulation", turn=1)
        for tag in ev.interpretation:
            assert "deception" not in tag.value.lower()
            assert "lie" not in tag.value.lower()
