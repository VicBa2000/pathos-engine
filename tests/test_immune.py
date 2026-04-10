"""Tests para Emotional Immune System - proteccion contra trauma emocional."""

import pytest

from pathos.engine.immune import (
    COMPARTMENT_STREAK_THRESHOLD,
    DISSOCIATION_STREAK_THRESHOLD,
    NUMBING_STREAK_THRESHOLD,
    apply_immune_protection,
    get_immune_prompt_info,
    update_immune_state,
    _extract_topic_keywords,
    _topic_overlap,
)
from pathos.models.emotion import BodyState, EmotionalState, PrimaryEmotion
from pathos.models.immune import ImmuneState, ProtectionMode, default_immune_state


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_state(
    valence: float = 0.0,
    arousal: float = 0.5,
    intensity: float = 0.5,
    openness: float = 0.5,
    warmth: float = 0.5,
    tension: float = 0.3,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        body_state=BodyState(openness=openness, warmth=warmth, tension=tension),
    )


def _traumatic_state(intensity: float = 0.85) -> EmotionalState:
    """Estado de trauma: alta intensidad + valence negativo."""
    return _make_state(valence=-0.7, arousal=0.8, intensity=intensity)


def _positive_state() -> EmotionalState:
    """Estado positivo (no trauma)."""
    return _make_state(valence=0.5, arousal=0.5, intensity=0.5)


def _simulate_trauma_streak(turns: int, stimulus: str = "esto es horrible y doloroso") -> ImmuneState:
    """Simula N turnos consecutivos de trauma."""
    immune = default_immune_state()
    state = _traumatic_state()
    for _ in range(turns):
        immune = update_immune_state(immune, state, stimulus)
    return immune


# ── Model tests ─────────────────────────────────────────────────────────

class TestImmuneModel:
    def test_default_state(self) -> None:
        state = default_immune_state()
        assert state.protection_mode == ProtectionMode.NONE
        assert state.negative_streak == 0
        assert state.protection_strength == 0.0
        assert state.reactivity_dampening == 0.0
        assert state.compartmentalized_topics == []
        assert state.total_activations == 0

    def test_protection_mode_enum(self) -> None:
        assert ProtectionMode.NONE.value == "none"
        assert ProtectionMode.NUMBING.value == "numbing"
        assert ProtectionMode.DISSOCIATION.value == "dissociation"
        assert ProtectionMode.COMPARTMENTALIZATION.value == "compartmentalization"


# ── Keyword extraction ──────────────────────────────────────────────────

class TestKeywordExtraction:
    def test_extract_keywords(self) -> None:
        keywords = _extract_topic_keywords("me siento terrible por la muerte de mi abuela")
        assert len(keywords) > 0
        assert "siento" in keywords
        assert "terrible" in keywords

    def test_short_words_excluded(self) -> None:
        keywords = _extract_topic_keywords("a de la en por")
        assert keywords == []

    def test_max_5_keywords(self) -> None:
        keywords = _extract_topic_keywords("alpha bravo charlie delta echo foxtrot golf hotel")
        assert len(keywords) <= 5

    def test_topic_overlap_detected(self) -> None:
        assert _topic_overlap("hablemos de la muerte de mi abuela", ["muerte", "abuela"])

    def test_topic_overlap_not_detected(self) -> None:
        assert not _topic_overlap("que buen dia hace hoy", ["muerte", "abuela"])

    def test_topic_overlap_needs_2_matches(self) -> None:
        """Necesita al menos 2 keywords para ser considerado overlap."""
        assert not _topic_overlap("mi abuela cocina bien", ["muerte", "abuela"])


# ── Trauma detection ────────────────────────────────────────────────────

class TestTraumaDetection:
    def test_no_trauma_no_streak(self) -> None:
        immune = default_immune_state()
        state = _positive_state()
        new = update_immune_state(immune, state, "hola como estas")
        assert new.negative_streak == 0
        assert new.protection_mode == ProtectionMode.NONE

    def test_trauma_increments_streak(self) -> None:
        immune = default_immune_state()
        state = _traumatic_state()
        new = update_immune_state(immune, state, "esto es terrible")
        assert new.negative_streak == 1

    def test_consecutive_trauma_builds_streak(self) -> None:
        immune = _simulate_trauma_streak(4)
        assert immune.negative_streak == 4

    def test_positive_turn_decrements_streak(self) -> None:
        immune = _simulate_trauma_streak(3)
        assert immune.negative_streak == 3
        new = update_immune_state(immune, _positive_state(), "todo bien")
        assert new.negative_streak == 2

    def test_streak_decays_to_zero(self) -> None:
        immune = _simulate_trauma_streak(2)
        for _ in range(3):
            immune = update_immune_state(immune, _positive_state(), "todo bien")
        assert immune.negative_streak == 0

    def test_low_intensity_not_trauma(self) -> None:
        """Intensidad debajo del umbral no cuenta como trauma."""
        immune = default_immune_state()
        state = _make_state(valence=-0.8, intensity=0.5)  # intense but below threshold
        new = update_immune_state(immune, state, "esto es malo")
        assert new.negative_streak == 0

    def test_positive_valence_not_trauma(self) -> None:
        """Valence positivo no cuenta como trauma aun con alta intensidad."""
        immune = default_immune_state()
        state = _make_state(valence=0.5, intensity=0.9)
        new = update_immune_state(immune, state, "esto es increible")
        assert new.negative_streak == 0

    def test_peak_intensity_tracked(self) -> None:
        immune = default_immune_state()
        state1 = _traumatic_state(intensity=0.75)
        immune = update_immune_state(immune, state1, "malo")
        state2 = _traumatic_state(intensity=0.95)
        immune = update_immune_state(immune, state2, "peor")
        assert immune.peak_negative_intensity == 0.95


# ── Protection activation ──────────────────────────────────────────────

class TestProtectionActivation:
    def test_numbing_activates_at_threshold(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        assert immune.protection_mode == ProtectionMode.NUMBING
        assert immune.protection_strength > 0
        assert immune.reactivity_dampening > 0

    def test_numbing_not_active_before_threshold(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD - 1)
        assert immune.protection_mode == ProtectionMode.NONE

    def test_dissociation_activates_at_threshold(self) -> None:
        immune = _simulate_trauma_streak(DISSOCIATION_STREAK_THRESHOLD)
        assert immune.protection_mode == ProtectionMode.DISSOCIATION
        assert immune.protection_strength > immune.reactivity_dampening

    def test_compartmentalization_activates_at_threshold(self) -> None:
        immune = _simulate_trauma_streak(COMPARTMENT_STREAK_THRESHOLD)
        assert immune.protection_mode == ProtectionMode.COMPARTMENTALIZATION
        assert len(immune.compartmentalized_topics) > 0

    def test_protection_strength_increases_with_streak(self) -> None:
        immune3 = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        immune6 = _simulate_trauma_streak(DISSOCIATION_STREAK_THRESHOLD)
        assert immune6.protection_strength > immune3.protection_strength

    def test_total_activations_counted(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        assert immune.total_activations == 1

    def test_compartmentalized_topics_limited(self) -> None:
        immune = _simulate_trauma_streak(COMPARTMENT_STREAK_THRESHOLD + 5)
        assert len(immune.compartmentalized_topics) <= 5


# ── Protection application ──────────────────────────────────────────────

class TestProtectionApplication:
    def test_no_protection_no_change(self) -> None:
        immune = default_immune_state()
        state = _traumatic_state()
        protected = apply_immune_protection(state, immune, "test")
        assert protected.intensity == state.intensity
        assert protected.valence == state.valence

    def test_numbing_reduces_intensity(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD + 1)
        state = _traumatic_state()
        protected = apply_immune_protection(state, immune, "test")
        assert protected.intensity < state.intensity

    def test_numbing_reduces_arousal(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD + 1)
        state = _traumatic_state()
        protected = apply_immune_protection(state, immune, "test")
        assert protected.arousal < state.arousal

    def test_dissociation_moves_valence_toward_zero(self) -> None:
        immune = _simulate_trauma_streak(DISSOCIATION_STREAK_THRESHOLD + 1)
        state = _traumatic_state()
        protected = apply_immune_protection(state, immune, "test")
        assert abs(protected.valence) < abs(state.valence)

    def test_dissociation_reduces_openness(self) -> None:
        immune = _simulate_trauma_streak(DISSOCIATION_STREAK_THRESHOLD + 1)
        state = _make_state(valence=-0.8, intensity=0.9, openness=0.7)
        protected = apply_immune_protection(state, immune, "test")
        assert protected.body_state.openness < state.body_state.openness

    def test_compartmentalization_strong_on_topic(self) -> None:
        stimulus = "esto es horrible y doloroso"
        immune = _simulate_trauma_streak(COMPARTMENT_STREAK_THRESHOLD)
        state = _traumatic_state()
        # Same topic as the trauma
        protected = apply_immune_protection(state, immune, stimulus)
        assert protected.intensity < state.intensity * 0.5

    def test_compartmentalization_mild_off_topic(self) -> None:
        immune = _simulate_trauma_streak(COMPARTMENT_STREAK_THRESHOLD)
        state = _make_state(valence=-0.5, intensity=0.6)
        # Different topic entirely
        protected = apply_immune_protection(state, immune, "el clima esta nublado hoy")
        # Should still have some effect but much less
        assert protected.intensity >= state.intensity * 0.6

    def test_all_values_in_range(self) -> None:
        """Protected state values always within valid ranges."""
        for streak in [3, 6, 8, 10]:
            immune = _simulate_trauma_streak(streak)
            state = _traumatic_state()
            protected = apply_immune_protection(state, immune, "test")
            assert -1 <= protected.valence <= 1
            assert 0 <= protected.arousal <= 1
            assert 0 <= protected.intensity <= 1
            assert 0 <= protected.body_state.openness <= 1
            assert 0 <= protected.body_state.warmth <= 1


# ── Recovery ────────────────────────────────────────────────────────────

class TestRecovery:
    def test_recovery_starts_on_positive_turn(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        assert immune.protection_mode == ProtectionMode.NUMBING
        recovered = update_immune_state(immune, _positive_state(), "todo bien")
        assert recovered.recovery_turns == 1

    def test_recovery_reduces_strength(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD + 2)
        strength_before = immune.protection_strength
        recovered = update_immune_state(immune, _positive_state(), "todo bien")
        assert recovered.protection_strength < strength_before

    def test_recovery_reduces_dampening(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD + 2)
        dampening_before = immune.reactivity_dampening
        recovered = update_immune_state(immune, _positive_state(), "todo bien")
        assert recovered.reactivity_dampening < dampening_before

    def test_full_recovery_returns_to_none(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        # Enough positive turns to fully recover
        for _ in range(20):
            immune = update_immune_state(immune, _positive_state(), "todo genial")
        assert immune.protection_mode == ProtectionMode.NONE
        assert immune.protection_strength == 0.0
        assert immune.reactivity_dampening == 0.0

    def test_trauma_during_recovery_re_escalates(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        # One positive turn
        immune = update_immune_state(immune, _positive_state(), "ok")
        assert immune.recovery_turns == 1
        # Trauma returns
        immune = update_immune_state(immune, _traumatic_state(), "otra vez mal")
        assert immune.recovery_turns == 0
        assert immune.negative_streak > 0


# ── Behavior modifier info ──────────────────────────────────────────────

class TestPromptInfo:
    def test_no_protection_returns_none(self) -> None:
        assert get_immune_prompt_info(default_immune_state()) is None

    def test_numbing_returns_info(self) -> None:
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD)
        info = get_immune_prompt_info(immune)
        assert info is not None
        assert "numbing" in info.lower() or "proteccion" in info.lower()

    def test_dissociation_returns_info(self) -> None:
        immune = _simulate_trauma_streak(DISSOCIATION_STREAK_THRESHOLD)
        info = get_immune_prompt_info(immune)
        assert info is not None
        assert "dissociation" in info.lower() or "desconect" in info.lower()

    def test_compartmentalization_returns_info(self) -> None:
        immune = _simulate_trauma_streak(COMPARTMENT_STREAK_THRESHOLD)
        info = get_immune_prompt_info(immune)
        assert info is not None
        assert "compartiment" in info.lower() or "compartmental" in info.lower()

    def test_recovery_mentioned(self) -> None:
        # Need enough turns for protection_strength to survive one recovery step
        immune = _simulate_trauma_streak(NUMBING_STREAK_THRESHOLD + 3)
        immune = update_immune_state(immune, _positive_state(), "ok")
        info = get_immune_prompt_info(immune)
        assert info is not None
        assert "recovery" in info.lower()

    def test_transparency_language(self) -> None:
        """All protection messages include transparency language."""
        for streak in [NUMBING_STREAK_THRESHOLD, DISSOCIATION_STREAK_THRESHOLD, COMPARTMENT_STREAK_THRESHOLD]:
            immune = _simulate_trauma_streak(streak)
            info = get_immune_prompt_info(immune)
            assert info is not None
            # Should mention it's a protection mechanism
            assert "protec" in info.lower()
