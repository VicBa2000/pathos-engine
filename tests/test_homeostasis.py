"""Tests para Homeostasis Emocional (Fase 3 - baseline shift avanzado)."""

from pathos.engine.homeostasis import (
    BASE_SHIFT_RATE,
    BASELINE_SHIFT_THRESHOLD,
    MAX_BASELINE_DRIFT,
    RECOVERY_START_TURNS,
    _compute_shift_rate,
    regulate,
)
from pathos.models.emotion import BodyState, EmotionalState, Mood, PrimaryEmotion


def _make_state(
    valence: float = 0.0,
    arousal: float = 0.5,
    intensity: float = 0.5,
    primary: PrimaryEmotion = PrimaryEmotion.JOY,
    baseline_valence: float = 0.1,
    baseline_arousal: float = 0.3,
    energy: float = 0.5,
    tension: float = 0.5,
    stability: float = 0.7,
    extreme_event_count: int = 0,
    turns_since_extreme: int = 0,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        primary_emotion=primary,
        body_state=BodyState(energy=energy, tension=tension),
        mood=Mood(
            baseline_valence=baseline_valence,
            baseline_arousal=baseline_arousal,
            stability=stability,
            extreme_event_count=extreme_event_count,
            turns_since_extreme=turns_since_extreme,
            original_baseline_valence=0.1,
            original_baseline_arousal=0.3,
        ),
    )


# --- Tests de Decay (sin cambios de logica) ---


class TestDecay:
    def test_valence_decays_toward_baseline(self) -> None:
        state = _make_state(valence=0.8, baseline_valence=0.1)
        regulated = regulate(state, turns_elapsed=1)
        assert abs(regulated.valence - 0.1) < abs(0.8 - 0.1)

    def test_arousal_decays_toward_baseline(self) -> None:
        state = _make_state(arousal=0.9, baseline_arousal=0.3)
        regulated = regulate(state, turns_elapsed=1)
        assert abs(regulated.arousal - 0.3) < abs(0.9 - 0.3)

    def test_intensity_decays(self) -> None:
        state = _make_state(intensity=0.7)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.intensity < 0.7

    def test_multiple_turns_decay_more(self) -> None:
        state = _make_state(valence=0.8, baseline_valence=0.1)
        reg1 = regulate(state, turns_elapsed=1)
        reg3 = regulate(state, turns_elapsed=3)
        assert abs(reg3.valence - 0.1) < abs(reg1.valence - 0.1)

    def test_low_intensity_becomes_neutral(self) -> None:
        state2 = _make_state(intensity=0.08, primary=PrimaryEmotion.ANGER)
        regulated2 = regulate(state2, turns_elapsed=1)
        assert regulated2.primary_emotion == PrimaryEmotion.NEUTRAL


# --- Tests de Baseline Shift Basico ---


class TestBaselineShift:
    def test_extreme_intensity_shifts_baseline(self) -> None:
        state = _make_state(
            valence=-0.8, intensity=0.9, baseline_valence=0.1,
            primary=PrimaryEmotion.ANGER,
        )
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.baseline_valence < 0.1

    def test_moderate_intensity_no_shift(self) -> None:
        state = _make_state(
            valence=-0.8, intensity=0.5, baseline_valence=0.1,
            primary=PrimaryEmotion.SADNESS,
        )
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.baseline_valence == 0.1


# --- Tests de Sensitizacion ---


class TestSensitization:
    """Eventos extremos repetidos = shift mas rapido."""

    def test_shift_rate_increases_with_extreme_count(self) -> None:
        rate_0 = _compute_shift_rate(extreme_count=0, stability=0.5)
        rate_5 = _compute_shift_rate(extreme_count=5, stability=0.5)
        assert rate_5 > rate_0

    def test_shift_rate_capped(self) -> None:
        rate = _compute_shift_rate(extreme_count=100, stability=0.0)
        from pathos.engine.homeostasis import MAX_SHIFT_RATE
        # Rate nunca excede MAX_SHIFT_RATE * max_resistance
        assert rate <= MAX_SHIFT_RATE * 1.0

    def test_repeated_extremes_shift_more(self) -> None:
        # Primera experiencia extrema
        state1 = _make_state(
            valence=-0.9, intensity=0.95, baseline_valence=0.1,
            extreme_event_count=0,
        )
        reg1 = regulate(state1, turns_elapsed=1)
        shift1 = abs(reg1.mood.baseline_valence - 0.1)

        # Despues de 5 eventos extremos previos
        state5 = _make_state(
            valence=-0.9, intensity=0.95, baseline_valence=0.1,
            extreme_event_count=5,
        )
        reg5 = regulate(state5, turns_elapsed=1)
        shift5 = abs(reg5.mood.baseline_valence - 0.1)

        assert shift5 > shift1

    def test_extreme_event_count_increments(self) -> None:
        state = _make_state(intensity=0.9, extreme_event_count=3)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.extreme_event_count == 4

    def test_no_increment_below_threshold(self) -> None:
        state = _make_state(intensity=0.5, extreme_event_count=3)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.extreme_event_count == 3


# --- Tests de Stability como resistencia ---


class TestStabilityResistance:
    """Mayor stability = mas resistencia al baseline shift."""

    def test_high_stability_resists_shift(self) -> None:
        state_stable = _make_state(
            valence=-0.9, intensity=0.95, baseline_valence=0.1,
            stability=0.9,
        )
        state_unstable = _make_state(
            valence=-0.9, intensity=0.95, baseline_valence=0.1,
            stability=0.1,
        )
        reg_stable = regulate(state_stable, turns_elapsed=1)
        reg_unstable = regulate(state_unstable, turns_elapsed=1)

        shift_stable = abs(reg_stable.mood.baseline_valence - 0.1)
        shift_unstable = abs(reg_unstable.mood.baseline_valence - 0.1)
        assert shift_unstable > shift_stable

    def test_stability_decreases_on_extreme_event(self) -> None:
        state = _make_state(intensity=0.9, stability=0.7)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.stability < 0.7


# --- Tests de Max Drift ---


class TestMaxDrift:
    """El baseline no puede alejarse mas de MAX_BASELINE_DRIFT del original."""

    def test_baseline_respects_max_drift(self) -> None:
        # Simular muchos eventos extremos negativos
        state = _make_state(
            valence=-1.0, intensity=1.0, baseline_valence=0.1,
            extreme_event_count=50, stability=0.0,
        )
        regulated = regulate(state, turns_elapsed=1)
        drift = abs(regulated.mood.baseline_valence - 0.1)
        assert drift <= MAX_BASELINE_DRIFT + 0.001  # Small float tolerance


# --- Tests de Recovery ---


class TestRecovery:
    """El baseline se recupera gradualmente sin eventos extremos."""

    def test_no_recovery_before_threshold(self) -> None:
        state = _make_state(
            intensity=0.3, baseline_valence=-0.2,
            turns_since_extreme=RECOVERY_START_TURNS - 2,
        )
        regulated = regulate(state, turns_elapsed=1)
        # turns_since_extreme se incrementa pero sigue bajo el threshold
        assert regulated.mood.baseline_valence == -0.2  # No extreme, no shift

    def test_recovery_after_threshold(self) -> None:
        state = _make_state(
            intensity=0.3, baseline_valence=-0.2,
            turns_since_extreme=RECOVERY_START_TURNS + 1,
        )
        regulated = regulate(state, turns_elapsed=1)
        # Baseline deberia moverse ligeramente hacia el original (0.1)
        assert regulated.mood.baseline_valence > -0.2

    def test_stability_recovers(self) -> None:
        state = _make_state(
            intensity=0.3, stability=0.4,
            turns_since_extreme=RECOVERY_START_TURNS + 1,
        )
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.stability > 0.4

    def test_turns_since_extreme_increments(self) -> None:
        state = _make_state(intensity=0.3, turns_since_extreme=2)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.turns_since_extreme == 3

    def test_turns_since_extreme_resets_on_extreme(self) -> None:
        state = _make_state(intensity=0.9, turns_since_extreme=10)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.mood.turns_since_extreme == 0


# --- Tests de Body Regulation (sin cambios) ---


class TestBodyRegulation:
    def test_tension_decays(self) -> None:
        state = _make_state(tension=0.8)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.body_state.tension < 0.8

    def test_high_arousal_drains_energy(self) -> None:
        state = _make_state(arousal=0.9, energy=0.5)
        regulated = regulate(state, turns_elapsed=1)
        assert regulated.body_state.energy < 0.5


# --- Tests de Rangos ---


class TestRanges:
    def test_all_in_range_after_regulation(self) -> None:
        extremes = [
            _make_state(valence=1, arousal=1, intensity=1, energy=1, tension=1),
            _make_state(valence=-1, arousal=0, intensity=0, energy=0, tension=0),
            _make_state(
                valence=-1, arousal=1, intensity=1, stability=0,
                extreme_event_count=50, baseline_valence=-0.5,
            ),
        ]
        for state in extremes:
            regulated = regulate(state, turns_elapsed=5)
            assert -1 <= regulated.valence <= 1
            assert 0 <= regulated.arousal <= 1
            assert 0 <= regulated.intensity <= 1
            assert 0 <= regulated.body_state.energy <= 1
            assert 0 <= regulated.body_state.tension <= 1
            assert -1 <= regulated.mood.baseline_valence <= 1
            assert 0 <= regulated.mood.baseline_arousal <= 1
            assert 0 <= regulated.mood.stability <= 1
