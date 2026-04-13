"""Tests for Interoception — body-state feedback ascendente."""

from pathos.engine.interoception import (
    ENERGY_LOW,
    LOW_ENERGY_THRESHOLD_TURNS,
    MAX_PERTURBATION,
    TENSION_HIGH,
    TENSION_THRESHOLD_TURNS,
    WARMTH_HIGH,
    WARMTH_THRESHOLD_TURNS,
    InteroceptiveResult,
    InteroceptiveState,
    compute_interoceptive_feedback,
    update_interoceptive_state,
)
from pathos.models.emotion import BodyState


class TestUpdateInteroceptiveState:
    """Tests for duration tracking of body state."""

    def test_high_tension_increments(self) -> None:
        body = BodyState(tension=0.8)
        state = InteroceptiveState(high_tension_turns=2)
        new = update_interoceptive_state(state, body)
        assert new.high_tension_turns == 3

    def test_tension_below_threshold_resets(self) -> None:
        body = BodyState(tension=0.5)
        state = InteroceptiveState(high_tension_turns=5)
        new = update_interoceptive_state(state, body)
        assert new.high_tension_turns == 0

    def test_low_energy_increments(self) -> None:
        body = BodyState(energy=0.1)
        state = InteroceptiveState(low_energy_turns=1)
        new = update_interoceptive_state(state, body)
        assert new.low_energy_turns == 2

    def test_energy_above_threshold_resets(self) -> None:
        body = BodyState(energy=0.5)
        state = InteroceptiveState(low_energy_turns=4)
        new = update_interoceptive_state(state, body)
        assert new.low_energy_turns == 0

    def test_high_warmth_increments(self) -> None:
        body = BodyState(warmth=0.8)
        state = InteroceptiveState(high_warmth_turns=1)
        new = update_interoceptive_state(state, body)
        assert new.high_warmth_turns == 2

    def test_warmth_below_threshold_resets(self) -> None:
        body = BodyState(warmth=0.4)
        state = InteroceptiveState(high_warmth_turns=3)
        new = update_interoceptive_state(state, body)
        assert new.high_warmth_turns == 0

    def test_all_counters_independent(self) -> None:
        body = BodyState(tension=0.9, energy=0.1, warmth=0.8)
        state = InteroceptiveState()
        new = update_interoceptive_state(state, body)
        assert new.high_tension_turns == 1
        assert new.low_energy_turns == 1
        assert new.high_warmth_turns == 1

    def test_fresh_state_starts_at_zero(self) -> None:
        state = InteroceptiveState()
        assert state.high_tension_turns == 0
        assert state.low_energy_turns == 0
        assert state.high_warmth_turns == 0


class TestComputeInteroceptiveFeedback:
    """Tests for the feedback computation."""

    def test_no_feedback_below_threshold_turns(self) -> None:
        body = BodyState(tension=0.9)
        state = InteroceptiveState(high_tension_turns=2)  # Below TENSION_THRESHOLD_TURNS
        result = compute_interoceptive_feedback(state, body)
        assert not result.active

    def test_tension_feedback_at_threshold(self) -> None:
        body = BodyState(tension=0.85)
        state = InteroceptiveState(high_tension_turns=TENSION_THRESHOLD_TURNS)
        result = compute_interoceptive_feedback(state, body)
        assert result.active
        assert result.source == "tension"
        assert result.valence_delta < 0  # Tension → negative valence
        assert result.arousal_delta > 0  # Tension → increased arousal

    def test_low_energy_feedback_at_threshold(self) -> None:
        body = BodyState(energy=0.1)
        state = InteroceptiveState(low_energy_turns=LOW_ENERGY_THRESHOLD_TURNS)
        result = compute_interoceptive_feedback(state, body)
        assert result.active
        assert result.source == "low_energy"
        assert result.valence_delta < 0  # Low energy → sadness/apathy
        assert result.arousal_delta < 0  # Low energy → lower arousal

    def test_warmth_feedback_at_threshold(self) -> None:
        body = BodyState(warmth=0.85)
        state = InteroceptiveState(high_warmth_turns=WARMTH_THRESHOLD_TURNS)
        result = compute_interoceptive_feedback(state, body)
        assert result.active
        assert result.source == "warmth"
        assert result.valence_delta > 0  # Warmth → positive valence
        assert result.arousal_delta == 0.0  # Warmth doesn't affect arousal

    def test_perturbation_clamped(self) -> None:
        body = BodyState(tension=1.0)
        state = InteroceptiveState(high_tension_turns=20)
        result = compute_interoceptive_feedback(state, body)
        assert abs(result.valence_delta) <= MAX_PERTURBATION
        assert abs(result.arousal_delta) <= MAX_PERTURBATION

    def test_no_feedback_normal_body(self) -> None:
        body = BodyState(tension=0.3, energy=0.5, warmth=0.5)
        state = InteroceptiveState()
        result = compute_interoceptive_feedback(state, body)
        assert not result.active
        assert result.valence_delta == 0.0
        assert result.arousal_delta == 0.0

    def test_strongest_effect_wins(self) -> None:
        """When multiple body states are extreme, only the strongest applies."""
        body = BodyState(tension=0.95, energy=0.05, warmth=0.9)
        state = InteroceptiveState(
            high_tension_turns=5,
            low_energy_turns=5,
            high_warmth_turns=5,
        )
        result = compute_interoceptive_feedback(state, body)
        assert result.active
        # Should pick the strongest effect (tension or low_energy, not warmth)
        assert result.source in ("tension", "low_energy")

    def test_duration_factor_increases_effect(self) -> None:
        """Longer duration → stronger effect."""
        body = BodyState(tension=0.85)
        result_short = compute_interoceptive_feedback(
            InteroceptiveState(high_tension_turns=TENSION_THRESHOLD_TURNS), body,
        )
        result_long = compute_interoceptive_feedback(
            InteroceptiveState(high_tension_turns=TENSION_THRESHOLD_TURNS + 3), body,
        )
        assert abs(result_long.valence_delta) >= abs(result_short.valence_delta)

    def test_borderline_tension_exactly_at_threshold(self) -> None:
        body = BodyState(tension=TENSION_HIGH)
        state = InteroceptiveState(high_tension_turns=TENSION_THRESHOLD_TURNS)
        result = compute_interoceptive_feedback(state, body)
        # At exactly the threshold, excess is 0 → no meaningful delta
        assert result.valence_delta == 0.0 or abs(result.valence_delta) < 0.01

    def test_borderline_energy_exactly_at_threshold(self) -> None:
        body = BodyState(energy=ENERGY_LOW)
        state = InteroceptiveState(low_energy_turns=LOW_ENERGY_THRESHOLD_TURNS)
        result = compute_interoceptive_feedback(state, body)
        # At exactly the threshold, deficit is 0 → no meaningful delta
        assert result.valence_delta == 0.0 or abs(result.valence_delta) < 0.01


class TestInteroceptiveResult:
    """Tests for the result dataclass."""

    def test_default_inactive(self) -> None:
        result = InteroceptiveResult()
        assert not result.active
        assert result.valence_delta == 0.0
        assert result.arousal_delta == 0.0
        assert result.source == ""

    def test_active_has_source(self) -> None:
        body = BodyState(tension=0.9)
        state = InteroceptiveState(high_tension_turns=TENSION_THRESHOLD_TURNS)
        result = compute_interoceptive_feedback(state, body)
        if result.active:
            assert result.source != ""


class TestMultiTurnSimulation:
    """Test interoception over multiple simulated turns."""

    def test_gradual_tension_buildup(self) -> None:
        """Tension that builds over turns eventually generates feedback."""
        body = BodyState(tension=0.8)
        state = InteroceptiveState()

        feedback_activated = False
        for turn in range(10):
            state = update_interoceptive_state(state, body)
            result = compute_interoceptive_feedback(state, body)
            if result.active:
                feedback_activated = True
                assert result.source == "tension"
                break

        assert feedback_activated
        assert state.high_tension_turns >= TENSION_THRESHOLD_TURNS

    def test_tension_release_stops_feedback(self) -> None:
        """When tension drops, counter resets and feedback stops."""
        tense_body = BodyState(tension=0.9)
        relaxed_body = BodyState(tension=0.3)

        state = InteroceptiveState(high_tension_turns=5)
        state = update_interoceptive_state(state, relaxed_body)
        assert state.high_tension_turns == 0

        result = compute_interoceptive_feedback(state, relaxed_body)
        assert not result.active

    def test_mixed_body_state_sequence(self) -> None:
        """Body state changes across turns produce correct tracking."""
        state = InteroceptiveState()

        # Turn 1-3: high tension
        for _ in range(3):
            state = update_interoceptive_state(state, BodyState(tension=0.8))
        assert state.high_tension_turns == 3

        # Turn 4: tension drops
        state = update_interoceptive_state(state, BodyState(tension=0.4))
        assert state.high_tension_turns == 0

        # Turn 5-6: low energy
        for _ in range(2):
            state = update_interoceptive_state(state, BodyState(energy=0.1))
        assert state.low_energy_turns == 2
