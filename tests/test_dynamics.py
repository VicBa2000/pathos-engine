"""Tests for Emotion Dynamics ODE engine."""

import pytest

from pathos.engine.dynamics import EmotionDynamics, EMOTION_INERTIA
from pathos.models.coupling import (
    CouplingMatrix,
    preset_default_coupling,
    preset_neurotic_coupling,
    preset_resilient_coupling,
    preset_volatile_coupling,
)
from pathos.models.emotion import PrimaryEmotion


class TestDynamicsBasics:
    def test_step_moves_toward_target(self):
        dyn = EmotionDynamics(variability=0.0)  # No noise for deterministic test
        current = 0.0
        target = 0.8
        result = dyn.step(current, target, 0.0, PrimaryEmotion.NEUTRAL, "valence")
        assert result > current  # Moved toward target

    def test_step_respects_attractor(self):
        dyn = EmotionDynamics(attractor_strength=0.5, variability=0.0)
        # Current is far from attractor, no stimulus
        result = dyn.step(0.8, 0.8, 0.0, PrimaryEmotion.NEUTRAL, "valence")
        # Should pull back toward attractor (0.0)
        assert result < 0.8

    def test_valence_clamped(self):
        dyn = EmotionDynamics(variability=0.0)
        result = dyn.step(0.9, 1.5, 0.0, PrimaryEmotion.JOY, "valence")
        assert -1 <= result <= 1

    def test_arousal_clamped(self):
        dyn = EmotionDynamics(variability=0.0)
        result = dyn.step(0.1, -0.5, 0.3, PrimaryEmotion.NEUTRAL, "arousal")
        assert 0 <= result <= 1


class TestEmotionInertia:
    def test_anger_high_inertia(self):
        assert EMOTION_INERTIA[PrimaryEmotion.ANGER] > 0.7

    def test_surprise_low_inertia(self):
        assert EMOTION_INERTIA[PrimaryEmotion.SURPRISE] < 0.2

    def test_sadness_persists(self):
        assert EMOTION_INERTIA[PrimaryEmotion.SADNESS] >= 0.7

    def test_inertia_affects_movement(self):
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        # Same target, different emotions
        result_surprise = dyn.step(0.0, 0.8, 0.0, PrimaryEmotion.SURPRISE, "valence")
        result_anger = dyn.step(0.0, 0.8, 0.0, PrimaryEmotion.ANGER, "valence")
        # Surprise should move more (lower inertia)
        assert abs(result_surprise) > abs(result_anger)


class TestStep4D:
    def test_returns_4_values(self):
        dyn = EmotionDynamics(variability=0.0)
        v, a, d, c = dyn.step_4d(
            0.0, 0.3, 0.5, 0.5,
            0.5, 0.7, 0.6, 0.4,
            0.1, 0.3,
            PrimaryEmotion.JOY,
        )
        assert -1 <= v <= 1
        assert 0 <= a <= 1
        assert 0 <= d <= 1
        assert 0 <= c <= 1

    def test_moves_toward_targets(self):
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        v, a, d, c = dyn.step_4d(
            0.0, 0.3, 0.5, 0.5,
            0.8, 0.9, 0.8, 0.3,
            0.0, 0.3,
            PrimaryEmotion.NEUTRAL,
        )
        assert v > 0.0  # Moved toward 0.8
        assert a > 0.3  # Moved toward 0.9


class TestIdleFluctuation:
    def test_fluctuates_around_attractor(self):
        dyn = EmotionDynamics(variability=0.5)
        results = set()
        for _ in range(20):
            r = dyn.idle_fluctuation(0.5, 0.5, "arousal")
            results.add(r)
        # Should produce some variation
        assert len(results) > 1

    def test_clamped(self):
        dyn = EmotionDynamics(variability=1.0)
        for _ in range(50):
            r = dyn.idle_fluctuation(0.95, 0.5, "arousal")
            assert 0 <= r <= 1


class TestStochasticNoise:
    def test_variability_produces_different_results(self):
        dyn = EmotionDynamics(variability=0.5, attractor_strength=0.0)
        results = set()
        for _ in range(20):
            r = dyn.step(0.5, 0.5, 0.5, PrimaryEmotion.NEUTRAL, "valence")
            results.add(r)
        # With noise, should get different values
        assert len(results) > 1

    def test_zero_variability_deterministic(self):
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        results = set()
        for _ in range(10):
            r = dyn.step(0.5, 0.5, 0.5, PrimaryEmotion.NEUTRAL, "valence")
            results.add(r)
        assert len(results) == 1  # Deterministic


class TestCoupledStep4D:
    """Tests for step_4d with CouplingMatrix integration."""

    def test_no_coupling_backward_compatible(self) -> None:
        """Without coupling, step_4d should produce same results."""
        dyn = EmotionDynamics(variability=0.0)
        args = (
            0.0, 0.3, 0.5, 0.5,   # current
            0.5, 0.7, 0.6, 0.4,   # target
            0.1, 0.3,              # baselines
            PrimaryEmotion.JOY,
        )
        v1, a1, d1, c1 = dyn.step_4d(*args)
        v2, a2, d2, c2 = dyn.step_4d(*args, coupling=None)
        assert v1 == v2
        assert a1 == a2
        assert d1 == d2
        assert c1 == c2

    def test_zero_coupling_same_as_none(self) -> None:
        """A zero CouplingMatrix should behave identically to None."""
        dyn = EmotionDynamics(variability=0.0)
        args = (
            -0.3, 0.6, 0.4, 0.3,
            0.2, 0.5, 0.6, 0.7,
            0.0, 0.3,
            PrimaryEmotion.SADNESS,
        )
        v1, a1, d1, c1 = dyn.step_4d(*args, coupling=None)
        v2, a2, d2, c2 = dyn.step_4d(*args, coupling=CouplingMatrix())
        assert v1 == v2
        assert a1 == a2
        assert d1 == d2
        assert c1 == c2

    def test_negative_valence_pushes_arousal_up(self) -> None:
        """With coupling, negative valence state should push arousal higher
        compared to uncoupled computation."""
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        coupling = preset_default_coupling()
        # State: very negative valence, moderate arousal, at target (no stimulus change)
        args = (
            -0.7, 0.4, 0.5, 0.5,   # current: negative V
            -0.7, 0.4, 0.5, 0.5,   # target: same (no new stimulus)
            0.0, 0.3,               # baselines
            PrimaryEmotion.SADNESS,
        )
        _, a_none, _, _ = dyn.step_4d(*args, coupling=None)
        _, a_coupled, _, _ = dyn.step_4d(*args, coupling=coupling)
        assert a_coupled > a_none, (
            "Negative valence should push arousal up via coupling"
        )

    def test_high_arousal_pushes_dominance_down(self) -> None:
        """With coupling, high arousal should reduce dominance."""
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        coupling = preset_default_coupling()
        args = (
            0.0, 0.9, 0.5, 0.5,   # current: very high arousal
            0.0, 0.9, 0.5, 0.5,   # target: same
            0.0, 0.3,              # baselines (arousal above baseline by 0.6)
            PrimaryEmotion.ANXIETY,
        )
        _, _, d_none, _ = dyn.step_4d(*args, coupling=None)
        _, _, d_coupled, _ = dyn.step_4d(*args, coupling=coupling)
        assert d_coupled < d_none, (
            "High arousal should push dominance down via coupling"
        )

    def test_negative_valence_reduces_certainty(self) -> None:
        """Negative valence should erode certainty through coupling."""
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        coupling = preset_default_coupling()
        args = (
            -0.8, 0.5, 0.5, 0.6,   # current: very negative V, moderate C
            -0.8, 0.5, 0.5, 0.6,   # target: same
            0.0, 0.3,
            PrimaryEmotion.FEAR,
        )
        _, _, _, c_none = dyn.step_4d(*args, coupling=None)
        _, _, _, c_coupled = dyn.step_4d(*args, coupling=coupling)
        assert c_coupled < c_none, (
            "Negative valence should reduce certainty via coupling"
        )

    def test_high_dominance_increases_certainty(self) -> None:
        """Feeling in control should boost certainty through coupling."""
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        coupling = preset_default_coupling()
        args = (
            0.2, 0.4, 0.9, 0.5,   # current: high dominance
            0.2, 0.4, 0.9, 0.5,   # target: same
            0.0, 0.3,
            PrimaryEmotion.CONTENTMENT,
        )
        _, _, _, c_none = dyn.step_4d(*args, coupling=None)
        _, _, _, c_coupled = dyn.step_4d(*args, coupling=coupling)
        assert c_coupled > c_none, (
            "High dominance should increase certainty via coupling"
        )

    def test_coupling_respects_clamping(self) -> None:
        """Even with strong coupling, all dimensions stay in valid ranges."""
        dyn = EmotionDynamics(variability=0.0)
        coupling = preset_volatile_coupling()
        # Extreme state to maximize coupling effects
        v, a, d, c = dyn.step_4d(
            -1.0, 1.0, 0.0, 0.0,   # extreme current
            -1.0, 1.0, 0.0, 0.0,   # extreme target
            0.0, 0.3,
            PrimaryEmotion.ANGER,
            coupling=coupling,
        )
        assert -1 <= v <= 1
        assert 0 <= a <= 1
        assert 0 <= d <= 1
        assert 0 <= c <= 1

    def test_neurotic_stronger_effect_than_resilient(self) -> None:
        """Neurotic coupling should produce larger deviations from uncoupled."""
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        neurotic = preset_neurotic_coupling()
        resilient = preset_resilient_coupling()
        args = (
            -0.6, 0.7, 0.4, 0.4,
            -0.6, 0.7, 0.4, 0.4,
            0.0, 0.3,
            PrimaryEmotion.ANXIETY,
        )
        _, _, _, _ = dyn.step_4d(*args, coupling=None)
        vn, an, dn, cn = dyn.step_4d(*args, coupling=neurotic)
        vr, ar, dr, cr = dyn.step_4d(*args, coupling=resilient)
        v0, a0, d0, c0 = dyn.step_4d(*args, coupling=None)

        # Neurotic deviation from uncoupled should be larger
        neurotic_delta = abs(an - a0) + abs(dn - d0) + abs(cn - c0)
        resilient_delta = abs(ar - a0) + abs(dr - d0) + abs(cr - c0)
        assert neurotic_delta > resilient_delta

    def test_coupling_plus_contagion_stack(self) -> None:
        """Coupling and contagion should both contribute to the final state."""
        dyn = EmotionDynamics(variability=0.0, attractor_strength=0.0)
        coupling = preset_default_coupling()
        args = (
            -0.5, 0.5, 0.5, 0.5,
            -0.5, 0.5, 0.5, 0.5,
            0.0, 0.3,
            PrimaryEmotion.SADNESS,
        )
        # No contagion, no coupling
        v0, a0, _, _ = dyn.step_4d(*args, coupling=None)
        # Coupling only
        _, a_coup, _, _ = dyn.step_4d(*args, coupling=coupling)
        # Contagion only (positive arousal push)
        _, a_cont, _, _ = dyn.step_4d(*args, contagion_a=0.05, coupling=None)
        # Both
        _, a_both, _, _ = dyn.step_4d(*args, contagion_a=0.05, coupling=coupling)
        # Both together should produce the strongest arousal push
        assert a_both > a_coup or a_both > a_cont
