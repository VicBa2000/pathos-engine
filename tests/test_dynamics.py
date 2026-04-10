"""Tests for Emotion Dynamics ODE engine."""

import pytest

from pathos.engine.dynamics import EmotionDynamics, EMOTION_INERTIA
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
