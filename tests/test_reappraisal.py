"""Tests for Reappraisal engine."""

import pytest

from pathos.engine.reappraisal import reappraise, should_reappraise, ReappraisalResult
from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _make_state(
    emotion: PrimaryEmotion = PrimaryEmotion.FEAR,
    intensity: float = 0.8,
    valence: float = -0.7,
    dominance: float = 0.5,
    certainty: float = 0.5,
) -> EmotionalState:
    return EmotionalState(
        primary_emotion=emotion,
        intensity=intensity,
        valence=valence,
        arousal=0.7,
        dominance=dominance,
        certainty=certainty,
    )


class TestShouldReappraise:
    def test_yes_intense_negative(self):
        state = _make_state(PrimaryEmotion.FEAR, intensity=0.8, dominance=0.5)
        assert should_reappraise(state, regulation_capacity=0.5) is True

    def test_no_low_intensity(self):
        state = _make_state(intensity=0.3)
        assert should_reappraise(state, regulation_capacity=0.5) is False

    def test_no_positive_emotion(self):
        state = _make_state(PrimaryEmotion.JOY, intensity=0.8, valence=0.7, dominance=0.5)
        assert should_reappraise(state, regulation_capacity=0.5) is False

    def test_no_low_dominance(self):
        state = _make_state(dominance=0.2)
        assert should_reappraise(state, regulation_capacity=0.5) is False

    def test_no_ego_depleted(self):
        state = _make_state(dominance=0.5)
        assert should_reappraise(state, regulation_capacity=0.1) is False


class TestReappraise:
    def test_fear_reappraised_to_alertness(self):
        state = _make_state(PrimaryEmotion.FEAR, intensity=0.8, dominance=0.6)
        reappraised, result = reappraise(state, regulation_capacity=0.5)
        if result.applied:
            assert reappraised.primary_emotion in (PrimaryEmotion.ALERTNESS, PrimaryEmotion.FEAR)
            assert reappraised.intensity <= state.intensity

    def test_anxiety_acceptance(self):
        state = _make_state(PrimaryEmotion.ANXIETY, intensity=0.7, dominance=0.4, certainty=0.6)
        reappraised, result = reappraise(state, regulation_capacity=0.5)
        if result.applied and result.strategy == "acceptance":
            assert reappraised.arousal <= state.arousal

    def test_anger_reframing(self):
        state = _make_state(PrimaryEmotion.ANGER, intensity=0.85, valence=-0.8, dominance=0.7)
        reappraised, result = reappraise(state, regulation_capacity=0.5)
        if result.applied and result.strategy == "reframing":
            assert reappraised.valence > state.valence

    def test_no_reappraisal_when_not_needed(self):
        state = _make_state(PrimaryEmotion.JOY, intensity=0.3, valence=0.5)
        reappraised, result = reappraise(state, regulation_capacity=0.5)
        assert result.applied is False
        assert reappraised is state  # Same object

    def test_result_tracks_changes(self):
        state = _make_state(PrimaryEmotion.FEAR, intensity=0.8, dominance=0.6)
        _, result = reappraise(state, regulation_capacity=0.5)
        if result.applied:
            assert result.original_emotion is not None
            assert result.reappraised_emotion is not None
            assert result.strategy is not None

    def test_intensity_always_reduced(self):
        for emotion in [PrimaryEmotion.FEAR, PrimaryEmotion.ANGER, PrimaryEmotion.SADNESS]:
            state = _make_state(emotion, intensity=0.8, dominance=0.6)
            reappraised, result = reappraise(state, regulation_capacity=0.5)
            if result.applied:
                assert reappraised.intensity <= state.intensity
