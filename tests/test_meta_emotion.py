"""Tests for Meta-Emotional Awareness engine."""

import pytest

from pathos.engine.meta import generate_meta_emotion, MetaEmotion
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.values import default_value_system


def _make_state(
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    intensity: float = 0.5,
    duration: int = 0,
    valence: float = 0.0,
) -> EmotionalState:
    return EmotionalState(
        primary_emotion=emotion,
        intensity=intensity,
        duration=duration,
        valence=valence,
    )


class TestMetaEmotionGeneration:
    def test_no_meta_for_low_intensity(self):
        current = _make_state(intensity=0.1)
        previous = _make_state()
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values)
        assert result is None

    def test_satisfaction_after_regulation(self):
        current = _make_state(PrimaryEmotion.CONTEMPLATION, intensity=0.4)
        previous = _make_state(PrimaryEmotion.ANGER, intensity=0.8)
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values, regulation_success=True)
        assert result is not None
        assert result.meta_response == "satisfaction"
        assert result.target_emotion == PrimaryEmotion.ANGER

    def test_conflict_anger_vs_compassion(self):
        current = _make_state(PrimaryEmotion.ANGER, intensity=0.7, valence=-0.7)
        previous = _make_state()
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values)
        assert result is not None
        assert result.meta_response == "conflict"
        assert "compassion" in result.reason

    def test_conflict_indifference_vs_compassion(self):
        current = _make_state(PrimaryEmotion.INDIFFERENCE, intensity=0.6)
        previous = _make_state()
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values)
        assert result is not None
        assert result.meta_response == "conflict"

    def test_curiosity_for_new_emotion(self):
        current = _make_state(PrimaryEmotion.MIXED, intensity=0.7)
        previous = _make_state(PrimaryEmotion.NEUTRAL)
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values, is_new_emotion=True)
        assert result is not None
        assert result.meta_response == "curiosity"

    def test_discomfort_for_intense_unregulated(self):
        # Use ANXIETY (not FEAR) to avoid value conflict with "truth"
        current = _make_state(PrimaryEmotion.ANXIETY, intensity=0.9)
        previous = _make_state()
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values, regulation_success=False)
        assert result is not None
        assert result.meta_response == "discomfort"

    def test_acceptance_for_sustained(self):
        current = _make_state(PrimaryEmotion.CONTEMPLATION, intensity=0.5, duration=3)
        previous = _make_state(PrimaryEmotion.CONTEMPLATION, intensity=0.5, duration=2)
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values)
        assert result is not None
        assert result.meta_response == "acceptance"


class TestMetaEmotionModel:
    def test_intensity_range(self):
        current = _make_state(PrimaryEmotion.ANGER, intensity=0.8)
        previous = _make_state(PrimaryEmotion.ANGER, intensity=0.9)
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values, regulation_success=True)
        if result:
            assert 0 <= result.intensity <= 1

    def test_reason_not_empty(self):
        current = _make_state(PrimaryEmotion.ANGER, intensity=0.7)
        previous = _make_state()
        values = default_value_system()
        result = generate_meta_emotion(current, previous, values)
        if result:
            assert len(result.reason) > 0
