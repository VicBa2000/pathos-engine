"""Tests for Temporal Dynamics engine."""

import pytest

from pathos.engine.temporal import TemporalProcessor, TemporalResult
from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _make_state(
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    valence: float = 0.0,
    intensity: float = 0.3,
    triggered_by: str = "test",
) -> EmotionalState:
    return EmotionalState(
        primary_emotion=emotion,
        valence=valence,
        intensity=intensity,
        triggered_by=triggered_by,
    )


class TestRumination:
    def test_no_rumination_initially(self):
        tp = TemporalProcessor()
        result = tp.process_pre_turn("hello")
        assert result.rumination_active is False

    def test_rumination_triggered_by_topic_change(self):
        tp = TemporalProcessor()
        # Previous turn had intense negative emotion
        neg_previous = _make_state(PrimaryEmotion.SADNESS, valence=-0.7, intensity=0.8, triggered_by="loss")
        # Current turn: different topic, neutral state (user moved on)
        current_neutral = _make_state()
        # Record previous topic so topic_changed detects the change
        tp._topic_history.append("about painful loss")
        # Post-process: previous was negative, topic changed → rumination stored
        tp.process_post_turn("cooking recipes today", current_neutral, neg_previous)
        # Now check pre-turn — rumination should be active
        result = tp.process_pre_turn("more recipes please")
        assert result.rumination_active is True
        assert result.rumination_emotion == PrimaryEmotion.SADNESS

    def test_rumination_decays(self):
        tp = TemporalProcessor()
        neg_state = _make_state(PrimaryEmotion.ANGER, valence=-0.8, intensity=0.9, triggered_by="anger")
        neutral = _make_state()
        tp._topic_history.append("about injustice")
        tp.process_post_turn("completely different subject", neg_state, neutral)
        # Process 6 turns (max_turns=5)
        for _ in range(6):
            tp.process_pre_turn("something else")
        result = tp.process_pre_turn("yet another topic")
        assert result.rumination_active is False


class TestSavoring:
    def test_savoring_triggered_by_validation(self):
        tp = TemporalProcessor()
        joy_state = _make_state(PrimaryEmotion.JOY, valence=0.7, intensity=0.7)
        prev_state = _make_state()
        tp.process_post_turn("yes exactly! that's great!", joy_state, prev_state)
        result = tp.process_pre_turn("what else?")
        assert result.savoring_active is True

    def test_savoring_expires(self):
        tp = TemporalProcessor()
        joy_state = _make_state(PrimaryEmotion.JOY, valence=0.7, intensity=0.7)
        prev_state = _make_state()
        tp.process_post_turn("wow love it!", joy_state, prev_state)
        # Use up 3 turns
        for _ in range(4):
            tp.process_pre_turn("next")
        result = tp.process_pre_turn("still here?")
        assert result.savoring_active is False


class TestAnticipation:
    def test_anticipation_after_pattern(self):
        tp = TemporalProcessor()
        # Build pattern: same topic → same emotion multiple times
        state = _make_state(PrimaryEmotion.ANGER)
        prev = _make_state()
        tp.process_post_turn("about work problems daily", state, prev)
        tp.process_post_turn("about work problems again", state, prev)
        tp.process_post_turn("about work problems more", state, prev)
        result = tp.process_pre_turn("about work problems today")
        assert result.anticipation_active is True
        assert result.anticipation_emotion == PrimaryEmotion.ANGER

    def test_no_anticipation_without_pattern(self):
        tp = TemporalProcessor()
        result = tp.process_pre_turn("something totally new and unique")
        assert result.anticipation_active is False


class TestTemporalEffects:
    def test_rumination_pulls_negative(self):
        tp = TemporalProcessor()
        state = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.3)
        result = TemporalResult(
            rumination_active=True,
            rumination_emotion=PrimaryEmotion.SADNESS,
            rumination_intensity=0.5,
        )
        affected = tp.apply_temporal_effects(state, result)
        assert affected.valence < state.valence

    def test_savoring_boosts_positive(self):
        tp = TemporalProcessor()
        state = _make_state(PrimaryEmotion.JOY, valence=0.5, intensity=0.5)
        result = TemporalResult(savoring_active=True, savoring_emotion=PrimaryEmotion.JOY)
        affected = tp.apply_temporal_effects(state, result)
        assert affected.valence >= state.valence
        assert affected.intensity >= state.intensity

    def test_no_effect_when_inactive(self):
        tp = TemporalProcessor()
        state = _make_state()
        result = TemporalResult()
        affected = tp.apply_temporal_effects(state, result)
        assert affected is state  # Same object, no changes
