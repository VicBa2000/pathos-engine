"""Tests for Active Emotional Regulation engine."""

import pytest

from pathos.engine.regulation import EmotionalRegulator, RegulationResult
from pathos.models.emotion import BodyState, EmotionalState, PrimaryEmotion


def _make_state(
    emotion: PrimaryEmotion = PrimaryEmotion.ANGER,
    intensity: float = 0.8,
    valence: float = -0.7,
    arousal: float = 0.8,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        certainty=0.5,
        primary_emotion=emotion,
        intensity=intensity,
        body_state=BodyState(energy=0.6, tension=0.7, openness=0.4, warmth=0.3),
    )


class TestRegulationBasics:
    def test_no_regulation_low_intensity(self):
        reg = EmotionalRegulator()
        state = _make_state(intensity=0.3)
        regulated, result = reg.regulate(state)
        assert result.strategy_used is None
        assert regulated.intensity == state.intensity

    def test_regulation_triggers_high_intensity(self):
        reg = EmotionalRegulator()
        state = _make_state(intensity=0.85)
        regulated, result = reg.regulate(state)
        assert result.strategy_used is not None

    def test_capacity_depletes(self):
        reg = EmotionalRegulator(regulation_capacity=0.5)
        state = _make_state(intensity=0.85)
        reg.regulate(state)
        assert reg.regulation_capacity < 0.5


class TestRegulationStrategies:
    def test_reappraisal_reduces_intensity(self):
        reg = EmotionalRegulator(regulation_capacity=0.8)
        state = _make_state(emotion=PrimaryEmotion.ANGER, intensity=0.9, valence=-0.8)
        regulated, result = reg.regulate(state)
        if result.strategy_used == "reappraisal":
            assert regulated.intensity < state.intensity

    def test_suppression_creates_dissonance(self):
        reg = EmotionalRegulator(regulation_capacity=0.3)
        state = _make_state(intensity=0.75)
        regulated, result = reg.regulate(state)
        if result.strategy_used == "suppression":
            assert reg.suppression_dissonance > 0

    def test_expression_reduces_tension(self):
        reg = EmotionalRegulator(suppression_dissonance=0.6)
        state = _make_state(intensity=0.75)
        regulated, result = reg.regulate(state)
        if result.strategy_used == "expression":
            assert regulated.body_state.tension < state.body_state.tension

    def test_distraction_for_anxiety(self):
        reg = EmotionalRegulator(regulation_capacity=0.6)
        state = _make_state(emotion=PrimaryEmotion.ANXIETY, intensity=0.8, valence=-0.4, arousal=0.7)
        _, result = reg.regulate(state)
        assert result.strategy_used in ("distraction", "suppression", "reappraisal")


class TestEgoDepletion:
    def test_breakthrough_when_depleted(self):
        reg = EmotionalRegulator(regulation_capacity=0.05)
        state = _make_state(intensity=0.9)
        regulated, result = reg.regulate(state)
        assert result.breakthrough is True
        assert regulated.intensity >= state.intensity  # Amplified

    def test_recovery_between_turns(self):
        reg = EmotionalRegulator(regulation_capacity=0.3)
        reg.recover(0.7)
        assert reg.regulation_capacity > 0.3

    def test_recovery_capped_at_personality_base(self):
        reg = EmotionalRegulator(regulation_capacity=0.65)
        reg.recover(0.7)
        assert reg.regulation_capacity <= 0.7

    def test_consecutive_regulation_increases_cost(self):
        reg = EmotionalRegulator(regulation_capacity=0.9)
        state = _make_state(intensity=0.85)
        reg.regulate(state)
        cap_after_first = reg.regulation_capacity
        state2 = _make_state(intensity=0.85)
        reg.regulate(state2)
        cap_after_second = reg.regulation_capacity
        # Second regulation should cost more
        first_cost = 0.9 - cap_after_first
        second_cost = cap_after_first - cap_after_second
        assert second_cost >= first_cost

    def test_breakthrough_count(self):
        reg = EmotionalRegulator(regulation_capacity=0.05)
        state = _make_state(intensity=0.9)
        reg.regulate(state)
        assert reg.breakthroughs_count == 1


class TestDissonance:
    def test_dissonance_triggers_expression(self):
        reg = EmotionalRegulator(suppression_dissonance=0.6, regulation_capacity=0.5)
        state = _make_state(intensity=0.75)
        _, result = reg.regulate(state)
        assert result.strategy_used == "expression"

    def test_dissonance_decays_with_recovery(self):
        reg = EmotionalRegulator(suppression_dissonance=0.5)
        reg.recover()
        assert reg.suppression_dissonance < 0.5
