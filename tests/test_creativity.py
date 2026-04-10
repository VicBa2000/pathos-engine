"""Tests para Emotional Creativity - modulacion de estructura de pensamiento."""

import pytest

from pathos.engine.creativity import (
    compute_creativity,
    compute_creativity_level,
    compute_temperature_modifier,
    determine_thinking_mode,
)
from pathos.models.creativity import CreativityState, ThinkingMode, default_creativity_state
from pathos.models.emotion import BodyState, EmotionalState, PrimaryEmotion


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_state(
    primary: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.3,
    dominance: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.3,
    openness: float = 0.5,
    energy: float = 0.5,
    stack: dict[str, float] | None = None,
) -> EmotionalState:
    return EmotionalState(
        primary_emotion=primary,
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        certainty=certainty,
        intensity=intensity,
        body_state=BodyState(openness=openness, energy=energy),
        emotional_stack=stack or {"neutral": 1.0},
    )


# ── Model tests ─────────────────────────────────────────────────────────

class TestCreativityModel:
    def test_default_state(self) -> None:
        state = default_creativity_state()
        assert state.thinking_mode == ThinkingMode.STANDARD
        assert state.creativity_level == 0.0
        assert state.temperature_modifier == 0.0
        assert state.active_instructions == []
        assert state.triggered_by == []

    def test_thinking_mode_enum(self) -> None:
        assert ThinkingMode.DIVERGENT.value == "divergent"
        assert ThinkingMode.REFLECTIVE.value == "reflective"
        assert ThinkingMode.FOCUSED.value == "focused"
        assert ThinkingMode.PREVENTIVE.value == "preventive"
        assert ThinkingMode.SYNTHESIZING.value == "synthesizing"
        assert ThinkingMode.EXPLORATORY.value == "exploratory"
        assert ThinkingMode.NURTURING.value == "nurturing"
        assert ThinkingMode.STANDARD.value == "standard"

    def test_creativity_state_clamping(self) -> None:
        state = CreativityState(creativity_level=0.0, temperature_modifier=-0.3)
        assert state.creativity_level >= 0.0
        assert state.temperature_modifier >= -0.3


# ── Thinking mode determination ─────────────────────────────────────────

class TestThinkingMode:
    def test_neutral_returns_standard(self) -> None:
        state = _make_state(stack={"neutral": 1.0})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.STANDARD

    def test_joy_triggers_divergent(self) -> None:
        state = _make_state(stack={"joy": 0.5, "excitement": 0.3, "neutral": 0.2})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.DIVERGENT
        assert "joy" in triggers

    def test_excitement_triggers_divergent(self) -> None:
        state = _make_state(stack={"excitement": 0.6, "hope": 0.2, "neutral": 0.2})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.DIVERGENT

    def test_sadness_triggers_reflective(self) -> None:
        state = _make_state(stack={"sadness": 0.5, "disappointment": 0.2, "neutral": 0.3})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.REFLECTIVE
        assert "sadness" in triggers

    def test_anger_triggers_focused(self) -> None:
        state = _make_state(stack={"anger": 0.5, "frustration": 0.3, "neutral": 0.2})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.FOCUSED
        assert "anger" in triggers

    def test_fear_triggers_preventive(self) -> None:
        state = _make_state(stack={"fear": 0.4, "anxiety": 0.3, "neutral": 0.3})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.PREVENTIVE
        assert "fear" in triggers

    def test_contemplation_triggers_synthesizing(self) -> None:
        state = _make_state(stack={"contemplation": 0.5, "mixed": 0.2, "neutral": 0.3})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.SYNTHESIZING
        assert "contemplation" in triggers

    def test_surprise_triggers_exploratory(self) -> None:
        state = _make_state(stack={"surprise": 0.5, "neutral": 0.5})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.EXPLORATORY
        assert "surprise" in triggers

    def test_gratitude_triggers_nurturing(self) -> None:
        state = _make_state(stack={"gratitude": 0.4, "contentment": 0.3, "neutral": 0.3})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.NURTURING
        assert "gratitude" in triggers

    def test_low_activation_returns_standard(self) -> None:
        """Even emotional stack with no dominant mode → standard."""
        state = _make_state(stack={
            "joy": 0.05, "sadness": 0.05, "anger": 0.05,
            "fear": 0.05, "neutral": 0.8,
        })
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.STANDARD

    def test_competing_modes_picks_strongest(self) -> None:
        """When divergent and focused compete, strongest activation wins."""
        state = _make_state(stack={
            "joy": 0.15, "excitement": 0.15,  # divergent total: 0.30
            "anger": 0.20, "frustration": 0.05,  # focused total: 0.25
            "neutral": 0.45,
        })
        mode, _ = determine_thinking_mode(state)
        assert mode == ThinkingMode.DIVERGENT

    def test_empty_stack_returns_standard(self) -> None:
        state = _make_state(stack={})
        mode, triggers = determine_thinking_mode(state)
        assert mode == ThinkingMode.STANDARD
        assert triggers == []

    def test_indifference_not_mapped(self) -> None:
        """Indifference has no specific mode - should be standard."""
        state = _make_state(stack={"indifference": 0.8, "neutral": 0.2})
        mode, _ = determine_thinking_mode(state)
        assert mode == ThinkingMode.STANDARD


# ── Creativity level ────────────────────────────────────────────────────

class TestCreativityLevel:
    def test_high_openness_high_creativity(self) -> None:
        state = _make_state(openness=0.9, arousal=0.6, intensity=0.7)
        level = compute_creativity_level(state, ThinkingMode.DIVERGENT)
        assert level > 0.7

    def test_low_openness_low_creativity(self) -> None:
        state = _make_state(openness=0.2, arousal=0.3, intensity=0.1)
        level = compute_creativity_level(state, ThinkingMode.STANDARD)
        assert level < 0.35

    def test_divergent_mode_amplifies(self) -> None:
        state = _make_state(openness=0.6, arousal=0.6, intensity=0.5)
        level_div = compute_creativity_level(state, ThinkingMode.DIVERGENT)
        level_std = compute_creativity_level(state, ThinkingMode.STANDARD)
        assert level_div > level_std

    def test_exploratory_mode_amplifies(self) -> None:
        state = _make_state(openness=0.6, arousal=0.6, intensity=0.5)
        level_exp = compute_creativity_level(state, ThinkingMode.EXPLORATORY)
        level_std = compute_creativity_level(state, ThinkingMode.STANDARD)
        assert level_exp > level_std

    def test_creativity_clamped_to_01(self) -> None:
        state = _make_state(openness=1.0, arousal=0.65, intensity=1.0)
        level = compute_creativity_level(state, ThinkingMode.DIVERGENT)
        assert 0.0 <= level <= 1.0

    def test_yerkes_dodson_arousal(self) -> None:
        """Moderate arousal (~0.65) should give higher creativity than extremes."""
        state_mid = _make_state(openness=0.6, arousal=0.65, intensity=0.5)
        state_low = _make_state(openness=0.6, arousal=0.1, intensity=0.5)
        state_high = _make_state(openness=0.6, arousal=1.0, intensity=0.5)

        level_mid = compute_creativity_level(state_mid, ThinkingMode.STANDARD)
        level_low = compute_creativity_level(state_low, ThinkingMode.STANDARD)
        level_high = compute_creativity_level(state_high, ThinkingMode.STANDARD)

        assert level_mid > level_low
        assert level_mid > level_high


# ── Temperature modifier ────────────────────────────────────────────────

class TestTemperatureModifier:
    def test_high_creativity_positive_modifier(self) -> None:
        mod = compute_temperature_modifier(0.9, ThinkingMode.DIVERGENT, BodyState(openness=0.8))
        assert mod > 0

    def test_low_creativity_negative_modifier(self) -> None:
        mod = compute_temperature_modifier(0.1, ThinkingMode.PREVENTIVE, BodyState(openness=0.3))
        assert mod < 0

    def test_modifier_in_range(self) -> None:
        for mode in ThinkingMode:
            for creativity in [0.0, 0.5, 1.0]:
                for openness in [0.0, 0.5, 1.0]:
                    mod = compute_temperature_modifier(
                        creativity, mode, BodyState(openness=openness),
                    )
                    assert -0.3 <= mod <= 0.3

    def test_focused_mode_lowers_temperature(self) -> None:
        mod_focused = compute_temperature_modifier(0.5, ThinkingMode.FOCUSED, BodyState())
        mod_divergent = compute_temperature_modifier(0.5, ThinkingMode.DIVERGENT, BodyState())
        assert mod_focused < mod_divergent

    def test_high_openness_slight_boost(self) -> None:
        mod_open = compute_temperature_modifier(0.5, ThinkingMode.STANDARD, BodyState(openness=0.9))
        mod_closed = compute_temperature_modifier(0.5, ThinkingMode.STANDARD, BodyState(openness=0.3))
        assert mod_open > mod_closed


# ── Full compute_creativity ─────────────────────────────────────────────

class TestComputeCreativity:
    def test_neutral_state_standard_mode(self) -> None:
        state = _make_state()
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.STANDARD
        assert result.active_instructions == []

    def test_joyful_state_divergent_with_instructions(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.JOY,
            valence=0.7, arousal=0.7, intensity=0.7, openness=0.8,
            stack={"joy": 0.5, "excitement": 0.3, "neutral": 0.2},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.DIVERGENT
        assert len(result.active_instructions) > 0
        assert result.creativity_level > 0.5
        assert result.temperature_modifier > 0
        assert "joy" in result.triggered_by

    def test_sad_state_reflective(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.SADNESS,
            valence=-0.6, arousal=0.2, intensity=0.6, openness=0.4,
            stack={"sadness": 0.6, "disappointment": 0.2, "neutral": 0.2},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.REFLECTIVE
        assert "sadness" in result.triggered_by

    def test_angry_state_focused(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.ANGER,
            valence=-0.7, arousal=0.8, intensity=0.8, openness=0.3,
            stack={"anger": 0.6, "frustration": 0.2, "neutral": 0.2},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.FOCUSED
        assert result.temperature_modifier < 0  # focused lowers temperature

    def test_fearful_state_preventive(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.FEAR,
            valence=-0.5, arousal=0.7, intensity=0.6, openness=0.3,
            stack={"fear": 0.5, "anxiety": 0.3, "neutral": 0.2},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.PREVENTIVE

    def test_surprise_state_exploratory(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.SURPRISE,
            valence=0.1, arousal=0.8, intensity=0.7, openness=0.7,
            stack={"surprise": 0.6, "alertness": 0.2, "neutral": 0.2},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.EXPLORATORY

    def test_contemplative_state_synthesizing(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.CONTEMPLATION,
            valence=0.1, arousal=0.3, intensity=0.4, openness=0.6,
            stack={"contemplation": 0.5, "mixed": 0.2, "neutral": 0.3},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.SYNTHESIZING

    def test_gratitude_state_nurturing(self) -> None:
        state = _make_state(
            primary=PrimaryEmotion.GRATITUDE,
            valence=0.6, arousal=0.3, intensity=0.5, openness=0.6,
            stack={"gratitude": 0.5, "contentment": 0.3, "neutral": 0.2},
        )
        result = compute_creativity(state)
        assert result.thinking_mode == ThinkingMode.NURTURING

    def test_high_creativity_more_instructions(self) -> None:
        """Higher creativity level should produce more instructions."""
        state_high = _make_state(
            openness=0.9, arousal=0.65, intensity=0.8,
            stack={"joy": 0.6, "excitement": 0.3, "neutral": 0.1},
        )
        state_low = _make_state(
            openness=0.4, arousal=0.3, intensity=0.3,
            stack={"joy": 0.4, "neutral": 0.6},
        )
        result_high = compute_creativity(state_high)
        result_low = compute_creativity(state_low)
        assert len(result_high.active_instructions) >= len(result_low.active_instructions)

    def test_creativity_level_in_range(self) -> None:
        """Creativity level always 0-1."""
        for primary in PrimaryEmotion:
            state = _make_state(
                primary=primary, valence=0.5, arousal=0.5,
                intensity=0.5, openness=0.5,
                stack={primary.value: 0.6, "neutral": 0.4},
            )
            result = compute_creativity(state)
            assert 0.0 <= result.creativity_level <= 1.0

    def test_temperature_modifier_in_range(self) -> None:
        """Temperature modifier always within bounds."""
        for primary in PrimaryEmotion:
            state = _make_state(
                primary=primary, valence=0.5, arousal=0.5,
                intensity=0.5, openness=0.5,
                stack={primary.value: 0.6, "neutral": 0.4},
            )
            result = compute_creativity(state)
            assert -0.3 <= result.temperature_modifier <= 0.3


# ── Integration with behavior modifier ──────────────────────────────────

class TestBehaviorIntegration:
    def test_creativity_in_behavior_prompt(self) -> None:
        from pathos.engine.behavior import generate_behavior_modifier

        state = _make_state(
            primary=PrimaryEmotion.JOY,
            valence=0.7, arousal=0.7, intensity=0.7, openness=0.8,
            stack={"joy": 0.5, "excitement": 0.3, "neutral": 0.2},
        )
        creativity = compute_creativity(state)
        prompt = generate_behavior_modifier(state, creativity=creativity)

        assert "MODO COGNITIVO" in prompt
        assert "divergent" in prompt

    def test_standard_mode_not_in_prompt(self) -> None:
        from pathos.engine.behavior import generate_behavior_modifier

        state = _make_state()
        creativity = compute_creativity(state)
        prompt = generate_behavior_modifier(state, creativity=creativity)

        assert "MODO COGNITIVO" not in prompt

    def test_no_creativity_param_works(self) -> None:
        """Backward compat: behavior modifier works without creativity."""
        from pathos.engine.behavior import generate_behavior_modifier

        state = _make_state()
        prompt = generate_behavior_modifier(state)
        assert "TU ESTADO INTERNO" in prompt
