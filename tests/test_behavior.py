"""Tests para el Behavior Modifier (Fase 2: 19 emociones + blended stack)."""

from pathos.engine.behavior import (
    EMOTION_EFFECTS,
    RAW_EMOTION_EFFECTS,
    generate_behavior_modifier,
    generate_blended_effects,
    generate_raw_behavior_modifier,
    generate_simple_behavior_modifier,
)
from pathos.models.emotion import BodyState, EmotionalState, PrimaryEmotion


def test_neutral_state_produces_valid_prompt() -> None:
    state = EmotionalState()
    prompt = generate_behavior_modifier(state)
    assert "ESTADO INTERNO ACTUAL" in prompt
    assert "neutral" in prompt


def test_high_warmth_adds_calida() -> None:
    state = EmotionalState(
        body_state=BodyState(warmth=0.8),
        primary_emotion=PrimaryEmotion.JOY,
        intensity=0.5,
    )
    prompt = generate_behavior_modifier(state)
    assert "CALIDA" in prompt


def test_low_energy_adds_breves() -> None:
    state = EmotionalState(
        body_state=BodyState(energy=0.2),
        primary_emotion=PrimaryEmotion.SADNESS,
        intensity=0.5,
    )
    prompt = generate_behavior_modifier(state)
    assert "breves" in prompt


def test_intense_emotion_shows_effect() -> None:
    state = EmotionalState(
        primary_emotion=PrimaryEmotion.ANGER,
        intensity=0.8,
        valence=-0.6,
        arousal=0.8,
    )
    prompt = generate_behavior_modifier(state)
    assert "URGENCIA" in prompt


def test_anxiety_shows_effect() -> None:
    state = EmotionalState(
        primary_emotion=PrimaryEmotion.ANXIETY,
        intensity=0.6,
    )
    prompt = generate_behavior_modifier(state)
    assert "NEGATIVO" in prompt


def test_gratitude_shows_effect() -> None:
    state = EmotionalState(
        primary_emotion=PrimaryEmotion.GRATITUDE,
        intensity=0.6,
    )
    prompt = generate_behavior_modifier(state)
    assert "VALIOSO" in prompt


def test_secondary_emotion_in_prompt() -> None:
    state = EmotionalState(
        primary_emotion=PrimaryEmotion.ANGER,
        secondary_emotion=PrimaryEmotion.ANXIETY,
        intensity=0.7,
    )
    prompt = generate_behavior_modifier(state)
    assert "anxiety" in prompt


def test_negative_valence_shows_riesgos() -> None:
    state = EmotionalState(valence=-0.5, intensity=0.3)
    prompt = generate_behavior_modifier(state)
    assert "RIESGOS" in prompt


def test_positive_valence_shows_oportunidades() -> None:
    state = EmotionalState(valence=0.5, intensity=0.3)
    prompt = generate_behavior_modifier(state)
    assert "OPORTUNIDADES" in prompt


# --- Blended Emotional Stack tests ---


class TestGenerateBlendedEffects:
    """Tests for the composicional blend of the emotional stack."""

    def test_empty_stack_returns_empty(self) -> None:
        assert generate_blended_effects({}, EMOTION_EFFECTS) == ""

    def test_default_neutral_stack_returns_empty(self) -> None:
        """Default uncomputed stack (neutral=1.0 only) returns empty for fallback."""
        assert generate_blended_effects({"neutral": 1.0}, EMOTION_EFFECTS) == ""

    def test_single_dominant_emotion(self) -> None:
        stack = {"joy": 0.8, "contentment": 0.05, "neutral": 0.05}
        result = generate_blended_effects(stack, EMOTION_EFFECTS)
        assert "EXPANSION" in result
        assert "Estado dominante" in result  # Single emotion format
        assert "MEZCLA" not in result

    def test_two_emotions_blend(self) -> None:
        stack = {"sadness": 0.5, "gratitude": 0.3, "neutral": 0.1}
        result = generate_blended_effects(stack, EMOTION_EFFECTS)
        assert "MEZCLA" in result
        assert "REFLEXIVIDAD" in result  # sadness effect
        assert "VALIOSO" in result  # gratitude effect

    def test_three_emotions_blend(self) -> None:
        stack = {"anger": 0.4, "sadness": 0.3, "hope": 0.2, "neutral": 0.05}
        result = generate_blended_effects(stack, EMOTION_EFFECTS)
        assert "MEZCLA" in result
        assert "URGENCIA" in result  # anger
        assert "REFLEXIVIDAD" in result  # sadness
        assert "RESULTADO POSITIVO" in result  # hope

    def test_percentages_shown(self) -> None:
        stack = {"joy": 0.6, "excitement": 0.3, "neutral": 0.05}
        result = generate_blended_effects(stack, EMOTION_EFFECTS)
        assert "%" in result

    def test_min_activation_filters(self) -> None:
        stack = {"anger": 0.8, "joy": 0.05, "fear": 0.02}
        result = generate_blended_effects(stack, EMOTION_EFFECTS, min_activation=0.10)
        assert "URGENCIA" in result  # anger passes
        assert "EXPANSION" not in result  # joy filtered out (0.05 < 0.10)

    def test_max_emotions_limit(self) -> None:
        stack = {
            "anger": 0.3, "sadness": 0.25, "fear": 0.2,
            "anxiety": 0.15, "hope": 0.1,
        }
        result = generate_blended_effects(stack, EMOTION_EFFECTS, max_emotions=2)
        # Only top 2: anger and sadness
        assert "URGENCIA" in result
        assert "REFLEXIVIDAD" in result
        # 3rd and beyond filtered
        assert "ALERTA" not in result  # fear

    def test_raw_effects_blend(self) -> None:
        stack = {"anger": 0.5, "sadness": 0.3, "neutral": 0.1}
        result = generate_blended_effects(stack, RAW_EMOTION_EFFECTS)
        assert "FURIA" in result or "DESTRUIR" in result  # raw anger
        assert "TRISTEZA" in result  # raw sadness

    def test_invalid_emotion_name_skipped(self) -> None:
        stack = {"joy": 0.5, "nonexistent_emotion": 0.4}
        result = generate_blended_effects(stack, EMOTION_EFFECTS)
        # Should still work, just skip the invalid one
        assert "EXPANSION" in result


class TestBlendedInModifiers:
    """Tests that blended effects integrate correctly in the three modifier functions."""

    def _make_blended_state(self) -> EmotionalState:
        return EmotionalState(
            primary_emotion=PrimaryEmotion.SADNESS,
            secondary_emotion=PrimaryEmotion.GRATITUDE,
            intensity=0.6,
            valence=-0.3,
            arousal=0.4,
            emotional_stack={
                "sadness": 0.45,
                "gratitude": 0.25,
                "hope": 0.15,
                "neutral": 0.08,
            },
        )

    def test_full_modifier_uses_blend(self) -> None:
        state = self._make_blended_state()
        prompt = generate_behavior_modifier(state)
        assert "MEZCLA" in prompt
        assert "REFLEXIVIDAD" in prompt  # sadness
        assert "VALIOSO" in prompt  # gratitude

    def test_raw_modifier_uses_blend(self) -> None:
        state = self._make_blended_state()
        prompt = generate_raw_behavior_modifier(state)
        assert "TRISTEZA" in prompt  # raw sadness

    def test_simple_modifier_uses_blend_max_2(self) -> None:
        state = self._make_blended_state()
        prompt = generate_simple_behavior_modifier(state)
        # Simple mode limits to 2 emotions
        assert "REFLEXIVIDAD" in prompt  # sadness (top 1)
        assert "VALIOSO" in prompt  # gratitude (top 2)

    def test_fallback_when_stack_is_default(self) -> None:
        """When stack is uncomputed, should use primary_emotion effect."""
        state = EmotionalState(
            primary_emotion=PrimaryEmotion.ANGER,
            intensity=0.8,
            valence=-0.6,
            arousal=0.8,
            # emotional_stack left as default {"neutral": 1.0}
        )
        prompt = generate_behavior_modifier(state)
        assert "URGENCIA" in prompt  # anger primary fallback
