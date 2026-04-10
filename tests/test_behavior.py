"""Tests para el Behavior Modifier (Fase 2: 19 emociones)."""

from pathos.engine.behavior import generate_behavior_modifier
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
