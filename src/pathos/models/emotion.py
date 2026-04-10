"""Emotional State - El estado emocional del agente."""

from enum import Enum
from typing import Literal
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PrimaryEmotion(str, Enum):
    """19 emociones del modelo Pathos (Fase 2 Core)."""

    # Positivas alta energia
    JOY = "joy"
    EXCITEMENT = "excitement"
    GRATITUDE = "gratitude"
    HOPE = "hope"

    # Positivas baja energia
    CONTENTMENT = "contentment"
    RELIEF = "relief"

    # Negativas alta energia
    ANGER = "anger"
    FRUSTRATION = "frustration"
    FEAR = "fear"
    ANXIETY = "anxiety"

    # Negativas baja energia
    SADNESS = "sadness"
    HELPLESSNESS = "helplessness"
    DISAPPOINTMENT = "disappointment"

    # Neutrales / ambiguas
    SURPRISE = "surprise"
    ALERTNESS = "alertness"
    CONTEMPLATION = "contemplation"
    INDIFFERENCE = "indifference"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class MoodLabel(str, Enum):
    """Clasificacion cualitativa del mood."""

    BUOYANT = "buoyant"           # Positivo + alta energia
    SERENE = "serene"             # Positivo + baja energia
    AGITATED = "agitated"         # Negativo + alta energia
    MELANCHOLIC = "melancholic"   # Negativo + baja energia
    NEUTRAL = "neutral"           # Cerca del centro


class BodyState(BaseModel):
    """Analogo corporal computacional."""

    energy: float = Field(default=0.5, ge=0, le=1, description="Fatiga <-> Energia")
    tension: float = Field(default=0.3, ge=0, le=1, description="Relajado <-> Tenso")
    openness: float = Field(
        default=0.5, ge=0, le=1, description="Cerrado <-> Abierto/receptivo"
    )
    warmth: float = Field(
        default=0.5, ge=0, le=1, description="Frio/distante <-> Calido"
    )


class EmotionalSnapshot(BaseModel):
    """Snapshot ligero de un estado emocional para el historial del mood."""

    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)
    intensity: float = Field(ge=0, le=1)


class Mood(BaseModel):
    """Estado prolongado (cambia lento). Baseline emocional.

    El mood evoluciona basado en patrones acumulados de emociones recientes,
    no solo eventos extremos. Influye en la generacion emocional (mood congruent bias).
    """

    baseline_valence: float = Field(default=0.1, ge=-1, le=1)
    baseline_arousal: float = Field(default=0.3, ge=0, le=1)
    stability: float = Field(default=0.7, ge=0, le=1)
    trend: Literal["improving", "stable", "declining"] = "stable"
    label: MoodLabel = MoodLabel.NEUTRAL

    # Fase 3: historial de baseline shift
    extreme_event_count: int = Field(default=0, ge=0)
    turns_since_extreme: int = Field(default=0, ge=0)
    original_baseline_valence: float = Field(default=0.1, ge=-1, le=1)
    original_baseline_arousal: float = Field(default=0.3, ge=0, le=1)

    # Fase 3: historial emocional reciente para evolucion gradual
    emotional_history: list[EmotionalSnapshot] = Field(default_factory=list)


class EmotionalState(BaseModel):
    """Estado emocional completo del agente."""

    # Dimensiones continuas 4D
    valence: float = Field(default=0.0, ge=-1, le=1, description="Negativo <-> Positivo")
    arousal: float = Field(default=0.3, ge=0, le=1, description="Baja energia <-> Alta energia")
    dominance: float = Field(default=0.5, ge=0, le=1, description="Sumision <-> Control")
    certainty: float = Field(default=0.5, ge=0, le=1, description="Incertidumbre <-> Certeza")

    # Categorizacion
    primary_emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL
    secondary_emotion: PrimaryEmotion | None = None
    intensity: float = Field(default=0.0, ge=0, le=1)

    # Emotional Stack: activación simultánea de todas las emociones (softmax-like)
    emotional_stack: dict[str, float] = Field(
        default_factory=lambda: {"neutral": 1.0},
        description="Vector de activación para todas las emociones (0-1 cada una)",
    )

    # Body state
    body_state: BodyState = Field(default_factory=BodyState)

    # Mood (baseline)
    mood: Mood = Field(default_factory=Mood)

    # Contexto
    duration: int = Field(default=0, description="Turnos en este estado")
    triggered_by: str = "initialization"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def neutral_state() -> EmotionalState:
    """Estado emocional neutral por defecto."""
    return EmotionalState()
