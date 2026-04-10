"""Emotional Creativity - Modelo de datos.

El estado emocional modula la ESTRUCTURA del pensamiento,
no solo el tono. Cada emocion produce un modo de pensamiento distinto.

Base teorica:
- Isen (1999) "Positive Affect and Creativity"
- Baas et al. (2008) "A Meta-Analysis of Mood and Creativity"
"""

from enum import Enum

from pydantic import BaseModel, Field


class ThinkingMode(str, Enum):
    """Modos de pensamiento modulados por la emocion."""

    DIVERGENT = "divergent"        # Joy/excitement: analogias lejanas, ideas nuevas
    REFLECTIVE = "reflective"      # Sadness: analisis profundo, insight
    FOCUSED = "focused"            # Anger: eliminar obstaculos, ser directo
    PREVENTIVE = "preventive"      # Fear/anxiety: edge cases, cautela
    SYNTHESIZING = "synthesizing"  # Contemplation: big picture, conexiones
    EXPLORATORY = "exploratory"    # Surprise: buscar lo inesperado
    NURTURING = "nurturing"        # Gratitude/contentment: consolidar, cuidar
    STANDARD = "standard"          # Neutral/low intensity: pensamiento normal


class CreativityState(BaseModel):
    """Estado del modulador de creatividad."""

    thinking_mode: ThinkingMode = ThinkingMode.STANDARD
    creativity_level: float = Field(
        default=0.0, ge=0, le=1,
        description="Nivel de creatividad activo (0=estandar, 1=maximo)",
    )
    temperature_modifier: float = Field(
        default=0.0, ge=-0.3, le=0.3,
        description="Ajuste de temperatura para el LLM (-0.3 a +0.3)",
    )
    active_instructions: list[str] = Field(
        default_factory=list,
        description="Instrucciones de pensamiento inyectadas en el prompt",
    )
    triggered_by: list[str] = Field(
        default_factory=list,
        description="Emociones que activaron este modo",
    )


def default_creativity_state() -> CreativityState:
    """Estado de creatividad por defecto (estandar)."""
    return CreativityState()
