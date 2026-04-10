"""User Model - Social Cognition and Theory of Mind.

Modela activamente al usuario como agente emocional.
La relación modula las respuestas emocionales:
- Un insulto de alto rapport genera hurt/disappointment
- Un insulto de bajo rapport genera indifference/contempt

Rapport crece lento y se rompe rápido (asimétrico como en humanos).
"""

from typing import Literal

from pydantic import BaseModel, Field


class UserModel(BaseModel):
    """Modelo computacional del usuario."""

    perceived_intent: float = Field(
        default=0.3, ge=-1, le=1,
        description="-1=hostil, 0=neutral, 1=benevolente",
    )
    perceived_engagement: float = Field(
        default=0.5, ge=0, le=1,
        description="0=desinteresado, 1=comprometido",
    )
    rapport: float = Field(
        default=0.3, ge=0, le=1,
        description="Nivel de conexión/confianza establecida",
    )
    communication_style: Literal["formal", "casual", "technical", "emotional", "unknown"] = Field(
        default="unknown",
        description="Estilo detectado del usuario",
    )
    emotional_reciprocity: float = Field(
        default=0.5, ge=0, le=1,
        description="¿El usuario responde a las emociones del sistema?",
    )
    trust_level: float = Field(
        default=0.5, ge=0, le=1,
        description="Nivel de confianza actual",
    )
    interaction_count: int = Field(
        default=0, ge=0,
        description="Número de interacciones totales",
    )

    # Trust trajectory (últimos 10 valores)
    trust_trajectory: list[float] = Field(
        default_factory=list,
        description="Historial de confianza para detectar tendencia",
    )


def default_user_model() -> UserModel:
    """Modelo de usuario por defecto (desconocido, neutral)."""
    return UserModel()
