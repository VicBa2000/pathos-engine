"""Shadow State - Estado emocional espejo del usuario (Emotion Contagion).

El contagio emocional es PRE-cognitivo: no pasa por el appraisal.
Como los mirror neurons, el agente "absorbe" la emocion del usuario
antes de evaluarla cognitivamente.

Basado en:
- Hatfield et al. (1993) "Emotional Contagion"
- Preston & de Waal (2002) Perception-Action Model
"""

from pydantic import BaseModel, Field


class ShadowState(BaseModel):
    """Estado emocional espejo del usuario.

    Representa la emocion inferida del usuario que el agente
    "absorbe" pre-cognitivamente. Decae entre turnos.
    """

    # Dimensiones emocionales inferidas del usuario
    valence: float = Field(default=0.0, ge=-1, le=1, description="Valencia emocional del usuario")
    arousal: float = Field(default=0.3, ge=0, le=1, description="Activacion emocional del usuario")

    # Fuerza de la senal emocional detectada (0=nada, 1=muy clara)
    signal_strength: float = Field(
        default=0.0, ge=0, le=1,
        description="Claridad de la senal emocional del usuario",
    )

    # Contagion acumulado (cuanto se ha contagiado el agente en total)
    accumulated_contagion: float = Field(
        default=0.0, ge=0, le=1,
        description="Contagio acumulado en la sesion",
    )

    # Turnos desde ultima deteccion fuerte
    turns_since_strong_signal: int = Field(
        default=0, ge=0,
        description="Turnos desde ultima senal emocional fuerte",
    )


def default_shadow_state() -> ShadowState:
    """Shadow state por defecto (neutral, sin senal)."""
    return ShadowState()
