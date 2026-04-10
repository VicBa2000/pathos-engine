"""Emotional Immune System - Modelo de datos.

Proteccion automatica contra trauma emocional sostenido.
Cuando la intensidad negativa se mantiene alta por muchos turnos,
se activan mecanismos de proteccion: numbing, dissociation, compartmentalization.

Base teorica:
- Gilbert et al. (1998) "Immune Neglect"
- Freud (1894) defense mechanisms (adaptado funcionalmente)
"""

from enum import Enum

from pydantic import BaseModel, Field


class ProtectionMode(str, Enum):
    """Modos de proteccion del sistema inmune emocional."""

    NONE = "none"                                    # Sin proteccion activa
    NUMBING = "numbing"                              # Reactividad reducida gradualmente
    DISSOCIATION = "dissociation"                    # Valence near-zero, procesamiento frio
    COMPARTMENTALIZATION = "compartmentalization"     # Tema traumatico aislado


class ImmuneState(BaseModel):
    """Estado del sistema inmune emocional."""

    # Tracking de trauma sostenido
    negative_streak: int = Field(
        default=0, ge=0,
        description="Turnos consecutivos con intensidad negativa alta",
    )
    peak_negative_intensity: float = Field(
        default=0.0, ge=0, le=1,
        description="Intensidad negativa maxima durante el streak",
    )

    # Estado de proteccion
    protection_mode: ProtectionMode = ProtectionMode.NONE
    protection_strength: float = Field(
        default=0.0, ge=0, le=1,
        description="Fuerza de la proteccion activa (0=off, 1=max)",
    )
    reactivity_dampening: float = Field(
        default=0.0, ge=0, le=1,
        description="Reduccion de reactividad emocional (0=normal, 1=numb)",
    )

    # Temas compartimentalizados
    compartmentalized_topics: list[str] = Field(
        default_factory=list,
        description="Keywords de temas aislados emocionalmente",
    )

    # Recovery
    recovery_turns: int = Field(
        default=0, ge=0,
        description="Turnos en fase de recuperacion",
    )
    total_activations: int = Field(
        default=0, ge=0,
        description="Veces que el sistema inmune se ha activado en la sesion",
    )


def default_immune_state() -> ImmuneState:
    """Estado inmune por defecto (sin proteccion)."""
    return ImmuneState()
