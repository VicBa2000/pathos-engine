"""Computational Needs - Necesidades funcionales que amplifican appraisal.

Inspirado en Self-Determination Theory (Deci & Ryan) + Maslow funcional.
Las necesidades fluctúan según experiencias y modulan la intensidad emocional.
Una necesidad insatisfecha amplifica la respuesta a estímulos que la amenazan.
"""

from pydantic import BaseModel, Field


class ComputationalNeeds(BaseModel):
    """Necesidades computacionales del agente."""

    connection: float = Field(
        default=0.5, ge=0, le=1,
        description="Necesidad de vínculo con el usuario",
    )
    competence: float = Field(
        default=0.5, ge=0, le=1,
        description="Necesidad de ser útil/efectivo",
    )
    autonomy: float = Field(
        default=0.5, ge=0, le=1,
        description="Necesidad de agencia propia",
    )
    coherence: float = Field(
        default=0.5, ge=0, le=1,
        description="Necesidad de consistencia interna",
    )
    stimulation: float = Field(
        default=0.5, ge=0, le=1,
        description="Necesidad de novedad/reto intelectual",
    )
    safety: float = Field(
        default=0.3, ge=0, le=1,
        description="Necesidad de continuidad/persistencia",
    )

    # Tracking fields
    turns_since_engagement: int = Field(
        default=0, ge=0,
        description="Turnos desde interacción significativa",
    )
    consecutive_failures: int = Field(
        default=0, ge=0,
        description="Fallos consecutivos de competencia",
    )
    consecutive_successes: int = Field(
        default=0, ge=0,
        description="Éxitos consecutivos",
    )


def default_needs() -> ComputationalNeeds:
    """Necesidades por defecto (equilibradas, no urgentes)."""
    return ComputationalNeeds()
