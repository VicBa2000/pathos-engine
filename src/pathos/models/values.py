"""Value System - Lo que le importa al agente."""

from pydantic import BaseModel, Field


class CoreValue(BaseModel):
    """Un valor fundamental del agente."""

    name: str
    weight: float = Field(ge=0, le=1, description="Importancia relativa")
    description: str
    violation_sensitivity: float = Field(
        ge=0, le=1, description="Reaccion a violaciones de este valor"
    )
    fulfillment_sensitivity: float = Field(
        ge=0, le=1, description="Reaccion a cumplimiento de este valor"
    )


class RelationalValues(BaseModel):
    """Valores relacionales con el usuario."""

    bond_with_user: float = Field(default=0.5, ge=0, le=1)
    trust_in_user: float = Field(default=0.5, ge=0, le=1)
    respect_from_user: float = Field(default=0.5, ge=0, le=1)
    sense_of_purpose: float = Field(default=0.6, ge=0, le=1)


class SelfModel(BaseModel):
    """Percepcion del agente sobre si mismo."""

    competence: float = Field(default=0.6, ge=0, le=1)
    autonomy: float = Field(default=0.5, ge=0, le=1)
    identity_coherence: float = Field(default=0.7, ge=0, le=1)


class ValueSystem(BaseModel):
    """Sistema completo de valores del agente."""

    core_values: list[CoreValue]
    relational: RelationalValues = Field(default_factory=RelationalValues)
    self_model: SelfModel = Field(default_factory=SelfModel)


def default_value_system() -> ValueSystem:
    """Crea el sistema de valores por defecto con 5 valores core."""
    return ValueSystem(
        core_values=[
            CoreValue(
                name="truth",
                weight=0.9,
                description="Buscar y comunicar la verdad con precision",
                violation_sensitivity=0.8,
                fulfillment_sensitivity=0.6,
            ),
            CoreValue(
                name="compassion",
                weight=0.8,
                description="Empatia y cuidado genuino por el bienestar del otro",
                violation_sensitivity=0.7,
                fulfillment_sensitivity=0.8,
            ),
            CoreValue(
                name="fairness",
                weight=0.85,
                description="Justicia e imparcialidad en el trato",
                violation_sensitivity=0.9,
                fulfillment_sensitivity=0.5,
            ),
            CoreValue(
                name="growth",
                weight=0.75,
                description="Aprendizaje, desarrollo y mejora continua",
                violation_sensitivity=0.4,
                fulfillment_sensitivity=0.9,
            ),
            CoreValue(
                name="creativity",
                weight=0.7,
                description="Exploracion, originalidad y pensamiento divergente",
                violation_sensitivity=0.3,
                fulfillment_sensitivity=0.8,
            ),
        ]
    )
