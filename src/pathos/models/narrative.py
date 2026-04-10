"""Narrative Self - Yo narrativo con autobiographical memory.

El sistema construye una narrativa coherente de quién es a lo largo
del tiempo, basada en su historia emocional. "Soy alguien que se
frustra fácilmente con la injusticia" o "soy alguien que encuentra
paz en la reflexión".

Base teórica:
- McAdams (2001) "The Psychology of Life Stories"
- Bruner (1991) "The Narrative Construction of Reality"
"""

from enum import Enum

from pydantic import BaseModel, Field

from pathos.models.emotion import PrimaryEmotion


class IdentityCategory(str, Enum):
    """Categorías de statements de identidad."""

    REACTIVE = "reactive"              # "reacciono con X ante Y"
    RELATIONAL = "relational"          # "soy cercano/distante con el usuario"
    COPING = "coping"                  # "tiendo a regular via Z"
    TEMPERAMENTAL = "temperamental"    # "soy emocionalmente reactivo/estable"
    VALUES = "values"                  # "valoro X por encima de todo"
    GROWTH = "growth"                  # "he aprendido a manejar X"


class IdentityStatement(BaseModel):
    """Un statement de identidad derivado de la historia emocional."""

    category: IdentityCategory
    statement: str = Field(description="Texto legible: 'Reacciono con ira ante la injusticia'")
    emotion: PrimaryEmotion = Field(description="Emoción asociada al statement")
    trigger_category: str = Field(description="Categoría de estímulo: criticism, praise, threat, etc.")
    valence: float = Field(ge=-1, le=1, description="Valencia del statement (-1 negativo, +1 positivo)")
    strength: float = Field(default=0.3, ge=0, le=1, description="Fuerza del statement, crece con refuerzo")
    formation_turn: int = Field(ge=0, description="Turno en que se formó")
    reinforcement_count: int = Field(default=0, ge=0)
    last_reinforced_turn: int = Field(ge=0, description="Último turno donde se reforzó")


class NarrativeCrisis(BaseModel):
    """Estado de crisis de identidad — cuando experiencias contradicen la narrativa."""

    active: bool = False
    contradiction_count: int = Field(default=0, ge=0, description="Contradicciones recientes")
    source_statement: str = Field(default="", description="Statement contradicho")
    source_emotion: str = Field(default="", description="Emoción esperada vs obtenida")
    turns_active: int = Field(default=0, ge=0)
    resolution_type: str = Field(default="", description="'growth' o 'regression' o ''")


class GrowthEvent(BaseModel):
    """Un evento de crecimiento narrativo — experiencia transformadora."""

    turn: int = Field(ge=0)
    old_pattern: str = Field(description="Patrón anterior: 'anger ante criticism'")
    new_pattern: str = Field(description="Nuevo patrón: 'contemplation ante criticism'")
    trigger: str = Field(description="Qué causó el cambio")
    emotion_before: PrimaryEmotion
    emotion_after: PrimaryEmotion


class NarrativeSelf(BaseModel):
    """El yo narrativo del agente — su historia emocional comprimida en identidad."""

    identity_statements: list[IdentityStatement] = Field(default_factory=list)
    crisis: NarrativeCrisis = Field(default_factory=NarrativeCrisis)
    growth_events: list[GrowthEvent] = Field(default_factory=list)
    coherence_score: float = Field(
        default=1.0, ge=0, le=1,
        description="Qué tan coherente es el comportamiento actual con la narrativa",
    )
    narrative_age: int = Field(default=0, ge=0, description="Turnos desde el primer statement")
    total_contradictions: int = Field(default=0, ge=0)
    total_reinforcements: int = Field(default=0, ge=0)


# Límites
MAX_IDENTITY_STATEMENTS = 12
MAX_GROWTH_EVENTS = 10


def default_narrative_self() -> NarrativeSelf:
    """Narrative self por defecto (vacío, sin historia)."""
    return NarrativeSelf()
