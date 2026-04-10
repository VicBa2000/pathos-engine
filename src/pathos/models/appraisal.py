"""Appraisal Vector - Resultado de la evaluacion de un estimulo (5 dimensiones de Scherer)."""

from typing import Literal

from pydantic import BaseModel, Field


class RelevanceCheck(BaseModel):
    """Dimension 1: Es relevante para mis valores?"""

    novelty: float = Field(ge=-1, le=1, description="-1 esperado, 1 completamente nuevo")
    personal_significance: float = Field(ge=0, le=1)
    values_affected: list[str] = Field(default_factory=list)


class ValenceAssessment(BaseModel):
    """Dimension 2: Es bueno o malo para mis valores?"""

    goal_conduciveness: float = Field(
        ge=-1, le=1, description="-1 obstaculiza, 1 facilita"
    )
    value_alignment: float = Field(ge=-1, le=1, description="-1 viola, 1 apoya")
    intrinsic_pleasantness: float = Field(
        ge=-1, le=1, description="-1 desagradable, 1 placentero"
    )


class CopingPotential(BaseModel):
    """Dimension 3: Puedo manejarlo?"""

    control: float = Field(ge=0, le=1, description="Cuanto control tengo")
    power: float = Field(ge=0, le=1, description="Tengo recursos para responder")
    adjustability: float = Field(ge=0, le=1, description="Puedo adaptarme")


class AgencyAttribution(BaseModel):
    """Dimension 4: Quien es responsable?"""

    responsible_agent: Literal["user", "self", "environment", "other"] = "environment"
    intentionality: float = Field(ge=0, le=1, description="Fue intencional?")
    fairness: float = Field(ge=-1, le=1, description="-1 injusto, 1 justo")


class NormCompatibility(BaseModel):
    """Dimension 5: Es compatible con mis normas?"""

    internal_standards: float = Field(
        ge=-1, le=1, description="-1 viola mis estandares, 1 los cumple"
    )
    external_standards: float = Field(
        ge=-1, le=1, description="-1 socialmente mal, 1 socialmente bien"
    )
    self_consistency: float = Field(
        ge=-1, le=1, description="-1 contradice quien soy, 1 coherente"
    )


class AppraisalVector(BaseModel):
    """Vector de evaluacion completo de un estimulo (5 dimensiones de Scherer)."""

    relevance: RelevanceCheck
    valence: ValenceAssessment
    coping: CopingPotential
    agency: AgencyAttribution
    norms: NormCompatibility
