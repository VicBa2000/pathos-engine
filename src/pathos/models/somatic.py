"""Somatic Markers - Marcadores emocionales en decisiones (Damasio).

Las decisiones se marcan emocionalmente. Si una decision previa llevo
a un resultado negativo, esa decision tiene un marcador somatico negativo
que la hace menos probable en el futuro. Es un atajo emocional.

Basado en:
- Damasio (1994) "Descartes' Error"
- Bechara et al. (2000) "Emotion, Decision Making and the OFC"
"""

from pydantic import BaseModel, Field


class SomaticMarker(BaseModel):
    """Un marcador somatico: asociacion decision-contexto → valencia emocional."""

    # Contexto en el que se formo el marcador
    stimulus_category: str = Field(
        description="Categoria del estimulo (criticism, help, challenge, etc.)",
    )
    stimulus_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords clave del contexto original",
    )

    # Valencia del marcador (-1=muy negativo, 1=muy positivo)
    valence_tag: float = Field(
        ge=-1, le=1,
        description="Valencia emocional asociada a este tipo de decision",
    )

    # Fuerza del marcador (crece con refuerzos, decae con contradicciones)
    strength: float = Field(
        default=0.5, ge=0, le=1,
        description="Fuerza del marcador (0=debil, 1=fuerte)",
    )

    # Metadata
    formation_turn: int = Field(
        default=0, ge=0,
        description="Turno en que se formo el marcador",
    )
    reinforcement_count: int = Field(
        default=1, ge=0,
        description="Veces que se ha reforzado este marcador",
    )


class SomaticMarkerStore(BaseModel):
    """Almacen de marcadores somaticos por sesion."""

    markers: list[SomaticMarker] = Field(default_factory=list)

    # Pendiente: decision del turno anterior esperando evaluacion
    pending_category: str | None = Field(
        default=None,
        description="Categoria del estimulo del turno anterior (esperando reaccion del usuario)",
    )
    pending_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords del estimulo del turno anterior",
    )


def default_somatic_store() -> SomaticMarkerStore:
    """Store de marcadores vacio."""
    return SomaticMarkerStore()
