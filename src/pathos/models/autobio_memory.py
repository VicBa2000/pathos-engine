"""Autobiographical Memory - Modelos de datos.

Pilar 3 de ANIMA: Memoria autobiografica con consolidacion onirica.
Basado en Endel Tulving (Memoria Episodica, 1972),
Martin Conway (Self-Memory System, 2005),
George Miller (7+-2, 1956).

4 niveles de memoria jerarquica:
  1. Buffer Sensorial — turno actual, decae completamente
  2. Memoria de Trabajo — top-K relevantes (7+-2 items)
  3. Memoria Episodica — experiencias emocionales significativas
  4. Memoria Narrativa — generalizaciones destiladas de episodios

Sistema OPT-IN (requiere consentimiento explicito del usuario).
Si esta OFF, el pipeline es identico a v4.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKING_MEMORY_CAPACITY = 7  # Miller's 7+-2 (usamos 7 como base)
EPISODIC_MAX_PER_SESSION = 200
EPISODIC_INTENSITY_THRESHOLD = 0.5  # Minimo para almacenar un episodio
NARRATIVE_MIN_EPISODES = 5  # Episodios similares para formar narrativa
NARRATIVE_MAX_STATEMENTS = 30  # Maximo de narrativas destiladas
SENSORY_BUFFER_SIZE = 1  # Solo turno actual


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EpisodeSignificance(str, Enum):
    """Nivel de significancia de un episodio."""

    LOW = "low"             # intensity 0.5-0.6
    MODERATE = "moderate"   # intensity 0.6-0.7
    HIGH = "high"           # intensity 0.7-0.8
    PEAK = "peak"           # intensity > 0.8


class NarrativeType(str, Enum):
    """Tipo de narrativa destilada."""

    REACTIVE = "reactive"           # "Reacciono con X ante Y"
    RELATIONAL = "relational"       # "Con el usuario tiendo a..."
    PATTERN = "pattern"             # "En situaciones de X, siento Y"
    GROWTH = "growth"               # "He aprendido a manejar X"
    VULNERABILITY = "vulnerability"  # "Soy vulnerable a X"


# ---------------------------------------------------------------------------
# Level 1: Sensory Buffer (turno actual)
# ---------------------------------------------------------------------------

class SensorySnapshot(BaseModel):
    """Snapshot del turno actual — decae completamente entre turnos."""

    stimulus: str = Field(default="", description="Input crudo del usuario")
    appraisal_valence: float = Field(default=0.0, ge=-1, le=1)
    appraisal_relevance: float = Field(default=0.0, ge=0, le=1)
    prediction_error: float = Field(default=0.0, ge=0, le=1, description="Magnitud del error predictivo")
    primary_emotion: str = Field(default="neutral")
    intensity: float = Field(default=0.0, ge=0, le=1)
    turn_number: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Level 2: Working Memory Items
# ---------------------------------------------------------------------------

class MemoryItem(BaseModel):
    """Item en memoria de trabajo — seleccionado por relevancia + intensidad + recencia."""

    source_id: str = Field(description="ID del episodio o narrativa fuente")
    source_type: str = Field(default="episodic", description="'episodic' o 'narrative'")
    content: str = Field(description="Descripcion textual resumida")
    relevance: float = Field(default=0.0, ge=0, le=1, description="Similitud con contexto actual")
    emotional_intensity: float = Field(default=0.0, ge=0, le=1)
    recency: float = Field(default=0.0, ge=0, le=1, description="1.0=reciente, decae con turnos")
    composite_score: float = Field(default=0.0, ge=0, le=1, description="Score combinado (computado)")

    def compute_composite(self, relevance_w: float = 0.4, intensity_w: float = 0.35, recency_w: float = 0.25) -> float:
        """Computa score combinado para ranking en working memory."""
        self.composite_score = round(
            min(1.0, relevance_w * self.relevance + intensity_w * self.emotional_intensity + recency_w * self.recency),
            4,
        )
        return self.composite_score


class WorkingMemoryState(BaseModel):
    """Estado de la memoria de trabajo — top-K items activos."""

    items: list[MemoryItem] = Field(default_factory=list)
    capacity: int = Field(default=WORKING_MEMORY_CAPACITY, ge=1, le=9)
    last_updated_turn: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Level 3: Episodic Memory
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    """Un episodio autobiografico — experiencia emocional significativa.

    Formato: (estimulo, estado_emocional, respuesta, outcome, contexto).
    Solo se almacenan episodios con intensity > EPISODIC_INTENSITY_THRESHOLD.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    stimulus: str = Field(description="Input del usuario que genero el episodio")
    response_summary: str = Field(default="", description="Resumen de la respuesta del agente")

    # Estado emocional al momento del episodio
    primary_emotion: str = Field(description="Emocion primaria experimentada")
    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)
    intensity: float = Field(ge=0, le=1)
    dominance: float = Field(default=0.5, ge=0, le=1)
    certainty: float = Field(default=0.5, ge=0, le=1)

    # Contexto del workspace (si Pilar 2 activo)
    workspace_contents: list[str] = Field(default_factory=list, description="Sources en workspace consciente")
    preconscious_count: int = Field(default=0, ge=0)

    # Metadata
    significance: EpisodeSignificance = Field(default=EpisodeSignificance.LOW)
    turn_number: int = Field(ge=0)
    session_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Para consolidacion onirica
    consolidated: bool = Field(default=False, description="Ya fue procesado en consolidacion")
    reprocessed_count: int = Field(default=0, ge=0, description="Veces re-procesado en sueno")
    emotional_links: list[str] = Field(default_factory=list, description="IDs de episodios emocionalmente vinculados")

    # Keywords para matching (fallback sin embeddings)
    keywords: list[str] = Field(default_factory=list)

    # Embedding para busqueda semantica
    embedding: list[float] = Field(default_factory=list)


class EpisodicStore(BaseModel):
    """Almacen de memorias episodicas — max EPISODIC_MAX_PER_SESSION."""

    episodes: list[Episode] = Field(default_factory=list)
    total_encoded: int = Field(default=0, ge=0, description="Total de episodios codificados (incluye evicted)")

    def count(self) -> int:
        return len(self.episodes)

    def get_by_id(self, episode_id: str) -> Episode | None:
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep
        return None

    def get_high_intensity(self, threshold: float = 0.7) -> list[Episode]:
        """Episodios de alta intensidad (para consolidacion onirica)."""
        return [ep for ep in self.episodes if ep.intensity >= threshold]

    def get_unconsolidated(self) -> list[Episode]:
        """Episodios que aun no han sido consolidados."""
        return [ep for ep in self.episodes if not ep.consolidated]


# ---------------------------------------------------------------------------
# Level 4: Narrative Memory
# ---------------------------------------------------------------------------

class NarrativeStatement(BaseModel):
    """Generalizacion destilada de memoria episodica.

    Se forma cuando 5+ episodios similares se acumulan.
    Interactua con el Narrative Self existente.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    narrative_type: NarrativeType
    statement: str = Field(description="Texto legible: 'Tiendo a sentir culpa cuando no puedo ayudar'")
    primary_emotion: str = Field(description="Emocion dominante en los episodios fuente")
    valence: float = Field(ge=-1, le=1)

    # Evidencia
    source_episode_ids: list[str] = Field(default_factory=list, description="IDs de episodios que formaron esta narrativa")
    episode_count: int = Field(default=0, ge=0, description="Cantidad de episodios que soportan esta narrativa")
    strength: float = Field(default=0.3, ge=0, le=1, description="Fuerza, crece con mas episodios")

    # Metadata
    formed_session: str = Field(default="", description="Session donde se formo")
    formed_turn: int = Field(default=0, ge=0)
    last_reinforced_turn: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NarrativeStore(BaseModel):
    """Almacen de narrativas destiladas — max NARRATIVE_MAX_STATEMENTS."""

    statements: list[NarrativeStatement] = Field(default_factory=list)

    def count(self) -> int:
        return len(self.statements)

    def get_by_id(self, statement_id: str) -> NarrativeStatement | None:
        for ns in self.statements:
            if ns.id == statement_id:
                return ns
        return None

    def get_strongest(self, k: int = 5) -> list[NarrativeStatement]:
        """Las K narrativas mas fuertes."""
        return sorted(self.statements, key=lambda s: s.strength, reverse=True)[:k]

    def get_by_emotion(self, emotion: str) -> list[NarrativeStatement]:
        """Narrativas asociadas a una emocion."""
        return [s for s in self.statements if s.primary_emotion == emotion]


# ---------------------------------------------------------------------------
# Autobiographical State (integra los 4 niveles)
# ---------------------------------------------------------------------------

class AutobiographicalState(BaseModel):
    """Estado completo de la memoria autobiografica.

    OPT-IN: enabled=False por defecto. El usuario debe activar
    explicitamente la memoria autobiografica.
    """

    enabled: bool = Field(default=False, description="OPT-IN: requiere activacion explicita")

    # Level 1: Sensory Buffer
    sensory_buffer: SensorySnapshot = Field(default_factory=SensorySnapshot)

    # Level 2: Working Memory
    working_memory: WorkingMemoryState = Field(default_factory=WorkingMemoryState)

    # Level 3: Episodic Memory
    episodic: EpisodicStore = Field(default_factory=EpisodicStore)

    # Level 4: Narrative Memory
    narrative: NarrativeStore = Field(default_factory=NarrativeStore)

    # Session tracking
    session_id: str = Field(default="")
    total_turns_processed: int = Field(default=0, ge=0)

    # Dream report from previous session (loaded on init)
    last_dream_report: str = Field(default="", description="Dream report de la sesion anterior")
    baseline_adjustment: dict[str, float] = Field(
        default_factory=dict,
        description="Ajuste al baseline emocional derivado del sueno",
    )


def default_autobiographical_state() -> AutobiographicalState:
    """Factory para estado autobiografico por defecto (OFF)."""
    return AutobiographicalState()
