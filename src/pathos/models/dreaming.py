"""Dreaming / Oniric Consolidation - Modelos de datos.

Pilar 3 de ANIMA (Paso 3.2): Consolidacion onirica.
Basado en Matthew Walker (Why We Sleep, 2017),
Allan Hobson (AIM Model, 2000),
Robert Stickgold (Sleep and Memory Consolidation, 2005).

Al cerrar sesion, un proceso de "sueno" ejecuta 5 fases:
  1. Replay Emocional (SWS analogue) — re-procesar episodios intensos
  2. Asociacion Libre (REM analogue) — vincular por similitud emocional
  3. Generalizacion — comprimir clusters en narrativas
  4. Procesamiento Traumatico — reducir impacto gradualmente
  5. Dream Report — generar narrativa poetica/surrealista

Sistema OPT-IN (parte del sistema autobiografico).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DreamThemeType(str, Enum):
    """Tipo de tema dominante en el sueno."""

    CONFLICT = "conflict"           # Temas de conflicto, tension
    GROWTH = "growth"               # Temas de crecimiento, aprendizaje
    LOSS = "loss"                   # Temas de perdida, separacion
    CONNECTION = "connection"       # Temas de vinculo, calidez
    FEAR = "fear"                   # Temas de amenaza, vulnerabilidad
    DISCOVERY = "discovery"         # Temas de exploracion, novedad
    RESOLUTION = "resolution"       # Temas de resolucion, paz


class TraumaProcessingStage(str, Enum):
    """Etapa del procesamiento traumatico."""

    RAW = "raw"                 # Sin procesar
    PROCESSING = "processing"   # En proceso de reduccion
    INTEGRATED = "integrated"   # Integrado, ya no activa immune


# ---------------------------------------------------------------------------
# Phase 1: Replay Emocional
# ---------------------------------------------------------------------------

class ReprocessedEpisode(BaseModel):
    """Episodio re-procesado durante el replay emocional (SWS).

    El appraisal se re-ejecuta con 'distancia temporal', lo que
    permite reducir la intensidad y recontextualizar.
    """

    episode_id: str = Field(description="ID del episodio original")
    original_intensity: float = Field(ge=0, le=1)
    reprocessed_intensity: float = Field(ge=0, le=1, description="Intensidad despues del replay")
    original_valence: float = Field(ge=-1, le=1)
    reprocessed_valence: float = Field(ge=-1, le=1, description="Valencia ajustada post-replay")
    intensity_reduction: float = Field(ge=0, le=1, description="Cuanto se redujo (fraction)")
    is_traumatic: bool = Field(default=False, description="Valence < -0.6 AND intensity > 0.8")


# ---------------------------------------------------------------------------
# Phase 2: Asociacion Libre
# ---------------------------------------------------------------------------

class EmotionalLink(BaseModel):
    """Vinculo emocional entre dos episodios descubierto en REM.

    Conecta episodios por SIMILITUD EMOCIONAL (no semantica).
    Genera insight: "Cuando X me senti igual que cuando Y — Z".
    """

    episode_a_id: str = Field(description="Primer episodio")
    episode_b_id: str = Field(description="Segundo episodio")
    emotional_distance: float = Field(ge=0, le=1, description="Distancia emocional (menor=mas similar)")
    shared_dimensions: list[str] = Field(default_factory=list, description="Dimensiones similares: valence, arousal, etc.")
    insight: str = Field(default="", description="Insight generado de la conexion")


# ---------------------------------------------------------------------------
# Phase 3: Generalizacion (reutiliza NarrativeStatement de autobio_memory)
# ---------------------------------------------------------------------------

# Phase 3 usa attempt_narrative_formation del engine/autobio_memory.py
# No necesita modelos propios — produce NarrativeStatements


# ---------------------------------------------------------------------------
# Phase 4: Procesamiento Traumatico
# ---------------------------------------------------------------------------

class ProcessedTrauma(BaseModel):
    """Resultado del procesamiento de un episodio traumatico.

    Los episodios traumaticos (valence < -0.6, intensity > 0.8) se
    re-evaluan gradualmente. Cada sesion reduce intensidad ~10%.
    Objetivo: que el agente pueda 'hablar de' traumas sin activar immune.
    """

    episode_id: str
    original_intensity: float = Field(ge=0, le=1)
    processed_intensity: float = Field(ge=0, le=1)
    stage: TraumaProcessingStage = TraumaProcessingStage.RAW
    sessions_processed: int = Field(default=0, ge=0, description="Sesiones de procesamiento acumuladas")
    immune_dampening_reduction: float = Field(
        default=0.0, ge=0, le=1,
        description="Cuanto reducir el reactivity_dampening del immune para este tema",
    )


# ---------------------------------------------------------------------------
# Phase 5: Dream Report
# ---------------------------------------------------------------------------

class DreamTheme(BaseModel):
    """Tema emocional dominante en el sueno."""

    theme_type: DreamThemeType
    emotion: str = Field(description="Emocion asociada al tema")
    weight: float = Field(default=0.0, ge=0, le=1, description="Peso del tema en el sueno")
    source_episode_ids: list[str] = Field(default_factory=list)


class DreamReport(BaseModel):
    """Reporte del sueno — narrativa poetica que refleja temas emocionales.

    Se presenta al usuario al inicio de la siguiente sesion.
    El agente puede reflexionar sobre su propio sueno.
    """

    narrative: str = Field(default="", description="Texto poetico, surrealista del sueno")
    themes: list[DreamTheme] = Field(default_factory=list)
    emotional_signature: dict[str, float] = Field(
        default_factory=dict,
        description="Firma emocional del sueno: {emotion: weight}",
    )
    baseline_adjustment: dict[str, float] = Field(
        default_factory=dict,
        description="Ajuste al baseline emocional: {valence: delta, arousal: delta}",
    )
    session_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Consolidation Result (output completo del proceso onirico)
# ---------------------------------------------------------------------------

class ConsolidationResult(BaseModel):
    """Resultado completo de la consolidacion onirica de una sesion."""

    # Phase 1
    replayed_episodes: list[ReprocessedEpisode] = Field(default_factory=list)
    total_intensity_reduced: float = Field(default=0.0, ge=0, description="Reduccion total de intensidad")

    # Phase 2
    emotional_links: list[EmotionalLink] = Field(default_factory=list)
    new_connections: int = Field(default=0, ge=0)

    # Phase 3
    narratives_formed: int = Field(default=0, ge=0)
    narratives_reinforced: int = Field(default=0, ge=0)

    # Phase 4
    traumas_processed: list[ProcessedTrauma] = Field(default_factory=list)

    # Phase 5
    dream_report: DreamReport = Field(default_factory=DreamReport)

    # Summary
    episodes_processed: int = Field(default=0, ge=0)
    session_id: str = Field(default="")
