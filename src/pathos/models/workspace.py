"""Global Workspace - Modelos de datos.

Pilar 2 de ANIMA: Consciencia funcional via competicion.
Basado en Bernard Baars (Global Workspace Theory, 1988),
Stanislas Dehaene (Global Neuronal Workspace, 2014),
Giulio Tononi (IIT, 2004) — principio de integracion informacional.

El pipeline genera candidatos desde cada sistema emocional. Los candidatos
compiten por un workspace limitado (top-K=5). Solo lo que entra al workspace
es "consciente" para el agente. El resto opera en el preconsciente:
influye indirectamente (priming, mood, somatic echo) pero el agente
no "sabe" que lo tiene.

Sistema TOGGLEABLE (default OFF). Si esta OFF, todos los sistemas
contribuyen directamente como en v4 (sin filtro de consciencia).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Workspace Candidate
# ---------------------------------------------------------------------------

class WorkspaceCandidate(BaseModel):
    """Un candidato que compite por entrar al workspace consciente.

    Cada sistema emocional genera 0-1 candidatos por turno.
    La saliency determina su probabilidad de entrar al workspace.
    """

    source: str = Field(description="Sistema que genera el candidato (ej: 'appraisal', 'schema')")
    content: str = Field(description="Descripcion textual del estado/insight")
    urgency: float = Field(default=0.0, ge=0, le=1, description="Amenaza inmediata=1, reflexion=0.2")
    relevance: float = Field(default=0.0, ge=0, le=1, description="Conexion con el estimulo actual")
    emotional_intensity: float = Field(default=0.0, ge=0, le=1, description="Intensidad emocional")

    # Computado: urgency * relevance * emotional_intensity
    saliency: float = Field(default=0.0, ge=0, le=1, description="Saliency total (computado)")

    # Metadata para coaliciones
    emotion_tag: str = Field(default="neutral", description="Emocion asociada (para coaliciones)")
    category: str = Field(default="general", description="Categoria semantica (para coaliciones)")

    # Persistencia en preconsciente
    preconscious_turns: int = Field(default=0, ge=0, description="Turnos en el preconsciente")

    def compute_saliency(self) -> float:
        """Computa y actualiza la saliency."""
        self.saliency = round(self.urgency * self.relevance * self.emotional_intensity, 4)
        return self.saliency


# ---------------------------------------------------------------------------
# Coalition (candidatos que se refuerzan mutuamente)
# ---------------------------------------------------------------------------

class Coalition(BaseModel):
    """Grupo de candidatos que se refuerzan mutuamente.

    Candidatos con misma emocion, misma categoria o misma fuente
    forman coaliciones. La coalicion tiene un bonus de coherencia
    que amplifica su saliency total.
    """

    members: list[WorkspaceCandidate] = Field(default_factory=list)
    total_saliency: float = Field(default=0.0, ge=0, description="Suma de saliencies de miembros")
    coherence: float = Field(default=1.0, ge=1.0, le=2.0, description="Bonus de coherencia (1.0-2.0)")
    effective_saliency: float = Field(default=0.0, ge=0, description="total_saliency * coherence")
    bond_type: str = Field(default="none", description="Tipo de vinculo: emotion, category, source")

    def compute_effective_saliency(self) -> float:
        """Computa saliency efectiva con bonus de coherencia."""
        self.total_saliency = sum(m.saliency for m in self.members)
        self.effective_saliency = round(self.total_saliency * self.coherence, 4)
        return self.effective_saliency


# ---------------------------------------------------------------------------
# Workspace Result
# ---------------------------------------------------------------------------

class WorkspaceResult(BaseModel):
    """Resultado de la competicion por el workspace.

    conscious: candidatos que entraron al workspace (top-K).
    preconscious: candidatos que no entraron (el resto).
    integration_score: cuanto se conectan los contenidos entre si (IIT).
    """

    conscious: list[WorkspaceCandidate] = Field(
        default_factory=list,
        description="Candidatos en el workspace consciente (max K)",
    )
    preconscious: list[WorkspaceCandidate] = Field(
        default_factory=list,
        description="Candidatos en el buffer preconsciente",
    )
    coalitions_formed: int = Field(
        default=0, ge=0,
        description="Numero de coaliciones formadas en la competicion",
    )
    integration_score: float = Field(
        default=0.0, ge=0, le=1,
        description="Integracion informacional del workspace (IIT-inspired)",
    )
    workspace_stability: float = Field(
        default=0.0, ge=0, le=1,
        description="Cuanto cambio el workspace respecto al turno anterior",
    )
    total_candidates: int = Field(default=0, ge=0)
    filtered_noise: int = Field(default=0, ge=0, description="Candidatos descartados por ruido")


# ---------------------------------------------------------------------------
# Preconscious Buffer
# ---------------------------------------------------------------------------

class PreconsciousBuffer(BaseModel):
    """Buffer de candidatos preconscientes con persistencia temporal.

    Los candidatos persisten entre turnos. Su saliency se incrementa
    progresivamente (+0.05/turno) hasta que eventualmente irrumpen
    en el workspace.
    """

    candidates: list[WorkspaceCandidate] = Field(default_factory=list)
    max_size: int = Field(default=20, ge=5)

    # Contribuciones indirectas del preconsciente
    mood_valence_contribution: float = Field(
        default=0.0, ge=-1, le=1,
        description="Contribucion del preconsciente al mood (30% del peso consciente)",
    )
    mood_arousal_contribution: float = Field(
        default=0.0, ge=0, le=1,
        description="Contribucion del preconsciente al arousal del mood",
    )
    somatic_tension_echo: float = Field(
        default=0.0, ge=0, le=1,
        description="Eco somatico: tension corporal sin causa consciente",
    )

    def add_candidates(self, new_candidates: list[WorkspaceCandidate]) -> None:
        """Agrega candidatos al buffer, incrementando persistencia de existentes."""
        # Incrementar turns de los que ya estaban
        for existing in self.candidates:
            existing.preconscious_turns += 1

        # Agregar nuevos
        for c in new_candidates:
            # Si ya existe uno del mismo source, actualizar
            replaced = False
            for i, existing in enumerate(self.candidates):
                if existing.source == c.source:
                    self.candidates[i] = c
                    replaced = True
                    break
            if not replaced:
                self.candidates.append(c)

        # Limitar tamaño
        if len(self.candidates) > self.max_size:
            # Eliminar los de menor saliency
            self.candidates.sort(key=lambda x: x.saliency, reverse=True)
            self.candidates = self.candidates[:self.max_size]

    def get_persistent(self, min_turns: int = 3) -> list[WorkspaceCandidate]:
        """Retorna candidatos que persisten min_turns o mas turnos."""
        return [c for c in self.candidates if c.preconscious_turns >= min_turns]

    def remove_by_source(self, source: str) -> None:
        """Elimina candidatos de un source (porque entraron al workspace)."""
        self.candidates = [c for c in self.candidates if c.source != source]


# ---------------------------------------------------------------------------
# Consciousness State (estado completo por sesion)
# ---------------------------------------------------------------------------

class ConsciousnessState(BaseModel):
    """Estado completo del sistema de consciencia funcional por sesion.

    Se anade a SessionState. TOGGLEABLE (default OFF).
    Si esta OFF, el pipeline funciona como v4 (sin filtro).
    """

    enabled: bool = Field(
        default=False,
        description="Si el Global Workspace esta activo",
    )

    # Resultado del turno actual
    current_result: WorkspaceResult | None = Field(
        default=None,
        description="Resultado de la competicion del turno actual",
    )

    # Buffer preconsciente (persiste entre turnos)
    preconscious: PreconsciousBuffer = Field(default_factory=PreconsciousBuffer)

    # Historial de integracion (para tracking)
    integration_history: list[float] = Field(
        default_factory=list,
        description="Ultimos N integration scores",
    )

    # Workspace del turno anterior (para calcular estabilidad)
    previous_workspace_sources: list[str] = Field(
        default_factory=list,
        description="Sources del workspace del turno anterior",
    )


def default_consciousness_state() -> ConsciousnessState:
    """Estado de consciencia por defecto (desactivado)."""
    return ConsciousnessState()
