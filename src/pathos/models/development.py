"""Desarrollo Ontogenico - Modelos de datos.

Pilar 4 de ANIMA: El agente crece con la experiencia.
Basado en Jean Piaget (Desarrollo Cognitivo, 1952),
Erik Erikson (Desarrollo Psicosocial, 1959),
Lev Vygotsky (ZPD, 1978),
Lawrence Kohlberg (Desarrollo Moral, 1958).

5 etapas de desarrollo emocional, cada una desbloquea sistemas:
  1. SENSORIMOTOR (0-50) — emociones basicas, alta reactividad
  2. PREOPERATIONAL (50-200) — emociones complejas, schemas, contagion
  3. CONCRETE_OPERATIONAL (200-500) — 19 emociones, meta, regulacion
  4. FORMAL_OPERATIONAL (500-1500) — todos los sistemas avanzados
  5. POST_FORMAL (1500+) — sabiduria, descubrimiento, fenomenologia

Sistema TOGGLEABLE (default OFF). Si esta OFF, todos los sistemas
activos como en v4 (sin gating por etapa).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DevelopmentStage(str, Enum):
    """Etapas de desarrollo emocional."""

    SENSORIMOTOR = "sensorimotor"              # 0-50 turnos
    PREOPERATIONAL = "preoperational"          # 50-200 turnos
    CONCRETE_OPERATIONAL = "concrete_operational"  # 200-500 turnos
    FORMAL_OPERATIONAL = "formal_operational"  # 500-1500 turnos
    POST_FORMAL = "post_formal"                # 1500+ turnos


class DevelopmentSpeed(str, Enum):
    """Velocidad de desarrollo."""

    GLACIAL = "glacial"          # x0.25
    NATURAL = "natural"          # x1.0
    ACCELERATED = "accelerated"  # x4.0
    FAST = "fast"                # x10.0
    CUSTOM = "custom"            # user-defined


class TransitionMode(str, Enum):
    """Modo de transicion entre etapas."""

    AUTO = "auto"      # Transiciona automaticamente al cumplir criterios
    MANUAL = "manual"  # Espera aprobacion del usuario


# ---------------------------------------------------------------------------
# Speed multipliers
# ---------------------------------------------------------------------------

SPEED_MULTIPLIERS: dict[DevelopmentSpeed, float] = {
    DevelopmentSpeed.GLACIAL: 0.25,
    DevelopmentSpeed.NATURAL: 1.0,
    DevelopmentSpeed.ACCELERATED: 4.0,
    DevelopmentSpeed.FAST: 10.0,
}

# ---------------------------------------------------------------------------
# Base thresholds (NATURAL speed, x1.0)
# ---------------------------------------------------------------------------

BASE_THRESHOLDS: dict[DevelopmentStage, int] = {
    DevelopmentStage.SENSORIMOTOR: 0,
    DevelopmentStage.PREOPERATIONAL: 50,
    DevelopmentStage.CONCRETE_OPERATIONAL: 200,
    DevelopmentStage.FORMAL_OPERATIONAL: 500,
    DevelopmentStage.POST_FORMAL: 1500,
}

# Stage order for progression
STAGE_ORDER: list[DevelopmentStage] = [
    DevelopmentStage.SENSORIMOTOR,
    DevelopmentStage.PREOPERATIONAL,
    DevelopmentStage.CONCRETE_OPERATIONAL,
    DevelopmentStage.FORMAL_OPERATIONAL,
    DevelopmentStage.POST_FORMAL,
]

# ---------------------------------------------------------------------------
# Emotions available per stage
# ---------------------------------------------------------------------------

# 6 basic emotions for stage 1
STAGE_1_EMOTIONS: set[str] = {
    "joy", "sadness", "fear", "anger", "surprise", "contentment",
}

# Stage 2 adds complex emotions
STAGE_2_EMOTIONS: set[str] = STAGE_1_EMOTIONS | {
    "gratitude", "hope", "frustration", "anxiety",
}

# Stage 3: all 19 primary emotions
STAGE_3_EMOTIONS: set[str] = STAGE_2_EMOTIONS | {
    "excitement", "relief", "helplessness", "disappointment",
    "alertness", "contemplation", "indifference", "mixed", "neutral",
}

# Stage 4+: same 19 + emergent emotions (handled by generator)
STAGE_4_EMOTIONS: set[str] = STAGE_3_EMOTIONS

# ---------------------------------------------------------------------------
# Systems gated per stage
# Systems listed are CUMULATIVE: each stage includes all previous
# ---------------------------------------------------------------------------

# Systems unlocked at each stage (in addition to all previous)
SYSTEMS_UNLOCKED_AT: dict[DevelopmentStage, set[str]] = {
    DevelopmentStage.SENSORIMOTOR: {
        "appraisal",
        "generator",
        "homeostasis",
        "body_state",
    },
    DevelopmentStage.PREOPERATIONAL: {
        "schemas",
        "contagion",
        "mood",
        "needs",
    },
    DevelopmentStage.CONCRETE_OPERATIONAL: {
        "meta_emotion",
        "regulation",
        "social",
        "forecasting",
        "self_inquiry",
        "temporal",
        "drives",
    },
    DevelopmentStage.FORMAL_OPERATIONAL: {
        "reappraisal",
        "creativity",
        "immune",
        "narrative",
        "somatic",
        "workspace",
    },
    DevelopmentStage.POST_FORMAL: {
        "discovery",
        "phenomenology",
        "dialectical",
    },
}


def get_cumulative_systems(stage: DevelopmentStage) -> set[str]:
    """Returns all systems available at a given stage (cumulative)."""
    systems: set[str] = set()
    for s in STAGE_ORDER:
        systems |= SYSTEMS_UNLOCKED_AT[s]
        if s == stage:
            break
    return systems


def get_emotions_for_stage(stage: DevelopmentStage) -> set[str]:
    """Returns the set of emotion names available at a given stage."""
    if stage == DevelopmentStage.SENSORIMOTOR:
        return STAGE_1_EMOTIONS
    elif stage == DevelopmentStage.PREOPERATIONAL:
        return STAGE_2_EMOTIONS
    elif stage == DevelopmentStage.CONCRETE_OPERATIONAL:
        return STAGE_3_EMOTIONS
    else:
        return STAGE_4_EMOTIONS


# ---------------------------------------------------------------------------
# Qualitative transition criteria (base values for NATURAL speed)
# ---------------------------------------------------------------------------

class TransitionCriteria(BaseModel):
    """Criterios cualitativos para transicionar a la siguiente etapa."""

    min_experience: int = 0
    min_distinct_emotions: int = 0
    min_high_intensity_episodes: int = 0
    min_schemas_formed: int = 0
    min_episodic_memories: int = 0
    min_identities: int = 0
    min_identity_crises_resolved: int = 0
    min_regulation_uses: int = 0


# Base criteria (NATURAL speed x1.0)
BASE_TRANSITION_CRITERIA: dict[DevelopmentStage, TransitionCriteria] = {
    # To transition FROM sensorimotor TO preoperational
    DevelopmentStage.PREOPERATIONAL: TransitionCriteria(
        min_experience=50,
        min_distinct_emotions=4,
        min_high_intensity_episodes=1,
    ),
    # To transition FROM preoperational TO concrete_operational
    DevelopmentStage.CONCRETE_OPERATIONAL: TransitionCriteria(
        min_experience=200,
        min_schemas_formed=3,
        min_episodic_memories=10,
    ),
    # To transition FROM concrete_operational TO formal_operational
    DevelopmentStage.FORMAL_OPERATIONAL: TransitionCriteria(
        min_experience=500,
        min_identities=5,
        min_identity_crises_resolved=1,
        min_regulation_uses=20,
    ),
    # To transition FROM formal_operational TO post_formal
    DevelopmentStage.POST_FORMAL: TransitionCriteria(
        min_experience=1500,
    ),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DevelopmentConfig(BaseModel):
    """Configuracion del sistema de desarrollo."""

    speed: DevelopmentSpeed = DevelopmentSpeed.NATURAL
    speed_multiplier: float = Field(default=1.0, ge=0.1, le=20.0)
    initial_stage: DevelopmentStage = DevelopmentStage.SENSORIMOTOR
    transition_mode: TransitionMode = TransitionMode.AUTO


# ---------------------------------------------------------------------------
# Transition event
# ---------------------------------------------------------------------------

class TransitionEvent(BaseModel):
    """Registro de una transicion de etapa."""

    from_stage: DevelopmentStage
    to_stage: DevelopmentStage
    at_experience: int
    turn_number: int = 0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DevelopmentState(BaseModel):
    """Estado del desarrollo ontogenico de un agente."""

    enabled: bool = False
    config: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    current_stage: DevelopmentStage = DevelopmentStage.SENSORIMOTOR
    total_experience: int = 0  # turnos acumulados cross-session

    # Tracking cualitativo
    distinct_emotions_experienced: set[str] = Field(default_factory=set)
    high_intensity_episodes: int = 0
    regulation_uses: int = 0

    # Transition
    transition_history: list[TransitionEvent] = Field(default_factory=list)
    pending_transition: DevelopmentStage | None = None  # For manual mode

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self, **kwargs: object) -> dict:
        """Override to serialize sets as sorted lists."""
        d = super().model_dump(**kwargs)
        if isinstance(d.get("distinct_emotions_experienced"), set):
            d["distinct_emotions_experienced"] = sorted(d["distinct_emotions_experienced"])
        return d


def default_development_state() -> DevelopmentState:
    """Creates a default (disabled) development state."""
    return DevelopmentState()
