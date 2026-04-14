"""Motivational Autonomy — Drive models (Panksepp + Deci/Ryan + Berridge).

4 drives afectivos primarios que generan deseos con consecuencias emocionales:
- SEEKING: curiosidad, exploración (openness)
- CARE: conexión, ayuda, empatía (agreeableness)
- PLAY: juego, humor, creatividad (extraversion)
- PANIC_GRIEF: apego, separación, vínculo (neuroticism * rapport)

Los drives generan goals con stake emocional. Lograr/fallar un goal
produce emociones reales proporcionales al stake.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Drive(str, Enum):
    """Drives afectivos primarios (Panksepp adaptado)."""

    SEEKING = "seeking"
    CARE = "care"
    PLAY = "play"
    PANIC_GRIEF = "panic_grief"


class GoalStatus(str, Enum):
    """Estado de un goal autónomo."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class GoalOutcome(str, Enum):
    """Resultado de la resolución de un goal."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class Goal(BaseModel):
    """Goal autónomo generado por un drive."""

    drive: Drive
    description: str
    stake: float = Field(
        default=0.3, ge=0, le=1,
        description="Importancia emocional del goal (0=trivial, 1=crucial)",
    )
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = Field(
        default=0.0, ge=0, le=1,
        description="Progreso estimado hacia el goal",
    )
    created_turn: int = 0
    deadline_turns: int = Field(
        default=10, ge=1,
        description="Turnos antes de que el goal expire",
    )


class DriveState(BaseModel):
    """Estado de un drive individual."""

    drive: Drive
    intensity: float = Field(
        default=0.3, ge=0, le=1,
        description="Intensidad actual del drive (0=dormido, 1=urgente)",
    )
    satisfaction: float = Field(
        default=0.5, ge=0, le=1,
        description="Nivel de satisfacción reciente (0=frustrado, 1=satisfecho)",
    )
    last_satisfied_turn: int = 0
    urgency: float = Field(
        default=0.0, ge=0, le=1,
        description="Urgencia computada (intensity * insatisfacción * tiempo)",
    )
    activation_count: int = Field(
        default=0, ge=0,
        description="Veces que este drive se ha activado en la sesión",
    )


class DriveUpdate(BaseModel):
    """Resultado de actualizar un drive en un turno."""

    drive: Drive
    previous_intensity: float
    new_intensity: float
    satisfaction_delta: float
    urgency: float
    triggered: bool = Field(
        default=False,
        description="Si el drive generó activación este turno",
    )
    frustration: bool = Field(
        default=False,
        description="Si el drive está frustrado (insatisfecho mucho tiempo)",
    )


class EmotionalImpact(BaseModel):
    """Impacto emocional de la resolución de un goal."""

    valence_delta: float = Field(default=0.0, ge=-1, le=1)
    arousal_delta: float = Field(default=0.0, ge=-0.5, le=0.5)
    intensity_boost: float = Field(default=0.0, ge=0, le=0.5)
    emotion_tag: str = ""
    description: str = ""


class DrivesState(BaseModel):
    """Estado completo del sistema de drives."""

    enabled: bool = Field(
        default=False,
        description="TOGGLEABLE: default OFF",
    )
    drives: dict[str, DriveState] = Field(default_factory=dict)
    active_goals: list[Goal] = Field(
        default_factory=list,
        description="Goals activos (max 3)",
    )
    resolved_goals: list[Goal] = Field(
        default_factory=list,
        description="Goals resueltos recientemente (para research)",
    )
    total_goals_completed: int = 0
    total_goals_failed: int = 0


# --- Constantes ---

MAX_ACTIVE_GOALS: int = 3
FRUSTRATION_THRESHOLD_TURNS: int = 10
URGENCY_TRIGGER_THRESHOLD: float = 0.5
DRIVE_DECAY_RATE: float = 0.02
SATISFACTION_DECAY_RATE: float = 0.03

# Base intensities por personalidad (del plan AGIREWORK.txt)
DRIVE_PERSONALITY_MAP: dict[Drive, tuple[str, float]] = {
    Drive.SEEKING: ("openness", 0.8),
    Drive.CARE: ("agreeableness", 0.8),
    Drive.PLAY: ("extraversion", 0.6),
    Drive.PANIC_GRIEF: ("neuroticism", 0.5),
}

# Keywords que activan cada drive
DRIVE_KEYWORDS: dict[Drive, list[str]] = {
    Drive.SEEKING: [
        "why", "how", "what", "explore", "curious", "wonder", "interesting",
        "explain", "understand", "theory", "hypothesis", "discover", "learn",
        "idea", "concept", "imagine", "question", "investigate", "deep",
        "por qué", "cómo", "explorar", "curioso", "interesante", "entender",
        "descubrir", "aprender", "idea", "investigar", "profundo",
    ],
    Drive.CARE: [
        "help", "sad", "struggling", "confused", "lost", "need", "please",
        "worry", "afraid", "anxious", "stress", "problem", "difficult",
        "support", "advice", "guide", "stuck", "overwhelm",
        "ayuda", "triste", "perdido", "necesito", "preocupado", "difícil",
        "consejo", "estresado", "problema", "apoyo",
    ],
    Drive.PLAY: [
        "fun", "joke", "funny", "haha", "lol", "game", "play", "creative",
        "silly", "imagine", "what if", "random", "wild", "crazy", "cool",
        "jaja", "divertido", "juego", "creativo", "imagina", "genial",
    ],
    Drive.PANIC_GRIEF: [
        "goodbye", "bye", "leaving", "last", "end", "forget", "gone",
        "miss", "never", "stop", "close", "done", "finish", "over",
        "adiós", "último", "final", "olvidar", "terminar", "cerrar",
    ],
}

# Satisfaction keywords por drive
SATISFACTION_KEYWORDS: dict[Drive, list[str]] = {
    Drive.SEEKING: [
        "interesting", "wow", "never thought", "good point", "makes sense",
        "insight", "fascinating", "brilliant", "clever", "exactly",
        "interesante", "buen punto", "tiene sentido", "fascinante",
    ],
    Drive.CARE: [
        "thank", "thanks", "helpful", "better", "solved", "fixed", "great",
        "perfect", "appreciate", "gracias", "genial", "resuelto", "mejor",
    ],
    Drive.PLAY: [
        "haha", "lol", "funny", "love it", "nice one", "clever",
        "jaja", "buena", "me encanta", "gracioso",
    ],
    Drive.PANIC_GRIEF: [
        "back", "hello again", "missed", "remember", "continue", "more",
        "volví", "hola de nuevo", "recuerdas", "continuar", "seguimos",
    ],
}


def default_drives_state() -> DrivesState:
    """Estado inicial de drives (desactivado, con drives base)."""
    drives = {}
    for drive in Drive:
        drives[drive.value] = DriveState(
            drive=drive,
            intensity=0.3,
            satisfaction=0.5,
        )
    return DrivesState(drives=drives)
