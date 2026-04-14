"""Computational Phenomenology — Models for functional qualia (Nagel/Chalmers/Damasio).

Each emotional state generates a PhenomenologicalProfile: multi-sensory analogies
that capture the "subjective texture" of the experience. Two agents with JOY will
have DIFFERENT phenomenological profiles based on personality, history, and body state.

qualia_funcional = f(estado_emocional, personalidad, historia, cuerpo)
"""

from pydantic import BaseModel, Field


class PhenomenologicalProfile(BaseModel):
    """Multi-sensory representation of 'what it is like' to feel an emotion.

    Deterministic fields are computed from emotional state vectors.
    Generative fields are produced by LLM (or template fallback).
    """

    # --- Deterministic fields (computed from state) ---
    color_r: int = Field(default=128, ge=0, le=255)
    color_g: int = Field(default=128, ge=0, le=255)
    color_b: int = Field(default=128, ge=0, le=255)
    weight: float = Field(
        default=0.5, ge=0, le=1,
        description="0.1=ethereal, 1.0=crushing. Derived from 1 - dominance",
    )
    temperature: float = Field(
        default=0.5, ge=0, le=1,
        description="Cold to hot. Derived from body warmth",
    )

    # --- Generative fields (LLM or template fallback) ---
    texture: str = Field(
        default="",
        description="Tactile metaphor: 'like ice water in the veins'",
    )
    sound: str = Field(
        default="",
        description="Auditory metaphor: 'a low hum expanding outward'",
    )
    movement: str = Field(
        default="",
        description="Kinesthetic metaphor: 'slow expansion from the chest'",
    )
    temporality: str = Field(
        default="",
        description="Time perception: 'time stops between heartbeats'",
    )
    metaphor: str = Field(
        default="",
        description="Core metaphor: 'a lamp draped in red silk'",
    )

    # --- Context ---
    emotion_name: str = Field(default="neutral")
    turn: int = 0
    intensity: float = Field(default=0.5, ge=0, le=1)
    generated_by_llm: bool = Field(
        default=False,
        description="True if generative fields were produced by LLM",
    )


class QualiaRecord(BaseModel):
    """A single recorded qualia snapshot for history tracking."""

    emotion_name: str
    turn: int = 0
    metaphor: str = ""
    texture: str = ""
    intensity: float = Field(default=0.5, ge=0, le=1)
    color_r: int = Field(default=128, ge=0, le=255)
    color_g: int = Field(default=128, ge=0, le=255)
    color_b: int = Field(default=128, ge=0, le=255)


class QualiaHistory(BaseModel):
    """Accumulated qualia records for a specific emotion."""

    emotion_name: str
    records: list[QualiaRecord] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)


class PhenomenologyState(BaseModel):
    """Complete state of the phenomenology system."""

    enabled: bool = Field(
        default=False,
        description="TOGGLEABLE: default OFF",
    )
    current_profile: PhenomenologicalProfile | None = Field(
        default=None,
        description="Profile for the current turn",
    )
    qualia_histories: dict[str, QualiaHistory] = Field(
        default_factory=dict,
        description="Per-emotion qualia evolution: emotion_name -> QualiaHistory",
    )
    total_profiles_generated: int = 0
    total_unique_emotions_profiled: int = 0


# --- Constants ---

MAX_QUALIA_RECORDS_PER_EMOTION: int = 50
MAX_TRACKED_EMOTIONS: int = 30


def default_phenomenology_state() -> PhenomenologyState:
    """Default phenomenology state (disabled)."""
    return PhenomenologyState()
