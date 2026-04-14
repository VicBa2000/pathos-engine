"""Emotional Discovery — Models for emergent emotion detection (Barrett).

Each agent develops a unique emotional vocabulary through experience.
Novel emotional states that don't map to known prototypes are detected,
clustered, and named — creating emotions the programmer never anticipated.
"""

from pydantic import BaseModel, Field


class EmotionalVector(BaseModel):
    """4D emotional coordinates (same as generator prototypes)."""

    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)
    dominance: float = Field(ge=0, le=1)
    certainty: float = Field(ge=0, le=1)


class BodySignature(BaseModel):
    """Body state signature for a discovered emotion."""

    tension: float = Field(default=0.5, ge=0, le=1)
    energy: float = Field(default=0.5, ge=0, le=1)
    openness: float = Field(default=0.5, ge=0, le=1)
    warmth: float = Field(default=0.5, ge=0, le=1)


class NovelEmotionalState(BaseModel):
    """A single instance of an emotional state that didn't match any prototype."""

    vector: EmotionalVector
    body: BodySignature
    context: str = Field(
        default="",
        description="Stimulus/context where this state was experienced",
    )
    min_distance: float = Field(
        default=0.0, ge=0,
        description="Distance to closest known prototype",
    )
    closest_known: str = Field(
        default="neutral",
        description="Name of closest known emotion",
    )
    turn: int = 0
    intensity: float = Field(default=0.5, ge=0, le=1)


class DiscoveredEmotion(BaseModel):
    """An emotion discovered through clustering of novel states."""

    name: str = Field(
        default="unnamed",
        description="Neologism invented by the agent",
    )
    description: str = Field(
        default="",
        description="One-sentence description of the emotion",
    )
    vector: EmotionalVector
    body_signature: BodySignature
    contexts: list[str] = Field(
        default_factory=list,
        description="Contexts where this emotion was experienced",
    )
    first_experienced_turn: int = 0
    frequency: int = Field(
        default=0, ge=0,
        description="Times this emotion has been experienced",
    )
    named: bool = Field(
        default=False,
        description="Whether the agent has named this emotion",
    )
    cluster_size: int = Field(
        default=0, ge=0,
        description="Number of novel states that formed this cluster",
    )


class DiscoveryState(BaseModel):
    """Complete state of the emotional discovery system."""

    enabled: bool = Field(
        default=False,
        description="TOGGLEABLE: default OFF",
    )
    novel_history: list[NovelEmotionalState] = Field(
        default_factory=list,
        description="Buffer of novel states awaiting clustering",
    )
    discovered_emotions: list[DiscoveredEmotion] = Field(
        default_factory=list,
        description="Emotions discovered through clustering",
    )
    total_novel_detected: int = 0
    total_emotions_discovered: int = 0
    last_detection_turn: int = 0


# --- Constants ---

NOVELTY_THRESHOLD: float = 0.35
CLUSTER_DISTANCE_THRESHOLD: float = 0.25
MIN_CLUSTER_SIZE: int = 3
MAX_NOVEL_BUFFER: int = 50
MAX_DISCOVERED_EMOTIONS: int = 20
MAX_CONTEXTS_PER_EMOTION: int = 5


def default_discovery_state() -> DiscoveryState:
    """Default discovery state (disabled)."""
    return DiscoveryState()
