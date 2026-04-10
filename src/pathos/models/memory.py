"""Emotional Memory - Memorias emocionales del agente."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class EmotionalMemory(BaseModel):
    """Una memoria emocional individual."""

    id: str
    stimulus: str
    intensity_at_time: float = Field(ge=0, le=1)
    valence_at_time: float = Field(ge=-1, le=1)
    primary_emotion: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Keywords extraidas del estimulo para matching fallback
    keywords: list[str] = Field(default_factory=list)

    # Embedding vector para busqueda semantica (Fase 3)
    embedding: list[float] = Field(default_factory=list)
