"""External Signals — Models for real-world stimulus integration.

Defines the data structures for external signals (physiological, behavioral,
environmental) that can be fused into the emotional pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---- Signal Source Definition ----

SIGNAL_SOURCES: dict[str, dict[str, str | float]] = {
    "facial_au": {
        "label": "Facial Action Units",
        "description": "Facial expression analysis (AU coding)",
        "base_weight": 0.6,
        "category": "behavioral",
    },
    "keyboard_dynamics": {
        "label": "Keyboard Dynamics",
        "description": "Typing speed, pressure, pause patterns",
        "base_weight": 0.5,
        "category": "behavioral",
    },
    "time_of_day": {
        "label": "Time of Day",
        "description": "Circadian influence on mood (subtle)",
        "base_weight": 0.3,
        "category": "environmental",
    },
    "weather": {
        "label": "Weather",
        "description": "Weather-based mood modulation (Schwarz 1983)",
        "base_weight": 0.2,
        "category": "environmental",
    },
}


class SignalSourceConfig(BaseModel):
    """Configuration for a single signal source."""

    enabled: bool = Field(default=False, description="Whether this signal source is active")
    valence_hint: float = Field(default=0.0, ge=-1.0, le=1.0, description="Valence direction (-1 to 1)")
    arousal_hint: float = Field(default=0.5, ge=0.0, le=1.0, description="Arousal level (0 to 1)")
    dominance_hint: float | None = Field(default=None, ge=0.0, le=1.0, description="Dominance (0 to 1), optional")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Signal reliability (0 to 1)")


class ExternalSignalsConfig(BaseModel):
    """Session-level configuration for all external signal sources.

    Each source can be toggled independently. Only enabled sources
    are injected into the emotional pipeline.
    """

    enabled: bool = Field(default=False, description="Master toggle — if False, no signals processed")
    sources: dict[str, SignalSourceConfig] = Field(
        default_factory=lambda: {name: SignalSourceConfig() for name in SIGNAL_SOURCES},
    )

    @property
    def active_sources(self) -> dict[str, SignalSourceConfig]:
        """Return only enabled sources (respects master toggle)."""
        if not self.enabled:
            return {}
        return {name: cfg for name, cfg in self.sources.items() if cfg.enabled}

    @property
    def active_count(self) -> int:
        return len(self.active_sources)


def default_signals_config() -> ExternalSignalsConfig:
    """Default config: all signals disabled."""
    return ExternalSignalsConfig()


class ProcessedSignal(BaseModel):
    """Result of processing a single external signal."""

    source: str
    valence_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)


class FusedSignalResult(BaseModel):
    """Result of fusing multiple external signals into a single modulation."""

    valence_modulation: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal_modulation: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance_modulation: float = Field(default=0.0, ge=-1.0, le=1.0)
    total_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    signal_count: int = 0
    contributions: list[ProcessedSignal] = Field(default_factory=list)

    @property
    def has_effect(self) -> bool:
        """True if any modulation is non-zero."""
        return (
            abs(self.valence_modulation) > 1e-6
            or abs(self.arousal_modulation) > 1e-6
            or abs(self.dominance_modulation) > 1e-6
        )
