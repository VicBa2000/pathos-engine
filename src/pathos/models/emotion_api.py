"""Emotion API as a Service — Request/Response models.

Standalone API that processes stimuli through the full emotional pipeline
WITHOUT requiring an LLM. Any application can use Pathos as its emotional layer.

Usage:
    POST /api/v1/emotion/process    — process a single stimulus
    POST /api/v1/emotion/batch      — process multiple stimuli
    GET  /api/v1/emotion/state      — get current session state
    POST /api/v1/emotion/configure  — configure personality/values
    POST /api/v1/emotion/reset      — reset a session
    GET  /api/v1/emotion/presets    — list personality presets
    GET  /api/v1/health             — health check
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pathos.models.emotion import EmotionalState


# ---- External Signals ----

class ExternalSignal(BaseModel):
    """A real-world signal that modulates the emotional pipeline.

    Sources can be physiological (heart rate, GSR), behavioral (typing speed,
    facial AU), environmental (time of day, weather), or custom.
    """

    source: str = Field(
        ...,
        description="Signal source identifier: 'facial_au', "
                    "'keyboard_dynamics', 'time_of_day', 'weather'",
    )
    valence_hint: float | None = Field(
        default=None, ge=-1.0, le=1.0,
        description="Suggested valence direction from this signal (-1 to 1). "
                    "None if signal doesn't carry valence information.",
    )
    arousal_hint: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Suggested arousal level from this signal (0 to 1). "
                    "None if signal doesn't carry arousal information.",
    )
    dominance_hint: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Suggested dominance from this signal (0 to 1). Optional.",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How reliable this signal is (0=noise, 1=ground truth).",
    )
    raw_data: dict[str, object] = Field(
        default_factory=dict,
        description="Raw sensor data for logging/debugging. Not used in computation.",
    )


# ---- API Configuration ----

class EmotionAPIConfig(BaseModel):
    """Configuration for what systems to activate in the API pipeline."""

    advanced_mode: bool = Field(
        default=True,
        description="Enable advanced systems (schemas, contagion, narrative, etc.)",
    )
    include_coupling: bool = Field(
        default=True,
        description="Enable cross-dimensional ODE coupling.",
    )
    include_voice_params: bool = Field(
        default=False,
        description="Generate voice parameters in response.",
    )
    include_pipeline_trace: bool = Field(
        default=False,
        description="Include detailed pipeline trace in response.",
    )
    include_behavior_prompt: bool = Field(
        default=False,
        description="Include the generated behavior modifier prompt.",
    )
    detail_level: str = Field(
        default="standard",
        pattern=r"^(minimal|standard|full)$",
        description="Response detail level: 'minimal' (state only), "
                    "'standard' (state + primary details), 'full' (everything).",
    )


# ---- Requests ----

class EmotionProcessRequest(BaseModel):
    """Process a single stimulus through the emotional pipeline."""

    stimulus: str = Field(
        ..., max_length=10000,
        description="The text stimulus to process emotionally.",
    )
    session_id: str = Field(
        default="api-default",
        pattern=r"^[\w\-]+$",
        description="Session ID for state persistence. Defaults to 'api-default'. "
                    "Use different IDs for independent emotional contexts.",
    )
    personality: dict[str, float] | None = Field(
        default=None,
        description="Partial Big Five override: {'openness': 0.8, 'neuroticism': 0.3, ...}. "
                    "Only provided fields are overridden. Applied on first request or to reconfigure.",
    )
    external_signals: list[ExternalSignal] = Field(
        default_factory=list,
        description="Real-world signals to fuse into the emotional pipeline.",
    )
    config: EmotionAPIConfig = Field(
        default_factory=EmotionAPIConfig,
        description="Pipeline configuration for this request.",
    )


class EmotionBatchRequest(BaseModel):
    """Process multiple stimuli in sequence (same session)."""

    stimuli: list[str] = Field(
        ..., min_length=1, max_length=50,
        description="List of stimuli to process sequentially (max 50).",
    )
    session_id: str = Field(
        default="api-default",
        pattern=r"^[\w\-]+$",
    )
    personality: dict[str, float] | None = None
    external_signals: list[ExternalSignal] = Field(default_factory=list)
    config: EmotionAPIConfig = Field(default_factory=EmotionAPIConfig)


class EmotionConfigureRequest(BaseModel):
    """Configure personality and values for a session."""

    session_id: str = Field(
        default="api-default",
        pattern=r"^[\w\-]+$",
    )
    personality: dict[str, float] | None = Field(
        default=None,
        description="Big Five partial override.",
    )
    values: dict[str, float] | None = Field(
        default=None,
        description="Value weight override: {'truth': 0.9, 'compassion': 0.8, ...}.",
    )
    reset_state: bool = Field(
        default=False,
        description="If True, reset emotional state to neutral after reconfiguration.",
    )


# ---- Responses ----

class ExternalSignalContribution(BaseModel):
    """How an external signal contributed to the emotional state."""

    source: str
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    dominance_delta: float = 0.0
    weight_applied: float = 0.0


class EmotionProcessResponse(BaseModel):
    """Response from processing a single stimulus."""

    # Always present
    session_id: str
    turn_number: int
    emotional_state: EmotionalState

    # Standard detail
    primary_emotion: str = ""
    secondary_emotion: str | None = None
    intensity: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    certainty: float = 0.5

    # Body state
    energy: float = 0.5
    tension: float = 0.3
    openness: float = 0.5
    warmth: float = 0.5

    # Mood
    mood_label: str = "neutral"
    mood_trend: str = "stable"

    # Emotional stack (top activations)
    top_emotions: dict[str, float] = Field(default_factory=dict)

    # External signals contribution
    external_contributions: list[ExternalSignalContribution] = Field(default_factory=list)

    # Optional fields (controlled by config)
    voice_params: dict[str, object] | None = None
    pipeline_trace: dict[str, object] | None = None
    behavior_prompt: str | None = None
    coupling_contributions: dict[str, float] | None = None

    # Meta
    processing_time_ms: float = 0.0


class EmotionBatchResponse(BaseModel):
    """Response from batch processing."""

    session_id: str
    results: list[EmotionProcessResponse]
    total_processing_time_ms: float = 0.0


class EmotionStateResponse(BaseModel):
    """Current emotional state of a session."""

    session_id: str
    turn_number: int
    emotional_state: EmotionalState
    personality_summary: dict[str, float] = Field(default_factory=dict)
    active_systems: list[str] = Field(default_factory=list)


class EmotionConfigureResponse(BaseModel):
    """Response from configuring a session."""

    session_id: str
    personality_applied: dict[str, float]
    values_applied: dict[str, float] | None = None
    state_reset: bool = False


class PersonalityPreset(BaseModel):
    """A named personality preset."""

    name: str
    description: str
    traits: dict[str, float]


class EmotionPresetsResponse(BaseModel):
    """Available personality presets."""

    presets: list[PersonalityPreset]


class EmotionHealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "3.0.0"
    systems_count: int = 23
    active_sessions: int = 0
    coupling_enabled: bool = True
    external_signals_supported: list[str] = Field(
        default_factory=lambda: [
            "facial_au", "keyboard_dynamics",
            "time_of_day", "weather",
        ]
    )
