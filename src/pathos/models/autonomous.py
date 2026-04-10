"""Data models for Autonomous Research mode.

Defines all Pydantic models for research state, events, findings,
conclusions, and API request/response schemas.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from pathos.models.emotion import EmotionalState


class ResearchPipelineMode(str, Enum):
    """Pipeline mode for autonomous research."""

    NORMAL = "normal"    # Full regulation
    LITE = "lite"        # Fast keyword appraisal
    RAW = "raw"          # Unfiltered expression
    EXTREME = "extreme"  # Amplified + no regulation


class ResearchEventType(str, Enum):
    """Types of SSE events emitted during research."""

    TOPIC_PICKED = "topic_picked"
    SEARCH_STARTED = "search_started"
    SEARCH_RESULTS = "search_results"
    FINDING_PROCESSED = "finding_processed"
    EMOTIONAL_REFLECTION = "emotional_reflection"
    DEEP_THINKING = "deep_thinking"
    SUBTOPIC_PICKED = "subtopic_picked"
    CONCLUSION_FORMED = "conclusion_formed"
    TOPIC_COMPLETED = "topic_completed"
    STATE_UPDATE = "state_update"
    ERROR = "error"
    STOPPED = "stopped"


# --- Core data structures ---


class EmotionalReflection(BaseModel):
    """Agent's emotional self-inquiry about a finding."""

    how_it_feels: str = ""
    emotions_generated: str = ""
    emotional_insight: str = ""
    valence_before: float = 0.0
    valence_after: float = 0.0
    arousal_before: float = 0.0
    arousal_after: float = 0.0
    primary_emotion_before: str = "neutral"
    primary_emotion_after: str = "neutral"


class ProcessedFinding(BaseModel):
    """A single piece of information that went through the emotional pipeline."""

    source_url: str = ""
    source_title: str = ""
    content_snippet: str = ""
    emotional_reflection: EmotionalReflection = Field(default_factory=EmotionalReflection)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchConclusion(BaseModel):
    """Agent's emotionally-colored conclusion about a topic."""

    topic: str = ""
    conclusion_text: str = ""
    emotional_bias: str = ""
    confidence: float = 0.5
    primary_emotion: str = "neutral"
    intensity: float = 0.0
    pipeline_mode: str = "normal"


class DeepThinking(BaseModel):
    """Agent's autonomous questions, ideas, and subtopic from deep thinking."""

    questions: list[str] = Field(default_factory=list)
    ideas: list[str] = Field(default_factory=list)
    subtopic: str = ""
    primary_emotion: str = "neutral"
    intensity: float = 0.0


class ResearchTopic(BaseModel):
    """A topic the agent investigated."""

    query: str = ""
    subtopics: list[str] = Field(default_factory=list)
    findings: list[ProcessedFinding] = Field(default_factory=list)
    thinking: list[DeepThinking] = Field(default_factory=list)
    conclusions: list[ResearchConclusion] = Field(default_factory=list)
    started_at: str = ""
    completed_at: str | None = None


class AutonomousResearchState(BaseModel):
    """Full state of an autonomous research session."""

    session_id: str = ""
    pipeline_mode: ResearchPipelineMode = ResearchPipelineMode.NORMAL
    is_running: bool = False
    topics_researched: list[ResearchTopic] = Field(default_factory=list)
    current_topic: str | None = None
    total_findings: int = 0
    total_conclusions: int = 0
    started_at: str | None = None
    stopped_at: str | None = None
    chat_history: list[dict[str, str]] = Field(default_factory=list)


# --- SSE Events ---


class ResearchEvent(BaseModel):
    """An event emitted during autonomous research (sent via SSE)."""

    type: ResearchEventType
    data: dict[str, object] = Field(default_factory=dict)
    emotional_state: EmotionalState | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# --- API Request/Response ---


class StartResearchRequest(BaseModel):
    """Request to start autonomous research."""

    session_id: str = Field(default="")
    pipeline_mode: ResearchPipelineMode = ResearchPipelineMode.NORMAL
    seed_topics: list[str] = Field(default_factory=list)


class ResearchChatRequest(BaseModel):
    """Request to chat with the agent about its research."""

    message: str = Field(..., max_length=10000)
    session_id: str = ""


class ResearchSaveInfo(BaseModel):
    """Metadata for a saved research session."""

    filename: str
    session_id: str
    pipeline_mode: str
    topics_count: int
    findings_count: int
    conclusions_count: int
    saved_at: str
