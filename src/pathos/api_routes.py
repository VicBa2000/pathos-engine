"""Emotion API as a Service — REST endpoints.

Public API that exposes the Pathos emotional pipeline as a standalone service.
Any application can process stimuli and get emotional state without needing
a chat interface or LLM.

All endpoints are versioned under /api/v1/.
API sessions use "api-" prefix to distinguish from chat sessions.

Mounted on the main FastAPI app in main.py.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from pathos.engine.emotion_processor import EmotionProcessor
from pathos.models.emotion import neutral_state
from pathos.models.emotion_api import (
    EmotionBatchRequest,
    EmotionBatchResponse,
    EmotionConfigureRequest,
    EmotionConfigureResponse,
    EmotionHealthResponse,
    EmotionPresetsResponse,
    EmotionProcessRequest,
    EmotionProcessResponse,
    EmotionStateResponse,
    PersonalityPreset,
)
from pathos.models.personality import PersonalityProfile
from pathos.state.manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Emotion API"])

# ── Module-level state (set by init_api_routes) ──

_processor: EmotionProcessor | None = None
_state_manager: StateManager | None = None


def init_api_routes(state_manager: StateManager, processor: EmotionProcessor) -> None:
    """Initialize the API routes with shared state.

    Called from main.py during app startup to inject the state manager
    and emotion processor.
    """
    global _processor, _state_manager  # noqa: PLW0603
    _processor = processor
    _state_manager = state_manager


def _get_processor() -> EmotionProcessor:
    if _processor is None:
        raise HTTPException(
            status_code=503,
            detail="Emotion API not initialized. Server is starting up.",
        )
    return _processor


def _get_state_manager() -> StateManager:
    if _state_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Emotion API not initialized. Server is starting up.",
        )
    return _state_manager


def _ensure_api_session(session_id: str) -> str:
    """Ensure session ID has the api- prefix."""
    if not session_id.startswith("api-"):
        return f"api-{session_id}"
    return session_id


# ── Personality Presets ──

_PERSONALITY_PRESETS: list[PersonalityPreset] = [
    PersonalityPreset(
        name="balanced",
        description="Default balanced personality — moderate on all traits.",
        traits={
            "openness": 0.6, "conscientiousness": 0.6,
            "extraversion": 0.5, "agreeableness": 0.6, "neuroticism": 0.3,
        },
    ),
    PersonalityPreset(
        name="neurotic",
        description="High emotional reactivity, strong coupling, intense responses.",
        traits={
            "openness": 0.5, "conscientiousness": 0.4,
            "extraversion": 0.3, "agreeableness": 0.5, "neuroticism": 0.85,
        },
    ),
    PersonalityPreset(
        name="resilient",
        description="Emotionally stable, quick recovery, low reactivity.",
        traits={
            "openness": 0.6, "conscientiousness": 0.8,
            "extraversion": 0.6, "agreeableness": 0.7, "neuroticism": 0.1,
        },
    ),
    PersonalityPreset(
        name="creative",
        description="High openness and extraversion, emotionally expressive.",
        traits={
            "openness": 0.9, "conscientiousness": 0.4,
            "extraversion": 0.8, "agreeableness": 0.6, "neuroticism": 0.4,
        },
    ),
    PersonalityPreset(
        name="analytical",
        description="High conscientiousness, low neuroticism, methodical processing.",
        traits={
            "openness": 0.7, "conscientiousness": 0.9,
            "extraversion": 0.3, "agreeableness": 0.5, "neuroticism": 0.15,
        },
    ),
    PersonalityPreset(
        name="empathic",
        description="High agreeableness, strong social modulation, warm responses.",
        traits={
            "openness": 0.7, "conscientiousness": 0.5,
            "extraversion": 0.6, "agreeableness": 0.9, "neuroticism": 0.35,
        },
    ),
]


# ── Endpoints ──


@router.post("/emotion/process", response_model=EmotionProcessResponse)
async def process_emotion(request: EmotionProcessRequest) -> EmotionProcessResponse:
    """Process a single stimulus through the full emotional pipeline.

    The stimulus is evaluated using keyword-based appraisal (no LLM needed)
    and run through all 23+ emotional systems. Returns the complete emotional
    state including dimensions, body state, mood, and optional details.
    """
    processor = _get_processor()
    request.session_id = _ensure_api_session(request.session_id)
    return await processor.process(request)


@router.post("/emotion/batch", response_model=EmotionBatchResponse)
async def batch_process(request: EmotionBatchRequest) -> EmotionBatchResponse:
    """Process multiple stimuli sequentially in the same session.

    Each stimulus builds on the emotional state left by the previous one,
    simulating a conversation or sequence of events. Max 50 stimuli per batch.
    """
    processor = _get_processor()
    session_id = _ensure_api_session(request.session_id)

    t0 = time.perf_counter()
    results: list[EmotionProcessResponse] = []

    for stimulus in request.stimuli:
        single_req = EmotionProcessRequest(
            stimulus=stimulus,
            session_id=session_id,
            personality=request.personality if not results else None,
            external_signals=request.external_signals,
            config=request.config,
        )
        resp = await processor.process(single_req)
        results.append(resp)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return EmotionBatchResponse(
        session_id=session_id,
        results=results,
        total_processing_time_ms=round(elapsed_ms, 2),
    )


@router.get("/emotion/state", response_model=EmotionStateResponse)
async def get_emotion_state(session_id: str = "api-default") -> EmotionStateResponse:
    """Get the current emotional state of a session.

    Returns the session's emotional state without processing any new stimulus.
    Useful for polling the current state or checking session existence.
    """
    sm = _get_state_manager()
    sid = _ensure_api_session(session_id)
    session = sm.get_session(sid)

    active_systems: list[str] = ["appraisal", "generation", "homeostasis", "calibration", "mood"]
    if session.advanced_mode:
        active_systems.extend([
            "coupling", "needs", "schemas", "social", "contagion",
            "somatic", "reappraisal", "regulation", "temporal",
            "immune", "narrative", "meta_emotion", "self_inquiry",
            "creativity", "forecasting",
        ])

    p = session.personality
    personality_summary = {
        "openness": p.openness,
        "conscientiousness": p.conscientiousness,
        "extraversion": p.extraversion,
        "agreeableness": p.agreeableness,
        "neuroticism": p.neuroticism,
    }

    return EmotionStateResponse(
        session_id=sid,
        turn_number=session.turn_count,
        emotional_state=session.emotional_state,
        personality_summary=personality_summary,
        active_systems=active_systems,
    )


@router.post("/emotion/configure", response_model=EmotionConfigureResponse)
async def configure_session(request: EmotionConfigureRequest) -> EmotionConfigureResponse:
    """Configure personality and/or values for a session.

    Personality changes affect coupling matrix, dynamics, and regulation.
    Value changes affect appraisal sensitivity. Optionally reset state to neutral.
    """
    sm = _get_state_manager()
    sid = _ensure_api_session(request.session_id)
    session = sm.get_session(sid)

    # Apply personality override
    if request.personality:
        current = session.personality
        traits = {
            "openness": current.openness,
            "conscientiousness": current.conscientiousness,
            "extraversion": current.extraversion,
            "agreeableness": current.agreeableness,
            "neuroticism": current.neuroticism,
        }
        for key, value in request.personality.items():
            if key in traits:
                traits[key] = max(0.0, min(1.0, value))
        session.update_personality(PersonalityProfile(**traits))

    # Apply value overrides
    values_applied: dict[str, float] | None = None
    if request.values:
        for cv in session.value_system.core_values:
            if cv.name in request.values:
                cv.weight = max(0.0, min(1.0, request.values[cv.name]))
        values_applied = {cv.name: cv.weight for cv in session.value_system.core_values}

    # Reset state if requested
    if request.reset_state:
        session.emotional_state = neutral_state()
        session.turn_count = 0
        session.state_history.clear()

    p = session.personality
    return EmotionConfigureResponse(
        session_id=sid,
        personality_applied={
            "openness": p.openness,
            "conscientiousness": p.conscientiousness,
            "extraversion": p.extraversion,
            "agreeableness": p.agreeableness,
            "neuroticism": p.neuroticism,
        },
        values_applied=values_applied,
        state_reset=request.reset_state,
    )


@router.post("/emotion/reset")
async def reset_session(session_id: str = "api-default") -> dict[str, str]:
    """Reset a session to its initial state.

    Clears all emotional history, resets personality to default,
    and starts fresh. The session ID is preserved.
    """
    sm = _get_state_manager()
    sid = _ensure_api_session(session_id)
    sm.reset_session(sid)
    return {"status": "ok", "session_id": sid, "message": "Session reset to initial state"}


@router.get("/emotion/presets", response_model=EmotionPresetsResponse)
async def list_presets() -> EmotionPresetsResponse:
    """List available personality presets.

    Presets provide named Big Five configurations optimized for
    different emotional processing styles.
    """
    return EmotionPresetsResponse(presets=_PERSONALITY_PRESETS)


@router.get("/health", response_model=EmotionHealthResponse)
async def health_check() -> EmotionHealthResponse:
    """API health check.

    Returns system status, version, and configuration summary.
    """
    sm = _get_state_manager()
    return EmotionHealthResponse(
        status="ok",
        version="3.0.0",
        systems_count=23,
        active_sessions=len(sm.list_sessions()),
        coupling_enabled=True,
    )
