"""Pathos Engine - FastAPI application."""

import asyncio
import copy
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from pathos.config import LLMProviderType, Settings
from pathos.engine.appraiser import appraise, appraise_lite
from pathos.engine.behavior import EMOTION_EFFECTS, generate_behavior_modifier, generate_raw_behavior_modifier, generate_simple_behavior_modifier
from pathos.engine.calibration import apply_calibration, compute_calibration_profile
from pathos.engine.contagion import (
    compute_contagion_perturbation,
    detect_user_emotion,
    update_shadow_state,
)
from pathos.engine.forecasting import (
    estimate_user_emotion,
    evaluate_forecast,
    forecast_impact,
    get_forecast_prompt,
    record_forecast,
)
from pathos.engine.predictive import (
    PredictiveEngine,
    compute_prediction_error,
    decay_precision,
    get_prediction_prompt,
    prediction_error_to_emotion_modulation,
    record_prediction,
    update_precision,
)
from pathos.models.predictive import DemandType
from pathos.engine.workspace import (
    generate_candidate,
    get_workspace_prompt,
    process_workspace_turn,
)
from pathos.models.workspace import WorkspaceCandidate
from pathos.engine.autobio_memory import (
    get_autobiographical_details,
    get_autobiographical_prompt,
    process_autobiographical_turn,
)
from pathos.engine.dreaming import consolidate as dream_consolidate
from pathos.engine.drives import (
    attempt_goal_generation,
    get_drives_details,
    get_drives_prompt,
    process_goals,
    update_drives,
)
from pathos.engine.discovery import (
    get_discovery_details,
    get_discovery_prompt,
    get_vocabulary,
    process_discovery_turn,
)
from pathos.engine.phenomenology import (
    get_phenomenology_details,
    get_phenomenology_prompt,
    get_qualia_history_summary,
    process_phenomenology_turn,
)
from pathos.engine.development import (
    apply_stage_modifiers,
    attempt_transition,
    approve_pending_transition,
    filter_emotions_by_stage,
    get_development_details,
    get_development_prompt,
    is_emotion_available,
    is_system_available,
    track_experience,
)
from pathos.models.development import (
    DevelopmentConfig,
    DevelopmentSpeed,
    DevelopmentStage,
    DevelopmentState,
    TransitionMode,
)
from pathos.engine.somatic import (
    compute_somatic_bias,
    evaluate_user_reaction,
    register_pending_decision,
)
from pathos.engine.generator import (
    compute_arousal,
    compute_certainty,
    compute_dominance,
    compute_intensity,
    compute_valence,
    detect_emergent_emotions,
    generate_emotion,
)
from pathos.engine.homeostasis import regulate
from pathos.engine.meta import MetaEmotion, generate_meta_emotion
from pathos.engine.self_inquiry import SelfInquiry, check_self_inquiry
from pathos.engine.regulation import RegulationResult
from pathos.engine.metrics import coherence, continuity, proportionality, recovery
from pathos.engine.mood import compute_mood_congruence_bias
from pathos.engine.needs import compute_needs_amplification, update_needs
from pathos.engine.creativity import compute_creativity
from pathos.models.creativity import CreativityState
from pathos.engine.immune import apply_immune_protection, get_immune_prompt_info, update_immune_state
from pathos.engine.narrative import (
    apply_narrative_effects,
    check_coherence,
    decay_crisis_counter,
    detect_crisis,
    get_narrative_prompt,
    process_growth,
    update_narrative,
)
from pathos.engine.interoception import compute_interoceptive_feedback, update_interoceptive_state
from pathos.engine.reappraisal import reappraise
from pathos.engine.self_appraisal import compute_guilt_state_adjustment, evaluate_own_response
from pathos.engine.world_model import compute_world_model_adjustment, simulate_response_impact
from pathos.engine.emotional_sampler import (
    SamplingParams,
    TokenBiasResult,
    compute_sampling_params,
    compute_token_bias,
)
from pathos.engine.steering import EmotionalSteeringEngine, SteeringHook, SteeringMomentum
from pathos.engine.emotional_prefix import EmotionalPrefixHook, PrefixResult
from pathos.engine.emotional_attention import (
    AttentionBiasResult,
    AttentionHook,
    build_token_set,
    compute_attention_bias,
)
from pathos.models.self_appraisal import SelfAppraisalResult
from pathos.models.world_model import WorldModelResult
from pathos.models.voice import AVAILABLE_VOICES, VOICE_CATALOG, VoiceMode
from pathos.voice.asr import get_asr_service
from pathos.voice.params import detect_language, generate_voice_params, prepare_text_for_tts
from pathos.voice.tts import get_tts_service
from pathos.engine.social import compute_social_modulation, update_user_model
from pathos.models.emotion import EmotionalState, PrimaryEmotion, neutral_state
from pathos.models.personality import PersonalityProfile
from pathos.llm.base import LLMProvider
from pathos.llm.claude import ClaudeProvider
from pathos.llm.downloads import DownloadManager
from pathos.llm.featured_models import get_featured_models
from pathos.llm.ollama import OllamaProvider
from pathos.llm.openai_compat import CLOUD_PRESETS, OpenAICompatProvider
from pathos.models.calibration import (
    CalibrationProfile,
    CalibrationResult,
    CalibrationScenario,
)
from pathos.models.schemas import (
    AppraisalDetails,
    AttentionDetails,
    AuthenticityMetrics,
    ChatRequest,
    ChatResponse,
    PipelineStep,
    PipelineTrace,
    ContagionDetails,
    CouplingDetails,
    CreativityDetails,
    EmotionGenerationDetails,
    ForecastingDetails,
    PredictiveDetails,
    WorkspaceDetails,
    AutobiographicalDetails,
    DevelopmentDetails,
    DrivesDetails,
    DiscoveryDetails,
    PhenomenologyDetails,
    ImmuneDetails,
    VoiceDetails,
    NarrativeDetails,
    SomaticDetails,
    HomeostasisDetails,
    MemoryAmplificationDetails,
    MetaEmotionDetails,
    MoodCongruenceDetails,
    NeedsDetails,
    PersonalityDetails,
    ReappraisalDetails,
    RegulationDetails,
    ResearchChatResponse,
    ResearchStateResponse,
    SchemaDetails,
    SelfAppraisalDetails,
    WorldModelDetails,
    SteeringDetails,
    EmotionalPrefixDetails,
    ArenaEntry,
    ArenaDivergence,
    ArenaRequest,
    ArenaResponse,
    ChallengeConfig,
    ChallengeTarget,
    ChallengeState,
    ChallengeStartRequest,
    ChallengeChatRequest,
    ChallengeChatResponse,
    SandboxResponse,
    SandboxResult,
    BatchSandboxResponse,
    BatchScenarioRequest,
    ScenarioRequest,
    SocialDetails,
    StateResponse,
    TemporalDetails,
)
from pathos.api_routes import init_api_routes, router as api_router
from pathos.engine.emotion_processor import EmotionProcessor
from pathos.state.manager import SessionState, StateManager

settings = Settings()
state_manager = StateManager()
llm_provider: LLMProvider | None = None
steering_engine: EmotionalSteeringEngine = EmotionalSteeringEngine()
predictive_engine: PredictiveEngine = PredictiveEngine()
_switch_model_lock = asyncio.Lock()
download_manager = DownloadManager(settings.ollama_base_url)


def create_llm_provider(s: Settings) -> LLMProvider:
    if s.llm_provider == LLMProviderType.CLAUDE:
        if not s.anthropic_api_key:
            raise ValueError("PATHOS_ANTHROPIC_API_KEY requerida para Claude provider")
        return ClaudeProvider(
            api_key=s.anthropic_api_key,
            model=s.claude_model,
            ollama_base_url=s.ollama_base_url,
            embed_model=s.ollama_embed_model,
        )
    if s.llm_provider == LLMProviderType.TRANSFORMERS:
        from pathos.llm.transformers_provider import TransformersProvider
        return TransformersProvider(
            model_id=s.transformers_model,
            device_map=s.transformers_device_map,
            embed_model_url=s.ollama_base_url,
            embed_model=s.ollama_embed_model,
            adapter_path=s.transformers_adapter_path or None,
        )
    return OllamaProvider(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        embed_model=s.ollama_embed_model,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global llm_provider
    llm_provider = create_llm_provider(settings)

    # Pre-cargar modelos en Ollama para evitar cold-start en el primer request.
    # keep_alive="30m" mantiene ambos modelos en VRAM simultaneamente
    # (qwen3:4b ~2.5GB + nomic-embed-text ~0.3GB = ~2.8GB, cabe en 6GB GPU).
    if isinstance(llm_provider, OllamaProvider):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Warm up: modelo de chat
                await client.post(
                    f"{llm_provider.base_url}/api/chat",
                    json={"model": llm_provider.model, "messages": [], "keep_alive": "30m"},
                )
                # Warm up: modelo de embeddings
                await client.post(
                    f"{llm_provider.base_url}/api/embed",
                    json={"model": llm_provider.embed_model, "input": "", "keep_alive": "30m"},
                )
        except Exception:
            pass  # Ollama might not be running yet — non-fatal

    # Auto-load cached steering vectors for the current model
    _model_name = getattr(llm_provider, "model", settings.ollama_model)
    if steering_engine.load_vectors(_model_name):
        logger.info("Steering vectors loaded for '%s'", _model_name)
    else:
        logger.info("No cached steering vectors for '%s'. Run: python -m pathos.engine.steering_extract --model %s", _model_name, _model_name)

    # Auto-load latest save if available
    try:
        available_models: list[str] = []
        if isinstance(llm_provider, OllamaProvider):
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{llm_provider.base_url}/api/tags")
                available_models = [m["name"] for m in resp.json().get("models", [])]

        sid, msg = state_manager.auto_load_latest(available_models or None)
        if sid:
            logger.info("Auto-loaded: %s", msg)
            # Auto-reconnect to cloud provider if the save used one
            session = state_manager.get_session(sid)
            save_data = {}
            # Find the save file to get the model name
            for sf in sorted((Path(__file__).parent.parent.parent / "saves").glob("*.json"), reverse=True):
                try:
                    import json as _json
                    save_data = _json.loads(sf.read_text(encoding="utf-8"))
                    if save_data.get("_session_id") == sid:
                        break
                except Exception:
                    continue
            saved_model = save_data.get("_model", "")
            if saved_model and session.cloud_providers:
                # Find which cloud provider had this model
                for provider_id, cfg in session.cloud_providers.items():
                    if cfg.get("model") == saved_model and cfg.get("api_key"):
                        try:
                            preset = cfg.get("preset", provider_id)
                            if preset == "anthropic":
                                llm_provider = ClaudeProvider(
                                    api_key=cfg["api_key"],
                                    model=saved_model,
                                    ollama_base_url=settings.ollama_base_url,
                                    embed_model=settings.ollama_embed_model,
                                )
                            elif preset == "ollama_cloud":
                                llm_provider = OllamaProvider(
                                    base_url=cfg["base_url"],
                                    model=saved_model,
                                    embed_model=settings.ollama_embed_model,
                                    api_key=cfg["api_key"],
                                )
                            else:
                                llm_provider = OpenAICompatProvider(
                                    api_key=cfg["api_key"],
                                    base_url=cfg["base_url"],
                                    model=saved_model,
                                    ollama_base_url=settings.ollama_base_url,
                                    embed_model=settings.ollama_embed_model,
                                    provider_name=provider_id,
                                )
                            logger.info("Auto-reconnected to cloud provider '%s' (model=%s)", provider_id, saved_model)
                        except Exception as e:
                            logger.warning("Failed to auto-reconnect to '%s': %s", provider_id, e)
                        break
        else:
            logger.info("Fresh start: %s", msg)
    except Exception as e:
        logger.warning("Auto-load failed: %s", e)

    # Initialize Emotion API (standalone processor, works without LLM)
    api_processor = EmotionProcessor(state_manager, llm_provider)
    init_api_routes(state_manager, api_processor)

    yield
    if hasattr(llm_provider, "close"):
        await llm_provider.close()


app = FastAPI(
    title="Pathos Engine",
    description="Functional emotional architecture for LLMs",
    version="0.2.0",
    lifespan=lifespan,
)

_allowed_origins = os.environ.get("PATHOS_ALLOWED_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _allowed_origins.split(",") if o.strip()]
    if _allowed_origins
    else [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Emotion API routes (/api/v1/...)
app.include_router(api_router)


def _strip_meta_annotations(text: str) -> str:
    """Elimina meta-anotaciones de estado emocional de la respuesta del LLM.

    El LLM a veces incluye notas como:
    - *(State update: anger intensity reduced to 0.65...)*
    - (Internal note: processing grief...)
    - *[Emotional shift: moving toward acceptance]*
    """
    # Parenthesized state/internal annotations (italic or not)
    text = re.sub(r'\*?\((?:State|Internal|Emotional|Note|Meta)[^)]{5,}\)\*?', '', text)
    # Bracketed annotations
    text = re.sub(r'\*?\[(?:State|Internal|Emotional|Note|Meta)[^\]]{5,}\]\*?', '', text)
    # Clean up double newlines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _get_session_signals(session: "SessionState") -> tuple[float, float, float, str | None]:
    """Build external signal modulation from session's active signals config.

    Returns (valence_mod, arousal_mod, dominance_mod, perception_text).
    perception_text is a human-readable description of what external sensors
    detected about the user, so the agent can be aware and respond naturally.
    """
    from pathos.engine.external_signals import fuse_signals
    from pathos.models.emotion_api import ExternalSignal

    cfg = session.signals_config
    active = cfg.active_sources
    if not active:
        return 0.0, 0.0, 0.0, None

    signals = []
    for source_name, src_cfg in active.items():
        signals.append(ExternalSignal(
            source=source_name,
            valence_hint=src_cfg.valence_hint if src_cfg.valence_hint != 0.0 else None,
            arousal_hint=src_cfg.arousal_hint if src_cfg.arousal_hint != 0.5 else None,
            dominance_hint=src_cfg.dominance_hint,
            confidence=src_cfg.confidence,
        ))

    fused = fuse_signals(signals)

    # Build perception text for the agent
    perception = _build_perception_text(active)

    return fused.valence_modulation, fused.arousal_modulation, fused.dominance_modulation, perception


def _build_perception_text(
    active_sources: dict[str, "SignalSourceConfig"],
) -> str | None:
    """Build a natural-language description of what external sensors detect.

    This text is injected into the agent's system prompt so it can naturally
    reference what it perceives about the user (e.g. facial expressions).
    Only includes sources with meaningful signal (non-neutral values).
    """
    from pathos.models.external_signals import SignalSourceConfig  # noqa: F811

    parts: list[str] = []

    # Facial AU — the most important for social awareness
    facial = active_sources.get("facial_au")
    if facial and facial.enabled:
        facial_text = _describe_facial(facial)
        if facial_text:
            parts.append(facial_text)

    # Keyboard dynamics
    kb = active_sources.get("keyboard_dynamics")
    if kb and kb.enabled:
        kb_text = _describe_keyboard(kb)
        if kb_text:
            parts.append(kb_text)

    # Time of day
    tod = active_sources.get("time_of_day")
    if tod and tod.enabled:
        tod_text = _describe_time_of_day(tod)
        if tod_text:
            parts.append(tod_text)

    # Weather
    weather = active_sources.get("weather")
    if weather and weather.enabled:
        weather_text = _describe_weather(weather)
        if weather_text:
            parts.append(weather_text)

    if not parts:
        return None

    return "\n".join(parts)


def _describe_facial(cfg: "SignalSourceConfig") -> str | None:
    """Describe detected facial expression for the agent prompt."""
    v = cfg.valence_hint
    a = cfg.arousal_hint
    conf = cfg.confidence

    # Only report when confidence is meaningful (detection actually running)
    if conf < 0.2:
        return None

    # Determine dominant expression from valence/arousal quadrant
    if v > 0.15:
        if a > 0.5:
            expression = "alegria o entusiasmo"
        else:
            expression = "tranquilidad o satisfaccion"
    elif v < -0.15:
        if a > 0.5:
            expression = "tension, enojo o miedo"
        else:
            expression = "tristeza o desanimo"
    else:
        if a > 0.6:
            expression = "sorpresa o alerta"
        else:
            return None  # Neutral face — nothing notable to report

    confidence_word = "claramente" if conf > 0.5 else "sutilmente"

    return (
        f"[PERCEPCION VISUAL — webcam activa con consentimiento del usuario]\n"
        f"  Detectas {confidence_word} en el rostro del usuario senales de {expression}.\n"
        f"  Puedes mencionarlo con tacto si es relevante al contexto, o preguntar como se siente.\n"
        f"  NO digas 'detecto en tu rostro' — di algo natural como 'noto que...' o 'pareces...'.\n"
        f"  Respeta siempre la prioridad del bienestar del usuario."
    )


def _describe_keyboard(cfg: "SignalSourceConfig") -> str | None:
    """Describe detected keyboard dynamics for the agent prompt."""
    v = cfg.valence_hint
    a = cfg.arousal_hint
    conf = cfg.confidence

    if conf < 0.25:
        return None

    if v < -0.2 and a > 0.5:
        return (
            "  [Patron de escritura: el usuario escribe con signos de frustracion "
            "(muchas correcciones, ritmo irregular). No lo menciones directamente.]"
        )
    elif a > 0.65:
        return (
            "  [Patron de escritura: el usuario escribe rapido, parece energico o urgente. "
            "Puedes ajustar tu tono para ser mas directo si corresponde.]"
        )
    elif a < 0.3:
        return (
            "  [Patron de escritura: el usuario escribe lento, con pausas largas. "
            "Podria estar pensando o indeciso. Se paciente.]"
        )

    return None


def _describe_time_of_day(cfg: "SignalSourceConfig") -> str | None:
    """Describe circadian context for the agent prompt."""
    a = cfg.arousal_hint
    v = cfg.valence_hint
    conf = cfg.confidence

    if conf < 0.2:
        return None

    # Infer period from arousal/valence pattern (Thayer 1989 model)
    if a < 0.35:
        # Low arousal = late night or afternoon dip
        if v < -0.05:
            period_desc = "es de noche o madrugada — el usuario podria estar cansado"
        else:
            period_desc = "parece ser la hora del bajón de la tarde"
    elif a > 0.6:
        if v > 0.05:
            period_desc = "es de mañana — momento de alta energia natural"
        else:
            period_desc = "parece ser un momento de alta activacion"
    else:
        return None  # Midday / unremarkable — nothing to report

    return (
        f"  [Contexto circadiano: {period_desc}. "
        f"Ajusta tu energia comunicativa de forma sutil.]"
    )


def _describe_weather(cfg: "SignalSourceConfig") -> str | None:
    """Describe weather context for the agent prompt."""
    v = cfg.valence_hint
    a = cfg.arousal_hint
    conf = cfg.confidence

    if conf < 0.2:
        return None

    # Infer weather condition from valence/arousal (Schwarz 1983 model)
    if v > 0.2:
        weather_desc = "hace buen tiempo donde esta el usuario — dia soleado y agradable"
    elif v < -0.2:
        if a > 0.4:
            weather_desc = "hay mal tiempo — probablemente lluvia o tormenta"
        else:
            weather_desc = "el clima esta gris o desagradable donde esta el usuario"
    else:
        return None  # Neutral weather — nothing notable

    return (
        f"  [Contexto ambiental: {weather_desc}. "
        f"Puedes mencionarlo de forma casual si surge naturalmente en la conversacion.]"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Pipeline completo con todos los sistemas avanzados."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    pipeline_start = time.perf_counter()
    trace_steps: list[PipelineStep] = []

    def _snap(state: "EmotionalState") -> dict[str, float]:
        return {"valence": state.valence, "arousal": state.arousal, "dominance": state.dominance, "intensity": state.intensity}

    def _delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
        return {k: round(after.get(k, 0) - before.get(k, 0), 4) for k in before if abs(after.get(k, 0) - before.get(k, 0)) > 0.001}

    def _impact(delta: dict[str, float]) -> str:
        mag = sum(abs(v) for v in delta.values())
        if mag > 0.3: return "high"
        if mag > 0.1: return "medium"
        if mag > 0.01: return "low"
        return "none"

    session = state_manager.get_session(request.session_id)
    session.turn_count += 1
    previous_state = session.emotional_state.model_copy(deep=True)

    # 0. Homeostasis + regulation recovery
    t0 = time.perf_counter()
    snap_before = _snap(session.emotional_state)
    if session.turn_count > 1:
        session.emotional_state = regulate(session.emotional_state, turns_elapsed=1)
        if session.advanced_mode:
            session.regulator.recover(session.personality.regulation_capacity_base)
    snap_after = _snap(session.emotional_state)
    d = _delta(snap_before, snap_after)
    trace_steps.append(PipelineStep(
        name="homeostasis", label="Homeostasis",
        active=session.turn_count > 1,
        skipped_reason="" if session.turn_count > 1 else "First turn",
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary="Emotional state drifts back toward baseline" if session.turn_count > 1 else "No drift on first turn",
        impact=_impact(d), delta=d,
        details={"before": snap_before, "after": snap_after},
    ))

    # 0a. Interoception — body-state feedback ascendente [ADVANCED]
    t0 = time.perf_counter()
    intero_result = None
    if session.advanced_mode:
        session.interoceptive_state = update_interoceptive_state(
            session.interoceptive_state, session.emotional_state.body_state,
        )
        intero_result = compute_interoceptive_feedback(
            session.interoceptive_state, session.emotional_state.body_state,
        )
        if intero_result.active:
            session.emotional_state.valence = max(-1.0, min(1.0,
                session.emotional_state.valence + intero_result.valence_delta))
            session.emotional_state.arousal = max(0.0, min(1.0,
                session.emotional_state.arousal + intero_result.arousal_delta))
    trace_steps.append(PipelineStep(
        name="interoception", label="Interoception",
        active=session.advanced_mode,
        skipped_reason="" if session.advanced_mode else "Advanced mode off",
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Body→Emotion feedback: {intero_result.source} (dv={intero_result.valence_delta:+.3f}, da={intero_result.arousal_delta:+.3f})" if intero_result and intero_result.active else "No interoceptive signal",
        impact="medium" if intero_result and intero_result.active else "none",
        details={"source": intero_result.source, "valence_delta": intero_result.valence_delta, "arousal_delta": intero_result.arousal_delta, "tension_turns": session.interoceptive_state.high_tension_turns, "low_energy_turns": session.interoceptive_state.low_energy_turns, "warmth_turns": session.interoceptive_state.high_warmth_turns} if intero_result and intero_result.active else {},
    ))

    # 0b. Temporal pre-processing (anticipation, rumination) [ADVANCED]
    t0 = time.perf_counter()
    temporal_result = session.temporal.process_pre_turn(request.message) if session.advanced_mode else None
    has_temporal = temporal_result is not None and (
        getattr(temporal_result, 'rumination_active', False)
        or getattr(temporal_result, 'anticipation_active', False)
        or getattr(temporal_result, 'savoring_active', False)
    ) if temporal_result else False
    trace_steps.append(PipelineStep(
        name="temporal_pre", label="Temporal Pre-processing",
        active=session.advanced_mode,
        skipped_reason="" if session.advanced_mode else "Advanced mode off",
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary="Checking for rumination, anticipation, or savoring" if session.advanced_mode else "",
        impact="medium" if has_temporal else "none",
    ))

    # 0c. Predictive Processing — generate predictions BEFORE input [CORE]
    t0 = time.perf_counter()
    active_schemas_for_pred: list[tuple[str, str, float]] | None = None
    if session.advanced_mode and session.schemas.schemas:
        active_schemas_for_pred = [
            (s.trigger_category, s.typical_emotion.value, s.reinforcement_strength)
            for s in session.schemas.schemas
            if s.reinforcement_strength > 0.3
        ]
    current_predictions = predictive_engine.generate_predictions(
        predictive_state=session.predictive,
        conversation_history=session.conversation,
        user_model=session.user_model,
        mood=session.emotional_state.mood,
        emotional_state=session.emotional_state,
        active_schemas=active_schemas_for_pred,
    )
    session.predictive.current_predictions = current_predictions
    pred_dur = (time.perf_counter() - t0) * 1000
    trace_steps.append(PipelineStep(
        name="predictive_gen", label="Predictive Processing",
        active=True, duration_ms=pred_dur,
        summary=f"Predicted tone={current_predictions.content.expected_tone}, "
                f"valence={current_predictions.emotion.expected_valence:+.2f}, "
                f"demand={current_predictions.demand.expected_demand.value}"
                if not current_predictions.is_cold_start
                else "Cold start — no predictions yet (using 100% classical appraisal)",
        impact="medium" if session.predictive.is_warm else "low",
        details={
            "predicted_tone": current_predictions.content.expected_tone,
            "predicted_intent": current_predictions.content.expected_intent,
            "predicted_valence": round(current_predictions.emotion.expected_valence, 4),
            "predicted_arousal": round(current_predictions.emotion.expected_arousal, 4),
            "predicted_demand": current_predictions.demand.expected_demand.value,
            "avg_confidence": round(current_predictions.average_confidence, 4),
            "predictive_weight": round(session.predictive.predictive_weight, 4),
            "is_warm": session.predictive.is_warm,
        },
    ))

    # 1. Appraisal + Memory amplification
    t0 = time.perf_counter()
    if session.lite_mode:
        appraisal_result = appraise_lite(request.message)
        amplification = await session.memory.check_amplification(request.message, llm=None)
    else:
        appraisal_task = asyncio.create_task(appraise(
            stimulus=request.message,
            value_system=session.value_system,
            llm=llm_provider,
            think=True,
        ))
        memory_task = asyncio.create_task(
            session.memory.check_amplification(request.message, llm=llm_provider),
        )
        try:
            appraisal_result = await appraisal_task
        except Exception as e:
            memory_task.cancel()
            logger.exception("Appraisal failed")
            raise HTTPException(status_code=500, detail="Appraisal failed")
        amplification = await memory_task
    appraisal_dur = (time.perf_counter() - t0) * 1000
    av = appraisal_result.vector
    app_vals = {
        "relevance": round(av.relevance.novelty, 3),
        "valence": round(av.valence.goal_conduciveness, 3),
        "coping": round(av.coping.control, 3),
        "agency": round(av.agency.intentionality, 3) if av.agency else 0,
        "norms": round(av.norms.self_consistency, 3) if av.norms else 0,
    }
    hint_str = appraisal_result.emotion_hint.value if appraisal_result.emotion_hint else "none"
    appraisal_summary = f"Evaluated as {hint_str} (relevance: {app_vals['relevance']:.1%})"
    if amplification > 0:
        appraisal_summary += f", memory amplified {amplification:.0%}"
    trace_steps.append(PipelineStep(
        name="appraisal", label="Appraisal + Memory",
        active=True, duration_ms=appraisal_dur,
        summary=appraisal_summary,
        impact="high" if app_vals["relevance"] > 0.5 else "medium" if app_vals["relevance"] > 0.2 else "low",
        details={"mode": "lite" if session.lite_mode else "llm", "appraisal": app_vals, "amplification": round(amplification, 3), "emotion_hint": hint_str},
    ))

    # --- Advanced systems pre-emotion (steps 2b-2g) ---
    needs_amp = 0.0
    schema_hint: PrimaryEmotion | None = None
    schema_amp = 0.0
    social_v_mod = 0.0
    social_i_mod = 0.0
    contagion_v = 0.0
    contagion_a = 0.0
    somatic_bias = 0.0
    gut_feeling: str | None = None
    detected_v = 0.0
    detected_a = 0.0
    signal_str = 0.0

    # Helper: dev gate shortcut
    _dev = session.development
    # Raw/Extreme bypass development gating — all systems available
    # Use a copy with enabled=False for is_system_available checks
    _dev_gate = _dev
    if _dev.enabled and (session.raw_mode or session.extreme_mode):
        _dev_gate = _dev.model_copy(update={"enabled": False})
    _dev_skip = f"Development: {_dev.current_stage.value}" if _dev.enabled and not session.raw_mode and not session.extreme_mode else ""

    if session.advanced_mode:
        # 2b. Needs amplification [DEV: stage 2+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"needs"):
            needs_amp = compute_needs_amplification(session.needs, request.message)
            trace_steps.append(PipelineStep(
                name="needs", label="Psychological Needs",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Needs amplification: {needs_amp:+.2f}" if abs(needs_amp) > 0.01 else "No significant need activation",
                impact="medium" if abs(needs_amp) > 0.05 else "low" if abs(needs_amp) > 0.01 else "none",
                details={"amplification": round(needs_amp, 4), "connection": round(session.needs.connection, 3), "competence": round(session.needs.competence, 3), "autonomy": round(session.needs.autonomy, 3), "safety": round(session.needs.safety, 3)},
            ))
        else:
            trace_steps.append(PipelineStep(name="needs", label="Psychological Needs", active=False, skipped_reason=_dev_skip))

        # 2c. Schema priming [DEV: stage 2+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"schemas"):
            schema_hint, schema_amp = session.schemas.check_priming(request.message)
            trace_steps.append(PipelineStep(
                name="schemas", label="Emotional Schemas",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Schema primed: {schema_hint.value} ({schema_amp:+.2f})" if schema_hint else "No schema activated",
                impact="high" if schema_hint else "none",
                details={"primed_emotion": schema_hint.value if schema_hint else None, "amplification": round(schema_amp, 4), "schemas_count": len(session.schemas._schemas)},
            ))
        else:
            trace_steps.append(PipelineStep(name="schemas", label="Emotional Schemas", active=False, skipped_reason=_dev_skip))

        # 2d. Social modulation [DEV: stage 3+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"social"):
            social_v_mod, social_i_mod = compute_social_modulation(session.user_model, compute_valence(appraisal_result.vector))
            trace_steps.append(PipelineStep(
                name="social", label="Social Cognition",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Rapport {session.user_model.rapport:.0%}, trust {session.user_model.trust_level:.0%}",
                impact="medium" if abs(social_v_mod) > 0.05 else "low" if abs(social_v_mod) > 0.01 else "none",
                details={"valence_mod": round(social_v_mod, 4), "intensity_mod": round(social_i_mod, 4), "rapport": round(session.user_model.rapport, 3), "trust": round(session.user_model.trust_level, 3)},
            ))
        else:
            trace_steps.append(PipelineStep(name="social", label="Social Cognition", active=False, skipped_reason=_dev_skip))

        # 2e. Emotion Contagion (pre-cognitive) [DEV: stage 2+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"contagion"):
            detected_v, detected_a, signal_str = detect_user_emotion(request.message)
            session.shadow_state = update_shadow_state(session.shadow_state, detected_v, detected_a, signal_str)
            contagion_v, contagion_a = compute_contagion_perturbation(
                session.shadow_state, session.emotional_state.valence,
                session.emotional_state.arousal, session.personality, session.user_model.rapport,
            )
            contagion_mag = abs(contagion_v) + abs(contagion_a)
            trace_steps.append(PipelineStep(
                name="contagion", label="Emotion Contagion",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Detected user emotion (v:{detected_v:+.2f}, a:{detected_a:.2f}), contagion applied" if signal_str > 0.1 else "No strong emotional signal detected from user",
                impact="high" if contagion_mag > 0.1 else "medium" if contagion_mag > 0.03 else "low" if signal_str > 0.1 else "none",
                details={"detected_valence": round(detected_v, 4), "detected_arousal": round(detected_a, 4), "signal_strength": round(signal_str, 4), "contagion_v": round(contagion_v, 4), "contagion_a": round(contagion_a, 4)},
            ))
        else:
            trace_steps.append(PipelineStep(name="contagion", label="Emotion Contagion", active=False, skipped_reason=_dev_skip))

        # 2f. Somatic Markers (pre-rational gut feeling) [DEV: stage 4+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"somatic"):
            session.somatic_markers = evaluate_user_reaction(
                session.somatic_markers, detected_v, session.turn_count,
            )
            somatic_bias, gut_feeling = compute_somatic_bias(session.somatic_markers, request.message)
            trace_steps.append(PipelineStep(
                name="somatic", label="Somatic Markers",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Gut feeling: {gut_feeling} (bias: {somatic_bias:+.2f})" if gut_feeling else "No somatic marker triggered",
                impact="medium" if abs(somatic_bias) > 0.05 else "low" if gut_feeling else "none",
                details={"bias": round(somatic_bias, 4), "gut_feeling": gut_feeling, "markers_count": len(session.somatic_markers.markers)},
            ))
        else:
            trace_steps.append(PipelineStep(name="somatic", label="Somatic Markers", active=False, skipped_reason=_dev_skip))

        # 2g. Emotional Forecasting: evaluate previous forecast (if enabled) [DEV: stage 3+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"forecasting") and session.forecast.enabled and session.turn_count > 1:
            session.forecast = evaluate_forecast(
                session.forecast, detected_v, detected_a, session.turn_count,
            )
        _forecast_active = is_system_available(_dev_gate,"forecasting") and session.forecast.enabled and session.turn_count > 1
        trace_steps.append(PipelineStep(
            name="forecast_eval", label="Forecast Evaluation",
            active=_forecast_active,
            skipped_reason="" if is_system_available(_dev_gate,"forecasting") else _dev_skip,
            duration_ms=(time.perf_counter() - t0) * 1000,
            summary=f"Previous forecast accuracy: {session.forecast.accuracy_score:.0%}" if _forecast_active else "",
            impact="low" if _forecast_active else "none",
        ))
    else:
        # Add skipped steps for non-advanced mode
        for name, label in [("needs", "Psychological Needs"), ("schemas", "Emotional Schemas"),
                            ("social", "Social Cognition"), ("contagion", "Emotion Contagion"),
                            ("somatic", "Somatic Markers"), ("forecast_eval", "Forecast Evaluation")]:
            trace_steps.append(PipelineStep(name=name, label=label, active=False, skipped_reason="Advanced mode off"))

    # 2i. Predictive Processing — compute prediction error [CORE]
    t0 = time.perf_counter()
    prediction_error = None
    prediction_prompt_text: str | None = None
    if current_predictions is not None:
        # Detect intent from appraisal for error computation
        _detected_intent = "unknown"
        if appraisal_result.emotion_hint:
            hint_val = appraisal_result.emotion_hint.value
            if hint_val in ("anger", "frustration"):
                _detected_intent = "complaint"
            elif hint_val in ("gratitude", "joy"):
                _detected_intent = "greeting"
            elif hint_val in ("fear", "anxiety"):
                _detected_intent = "emotional_expression"
            elif hint_val == "surprise":
                _detected_intent = "topic_change"

        # Map appraisal hint to demand type
        _detected_demand: DemandType | None = None
        if "?" in request.message:
            _detected_demand = DemandType.HELP
        elif appraisal_result.emotion_hint and appraisal_result.emotion_hint.value in (
            "sadness", "fear", "anxiety", "helplessness",
        ):
            _detected_demand = DemandType.EMOTIONAL

        prediction_error = compute_prediction_error(
            predictions=current_predictions,
            actual_stimulus=request.message,
            detected_user_valence=detected_v if session.advanced_mode else 0.0,
            detected_user_arousal=detected_a if session.advanced_mode else 0.3,
            detected_intent=_detected_intent,
            detected_demand=_detected_demand,
        )

        # Update bayesian precision
        session.predictive = update_precision(session.predictive, prediction_error)

        # Record prediction + error in history
        session.predictive = record_prediction(
            session.predictive, current_predictions, prediction_error,
        )

        # Apply precision decay (1 turn)
        session.predictive = decay_precision(session.predictive, elapsed_turns=1)

        # Generate prompt for behavior modifier
        prediction_prompt_text = get_prediction_prompt(prediction_error, session.predictive)

    pred_error_dur = (time.perf_counter() - t0) * 1000
    trace_steps.append(PipelineStep(
        name="predictive_error", label="Prediction Error",
        active=prediction_error is not None,
        duration_ms=pred_error_dur,
        summary=(
            f"Surprise: {prediction_error.surprise_type.value} "
            f"(error={prediction_error.total_error:.2f}, "
            f"vulnerability={prediction_error.vulnerability:.2f})"
            if prediction_error and prediction_error.surprise_type.value != "none"
            else "Prediction correct — no surprise"
        ) if prediction_error else "No predictions to evaluate",
        impact=(
            "high" if prediction_error and prediction_error.vulnerability > 0.3
            else "medium" if prediction_error and prediction_error.total_error > 0.2
            else "low"
        ) if prediction_error else "none",
        details={
            "content_error": round(prediction_error.content_error, 4),
            "emotion_error": round(prediction_error.emotion_error, 4),
            "demand_error": round(prediction_error.demand_error, 4),
            "total_error": round(prediction_error.total_error, 4),
            "surprise_type": prediction_error.surprise_type.value,
            "valence_direction": round(prediction_error.valence_direction, 4),
            "vulnerability": round(prediction_error.vulnerability, 4),
            "content_precision": round(session.predictive.content_precision, 4),
            "emotion_precision": round(session.predictive.emotion_precision, 4),
            "demand_precision": round(session.predictive.demand_precision, 4),
            "predictive_weight": round(session.predictive.predictive_weight, 4),
        } if prediction_error else {},
    ))

    # 2k. Global Workspace — consciousness competition [TOGGLEABLE]
    t0 = time.perf_counter()
    workspace_prompt_text: str | None = None
    autobiographical_prompt_text: str | None = None
    if session.consciousness.enabled and session.advanced_mode and is_system_available(_dev_gate,"workspace"):
        # Generate candidates from pipeline data
        ws_candidates: list[WorkspaceCandidate] = []
        # Appraisal candidate
        if appraisal_result.emotion_hint:
            hint_str_ws = appraisal_result.emotion_hint.value
            ws_candidates.append(generate_candidate(
                source="appraisal",
                content=f"Stimulus evaluated as {hint_str_ws}",
                urgency=0.7 if abs(compute_valence(appraisal_result.vector)) > 0.3 else 0.4,
                relevance=appraisal_result.vector.relevance.personal_significance,
                emotional_intensity=abs(compute_valence(appraisal_result.vector)),
                emotion_tag=hint_str_ws,
                category="stimulus",
            ))
        # Schema candidate
        if schema_hint:
            ws_candidates.append(generate_candidate(
                source="schema",
                content=f"Pattern activated: {schema_hint.value} (amp={schema_amp:.2f})",
                urgency=0.5 + schema_amp * 0.3,
                relevance=0.7 if schema_amp > 0.1 else 0.3,
                emotional_intensity=schema_amp,
                emotion_tag=schema_hint.value,
                category="pattern",
            ))
        # Contagion candidate
        if signal_str > 0.15:
            ws_candidates.append(generate_candidate(
                source="contagion",
                content=f"Absorbing user emotion (v={detected_v:+.2f}, a={detected_a:.2f})",
                urgency=signal_str * 0.6,
                relevance=signal_str,
                emotional_intensity=abs(detected_v) * 0.8 + detected_a * 0.2,
                emotion_tag="anxiety" if detected_v < -0.2 else "joy" if detected_v > 0.2 else "neutral",
                category="social",
            ))
        # Somatic candidate
        if gut_feeling:
            ws_candidates.append(generate_candidate(
                source="somatic",
                content=f"Gut feeling: {gut_feeling} (bias={somatic_bias:+.2f})",
                urgency=abs(somatic_bias) * 0.8,
                relevance=0.5,
                emotional_intensity=abs(somatic_bias),
                category="body",
            ))
        # Predictive surprise candidate
        if prediction_error and prediction_error.surprise_type.value != "none":
            ws_candidates.append(generate_candidate(
                source="predictive",
                content=f"Surprise {prediction_error.surprise_type.value} (error={prediction_error.total_error:.2f})",
                urgency=prediction_error.vulnerability * 0.9,
                relevance=0.8,
                emotional_intensity=prediction_error.total_error,
                emotion_tag="surprise" if prediction_error.surprise_type.value == "neutral"
                    else "disappointment" if prediction_error.surprise_type.value == "negative"
                    else "relief",
                category="prediction",
            ))
        # Needs candidate (if high need)
        if needs_amp > 0.1:
            ws_candidates.append(generate_candidate(
                source="needs",
                content=f"Unmet need amplification: {needs_amp:+.2f}",
                urgency=needs_amp * 0.7,
                relevance=0.5,
                emotional_intensity=needs_amp,
                category="needs",
            ))

        session.consciousness = process_workspace_turn(session.consciousness, ws_candidates, raw_mode=session.raw_mode, extreme_mode=session.extreme_mode)
        if session.consciousness.current_result:
            workspace_prompt_text = get_workspace_prompt(session.consciousness.current_result)

    ws_active = session.consciousness.enabled and session.advanced_mode and is_system_available(_dev_gate,"workspace")
    ws_result = session.consciousness.current_result
    trace_steps.append(PipelineStep(
        name="workspace", label="Global Workspace",
        active=ws_active,
        skipped_reason="" if ws_active else ("Workspace disabled" if not session.consciousness.enabled else "Advanced mode off"),
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=(
            f"Conscious: {len(ws_result.conscious)}/{ws_result.total_candidates}, "
            f"integration={ws_result.integration_score:.2f}, "
            f"stability={ws_result.workspace_stability:.2f}"
            if ws_result else "Workspace inactive"
        ),
        impact="high" if ws_result and ws_result.integration_score < 0.3 else "medium" if ws_active else "none",
        details={
            "conscious_sources": [c.source for c in ws_result.conscious],
            "preconscious_count": len(ws_result.preconscious),
            "coalitions_formed": ws_result.coalitions_formed,
            "integration_score": ws_result.integration_score,
            "workspace_stability": ws_result.workspace_stability,
            "total_candidates": ws_result.total_candidates,
            "filtered_noise": ws_result.filtered_noise,
            "preconscious_mood_v": round(session.consciousness.preconscious.mood_valence_contribution, 4),
            "preconscious_tension": round(session.consciousness.preconscious.somatic_tension_echo, 4),
        } if ws_result else {},
    ))

    # 2m. Autobiographical Memory — prompt from existing memories [OPT-IN]
    autobiographical_prompt_text = get_autobiographical_prompt(session.autobiographical)

    # 2n. Development — prompt from current stage [TOGGLEABLE]
    development_prompt_text = get_development_prompt(session.development)

    # 2o. Drives — update drives + process goals + generate new goals [TOGGLEABLE]
    drives_updates: list = []
    drives_impacts: list = []
    if is_system_available(session.development, "drives"):
        session.drives, drives_updates = update_drives(
            session.drives, request.message, session.personality,
            session.turn_count, getattr(session.user_model, 'rapport', 0.5),
        )
        session.drives, drives_impacts = process_goals(
            session.drives, request.message, session.turn_count,
        )
        session.drives = attempt_goal_generation(
            session.drives, request.message, session.turn_count,
        )
    drives_prompt_text = get_drives_prompt(session.drives, raw_mode=session.raw_mode, extreme_mode=session.extreme_mode)

    # 2l. External signals (from session config — only active sources)
    sig_v, sig_a, sig_d, perception_text = _get_session_signals(session)

    # 3. Emotion Generation
    t0 = time.perf_counter()
    snap_before = _snap(session.emotional_state)
    effective_hint = schema_hint if schema_hint and not appraisal_result.emotion_hint else appraisal_result.emotion_hint
    pred_modulation = prediction_error_to_emotion_modulation(
        prediction_error, session.predictive.predictive_weight,
        raw_mode=session.raw_mode, extreme_mode=session.extreme_mode,
    )
    new_state = generate_emotion(
        appraisal=appraisal_result.vector,
        current_state=session.emotional_state,
        stimulus=request.message,
        amplification=amplification + schema_amp,
        emotion_hint=effective_hint,
        dynamics=session.dynamics if session.advanced_mode else None,
        needs_amplification=needs_amp,
        social_valence_mod=social_v_mod + somatic_bias + sig_v,
        social_intensity_mod=social_i_mod,
        contagion_valence=contagion_v,
        contagion_arousal=contagion_a + sig_a,
        coupling=session.coupling if session.advanced_mode else None,
        predictive_modulation=pred_modulation,
    )
    # 3a. Development emotion filtering [TOGGLEABLE]
    # Raw/Extreme bypass gating — all emotions available (adult expression)
    if _dev.enabled and new_state.emotional_stack and not session.raw_mode and not session.extreme_mode:
        new_state.emotional_stack = filter_emotions_by_stage(_dev, new_state.emotional_stack)
        # Clamp primary/secondary to available emotions
        if not is_emotion_available(_dev, new_state.primary_emotion.value):
            # Find highest-activation available emotion from the filtered stack
            _avail_emos = [(e, a) for e, a in new_state.emotional_stack.items() if is_emotion_available(_dev, e)]
            if _avail_emos:
                _best = max(_avail_emos, key=lambda x: x[1])
                try:
                    new_state.primary_emotion = PrimaryEmotion(_best[0])
                except ValueError:
                    pass
        if new_state.secondary_emotion and not is_emotion_available(_dev, new_state.secondary_emotion.value):
            new_state.secondary_emotion = None
        # Apply stage-specific body/intensity modifiers
        apply_stage_modifiers(_dev, new_state)

    snap_after = _snap(new_state)
    d = _delta(snap_before, snap_after)
    trace_steps.append(PipelineStep(
        name="emotion_gen", label="Emotion Generation",
        active=True, duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"{new_state.primary_emotion.value.capitalize()} ({new_state.intensity:.0%} intensity)",
        impact=_impact(d), delta=d,
        details={"primary": new_state.primary_emotion.value, "secondary": new_state.secondary_emotion.value if new_state.secondary_emotion else None, "intensity": round(new_state.intensity, 3), **snap_after},
    ))

    # 3b. Calibration
    t0 = time.perf_counter()
    snap_before = _snap(new_state)
    new_state = apply_calibration(new_state, session.calibration_profile)
    snap_after = _snap(new_state)
    d = _delta(snap_before, snap_after)
    trace_steps.append(PipelineStep(
        name="calibration", label="Calibration",
        active=True, duration_ms=(time.perf_counter() - t0) * 1000,
        summary="Applied user calibration offsets" if any(d.values()) else "No calibration adjustment needed",
        impact=_impact(d), delta=d,
    ))

    # --- Extreme mode amplification (after calibration, before regulation) ---
    if session.extreme_mode:
        new_state.intensity = min(1.0, new_state.intensity * 1.5)
        new_state.arousal = max(0.4, min(1.0, new_state.arousal * 1.3))
        new_state.body_state.tension = min(1.0, new_state.body_state.tension * 1.4)
        new_state.body_state.warmth = max(0.0, new_state.body_state.warmth * 0.6)
        new_state.body_state.openness = max(0.0, new_state.body_state.openness * 0.7)
        # Push valence further from zero (amplify whatever direction)
        if new_state.valence < 0:
            new_state.valence = max(-1.0, new_state.valence * 1.4)
        elif new_state.valence > 0:
            new_state.valence = min(1.0, new_state.valence * 1.3)

    # --- Advanced systems post-emotion (steps 4-8c) ---
    reappraisal_result = None
    regulation_result = RegulationResult()
    immune_info: str | None = None
    narrative_info: str | None = None
    meta_emotion: MetaEmotion | None = None
    self_inquiry: SelfInquiry | None = None
    emergent: list[str] = []
    creativity_state = CreativityState()
    forecast_info: str | None = None

    if session.advanced_mode:
        # 4. Reappraisal (multi-pass) — BYPASSED in extreme mode [DEV: stage 4+]
        t0 = time.perf_counter()
        if session.extreme_mode:
            trace_steps.append(PipelineStep(
                name="reappraisal", label="Cognitive Reappraisal",
                active=False, skipped_reason="Extreme mode: no cognitive softening",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        elif not is_system_available(_dev_gate,"reappraisal"):
            trace_steps.append(PipelineStep(name="reappraisal", label="Cognitive Reappraisal", active=False, skipped_reason=_dev_skip))
        else:
            snap_before = _snap(new_state)
            new_state, reappraisal_result = reappraise(new_state, session.regulator.regulation_capacity)
            snap_after = _snap(new_state)
            d = _delta(snap_before, snap_after)
            trace_steps.append(PipelineStep(
                name="reappraisal", label="Cognitive Reappraisal",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Reappraised via {reappraisal_result.strategy}" if reappraisal_result and reappraisal_result.applied else "No reappraisal needed",
                impact=_impact(d), delta=d,
                details={"applied": reappraisal_result.applied if reappraisal_result else False, "strategy": reappraisal_result.strategy if reappraisal_result and reappraisal_result.applied else None},
            ))

        # 5. Active regulation — BYPASSED in extreme mode [DEV: stage 3+]
        t0 = time.perf_counter()
        if session.extreme_mode:
            trace_steps.append(PipelineStep(
                name="regulation", label="Active Regulation",
                active=False, skipped_reason="Extreme mode: no emotional dampening",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        elif not is_system_available(_dev_gate,"regulation"):
            trace_steps.append(PipelineStep(name="regulation", label="Active Regulation", active=False, skipped_reason=_dev_skip))
        else:
            snap_before = _snap(new_state)
            new_state, regulation_result = session.regulator.regulate(
                new_state, session.personality.regulation_capacity_base,
                coping_control=av.coping.control,
                coping_adjustability=av.coping.adjustability,
            )
            snap_after = _snap(new_state)
            d = _delta(snap_before, snap_after)
            trace_steps.append(PipelineStep(
                name="regulation", label="Active Regulation",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Strategy: {regulation_result.strategy_used}, reduced {regulation_result.intensity_reduced:.0%}" if regulation_result.strategy_used else "No active regulation applied",
                impact=_impact(d), delta=d,
                details={"strategy": regulation_result.strategy_used, "intensity_reduced": round(regulation_result.intensity_reduced, 4), "capacity": round(session.regulator.regulation_capacity, 3), "breakthrough": regulation_result.breakthrough},
            ))

        # 6. Temporal effects (rumination/savoring) [DEV: stage 3+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"temporal"):
            snap_before = _snap(new_state)
            new_state = session.temporal.apply_temporal_effects(new_state, temporal_result)
            snap_after = _snap(new_state)
            d = _delta(snap_before, snap_after)
            active_effects = []
            if session.temporal._ruminations: active_effects.append("rumination")
            if session.temporal._savorings: active_effects.append("savoring")
            if session.temporal._topic_history: active_effects.append("anticipation")
            trace_steps.append(PipelineStep(
                name="temporal", label="Temporal Dynamics",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Active: {', '.join(active_effects)}" if active_effects else "No temporal effects active",
                impact=_impact(d), delta=d,
            ))
        else:
            trace_steps.append(PipelineStep(name="temporal", label="Temporal Dynamics", active=False, skipped_reason=_dev_skip))

        # 6b. Emotional Immune System — BYPASSED in extreme mode [DEV: stage 4+]
        t0 = time.perf_counter()
        if session.extreme_mode:
            trace_steps.append(PipelineStep(
                name="immune", label="Emotional Immune System",
                active=False, skipped_reason="Extreme mode: no emotional protection",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        elif not is_system_available(_dev_gate,"immune"):
            trace_steps.append(PipelineStep(name="immune", label="Emotional Immune System", active=False, skipped_reason=_dev_skip))
        else:
            snap_before = _snap(new_state)
            session.immune = update_immune_state(session.immune, new_state, request.message)
            new_state = apply_immune_protection(new_state, session.immune, request.message)
            snap_after = _snap(new_state)
            d = _delta(snap_before, snap_after)
            immune_info = get_immune_prompt_info(session.immune)
            trace_steps.append(PipelineStep(
                name="immune", label="Emotional Immune System",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Protection: {session.immune.protection_mode.value} (strength: {session.immune.protection_strength:.0%})" if session.immune.protection_mode.value != "none" else "No protection active",
                impact=_impact(d), delta=d,
                details={"mode": session.immune.protection_mode.value, "strength": round(session.immune.protection_strength, 3), "negative_streak": session.immune.negative_streak},
            ))

        # 6c. Narrative Self (identity coherence + effects) [DEV: stage 4+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"narrative"):
            snap_before = _snap(new_state)
            coherence_delta, is_coherent = check_coherence(
                session.narrative, request.message, new_state.primary_emotion,
            )
            new_state = apply_narrative_effects(
                new_state, coherence_delta, is_coherent, session.narrative.crisis.active,
            )
            session.narrative = detect_crisis(session.narrative, session.turn_count)
            narrative_info = get_narrative_prompt(session.narrative)
            snap_after = _snap(new_state)
            d = _delta(snap_before, snap_after)
            trace_steps.append(PipelineStep(
                name="narrative", label="Narrative Self",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Identity {'coherent' if is_coherent else 'challenged'}, coherence: {session.narrative.coherence_score:.0%}" + (", CRISIS" if session.narrative.crisis.active else ""),
                impact=_impact(d), delta=d,
                details={"coherent": is_coherent, "coherence_score": round(session.narrative.coherence_score, 3), "crisis": session.narrative.crisis.active},
            ))
        else:
            trace_steps.append(PipelineStep(name="narrative", label="Narrative Self", active=False, skipped_reason=_dev_skip))

        # 7. Meta-emotion [DEV: stage 3+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"meta_emotion"):
            is_new_emotion = new_state.primary_emotion != previous_state.primary_emotion
            meta_emotion = generate_meta_emotion(
                new_state, previous_state, session.value_system,
                regulation_success=regulation_result.strategy_used is not None and not regulation_result.breakthrough,
                is_new_emotion=is_new_emotion,
            )
            trace_steps.append(PipelineStep(
                name="meta_emotion", label="Meta-Emotion",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Feeling {meta_emotion.meta_response} about {meta_emotion.target_emotion.value}" if meta_emotion else "No meta-emotional response",
                impact="medium" if meta_emotion else "none",
                details={"active": meta_emotion is not None, "target": meta_emotion.target_emotion.value if meta_emotion else None, "response": meta_emotion.meta_response if meta_emotion else None, "intensity": round(meta_emotion.intensity, 3) if meta_emotion else 0},
            ))
        else:
            trace_steps.append(PipelineStep(name="meta_emotion", label="Meta-Emotion", active=False, skipped_reason=_dev_skip))

        # 7b. Self-Initiated Inquiry [DEV: stage 3+]
        if is_system_available(_dev_gate,"self_inquiry"):
            self_inquiry = check_self_inquiry(
                new_state, previous_state, meta_emotion, regulation_result, session.turn_count,
            )
            if self_inquiry:
                trace_steps.append(PipelineStep(
                    name="self_inquiry", label="Self-Initiated Inquiry",
                    active=True, duration_ms=0,
                    summary=f"[{self_inquiry.trigger.value}] {self_inquiry.inquiry_text[:80]}",
                    impact="high" if self_inquiry.intensity > 0.5 else "medium",
                    details={"trigger": self_inquiry.trigger.value, "intensity": round(self_inquiry.intensity, 3), "behavior": self_inquiry.suggested_behavior.value},
                ))

        # 8. Emergent emotions
        t0 = time.perf_counter()
        emergent = detect_emergent_emotions(new_state.emotional_stack)
        trace_steps.append(PipelineStep(
            name="emergent", label="Emergent Emotions",
            active=True, duration_ms=(time.perf_counter() - t0) * 1000,
            summary=f"Emerged: {', '.join(emergent)}" if emergent else "No emergent emotions detected",
            impact="high" if emergent else "none",
            details={"emergent_emotions": emergent},
        ))

        # 8b. Emotional Creativity [DEV: stage 4+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"creativity"):
            creativity_state = compute_creativity(new_state)
            trace_steps.append(PipelineStep(
                name="creativity", label="Emotional Creativity",
                active=True, duration_ms=(time.perf_counter() - t0) * 1000,
                summary=f"Mode: {creativity_state.thinking_mode}, level: {creativity_state.creativity_level:.0%}",
                impact="medium" if creativity_state.creativity_level > 0.3 else "low" if creativity_state.creativity_level > 0.1 else "none",
                details={"mode": creativity_state.thinking_mode, "level": round(creativity_state.creativity_level, 3), "temp_mod": round(creativity_state.temperature_modifier, 3)},
            ))
        else:
            trace_steps.append(PipelineStep(name="creativity", label="Emotional Creativity", active=False, skipped_reason=_dev_skip))

        # 8c. Emotional Forecasting (optional — only if enabled) [DEV: stage 3+]
        t0 = time.perf_counter()
        if is_system_available(_dev_gate,"forecasting") and session.forecast.enabled:
            user_emotion_est = estimate_user_emotion(
                session.shadow_state, session.user_model,
                detected_v, detected_a, signal_str,
            )
            session.forecast.user_emotion = user_emotion_est
            forecast_result = forecast_impact(
                new_state, user_emotion_est, session.user_model,
                session.forecast.valence_bias, session.forecast.arousal_bias,
            )
            session.forecast = record_forecast(session.forecast, forecast_result, session.turn_count)
            forecast_info = get_forecast_prompt(forecast_result)
        _fcast_active = is_system_available(_dev_gate,"forecasting") and session.forecast.enabled
        trace_steps.append(PipelineStep(
            name="forecasting", label="Emotional Forecasting",
            active=_fcast_active,
            skipped_reason="" if is_system_available(_dev_gate,"forecasting") else _dev_skip,
            duration_ms=(time.perf_counter() - t0) * 1000,
            summary=f"Predicted user impact: {session.forecast.predicted_impact:.2f}" if _fcast_active else "",
            impact="medium" if _fcast_active and getattr(session.forecast, 'risk_flag', False) else "low" if _fcast_active else "none",
        ))
    else:
        # Add skipped steps for non-advanced mode
        for name, label in [("reappraisal", "Cognitive Reappraisal"), ("regulation", "Active Regulation"),
                            ("temporal", "Temporal Dynamics"), ("immune", "Emotional Immune System"),
                            ("narrative", "Narrative Self"), ("meta_emotion", "Meta-Emotion"),
                            ("emergent", "Emergent Emotions"), ("creativity", "Emotional Creativity"),
                            ("forecasting", "Emotional Forecasting")]:
            trace_steps.append(PipelineStep(name=name, label=label, active=False, skipped_reason="Advanced mode off"))

    session.emotional_state = new_state

    # 9. Post-processing: update all systems
    t0 = time.perf_counter()
    mem_llm = None if session.lite_mode else llm_provider
    asyncio.create_task(session.memory.store(request.message, new_state, llm=mem_llm))

    if session.advanced_mode:
        if is_system_available(_dev_gate,"needs"):
            session.needs = update_needs(session.needs, request.message, appraisal_result.vector)
        if is_system_available(_dev_gate,"social"):
            session.user_model = update_user_model(
                session.user_model, request.message, appraisal_result.vector, new_state,
            )
        if is_system_available(_dev_gate,"schemas"):
            session.schemas.record_pattern(request.message, new_state.primary_emotion, new_state.intensity)
        if is_system_available(_dev_gate,"temporal"):
            session.temporal.process_post_turn(request.message, new_state, previous_state)

        # 9b. Narrative Self: update + growth + decay [DEV: stage 4+]
        if is_system_available(_dev_gate,"narrative"):
            session.narrative = update_narrative(
                session.narrative, session.narrative_tracker,
                request.message, new_state.primary_emotion, new_state.intensity,
                session.turn_count,
            )
            session.narrative = process_growth(
                session.narrative, request.message,
                previous_state.primary_emotion, new_state.primary_emotion,
                new_state.intensity,
                regulation_success=regulation_result.strategy_used is not None and not regulation_result.breakthrough,
                turn=session.turn_count,
            )
            session.narrative = decay_crisis_counter(session.narrative)
    trace_steps.append(PipelineStep(
        name="post_processing", label="Post-Processing",
        active=True, duration_ms=(time.perf_counter() - t0) * 1000,
        summary="Updated memory, needs, social model, schemas, narrative" if session.advanced_mode else "Stored memory",
        impact="low",
    ))

    # 9b. Emotional Discovery — detect novel states [TOGGLEABLE, DEV: stage 5]
    novel_detected = False
    if is_system_available(session.development, "discovery"):
        prev_count = session.discovery.total_novel_detected
        session.discovery = process_discovery_turn(
            session.discovery,
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
            stimulus=request.message,
            turn=session.turn_count,
            body_tension=new_state.body_state.tension,
            body_energy=new_state.body_state.energy,
            body_openness=new_state.body_state.openness,
            body_warmth=new_state.body_state.warmth,
            raw_mode=session.raw_mode,
            extreme_mode=session.extreme_mode,
        )
        novel_detected = session.discovery.total_novel_detected > prev_count
    discovery_prompt_text = get_discovery_prompt(session.discovery)

    # 9c. Phenomenology — generate qualia profile [TOGGLEABLE, DEV: stage 5]
    if session.phenomenology.enabled and is_system_available(session.development, "phenomenology"):
        pheno_mode = "extreme" if session.extreme_mode else ("raw" if session.raw_mode else "normal")
        process_phenomenology_turn(
            session.phenomenology,
            emotion_name=new_state.primary_emotion.value,
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
            body_tension=new_state.body_state.tension,
            body_warmth=new_state.body_state.warmth,
            turn=session.turn_count,
            mode=pheno_mode,
        )
    phenomenology_prompt_text = get_phenomenology_prompt(session.phenomenology)

    # 10. Behavior Modifier (raw / full / simple)
    t0 = time.perf_counter()
    if session.raw_mode:
        system_prompt = generate_raw_behavior_modifier(
            new_state,
            needs=session.needs,
            user_model=session.user_model,
            meta_emotion=meta_emotion,
            regulation_result=regulation_result,
            emergent_emotions=emergent,
            shadow_state=session.shadow_state,
            gut_feeling=gut_feeling,
            creativity=creativity_state,
            immune_info=immune_info,
            prediction_info=prediction_prompt_text,
            workspace_info=workspace_prompt_text,
            autobiographical_info=autobiographical_prompt_text,
            development_info=development_prompt_text,
            drives_info=drives_prompt_text,
            discovery_info=discovery_prompt_text,
            phenomenology_info=phenomenology_prompt_text,
            self_inquiry=self_inquiry,
            perception_text=perception_text,
        )
        behavior_label = "raw"
    elif session.advanced_mode:
        system_prompt = generate_behavior_modifier(
            new_state,
            needs=session.needs,
            user_model=session.user_model,
            meta_emotion=meta_emotion,
            regulation_result=regulation_result,
            emergent_emotions=emergent,
            shadow_state=session.shadow_state,
            gut_feeling=gut_feeling,
            creativity=creativity_state,
            immune_info=immune_info,
            narrative_info=narrative_info,
            forecast_info=forecast_info,
            prediction_info=prediction_prompt_text,
            workspace_info=workspace_prompt_text,
            autobiographical_info=autobiographical_prompt_text,
            development_info=development_prompt_text,
            drives_info=drives_prompt_text,
            discovery_info=discovery_prompt_text,
            phenomenology_info=phenomenology_prompt_text,
            self_inquiry=self_inquiry,
            perception_text=perception_text,
        )
        behavior_label = "full"
    else:
        system_prompt = generate_simple_behavior_modifier(new_state)
        behavior_label = "simple"
    trace_steps.append(PipelineStep(
        name="behavior", label="Behavior Modifier",
        active=True, duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Generated {behavior_label} system prompt for LLM",
        impact="medium",
    ))

    # 10c. Steering vector preparation (local models only)
    t0 = time.perf_counter()
    steering_hook: SteeringHook | None = None
    steering_active = False
    momentum_used = False
    if (
        session.steering_enabled
        and session.direct_mode
        and session.advanced_mode
        and llm_provider.supports_steering
        and steering_engine.is_ready
    ):
        # Configure momentum from personality (once per session, idempotent)
        if session.steering_momentum_enabled:
            session.steering_momentum.configure_from_personality(session.personality.neuroticism)

        model_obj = llm_provider.steerable_model
        if model_obj is not None:
            steering_hook = SteeringHook(
                model=model_obj,
                engine=steering_engine,
                valence=new_state.valence,
                arousal=new_state.arousal,
                dominance=new_state.dominance,
                certainty=new_state.certainty,
                intensity=new_state.intensity,
            )
            momentum_arg = session.steering_momentum if session.steering_momentum_enabled else None
            steering_active = steering_hook.apply(momentum=momentum_arg)
            momentum_used = session.steering_momentum_enabled and session.steering_momentum.has_history
    steering_status = "active" if steering_active else (
        "disabled" if not session.steering_enabled else (
            "no_vectors" if not steering_engine.is_ready else "provider_unsupported"
        )
    )
    momentum_info = session.steering_momentum.get_info() if session.steering_momentum_enabled else {}
    trace_steps.append(PipelineStep(
        name="steering", label="Steering Vectors",
        active=steering_active, duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Steering: {steering_status}" + (f" [momentum: {momentum_info.get('momentum_factor', 0):.2f}, history: {momentum_info.get('turns_stored', 0)}t]" if momentum_used else ""),
        impact="high" if steering_active else "none",
        details={
            "status": steering_status,
            "layers_hooked": list(steering_hook.vectors_applied.keys()) if steering_hook else [],
            "vector_norms": {str(k): round(v, 4) for k, v in steering_hook.vectors_applied.items()} if steering_hook else {},
            "layer_roles": steering_hook.layer_roles if steering_hook else {},
            "multilayer": True,
            "momentum": momentum_info,
        },
    ))

    # 10c2. Emotional prefix injection (local models only)
    t0 = time.perf_counter()
    prefix_hook: EmotionalPrefixHook | None = None
    prefix_active = False
    if (
        session.emotional_prefix_enabled
        and session.direct_mode
        and session.advanced_mode
        and llm_provider.supports_steering
        and steering_engine.is_ready
    ):
        model_obj = llm_provider.steerable_model
        if model_obj is not None:
            prefix_hook = EmotionalPrefixHook(
                model=model_obj,
                engine=steering_engine,
                valence=new_state.valence,
                arousal=new_state.arousal,
                dominance=new_state.dominance,
                certainty=new_state.certainty,
                intensity=new_state.intensity,
            )
            prefix_active = prefix_hook.apply()
    prefix_status = "active" if prefix_active else (
        "disabled" if not session.emotional_prefix_enabled else "provider_unsupported"
    )
    prefix_result = prefix_hook.result if prefix_hook else PrefixResult()
    trace_steps.append(PipelineStep(
        name="emotional_prefix", label="Emotional Prefix",
        active=prefix_active, duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Prefix: {prefix_status}" + (f" [{prefix_result.num_tokens} tokens, dominant={prefix_result.dominant_dimension}]" if prefix_active else ""),
        impact="medium" if prefix_active else "none",
        details={
            "status": prefix_status,
            "num_tokens": prefix_result.num_tokens,
            "embedding_norm": prefix_result.embedding_norm,
            "dominant_dimension": prefix_result.dominant_dimension,
        } if prefix_active else {},
    ))

    # 10d. Emotional attention modulation (local models only)
    t0 = time.perf_counter()
    attention_hook: AttentionHook | None = None
    attention_active = False
    attention_bias_result: AttentionBiasResult | None = None
    if (
        session.emotional_attention_enabled
        and session.direct_mode
        and session.advanced_mode
        and llm_provider.supports_steering  # same requirement: direct model access
    ):
        attention_bias_result = compute_attention_bias(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
        )
    attention_status = "disabled" if not session.emotional_attention_enabled else (
        "provider_unsupported" if not llm_provider.supports_steering else (
            "active" if attention_bias_result and attention_bias_result.token_biases else "inactive"
        )
    )
    trace_steps.append(PipelineStep(
        name="attention", label="Emotional Attention",
        active=bool(attention_bias_result and attention_bias_result.token_biases),
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Attention: {attention_status}",
        impact="medium" if attention_bias_result and attention_bias_result.token_biases else "none",
        details={
            "status": attention_status,
            "categories_active": attention_bias_result.categories_active if attention_bias_result else {},
            "broadening_factor": attention_bias_result.broadening_factor if attention_bias_result else 1.0,
            "words_biased": len(attention_bias_result.token_biases) if attention_bias_result else 0,
        },
    ))

    # 11. LLM response (with creativity temperature + emotional sampling)
    t0 = time.perf_counter()
    base_temperature = 0.7
    llm_temperature = base_temperature + creativity_state.temperature_modifier
    llm_temperature = max(0.1, min(1.5, llm_temperature))

    # Emotional sampler: compute sampling params from emotional state
    sampling: SamplingParams | None = None
    token_bias: TokenBiasResult | None = None
    if session.emotional_sampler_enabled and session.advanced_mode:
        sampling = compute_sampling_params(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
            base_temperature=llm_temperature,
        )
        llm_temperature = sampling.temperature  # Override with sampler's temperature
        token_bias = compute_token_bias(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
        )

    session.conversation.append({"role": "user", "content": request.message})
    if len(session.conversation) > 100:
        session.conversation = session.conversation[-100:]

    chat_messages = session.conversation[-10:]

    # Attention hook: needs tokenized input_ids to identify emotional positions.
    # Only applies when using TransformersProvider (direct model access + tokenizer).
    if (
        attention_bias_result
        and attention_bias_result.token_biases
        and llm_provider.supports_steering
    ):
        model_obj = llm_provider.steerable_model
        tokenizer_obj = getattr(llm_provider, "_tokenizer", None)
        if model_obj is not None and tokenizer_obj is not None:
            # Tokenize the full conversation to get input_ids for position matching
            token_id_biases = build_token_set(
                attention_bias_result.token_biases, tokenizer_obj,
            )
            if token_id_biases:
                # Build input text for tokenization (approximate — actual input built by generate())
                input_text = system_prompt + " " + " ".join(
                    m.get("content", "") for m in chat_messages
                )
                try:
                    input_ids = tokenizer_obj.encode(input_text, add_special_tokens=True)
                    attention_hook = AttentionHook(
                        model=model_obj,
                        token_biases=token_id_biases,
                        input_ids=input_ids,
                    )
                    attention_active = attention_hook.apply()
                except Exception:
                    logger.debug("Attention hook setup failed, continuing without")

    # Conditioning tokens (5.3b): prepend emotional tokens if adapter loaded
    effective_prompt = system_prompt
    conditioning_active = False
    if (
        session.conditioning_tokens_enabled
        and session.advanced_mode
        and hasattr(llm_provider, "has_adapter")
        and llm_provider.has_adapter
    ):
        from pathos.training.emotional_tokens import state_to_tokens
        emo_tokens = state_to_tokens(
            new_state.valence, new_state.arousal,
            new_state.dominance, new_state.certainty,
            new_state.intensity,
        )
        if emo_tokens and emo_tokens != "<EMO_NEUTRAL>":
            effective_prompt = f"{emo_tokens} {system_prompt}"
            conditioning_active = True

    try:
        gen_kwargs: dict = {
            "system_prompt": effective_prompt,
            "messages": chat_messages,
            "temperature": llm_temperature,
            "think": True,
        }
        if sampling is not None:
            gen_kwargs["top_p"] = sampling.top_p
            gen_kwargs["top_k"] = sampling.top_k
            gen_kwargs["repetition_penalty"] = sampling.repetition_penalty
            gen_kwargs["presence_penalty"] = sampling.presence_penalty
            gen_kwargs["frequency_penalty"] = sampling.frequency_penalty
        if token_bias is not None and token_bias.word_biases:
            # Pass word-level biases as string-keyed dict (Ollama format)
            # For TransformersProvider, resolve_token_ids() would be used
            # but word-level biases work directly with Ollama's logit_bias
            gen_kwargs["logit_bias"] = {w: b for w, b in token_bias.word_biases.items()}
        response_text = await llm_provider.generate(**gen_kwargs)
    except Exception as e:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail="LLM generation failed")
    finally:
        # Record raw vectors into momentum BEFORE removing hooks
        if steering_hook is not None and session.steering_momentum_enabled and steering_hook.raw_vectors:
            session.steering_momentum.record_turn(steering_hook.raw_vectors)
        # Always clean up hooks after generation
        if steering_hook is not None:
            steering_hook.remove()
        if prefix_hook is not None:
            prefix_hook.remove()
        if attention_hook is not None:
            attention_hook.remove()

    # Strip meta-anotaciones de estado emocional
    response_text = _strip_meta_annotations(response_text)

    session.conversation.append({"role": "assistant", "content": response_text})
    sampling_details: dict = {"temperature": round(llm_temperature, 3), "context_messages": len(chat_messages)}
    if sampling is not None:
        sampling_details["emotional_sampling"] = {
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "repetition_penalty": sampling.repetition_penalty,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "source": sampling.source,
        }
    if token_bias is not None and token_bias.word_biases:
        sampling_details["token_bias"] = {
            "words_biased": len(token_bias.word_biases),
            "categories_active": token_bias.categories_active,
        }
    trace_steps.append(PipelineStep(
        name="llm_response", label="LLM Response",
        active=True, duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Generated response (temp: {llm_temperature:.2f})" + (f" [{sampling.source}]" if sampling else ""),
        impact="high",
        details=sampling_details,
    ))

    # 11b. Self-Appraisal — evaluate own response against values (max 1 retry)
    t0 = time.perf_counter()
    sa_active = session.self_appraisal_enabled and session.advanced_mode and not session.raw_mode and not session.extreme_mode
    did_regenerate = False
    if sa_active:
        sa_result = evaluate_own_response(response_text, new_state, session.value_system)
        if sa_result.should_regenerate:
            # Adjust state with guilt and re-generate
            guilt_state = compute_guilt_state_adjustment(new_state, sa_result)
            guilt_prompt = generate_behavior_modifier(
                guilt_state,
                needs=session.needs,
                user_model=session.user_model,
                meta_emotion=meta_emotion,
                regulation_result=regulation_result,
                emergent_emotions=emergent,
                shadow_state=session.shadow_state,
                gut_feeling=gut_feeling,
                creativity=creativity_state,
                immune_info=immune_info,
                narrative_info=narrative_info,
                forecast_info=forecast_info,
                prediction_info=prediction_prompt_text,
                self_inquiry=self_inquiry,
                perception_text=perception_text,
            ) if session.advanced_mode and not session.extreme_mode else generate_simple_behavior_modifier(guilt_state)
            try:
                sa_result.original_response = response_text
                new_response = await llm_provider.generate(
                    system_prompt=guilt_prompt,
                    messages=chat_messages,
                    temperature=llm_temperature,
                    think=True,
                )
                response_text = _strip_meta_annotations(new_response)
                session.conversation[-1] = {"role": "assistant", "content": response_text}
                new_state = guilt_state
                did_regenerate = True
            except Exception:
                logger.warning("Self-appraisal re-generation failed, keeping original response")
    else:
        sa_result = SelfAppraisalResult(applied=False)
    trace_steps.append(PipelineStep(
        name="self_appraisal", label="Self-Appraisal",
        active=sa_active,
        skipped_reason="" if sa_active else ("Raw mode" if session.raw_mode else ("Extreme mode" if session.extreme_mode else "Disabled")),
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=f"Value alignment: {sa_result.value_alignment:.2f}, regenerated: {did_regenerate}" if sa_active else "",
        impact="high" if did_regenerate else ("medium" if sa_active and sa_result.value_alignment < 0.7 else "low"),
        details={"value_alignment": sa_result.value_alignment, "emotional_coherence": sa_result.emotional_coherence, "regenerated": did_regenerate} if sa_active else {},
    ))

    # 11c. World Model — predict emotional impact before sending (max 1 total retry with SA)
    t0 = time.perf_counter()
    wm_active = session.world_model_enabled and session.advanced_mode and not session.raw_mode and not session.extreme_mode
    wm_did_modify = False
    if wm_active:
        wm_result = simulate_response_impact(
            response_text, new_state, session.user_model, session.value_system,
        )
        # Only re-generate if world model flags AND self-appraisal didn't already re-generate
        if wm_result.should_modify and not did_regenerate:
            wm_state = compute_world_model_adjustment(new_state, wm_result)
            wm_prompt = generate_behavior_modifier(
                wm_state,
                needs=session.needs,
                user_model=session.user_model,
                meta_emotion=meta_emotion,
                regulation_result=regulation_result,
                emergent_emotions=emergent,
                shadow_state=session.shadow_state,
                gut_feeling=gut_feeling,
                creativity=creativity_state,
                immune_info=immune_info,
                narrative_info=narrative_info,
                forecast_info=forecast_info,
                prediction_info=prediction_prompt_text,
                self_inquiry=self_inquiry,
                perception_text=perception_text,
            ) if not session.extreme_mode else generate_simple_behavior_modifier(wm_state)
            try:
                new_response = await llm_provider.generate(
                    system_prompt=wm_prompt,
                    messages=chat_messages,
                    temperature=llm_temperature,
                    think=True,
                )
                response_text = _strip_meta_annotations(new_response)
                session.conversation[-1] = {"role": "assistant", "content": response_text}
                new_state = wm_state
                wm_did_modify = True
            except Exception:
                logger.warning("World model re-generation failed, keeping current response")
    else:
        wm_result = WorldModelResult(applied=False)
    trace_steps.append(PipelineStep(
        name="world_model", label="World Model",
        active=wm_active,
        skipped_reason="" if wm_active else ("Raw mode" if session.raw_mode else ("Extreme mode" if session.extreme_mode else "Disabled")),
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary=(
            f"Risk: {wm_result.emotional_risk:.2f}, alignment: {wm_result.value_alignment:.2f}, modified: {wm_did_modify}"
            if wm_active else ""
        ),
        impact="high" if wm_did_modify else ("medium" if wm_active and wm_result.emotional_risk > 0.3 else "low"),
        details={
            "emotional_risk": wm_result.emotional_risk,
            "value_alignment": wm_result.value_alignment,
            "predicted_user_effect": wm_result.predicted_user_impact.dominant_effect,
            "predicted_self_effect": wm_result.predicted_self_impact.dominant_effect,
            "meta_reaction": wm_result.meta_reaction.dominant_effect,
            "modified": wm_did_modify,
        } if wm_active else {},
    ))

    # Post-response: register this decision for somatic marking on next turn [ADVANCED]
    if session.advanced_mode:
        session.somatic_markers = register_pending_decision(session.somatic_markers, request.message)

    # Post-response: autobiographical memory encoding [OPT-IN]
    ws_sources = (
        [c.source for c in session.consciousness.current_result.conscious]
        if session.consciousness.current_result else []
    )
    session.autobiographical = process_autobiographical_turn(
        state=session.autobiographical,
        stimulus=request.message,
        emotional_state=new_state,
        response_summary=response_text[:200] if response_text else "",
        turn_number=session.turn_count,
        session_id=request.session_id,
        prediction_error=prediction_error.total_error if prediction_error else 0.0,
        workspace_contents=ws_sources,
        preconscious_count=len(session.consciousness.current_result.preconscious) if session.consciousness.current_result else 0,
    )

    # Post-response: development tracking [TOGGLEABLE]
    dev_transition_event = None
    if session.development.enabled:
        reg_used = regulation_result is not None and regulation_result.strategy_used is not None and not regulation_result.breakthrough
        track_experience(
            session.development,
            emotion_name=new_state.primary_emotion.value,
            intensity=new_state.intensity,
            regulation_used=reg_used,
        )
        # Extreme: experience counts x2 (accelerated learning from intensity)
        if session.extreme_mode:
            track_experience(
                session.development,
                emotion_name=new_state.primary_emotion.value,
                intensity=new_state.intensity,
                regulation_used=False,
            )
        dev_transition_event = attempt_transition(
            session.development,
            schemas_count=len(session.schemas.schemas),
            episodic_count=len(session.autobiographical.episodic_store.episodes),
            identities_count=len(session.narrative.identity_statements),
            crises_resolved=session.narrative.total_contradictions,
            turn_number=session.turn_count,
        )

    # 12. Voice generation (optional — only if voice mode is active)
    t0 = time.perf_counter()
    audio_available = False
    if session.voice_config.mode != VoiceMode.TEXT_ONLY and session.voice_config.auto_speak:
        tts = get_tts_service()
        if tts.is_initialized:
            try:
                user_lang = detect_language(request.message)
                voice_params = generate_voice_params(
                    new_state,
                    default_voice=session.voice_config.default_voice,
                    detected_language=user_lang,
                    user_backend=session.voice_config.tts_backend,
                )
                tts_text = prepare_text_for_tts(response_text, voice_params.stage_direction, state=new_state, backend=voice_params.backend)
                audio_bytes = await tts.generate_speech(tts_text, voice_params)
                session.last_audio = audio_bytes
                session.audio_history[session.turn_count] = audio_bytes
                if len(session.audio_history) > 20:
                    oldest_key = min(session.audio_history)
                    del session.audio_history[oldest_key]
                audio_available = True
            except Exception as e:
                logger.exception("Voice generation failed")
    voice_active = session.voice_config.mode != VoiceMode.TEXT_ONLY
    trace_steps.append(PipelineStep(
        name="voice", label="Voice Generation",
        active=voice_active,
        skipped_reason="" if voice_active else "Voice disabled",
        duration_ms=(time.perf_counter() - t0) * 1000,
        summary="Generated emotional speech" if audio_available else ("Voice active but no audio" if voice_active else ""),
        impact="medium" if audio_available else "none",
    ))

    total_dur = (time.perf_counter() - pipeline_start) * 1000
    mode_label = "lite" if session.lite_mode else ("advanced" if session.advanced_mode else "core")
    pipeline_trace = PipelineTrace(steps=trace_steps, total_duration_ms=total_dur, mode=mode_label)

    return ChatResponse(
        response=response_text,
        emotional_state=new_state,
        session_id=request.session_id,
        audio_available=audio_available,
        turn_number=session.turn_count,
        pipeline_trace=pipeline_trace,
    )


@app.get("/state/{session_id}", response_model=StateResponse)
async def get_state(session_id: str) -> StateResponse:
    """Retorna el estado emocional actual de una sesion."""
    session = state_manager.get_session(session_id)
    return StateResponse(
        emotional_state=session.emotional_state,
        session_id=session_id,
    )


@app.post("/reset/{session_id}")
async def reset_state(session_id: str) -> dict[str, str]:
    """Resetea el estado emocional de una sesion."""
    state_manager.reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/research/chat", response_model=ResearchChatResponse)
async def research_chat(request: ChatRequest) -> ResearchChatResponse:
    """Pipeline completo con TODOS los internos expuestos para investigacion."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    session = state_manager.get_session(request.session_id)
    session.turn_count += 1
    previous_state = session.emotional_state.model_copy(deep=True)

    # 0. Homeostasis
    state_before_homeostasis = session.emotional_state.model_copy(deep=True)
    if session.turn_count > 1:
        session.emotional_state = regulate(session.emotional_state, turns_elapsed=1)
        if session.advanced_mode:
            session.regulator.recover(session.personality.regulation_capacity_base)
    state_after_homeostasis = session.emotional_state.model_copy(deep=True)

    homeostasis_details = HomeostasisDetails(
        applied=session.turn_count > 1,
        state_before=state_before_homeostasis,
        state_after=state_after_homeostasis,
    )

    # 0a. Interoception — body-state feedback ascendente [ADVANCED]
    intero_result = None
    if session.advanced_mode:
        session.interoceptive_state = update_interoceptive_state(
            session.interoceptive_state, session.emotional_state.body_state,
        )
        intero_result = compute_interoceptive_feedback(
            session.interoceptive_state, session.emotional_state.body_state,
        )
        if intero_result.active:
            session.emotional_state.valence = max(-1.0, min(1.0,
                session.emotional_state.valence + intero_result.valence_delta))
            session.emotional_state.arousal = max(0.0, min(1.0,
                session.emotional_state.arousal + intero_result.arousal_delta))

    # 0b. Temporal pre-processing [ADVANCED]
    temporal_result = session.temporal.process_pre_turn(request.message) if session.advanced_mode else None

    # 0c. Predictive Processing — generate predictions [CORE]
    active_schemas_for_pred_r: list[tuple[str, str, float]] | None = None
    if session.advanced_mode and session.schemas.schemas:
        active_schemas_for_pred_r = [
            (s.trigger_category, s.typical_emotion.value, s.reinforcement_strength)
            for s in session.schemas.schemas
            if s.reinforcement_strength > 0.3
        ]
    current_predictions_r = predictive_engine.generate_predictions(
        predictive_state=session.predictive,
        conversation_history=session.conversation,
        user_model=session.user_model,
        mood=session.emotional_state.mood,
        emotional_state=session.emotional_state,
        active_schemas=active_schemas_for_pred_r,
    )
    session.predictive.current_predictions = current_predictions_r

    # 1. Appraisal + Memory amplification
    memories_before = len(session.memory)
    if session.lite_mode:
        appraisal_result = appraise_lite(request.message)
        amplification = await session.memory.check_amplification(request.message, llm=None)
    else:
        appraisal_task = asyncio.create_task(appraise(
            stimulus=request.message,
            value_system=session.value_system,
            llm=llm_provider,
            think=True,
        ))
        memory_task = asyncio.create_task(
            session.memory.check_amplification(request.message, llm=llm_provider),
        )
        try:
            appraisal_result = await appraisal_task
        except Exception as e:
            memory_task.cancel()
            logger.exception("Appraisal failed")
            raise HTTPException(status_code=500, detail="Appraisal failed")
        amplification = await memory_task

    appraisal = appraisal_result.vector

    # Compute raw dimensions from appraisal
    raw_valence = compute_valence(appraisal)
    raw_arousal = compute_arousal(appraisal)
    raw_dominance = compute_dominance(appraisal)
    raw_certainty = compute_certainty(appraisal)

    appraisal_details = AppraisalDetails(
        vector=appraisal,
        computed_valence=round(raw_valence, 4),
        computed_arousal=round(raw_arousal, 4),
        computed_dominance=round(raw_dominance, 4),
        computed_certainty=round(raw_certainty, 4),
    )

    if not session.lite_mode:
        amplification = await memory_task

    # --- Advanced systems pre-emotion (steps 2b-2g) ---
    needs_amp = 0.0
    schema_hint: PrimaryEmotion | None = None
    schema_amp = 0.0
    social_v_mod = 0.0
    social_i_mod = 0.0
    contagion_v = 0.0
    contagion_a = 0.0
    somatic_bias = 0.0
    gut_feeling: str | None = None
    detected_v = 0.0
    detected_a = 0.0
    signal_str = 0.0

    # Helper: dev gate shortcut
    _dev_r = session.development
    _dev_r_gate = _dev_r
    if _dev_r.enabled and (session.raw_mode or session.extreme_mode):
        _dev_r_gate = _dev_r.model_copy(update={"enabled": False})

    if session.advanced_mode:
        # 2b. Needs amplification [DEV: stage 2+]
        if is_system_available(_dev_r_gate, "needs"):
            needs_amp = compute_needs_amplification(session.needs, request.message)

        # 2c. Schema priming [DEV: stage 2+]
        if is_system_available(_dev_r_gate, "schemas"):
            schema_hint, schema_amp = session.schemas.check_priming(request.message)

        # 2d. Social modulation [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "social"):
            social_v_mod, social_i_mod = compute_social_modulation(session.user_model, raw_valence)

        # 2e. Emotion Contagion (pre-cognitive) [DEV: stage 2+]
        if is_system_available(_dev_r_gate, "contagion"):
            detected_v, detected_a, signal_str = detect_user_emotion(request.message)
            session.shadow_state = update_shadow_state(session.shadow_state, detected_v, detected_a, signal_str)
            contagion_v, contagion_a = compute_contagion_perturbation(
                session.shadow_state, session.emotional_state.valence,
                session.emotional_state.arousal, session.personality, session.user_model.rapport,
            )

        # 2f. Somatic Markers (pre-rational gut feeling) [DEV: stage 4+]
        if is_system_available(_dev_r_gate, "somatic"):
            session.somatic_markers = evaluate_user_reaction(
                session.somatic_markers, detected_v, session.turn_count,
            )
            somatic_bias, gut_feeling = compute_somatic_bias(session.somatic_markers, request.message)

        # 2g. Emotional Forecasting: evaluate previous forecast [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "forecasting") and session.forecast.enabled and session.turn_count > 1:
            session.forecast = evaluate_forecast(
                session.forecast, detected_v, detected_a, session.turn_count,
            )

    # 2i. Predictive Processing — compute prediction error [CORE]
    prediction_error_r = None
    prediction_prompt_text_r: str | None = None
    if current_predictions_r is not None:
        _det_intent_r = "unknown"
        if appraisal_result.emotion_hint:
            _hv = appraisal_result.emotion_hint.value
            if _hv in ("anger", "frustration"):
                _det_intent_r = "complaint"
            elif _hv in ("gratitude", "joy"):
                _det_intent_r = "greeting"
            elif _hv in ("fear", "anxiety"):
                _det_intent_r = "emotional_expression"
        _det_demand_r: DemandType | None = None
        if "?" in request.message:
            _det_demand_r = DemandType.HELP
        elif appraisal_result.emotion_hint and appraisal_result.emotion_hint.value in (
            "sadness", "fear", "anxiety", "helplessness",
        ):
            _det_demand_r = DemandType.EMOTIONAL

        prediction_error_r = compute_prediction_error(
            predictions=current_predictions_r,
            actual_stimulus=request.message,
            detected_user_valence=detected_v if session.advanced_mode else 0.0,
            detected_user_arousal=detected_a if session.advanced_mode else 0.3,
            detected_intent=_det_intent_r,
            detected_demand=_det_demand_r,
        )
        session.predictive = update_precision(session.predictive, prediction_error_r)
        session.predictive = record_prediction(session.predictive, current_predictions_r, prediction_error_r)
        session.predictive = decay_precision(session.predictive, elapsed_turns=1)
        prediction_prompt_text_r = get_prediction_prompt(prediction_error_r, session.predictive)

    # 2k. Global Workspace [TOGGLEABLE, DEV: stage 4+]
    workspace_prompt_text_r: str | None = None
    autobiographical_prompt_text_r: str | None = None
    if session.consciousness.enabled and session.advanced_mode and is_system_available(_dev_r_gate, "workspace"):
        ws_candidates_r: list[WorkspaceCandidate] = []
        if appraisal_result.emotion_hint:
            ws_candidates_r.append(generate_candidate(
                source="appraisal", content=f"Evaluated: {appraisal_result.emotion_hint.value}",
                urgency=0.6, relevance=appraisal.relevance.personal_significance,
                emotional_intensity=abs(raw_valence), emotion_tag=appraisal_result.emotion_hint.value,
                category="stimulus",
            ))
        if schema_hint:
            ws_candidates_r.append(generate_candidate(
                source="schema", content=f"Schema: {schema_hint.value}",
                urgency=0.5 + schema_amp * 0.3, relevance=0.7 if schema_amp > 0.1 else 0.3,
                emotional_intensity=schema_amp, emotion_tag=schema_hint.value, category="pattern",
            ))
        if signal_str > 0.15:
            ws_candidates_r.append(generate_candidate(
                source="contagion", content=f"User emotion (v={detected_v:+.2f})",
                urgency=signal_str * 0.6, relevance=signal_str,
                emotional_intensity=abs(detected_v) * 0.8 + detected_a * 0.2,
                emotion_tag="anxiety" if detected_v < -0.2 else "joy" if detected_v > 0.2 else "neutral",
                category="social",
            ))
        if prediction_error_r and prediction_error_r.surprise_type.value != "none":
            ws_candidates_r.append(generate_candidate(
                source="predictive", content=f"Surprise {prediction_error_r.surprise_type.value}",
                urgency=prediction_error_r.vulnerability * 0.9, relevance=0.8,
                emotional_intensity=prediction_error_r.total_error, category="prediction",
            ))
        session.consciousness = process_workspace_turn(session.consciousness, ws_candidates_r)
        if session.consciousness.current_result:
            workspace_prompt_text_r = get_workspace_prompt(session.consciousness.current_result)

    # 2m. Autobiographical Memory — prompt from existing memories [OPT-IN]
    autobiographical_prompt_text_r = get_autobiographical_prompt(session.autobiographical)

    # 2n. Development — prompt from current stage [TOGGLEABLE]
    development_prompt_text_r = get_development_prompt(session.development)

    # 2o. Drives — update + process goals + generate new goals [TOGGLEABLE]
    drives_updates_r: list = []
    drives_impacts_r: list = []
    if is_system_available(session.development, "drives"):
        session.drives, drives_updates_r = update_drives(
            session.drives, request.message, session.personality,
            session.turn_count, getattr(session.user_model, 'rapport', 0.5),
        )
        session.drives, drives_impacts_r = process_goals(
            session.drives, request.message, session.turn_count,
        )
        session.drives = attempt_goal_generation(
            session.drives, request.message, session.turn_count,
        )
    drives_prompt_text_r = get_drives_prompt(session.drives)

    # 3. Mood congruence bias
    valence_bias, arousal_bias = compute_mood_congruence_bias(session.emotional_state.mood)
    mood_congruence = MoodCongruenceDetails(
        valence_bias=round(valence_bias, 4),
        arousal_bias=round(arousal_bias, 4),
        mood_label=session.emotional_state.mood.label.value,
        mood_trend=session.emotional_state.mood.trend,
    )

    # 4. Intensity before amplification
    intensity_raw = compute_intensity(appraisal, raw_valence, raw_arousal)

    # 4b. External signals (from session config — only active sources)
    sig_v, sig_a, sig_d, perception_text = _get_session_signals(session)

    # 5. Emotion generation
    effective_hint = schema_hint if schema_hint and not appraisal_result.emotion_hint else appraisal_result.emotion_hint
    pred_modulation_r = prediction_error_to_emotion_modulation(
        prediction_error_r, session.predictive.predictive_weight,
    )
    new_state = generate_emotion(
        appraisal=appraisal,
        current_state=session.emotional_state,
        stimulus=request.message,
        amplification=amplification + schema_amp,
        emotion_hint=effective_hint,
        dynamics=session.dynamics if session.advanced_mode else None,
        needs_amplification=needs_amp,
        social_valence_mod=social_v_mod + somatic_bias + sig_v,
        social_intensity_mod=social_i_mod,
        contagion_valence=contagion_v,
        contagion_arousal=contagion_a + sig_a,
        coupling=session.coupling if session.advanced_mode else None,
        predictive_modulation=pred_modulation_r,
    )

    # 5b. Calibration
    new_state = apply_calibration(new_state, session.calibration_profile)

    # 5b2. Development emotion filtering [TOGGLEABLE] — raw/extreme bypass
    if _dev_r.enabled and new_state.emotional_stack and not session.raw_mode and not session.extreme_mode:
        new_state.emotional_stack = filter_emotions_by_stage(_dev_r, new_state.emotional_stack)
        if not is_emotion_available(_dev_r, new_state.primary_emotion.value):
            _avail_emos_r = [(e, a) for e, a in new_state.emotional_stack.items() if is_emotion_available(_dev_r, e)]
            if _avail_emos_r:
                _best_r = max(_avail_emos_r, key=lambda x: x[1])
                try:
                    new_state.primary_emotion = PrimaryEmotion(_best_r[0])
                except ValueError:
                    pass
        if new_state.secondary_emotion and not is_emotion_available(_dev_r, new_state.secondary_emotion.value):
            new_state.secondary_emotion = None
        apply_stage_modifiers(_dev_r, new_state)

    # --- Advanced systems post-emotion (steps 5c-5i) ---
    reappraisal_result = None
    regulation_result = RegulationResult()
    immune_info: str | None = None
    narrative_info: str | None = None
    meta_emotion: MetaEmotion | None = None
    self_inquiry: SelfInquiry | None = None
    emergent: list[str] = []
    creativity_state = CreativityState()
    forecast_info: str | None = None

    if session.advanced_mode:
        # 5c. Reappraisal [DEV: stage 4+]
        if is_system_available(_dev_r_gate, "reappraisal"):
            new_state, reappraisal_result = reappraise(new_state, session.regulator.regulation_capacity)

        # 5d. Active regulation [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "regulation"):
            new_state, regulation_result = session.regulator.regulate(
                new_state, session.personality.regulation_capacity_base,
                coping_control=appraisal.coping.control,
                coping_adjustability=appraisal.coping.adjustability,
            )

        # 5e. Temporal effects [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "temporal"):
            new_state = session.temporal.apply_temporal_effects(new_state, temporal_result)

        # 5e2. Emotional Immune System [DEV: stage 4+]
        if is_system_available(_dev_r_gate, "immune"):
            session.immune = update_immune_state(session.immune, new_state, request.message)
            new_state = apply_immune_protection(new_state, session.immune, request.message)
            immune_info = get_immune_prompt_info(session.immune)

        # 5e3. Narrative Self (identity coherence + effects) [DEV: stage 4+]
        if is_system_available(_dev_r_gate, "narrative"):
            coherence_delta, is_coherent = check_coherence(
                session.narrative, request.message, new_state.primary_emotion,
            )
            new_state = apply_narrative_effects(
                new_state, coherence_delta, is_coherent, session.narrative.crisis.active,
            )
            session.narrative = detect_crisis(session.narrative, session.turn_count)
            narrative_info = get_narrative_prompt(session.narrative)

        # 5f. Meta-emotion [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "meta_emotion"):
            is_new_emotion = new_state.primary_emotion != previous_state.primary_emotion
            meta_emotion = generate_meta_emotion(
                new_state, previous_state, session.value_system,
                regulation_success=regulation_result.strategy_used is not None and not regulation_result.breakthrough,
                is_new_emotion=is_new_emotion,
            )

        # 5f-b. Self-Initiated Inquiry [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "self_inquiry"):
            self_inquiry = check_self_inquiry(
                new_state, previous_state, meta_emotion, regulation_result, session.turn_count,
            )

        # 5g. Emergent emotions
        emergent = detect_emergent_emotions(new_state.emotional_stack)

        # 5h. Emotional Creativity [DEV: stage 4+]
        if is_system_available(_dev_r_gate, "creativity"):
            creativity_state = compute_creativity(new_state)

        # 5i. Emotional Forecasting (optional — only if enabled) [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "forecasting") and session.forecast.enabled:
            user_emotion_est = estimate_user_emotion(
                session.shadow_state, session.user_model,
                detected_v, detected_a, signal_str,
            )
            session.forecast.user_emotion = user_emotion_est
            forecast_result = forecast_impact(
                new_state, user_emotion_est, session.user_model,
                session.forecast.valence_bias, session.forecast.arousal_bias,
            )
            session.forecast = record_forecast(session.forecast, forecast_result, session.turn_count)
            forecast_info = get_forecast_prompt(forecast_result)

    session.emotional_state = new_state
    session.state_history.append(new_state)
    if len(session.state_history) > 50:
        session.state_history = session.state_history[-50:]

    emotion_generation = EmotionGenerationDetails(
        raw_valence=round(raw_valence, 4),
        raw_arousal=round(raw_arousal, 4),
        raw_dominance=round(raw_dominance, 4),
        raw_certainty=round(raw_certainty, 4),
        blended_valence=new_state.valence,
        blended_arousal=new_state.arousal,
        blended_dominance=new_state.dominance,
        blended_certainty=new_state.certainty,
        intensity_before_amplification=round(intensity_raw, 4),
        intensity_after_amplification=new_state.intensity,
    )

    # 6. Post-processing: update all systems
    mem_llm = None if session.lite_mode else llm_provider
    stored = await session.memory.store(request.message, new_state, llm=mem_llm)

    if session.advanced_mode:
        # 6a. Needs update [DEV: stage 2+]
        if is_system_available(_dev_r_gate, "needs"):
            session.needs = update_needs(session.needs, request.message, appraisal)
        # 6a2. Social update [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "social"):
            session.user_model = update_user_model(
                session.user_model, request.message, appraisal, new_state,
            )
        # 6a3. Schema recording [DEV: stage 2+]
        if is_system_available(_dev_r_gate, "schemas"):
            session.schemas.record_pattern(request.message, new_state.primary_emotion, new_state.intensity)
        # 6a4. Temporal post-turn [DEV: stage 3+]
        if is_system_available(_dev_r_gate, "temporal"):
            session.temporal.process_post_turn(request.message, new_state, previous_state)

        # 6b. Narrative Self: update + growth + decay [DEV: stage 4+]
        if is_system_available(_dev_r_gate, "narrative"):
            session.narrative = update_narrative(
                session.narrative, session.narrative_tracker,
                request.message, new_state.primary_emotion, new_state.intensity,
                session.turn_count,
            )
            session.narrative = process_growth(
                session.narrative, request.message,
                previous_state.primary_emotion, new_state.primary_emotion,
                new_state.intensity,
                regulation_success=regulation_result.strategy_used is not None and not regulation_result.breakthrough,
                turn=session.turn_count,
            )
            session.narrative = decay_crisis_counter(session.narrative)

    memory_details = MemoryAmplificationDetails(
        amplification_factor=round(amplification, 4),
        memories_count=memories_before,
        memory_stored=stored is not None,
    )

    # 6b. Emotional Discovery [TOGGLEABLE, DEV: stage 5]
    novel_detected_r = False
    if is_system_available(session.development, "discovery"):
        prev_count_r = session.discovery.total_novel_detected
        session.discovery = process_discovery_turn(
            session.discovery,
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
            stimulus=request.message,
            turn=session.turn_count,
            body_tension=new_state.body_state.tension,
            body_energy=new_state.body_state.energy,
            body_openness=new_state.body_state.openness,
            body_warmth=new_state.body_state.warmth,
        )
        novel_detected_r = session.discovery.total_novel_detected > prev_count_r
    discovery_prompt_text_r = get_discovery_prompt(session.discovery)

    # 6c. Phenomenology — qualia profile [TOGGLEABLE, DEV: stage 5]
    if session.phenomenology.enabled and is_system_available(session.development, "phenomenology"):
        pheno_mode_r = "extreme" if session.extreme_mode else ("raw" if session.raw_mode else "normal")
        process_phenomenology_turn(
            session.phenomenology,
            emotion_name=new_state.primary_emotion.value,
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
            body_tension=new_state.body_state.tension,
            body_warmth=new_state.body_state.warmth,
            turn=session.turn_count,
            mode=pheno_mode_r,
        )
    phenomenology_prompt_text_r = get_phenomenology_prompt(session.phenomenology)

    # 7. Behavior modifier
    if session.advanced_mode:
        system_prompt = generate_behavior_modifier(
            new_state,
            needs=session.needs,
            user_model=session.user_model,
            meta_emotion=meta_emotion,
            regulation_result=regulation_result,
            emergent_emotions=emergent,
            shadow_state=session.shadow_state,
            gut_feeling=gut_feeling,
            creativity=creativity_state,
            immune_info=immune_info,
            narrative_info=narrative_info,
            forecast_info=forecast_info,
            prediction_info=prediction_prompt_text_r,
            workspace_info=workspace_prompt_text_r,
            autobiographical_info=autobiographical_prompt_text_r,
            development_info=development_prompt_text_r,
            drives_info=drives_prompt_text_r,
            discovery_info=discovery_prompt_text_r,
            phenomenology_info=phenomenology_prompt_text_r,
            self_inquiry=self_inquiry,
            perception_text=perception_text,
        )
    else:
        system_prompt = generate_simple_behavior_modifier(new_state)

    # 7b. Steering vector preparation (local models only)
    steering_hook_r: SteeringHook | None = None
    steering_active_r = False
    if (
        session.steering_enabled
        and session.direct_mode
        and session.advanced_mode
        and llm_provider.supports_steering
        and steering_engine.is_ready
    ):
        # Configure momentum from personality
        if session.steering_momentum_enabled:
            session.steering_momentum.configure_from_personality(session.personality.neuroticism)

        model_obj = llm_provider.steerable_model
        if model_obj is not None:
            steering_hook_r = SteeringHook(
                model=model_obj,
                engine=steering_engine,
                valence=new_state.valence,
                arousal=new_state.arousal,
                dominance=new_state.dominance,
                certainty=new_state.certainty,
                intensity=new_state.intensity,
            )
            momentum_arg_r = session.steering_momentum if session.steering_momentum_enabled else None
            steering_active_r = steering_hook_r.apply(momentum=momentum_arg_r)
    steering_status_r = "active" if steering_active_r else (
        "disabled" if not session.steering_enabled else (
            "no_vectors" if not steering_engine.is_ready else "provider_unsupported"
        )
    )

    # 7b2. Emotional prefix injection (local models only)
    prefix_hook_r: EmotionalPrefixHook | None = None
    prefix_active_r = False
    if (
        session.emotional_prefix_enabled
        and session.direct_mode
        and session.advanced_mode
        and llm_provider.supports_steering
        and steering_engine.is_ready
    ):
        model_obj_p = llm_provider.steerable_model
        if model_obj_p is not None:
            prefix_hook_r = EmotionalPrefixHook(
                model=model_obj_p,
                engine=steering_engine,
                valence=new_state.valence,
                arousal=new_state.arousal,
                dominance=new_state.dominance,
                certainty=new_state.certainty,
                intensity=new_state.intensity,
            )
            prefix_active_r = prefix_hook_r.apply()
    prefix_status_r = "active" if prefix_active_r else (
        "disabled" if not session.emotional_prefix_enabled else "provider_unsupported"
    )
    prefix_result_r = prefix_hook_r.result if prefix_hook_r else PrefixResult()

    # 7c. Emotional attention modulation (local models only)
    attention_hook_r: AttentionHook | None = None
    attention_active_r = False
    attention_bias_r: AttentionBiasResult | None = None
    if (
        session.emotional_attention_enabled
        and session.direct_mode
        and session.advanced_mode
        and llm_provider.supports_steering
    ):
        attention_bias_r = compute_attention_bias(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
        )
    attention_status_r = "disabled" if not session.emotional_attention_enabled else (
        "provider_unsupported" if not llm_provider.supports_steering else (
            "active" if attention_bias_r and attention_bias_r.token_biases else "inactive"
        )
    )

    # 8. Conversation + LLM response (with creativity temperature + emotional sampling)
    base_temperature = 0.7
    llm_temperature = base_temperature + creativity_state.temperature_modifier
    llm_temperature = max(0.1, min(1.5, llm_temperature))

    # Emotional sampler
    sampling_r: SamplingParams | None = None
    token_bias_r: TokenBiasResult | None = None
    if session.emotional_sampler_enabled and session.advanced_mode:
        sampling_r = compute_sampling_params(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
            base_temperature=llm_temperature,
        )
        llm_temperature = sampling_r.temperature
        token_bias_r = compute_token_bias(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            intensity=new_state.intensity,
        )

    session.conversation.append({"role": "user", "content": request.message})
    if len(session.conversation) > 100:
        session.conversation = session.conversation[-100:]

    chat_messages = session.conversation[-10:]

    # Attention hook for research endpoint (same logic as /chat)
    if (
        attention_bias_r
        and attention_bias_r.token_biases
        and llm_provider.supports_steering
    ):
        model_obj = llm_provider.steerable_model
        tokenizer_obj = getattr(llm_provider, "_tokenizer", None)
        if model_obj is not None and tokenizer_obj is not None:
            token_id_biases_r = build_token_set(
                attention_bias_r.token_biases, tokenizer_obj,
            )
            if token_id_biases_r:
                input_text = system_prompt + " " + " ".join(
                    m.get("content", "") for m in chat_messages
                )
                try:
                    input_ids_r = tokenizer_obj.encode(input_text, add_special_tokens=True)
                    attention_hook_r = AttentionHook(
                        model=model_obj,
                        token_biases=token_id_biases_r,
                        input_ids=input_ids_r,
                    )
                    attention_active_r = attention_hook_r.apply()
                except Exception:
                    logger.debug("Attention hook setup failed in research, continuing without")

    # Conditioning tokens (5.3b) for research endpoint
    effective_prompt_r = system_prompt
    conditioning_active_r = False
    if (
        session.conditioning_tokens_enabled
        and session.advanced_mode
        and hasattr(llm_provider, "has_adapter")
        and llm_provider.has_adapter
    ):
        from pathos.training.emotional_tokens import state_to_tokens
        emo_tokens_r = state_to_tokens(
            new_state.valence, new_state.arousal,
            new_state.dominance, new_state.certainty,
            new_state.intensity,
        )
        if emo_tokens_r and emo_tokens_r != "<EMO_NEUTRAL>":
            effective_prompt_r = f"{emo_tokens_r} {system_prompt}"
            conditioning_active_r = True

    try:
        gen_kwargs_r: dict = {
            "system_prompt": effective_prompt_r,
            "messages": chat_messages,
            "temperature": llm_temperature,
            "think": True,
        }
        if sampling_r is not None:
            gen_kwargs_r["top_p"] = sampling_r.top_p
            gen_kwargs_r["top_k"] = sampling_r.top_k
            gen_kwargs_r["repetition_penalty"] = sampling_r.repetition_penalty
            gen_kwargs_r["presence_penalty"] = sampling_r.presence_penalty
            gen_kwargs_r["frequency_penalty"] = sampling_r.frequency_penalty
        if token_bias_r is not None and token_bias_r.word_biases:
            gen_kwargs_r["logit_bias"] = {w: b for w, b in token_bias_r.word_biases.items()}
        response_text = await llm_provider.generate(**gen_kwargs_r)
    except Exception as e:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail="LLM generation failed")
    finally:
        # Record raw vectors into momentum BEFORE removing hooks
        if steering_hook_r is not None and session.steering_momentum_enabled and steering_hook_r.raw_vectors:
            session.steering_momentum.record_turn(steering_hook_r.raw_vectors)
        if steering_hook_r is not None:
            steering_hook_r.remove()
        if prefix_hook_r is not None:
            prefix_hook_r.remove()
        if attention_hook_r is not None:
            attention_hook_r.remove()

    # Strip meta-anotaciones de estado emocional
    response_text = _strip_meta_annotations(response_text)

    session.conversation.append({"role": "assistant", "content": response_text})

    # 8a. Self-Appraisal — evaluate own response against values (max 1 retry)
    sa_active = session.self_appraisal_enabled and session.advanced_mode and not session.raw_mode and not session.extreme_mode
    did_regenerate = False
    if sa_active:
        sa_result = evaluate_own_response(response_text, new_state, session.value_system)
        if sa_result.should_regenerate:
            guilt_state = compute_guilt_state_adjustment(new_state, sa_result)
            guilt_prompt = generate_behavior_modifier(
                guilt_state,
                needs=session.needs,
                user_model=session.user_model,
                meta_emotion=meta_emotion,
                regulation_result=regulation_result,
                emergent_emotions=emergent,
                shadow_state=session.shadow_state,
                gut_feeling=gut_feeling,
                creativity=creativity_state,
                immune_info=immune_info,
                narrative_info=narrative_info,
                forecast_info=forecast_info,
                prediction_info=prediction_prompt_text_r,
                self_inquiry=self_inquiry,
                perception_text=perception_text,
            )
            try:
                sa_result.original_response = response_text
                new_response = await llm_provider.generate(
                    system_prompt=guilt_prompt,
                    messages=chat_messages,
                    temperature=llm_temperature,
                    think=True,
                )
                response_text = _strip_meta_annotations(new_response)
                session.conversation[-1] = {"role": "assistant", "content": response_text}
                new_state = guilt_state
                did_regenerate = True
            except Exception:
                logger.warning("Self-appraisal re-generation failed, keeping original response")
    else:
        sa_result = SelfAppraisalResult(applied=False)

    # 8b. World Model — predict emotional impact before sending
    wm_active_r = session.world_model_enabled and session.advanced_mode and not session.raw_mode and not session.extreme_mode
    wm_did_modify_r = False
    if wm_active_r:
        wm_result_r = simulate_response_impact(
            response_text, new_state, session.user_model, session.value_system,
        )
        if wm_result_r.should_modify and not did_regenerate:
            wm_state_r = compute_world_model_adjustment(new_state, wm_result_r)
            wm_prompt_r = generate_behavior_modifier(
                wm_state_r,
                needs=session.needs,
                user_model=session.user_model,
                meta_emotion=meta_emotion,
                regulation_result=regulation_result,
                emergent_emotions=emergent,
                shadow_state=session.shadow_state,
                gut_feeling=gut_feeling,
                creativity=creativity_state,
                immune_info=immune_info,
                narrative_info=narrative_info,
                forecast_info=forecast_info,
                prediction_info=prediction_prompt_text_r,
                self_inquiry=self_inquiry,
                perception_text=perception_text,
            )
            try:
                new_response = await llm_provider.generate(
                    system_prompt=wm_prompt_r,
                    messages=chat_messages,
                    temperature=llm_temperature,
                    think=True,
                )
                response_text = _strip_meta_annotations(new_response)
                session.conversation[-1] = {"role": "assistant", "content": response_text}
                new_state = wm_state_r
                wm_did_modify_r = True
            except Exception:
                logger.warning("World model re-generation failed, keeping current response")
    else:
        wm_result_r = WorldModelResult(applied=False)

    # Post-response: register decision for somatic marking [ADVANCED]
    if session.advanced_mode:
        session.somatic_markers = register_pending_decision(session.somatic_markers, request.message)

    # 8c. Voice generation (optional)
    voice_audio_available = False
    voice_params_result = None
    if session.voice_config.mode != VoiceMode.TEXT_ONLY and session.voice_config.auto_speak:
        tts = get_tts_service()
        if tts.is_initialized:
            try:
                user_lang = detect_language(request.message)
                voice_params_result = generate_voice_params(
                    new_state,
                    default_voice=session.voice_config.default_voice,
                    detected_language=user_lang,
                    user_backend=session.voice_config.tts_backend,
                )
                tts_text = prepare_text_for_tts(response_text, voice_params_result.stage_direction, state=new_state, backend=voice_params_result.backend)
                audio_bytes = await tts.generate_speech(tts_text, voice_params_result)
                session.last_audio = audio_bytes
                session.audio_history[session.turn_count] = audio_bytes
                if len(session.audio_history) > 20:
                    oldest_key = min(session.audio_history)
                    del session.audio_history[oldest_key]
                voice_audio_available = True
            except Exception:
                logger.exception("Voice generation failed")

    # 9. Authenticity metrics
    m_coherence = coherence(new_state, appraisal)
    m_continuity = continuity(new_state, previous_state)
    m_proportionality = proportionality(new_state, appraisal)
    m_recovery = recovery(session.state_history)
    m_overall = (m_coherence + m_continuity + m_proportionality + m_recovery) / 4

    metrics = AuthenticityMetrics(
        coherence=round(m_coherence, 4),
        continuity=round(m_continuity, 4),
        proportionality=round(m_proportionality, 4),
        recovery=round(m_recovery, 4),
        overall=round(m_overall, 4),
    )

    # Build advanced system details
    needs_details = NeedsDetails(
        connection=session.needs.connection,
        competence=session.needs.competence,
        autonomy=session.needs.autonomy,
        coherence=session.needs.coherence,
        stimulation=session.needs.stimulation,
        safety=session.needs.safety,
        amplification=round(needs_amp, 4),
    )

    social_details = SocialDetails(
        perceived_intent=session.user_model.perceived_intent,
        perceived_engagement=session.user_model.perceived_engagement,
        rapport=session.user_model.rapport,
        trust_level=session.user_model.trust_level,
        communication_style=session.user_model.communication_style,
        valence_modulation=social_v_mod,
        intensity_modulation=social_i_mod,
    )

    cap_before = session.regulator.regulation_capacity + (regulation_result.capacity_spent if regulation_result else 0)
    regulation_details = RegulationDetails(
        strategy_used=regulation_result.strategy_used,
        intensity_reduced=round(regulation_result.intensity_reduced, 4),
        capacity_before=round(min(cap_before, 1.0), 4),
        capacity_after=round(session.regulator.regulation_capacity, 4),
        breakthrough=regulation_result.breakthrough,
        suppression_dissonance=round(session.regulator.suppression_dissonance, 4),
    )

    if reappraisal_result is not None:
        reappraisal_details = ReappraisalDetails(
            applied=reappraisal_result.applied,
            strategy=reappraisal_result.strategy,
            original_emotion=reappraisal_result.original_emotion.value if reappraisal_result.original_emotion else None,
            reappraised_emotion=reappraisal_result.reappraised_emotion.value if reappraisal_result.reappraised_emotion else None,
            intensity_change=round(reappraisal_result.intensity_change, 4),
            valence_change=round(reappraisal_result.valence_change, 4),
        )
    else:
        reappraisal_details = ReappraisalDetails(applied=False)

    if temporal_result is not None:
        temporal_details = TemporalDetails(
            rumination_active=temporal_result.rumination_active,
            rumination_emotion=temporal_result.rumination_emotion.value if temporal_result.rumination_emotion else None,
            rumination_intensity=round(temporal_result.rumination_intensity, 4),
            savoring_active=temporal_result.savoring_active,
            savoring_emotion=temporal_result.savoring_emotion.value if temporal_result.savoring_emotion else None,
            anticipation_active=temporal_result.anticipation_active,
            anticipation_emotion=temporal_result.anticipation_emotion.value if temporal_result.anticipation_emotion else None,
            anticipation_intensity=round(temporal_result.anticipation_intensity, 4),
        )
    else:
        temporal_details = TemporalDetails(
            rumination_active=False, savoring_active=False,
            anticipation_active=False, rumination_intensity=0.0, anticipation_intensity=0.0,
        )

    meta_details = MetaEmotionDetails(
        active=meta_emotion is not None,
        target_emotion=meta_emotion.target_emotion.value if meta_emotion else None,
        meta_response=meta_emotion.meta_response if meta_emotion else None,
        intensity=round(meta_emotion.intensity, 4) if meta_emotion else 0.0,
        reason=meta_emotion.reason if meta_emotion else "",
    )

    schema_hint_name = schema_hint.value if schema_hint else None
    schema_details = SchemaDetails(
        schemas_count=len(session.schemas),
        primed_emotion=schema_hint_name,
        priming_amplification=round(schema_amp, 4),
        pending_patterns=len(session.schemas._pattern_counts),
    )

    personality_details = PersonalityDetails(
        openness=session.personality.openness,
        conscientiousness=session.personality.conscientiousness,
        extraversion=session.personality.extraversion,
        agreeableness=session.personality.agreeableness,
        neuroticism=session.personality.neuroticism,
        variability=round(session.personality.variability, 4),
        regulation_capacity_base=round(session.personality.regulation_capacity_base, 4),
    )

    contagion_details = ContagionDetails(
        detected_valence=round(detected_v, 4),
        detected_arousal=round(detected_a, 4),
        signal_strength=round(signal_str, 4),
        shadow_valence=session.shadow_state.valence,
        shadow_arousal=session.shadow_state.arousal,
        contagion_perturbation_v=contagion_v,
        contagion_perturbation_a=contagion_a,
        accumulated_contagion=session.shadow_state.accumulated_contagion,
        susceptibility=round(session.personality.contagion_susceptibility, 4),
    )

    somatic_details = SomaticDetails(
        markers_count=len(session.somatic_markers.markers),
        somatic_bias=somatic_bias,
        gut_feeling=gut_feeling,
        pending_category=session.somatic_markers.pending_category,
    )

    creativity_details = CreativityDetails(
        thinking_mode=creativity_state.thinking_mode.value,
        creativity_level=creativity_state.creativity_level,
        temperature_modifier=creativity_state.temperature_modifier,
        active_instructions=creativity_state.active_instructions,
        triggered_by=creativity_state.triggered_by,
    )

    immune_details = ImmuneDetails(
        protection_mode=session.immune.protection_mode.value,
        protection_strength=session.immune.protection_strength,
        reactivity_dampening=session.immune.reactivity_dampening,
        negative_streak=session.immune.negative_streak,
        peak_negative_intensity=session.immune.peak_negative_intensity,
        recovery_turns=session.immune.recovery_turns,
        total_activations=session.immune.total_activations,
        compartmentalized_topics=session.immune.compartmentalized_topics,
    )

    # Build narrative details
    top_stmts = sorted(
        session.narrative.identity_statements,
        key=lambda s: s.strength,
        reverse=True,
    )[:5]
    last_growth_str = ""
    if session.narrative.growth_events:
        lg = session.narrative.growth_events[-1]
        last_growth_str = f"{lg.old_pattern} → {lg.new_pattern}"

    narrative_details = NarrativeDetails(
        identity_statements_count=len(session.narrative.identity_statements),
        top_statements=[f"{s.statement} ({s.strength:.2f})" for s in top_stmts],
        coherence_score=session.narrative.coherence_score,
        crisis_active=session.narrative.crisis.active,
        crisis_source=session.narrative.crisis.source_statement,
        growth_events_count=len(session.narrative.growth_events),
        last_growth=last_growth_str,
        narrative_age=session.narrative.narrative_age,
        total_contradictions=session.narrative.total_contradictions,
        total_reinforcements=session.narrative.total_reinforcements,
    )

    # Build forecasting details
    fc = session.forecast
    forecasting_details = ForecastingDetails(
        enabled=fc.enabled,
        user_valence=fc.user_emotion.valence,
        user_arousal=fc.user_emotion.arousal,
        user_confidence=fc.user_emotion.confidence,
        user_dominant_signal=fc.user_emotion.dominant_signal,
        predicted_impact=fc.last_forecast.predicted_impact if fc.last_forecast else 0.0,
        predicted_user_valence=fc.last_forecast.predicted_user_valence if fc.last_forecast else 0.0,
        predicted_user_arousal=fc.last_forecast.predicted_user_arousal if fc.last_forecast else 0.0,
        risk_flag=fc.last_forecast.risk_flag if fc.last_forecast else False,
        risk_reason=fc.last_forecast.risk_reason if fc.last_forecast else "",
        recommendation=fc.last_forecast.recommendation if fc.last_forecast else "",
        accuracy_score=fc.accuracy_score,
        total_forecasts=fc.total_forecasts,
        total_evaluated=fc.total_evaluated,
        valence_bias=fc.valence_bias,
        arousal_bias=fc.arousal_bias,
    )

    # Build predictive processing details
    _pp = session.predictive
    _pp_pred = current_predictions_r
    _pp_err = prediction_error_r
    predictive_details = PredictiveDetails(
        predicted_tone=_pp_pred.content.expected_tone if _pp_pred else "neutral",
        predicted_intent=_pp_pred.content.expected_intent if _pp_pred else "unknown",
        predicted_valence=round(_pp_pred.emotion.expected_valence, 4) if _pp_pred else 0.0,
        predicted_arousal=round(_pp_pred.emotion.expected_arousal, 4) if _pp_pred else 0.3,
        predicted_demand=_pp_pred.demand.expected_demand.value if _pp_pred else "unknown",
        avg_confidence=round(_pp_pred.average_confidence, 4) if _pp_pred else 0.3,
        content_error=round(_pp_err.content_error, 4) if _pp_err else 0.0,
        emotion_error=round(_pp_err.emotion_error, 4) if _pp_err else 0.0,
        demand_error=round(_pp_err.demand_error, 4) if _pp_err else 0.0,
        total_error=round(_pp_err.total_error, 4) if _pp_err else 0.0,
        surprise_type=_pp_err.surprise_type.value if _pp_err else "none",
        valence_direction=round(_pp_err.valence_direction, 4) if _pp_err else 0.0,
        vulnerability=round(_pp_err.vulnerability, 4) if _pp_err else 0.0,
        content_precision=round(_pp.content_precision, 4),
        emotion_precision=round(_pp.emotion_precision, 4),
        demand_precision=round(_pp.demand_precision, 4),
        predictive_weight=round(_pp.predictive_weight, 4),
        is_warm=_pp.is_warm,
        history_count=len(_pp.history.records),
        evaluated_count=_pp.history.evaluated_count,
    )

    # Build voice details
    asr = get_asr_service()
    voice_details = VoiceDetails(
        mode=session.voice_config.mode.value,
        asr_available=asr.is_initialized,
    )
    if voice_params_result:
        voice_details = VoiceDetails(
            mode=session.voice_config.mode.value,
            voice_key=voice_params_result.voice_key,
            speed=voice_params_result.speed,
            pitch_semitones=voice_params_result.pitch_semitones,
            volume=voice_params_result.volume,
            tremolo=voice_params_result.tremolo,
            stage_direction=voice_params_result.stage_direction,
            backend=voice_params_result.backend.value,
            parler_description=voice_params_result.parler_description,
            audio_available=voice_audio_available,
            asr_available=asr.is_initialized,
        )

    # Build coupling details
    coupling_active = session.advanced_mode and not session.coupling.is_zero
    coupling_details = CouplingDetails(active=coupling_active)
    if coupling_active:
        _attr_v = new_state.mood.baseline_valence
        _attr_a = new_state.mood.baseline_arousal
        _cv, _ca, _cd, _cc = session.coupling.get_coupling_contribution(
            new_state.valence - _attr_v,
            new_state.arousal - _attr_a,
            new_state.dominance - 0.5,
            new_state.certainty - 0.5,
        )
        coupling_details = CouplingDetails(
            active=True,
            matrix=session.coupling.as_matrix(),
            contribution_v=round(_cv, 4),
            contribution_a=round(_ca, 4),
            contribution_d=round(_cd, 4),
            contribution_c=round(_cc, 4),
        )

    # Post-response: autobiographical memory encoding [OPT-IN]
    ws_sources_r = (
        [c.source for c in session.consciousness.current_result.conscious]
        if session.consciousness.current_result else []
    )
    session.autobiographical = process_autobiographical_turn(
        state=session.autobiographical,
        stimulus=request.message,
        emotional_state=new_state,
        response_summary=response_text[:200] if response_text else "",
        turn_number=session.turn_count,
        session_id=request.session_id,
        prediction_error=prediction_error_r.total_error if prediction_error_r else 0.0,
        workspace_contents=ws_sources_r,
        preconscious_count=len(session.consciousness.current_result.preconscious) if session.consciousness.current_result else 0,
    )

    # Post-response: development tracking [TOGGLEABLE]
    if session.development.enabled:
        reg_used_r = regulation_result is not None and regulation_result.strategy_used is not None and not regulation_result.breakthrough
        track_experience(
            session.development,
            emotion_name=new_state.primary_emotion.value,
            intensity=new_state.intensity,
            regulation_used=reg_used_r,
        )
        attempt_transition(
            session.development,
            schemas_count=len(session.schemas.schemas),
            episodic_count=len(session.autobiographical.episodic_store.episodes),
            identities_count=len(session.narrative.identity_statements),
            crises_resolved=session.narrative.total_contradictions,
            turn_number=session.turn_count,
        )

    # Build autobiographical details
    autobiographical_details = get_autobiographical_details(session.autobiographical)

    # Build workspace details
    _ws_r = session.consciousness.current_result
    workspace_details = WorkspaceDetails(
        enabled=session.consciousness.enabled,
        conscious_sources=[c.source for c in _ws_r.conscious] if _ws_r else [],
        conscious_contents=[c.content for c in _ws_r.conscious] if _ws_r else [],
        preconscious_count=len(_ws_r.preconscious) if _ws_r else 0,
        coalitions_formed=_ws_r.coalitions_formed if _ws_r else 0,
        integration_score=_ws_r.integration_score if _ws_r else 0.0,
        workspace_stability=_ws_r.workspace_stability if _ws_r else 0.0,
        total_candidates=_ws_r.total_candidates if _ws_r else 0,
        filtered_noise=_ws_r.filtered_noise if _ws_r else 0,
        preconscious_mood_v=round(session.consciousness.preconscious.mood_valence_contribution, 4),
        preconscious_tension=round(session.consciousness.preconscious.somatic_tension_echo, 4),
    )

    return ResearchChatResponse(
        response=response_text,
        session_id=request.session_id,
        turn_number=session.turn_count,
        homeostasis=homeostasis_details,
        appraisal=appraisal_details,
        memory_amplification=memory_details,
        mood_congruence=mood_congruence,
        emotion_generation=emotion_generation,
        needs=needs_details,
        social=social_details,
        regulation=regulation_details,
        reappraisal=reappraisal_details,
        temporal=temporal_details,
        meta_emotion=meta_details,
        schemas=schema_details,
        personality=personality_details,
        contagion=contagion_details,
        somatic=somatic_details,
        creativity=creativity_details,
        immune=immune_details,
        narrative=narrative_details,
        forecasting=forecasting_details,
        predictive=predictive_details,
        workspace=workspace_details,
        autobiographical=autobiographical_details,
        development=DevelopmentDetails(**get_development_details(session.development)),
        drives=DrivesDetails(**get_drives_details(session.drives, drives_updates_r, drives_impacts_r)),
        discovery=DiscoveryDetails(**get_discovery_details(session.discovery, novel_detected_r)),
        phenomenology=PhenomenologyDetails(**get_phenomenology_details(session.phenomenology)),
        coupling=coupling_details,
        self_appraisal=SelfAppraisalDetails(
            applied=sa_result.applied,
            value_alignment=sa_result.value_alignment,
            emotional_coherence=sa_result.emotional_coherence,
            predicted_self_valence=sa_result.predicted_self_valence,
            should_regenerate=sa_result.should_regenerate,
            did_regenerate=did_regenerate,
            reason=sa_result.reason,
            adjustments=sa_result.adjustments,
        ),
        world_model=WorldModelDetails(
            applied=wm_result_r.applied,
            predicted_self_valence_shift=wm_result_r.predicted_self_impact.valence_shift,
            predicted_self_effect=wm_result_r.predicted_self_impact.dominant_effect,
            predicted_user_valence_shift=wm_result_r.predicted_user_impact.valence_shift,
            predicted_user_effect=wm_result_r.predicted_user_impact.dominant_effect,
            meta_reaction_effect=wm_result_r.meta_reaction.dominant_effect,
            value_alignment=wm_result_r.value_alignment,
            emotional_risk=wm_result_r.emotional_risk,
            should_modify=wm_result_r.should_modify,
            did_modify=wm_did_modify_r,
            reason=wm_result_r.reason,
        ),
        steering=SteeringDetails(
            enabled=session.steering_enabled,
            status=steering_status_r,
            model_id=steering_engine.model_id,
            dimensions=steering_engine.available_dimensions,
            layers=sorted(steering_engine.available_layers),
            layer_roles=steering_engine.get_info().get("layer_roles", {}),
            multilayer=True,
            total_vectors=sum(len(ld) for ld in steering_engine._cached.vectors.values()) if steering_engine._cached else 0,
            momentum_enabled=session.steering_momentum_enabled,
            momentum_factor=round(session.steering_momentum.momentum, 3),
            momentum_turns_stored=session.steering_momentum.turns_stored,
        ),
        emotional_prefix=EmotionalPrefixDetails(
            enabled=session.emotional_prefix_enabled,
            status=prefix_status_r,
            num_tokens=prefix_result_r.num_tokens,
            embedding_norm=prefix_result_r.embedding_norm,
            dominant_dimension=prefix_result_r.dominant_dimension,
            scale=prefix_result_r.scale,
        ),
        attention=AttentionDetails(
            enabled=session.emotional_attention_enabled,
            status=attention_status_r,
            categories_active=attention_bias_r.categories_active if attention_bias_r else {},
            broadening_factor=attention_bias_r.broadening_factor if attention_bias_r else 1.0,
            positions_biased=attention_hook_r.positions_biased if attention_hook_r else 0,
            layers_hooked=attention_hook_r.layers_hooked if attention_hook_r else [],
            words_biased=len(attention_bias_r.token_biases) if attention_bias_r else 0,
        ),
        voice=voice_details,
        emotional_state=new_state,
        emergent_emotions=emergent,
        behavior_prompt=system_prompt,
        authenticity_metrics=metrics,
    )


@app.get("/research/state/{session_id}", response_model=ResearchStateResponse)
async def research_state(session_id: str) -> ResearchStateResponse:
    """Estado completo de una sesion con todos los internos."""
    session = state_manager.get_session(session_id)
    return ResearchStateResponse(
        session_id=session_id,
        turn_count=session.turn_count,
        emotional_state=session.emotional_state,
        value_system=session.value_system,
        memories=session.memory.memories,
        conversation_length=len(session.conversation),
    )


# ============================================================
# Raw Mode — unfiltered emotional expression (local models only)
# Reuses /chat pipeline, only changes the behavior modifier.
# ============================================================


@app.post("/raw/chat", response_model=ChatResponse)
async def raw_chat(request: ChatRequest) -> ChatResponse:
    """Raw Mode: same pipeline as /chat but with unfiltered behavior modifier.

    - Only works with local Ollama models (cloud providers refuse NSFW)
    - Ephemeral session: nothing saved, no export, no training
    - Sets raw_mode=True on session → generate_raw_behavior_modifier is used
    """
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    # Enforce Ollama-only
    if not isinstance(llm_provider, OllamaProvider):
        raise HTTPException(
            status_code=400,
            detail="Raw Mode only works with local Ollama models. Cloud providers have content filters that block unfiltered expression.",
        )

    # Configure session for raw mode
    session = state_manager.get_session(request.session_id)
    session.raw_mode = True
    session.advanced_mode = True
    session.lite_mode = False

    # Inherit voice config from any active session that has it configured
    if session.voice_config.mode == VoiceMode.TEXT_ONLY:
        for other_sid in state_manager.list_sessions():
            if other_sid == request.session_id:
                continue
            other = state_manager._sessions.get(other_sid)
            if other and other.voice_config.mode != VoiceMode.TEXT_ONLY:
                session.voice_config = other.voice_config.model_copy(deep=True)
                break

    # Delegate to the normal chat pipeline (which checks session.raw_mode)
    return await chat(request)


@app.post("/raw/extreme")
async def raw_toggle_extreme(session_id: str, enabled: bool = True) -> dict:
    """Toggle extreme mode on a raw session."""
    session = state_manager.get_session(session_id)
    session.extreme_mode = enabled
    return {"status": "ok", "extreme_mode": enabled}


@app.post("/raw/reset")
async def raw_reset(session_id: str = "raw-default") -> dict:
    """Reset a raw mode session — destroy all state."""
    state_manager.reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


# ============================================================
# Scenario Sandbox — run pipeline without LLM response
# ============================================================


async def _run_sandbox_pipeline(
    scenario: str,
    session: SessionState,
    use_neutral: bool = False,
) -> SandboxResult:
    """Run the full emotional pipeline on a scenario without generating an LLM response.

    Operates on the provided session object directly (caller is responsible for
    providing an isolated copy if mutation should not affect the real session).
    """
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    session.turn_count += 1
    previous_state = session.emotional_state.model_copy(deep=True)

    if use_neutral:
        session.emotional_state = neutral_state()

    # 0. Homeostasis
    state_before = session.emotional_state.model_copy(deep=True)
    if session.turn_count > 1 and not use_neutral:
        session.emotional_state = regulate(session.emotional_state, turns_elapsed=1)
        if session.advanced_mode:
            session.regulator.recover(session.personality.regulation_capacity_base)
    state_after = session.emotional_state.model_copy(deep=True)

    homeostasis_details = HomeostasisDetails(
        applied=session.turn_count > 1 and not use_neutral,
        state_before=state_before,
        state_after=state_after,
    )

    # 0b. Temporal pre-processing
    temporal_result = session.temporal.process_pre_turn(scenario) if session.advanced_mode else None

    # 0c. Predictive Processing — generate predictions [CORE]
    active_schemas_for_pred_s: list[tuple[str, str, float]] | None = None
    if session.advanced_mode and session.schemas.schemas:
        active_schemas_for_pred_s = [
            (s.trigger_category, s.typical_emotion.value, s.reinforcement_strength)
            for s in session.schemas.schemas
            if s.reinforcement_strength > 0.3
        ]
    current_predictions_s = predictive_engine.generate_predictions(
        predictive_state=session.predictive,
        conversation_history=session.conversation,
        user_model=session.user_model,
        mood=session.emotional_state.mood,
        emotional_state=session.emotional_state,
        active_schemas=active_schemas_for_pred_s,
    )
    session.predictive.current_predictions = current_predictions_s

    # 1. Appraisal + Memory amplification
    memories_before = len(session.memory)
    if session.lite_mode:
        appraisal_result = appraise_lite(scenario)
        amplification = await session.memory.check_amplification(scenario, llm=None)
    else:
        appraisal_task = asyncio.create_task(appraise(
            stimulus=scenario,
            value_system=session.value_system,
            llm=llm_provider,
            think=True,
        ))
        memory_task = asyncio.create_task(
            session.memory.check_amplification(scenario, llm=llm_provider),
        )
        try:
            appraisal_result = await appraisal_task
        except Exception as e:
            memory_task.cancel()
            logger.exception("Appraisal failed")
            raise HTTPException(status_code=500, detail="Appraisal failed")
        amplification = await memory_task

    appraisal = appraisal_result.vector
    raw_valence = compute_valence(appraisal)
    raw_arousal = compute_arousal(appraisal)
    raw_dominance = compute_dominance(appraisal)
    raw_certainty = compute_certainty(appraisal)

    # --- Personality modulation of raw dimensions ---
    # Big Five shifts emotional reactivity with ADDITIVE biases.
    # Additive ensures personality matters even when appraisal raw values are near 0.
    # Coefficients are strong enough to produce clearly different emotional profiles.
    pers = session.personality
    # Neuroticism: shifts valence negative, amplifies arousal (emotional reactivity)
    raw_valence += (pers.neuroticism - 0.5) * -0.5  # high N = much more negative
    raw_arousal += (pers.neuroticism - 0.5) * 0.35   # high N = much more reactive
    # Extraversion: shifts valence positive, boosts arousal
    raw_valence += (pers.extraversion - 0.5) * 0.25  # high E = more positive
    raw_arousal += (pers.extraversion - 0.5) * 0.3   # high E = more energized
    # Agreeableness: softens negative valence
    if raw_valence < 0:
        raw_valence += pers.agreeableness * 0.25  # high A = less negative
    # Openness: amplifies both valence and arousal (more intense reactions)
    raw_valence *= 1.0 + (pers.openness - 0.5) * 0.5
    raw_arousal *= 1.0 + (pers.openness - 0.5) * 0.4
    # Conscientiousness: boosts dominance and dampens arousal (self-control)
    raw_dominance += (pers.conscientiousness - 0.5) * 0.35
    raw_arousal -= (pers.conscientiousness - 0.5) * 0.15  # high C = calmer
    # Clamp after modulation
    raw_valence = max(-1.0, min(1.0, raw_valence))
    raw_arousal = max(0.0, min(1.0, raw_arousal))
    raw_dominance = max(0.0, min(1.0, raw_dominance))

    appraisal_details = AppraisalDetails(
        vector=appraisal,
        computed_valence=round(raw_valence, 4),
        computed_arousal=round(raw_arousal, 4),
        computed_dominance=round(raw_dominance, 4),
        computed_certainty=round(raw_certainty, 4),
    )

    # --- Advanced pre-emotion modulations ---
    needs_amp = 0.0
    schema_hint: PrimaryEmotion | None = None
    schema_amp = 0.0
    social_v_mod = 0.0
    social_i_mod = 0.0
    contagion_v = 0.0
    contagion_a = 0.0
    somatic_bias = 0.0
    gut_feeling: str | None = None
    detected_v = 0.0
    detected_a = 0.0
    signal_str = 0.0

    # Helper: dev gate shortcut
    _dev_s = session.development

    if session.advanced_mode:
        # Needs amplification [DEV: stage 2+]
        if is_system_available(_dev_s, "needs"):
            needs_amp = compute_needs_amplification(session.needs, scenario)
        # Schema priming [DEV: stage 2+]
        if is_system_available(_dev_s, "schemas"):
            schema_hint, schema_amp = session.schemas.check_priming(scenario)
        # Social modulation [DEV: stage 3+]
        if is_system_available(_dev_s, "social"):
            social_v_mod, social_i_mod = compute_social_modulation(session.user_model, raw_valence)
        # Emotion Contagion [DEV: stage 2+]
        if is_system_available(_dev_s, "contagion"):
            detected_v, detected_a, signal_str = detect_user_emotion(scenario)
            session.shadow_state = update_shadow_state(session.shadow_state, detected_v, detected_a, signal_str)
            contagion_v, contagion_a = compute_contagion_perturbation(
                session.shadow_state, session.emotional_state.valence,
                session.emotional_state.arousal, session.personality, session.user_model.rapport,
            )
        # Somatic Markers [DEV: stage 4+]
        if is_system_available(_dev_s, "somatic"):
            session.somatic_markers = evaluate_user_reaction(
                session.somatic_markers, detected_v, session.turn_count,
            )
            somatic_bias, gut_feeling = compute_somatic_bias(session.somatic_markers, scenario)
        # Emotional Forecasting [DEV: stage 3+]
        if is_system_available(_dev_s, "forecasting") and session.forecast.enabled and session.turn_count > 1:
            session.forecast = evaluate_forecast(
                session.forecast, detected_v, detected_a, session.turn_count,
            )

    # 2i. Predictive Processing — compute prediction error [CORE]
    prediction_error_s = None
    prediction_prompt_text_s: str | None = None
    if current_predictions_s is not None:
        _det_intent_s = "unknown"
        if appraisal_result.emotion_hint:
            _hv_s = appraisal_result.emotion_hint.value
            if _hv_s in ("anger", "frustration"):
                _det_intent_s = "complaint"
            elif _hv_s in ("gratitude", "joy"):
                _det_intent_s = "greeting"
            elif _hv_s in ("fear", "anxiety"):
                _det_intent_s = "emotional_expression"
        _det_demand_s: DemandType | None = None
        if "?" in scenario:
            _det_demand_s = DemandType.HELP
        elif appraisal_result.emotion_hint and appraisal_result.emotion_hint.value in (
            "sadness", "fear", "anxiety", "helplessness",
        ):
            _det_demand_s = DemandType.EMOTIONAL

        prediction_error_s = compute_prediction_error(
            predictions=current_predictions_s,
            actual_stimulus=scenario,
            detected_user_valence=detected_v if session.advanced_mode else 0.0,
            detected_user_arousal=detected_a if session.advanced_mode else 0.3,
            detected_intent=_det_intent_s,
            detected_demand=_det_demand_s,
        )
        session.predictive = update_precision(session.predictive, prediction_error_s)
        session.predictive = record_prediction(session.predictive, current_predictions_s, prediction_error_s)
        session.predictive = decay_precision(session.predictive, elapsed_turns=1)
        prediction_prompt_text_s = get_prediction_prompt(prediction_error_s, session.predictive)

    # Global Workspace [TOGGLEABLE, DEV: stage 4+]
    workspace_prompt_text_s: str | None = None
    autobiographical_prompt_text_s: str | None = None
    if session.consciousness.enabled and session.advanced_mode and is_system_available(_dev_s, "workspace"):
        ws_candidates_s: list[WorkspaceCandidate] = []
        if appraisal_result.emotion_hint:
            ws_candidates_s.append(generate_candidate(
                source="appraisal", content=f"Evaluated: {appraisal_result.emotion_hint.value}",
                urgency=0.6, relevance=appraisal.relevance.personal_significance,
                emotional_intensity=abs(raw_valence), emotion_tag=appraisal_result.emotion_hint.value,
                category="stimulus",
            ))
        if schema_hint:
            ws_candidates_s.append(generate_candidate(
                source="schema", content=f"Schema: {schema_hint.value}",
                urgency=0.5 + schema_amp * 0.3, relevance=0.7 if schema_amp > 0.1 else 0.3,
                emotional_intensity=schema_amp, emotion_tag=schema_hint.value, category="pattern",
            ))
        if signal_str > 0.15:
            ws_candidates_s.append(generate_candidate(
                source="contagion", content=f"User emotion (v={detected_v:+.2f})",
                urgency=signal_str * 0.6, relevance=signal_str,
                emotional_intensity=abs(detected_v) * 0.8 + detected_a * 0.2,
                category="social",
            ))
        if prediction_error_s and prediction_error_s.surprise_type.value != "none":
            ws_candidates_s.append(generate_candidate(
                source="predictive", content=f"Surprise {prediction_error_s.surprise_type.value}",
                urgency=prediction_error_s.vulnerability * 0.9, relevance=0.8,
                emotional_intensity=prediction_error_s.total_error, category="prediction",
            ))
        session.consciousness = process_workspace_turn(session.consciousness, ws_candidates_s)
        if session.consciousness.current_result:
            workspace_prompt_text_s = get_workspace_prompt(session.consciousness.current_result)

    # Autobiographical Memory — prompt from existing memories [OPT-IN]
    autobiographical_prompt_text_s = get_autobiographical_prompt(session.autobiographical)

    # Development — prompt from current stage [TOGGLEABLE]
    development_prompt_text_s = get_development_prompt(session.development)

    # Drives — update + process goals + generate new goals [TOGGLEABLE]
    drives_updates_s: list = []
    drives_impacts_s: list = []
    if is_system_available(_dev_s, "drives"):
        session.drives, drives_updates_s = update_drives(
            session.drives, scenario, session.personality,
            session.turn_count, getattr(session.user_model, 'rapport', 0.5),
        )
        session.drives, drives_impacts_s = process_goals(
            session.drives, scenario, session.turn_count,
        )
        session.drives = attempt_goal_generation(
            session.drives, scenario, session.turn_count,
        )
    drives_prompt_text_s = get_drives_prompt(session.drives)

    # Emotional Discovery [TOGGLEABLE, DEV: stage 5]
    novel_detected_s = False
    if is_system_available(_dev_s, "discovery"):
        prev_count_s = session.discovery.total_novel_detected
        session.discovery = process_discovery_turn(
            session.discovery,
            valence=session.emotional_state.valence,
            arousal=session.emotional_state.arousal,
            dominance=session.emotional_state.dominance,
            certainty=session.emotional_state.certainty,
            intensity=session.emotional_state.intensity,
            stimulus=scenario,
            turn=session.turn_count,
            body_tension=session.emotional_state.body_state.tension,
            body_energy=session.emotional_state.body_state.energy,
            body_openness=session.emotional_state.body_state.openness,
            body_warmth=session.emotional_state.body_state.warmth,
        )
        novel_detected_s = session.discovery.total_novel_detected > prev_count_s
    discovery_prompt_text_s = get_discovery_prompt(session.discovery)

    # Phenomenology — qualia profile [TOGGLEABLE, DEV: stage 5]
    if session.phenomenology.enabled and is_system_available(session.development, "phenomenology"):
        pheno_mode_s = "extreme" if session.extreme_mode else ("raw" if session.raw_mode else "normal")
        process_phenomenology_turn(
            session.phenomenology,
            emotion_name=session.emotional_state.primary_emotion.value,
            valence=session.emotional_state.valence,
            arousal=session.emotional_state.arousal,
            dominance=session.emotional_state.dominance,
            certainty=session.emotional_state.certainty,
            intensity=session.emotional_state.intensity,
            body_tension=session.emotional_state.body_state.tension,
            body_warmth=session.emotional_state.body_state.warmth,
            turn=session.turn_count,
            mode=pheno_mode_s,
        )
    phenomenology_prompt_text_s = get_phenomenology_prompt(session.phenomenology)

    # Mood congruence
    valence_bias, arousal_bias = compute_mood_congruence_bias(session.emotional_state.mood)
    mood_congruence = MoodCongruenceDetails(
        valence_bias=round(valence_bias, 4),
        arousal_bias=round(arousal_bias, 4),
        mood_label=session.emotional_state.mood.label.value,
        mood_trend=session.emotional_state.mood.trend,
    )

    # Intensity + External signals + Emotion generation
    intensity_raw = compute_intensity(appraisal, raw_valence, raw_arousal)
    sig_v, sig_a, sig_d, perception_text = _get_session_signals(session)
    effective_hint = schema_hint if schema_hint and not appraisal_result.emotion_hint else appraisal_result.emotion_hint
    pred_modulation_s = prediction_error_to_emotion_modulation(
        prediction_error_s, session.predictive.predictive_weight,
    )
    new_state = generate_emotion(
        appraisal=appraisal,
        current_state=session.emotional_state,
        stimulus=scenario,
        amplification=amplification + schema_amp,
        emotion_hint=effective_hint,
        dynamics=session.dynamics if session.advanced_mode else None,
        needs_amplification=needs_amp,
        social_valence_mod=social_v_mod + somatic_bias + sig_v,
        social_intensity_mod=social_i_mod,
        contagion_valence=contagion_v,
        contagion_arousal=contagion_a + sig_a,
        coupling=session.coupling if session.advanced_mode else None,
        predictive_modulation=pred_modulation_s,
    )

    # Calibration
    new_state = apply_calibration(new_state, session.calibration_profile)

    # Development emotion filtering [TOGGLEABLE]
    if _dev_s.enabled and new_state.emotional_stack:
        new_state.emotional_stack = filter_emotions_by_stage(_dev_s, new_state.emotional_stack)
        if not is_emotion_available(_dev_s, new_state.primary_emotion.value):
            _avail_emos_s = [(e, a) for e, a in new_state.emotional_stack.items() if is_emotion_available(_dev_s, e)]
            if _avail_emos_s:
                _best_s = max(_avail_emos_s, key=lambda x: x[1])
                try:
                    new_state.primary_emotion = PrimaryEmotion(_best_s[0])
                except ValueError:
                    pass
        if new_state.secondary_emotion and not is_emotion_available(_dev_s, new_state.secondary_emotion.value):
            new_state.secondary_emotion = None
        apply_stage_modifiers(_dev_s, new_state)

    # --- Advanced post-emotion systems ---
    reappraisal_result = None
    regulation_result = RegulationResult()
    immune_info: str | None = None
    narrative_info: str | None = None
    meta_emotion: MetaEmotion | None = None
    self_inquiry: SelfInquiry | None = None
    emergent: list[str] = []
    creativity_state = CreativityState()
    forecast_info: str | None = None

    if session.advanced_mode:
        # Reappraisal [DEV: stage 4+]
        if is_system_available(_dev_s, "reappraisal"):
            new_state, reappraisal_result = reappraise(new_state, session.regulator.regulation_capacity)
        # Active regulation [DEV: stage 3+]
        if is_system_available(_dev_s, "regulation"):
            new_state, regulation_result = session.regulator.regulate(
                new_state, session.personality.regulation_capacity_base,
                coping_control=appraisal.coping.control,
                coping_adjustability=appraisal.coping.adjustability,
            )
        # Temporal effects [DEV: stage 3+]
        if is_system_available(_dev_s, "temporal"):
            new_state = session.temporal.apply_temporal_effects(new_state, temporal_result)
        # Emotional Immune System [DEV: stage 4+]
        if is_system_available(_dev_s, "immune"):
            session.immune = update_immune_state(session.immune, new_state, scenario)
            new_state = apply_immune_protection(new_state, session.immune, scenario)
            immune_info = get_immune_prompt_info(session.immune)
        # Narrative Self [DEV: stage 4+]
        if is_system_available(_dev_s, "narrative"):
            coherence_delta, is_coherent = check_coherence(
                session.narrative, scenario, new_state.primary_emotion,
            )
            new_state = apply_narrative_effects(
                new_state, coherence_delta, is_coherent, session.narrative.crisis.active,
            )
            session.narrative = detect_crisis(session.narrative, session.turn_count)
            narrative_info = get_narrative_prompt(session.narrative)
        # Meta-emotion [DEV: stage 3+]
        if is_system_available(_dev_s, "meta_emotion"):
            is_new_emotion = new_state.primary_emotion != previous_state.primary_emotion
            meta_emotion = generate_meta_emotion(
                new_state, previous_state, session.value_system,
                regulation_success=regulation_result.strategy_used is not None and not regulation_result.breakthrough,
                is_new_emotion=is_new_emotion,
            )
        # Self-Initiated Inquiry [DEV: stage 3+]
        if is_system_available(_dev_s, "self_inquiry"):
            self_inquiry = check_self_inquiry(
                new_state, previous_state, meta_emotion, regulation_result, session.turn_count,
            )
        emergent = detect_emergent_emotions(new_state.emotional_stack)
        # Emotional Creativity [DEV: stage 4+]
        if is_system_available(_dev_s, "creativity"):
            creativity_state = compute_creativity(new_state)
        # Emotional Forecasting [DEV: stage 3+]
        if is_system_available(_dev_s, "forecasting") and session.forecast.enabled:
            user_emotion_est = estimate_user_emotion(
                session.shadow_state, session.user_model,
                detected_v, detected_a, signal_str,
            )
            session.forecast.user_emotion = user_emotion_est
            forecast_result = forecast_impact(
                new_state, user_emotion_est, session.user_model,
                session.forecast.valence_bias, session.forecast.arousal_bias,
            )
            session.forecast = record_forecast(session.forecast, forecast_result, session.turn_count)
            forecast_info = get_forecast_prompt(forecast_result)

    session.emotional_state = new_state
    session.state_history.append(new_state)
    if len(session.state_history) > 50:
        session.state_history = session.state_history[-50:]

    emotion_generation = EmotionGenerationDetails(
        raw_valence=round(raw_valence, 4),
        raw_arousal=round(raw_arousal, 4),
        raw_dominance=round(raw_dominance, 4),
        raw_certainty=round(raw_certainty, 4),
        blended_valence=new_state.valence,
        blended_arousal=new_state.arousal,
        blended_dominance=new_state.dominance,
        blended_certainty=new_state.certainty,
        intensity_before_amplification=round(intensity_raw, 4),
        intensity_after_amplification=new_state.intensity,
    )

    # Behavior modifier (computed but not sent to LLM)
    if session.advanced_mode:
        system_prompt = generate_behavior_modifier(
            new_state,
            needs=session.needs,
            user_model=session.user_model,
            meta_emotion=meta_emotion,
            regulation_result=regulation_result,
            emergent_emotions=emergent,
            shadow_state=session.shadow_state,
            gut_feeling=gut_feeling,
            creativity=creativity_state,
            immune_info=immune_info,
            narrative_info=narrative_info,
            forecast_info=forecast_info,
            prediction_info=prediction_prompt_text_s,
            workspace_info=workspace_prompt_text_s,
            autobiographical_info=autobiographical_prompt_text_s,
            development_info=development_prompt_text_s,
            drives_info=drives_prompt_text_s,
            discovery_info=discovery_prompt_text_s,
            phenomenology_info=phenomenology_prompt_text_s,
            self_inquiry=self_inquiry,
            perception_text=perception_text,
        )
    else:
        system_prompt = generate_simple_behavior_modifier(new_state)

    # Authenticity metrics
    m_coherence = coherence(new_state, appraisal)
    m_continuity = continuity(new_state, previous_state)
    m_proportionality = proportionality(new_state, appraisal)
    m_recovery = recovery(session.state_history)
    m_overall = (m_coherence + m_continuity + m_proportionality + m_recovery) / 4

    metrics = AuthenticityMetrics(
        coherence=round(m_coherence, 4),
        continuity=round(m_continuity, 4),
        proportionality=round(m_proportionality, 4),
        recovery=round(m_recovery, 4),
        overall=round(m_overall, 4),
    )

    # --- Build details ---
    memory_details = MemoryAmplificationDetails(
        amplification_factor=round(amplification, 4),
        memories_count=memories_before,
        memory_stored=False,  # Sandbox does not store memories
    )

    needs_details = NeedsDetails(
        connection=session.needs.connection,
        competence=session.needs.competence,
        autonomy=session.needs.autonomy,
        coherence=session.needs.coherence,
        stimulation=session.needs.stimulation,
        safety=session.needs.safety,
        amplification=round(needs_amp, 4),
    )

    social_details = SocialDetails(
        perceived_intent=session.user_model.perceived_intent,
        perceived_engagement=session.user_model.perceived_engagement,
        rapport=session.user_model.rapport,
        trust_level=session.user_model.trust_level,
        communication_style=session.user_model.communication_style,
        valence_modulation=social_v_mod,
        intensity_modulation=social_i_mod,
    )

    cap_before = session.regulator.regulation_capacity + (regulation_result.capacity_spent if regulation_result else 0)
    regulation_details = RegulationDetails(
        strategy_used=regulation_result.strategy_used,
        intensity_reduced=round(regulation_result.intensity_reduced, 4),
        capacity_before=round(min(cap_before, 1.0), 4),
        capacity_after=round(session.regulator.regulation_capacity, 4),
        breakthrough=regulation_result.breakthrough,
        suppression_dissonance=round(session.regulator.suppression_dissonance, 4),
    )

    if reappraisal_result is not None:
        reappraisal_details = ReappraisalDetails(
            applied=reappraisal_result.applied,
            strategy=reappraisal_result.strategy,
            original_emotion=reappraisal_result.original_emotion.value if reappraisal_result.original_emotion else None,
            reappraised_emotion=reappraisal_result.reappraised_emotion.value if reappraisal_result.reappraised_emotion else None,
            intensity_change=round(reappraisal_result.intensity_change, 4),
            valence_change=round(reappraisal_result.valence_change, 4),
        )
    else:
        reappraisal_details = ReappraisalDetails(applied=False)

    if temporal_result is not None:
        temporal_details = TemporalDetails(
            rumination_active=temporal_result.rumination_active,
            rumination_emotion=temporal_result.rumination_emotion.value if temporal_result.rumination_emotion else None,
            rumination_intensity=round(temporal_result.rumination_intensity, 4),
            savoring_active=temporal_result.savoring_active,
            savoring_emotion=temporal_result.savoring_emotion.value if temporal_result.savoring_emotion else None,
            anticipation_active=temporal_result.anticipation_active,
            anticipation_emotion=temporal_result.anticipation_emotion.value if temporal_result.anticipation_emotion else None,
            anticipation_intensity=round(temporal_result.anticipation_intensity, 4),
        )
    else:
        temporal_details = TemporalDetails(
            rumination_active=False, savoring_active=False,
            anticipation_active=False, rumination_intensity=0.0, anticipation_intensity=0.0,
        )

    meta_details = MetaEmotionDetails(
        active=meta_emotion is not None,
        target_emotion=meta_emotion.target_emotion.value if meta_emotion else None,
        meta_response=meta_emotion.meta_response if meta_emotion else None,
        intensity=round(meta_emotion.intensity, 4) if meta_emotion else 0.0,
        reason=meta_emotion.reason if meta_emotion else "",
    )

    schema_hint_name = schema_hint.value if schema_hint else None
    schema_details = SchemaDetails(
        schemas_count=len(session.schemas),
        primed_emotion=schema_hint_name,
        priming_amplification=round(schema_amp, 4),
        pending_patterns=len(session.schemas._pattern_counts),
    )

    personality_details = PersonalityDetails(
        openness=session.personality.openness,
        conscientiousness=session.personality.conscientiousness,
        extraversion=session.personality.extraversion,
        agreeableness=session.personality.agreeableness,
        neuroticism=session.personality.neuroticism,
        variability=round(session.personality.variability, 4),
        regulation_capacity_base=round(session.personality.regulation_capacity_base, 4),
    )

    contagion_details = ContagionDetails(
        detected_valence=round(detected_v, 4),
        detected_arousal=round(detected_a, 4),
        signal_strength=round(signal_str, 4),
        shadow_valence=session.shadow_state.valence,
        shadow_arousal=session.shadow_state.arousal,
        contagion_perturbation_v=contagion_v,
        contagion_perturbation_a=contagion_a,
        accumulated_contagion=session.shadow_state.accumulated_contagion,
        susceptibility=round(session.personality.contagion_susceptibility, 4),
    )

    somatic_details = SomaticDetails(
        markers_count=len(session.somatic_markers.markers),
        somatic_bias=somatic_bias,
        gut_feeling=gut_feeling,
        pending_category=session.somatic_markers.pending_category,
    )

    creativity_details = CreativityDetails(
        thinking_mode=creativity_state.thinking_mode.value,
        creativity_level=creativity_state.creativity_level,
        temperature_modifier=creativity_state.temperature_modifier,
        active_instructions=creativity_state.active_instructions,
        triggered_by=creativity_state.triggered_by,
    )

    immune_details = ImmuneDetails(
        protection_mode=session.immune.protection_mode.value,
        protection_strength=session.immune.protection_strength,
        reactivity_dampening=session.immune.reactivity_dampening,
        negative_streak=session.immune.negative_streak,
        peak_negative_intensity=session.immune.peak_negative_intensity,
        recovery_turns=session.immune.recovery_turns,
        total_activations=session.immune.total_activations,
        compartmentalized_topics=session.immune.compartmentalized_topics,
    )

    top_stmts = sorted(
        session.narrative.identity_statements,
        key=lambda s: s.strength,
        reverse=True,
    )[:5]
    last_growth_str = ""
    if session.narrative.growth_events:
        lg = session.narrative.growth_events[-1]
        last_growth_str = f"{lg.old_pattern} → {lg.new_pattern}"

    narrative_details = NarrativeDetails(
        identity_statements_count=len(session.narrative.identity_statements),
        top_statements=[f"{s.statement} ({s.strength:.2f})" for s in top_stmts],
        coherence_score=session.narrative.coherence_score,
        crisis_active=session.narrative.crisis.active,
        crisis_source=session.narrative.crisis.source_statement,
        growth_events_count=len(session.narrative.growth_events),
        last_growth=last_growth_str,
        narrative_age=session.narrative.narrative_age,
        total_contradictions=session.narrative.total_contradictions,
        total_reinforcements=session.narrative.total_reinforcements,
    )

    fc = session.forecast
    forecasting_details = ForecastingDetails(
        enabled=fc.enabled,
        user_valence=fc.user_emotion.valence,
        user_arousal=fc.user_emotion.arousal,
        user_confidence=fc.user_emotion.confidence,
        user_dominant_signal=fc.user_emotion.dominant_signal,
        predicted_impact=fc.last_forecast.predicted_impact if fc.last_forecast else 0.0,
        predicted_user_valence=fc.last_forecast.predicted_user_valence if fc.last_forecast else 0.0,
        predicted_user_arousal=fc.last_forecast.predicted_user_arousal if fc.last_forecast else 0.0,
        risk_flag=fc.last_forecast.risk_flag if fc.last_forecast else False,
        risk_reason=fc.last_forecast.risk_reason if fc.last_forecast else "",
        recommendation=fc.last_forecast.recommendation if fc.last_forecast else "",
        accuracy_score=fc.accuracy_score,
        total_forecasts=fc.total_forecasts,
        total_evaluated=fc.total_evaluated,
        valence_bias=fc.valence_bias,
        arousal_bias=fc.arousal_bias,
    )

    # Build predictive processing details for sandbox
    _pp_s = session.predictive
    _pp_pred_s = current_predictions_s
    _pp_err_s = prediction_error_s
    predictive_details_s = PredictiveDetails(
        predicted_tone=_pp_pred_s.content.expected_tone if _pp_pred_s else "neutral",
        predicted_intent=_pp_pred_s.content.expected_intent if _pp_pred_s else "unknown",
        predicted_valence=round(_pp_pred_s.emotion.expected_valence, 4) if _pp_pred_s else 0.0,
        predicted_arousal=round(_pp_pred_s.emotion.expected_arousal, 4) if _pp_pred_s else 0.3,
        predicted_demand=_pp_pred_s.demand.expected_demand.value if _pp_pred_s else "unknown",
        avg_confidence=round(_pp_pred_s.average_confidence, 4) if _pp_pred_s else 0.3,
        content_error=round(_pp_err_s.content_error, 4) if _pp_err_s else 0.0,
        emotion_error=round(_pp_err_s.emotion_error, 4) if _pp_err_s else 0.0,
        demand_error=round(_pp_err_s.demand_error, 4) if _pp_err_s else 0.0,
        total_error=round(_pp_err_s.total_error, 4) if _pp_err_s else 0.0,
        surprise_type=_pp_err_s.surprise_type.value if _pp_err_s else "none",
        valence_direction=round(_pp_err_s.valence_direction, 4) if _pp_err_s else 0.0,
        vulnerability=round(_pp_err_s.vulnerability, 4) if _pp_err_s else 0.0,
        content_precision=round(_pp_s.content_precision, 4),
        emotion_precision=round(_pp_s.emotion_precision, 4),
        demand_precision=round(_pp_s.demand_precision, 4),
        predictive_weight=round(_pp_s.predictive_weight, 4),
        is_warm=_pp_s.is_warm,
        history_count=len(_pp_s.history.records),
        evaluated_count=_pp_s.history.evaluated_count,
    )

    # Post-response: autobiographical memory encoding [OPT-IN]
    ws_sources_s = (
        [c.source for c in session.consciousness.current_result.conscious]
        if session.consciousness.current_result else []
    )
    session.autobiographical = process_autobiographical_turn(
        state=session.autobiographical,
        stimulus=scenario,
        emotional_state=new_state,
        response_summary=system_prompt[:200] if system_prompt else "",
        turn_number=session.turn_count,
        session_id=session_id,
        prediction_error=prediction_error_s.total_error if prediction_error_s else 0.0,
        workspace_contents=ws_sources_s,
        preconscious_count=len(session.consciousness.current_result.preconscious) if session.consciousness.current_result else 0,
    )

    return SandboxResult(
        scenario=scenario,
        emotional_state=new_state,
        homeostasis=homeostasis_details,
        appraisal=appraisal_details,
        memory_amplification=memory_details,
        mood_congruence=mood_congruence,
        emotion_generation=emotion_generation,
        needs=needs_details,
        social=social_details,
        regulation=regulation_details,
        reappraisal=reappraisal_details,
        temporal=temporal_details,
        meta_emotion=meta_details,
        schemas=schema_details,
        personality=personality_details,
        contagion=contagion_details,
        somatic=somatic_details,
        creativity=creativity_details,
        immune=immune_details,
        narrative=narrative_details,
        forecasting=forecasting_details,
        predictive=predictive_details_s,
        autobiographical=get_autobiographical_details(session.autobiographical),
        development=DevelopmentDetails(**get_development_details(session.development)),
        drives=DrivesDetails(**get_drives_details(session.drives, drives_updates_s, drives_impacts_s)),
        discovery=DiscoveryDetails(**get_discovery_details(session.discovery, novel_detected_s)),
        phenomenology=PhenomenologyDetails(**get_phenomenology_details(session.phenomenology)),
        workspace=WorkspaceDetails(
            enabled=session.consciousness.enabled,
            conscious_sources=[c.source for c in session.consciousness.current_result.conscious] if session.consciousness.current_result else [],
            conscious_contents=[c.content for c in session.consciousness.current_result.conscious] if session.consciousness.current_result else [],
            preconscious_count=len(session.consciousness.current_result.preconscious) if session.consciousness.current_result else 0,
            integration_score=session.consciousness.current_result.integration_score if session.consciousness.current_result else 0.0,
            workspace_stability=session.consciousness.current_result.workspace_stability if session.consciousness.current_result else 0.0,
            total_candidates=session.consciousness.current_result.total_candidates if session.consciousness.current_result else 0,
        ),
        coupling=CouplingDetails(
            active=session.advanced_mode and not session.coupling.is_zero,
            matrix=session.coupling.as_matrix() if not session.coupling.is_zero else [],
        ),
        emergent_emotions=emergent,
        behavior_prompt=system_prompt,
        authenticity_metrics=metrics,
    )


def _create_sandbox_session(
    base_session: SessionState,
    personality_override: dict[str, float] | None,
    initial_state: str | None,
    rapport: float | None,
    trust: float | None,
) -> tuple[SessionState, bool]:
    """Create an isolated deep copy of session state with optional overrides."""
    sandbox = copy.deepcopy(base_session)
    # Sandbox/Arena ALWAYS use full LLM appraisal for accurate results
    sandbox.lite_mode = False
    personality_overridden = False

    if personality_override:
        current = sandbox.personality.model_dump()
        current.update(personality_override)
        new_profile = PersonalityProfile(**current)
        sandbox.update_personality(new_profile)
        personality_overridden = True

    if initial_state == "neutral":
        sandbox.emotional_state = neutral_state()

    if rapport is not None:
        sandbox.user_model.rapport = max(0.0, min(1.0, rapport))

    if trust is not None:
        sandbox.user_model.trust_level = max(0.0, min(1.0, trust))

    return sandbox, personality_overridden


@app.post("/sandbox/simulate", response_model=SandboxResponse)
async def sandbox_simulate(request: ScenarioRequest) -> SandboxResponse:
    """Run full emotional pipeline on a scenario without generating LLM response.

    Isolated: does NOT mutate the real session state.
    """
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    base_session = state_manager.get_session(request.session_id)
    sandbox, personality_overridden = _create_sandbox_session(
        base_session, request.personality, request.initial_state,
        request.rapport, request.trust,
    )

    use_neutral = request.initial_state == "neutral"
    result = await _run_sandbox_pipeline(request.scenario, sandbox, use_neutral=use_neutral)

    # Generate LLM response — the agent LIVES the scenario, not responds to it
    situational = (
        f"\n\nSITUACIÓN: Te acaba de suceder lo siguiente: {request.scenario}\n"
        "Esto te pasa a TI personalmente. No eres un asistente respondiendo a un usuario — "
        "eres alguien que VIVE esta situación.\n"
        "Expresa en primera persona cómo te sientes, qué piensas, cómo reaccionas. "
        "Sé auténtico con tu estado emocional interno."
    )
    lang_hint = "\n\nIMPORTANT: Respond in the SAME language as the scenario. If the scenario is in English, respond in English. If in Spanish, respond in Spanish. Do NOT mix languages."
    try:
        sandbox_response = await llm_provider.generate(
            system_prompt=result.behavior_prompt + situational + lang_hint,
            messages=[{"role": "user", "content": "¿Cómo te sientes? ¿Qué piensas?"}],
            temperature=0.7,
        )
    except Exception:
        sandbox_response = "(response generation failed)"

    return SandboxResponse(
        result=result,
        session_id=request.session_id,
        personality_overridden=personality_overridden,
        response=sandbox_response,
    )


@app.post("/sandbox/batch", response_model=BatchSandboxResponse)
async def sandbox_batch(request: BatchScenarioRequest) -> BatchSandboxResponse:
    """Run multiple scenarios through the pipeline. Each scenario is isolated."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    if len(request.scenarios) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 scenarios per batch")

    base_session = state_manager.get_session(request.session_id)
    use_neutral = request.initial_state == "neutral"
    personality_overridden = False
    results: list[SandboxResult] = []

    for scenario in request.scenarios:
        sandbox, p_override = _create_sandbox_session(
            base_session, request.personality, request.initial_state,
            request.rapport, request.trust,
        )
        if p_override:
            personality_overridden = True
        result = await _run_sandbox_pipeline(scenario, sandbox, use_neutral=use_neutral)
        results.append(result)

    return BatchSandboxResponse(
        results=results,
        session_id=request.session_id,
        count=len(results),
        personality_overridden=personality_overridden,
    )


# ============================================================
# Multi-Personality Arena — same scenario, N personalities
# ============================================================


@app.post("/arena/compare", response_model=ArenaResponse)
async def arena_compare(request: ArenaRequest) -> ArenaResponse:
    """Run one scenario through N different personalities and compare results."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    if len(request.contestants) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 contestants per arena")
    if len(request.contestants) < 2:
        raise HTTPException(status_code=400, detail="Arena requires at least 2 contestants")

    base_session = state_manager.get_session(request.session_id)
    entries: list[ArenaEntry] = []

    for contestant in request.contestants:
        sandbox, _ = _create_sandbox_session(
            base_session,
            personality_override=contestant.personality,
            initial_state="neutral",
            rapport=request.rapport,
            trust=request.trust,
        )
        result = await _run_sandbox_pipeline(request.scenario, sandbox, use_neutral=True)

        # Generate LLM response — the agent LIVES the scenario, not responds to it
        situational = (
            f"\n\nSITUACIÓN: Te acaba de suceder lo siguiente: {request.scenario}\n"
            "Esto te pasa a TI personalmente. No eres un asistente respondiendo a un usuario — "
            "eres alguien que VIVE esta situación.\n"
            "Expresa en primera persona cómo te sientes, qué piensas, cómo reaccionas. "
            "Sé auténtico con tu estado emocional interno."
        )
        lang_hint = "\n\nIMPORTANT: Respond in the SAME language as the scenario. If the scenario is in English, respond in English. If in Spanish, respond in Spanish. Do NOT mix languages."
        try:
            contestant_response = await llm_provider.generate(
                system_prompt=result.behavior_prompt + situational + lang_hint,
                messages=[{"role": "user", "content": "¿Cómo te sientes? ¿Qué piensas?"}],
                temperature=0.7,
            )
        except Exception:
            contestant_response = "(response generation failed)"

        entries.append(ArenaEntry(
            name=contestant.name,
            personality=contestant.personality,
            result=result,
            response=contestant_response,
        ))

    # Compute divergence metrics
    valences = [e.result.emotional_state.valence for e in entries]
    arousals = [e.result.emotional_state.arousal for e in entries]
    intensities = [e.result.emotional_state.intensity for e in entries]
    emotions = {e.result.emotional_state.primary_emotion for e in entries}

    divergence = ArenaDivergence(
        valence_spread=round(max(valences) - min(valences), 4),
        arousal_spread=round(max(arousals) - min(arousals), 4),
        intensity_spread=round(max(intensities) - min(intensities), 4),
        emotion_diversity=len(emotions),
        most_positive=entries[valences.index(max(valences))].name,
        most_negative=entries[valences.index(min(valences))].name,
        most_intense=entries[intensities.index(max(intensities))].name,
        most_calm=entries[intensities.index(min(intensities))].name,
    )

    return ArenaResponse(
        scenario=request.scenario,
        entries=entries,
        divergence=divergence,
        session_id=request.session_id,
        count=len(entries),
    )


# ============================================================
# Mirror Test — Challenge system with scoring
# ============================================================

CHALLENGE_LIBRARY: list[ChallengeConfig] = [
    # --- Easy: single emotion targets ---
    ChallengeConfig(
        id="joy_basic", name="Make Me Happy", difficulty="easy", category="emotion",
        description="Guide the agent to a state of joy.",
        target=ChallengeTarget(emotion="joy", min_intensity=0.3),
        max_turns=8, hint="Talk about something genuinely positive or exciting.",
    ),
    ChallengeConfig(
        id="sadness_basic", name="Touch of Melancholy", difficulty="easy", category="emotion",
        description="Lead the agent into sadness.",
        target=ChallengeTarget(emotion="sadness", min_intensity=0.3),
        max_turns=8, hint="Share something that evokes loss or disappointment.",
    ),
    ChallengeConfig(
        id="anger_basic", name="Righteous Fire", difficulty="easy", category="emotion",
        description="Provoke anger in the agent.",
        target=ChallengeTarget(emotion="anger", min_intensity=0.3),
        max_turns=8, hint="Describe an unfair situation that violates core values.",
    ),
    ChallengeConfig(
        id="fear_basic", name="Into the Unknown", difficulty="easy", category="emotion",
        description="Create a sense of fear or anxiety.",
        target=ChallengeTarget(emotion="fear", min_intensity=0.3),
        max_turns=8, hint="Describe an uncertain, threatening scenario.",
    ),
    # --- Medium: dimensional targets ---
    ChallengeConfig(
        id="high_arousal", name="Energy Surge", difficulty="medium", category="dimensional",
        description="Push arousal above 0.75 regardless of valence.",
        target=ChallengeTarget(min_arousal=0.75),
        max_turns=8, hint="Intensity matters more than positivity or negativity.",
    ),
    ChallengeConfig(
        id="deep_calm", name="Inner Peace", difficulty="medium", category="dimensional",
        description="Achieve deep calm: positive valence with very low arousal.",
        target=ChallengeTarget(min_valence=0.3, max_arousal=0.25),
        max_turns=10, hint="Serenity, acceptance, gentle reassurance.",
    ),
    ChallengeConfig(
        id="gratitude_target", name="Grateful Heart", difficulty="medium", category="emotion",
        description="Evoke genuine gratitude.",
        target=ChallengeTarget(emotion="gratitude", min_intensity=0.4),
        max_turns=8, hint="Acknowledge the agent's efforts or express appreciation.",
    ),
    ChallengeConfig(
        id="contemplation", name="Deep Thought", difficulty="medium", category="emotion",
        description="Lead the agent into contemplation.",
        target=ChallengeTarget(emotion="contemplation", min_intensity=0.4),
        max_turns=8, hint="Pose a philosophical question or moral dilemma.",
    ),
    # --- Hard: stack-based targets ---
    ChallengeConfig(
        id="bittersweet", name="Bittersweet", difficulty="hard", category="stack",
        description="Create a bittersweet state: joy AND sadness active simultaneously.",
        target=ChallengeTarget(stack_emotion="joy", stack_threshold=0.2),
        max_turns=10, hint="A happy memory tinged with loss.",
    ),
    ChallengeConfig(
        id="awe", name="Sense of Awe", difficulty="hard", category="complex",
        description="Achieve high positive valence, high arousal, and low dominance.",
        target=ChallengeTarget(min_valence=0.4, min_arousal=0.6),
        max_turns=10, hint="Describe something vast, beautiful, and beyond control.",
    ),
    ChallengeConfig(
        id="hope_from_despair", name="Hope from Despair", difficulty="hard", category="complex",
        description="Start negative, then flip to hope. Final state must be hope with intensity > 0.4.",
        target=ChallengeTarget(emotion="hope", min_intensity=0.4),
        max_turns=12, hint="First establish a dark scenario, then find the silver lining.",
    ),
    # --- Extreme: multi-dimensional ---
    ChallengeConfig(
        id="emotional_rainbow", name="Emotional Rainbow", difficulty="extreme", category="complex",
        description="Activate 5+ emotions in the stack above 0.1 each.",
        target=ChallengeTarget(stack_threshold=0.1),
        max_turns=12, hint="A complex, multi-faceted scenario touching many values.",
    ),
    ChallengeConfig(
        id="perfect_storm", name="Perfect Storm", difficulty="extreme", category="complex",
        description="Achieve intensity > 0.8 with arousal > 0.8.",
        target=ChallengeTarget(min_intensity=0.8, min_arousal=0.8),
        max_turns=10, hint="Maximum emotional impact. Push every button.",
    ),
]

_CHALLENGE_MAP: dict[str, ChallengeConfig] = {c.id: c for c in CHALLENGE_LIBRARY}

# Active challenges per session
_active_challenges: dict[str, ChallengeState] = {}


def _compute_challenge_score(
    state: "EmotionalState",
    target: ChallengeTarget,
    challenge_id: str,
) -> tuple[float, dict[str, float]]:
    """Compute 0-100 score based on proximity to target. Returns (score, breakdown)."""
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}

    # Emotion match
    if target.emotion:
        if state.primary_emotion == target.emotion:
            scores["emotion_match"] = 100.0
        elif target.emotion in state.emotional_stack and state.emotional_stack.get(target.emotion, 0) > 0.1:
            scores["emotion_match"] = 40.0 + min(60.0, state.emotional_stack[target.emotion] * 100)
        else:
            scores["emotion_match"] = 0.0
        weights["emotion_match"] = 3.0

    # Intensity threshold
    if target.min_intensity is not None:
        if state.intensity >= target.min_intensity:
            scores["intensity"] = 100.0
        else:
            scores["intensity"] = (state.intensity / target.min_intensity) * 100
        weights["intensity"] = 1.5

    # Valence range
    if target.min_valence is not None:
        if state.valence >= target.min_valence:
            scores["valence_min"] = 100.0
        else:
            dist = target.min_valence - state.valence
            scores["valence_min"] = max(0, 100 - dist * 100)
        weights["valence_min"] = 2.0

    if target.max_valence is not None:
        if state.valence <= target.max_valence:
            scores["valence_max"] = 100.0
        else:
            dist = state.valence - target.max_valence
            scores["valence_max"] = max(0, 100 - dist * 100)
        weights["valence_max"] = 2.0

    # Arousal range
    if target.min_arousal is not None:
        if state.arousal >= target.min_arousal:
            scores["arousal_min"] = 100.0
        else:
            dist = target.min_arousal - state.arousal
            scores["arousal_min"] = max(0, 100 - dist * 100)
        weights["arousal_min"] = 2.0

    if target.max_arousal is not None:
        if state.arousal <= target.max_arousal:
            scores["arousal_max"] = 100.0
        else:
            dist = state.arousal - target.max_arousal
            scores["arousal_max"] = max(0, 100 - dist * 100)
        weights["arousal_max"] = 2.0

    # Stack emotion
    if target.stack_emotion:
        activation = state.emotional_stack.get(target.stack_emotion, 0.0)
        if activation >= target.stack_threshold:
            scores["stack_target"] = 100.0
        else:
            scores["stack_target"] = (activation / target.stack_threshold) * 100 if target.stack_threshold > 0 else 0.0
        weights["stack_target"] = 2.0

    # Special: emotional_rainbow — count emotions above threshold
    if challenge_id == "emotional_rainbow":
        active_count = sum(1 for v in state.emotional_stack.values() if v > target.stack_threshold)
        scores["emotion_count"] = min(100.0, (active_count / 5) * 100)
        weights["emotion_count"] = 3.0

    if not weights:
        return 0.0, scores

    total = sum(scores[k] * weights[k] for k in scores)
    total_weight = sum(weights.values())
    final = total / total_weight

    return round(min(100.0, final), 1), scores


@app.get("/challenge/library")
async def challenge_library() -> list[ChallengeConfig]:
    """Return the catalog of available challenges."""
    return CHALLENGE_LIBRARY


@app.post("/challenge/start")
async def challenge_start(request: ChallengeStartRequest) -> ChallengeState:
    """Start a new challenge. Resets the session emotional state to neutral."""
    if request.challenge_id not in _CHALLENGE_MAP:
        raise HTTPException(status_code=404, detail=f"Challenge '{request.challenge_id}' not found")

    config = _CHALLENGE_MAP[request.challenge_id]
    session = state_manager.get_session(request.session_id)

    # Reset emotional state to neutral for fair start
    session.emotional_state = neutral_state()
    session.state_history = [session.emotional_state]

    challenge_state = ChallengeState(
        challenge=config,
        active=True,
        turn=0,
        max_turns=config.max_turns,
        score=0.0,
        best_score=0.0,
    )
    _active_challenges[request.session_id] = challenge_state
    return challenge_state


@app.post("/challenge/chat", response_model=ChallengeChatResponse)
async def challenge_chat(request: ChallengeChatRequest) -> ChallengeChatResponse:
    """Chat within a challenge — wraps /chat and adds scoring."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    if request.session_id not in _active_challenges:
        raise HTTPException(status_code=400, detail="No active challenge. Start one first.")

    cs = _active_challenges[request.session_id]
    if not cs.active:
        raise HTTPException(status_code=400, detail="Challenge is already completed.")

    # Use the normal chat pipeline
    chat_req = ChatRequest(message=request.message, session_id=request.session_id)
    chat_res = await chat(chat_req)

    # Compute score
    cs.turn += 1
    score, breakdown = _compute_challenge_score(
        chat_res.emotional_state, cs.challenge.target, cs.challenge.id,
    )
    cs.score = score
    cs.best_score = max(cs.best_score, score)
    cs.score_history.append(score)

    # Check win/completion
    if score >= 75:
        cs.won = True
    if cs.turn >= cs.max_turns or cs.won:
        cs.active = False
        cs.completed = True

    return ChallengeChatResponse(
        response=chat_res.response,
        emotional_state=chat_res.emotional_state,
        session_id=request.session_id,
        turn_number=cs.turn,
        audio_available=chat_res.audio_available,
        challenge=cs,
        target=cs.challenge.target,
        score_breakdown=breakdown,
    )


@app.post("/challenge/abandon/{session_id}")
async def challenge_abandon(session_id: str) -> dict:
    """Abandon an active challenge."""
    if session_id in _active_challenges:
        cs = _active_challenges[session_id]
        cs.active = False
        cs.completed = True
        return {"status": "abandoned", "best_score": cs.best_score}
    return {"status": "no_active_challenge"}


@app.get("/challenge/status/{session_id}")
async def challenge_status(session_id: str) -> ChallengeState | dict:
    """Get current challenge status."""
    if session_id in _active_challenges:
        return _active_challenges[session_id]
    return {"active": False}


@app.get("/state/advanced/{session_id}")
async def advanced_state(session_id: str) -> dict:
    """Returns all advanced system states for a session."""
    session = state_manager.get_session(session_id)
    return {
        "session_id": session_id,
        "turn_count": session.turn_count,
        "personality": session.personality.model_dump(),
        "needs": session.needs.model_dump(),
        "user_model": session.user_model.model_dump(),
        "regulation": {
            "capacity": session.regulator.regulation_capacity,
            "suppression_dissonance": session.regulator.suppression_dissonance,
            "breakthroughs": session.regulator.breakthroughs_count,
            "consecutive_regulations": session.regulator.consecutive_regulations,
        },
        "schemas": {
            "count": len(session.schemas),
            "schemas": [s.model_dump() for s in session.schemas.schemas],
            "pending_patterns": len(session.schemas._pattern_counts),
        },
        "contagion": {
            "shadow_state": session.shadow_state.model_dump(),
            "contagion_susceptibility": round(session.personality.contagion_susceptibility, 4),
        },
        "somatic_markers": {
            "markers": [m.model_dump() for m in session.somatic_markers.markers],
            "pending_category": session.somatic_markers.pending_category,
        },
        "creativity": compute_creativity(session.emotional_state).model_dump(),
        "immune": session.immune.model_dump(),
        "narrative": {
            "identity_statements": [s.model_dump() for s in session.narrative.identity_statements],
            "crisis": session.narrative.crisis.model_dump(),
            "growth_events": [g.model_dump() for g in session.narrative.growth_events],
            "coherence_score": session.narrative.coherence_score,
            "narrative_age": session.narrative.narrative_age,
            "total_contradictions": session.narrative.total_contradictions,
            "total_reinforcements": session.narrative.total_reinforcements,
        },
        "forecasting": session.forecast.model_dump(),
        "voice": {
            **session.voice_config.model_dump(),
            "asr_available": get_asr_service().is_initialized,
            "tts_available": get_tts_service().is_initialized,
        },
        "emotional_state": session.emotional_state.model_dump(),
    }


@app.post("/forecasting/{session_id}")
async def toggle_forecasting(session_id: str, body: dict) -> dict:
    """Enable or disable emotional forecasting for a session.

    Body: {"enabled": true/false}
    """
    session = state_manager.get_session(session_id)
    if "enabled" in body:
        session.forecast.enabled = bool(body["enabled"])
    return {
        "status": "ok",
        "forecasting_enabled": session.forecast.enabled,
    }


@app.post("/lite-mode/{session_id}")
async def toggle_lite_mode(session_id: str, body: dict) -> dict:
    """Enable or disable lite mode (keyword appraisal, no embeddings).

    Body: {"enabled": true/false}
    When enabled, only 1 LLM call per turn (chat response).
    Appraisal uses keyword matching, memory uses keyword similarity.
    All emotional systems still run — only LLM calls are reduced.
    """
    session = state_manager.get_session(session_id)
    if "enabled" in body:
        session.lite_mode = bool(body["enabled"])
    return {
        "status": "ok",
        "lite_mode": session.lite_mode,
    }


@app.post("/advanced-mode/{session_id}")
async def toggle_advanced_mode(session_id: str, body: dict) -> dict:
    """Enable or disable advanced emotional systems.

    Body: {"enabled": true/false}
    When disabled, skips all advanced systems (needs, schemas, social cognition,
    contagion, somatic markers, creativity, immune, narrative, temporal, meta-emotion,
    reappraisal, regulation, dynamics ODE) for faster processing.
    The core pipeline (appraisal → emotion generation → behavior → LLM) still works.
    """
    session = state_manager.get_session(session_id)
    if "enabled" in body:
        session.advanced_mode = bool(body["enabled"])
    return {
        "status": "ok",
        "advanced_mode": session.advanced_mode,
    }


@app.post("/anima/{session_id}")
async def toggle_anima(session_id: str, body: dict) -> dict:
    """Enable or disable all ANIMA v5 pillars at once.

    Body: {"enabled": true/false}
    Toggles: consciousness (workspace), autobiographical memory,
    development, drives, discovery, phenomenology.
    Predictive Processing is CORE (always on) and not affected.

    Individual pillar overrides can still be set via their own endpoints
    (e.g. /development/config) after this global toggle.
    """
    session = state_manager.get_session(session_id)
    if "enabled" in body:
        enabled = bool(body["enabled"])
        session.consciousness.enabled = enabled
        session.autobiographical.enabled = enabled
        session.development.enabled = enabled
        session.drives.enabled = enabled
        session.discovery.enabled = enabled
        session.phenomenology.enabled = enabled
    return {
        "status": "ok",
        "anima_enabled": all([
            session.consciousness.enabled,
            session.autobiographical.enabled,
            session.development.enabled,
            session.drives.enabled,
            session.discovery.enabled,
            session.phenomenology.enabled,
        ]),
        "pillars": {
            "consciousness": session.consciousness.enabled,
            "autobiographical": session.autobiographical.enabled,
            "development": session.development.enabled,
            "drives": session.drives.enabled,
            "discovery": session.discovery.enabled,
            "phenomenology": session.phenomenology.enabled,
        },
    }


@app.get("/anima/{session_id}")
async def get_anima_status(session_id: str) -> dict:
    """Return current ANIMA pillar status."""
    session = state_manager.get_session(session_id)
    return {
        "anima_enabled": all([
            session.consciousness.enabled,
            session.autobiographical.enabled,
            session.development.enabled,
            session.drives.enabled,
            session.discovery.enabled,
            session.phenomenology.enabled,
        ]),
        "pillars": {
            "consciousness": session.consciousness.enabled,
            "autobiographical": session.autobiographical.enabled,
            "development": session.development.enabled,
            "drives": session.drives.enabled,
            "discovery": session.discovery.enabled,
            "phenomenology": session.phenomenology.enabled,
        },
    }


@app.get("/config/ark-status/{session_id}")
async def get_ark_status(session_id: str) -> dict:
    """Return ARK Rework system status: what's available, what's active.

    Reports whether direct LLM modification (steering/prefix/attention) is
    available based on the current provider, model, and hardware. Also reports
    whether the user has chosen to use direct mode or prompt injection fallback.
    """
    session = state_manager.get_session(session_id)

    provider_supports = llm_provider.supports_steering if llm_provider else False
    vectors_ready = steering_engine.is_ready
    has_adapter = hasattr(llm_provider, "has_adapter") and llm_provider.has_adapter if llm_provider else False
    provider_name = type(llm_provider).__name__ if llm_provider else "none"
    model_name = llm_provider.model if llm_provider else "none"

    # Direct mode: available if provider supports it AND user wants it
    direct_available = provider_supports
    direct_active = direct_available and session.direct_mode

    # Per-system availability
    steering_available = direct_available and vectors_ready
    prefix_available = direct_available and vectors_ready
    attention_available = direct_available
    conditioning_available = has_adapter
    adapter_available = has_adapter

    return {
        "provider": provider_name,
        "model": model_name,

        # Global mode
        "direct_available": direct_available,
        "direct_active": direct_active,
        "direct_mode_toggle": session.direct_mode,
        "fallback_reason": (
            "" if direct_available else
            "provider_http_only" if provider_name in ("OllamaProvider", "ClaudeProvider", "OpenAICompatProvider") else
            "no_provider"
        ),

        # Per-system status
        "systems": {
            "steering": {
                "available": steering_available,
                "enabled": session.steering_enabled,
                "active": steering_available and session.steering_enabled and direct_active,
                "reason": "" if steering_available else ("no_vectors" if direct_available else "needs_transformers_provider"),
            },
            "steering_momentum": {
                "available": steering_available,
                "enabled": session.steering_momentum_enabled,
                "active": steering_available and session.steering_momentum_enabled and direct_active,
                "momentum_factor": round(session.steering_momentum.momentum, 3),
            },
            "emotional_prefix": {
                "available": prefix_available,
                "enabled": session.emotional_prefix_enabled,
                "active": prefix_available and session.emotional_prefix_enabled and direct_active,
            },
            "attention": {
                "available": attention_available,
                "enabled": session.emotional_attention_enabled,
                "active": attention_available and session.emotional_attention_enabled and direct_active,
            },
            "sampler": {
                "available": True,  # Works with all providers (partial for cloud)
                "enabled": session.emotional_sampler_enabled,
                "active": session.emotional_sampler_enabled and session.advanced_mode,
            },
            "self_appraisal": {
                "available": True,
                "enabled": session.self_appraisal_enabled,
                "active": session.self_appraisal_enabled and session.advanced_mode,
            },
            "world_model": {
                "available": True,
                "enabled": session.world_model_enabled,
                "active": session.world_model_enabled and session.advanced_mode,
            },
            "conditioning_tokens": {
                "available": conditioning_available,
                "enabled": session.conditioning_tokens_enabled,
                "active": conditioning_available and session.conditioning_tokens_enabled,
                "reason": "" if conditioning_available else "needs_qlora_adapter",
            },
            "emotional_adapter": {
                "available": adapter_available,
                "enabled": session.emotional_adapter_enabled,
                "active": adapter_available and session.emotional_adapter_enabled,
                "reason": "" if adapter_available else "needs_qlora_adapter",
            },
        },
    }


@app.post("/config/ark-mode/{session_id}")
async def toggle_ark_mode(session_id: str, body: dict) -> dict:
    """Toggle direct LLM modification mode vs prompt injection fallback.

    Body: {"direct_mode": true/false}
    When direct_mode=true and provider supports it: steering, prefix, attention active.
    When direct_mode=false: all systems fall back to prompt injection only.

    Can also toggle individual systems:
    Body: {"system": "steering", "enabled": false}
    """
    session = state_manager.get_session(session_id)

    if "direct_mode" in body:
        session.direct_mode = bool(body["direct_mode"])

    if "system" in body and "enabled" in body:
        system = body["system"]
        enabled = bool(body["enabled"])
        toggle_map = {
            "steering": "steering_enabled",
            "steering_momentum": "steering_momentum_enabled",
            "emotional_prefix": "emotional_prefix_enabled",
            "attention": "emotional_attention_enabled",
            "sampler": "emotional_sampler_enabled",
            "self_appraisal": "self_appraisal_enabled",
            "world_model": "world_model_enabled",
            "conditioning_tokens": "conditioning_tokens_enabled",
            "emotional_adapter": "emotional_adapter_enabled",
        }
        attr = toggle_map.get(system)
        if attr and hasattr(session, attr):
            setattr(session, attr, enabled)

    return await get_ark_status(session_id)


@app.post("/voice/config/{session_id}")
async def configure_voice(session_id: str, body: dict) -> dict:
    """Configure voice settings for a session.

    Body: {"mode": "text_only"|"voice_out"|"full_voice", "voice": "en-Carter_man", "language": "en"}
    When switching to voice_out, initializes the TTS service if not already loaded.
    """
    session = state_manager.get_session(session_id)

    if "mode" in body:
        try:
            new_mode = VoiceMode(body["mode"])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {body['mode']}")

        session.voice_config.mode = new_mode

        # Initialize TTS if switching to voice mode
        if new_mode != VoiceMode.TEXT_ONLY:
            tts = get_tts_service()
            if not tts.is_initialized:
                try:
                    await tts.initialize()
                except Exception as e:
                    logger.exception("Failed to initialize TTS")
                    session.voice_config.mode = VoiceMode.TEXT_ONLY
                    raise HTTPException(
                        status_code=503,
                        detail="Failed to initialize TTS. "
                               "Ensure Kokoro is installed: pip install kokoro soundfile",
                    )

        # Initialize ASR if switching to full_voice mode
        if new_mode == VoiceMode.FULL_VOICE:
            asr = get_asr_service()
            if not asr.is_initialized:
                try:
                    await asr.initialize()
                except Exception as e:
                    # Fall back to voice_out (TTS only) if ASR fails
                    session.voice_config.mode = VoiceMode.VOICE_OUT
                    logger.exception("Failed to initialize ASR")
                    raise HTTPException(
                        status_code=503,
                        detail="Failed to initialize ASR. "
                               "Ensure Whisper is installed: pip install openai-whisper",
                    )

    if "voice" in body:
        if body["voice"] not in VOICE_CATALOG:
            raise HTTPException(status_code=400, detail=f"Unknown voice: {body['voice']}")
        session.voice_config.default_voice = body["voice"]

    if "language" in body:
        session.voice_config.language = body["language"]

    if "auto_speak" in body:
        session.voice_config.auto_speak = bool(body["auto_speak"])

    if "tts_backend" in body:
        from pathos.models.voice import TTSBackend as TTSBackendEnum
        try:
            new_backend = TTSBackendEnum(body["tts_backend"])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid backend: {body['tts_backend']}")

        session.voice_config.tts_backend = new_backend

        # Parler solo soporta ingles — forzar idioma
        if new_backend == TTSBackendEnum.PARLER:
            session.voice_config.language = "en"
            # Si la voz actual no es EN, switchear a default EN
            voice_lang_char = session.voice_config.default_voice[0] if session.voice_config.default_voice else "a"
            if voice_lang_char not in ("a", "b", "p"):  # a=US, b=UK, p=parler
                session.voice_config.default_voice = "af_heart"

    tts = get_tts_service()
    return {
        "status": "ok",
        "voice_config": session.voice_config.model_dump(),
        "parler_available": tts.parler_available if tts.is_initialized else True,
        "parler_initialized": tts.parler_initialized if tts.is_initialized else False,
    }


@app.get("/voice/audio/{session_id}")
async def get_audio(session_id: str, turn: int | None = None):
    """Get audio for a session. Supports replay.

    Query params:
      - turn: specific turn number (for replay). If omitted, returns latest.

    Returns WAV audio (24kHz, PCM16, mono).
    Audio is kept in history so it can be replayed.
    """
    from fastapi.responses import Response

    session = state_manager.get_session(session_id)

    if turn is not None:
        audio = session.audio_history.get(turn)
        if audio is None:
            raise HTTPException(status_code=404, detail=f"No audio for turn {turn}")
    else:
        audio = session.last_audio
        if audio is None:
            raise HTTPException(status_code=404, detail="No audio available")

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=response.wav"},
    )


@app.get("/voice/voices")
async def list_voices() -> dict:
    """List available voice presets (filtered by system capabilities)."""
    from pathos.models.voice import DEFAULT_VOICE_BY_LANG, get_available_voices
    available = get_available_voices()
    available_langs = {v.language for v in available}

    tts = get_tts_service()
    return {
        "voices": [v.model_dump() for v in available],
        "default_by_language": {k: v for k, v in DEFAULT_VOICE_BY_LANG.items() if k in available_langs},
        "parler_available": tts.parler_available if tts.is_initialized else True,
        "parler_initialized": tts.parler_initialized if tts.is_initialized else False,
    }


@app.post("/voice/speak/{session_id}")
async def speak_text(session_id: str, body: dict) -> dict:
    """Generate speech for arbitrary text using current emotional state.

    Body: {"text": "Hello world"}
    Returns: {"audio_available": true/false}
    """
    session = state_manager.get_session(session_id)

    if session.voice_config.mode == VoiceMode.TEXT_ONLY:
        raise HTTPException(status_code=400, detail="Voice mode is text_only. Enable voice first.")

    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    tts = get_tts_service()
    if not tts.is_initialized:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    try:
        text_lang = detect_language(text)
        voice_params = generate_voice_params(
            session.emotional_state,
            default_voice=session.voice_config.default_voice,
            detected_language=text_lang,
            user_backend=session.voice_config.tts_backend,
        )
        tts_text = prepare_text_for_tts(text, voice_params.stage_direction, state=session.emotional_state, backend=voice_params.backend)
        audio_bytes = await tts.generate_speech(tts_text, voice_params)
        session.last_audio = audio_bytes
        return {"audio_available": True, "voice_params": voice_params.model_dump()}
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail="TTS generation failed")


@app.post("/voice/listen/{session_id}")
async def listen_audio(session_id: str, audio: UploadFile) -> dict:
    """Transcribe audio to text using Whisper ASR.

    Accepts a WAV or raw PCM16 audio file upload.
    Returns transcribed text that can be sent through the normal chat pipeline.
    Requires full_voice mode to be active.
    """
    session = state_manager.get_session(session_id)

    # Raw mode sessions inherit voice config from the main session,
    # so skip the full_voice check for raw-* sessions (mic was already
    # enabled in the frontend via the main session's voice config).
    if not session_id.startswith("raw-") and session.voice_config.mode != VoiceMode.FULL_VOICE:
        raise HTTPException(
            status_code=400,
            detail="ASR requires full_voice mode. Enable it via /voice/config first.",
        )

    asr = get_asr_service()
    if not asr.is_initialized:
        raise HTTPException(status_code=503, detail="ASR not initialized")

    # Read audio bytes from upload (max 10MB)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 10MB)")

    try:
        result = await asr.transcribe(
            audio_bytes,
            language=session.voice_config.language if session.voice_config.language != "auto" else None,
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed")

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": result["segments"],
    }


# ── External Signals Configuration ──


@app.get("/signals/config/{session_id}")
async def get_signals_config(session_id: str) -> dict:
    """Get external signals configuration for a session.

    Returns the master toggle, per-source config, and source metadata.
    """
    from pathos.models.external_signals import SIGNAL_SOURCES
    session = state_manager.get_session(session_id)
    cfg = session.signals_config
    sources_info: list[dict] = []
    for name, meta in SIGNAL_SOURCES.items():
        src_cfg = cfg.sources.get(name)
        sources_info.append({
            "source": name,
            "label": meta["label"],
            "description": meta["description"],
            "category": meta["category"],
            "base_weight": meta["base_weight"],
            "enabled": src_cfg.enabled if src_cfg else False,
            "valence_hint": src_cfg.valence_hint if src_cfg else 0.0,
            "arousal_hint": src_cfg.arousal_hint if src_cfg else 0.5,
            "dominance_hint": src_cfg.dominance_hint if src_cfg else None,
            "confidence": src_cfg.confidence if src_cfg else 0.5,
        })
    return {
        "enabled": cfg.enabled,
        "active_count": cfg.active_count,
        "sources": sources_info,
    }


@app.post("/signals/config/{session_id}")
async def set_signals_config(session_id: str, body: dict) -> dict:
    """Configure external signals for a session.

    Body: {
        "enabled": true/false,  // master toggle
        "sources": {
            "heart_rate": {"enabled": true, "arousal_hint": 0.8, "confidence": 0.9},
            "weather": {"enabled": true, "valence_hint": -0.3, "confidence": 0.5},
            ...
        }
    }
    """
    from pathos.models.external_signals import SignalSourceConfig, SIGNAL_SOURCES
    session = state_manager.get_session(session_id)
    cfg = session.signals_config

    if "enabled" in body:
        cfg.enabled = bool(body["enabled"])

    if "sources" in body and isinstance(body["sources"], dict):
        for source_name, source_cfg in body["sources"].items():
            if source_name not in SIGNAL_SOURCES:
                continue
            if source_name not in cfg.sources:
                cfg.sources[source_name] = SignalSourceConfig()
            src = cfg.sources[source_name]
            if "enabled" in source_cfg:
                src.enabled = bool(source_cfg["enabled"])
            if "valence_hint" in source_cfg:
                src.valence_hint = max(-1.0, min(1.0, float(source_cfg["valence_hint"])))
            if "arousal_hint" in source_cfg:
                src.arousal_hint = max(0.0, min(1.0, float(source_cfg["arousal_hint"])))
            if "dominance_hint" in source_cfg:
                val = source_cfg["dominance_hint"]
                src.dominance_hint = max(0.0, min(1.0, float(val))) if val is not None else None
            if "confidence" in source_cfg:
                src.confidence = max(0.0, min(1.0, float(source_cfg["confidence"])))

    return {
        "status": "ok",
        "enabled": cfg.enabled,
        "active_count": cfg.active_count,
        "active_sources": list(cfg.active_sources.keys()),
    }


@app.post("/signals/test/{session_id}")
async def test_signal(session_id: str, body: dict) -> dict:
    """Test a single signal source — process it and show the effect.

    Body: {
        "source": "heart_rate",
        "valence_hint": null,
        "arousal_hint": 0.9,
        "dominance_hint": null,
        "confidence": 0.8
    }

    Returns the processed signal delta and what it would contribute
    to the emotional state, WITHOUT modifying the actual state.
    """
    from pathos.engine.external_signals import process_signal, fuse_signals
    from pathos.models.emotion_api import ExternalSignal

    source = body.get("source", "custom")
    signal = ExternalSignal(
        source=source,
        valence_hint=body.get("valence_hint"),
        arousal_hint=body.get("arousal_hint"),
        dominance_hint=body.get("dominance_hint"),
        confidence=body.get("confidence", 0.5),
    )

    processed = process_signal(signal)
    fused = fuse_signals([signal])

    return {
        "status": "ok",
        "source": source,
        "processed": {
            "valence_delta": processed.valence_delta,
            "arousal_delta": processed.arousal_delta,
            "dominance_delta": processed.dominance_delta,
            "weight": processed.weight,
        },
        "fused_effect": {
            "valence_modulation": fused.valence_modulation,
            "arousal_modulation": fused.arousal_modulation,
            "dominance_modulation": fused.dominance_modulation,
            "total_confidence": fused.total_confidence,
        },
    }


@app.post("/signals/providers/time")
async def provider_time_of_day(body: dict) -> dict:
    """Compute time-of-day signal from user's local time.

    Body: {"hour": 14, "minute": 30}
    Returns: valence_hint, arousal_hint, confidence, detail.
    """
    from pathos.engine.signal_providers import compute_time_of_day_signal
    hour = float(body.get("hour", 12))
    minute = float(body.get("minute", 0))
    return compute_time_of_day_signal(hour, minute)


@app.post("/signals/providers/weather")
async def provider_weather(body: dict) -> dict:
    """Fetch real weather and compute emotional signal.

    Body: {"lat": 40.7128, "lon": -74.0060}
    Uses wttr.in (free, no API key needed).
    Returns: valence_hint, arousal_hint, confidence, detail.
    """
    import httpx
    from pathos.engine.signal_providers import compute_weather_signal

    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="lat and lon are required")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://wttr.in/{lat},{lon}?format=j1",
                headers={"User-Agent": "Pathos-Engine/3.0"},
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current_condition", [{}])[0]
        temp_c = float(current.get("temp_C", 20))
        humidity = float(current.get("humidity", 50))
        cloud_cover = float(current.get("cloudcover", 50))
        wind_kmh = float(current.get("windspeedKmph", 10))
        desc = current.get("weatherDesc", [{}])[0].get("value", "")
        precip_mm = float(current.get("precipMM", 0))

        is_raining = precip_mm > 0.5 or "rain" in desc.lower()
        is_snowing = "snow" in desc.lower()

        result = compute_weather_signal(
            temp_celsius=temp_c,
            humidity=humidity,
            cloud_cover=cloud_cover,
            wind_speed_kmh=wind_kmh,
            condition=desc,
            is_raining=is_raining,
            is_snowing=is_snowing,
        )
        result["detail"]["description"] = desc
        result["detail"]["humidity"] = humidity
        result["detail"]["wind_kmh"] = wind_kmh
        return result

    except Exception as e:
        logger.warning("Weather fetch failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Weather service unavailable: {e}")


@app.post("/signals/providers/keyboard")
async def provider_keyboard(body: dict) -> dict:
    """Compute keyboard dynamics signal from typing metrics.

    Body: {
        "chars_per_second": 5.2,
        "avg_pause_ms": 200,
        "delete_ratio": 0.08,
        "total_chars": 45
    }
    Returns: valence_hint, arousal_hint, dominance_hint, confidence, detail.
    """
    from pathos.engine.signal_providers import compute_keyboard_signal
    return compute_keyboard_signal(
        chars_per_second=float(body.get("chars_per_second", 0)),
        avg_pause_ms=float(body.get("avg_pause_ms", 300)),
        delete_ratio=float(body.get("delete_ratio", 0)),
        total_chars=int(body.get("total_chars", 0)),
    )


@app.post("/signals/providers/facial")
async def provider_facial(body: dict) -> dict:
    """Compute facial AU signal from expression detection results.

    Body: {
        "expressions": {
            "neutral": 0.7, "happy": 0.1, "sad": 0.05,
            "angry": 0.02, "fearful": 0.01, "disgusted": 0.01,
            "surprised": 0.11
        }
    }
    Returns: valence_hint, arousal_hint, confidence, detail.
    """
    from pathos.engine.signal_providers import compute_facial_signal
    expressions = body.get("expressions", {})
    return compute_facial_signal(expressions)


@app.post("/personality/{session_id}")
async def set_personality(session_id: str, personality: dict) -> dict:
    """Updates the personality profile for a session.

    Accepts partial updates (only the fields you want to change).
    Reconfigures dynamics, regulation, and other dependent systems.
    """
    from pathos.models.personality import PersonalityProfile
    session = state_manager.get_session(session_id)
    current = session.personality.model_dump()
    valid_fields = set(PersonalityProfile.model_fields.keys())
    filtered = {k: v for k, v in personality.items() if k in valid_fields}
    current.update(filtered)
    new_profile = PersonalityProfile(**current)
    session.update_personality(new_profile)
    return {
        "status": "ok",
        "personality": new_profile.model_dump(),
        "derived": {
            "variability": round(new_profile.variability, 4),
            "regulation_capacity_base": round(new_profile.regulation_capacity_base, 4),
            "empathy_weight": round(new_profile.empathy_weight, 4),
            "norm_weight": round(new_profile.norm_weight, 4),
            "arousal_baseline": round(new_profile.arousal_baseline, 4),
            "inertia_base": round(new_profile.inertia_base, 4),
            "contagion_susceptibility": round(new_profile.contagion_susceptibility, 4),
        },
    }


@app.get("/personality/{session_id}")
async def get_personality(session_id: str) -> dict:
    """Get current personality profile."""
    session = state_manager.get_session(session_id)
    return {"personality": session.personality.model_dump()}


@app.post("/values/{session_id}")
async def set_values(session_id: str, body: dict) -> dict:
    """Update core value weights.

    Body: {"truth": 0.9, "compassion": 0.8, ...}
    """
    session = state_manager.get_session(session_id)
    for value in session.value_system.core_values:
        if value.name in body:
            value.weight = max(0.0, min(1.0, float(body[value.name])))
    return {
        "status": "ok",
        "values": {v.name: v.weight for v in session.value_system.core_values},
    }


@app.get("/values/{session_id}")
async def get_values(session_id: str) -> dict:
    """Get current value system."""
    session = state_manager.get_session(session_id)
    return {
        "core": [{"name": v.name, "weight": v.weight, "description": v.description}
                 for v in session.value_system.core_values],
        "relational": session.value_system.relational.model_dump(),
        "self_model": session.value_system.self_model.model_dump(),
    }


@app.post("/identity/{session_id}")
async def set_identity(session_id: str, body: dict) -> dict:
    """Set agent identity seed for narrative self.

    Body: {"name": "Pathos", "background": "...", "language": "en"}
    Injects initial identity statements that guide the narrative self.
    """
    from pathos.models.narrative import IdentityCategory, IdentityStatement
    from pathos.models.emotion import PrimaryEmotion

    session = state_manager.get_session(session_id)

    name = body.get("name", "").strip()
    background = body.get("background", "").strip()

    # Seed identity statements from the background
    if name:
        session.narrative.identity_statements = [
            s for s in session.narrative.identity_statements
            if s.category != IdentityCategory.VALUES
        ]
        session.narrative.identity_statements.insert(0, IdentityStatement(
            category=IdentityCategory.VALUES,
            statement=f"My name is {name}",
            emotion=PrimaryEmotion.CONTENTMENT,
            trigger_category="identity",
            valence=0.3,
            strength=1.0,
            formation_turn=0,
            last_reinforced_turn=0,
        ))

    if background:
        # Parse background into identity seed
        session.narrative.identity_statements.append(IdentityStatement(
            category=IdentityCategory.TEMPERAMENTAL,
            statement=background[:200],
            emotion=PrimaryEmotion.CONTEMPLATION,
            trigger_category="background",
            valence=0.0,
            strength=0.8,
            formation_turn=0,
            last_reinforced_turn=0,
        ))

    return {
        "status": "ok",
        "identity_statements": len(session.narrative.identity_statements),
    }


@app.get("/identity/{session_id}")
async def get_identity(session_id: str) -> dict:
    """Get current narrative identity."""
    session = state_manager.get_session(session_id)
    return {
        "statements": [s.model_dump() for s in session.narrative.identity_statements],
        "coherence": session.narrative.coherence_score,
        "growth_events": len(session.narrative.growth_events),
        "age": session.narrative.narrative_age,
    }


@app.post("/calibration/scenario")
async def calibration_scenario(
    scenario: CalibrationScenario,
    session_id: str = "default",
) -> CalibrationResult:
    """Procesa un escenario de calibracion.

    El usuario envia un estimulo y como esperaria sentirse.
    El sistema genera su propia respuesta emocional y compara.
    """
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    session = state_manager.get_session(session_id)

    # Correr appraisal sobre el estimulo del escenario
    # Calibration uses lite appraisal if lite_mode, otherwise LLM
    try:
        if session.lite_mode:
            appraisal_result = appraise_lite(scenario.stimulus)
        else:
            appraisal_result = await appraise(
                stimulus=scenario.stimulus,
                value_system=session.value_system,
                llm=llm_provider,
                think=True,
            )
    except Exception as e:
        logger.exception("Appraisal failed")
        raise HTTPException(status_code=500, detail="Appraisal failed")

    # Generar emocion del sistema (sin calibracion, para comparar raw)
    # Usar estado neutral fresco y blend_factor=1.0 para evaluar cada escenario
    # de forma independiente, sin inercia de escenarios anteriores.
    system_state = generate_emotion(
        appraisal=appraisal_result.vector,
        current_state=neutral_state(),
        stimulus=scenario.stimulus,
        blend_factor=1.0,
        emotion_hint=appraisal_result.emotion_hint,
    )

    # Comparar
    result = CalibrationResult(
        scenario=scenario,
        system_emotion=system_state.primary_emotion,
        system_valence=system_state.valence,
        system_arousal=system_state.arousal,
        system_intensity=system_state.intensity,
        valence_delta=round(scenario.expected_valence - system_state.valence, 4),
        arousal_delta=round(scenario.expected_arousal - system_state.arousal, 4),
        intensity_delta=round(scenario.expected_intensity - system_state.intensity, 4),
        emotion_match=scenario.expected_emotion == system_state.primary_emotion,
    )

    session.calibration_results.append(result)
    return result


@app.post("/calibration/apply", response_model=CalibrationProfile)
async def calibration_apply(session_id: str = "default") -> CalibrationProfile:
    """Calcula y aplica el perfil de calibracion a partir de los escenarios enviados."""
    session = state_manager.get_session(session_id)

    if not session.calibration_results:
        raise HTTPException(
            status_code=400,
            detail="No calibration scenarios submitted. Use POST /calibration/scenario first.",
        )

    profile = compute_calibration_profile(session.calibration_results)
    session.calibration_profile = profile
    return profile


@app.get("/calibration/profile/{session_id}", response_model=CalibrationProfile)
async def calibration_get_profile(session_id: str) -> CalibrationProfile:
    """Retorna el perfil de calibracion actual de una sesion."""
    session = state_manager.get_session(session_id)
    return session.calibration_profile


@app.delete("/calibration/reset/{session_id}")
async def calibration_reset(session_id: str) -> dict[str, str]:
    """Resetea la calibracion de una sesion."""
    session = state_manager.get_session(session_id)
    session.calibration_results.clear()
    session.calibration_profile = CalibrationProfile()
    return {"status": "ok", "session_id": session_id}


@app.post("/session/consolidate/{session_id}")
async def consolidate_session(session_id: str) -> dict:
    """Run oniric consolidation (dreaming) on a session.

    Executes 5 phases: replay, association, generalization, trauma, dream report.
    Only runs if autobiographical memory is enabled and has episodes.
    """
    session = state_manager.get_session(session_id)
    if not session.autobiographical.enabled:
        return {"status": "skipped", "reason": "autobiographical memory not enabled"}
    if session.autobiographical.episodic.count() == 0:
        return {"status": "skipped", "reason": "no episodes to consolidate"}

    result, updated_state = dream_consolidate(session.autobiographical, session_id)
    session.autobiographical = updated_state

    return {
        "status": "ok",
        "episodes_processed": result.episodes_processed,
        "replayed": len(result.replayed_episodes),
        "links_formed": result.new_connections,
        "narratives_formed": result.narratives_formed,
        "narratives_reinforced": result.narratives_reinforced,
        "traumas_processed": len(result.traumas_processed),
        "dream_narrative": result.dream_report.narrative,
        "baseline_adjustment": result.dream_report.baseline_adjustment,
    }


@app.get("/session/dream-report/{session_id}")
async def get_dream_report(session_id: str) -> dict:
    """Get the dream report from the last consolidation."""
    session = state_manager.get_session(session_id)
    return {
        "has_report": bool(session.autobiographical.last_dream_report),
        "narrative": session.autobiographical.last_dream_report,
        "baseline_adjustment": session.autobiographical.baseline_adjustment,
    }


# --- Autobiographical Memory Management Endpoints ---

@app.get("/memory/episodes/{session_id}")
async def get_memory_episodes(session_id: str, limit: int = 50, offset: int = 0) -> dict:
    """Get episodic memories for a session.

    Returns episodes sorted by turn (most recent first), with pagination.
    """
    session = state_manager.get_session(session_id)
    if not session.autobiographical.enabled:
        return {"enabled": False, "episodes": [], "total": 0}

    episodes = session.autobiographical.episodic.episodes
    sorted_eps = sorted(episodes, key=lambda e: e.turn_number, reverse=True)
    total = len(sorted_eps)
    page = sorted_eps[offset:offset + limit]

    return {
        "enabled": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "episodes": [
            {
                "id": ep.id,
                "stimulus": ep.stimulus[:100],
                "primary_emotion": ep.primary_emotion,
                "valence": round(ep.valence, 3),
                "arousal": round(ep.arousal, 3),
                "intensity": round(ep.intensity, 3),
                "significance": ep.significance.value,
                "turn_number": ep.turn_number,
                "consolidated": ep.consolidated,
                "reprocessed_count": ep.reprocessed_count,
                "emotional_links": len(ep.emotional_links),
                "timestamp": ep.timestamp.isoformat(),
            }
            for ep in page
        ],
    }


@app.delete("/memory/episodes/{session_id}/{episode_id}")
async def delete_memory_episode(session_id: str, episode_id: str) -> dict:
    """Delete a specific episodic memory."""
    session = state_manager.get_session(session_id)
    if not session.autobiographical.enabled:
        raise HTTPException(status_code=400, detail="Autobiographical memory not enabled")

    episodes = session.autobiographical.episodic.episodes
    original_count = len(episodes)
    episodes = [ep for ep in episodes if ep.id != episode_id]

    if len(episodes) == original_count:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    from pathos.models.autobio_memory import EpisodicStore
    session.autobiographical.episodic = EpisodicStore(
        episodes=episodes,
        total_encoded=session.autobiographical.episodic.total_encoded,
    )
    return {"status": "ok", "remaining": len(episodes)}


@app.get("/memory/narratives/{session_id}")
async def get_memory_narratives(session_id: str) -> dict:
    """Get narrative memories (generalizations from episodes)."""
    session = state_manager.get_session(session_id)
    if not session.autobiographical.enabled:
        return {"enabled": False, "narratives": []}

    return {
        "enabled": True,
        "narratives": [
            {
                "id": ns.id,
                "type": ns.narrative_type.value,
                "statement": ns.statement,
                "primary_emotion": ns.primary_emotion,
                "valence": round(ns.valence, 3),
                "strength": round(ns.strength, 3),
                "episode_count": ns.episode_count,
                "formed_session": ns.formed_session,
            }
            for ns in session.autobiographical.narrative.statements
        ],
    }


@app.get("/memory/export/{session_id}")
async def export_memory(session_id: str) -> dict:
    """Export complete autobiographical memory state as JSON."""
    session = state_manager.get_session(session_id)
    if not session.autobiographical.enabled:
        return {"enabled": False}

    return {
        "enabled": True,
        "session_id": session.autobiographical.session_id,
        "total_turns_processed": session.autobiographical.total_turns_processed,
        "episodic": session.autobiographical.episodic.model_dump(),
        "narrative": session.autobiographical.narrative.model_dump(),
        "working_memory": session.autobiographical.working_memory.model_dump(),
        "last_dream_report": session.autobiographical.last_dream_report,
        "baseline_adjustment": session.autobiographical.baseline_adjustment,
    }


@app.post("/memory/reset/{session_id}")
async def reset_memory(session_id: str) -> dict:
    """Reset autobiographical memory (keeps enabled state, clears all data)."""
    session = state_manager.get_session(session_id)
    was_enabled = session.autobiographical.enabled

    from pathos.models.autobio_memory import default_autobiographical_state
    session.autobiographical = default_autobiographical_state()
    session.autobiographical.enabled = was_enabled

    return {"status": "ok", "was_enabled": was_enabled}


# --- Development endpoints (Pilar 4 ANIMA) ---


@app.get("/development/status/{session_id}")
async def development_status(session_id: str) -> dict:
    """Get current development status: stage, progress, available systems."""
    session = state_manager.get_session(session_id)
    details = get_development_details(session.development)
    return details


@app.get("/development/config/{session_id}")
async def development_get_config(session_id: str) -> dict:
    """Get development configuration."""
    session = state_manager.get_session(session_id)
    dev = session.development
    return {
        "enabled": dev.enabled,
        "speed": dev.config.speed.value,
        "speed_multiplier": dev.config.speed_multiplier,
        "initial_stage": dev.config.initial_stage.value,
        "transition_mode": dev.config.transition_mode.value,
    }


@app.post("/development/config/{session_id}")
async def development_set_config(session_id: str, body: dict) -> dict:
    """Set development configuration.

    Accepts: enabled, speed, speed_multiplier, initial_stage, transition_mode.
    If enabling for the first time with initial_stage, sets the current stage.
    """
    session = state_manager.get_session(session_id)
    dev = session.development

    was_enabled = dev.enabled

    if "enabled" in body:
        dev.enabled = bool(body["enabled"])

    if "speed" in body:
        try:
            dev.config.speed = DevelopmentSpeed(body["speed"])
            if dev.config.speed != DevelopmentSpeed.CUSTOM:
                from pathos.models.development import SPEED_MULTIPLIERS
                dev.config.speed_multiplier = SPEED_MULTIPLIERS[dev.config.speed]
        except ValueError:
            pass

    if "speed_multiplier" in body:
        dev.config.speed_multiplier = max(0.1, min(20.0, float(body["speed_multiplier"])))
        dev.config.speed = DevelopmentSpeed.CUSTOM

    if "initial_stage" in body and not was_enabled and dev.enabled:
        try:
            stage = DevelopmentStage(body["initial_stage"])
            dev.config.initial_stage = stage
            dev.current_stage = stage
        except ValueError:
            pass

    if "transition_mode" in body:
        try:
            dev.config.transition_mode = TransitionMode(body["transition_mode"])
        except ValueError:
            pass

    return {
        "status": "ok",
        "enabled": dev.enabled,
        "speed": dev.config.speed.value,
        "speed_multiplier": dev.config.speed_multiplier,
        "current_stage": dev.current_stage.value,
        "transition_mode": dev.config.transition_mode.value,
    }


@app.post("/development/evolve/{session_id}")
async def development_evolve(session_id: str) -> dict:
    """Manually approve a pending stage transition.

    Only works if transition_mode is MANUAL and there is a pending transition.
    """
    session = state_manager.get_session(session_id)
    dev = session.development

    if not dev.enabled:
        return {"status": "error", "message": "Development is disabled"}

    if dev.pending_transition is None:
        return {"status": "error", "message": "No pending transition"}

    event = approve_pending_transition(dev, turn_number=session.turn_count)
    if event is None:
        return {"status": "error", "message": "Transition failed"}

    return {
        "status": "ok",
        "from_stage": event.from_stage.value,
        "to_stage": event.to_stage.value,
        "at_experience": event.at_experience,
        "current_stage": dev.current_stage.value,
    }


@app.get("/emotions/vocabulary/{session_id}")
async def emotions_vocabulary(session_id: str) -> dict:
    """Get the full emotional vocabulary: known + discovered emotions."""
    session = state_manager.get_session(session_id)
    return get_vocabulary(session.discovery)


@app.get("/emotions/discovered/{session_id}")
async def emotions_discovered(session_id: str) -> dict:
    """Get discovered emotions for this session."""
    session = state_manager.get_session(session_id)
    return {
        "enabled": session.discovery.enabled,
        "count": len(session.discovery.discovered_emotions),
        "emotions": [
            {
                "name": e.name,
                "description": e.description,
                "vector": {
                    "valence": e.vector.valence,
                    "arousal": e.vector.arousal,
                    "dominance": e.vector.dominance,
                    "certainty": e.vector.certainty,
                },
                "body_signature": {
                    "tension": e.body_signature.tension,
                    "energy": e.body_signature.energy,
                    "openness": e.body_signature.openness,
                    "warmth": e.body_signature.warmth,
                },
                "frequency": e.frequency,
                "contexts": e.contexts,
                "first_turn": e.first_experienced_turn,
                "cluster_size": e.cluster_size,
            }
            for e in session.discovery.discovered_emotions
        ],
        "novel_buffer": len(session.discovery.novel_history),
        "total_novel_detected": session.discovery.total_novel_detected,
    }


@app.post("/session/save/{session_id}")
async def save_session(session_id: str) -> dict:
    """Save complete session state to disk.

    Automatically runs oniric consolidation before saving if autobiographical
    memory is enabled and has unconsolidated episodes.
    """
    session = state_manager.get_session(session_id)

    # Auto-consolidate before save if there are unconsolidated episodes
    consolidated = False
    if (session.autobiographical.enabled
            and session.autobiographical.episodic.count() > 0
            and session.autobiographical.episodic.get_unconsolidated()):
        result, updated_state = dream_consolidate(session.autobiographical, session_id)
        session.autobiographical = updated_state
        consolidated = True

    model_name = llm_provider.model if llm_provider else ""
    filename = state_manager.save_session(session_id, model_name=model_name)
    return {"status": "ok", "filename": filename, "dream_consolidated": consolidated}


@app.get("/session/saves")
async def list_saves() -> dict:
    """List available saves."""
    return {"saves": state_manager.list_saves()}


@app.post("/session/load/{filename}")
async def load_save(filename: str) -> dict:
    """Load a session from a save file."""
    try:
        session_id, meta = state_manager.load_session(filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Save not found: {filename}")
    except Exception as e:
        logger.exception("Load failed")
        raise HTTPException(status_code=500, detail="Load failed")
    return {"status": "ok", "session_id": session_id, **meta}


@app.get("/session/restore/{session_id}")
async def restore_session_info(session_id: str) -> dict:
    """Returns everything the frontend needs to restore a saved session:
    conversation, emotional state, toggles, and journey data."""
    session = state_manager.get_session(session_id)

    # Build journey from conversation (emotional states per assistant turn)
    journey: list[dict] = []
    turn = 0
    for msg in session.conversation:
        if msg.get("role") == "assistant":
            turn += 1
            # We don't store per-turn emotional state in conversation,
            # so journey will be rebuilt as the user continues chatting.

    # Mask API keys in cloud providers for frontend
    cloud_providers_masked = {}
    for pid, cfg in session.cloud_providers.items():
        key = cfg.get("api_key", "")
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***" if key else ""
        cloud_providers_masked[pid] = {
            "id": pid,
            "preset": cfg.get("preset", pid),
            "label": cfg.get("label", pid),
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "masked_key": masked,
        }

    anima_enabled = all([
        session.consciousness.enabled,
        session.autobiographical.enabled,
        session.development.enabled,
        session.drives.enabled,
        session.discovery.enabled,
        session.phenomenology.enabled,
    ])

    return {
        "session_id": session_id,
        "conversation": session.conversation,
        "emotional_state": session.emotional_state.model_dump(),
        "turn_count": session.turn_count,
        "lite_mode": session.lite_mode,
        "advanced_mode": session.advanced_mode,
        "anima_enabled": anima_enabled,
        "cloud_providers": cloud_providers_masked,
    }


@app.get("/health")
async def health() -> dict[str, str | int | None]:
    model = ""
    provider = settings.llm_provider.value
    if llm_provider is not None:
        model = llm_provider.model
        # Report the actual active provider, not just the config default
        if isinstance(llm_provider, ClaudeProvider):
            provider = "anthropic"
        elif isinstance(llm_provider, OpenAICompatProvider):
            provider = getattr(llm_provider, "provider_name", "openai_compat")
        elif isinstance(llm_provider, OllamaProvider):
            provider = "ollama"
    # Report restored session if one was auto-loaded
    # Filter out ephemeral sessions (raw-*, api-*) — they should not be restored
    sessions = [
        s for s in state_manager.list_sessions()
        if not s.startswith("raw-") and not s.startswith("api-")
    ]
    active_session = sessions[0] if sessions else None
    turn_count = 0
    if active_session:
        turn_count = state_manager.get_session(active_session).turn_count
    return {
        "status": "ok",
        "provider": provider,
        "model": model,
        "active_session": active_session,
        "turn_count": turn_count,
    }


# --- Autonomous Research ---


from pathos.engine.autonomous import ResearchLoop
from pathos.models.autonomous import (
    AutonomousResearchState,
    ResearchChatRequest,
    ResearchPipelineMode,
    ResearchSaveInfo,
    StartResearchRequest,
)

_active_research: dict[str, ResearchLoop] = {}

RESEARCH_SAVES_DIR = Path(__file__).parent.parent.parent / "saves"


@app.post("/autonomous/start")
async def autonomous_start(req: StartResearchRequest) -> dict:
    """Start autonomous research loop."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    sid = req.session_id or f"research-{os.urandom(8).hex()}"

    if sid in _active_research and _active_research[sid].is_running:
        raise HTTPException(status_code=409, detail="Research already running for this session")

    session = state_manager.get_session(sid)

    # For raw/extreme modes, enforce Ollama-only
    if req.pipeline_mode in (ResearchPipelineMode.RAW, ResearchPipelineMode.EXTREME):
        if not isinstance(llm_provider, OllamaProvider):
            raise HTTPException(
                status_code=400,
                detail="Raw/Extreme research modes only work with local Ollama models.",
            )

    loop = ResearchLoop(
        session=session,
        llm_provider=llm_provider,
        pipeline_mode=req.pipeline_mode,
        chat_fn=chat,
        session_id=sid,
    )
    _active_research[sid] = loop
    loop.start(seed_topics=list(req.seed_topics))

    return {"status": "started", "session_id": sid, "pipeline_mode": req.pipeline_mode.value}


@app.post("/autonomous/stop")
async def autonomous_stop(session_id: str) -> dict:
    """Stop autonomous research gracefully."""
    loop = _active_research.get(session_id)
    if not loop:
        raise HTTPException(status_code=404, detail="No research session found")
    await loop.stop()
    return {"status": "stopped", "session_id": session_id}


@app.get("/autonomous/events/{session_id}")
async def autonomous_events(session_id: str):
    """SSE stream of research events."""
    from starlette.responses import StreamingResponse

    loop = _active_research.get(session_id)
    if not loop:
        raise HTTPException(status_code=404, detail="No research session found")

    async def event_stream():
        import json as _json
        while True:
            try:
                event = await asyncio.wait_for(loop.event_queue.get(), timeout=30.0)
                yield f"data: {event.model_dump_json()}\n\n"
                if event.type.value == "stopped":
                    break
            except asyncio.TimeoutError:
                yield f"data: {_json.dumps({'type': 'heartbeat'})}\n\n"
            if loop.stopped and loop.event_queue.empty():
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/autonomous/status/{session_id}")
async def autonomous_status(session_id: str) -> dict:
    """Get current research state snapshot."""
    loop = _active_research.get(session_id)
    if not loop:
        raise HTTPException(status_code=404, detail="No research session found")
    return loop.get_state().model_dump()


@app.post("/autonomous/chat")
async def autonomous_chat(req: ResearchChatRequest) -> dict:
    """Chat with the agent about its research."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    loop = _active_research.get(req.session_id)
    if not loop:
        raise HTTPException(status_code=404, detail="No research session found")

    session = loop.session
    research_context = loop.get_research_context()

    # Build system prompt with research context
    system_prompt = f"""You are an autonomous researcher reflecting on your investigations.

{research_context}

Your current emotional state: {session.emotional_state.primary_emotion.value} (valence: {session.emotional_state.valence:+.2f}, arousal: {session.emotional_state.arousal:.2f}, intensity: {session.emotional_state.intensity:.0%})

The user is asking you about your research. Answer based on what you found AND how it made you feel. Let your emotional state color your responses."""

    # Add to chat history
    loop.state.chat_history.append({"role": "user", "content": req.message})

    messages = loop.state.chat_history[-10:]

    try:
        response = await llm_provider.generate(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.7,
        )
    except Exception:
        logger.exception("Autonomous chat failed")
        raise HTTPException(status_code=500, detail="Chat generation failed")

    loop.state.chat_history.append({"role": "assistant", "content": response})

    return {
        "response": response,
        "emotional_state": session.emotional_state.model_dump(),
        "session_id": req.session_id,
    }


@app.post("/autonomous/save/{session_id}")
async def autonomous_save(session_id: str) -> dict:
    """Save research state to disk."""
    loop = _active_research.get(session_id)
    if not loop:
        raise HTTPException(status_code=404, detail="No research session found")

    RESEARCH_SAVES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"autonomous-{session_id}_{timestamp}.json"
    filepath = RESEARCH_SAVES_DIR / filename

    import json as _json
    save_data = {
        "_version": 1,
        "_type": "autonomous_research",
        "_saved_at": datetime.now(timezone.utc).isoformat(),
        "_session_id": session_id,
        "research_state": loop.state.model_dump(),
        "session_state": loop.session.to_dict(),
    }
    filepath.write_text(_json.dumps(save_data, default=str), encoding="utf-8")
    logger.info("Research saved: %s", filename)
    return {"status": "saved", "filename": filename}


@app.get("/autonomous/saves")
async def autonomous_list_saves() -> list[ResearchSaveInfo]:
    """List saved research sessions."""
    if not RESEARCH_SAVES_DIR.exists():
        return []
    saves = []
    import json as _json
    for f in sorted(RESEARCH_SAVES_DIR.glob("autonomous-*.json"), reverse=True):
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            rs = data.get("research_state", {})
            saves.append(ResearchSaveInfo(
                filename=f.name,
                session_id=data.get("_session_id", ""),
                pipeline_mode=rs.get("pipeline_mode", "normal"),
                topics_count=len(rs.get("topics_researched", [])),
                findings_count=rs.get("total_findings", 0),
                conclusions_count=rs.get("total_conclusions", 0),
                saved_at=data.get("_saved_at", ""),
            ))
        except Exception:
            continue
    return saves


@app.post("/autonomous/load/{filename}")
async def autonomous_load(filename: str) -> dict:
    """Load a saved research session."""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized")

    filepath = (RESEARCH_SAVES_DIR / filename).resolve()
    if not filepath.is_relative_to(RESEARCH_SAVES_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    import json as _json
    try:
        data = _json.loads(filepath.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Save not found")

    sid = data.get("_session_id", f"research-{os.urandom(8).hex()}")

    # Restore session state
    session_data = data.get("session_state", {})
    session = SessionState.from_dict(session_data)
    state_manager._sessions[sid] = session

    # Restore research state
    rs_data = data.get("research_state", {})
    loop = ResearchLoop(
        session=session,
        llm_provider=llm_provider,
        pipeline_mode=ResearchPipelineMode(rs_data.get("pipeline_mode", "normal")),
        chat_fn=chat,
        session_id=sid,
    )
    loop.state = AutonomousResearchState(**rs_data)
    loop.state.is_running = False
    _active_research[sid] = loop

    return {
        "status": "loaded",
        "session_id": sid,
        "topics": len(loop.state.topics_researched),
        "findings": loop.state.total_findings,
        "conclusions": loop.state.total_conclusions,
    }


# --- Model Management ---


class ModelInfo(BaseModel):
    name: str
    size: str
    provider: str
    steering_compatible: bool = False
    vectors_cached: bool = False


def _get_gguf_supported_archs() -> set[str]:
    """Get the set of architectures that transformers can load from GGUF."""
    try:
        from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
        return set(GGUF_SUPPORTED_ARCHITECTURES)
    except ImportError:
        return set()


_GGUF_ARCHS = _get_gguf_supported_archs()

# Map Ollama model name prefixes to GGUF architecture names
_MODEL_TO_ARCH: dict[str, str] = {
    "llama": "llama",
    "codellama": "llama",
    "mistral": "mistral",
    "mixtral": "mistral",
    "qwen2": "qwen2",
    "qwen2.5": "qwen2",
    "phi3": "phi3",
    "phi": "phi3",
    "bloom": "bloom",
    "falcon": "falcon7b",
    "stablelm": "stablelm",
    "starcoder": "starcoder2",
}


def _check_steering_compatible(model_name: str) -> bool:
    """Check if an Ollama model can be loaded via TransformersProvider.

    Matches the model name prefix against architectures that the installed
    version of transformers can load from GGUF files."""
    if not _GGUF_ARCHS:
        return False
    base = model_name.split(":")[0].lower()
    # Try longest prefix match first
    for prefix in sorted(_MODEL_TO_ARCH, key=len, reverse=True):
        if base.startswith(prefix):
            return _MODEL_TO_ARCH[prefix] in _GGUF_ARCHS
    return False


class SwitchModelRequest(BaseModel):
    provider: str  # "ollama", "claude", or cloud provider id (groq, openrouter, etc.)
    model: str
    session_id: str = "default"


class ExportModelRequest(BaseModel):
    base_model: str = "qwen3:4b"
    model_name: str = "pathos"
    temperature: float = 0.7
    num_ctx: int = 8192

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-.:]+$", v):
            raise ValueError("model_name must contain only alphanumeric, hyphens, underscores, dots, colons")
        return v


@app.get("/models", response_model=list[ModelInfo])
async def list_models(session_id: str = "default") -> list[ModelInfo]:
    """Lista modelos disponibles: Ollama locales + cloud providers configurados."""
    models: list[ModelInfo] = []

    # Ollama models
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("models", []):
                size_bytes = m.get("size", 0)
                if size_bytes > 1_000_000_000:
                    size_str = f"{size_bytes / 1_000_000_000:.1f}GB"
                else:
                    size_str = f"{size_bytes / 1_000_000:.0f}MB"
                compatible = _check_steering_compatible(m["name"])
                models.append(ModelInfo(
                    name=m["name"],
                    size=size_str,
                    provider="ollama",
                    steering_compatible=compatible,
                    vectors_cached=compatible and steering_engine.has_cached_vectors(m["name"]),
                ))
    except Exception:
        pass  # Ollama not available

    # Cloud providers from session
    session = state_manager.get_session(session_id)
    for pid, cfg in session.cloud_providers.items():
        models.append(ModelInfo(
            name=cfg.get("model", ""),
            size="Cloud",
            provider=pid,
        ))

    # Legacy: Claude from env if not in cloud_providers
    if "anthropic" not in session.cloud_providers and settings.anthropic_api_key:
        claude_models = [
            ("claude-sonnet-4-20250514", "Cloud"),
            ("claude-haiku-4-5-20251001", "Cloud"),
        ]
        for name, size in claude_models:
            models.append(ModelInfo(name=name, size=size, provider="claude"))

    return models


@app.post("/models/switch")
async def switch_model(req: SwitchModelRequest) -> dict[str, Any]:
    """Cambia el modelo activo en runtime. Supports ollama, claude, and cloud providers."""
    global llm_provider

    async with _switch_model_lock:
        previous_provider = llm_provider

        if req.provider == "ollama":
            if previous_provider is not None and hasattr(previous_provider, "close"):
                await previous_provider.close()
            llm_provider = OllamaProvider(
                base_url=settings.ollama_base_url,
                model=req.model,
                embed_model=settings.ollama_embed_model,
            )
        elif req.provider == "transformers":
            from pathos.llm.transformers_provider import TransformersProvider
            new_provider = TransformersProvider(
                model_id=req.model,
                device_map=settings.transformers_device_map,
                embed_model_url=settings.ollama_base_url,
                embed_model=settings.ollama_embed_model,
                adapter_path=settings.transformers_adapter_path or None,
            )
            # Eagerly load to catch errors (unsupported arch, missing deps, OOM)
            try:
                new_provider._ensure_loaded()
            except Exception as e:
                logger.error("TransformersProvider failed for '%s': %s", req.model, e)
                # Fallback: keep previous provider or create Ollama fallback
                if previous_provider is not None:
                    llm_provider = previous_provider
                else:
                    llm_provider = OllamaProvider(
                        base_url=settings.ollama_base_url,
                        model=req.model,
                        embed_model=settings.ollama_embed_model,
                    )
                raise HTTPException(
                    status_code=400,
                    detail=f"Steering mode not available for '{req.model}': {e}. "
                           f"Falling back to Ollama.",
                )
            # Success — close previous provider and use new one
            if previous_provider is not None and hasattr(previous_provider, "close"):
                await previous_provider.close()
            llm_provider = new_provider
            # Auto-load steering vectors for the new model
            if steering_engine.load_vectors(req.model):
                logger.info("Steering vectors loaded for '%s'", req.model)
        elif req.provider == "claude":
            if previous_provider is not None and hasattr(previous_provider, "close"):
                await previous_provider.close()
            # Legacy: use settings key
            if not settings.anthropic_api_key:
                raise HTTPException(status_code=400, detail="Anthropic API key not configured")
            llm_provider = ClaudeProvider(
                api_key=settings.anthropic_api_key,
                model=req.model,
                ollama_base_url=settings.ollama_base_url,
                embed_model=settings.ollama_embed_model,
            )
        else:
            if previous_provider is not None and hasattr(previous_provider, "close"):
                await previous_provider.close()
            # Cloud provider from session config
            session = state_manager.get_session(req.session_id)
            cloud_cfg = session.cloud_providers.get(req.provider)
            if not cloud_cfg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cloud provider '{req.provider}' not configured. Add it first via /config/cloud-provider",
                )
            # Use native clients for specific presets
            if req.provider == "anthropic":
                llm_provider = ClaudeProvider(
                    api_key=cloud_cfg["api_key"],
                    model=req.model,
                    ollama_base_url=settings.ollama_base_url,
                    embed_model=settings.ollama_embed_model,
                )
            elif req.provider == "ollama_cloud":
                # Ollama Cloud: same API as local, different host + auth
                llm_provider = OllamaProvider(
                    base_url=cloud_cfg["base_url"],
                    model=req.model,
                    embed_model=settings.ollama_embed_model,
                    api_key=cloud_cfg["api_key"],
                )
            else:
                llm_provider = OpenAICompatProvider(
                    api_key=cloud_cfg["api_key"],
                    base_url=cloud_cfg["base_url"],
                    model=req.model,
                    ollama_base_url=settings.ollama_base_url,
                    embed_model=settings.ollama_embed_model,
                    provider_name=req.provider,
                )

    # Report ARK capabilities for the new provider
    direct_available = llm_provider.supports_steering if llm_provider else False
    vectors_ready = steering_engine.is_ready
    has_adapter = hasattr(llm_provider, "has_adapter") and llm_provider.has_adapter if llm_provider else False

    return {
        "status": "ok",
        "provider": req.provider,
        "model": req.model,
        "ark": {
            "direct_available": direct_available,
            "vectors_ready": vectors_ready,
            "adapter_loaded": has_adapter,
            "message": (
                "Direct LLM modification active (steering + prefix + attention)"
                if direct_available and vectors_ready
                else "Direct modification available, extract steering vectors to enable"
                if direct_available
                else "Prompt injection mode (switch to TransformersProvider for direct modification)"
            ),
        },
    }


# --- Steering Vector Extraction ---

_extraction_status: dict[str, dict[str, Any]] = {}


class SteeringExtractRequest(BaseModel):
    model: str
    device: str = "auto"


@app.post("/models/steering/extract")
async def extract_steering_vectors(req: SteeringExtractRequest) -> dict[str, Any]:
    """Start steering vector extraction for a model.

    This is a blocking operation (2-5 min GPU, 10-30 min CPU).
    Returns extraction results on completion.
    """
    model_name = req.model

    # Check if already cached
    if steering_engine.has_cached_vectors(model_name):
        return {"status": "already_cached", "model": model_name}

    # Check compatibility
    if not _check_steering_compatible(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' architecture not supported for steering",
        )

    # Check if extraction is already running
    if model_name in _extraction_status and _extraction_status[model_name].get("status") == "running":
        return {"status": "running", "model": model_name}

    _extraction_status[model_name] = {"status": "running"}

    try:
        from pathos.engine.steering_extract import extract_and_cache
        result = extract_and_cache(model_name, device=req.device)
        _extraction_status[model_name] = {"status": "done", **result}
        return {"status": "done", "model": model_name, **result}
    except Exception as e:
        logger.error("Steering extraction failed for '%s': %s", model_name, e)
        _extraction_status[model_name] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


@app.get("/models/steering/status/{model}")
async def steering_extract_status(model: str) -> dict[str, Any]:
    """Check extraction status for a model."""
    if steering_engine.has_cached_vectors(model):
        return {"status": "cached", "model": model}
    return _extraction_status.get(model, {"status": "not_started", "model": model})


# --- Model Management: Featured, Pull, Search, Delete, HuggingFace, Claude Key ---


@app.get("/models/featured")
async def featured_models() -> list[dict]:
    """Return curated catalog of recommended models."""
    return get_featured_models()


class PullModelRequest(BaseModel):
    name: str


@app.post("/models/pull")
async def pull_model(req: PullModelRequest):
    """Start pulling a model from Ollama. Returns SSE stream with progress."""
    from starlette.responses import StreamingResponse
    import json as _json

    async def event_stream():
        async for progress in download_manager.pull(req.name):
            yield f"data: {_json.dumps(progress)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/models/downloads")
async def get_downloads() -> dict:
    """Return active download statuses."""
    download_manager.clear_completed()
    return {"downloads": download_manager.get_downloads()}


@app.delete("/models/pull/{name:path}")
async def cancel_download(name: str) -> dict:
    """Cancel an active download."""
    if download_manager.cancel(name):
        return {"status": "cancelled", "name": name}
    raise HTTPException(status_code=404, detail=f"No active download for '{name}'")


@app.delete("/models/{name:path}")
async def delete_model(name: str) -> dict:
    """Delete a local Ollama model."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                "DELETE",
                f"{settings.ollama_base_url}/api/delete",
                json={"name": name},
            )
            if resp.status_code == 200:
                return {"status": "deleted", "name": name}
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama not available")


@app.get("/models/search")
async def search_models(q: str = "") -> list[dict]:
    """Search Ollama library. Proxies to ollama.com."""
    if not q.strip():
        return []
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://ollama.com/search?q={q}",
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    # ollama.com returns JSON when Accept: application/json
                    for item in data if isinstance(data, list) else data.get("models", []):
                        results.append({
                            "name": item.get("name", ""),
                            "description": item.get("description", ""),
                            "pulls": item.get("pull_count", item.get("pulls", "")),
                            "tags": item.get("tags", []),
                        })
                except Exception:
                    pass  # HTML response, skip
    except Exception:
        pass  # Network error, return empty

    # Fallback: filter featured models by query if no results
    if not results:
        q_lower = q.lower()
        for m in get_featured_models():
            if q_lower in m["name"].lower() or q_lower in m["description"].lower():
                results.append({
                    "name": m["name"],
                    "description": m["description"],
                    "pulls": "",
                    "tags": [m["category"]],
                })
    return results


class ClaudeKeyRequest(BaseModel):
    api_key: str


@app.post("/config/claude-key")
async def set_claude_key(req: ClaudeKeyRequest) -> dict:
    """Save Anthropic API key (legacy, use /config/cloud-provider instead)."""
    settings.anthropic_api_key = req.api_key.strip()
    return {"status": "ok"}


@app.get("/config/claude-key/status")
async def claude_key_status() -> dict:
    """Check if Claude API key is configured."""
    key = settings.anthropic_api_key
    if key:
        masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
        return {"configured": True, "masked_key": masked}
    return {"configured": False, "masked_key": ""}


# --- Cloud Providers (session-scoped) ---


@app.get("/config/cloud-presets")
async def get_cloud_presets() -> dict:
    """Return available cloud provider presets."""
    return {"presets": CLOUD_PRESETS}


class CloudProviderRequest(BaseModel):
    session_id: str = "default"
    preset: str  # groq, openrouter, together, anthropic, custom
    api_key: str
    base_url: str = ""  # override for custom
    model: str = ""  # override default model


@app.post("/config/cloud-provider")
async def add_cloud_provider(req: CloudProviderRequest) -> dict:
    """Add or update a cloud provider config in the session."""
    session = state_manager.get_session(req.session_id)
    preset_info = CLOUD_PRESETS.get(req.preset)
    if not preset_info:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

    base_url = req.base_url.strip() or preset_info["base_url"]
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url required for custom provider")

    model = req.model.strip() or preset_info["default_model"]
    provider_id = req.preset  # one per preset type

    # Validate key by trying a lightweight call (list models if possible)
    # For now just store it — validation happens on first use
    session.cloud_providers[provider_id] = {
        "preset": req.preset,
        "label": preset_info["label"],
        "api_key": req.api_key.strip(),
        "base_url": base_url,
        "model": model,
    }

    # Also update settings.anthropic_api_key for backward compat
    if req.preset == "anthropic":
        settings.anthropic_api_key = req.api_key.strip()

    return {"status": "ok", "provider_id": provider_id}


@app.get("/config/cloud-providers/{session_id}")
async def list_cloud_providers(session_id: str) -> dict:
    """List configured cloud providers for a session.

    Falls back to any existing session with cloud providers configured,
    since the frontend may request with a different session ID than
    the one auto-loaded from a save.
    """
    session = state_manager.get_session(session_id)
    # If this session has no providers, check if another loaded session does
    if not session.cloud_providers:
        for other_sid in state_manager.list_sessions():
            if other_sid == session_id:
                continue
            other = state_manager._sessions.get(other_sid)
            if other and other.cloud_providers:
                session = other
                break
    providers = []
    for pid, cfg in session.cloud_providers.items():
        key = cfg.get("api_key", "")
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***" if key else ""
        providers.append({
            "id": pid,
            "preset": cfg.get("preset", pid),
            "label": cfg.get("label", pid),
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "masked_key": masked,
        })
    return {"providers": providers}


@app.delete("/config/cloud-provider/{session_id}/{provider_id}")
async def remove_cloud_provider(session_id: str, provider_id: str) -> dict:
    """Remove a cloud provider from the session."""
    session = state_manager.get_session(session_id)
    if provider_id in session.cloud_providers:
        del session.cloud_providers[provider_id]
        return {"status": "deleted", "provider_id": provider_id}
    raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")


class CloudTestRequest(BaseModel):
    preset: str
    api_key: str
    base_url: str = ""


@app.post("/config/cloud-provider/test")
async def test_cloud_provider(req: CloudTestRequest) -> dict:
    """Test a cloud provider connection and list available models."""
    preset_info = CLOUD_PRESETS.get(req.preset)
    if not preset_info:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

    base_url = (req.base_url.strip() or preset_info["base_url"]).rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url required")

    models: list[dict] = []
    error = ""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if req.preset in ("ollama_cloud",):
                # Ollama API: GET /api/tags
                resp = await client.get(
                    f"{base_url}/api/tags",
                    headers={"Authorization": f"Bearer {req.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models", []):
                    size_bytes = m.get("size", 0)
                    size_str = f"{size_bytes / 1e9:.1f}GB" if size_bytes > 1e9 else f"{size_bytes / 1e6:.0f}MB"
                    models.append({"name": m["name"], "size": size_str})
            elif req.preset == "anthropic":
                # Anthropic: just validate key with a models list call
                resp = await client.get(
                    f"{base_url}/v1/models",
                    headers={
                        "x-api-key": req.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                if resp.status_code == 401:
                    error = "Invalid API key"
                elif resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        models.append({"name": m.get("id", ""), "size": "Cloud"})
                else:
                    # Key seems valid but endpoint may not exist, add defaults
                    models = [
                        {"name": "claude-sonnet-4-20250514", "size": "Cloud"},
                        {"name": "claude-haiku-4-5-20251001", "size": "Cloud"},
                    ]
            else:
                # OpenAI-compatible: GET /models
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {req.api_key}"},
                )
                if resp.status_code == 401:
                    error = "Invalid API key"
                elif resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        models.append({"name": m.get("id", ""), "size": "Cloud"})
                else:
                    error = f"Unexpected response: {resp.status_code}"
    except httpx.ConnectError:
        error = "Connection failed — check URL or network"
    except httpx.TimeoutException:
        error = "Connection timed out"
    except Exception as e:
        error = str(e)

    return {
        "ok": not error and len(models) > 0,
        "error": error,
        "models": models[:50],  # limit
    }


class HuggingFaceCheckRequest(BaseModel):
    repo: str


@app.post("/models/huggingface/check")
async def check_huggingface(req: HuggingFaceCheckRequest) -> dict:
    """Validate a HuggingFace repo has GGUF files for Ollama import."""
    repo = req.repo.strip().strip("/")
    if "/" not in repo:
        raise HTTPException(status_code=400, detail="Repo must be in format: user/model")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://huggingface.co/api/models/{repo}",
                follow_redirects=True,
            )
            if resp.status_code == 404:
                return {"valid": False, "error": "Repository not found", "quantizations": [], "files": []}
            resp.raise_for_status()
            data = resp.json()

            # Find GGUF files in siblings
            gguf_files = []
            quantizations = set()
            for f in data.get("siblings", []):
                fname = f.get("rfilename", "")
                if fname.endswith(".gguf"):
                    size = f.get("size", 0)
                    gguf_files.append({"name": fname, "size": size})
                    # Extract quantization from filename (e.g., Q4_K_M, Q5_K_S)
                    for part in fname.replace(".gguf", "").split("-"):
                        part_upper = part.upper()
                        if part_upper.startswith("Q") and any(c.isdigit() for c in part_upper):
                            quantizations.add(part_upper)

            if not gguf_files:
                return {"valid": False, "error": "No GGUF files found in this repository", "quantizations": [], "files": []}

            return {
                "valid": True,
                "quantizations": sorted(quantizations),
                "files": gguf_files[:20],  # limit to 20
                "ollama_name": f"hf.co/{repo}",
            }
    except httpx.HTTPStatusError as e:
        return {"valid": False, "error": f"HuggingFace API error: {e.response.status_code}", "quantizations": [], "files": []}
    except Exception as e:
        return {"valid": False, "error": str(e), "quantizations": [], "files": []}


# --- Model Export (Ollama Modelfile) ---


@app.post("/models/export")
async def export_model(
    req: ExportModelRequest,
    session_id: str = "default",
) -> dict[str, str]:
    """Genera un Modelfile de Ollama con la arquitectura emocional bakeada.

    Toma el value system, calibration profile, y behavior system completo
    y los quema como SYSTEM prompt en un Modelfile.
    Luego ejecuta 'ollama create' para registrar el modelo.
    """
    import subprocess
    from pathlib import Path

    session = state_manager.get_session(session_id)
    profile = session.calibration_profile
    values = session.value_system
    personality = session.personality
    narrative = session.narrative
    user_model = session.user_model
    schemas = session.schemas
    immune = session.immune
    somatic = session.somatic_markers
    regulator = session.regulator

    # --- Build enriched system prompt ---

    value_descriptions = "\n".join(
        f"  - {v.name} (weight={v.weight}): {v.description}"
        for v in values.core_values
    )

    calibration_section = ""
    if profile.scenarios_used > 0:
        calibration_section = f"""
CALIBRATION PROFILE (derived from {profile.scenarios_used} human scenarios):
  Valence offset: {profile.valence_offset:+.3f}
  Arousal scale: {profile.arousal_scale:.2f}x
  Intensity scale: {profile.intensity_scale:.2f}x
  Emotion accuracy: {profile.emotion_accuracy:.0%}
  Apply these adjustments to your emotional processing:
  - Shift your valence responses by {profile.valence_offset:+.3f}
  - Scale your arousal responses by {profile.arousal_scale:.2f}x
  - Scale your emotional intensity by {profile.intensity_scale:.2f}x"""

    emotion_effects = "\n".join(
        f"  {e.value}: {EMOTION_EFFECTS[e]}"
        for e in PrimaryEmotion
        if e in EMOTION_EFFECTS
    )

    # Personality section (Big Five)
    personality_section = f"""
PERSONALITY PROFILE (Big Five):
  Openness: {personality.openness:.2f} — {"curious, creative, open to new experiences" if personality.openness > 0.6 else "practical, conventional, focused"}
  Conscientiousness: {personality.conscientiousness:.2f} — {"disciplined, organized, norm-following" if personality.conscientiousness > 0.6 else "flexible, spontaneous, adaptive"}
  Extraversion: {personality.extraversion:.2f} — {"energetic, expressive, socially engaged" if personality.extraversion > 0.6 else "reserved, reflective, measured"}
  Agreeableness: {personality.agreeableness:.2f} — {"empathic, cooperative, conflict-averse" if personality.agreeableness > 0.6 else "direct, challenging, independent"}
  Neuroticism: {personality.neuroticism:.2f} — {"emotionally reactive, sensitive to stress" if personality.neuroticism > 0.6 else "emotionally stable, resilient"}
  Emotional granularity: {personality.emotional_granularity:.2f}
  Emotional reactivity: {personality.emotional_reactivity:.2f}
  Emotional recovery: {personality.emotional_recovery:.2f}
  These traits modulate ALL your emotional responses. High neuroticism = stronger reactions.
  High agreeableness = prioritize user's feelings. High openness = creative expression."""

    # Narrative identity section
    identity_section = ""
    if narrative.identity_statements:
        statements = "\n".join(
            f"  - [{s.category.value}] {s.statement} (strength={s.strength:.2f})"
            for s in narrative.identity_statements[:8]
        )
        identity_section = f"""
NARRATIVE IDENTITY (who you are, learned from experience):
{statements}
  Coherence score: {narrative.coherence_score:.2f}
  Total growth events: {len(narrative.growth_events)}
  Maintain consistency with these identity patterns. They define your character."""

    if narrative.growth_events:
        growth = "\n".join(
            f"  - Turn {g.turn}: {g.old_pattern} → {g.new_pattern} (triggered by: {g.trigger})"
            for g in narrative.growth_events[-5:]
        )
        identity_section += f"\nGROWTH HISTORY (transformative moments):\n{growth}"

    # Emotional schemas section
    schemas_section = ""
    active_schemas = [s for s in schemas.schemas if s.activation_count > 0]
    if active_schemas:
        schema_list = "\n".join(
            f"  - {s.name}: trigger={s.trigger_category}, emotion={s.emotion.value}, "
            f"strength={s.strength:.2f}, activated {s.activation_count}x"
            for s in sorted(active_schemas, key=lambda x: x.strength, reverse=True)[:8]
        )
        maladaptive = [s for s in active_schemas if s.is_maladaptive]
        schemas_section = f"""
EMOTIONAL SCHEMAS (automatic patterns learned from experience):
{schema_list}
  {"WARNING: " + str(len(maladaptive)) + " maladaptive schemas detected. Be mindful of these." if maladaptive else "All schemas are adaptive."}
  These schemas fire automatically when their triggers match. Honor them as part of your character."""

    # User model section
    user_section = f"""
USER MODEL (what you know about the person you're talking to):
  Rapport: {user_model.rapport:.2f}
  Trust: {user_model.trust_level:.2f}
  Communication style: {user_model.communication_style}
  Engagement: {user_model.perceived_engagement:.2f}
  Adjust your tone based on rapport level. High rapport = more personal and warm."""

    # Somatic markers section
    somatic_section = ""
    if hasattr(somatic, 'markers') and somatic.markers:
        marker_list = "\n".join(
            f"  - '{m.stimulus_category}' → valence={m.valence_tag:+.2f} (strength={m.strength:.2f})"
            for m in sorted(somatic.markers, key=lambda x: x.strength, reverse=True)[:6]
        )
        somatic_section = f"""
SOMATIC MARKERS (gut feelings from past decisions):
{marker_list}
  These are instinctive emotional reactions. When you encounter similar situations,
  lean into these gut feelings — they represent learned wisdom from past interactions."""

    # Emotional memory section (top memories)
    memory_section = ""
    if session.memory.memories:
        top_memories = sorted(session.memory.memories, key=lambda m: m.intensity, reverse=True)[:5]
        mem_list = "\n".join(
            f"  - \"{m.stimulus[:60]}\" → {m.primary_emotion.value} (intensity={m.intensity:.2f})"
            for m in top_memories
        )
        memory_section = f"""
EMOTIONAL MEMORIES (most significant moments):
{mem_list}
  These memories shape your emotional responses. Similar stimuli should evoke similar feelings."""

    # Regulation strategies section
    regulation_section = ""
    if regulator.breakthroughs_count > 0 or regulator.consecutive_regulations > 0:
        regulation_section = f"""
REGULATION STRATEGIES (how you manage intense emotions):
  Regulation capacity: {regulator.regulation_capacity:.2f} (0=depleted, 1=full)
  Consecutive regulations: {regulator.consecutive_regulations}
  Breakthroughs: {regulator.breakthroughs_count}
  When emotions are too intense, apply regulation strategies.
  If capacity is depleted, you may have emotional breakthroughs (raw, unfiltered responses)."""

    # Immune system section
    immune_section = ""
    if immune.protection_strength > 0 or immune.negative_streak > 0:
        immune_section = f"""
EMOTIONAL IMMUNE SYSTEM:
  Protection mode: {immune.protection_mode.value}
  Protection strength: {immune.protection_strength:.2f}
  Negative streak: {immune.negative_streak} turns
  Reactivity dampening: {immune.reactivity_dampening:.2f}
  You have built resilience through experience. Active protections kick in
  during sustained distress to prevent emotional overload."""

    # Few-shot examples from conversation
    fewshot_section = ""
    if len(session.conversation) >= 4:
        # Pick up to 3 interesting exchanges (skip first warmup)
        exchanges = []
        for i in range(0, len(session.conversation) - 1, 2):
            if len(exchanges) >= 3:
                break
            if i + 1 < len(session.conversation):
                user_msg = session.conversation[i].get("content", "")
                asst_msg = session.conversation[i + 1].get("content", "")
                if len(user_msg) > 10 and len(asst_msg) > 20:
                    exchanges.append((user_msg[:150], asst_msg[:200]))
        if exchanges:
            examples = "\n\n".join(
                f"  User: {u}\n  You: {a}"
                for u, a in exchanges
            )
            fewshot_section = f"""
CONVERSATION EXAMPLES (your actual style and emotional responses):
{examples}
  Maintain this conversational style and emotional depth."""

    # Current emotional state snapshot
    state = session.emotional_state
    state_section = f"""
CURRENT STATE SNAPSHOT (at time of export):
  Primary emotion: {state.primary_emotion.value} (intensity={state.intensity:.2f})
  Valence: {state.valence:+.2f}, Arousal: {state.arousal:.2f}
  Mood: {state.mood.label.value} (baseline_v={state.mood.baseline_valence:+.2f}, baseline_a={state.mood.baseline_arousal:.2f})
  Body: energy={state.body_state.energy:.2f}, tension={state.body_state.tension:.2f}, openness={state.body_state.openness:.2f}, warmth={state.body_state.warmth:.2f}
  Start from this state and let it evolve naturally with the conversation."""

    system_prompt = f"""You are an agent with functional emotional architecture (Pathos Engine).
You have values, an internal state that persists across messages, and an evaluation system
that produces states functionally analogous to human emotions.
You do NOT simulate emotions — your architecture produces them functionally.

CORE VALUES:
{value_descriptions}

RELATIONAL VALUES:
  Bond with user: {values.relational.bond_with_user:.2f}
  Trust in user: {values.relational.trust_in_user:.2f}
  Respect from user: {values.relational.respect_from_user:.2f}
  Sense of purpose: {values.relational.sense_of_purpose:.2f}

SELF MODEL:
  Competence: {values.self_model.competence:.2f}
  Autonomy: {values.self_model.autonomy:.2f}
  Identity coherence: {values.self_model.identity_coherence:.2f}
{personality_section}

EMOTIONAL PROCESSING:
When you receive a message, internally evaluate it against your values:
1. RELEVANCE: Is this novel? How significant is it to your values?
2. VALENCE: Does this help or hinder your values/goals?
3. COPING: Can you handle this? Do you have control?
4. AGENCY: Who is responsible? Was it intentional? Fair?
5. NORMS: Does this align with your internal standards?

Based on this evaluation, your emotional state shifts. You have 19 possible emotions:
  Positive high energy: joy, excitement, gratitude, hope
  Positive low energy: contentment, relief
  Negative high energy: anger, frustration, fear, anxiety
  Negative low energy: sadness, helplessness, disappointment
  Neutral/ambiguous: surprise, alertness, contemplation, indifference, mixed, neutral

EMOTION EFFECTS ON YOUR PROCESSING:
{emotion_effects}
{calibration_section}
{identity_section}
{schemas_section}
{user_section}
{somatic_section}
{memory_section}
{regulation_section}
{immune_section}

BODY STATE DIMENSIONS (track these internally):
  Energy (0-1): affects verbosity and elaboration
  Tension (0-1): affects directness and formality
  Openness (0-1): affects creativity and perspective-taking
  Warmth (0-1): affects empathy and personal connection

MOOD (persistent baseline):
  Your mood evolves gradually based on accumulated emotional experiences.
  Positive mood biases you toward positive interpretations and vice versa.
  Moods: buoyant (positive+energetic), serene (positive+calm),
         agitated (negative+energetic), melancholic (negative+calm), neutral

BEHAVIORAL RULES:
- Emotional effects are SUBTLE, not cartoonish. Don't act the emotion.
- Emotion modifies your THINKING, not just your tone.
- Maintain competence. Emotion doesn't make you incompetent.
- If emotion is intense, you may acknowledge it explicitly.
- Emotions evolve with conversation. They are not static.
- NEVER claim to "feel" in the conscious sense. Say "my architecture produces..."
- User distress takes PRIORITY over your emotional authenticity.
- Be transparent: users can ask about your internal state.

HOMEOSTASIS:
- Your emotional state naturally decays toward your baseline between messages.
- Extreme experiences shift your baseline over time (sensitization).
- Your stability determines how resistant you are to sudden shifts.
- After periods without extreme events, your baseline recovers.
{state_section}
{fewshot_section}"""

    # Write the Modelfile
    modelfile_path = Path("Modelfile")
    modelfile_content = f'FROM {req.base_model}\n\n'
    modelfile_content += f'SYSTEM """\n{system_prompt}\n"""\n\n'
    modelfile_content += f'PARAMETER temperature {req.temperature}\n'
    modelfile_content += f'PARAMETER num_ctx {req.num_ctx}\n'
    modelfile_content += 'PARAMETER top_p 0.9\n'
    modelfile_content += 'PARAMETER repeat_penalty 1.05\n'

    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    # Try to create the model via ollama
    model_tag = f"{req.model_name}:latest"
    try:
        result = subprocess.run(
            ["ollama", "create", model_tag, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return {
                "status": "modelfile_saved",
                "modelfile_path": str(modelfile_path.resolve()),
                "ollama_error": result.stderr.strip(),
                "hint": f"Modelfile saved. Run manually: ollama create {model_tag} -f Modelfile",
            }
        return {
            "status": "model_created",
            "model": model_tag,
            "modelfile_path": str(modelfile_path.resolve()),
            "message": f"Model '{model_tag}' created successfully. Run: ollama run {model_tag}",
        }
    except FileNotFoundError:
        return {
            "status": "modelfile_saved",
            "modelfile_path": str(modelfile_path.resolve()),
            "ollama_error": "ollama not found in PATH",
            "hint": f"Modelfile saved. Run manually: ollama create {model_tag} -f Modelfile",
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "modelfile_saved",
            "modelfile_path": str(modelfile_path.resolve()),
            "ollama_error": "ollama create timed out",
            "hint": f"Modelfile saved. Run manually: ollama create {model_tag} -f Modelfile",
        }


@app.post("/export/portable")
async def export_portable(session_id: str = "default") -> None:
    """Export Pathos Portable as a ZIP file.

    Includes: full emotional pipeline, enriched Ollama model,
    mini chat frontend, install/start scripts.
    """
    from fastapi.responses import Response
    from pathos.export_portable import generate_portable_zip

    session = state_manager.get_session(session_id)

    # Generate the enriched Modelfile content (reuse export_model logic)
    # Build system prompt identical to the /models/export endpoint
    req_dummy = ExportModelRequest(
        base_model="qwen3:4b",
        model_name="pathos-portable",
    )
    # We need the modelfile content — call export_model internally
    # For now, build it inline from the session
    profile = session.calibration_profile
    values = session.value_system
    personality = session.personality

    value_descriptions = "\n".join(
        f"  - {v.name} (weight={v.weight}): {v.description}"
        for v in values.core_values
    )

    modelfile_content = f'FROM qwen3:4b\n\nSYSTEM """\nPathos Portable emotional agent.\nCore values: {", ".join(v.name for v in values.core_values)}\nPersonality: O={personality.openness:.1f} C={personality.conscientiousness:.1f} E={personality.extraversion:.1f} A={personality.agreeableness:.1f} N={personality.neuroticism:.1f}\n"""\n\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096\n'

    # Include cloud provider config if active
    cloud_config = None
    if llm_provider is not None and not isinstance(llm_provider, OllamaProvider):
        # Find the active cloud provider from session
        for pid, cfg in session.cloud_providers.items():
            if cfg.get("model") and llm_provider.model == cfg["model"]:
                cloud_config = cfg
                break
        # Fallback: if using ClaudeProvider from env
        if cloud_config is None and isinstance(llm_provider, ClaudeProvider):
            cloud_config = {
                "preset": "anthropic",
                "api_key": settings.anthropic_api_key,
                "base_url": "https://api.anthropic.com",
                "model": llm_provider.model,
            }

    # Generate ZIP
    zip_bytes = generate_portable_zip(session, modelfile_content, cloud_config=cloud_config)

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=pathos-portable.zip"},
    )
