"""Standalone Emotion Processor — Runs the full emotional pipeline WITHOUT an LLM.

This is the core engine for the Emotion API as a Service. It takes a text stimulus,
processes it through all emotional systems (appraisal, generation, regulation,
advanced systems), and returns a complete EmotionProcessResponse.

The processor uses keyword-based appraisal by default (no LLM needed), but can
optionally use LLM-enhanced appraisal if a provider is supplied.

Usage:
    processor = EmotionProcessor(state_manager)
    response = await processor.process(request)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pathos.engine.appraiser import appraise, appraise_lite
from pathos.engine.behavior import generate_behavior_modifier, generate_simple_behavior_modifier
from pathos.engine.calibration import apply_calibration
from pathos.engine.contagion import (
    compute_contagion_perturbation,
    detect_user_emotion,
    update_shadow_state,
)
from pathos.engine.external_signals import fuse_signals
from pathos.engine.generator import (
    compute_arousal,
    compute_certainty,
    compute_dominance,
    compute_intensity,
    compute_valence,
    generate_emotion,
)
from pathos.engine.homeostasis import regulate
from pathos.engine.immune import apply_immune_protection, update_immune_state
from pathos.engine.meta import generate_meta_emotion
from pathos.engine.metrics import coherence, continuity, proportionality, recovery
from pathos.engine.mood import compute_mood_congruence_bias
from pathos.engine.narrative import (
    apply_narrative_effects,
    check_coherence,
    decay_crisis_counter,
    detect_crisis,
    process_growth,
    update_narrative,
)
from pathos.engine.needs import compute_needs_amplification, update_needs
from pathos.engine.reappraisal import reappraise
from pathos.engine.regulation import RegulationResult
from pathos.engine.self_inquiry import check_self_inquiry
from pathos.engine.social import compute_social_modulation, update_user_model
from pathos.engine.somatic import compute_somatic_bias, evaluate_user_reaction
from pathos.models.emotion import PrimaryEmotion
from pathos.models.emotion_api import (
    EmotionAPIConfig,
    EmotionProcessRequest,
    EmotionProcessResponse,
    ExternalSignalContribution,
)
from pathos.models.personality import PersonalityProfile
from pathos.state.manager import SessionState, StateManager

if TYPE_CHECKING:
    from pathos.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class EmotionProcessor:
    """Standalone emotional pipeline processor.

    Runs the full Pathos emotional pipeline on a text stimulus without
    requiring an LLM. All 23+ emotional systems are supported.

    The processor manages sessions via a StateManager and returns
    EmotionProcessResponse objects with the complete emotional state.
    """

    def __init__(
        self,
        state_manager: StateManager,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            state_manager: Manages per-session emotional state.
            llm_provider: Optional LLM for enhanced appraisal. If None,
                          keyword-based appraisal is used (fast, no dependencies).
        """
        self._state_manager = state_manager
        self._llm_provider = llm_provider

    @property
    def state_manager(self) -> StateManager:
        return self._state_manager

    async def process(self, request: EmotionProcessRequest) -> EmotionProcessResponse:
        """Process a single stimulus through the full emotional pipeline.

        Args:
            request: The stimulus and configuration to process.

        Returns:
            Complete emotional response with state, metrics, and optional details.
        """
        t0 = time.perf_counter()
        config = request.config
        session = self._state_manager.get_session(request.session_id)

        # Apply personality override if provided (first request or reconfigure)
        if request.personality:
            self._apply_personality(session, request.personality)

        # Configure session modes from API config
        session.advanced_mode = config.advanced_mode

        session.turn_count += 1
        previous_state = session.emotional_state.model_copy(deep=True)

        # ── Step 0: Homeostasis ──
        if session.turn_count > 1:
            session.emotional_state = regulate(session.emotional_state, turns_elapsed=1)
            if session.advanced_mode:
                session.regulator.recover(session.personality.regulation_capacity_base)

        # ── Step 0b: Temporal pre-processing [ADVANCED] ──
        temporal_result = None
        if session.advanced_mode:
            temporal_result = session.temporal.process_pre_turn(request.stimulus)

        # ── Step 1: Appraisal ──
        if self._llm_provider is not None:
            try:
                appraisal_result = await appraise(
                    stimulus=request.stimulus,
                    value_system=session.value_system,
                    llm=self._llm_provider,
                    think=True,
                )
            except Exception:
                logger.warning("LLM appraisal failed, falling back to keyword-based")
                appraisal_result = appraise_lite(request.stimulus)
        else:
            appraisal_result = appraise_lite(request.stimulus)

        appraisal = appraisal_result.vector

        # Compute raw dimensions
        raw_valence = compute_valence(appraisal)
        raw_arousal = compute_arousal(appraisal)
        raw_dominance = compute_dominance(appraisal)
        raw_certainty = compute_certainty(appraisal)

        # ── Step 1b: Memory amplification ──
        amplification = await session.memory.check_amplification(
            request.stimulus, llm=self._llm_provider,
        )

        # ── Step 2: External signals ──
        external_contributions: list[ExternalSignalContribution] = []
        ext_v_mod = 0.0
        ext_a_mod = 0.0
        ext_d_mod = 0.0

        if request.external_signals:
            fused = fuse_signals(request.external_signals)
            ext_v_mod = fused.valence_modulation
            ext_a_mod = fused.arousal_modulation
            ext_d_mod = fused.dominance_modulation

            for contrib in fused.contributions:
                external_contributions.append(ExternalSignalContribution(
                    source=contrib.source,
                    valence_delta=contrib.valence_delta,
                    arousal_delta=contrib.arousal_delta,
                    dominance_delta=contrib.dominance_delta,
                    weight_applied=contrib.weight,
                ))

        # ── Steps 2b-2f: Advanced pre-emotion systems ──
        needs_amp = 0.0
        schema_hint: PrimaryEmotion | None = None
        schema_amp = 0.0
        social_v_mod = 0.0
        social_i_mod = 0.0
        contagion_v = 0.0
        contagion_a = 0.0
        somatic_bias = 0.0

        if session.advanced_mode:
            # 2b. Needs amplification
            needs_amp = compute_needs_amplification(session.needs, request.stimulus)

            # 2c. Schema priming
            schema_hint, schema_amp = session.schemas.check_priming(request.stimulus)

            # 2d. Social modulation
            social_v_mod, social_i_mod = compute_social_modulation(
                session.user_model, raw_valence,
            )

            # 2e. Emotion Contagion
            detected_v, detected_a, signal_str = detect_user_emotion(request.stimulus)
            session.shadow_state = update_shadow_state(
                session.shadow_state, detected_v, detected_a, signal_str,
            )
            contagion_v, contagion_a = compute_contagion_perturbation(
                session.shadow_state,
                session.emotional_state.valence,
                session.emotional_state.arousal,
                session.personality,
                session.user_model.rapport,
            )

            # 2f. Somatic Markers
            session.somatic_markers = evaluate_user_reaction(
                session.somatic_markers, detected_v, session.turn_count,
            )
            somatic_bias, _ = compute_somatic_bias(session.somatic_markers, request.stimulus)

        # ── Step 3: Mood congruence bias ──
        valence_bias, arousal_bias = compute_mood_congruence_bias(
            session.emotional_state.mood,
        )

        # ── Step 4: Raw intensity ──
        intensity_raw = compute_intensity(appraisal, raw_valence, raw_arousal)

        # ── Step 5: Emotion generation ──
        # Inject external signal modulation into social_valence_mod
        effective_hint = (
            schema_hint
            if schema_hint and not appraisal_result.emotion_hint
            else appraisal_result.emotion_hint
        )
        new_state = generate_emotion(
            appraisal=appraisal,
            current_state=session.emotional_state,
            stimulus=request.stimulus,
            amplification=amplification + schema_amp,
            emotion_hint=effective_hint,
            dynamics=session.dynamics if session.advanced_mode else None,
            needs_amplification=needs_amp,
            social_valence_mod=social_v_mod + somatic_bias + ext_v_mod,
            social_intensity_mod=social_i_mod,
            contagion_valence=contagion_v,
            contagion_arousal=contagion_a + ext_a_mod,
            coupling=session.coupling if config.include_coupling and session.advanced_mode else None,
        )

        # 5b. Calibration
        new_state = apply_calibration(new_state, session.calibration_profile)

        # ── Steps 5c-5f: Advanced post-emotion systems ──
        regulation_result = RegulationResult()

        if session.advanced_mode:
            # 5c. Reappraisal
            new_state, _ = reappraise(new_state, session.regulator.regulation_capacity)

            # 5d. Active regulation
            new_state, regulation_result = session.regulator.regulate(
                new_state, session.personality.regulation_capacity_base,
            )

            # 5e. Temporal effects
            new_state = session.temporal.apply_temporal_effects(new_state, temporal_result)

            # 5e2. Emotional Immune System
            session.immune = update_immune_state(session.immune, new_state, request.stimulus)
            new_state = apply_immune_protection(new_state, session.immune, request.stimulus)

            # 5e3. Narrative Self
            coherence_delta, is_coherent = check_coherence(
                session.narrative, request.stimulus, new_state.primary_emotion,
            )
            new_state = apply_narrative_effects(
                new_state, coherence_delta, is_coherent, session.narrative.crisis.active,
            )
            session.narrative = detect_crisis(session.narrative, session.turn_count)

            # 5f. Meta-emotion
            is_new_emotion = new_state.primary_emotion != previous_state.primary_emotion
            meta_emotion = generate_meta_emotion(
                new_state, previous_state, session.value_system,
                regulation_success=(
                    regulation_result.strategy_used is not None
                    and not regulation_result.breakthrough
                ),
                is_new_emotion=is_new_emotion,
            )

            # 5f-b. Self-Initiated Inquiry
            check_self_inquiry(
                new_state, previous_state, meta_emotion,
                regulation_result, session.turn_count,
            )

        # ── Commit new state ──
        session.emotional_state = new_state
        session.state_history.append(new_state)
        if len(session.state_history) > 50:
            session.state_history = session.state_history[-50:]

        # ── Step 6: Post-processing updates ──
        await session.memory.store(request.stimulus, new_state, llm=self._llm_provider)

        if session.advanced_mode:
            session.needs = update_needs(session.needs, request.stimulus, appraisal)
            session.user_model = update_user_model(
                session.user_model, request.stimulus, appraisal, new_state,
            )
            session.schemas.record_pattern(
                request.stimulus, new_state.primary_emotion, new_state.intensity,
            )
            session.temporal.process_post_turn(request.stimulus, new_state, previous_state)

            # Narrative update + growth + decay
            session.narrative = update_narrative(
                session.narrative, session.narrative_tracker,
                request.stimulus, new_state.primary_emotion, new_state.intensity,
                session.turn_count,
            )
            session.narrative = process_growth(
                session.narrative, request.stimulus,
                previous_state.primary_emotion, new_state.primary_emotion,
                new_state.intensity,
                regulation_success=(
                    regulation_result.strategy_used is not None
                    and not regulation_result.breakthrough
                ),
                turn=session.turn_count,
            )
            session.narrative = decay_crisis_counter(session.narrative)

        # ── Step 7: Behavior modifier (optional) ──
        behavior_prompt: str | None = None
        if config.include_behavior_prompt:
            if session.advanced_mode:
                behavior_prompt = generate_behavior_modifier(new_state)
            else:
                behavior_prompt = generate_simple_behavior_modifier(new_state)

        # ── Step 8: Voice params (optional) ──
        voice_params: dict[str, object] | None = None
        if config.include_voice_params:
            try:
                from pathos.voice.params import generate_voice_params as gen_vp

                vp = gen_vp(new_state)
                voice_params = {
                    "speed": vp.speed,
                    "pitch_shift": vp.pitch_shift,
                    "voice_name": vp.voice_name,
                    "backend": vp.backend,
                    "stage_direction": vp.stage_direction or "",
                }
            except Exception:
                logger.debug("Voice params generation skipped (voice module not available)")

        # ── Step 9: Coupling contributions (optional) ──
        coupling_contribs: dict[str, float] | None = None
        if config.include_coupling and session.advanced_mode:
            # Attractors: mood baselines for V/A, 0.5 neutral for D/C
            mood = new_state.mood
            attr_v = mood.baseline_valence
            attr_a = mood.baseline_arousal
            dev_v = new_state.valence - attr_v
            dev_a = new_state.arousal - attr_a
            dev_d = new_state.dominance - 0.5
            dev_c = new_state.certainty - 0.5
            contrib = session.coupling.get_coupling_contribution(
                dev_v, dev_a, dev_d, dev_c,
            )
            coupling_contribs = {
                "valence": round(contrib[0], 6),
                "arousal": round(contrib[1], 6),
                "dominance": round(contrib[2], 6),
                "certainty": round(contrib[3], 6),
            }

        # ── Build pipeline trace (optional) ──
        pipeline_trace: dict[str, object] | None = None
        if config.include_pipeline_trace:
            pipeline_trace = {
                "appraisal_method": "llm" if self._llm_provider else "keyword",
                "raw_valence": round(raw_valence, 4),
                "raw_arousal": round(raw_arousal, 4),
                "raw_dominance": round(raw_dominance, 4),
                "raw_certainty": round(raw_certainty, 4),
                "intensity_raw": round(intensity_raw, 4),
                "amplification": round(amplification, 4),
                "advanced_mode": session.advanced_mode,
                "coupling_active": config.include_coupling and session.advanced_mode,
                "external_signals_count": len(request.external_signals),
                "external_valence_mod": round(ext_v_mod, 6),
                "external_arousal_mod": round(ext_a_mod, 6),
                "external_dominance_mod": round(ext_d_mod, 6),
            }

        # ── Build top emotions from stack ──
        top_emotions: dict[str, float] = {}
        if new_state.emotional_stack:
            sorted_stack = sorted(
                new_state.emotional_stack.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_emotions = {k: round(v, 4) for k, v in sorted_stack[:5]}

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return EmotionProcessResponse(
            session_id=request.session_id,
            turn_number=session.turn_count,
            emotional_state=new_state,
            primary_emotion=new_state.primary_emotion.value,
            secondary_emotion=(
                new_state.secondary_emotion.value if new_state.secondary_emotion else None
            ),
            intensity=new_state.intensity,
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominance=new_state.dominance,
            certainty=new_state.certainty,
            energy=new_state.body_state.energy,
            tension=new_state.body_state.tension,
            openness=new_state.body_state.openness,
            warmth=new_state.body_state.warmth,
            mood_label=new_state.mood.label.value,
            mood_trend=new_state.mood.trend,
            top_emotions=top_emotions,
            external_contributions=external_contributions,
            voice_params=voice_params,
            pipeline_trace=pipeline_trace,
            behavior_prompt=behavior_prompt,
            coupling_contributions=coupling_contribs,
            processing_time_ms=round(elapsed_ms, 2),
        )

    def _apply_personality(
        self, session: SessionState, overrides: dict[str, float],
    ) -> None:
        """Apply partial Big Five personality overrides to a session."""
        current = session.personality
        traits = {
            "openness": current.openness,
            "conscientiousness": current.conscientiousness,
            "extraversion": current.extraversion,
            "agreeableness": current.agreeableness,
            "neuroticism": current.neuroticism,
        }
        for key, value in overrides.items():
            if key in traits:
                traits[key] = max(0.0, min(1.0, value))

        new_personality = PersonalityProfile(**traits)
        session.update_personality(new_personality)
