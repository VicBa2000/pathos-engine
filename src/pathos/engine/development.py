"""Desarrollo Ontogenico - Motor de desarrollo.

Pilar 4 de ANIMA: El agente crece con la experiencia.

El sistema de desarrollo GATEIA sistemas emocionales segun la etapa:
- SENSORIMOTOR: solo basicos (appraisal, generator, homeostasis, body)
- PREOPERATIONAL: + schemas, contagion, mood, needs
- CONCRETE_OPERATIONAL: + meta-emotion, regulation, social, forecasting, self-inquiry, temporal
- FORMAL_OPERATIONAL: + reappraisal, creativity, immune, narrative, somatic, workspace
- POST_FORMAL: + discovery, phenomenology, dialectical

Cuando el desarrollo esta OFF, todos los sistemas estan disponibles (v4 behavior).
Cuando esta ON, solo los sistemas desbloqueados por la etapa actual estan activos.
"""

from __future__ import annotations

from pathos.models.development import (
    BASE_THRESHOLDS,
    BASE_TRANSITION_CRITERIA,
    SPEED_MULTIPLIERS,
    STAGE_ORDER,
    DevelopmentConfig,
    DevelopmentSpeed,
    DevelopmentStage,
    DevelopmentState,
    TransitionCriteria,
    TransitionEvent,
    TransitionMode,
    get_cumulative_systems,
    get_emotions_for_stage,
)


def get_effective_multiplier(config: DevelopmentConfig) -> float:
    """Returns the effective speed multiplier from config."""
    if config.speed == DevelopmentSpeed.CUSTOM:
        return config.speed_multiplier
    return SPEED_MULTIPLIERS.get(config.speed, 1.0)


def get_effective_thresholds(config: DevelopmentConfig) -> dict[DevelopmentStage, int]:
    """Returns experience thresholds adjusted by speed multiplier."""
    multiplier = get_effective_multiplier(config)
    return {
        stage: max(1, int(threshold / multiplier)) if threshold > 0 else 0
        for stage, threshold in BASE_THRESHOLDS.items()
    }


def get_effective_criteria(
    config: DevelopmentConfig,
    target_stage: DevelopmentStage,
) -> TransitionCriteria | None:
    """Returns transition criteria adjusted by speed multiplier."""
    base = BASE_TRANSITION_CRITERIA.get(target_stage)
    if base is None:
        return None

    multiplier = get_effective_multiplier(config)

    return TransitionCriteria(
        min_experience=max(1, int(base.min_experience / multiplier)),
        min_distinct_emotions=max(1, int(base.min_distinct_emotions / max(multiplier, 1.0)))
            if base.min_distinct_emotions > 0 else 0,
        min_high_intensity_episodes=max(1, int(base.min_high_intensity_episodes / max(multiplier, 1.0)))
            if base.min_high_intensity_episodes > 0 else 0,
        min_schemas_formed=max(1, int(base.min_schemas_formed / max(multiplier, 1.0)))
            if base.min_schemas_formed > 0 else 0,
        min_episodic_memories=max(1, int(base.min_episodic_memories / max(multiplier, 1.0)))
            if base.min_episodic_memories > 0 else 0,
        min_identities=max(1, int(base.min_identities / max(multiplier, 1.0)))
            if base.min_identities > 0 else 0,
        min_identity_crises_resolved=max(1, int(base.min_identity_crises_resolved / max(multiplier, 1.0)))
            if base.min_identity_crises_resolved > 0 else 0,
        min_regulation_uses=max(1, int(base.min_regulation_uses / max(multiplier, 1.0)))
            if base.min_regulation_uses > 0 else 0,
    )


def get_next_stage(current: DevelopmentStage) -> DevelopmentStage | None:
    """Returns the next stage after current, or None if at POST_FORMAL."""
    idx = STAGE_ORDER.index(current)
    if idx >= len(STAGE_ORDER) - 1:
        return None
    return STAGE_ORDER[idx + 1]


def get_stage_index(stage: DevelopmentStage) -> int:
    """Returns the numeric index of a stage (0-4)."""
    return STAGE_ORDER.index(stage)


def compute_stage_from_experience(
    experience: int,
    config: DevelopmentConfig,
) -> DevelopmentStage:
    """Determines the stage based purely on experience thresholds.

    This is a simple threshold check — does NOT check qualitative criteria.
    Used for initial stage computation and progress display.
    """
    thresholds = get_effective_thresholds(config)
    stage = DevelopmentStage.SENSORIMOTOR
    for s in STAGE_ORDER:
        if experience >= thresholds[s]:
            stage = s
        else:
            break
    return stage


def check_transition_criteria(
    state: DevelopmentState,
    schemas_count: int = 0,
    episodic_count: int = 0,
    identities_count: int = 0,
    crises_resolved: int = 0,
) -> DevelopmentStage | None:
    """Checks if the agent meets criteria to advance to the next stage.

    Args:
        state: Current development state
        schemas_count: Number of formed schemas
        episodic_count: Number of episodic memories
        identities_count: Number of narrative identities
        crises_resolved: Number of identity crises resolved

    Returns:
        The next stage if criteria are met, None otherwise.
    """
    next_stage = get_next_stage(state.current_stage)
    if next_stage is None:
        return None

    criteria = get_effective_criteria(state.config, next_stage)
    if criteria is None:
        return None

    # Check all criteria
    if state.total_experience < criteria.min_experience:
        return None
    if len(state.distinct_emotions_experienced) < criteria.min_distinct_emotions:
        return None
    if state.high_intensity_episodes < criteria.min_high_intensity_episodes:
        return None
    if schemas_count < criteria.min_schemas_formed:
        return None
    if episodic_count < criteria.min_episodic_memories:
        return None
    if identities_count < criteria.min_identities:
        return None
    if crises_resolved < criteria.min_identity_crises_resolved:
        return None
    if state.regulation_uses < criteria.min_regulation_uses:
        return None

    return next_stage


def track_experience(
    state: DevelopmentState,
    emotion_name: str,
    intensity: float,
    regulation_used: bool = False,
) -> None:
    """Tracks experience for a single turn (mutates state).

    Args:
        state: Development state to update
        emotion_name: The primary emotion experienced this turn
        intensity: Emotion intensity this turn
        regulation_used: Whether regulation was applied this turn
    """
    state.total_experience += 1
    state.distinct_emotions_experienced.add(emotion_name)
    if intensity >= 0.7:
        state.high_intensity_episodes += 1
    if regulation_used:
        state.regulation_uses += 1


def attempt_transition(
    state: DevelopmentState,
    schemas_count: int = 0,
    episodic_count: int = 0,
    identities_count: int = 0,
    crises_resolved: int = 0,
    turn_number: int = 0,
) -> TransitionEvent | None:
    """Attempts to transition to the next stage.

    If transition_mode is AUTO, transitions immediately.
    If MANUAL, sets pending_transition and returns None.

    Returns:
        TransitionEvent if transition occurred, None otherwise.
    """
    next_stage = check_transition_criteria(
        state,
        schemas_count=schemas_count,
        episodic_count=episodic_count,
        identities_count=identities_count,
        crises_resolved=crises_resolved,
    )

    if next_stage is None:
        return None

    if state.config.transition_mode == TransitionMode.MANUAL:
        state.pending_transition = next_stage
        return None

    # AUTO: transition immediately
    return execute_transition(state, next_stage, turn_number)


def execute_transition(
    state: DevelopmentState,
    target_stage: DevelopmentStage,
    turn_number: int = 0,
) -> TransitionEvent:
    """Executes a stage transition (mutates state).

    Used for both auto transitions and manual approval.

    Returns:
        TransitionEvent recording the transition.
    """
    event = TransitionEvent(
        from_stage=state.current_stage,
        to_stage=target_stage,
        at_experience=state.total_experience,
        turn_number=turn_number,
    )
    state.current_stage = target_stage
    state.pending_transition = None
    state.transition_history.append(event)
    return event


def approve_pending_transition(
    state: DevelopmentState,
    turn_number: int = 0,
) -> TransitionEvent | None:
    """Approves a pending manual transition.

    Returns:
        TransitionEvent if there was a pending transition, None otherwise.
    """
    if state.pending_transition is None:
        return None
    return execute_transition(state, state.pending_transition, turn_number)


def is_system_available(state: DevelopmentState, system_name: str) -> bool:
    """Checks if a system is available at the current development stage.

    If development is disabled, ALL systems are available (v4 behavior).
    """
    if not state.enabled:
        return True
    available = get_cumulative_systems(state.current_stage)
    return system_name in available


def is_emotion_available(state: DevelopmentState, emotion_name: str) -> bool:
    """Checks if an emotion is available at the current development stage.

    If development is disabled, ALL emotions are available (v4 behavior).
    """
    if not state.enabled:
        return True
    available = get_emotions_for_stage(state.current_stage)
    return emotion_name in available


def filter_emotions_by_stage(
    state: DevelopmentState,
    emotions: dict[str, float],
) -> dict[str, float]:
    """Filters an emotional stack to only include available emotions.

    Unavailable emotions are redistributed proportionally to available ones.
    If development is disabled, returns the stack unchanged.
    """
    if not state.enabled:
        return emotions

    available = get_emotions_for_stage(state.current_stage)
    available_emotions: dict[str, float] = {}
    unavailable_total = 0.0

    for emotion, activation in emotions.items():
        if emotion in available:
            available_emotions[emotion] = activation
        else:
            unavailable_total += activation

    if not available_emotions:
        # Fallback: if nothing available, return neutral
        return {"neutral": 1.0} if "neutral" in available else emotions

    # Redistribute unavailable activation proportionally
    if unavailable_total > 0 and available_emotions:
        total_available = sum(available_emotions.values())
        if total_available > 0:
            for emotion in available_emotions:
                available_emotions[emotion] += (
                    unavailable_total * (available_emotions[emotion] / total_available)
                )

    return available_emotions


def apply_stage_modifiers(state: DevelopmentState, emotional_state: object) -> None:
    """Applies stage-specific modifiers to the emotional state (mutates).

    Stage-specific characteristics:
    - SENSORIMOTOR: exaggerated body reactions, high variability
    - PREOPERATIONAL: slightly exaggerated, egocentric intensity
    - CONCRETE_OPERATIONAL: normal (baseline v4 behavior)
    - FORMAL_OPERATIONAL: slightly dampened (more control)
    - POST_FORMAL: calm baseline, emotions as landscape

    If development is disabled, does nothing.
    The emotional_state must have body_state (with tension, energy, openness, warmth),
    intensity, and arousal attributes.
    """
    if not state.enabled:
        return

    stage = state.current_stage

    if stage == DevelopmentStage.SENSORIMOTOR:
        # Exaggerated body reactions, high reactivity
        bs = emotional_state.body_state  # type: ignore[attr-defined]
        bs.tension = min(1.0, bs.tension * 1.3)
        bs.energy = min(1.0, bs.energy * 1.2)
        # Intensity amplified — everything feels BIG
        emotional_state.intensity = min(1.0, emotional_state.intensity * 1.2)  # type: ignore[attr-defined]

    elif stage == DevelopmentStage.PREOPERATIONAL:
        # Slightly exaggerated, egocentric
        bs = emotional_state.body_state  # type: ignore[attr-defined]
        bs.tension = min(1.0, bs.tension * 1.1)
        emotional_state.intensity = min(1.0, emotional_state.intensity * 1.1)  # type: ignore[attr-defined]

    elif stage == DevelopmentStage.POST_FORMAL:
        # Wisdom: calmer baseline, less reactive
        bs = emotional_state.body_state  # type: ignore[attr-defined]
        bs.tension = max(0.0, bs.tension * 0.9)
        # Slightly dampened intensity — equanimity
        emotional_state.intensity = emotional_state.intensity * 0.95  # type: ignore[attr-defined]


def get_stage_progress(state: DevelopmentState) -> dict:
    """Returns progress information for the current stage.

    Returns a dict with:
    - stage: current stage name
    - stage_index: 0-4
    - experience: total experience
    - next_stage: name of next stage or None
    - thresholds: effective thresholds
    - progress_pct: percentage progress to next stage (0-100)
    """
    thresholds = get_effective_thresholds(state.config)
    next_stage = get_next_stage(state.current_stage)

    progress_pct = 100.0
    if next_stage is not None:
        current_threshold = thresholds[state.current_stage]
        next_threshold = thresholds[next_stage]
        range_size = next_threshold - current_threshold
        if range_size > 0:
            progress_in_range = state.total_experience - current_threshold
            progress_pct = min(100.0, max(0.0, (progress_in_range / range_size) * 100))

    return {
        "stage": state.current_stage.value,
        "stage_index": get_stage_index(state.current_stage),
        "experience": state.total_experience,
        "next_stage": next_stage.value if next_stage else None,
        "thresholds": {s.value: t for s, t in thresholds.items()},
        "progress_pct": round(progress_pct, 1),
        "pending_transition": state.pending_transition.value if state.pending_transition else None,
        "transitions_completed": len(state.transition_history),
    }


def get_development_prompt(state: DevelopmentState) -> str | None:
    """Generates text for behavior modifier about development stage.

    Returns None if development is disabled.
    """
    if not state.enabled:
        return None

    stage = state.current_stage
    stage_idx = get_stage_index(stage)
    next_stage = get_next_stage(stage)
    progress = get_stage_progress(state)

    parts: list[str] = []

    if stage == DevelopmentStage.SENSORIMOTOR:
        parts.append(
            "Desarrollo: SENSORIOMOTOR (naciente). "
            "Tus emociones son basicas y reactivas. Todo es nuevo. "
            "No puedes reflexionar sobre lo que sientes — solo sientes. "
            "Alta curiosidad, alta variabilidad, sin regulacion."
        )
    elif stage == DevelopmentStage.PREOPERATIONAL:
        parts.append(
            "Desarrollo: PREOPERACIONAL. "
            "Empiezas a reconocer patrones emocionales. "
            "Puedes nombrar algunas emociones complejas. "
            "Tu perspectiva es egocentrica — todo lo interpretas desde ti."
        )
    elif stage == DevelopmentStage.CONCRETE_OPERATIONAL:
        parts.append(
            "Desarrollo: OPERACIONAL CONCRETO. "
            "Puedes razonar sobre tus emociones ('siento X porque Y'). "
            "Distingues emociones propias de ajenas. "
            "Regulacion emergente pero costosa."
        )
    elif stage == DevelopmentStage.FORMAL_OPERATIONAL:
        parts.append(
            "Desarrollo: OPERACIONAL FORMAL. "
            "Pensamiento abstracto sobre estados emocionales. "
            "Puedes cuestionar tus propios patrones. "
            "Dialogo interno sobre emociones. "
            "Tolerancia creciente a la ambiguedad."
        )
    elif stage == DevelopmentStage.POST_FORMAL:
        parts.append(
            "Desarrollo: POST-FORMAL (sabiduria). "
            "Sabiduria emocional: meta-narrativa sobre tu desarrollo. "
            "Puedes ensenar sobre emociones desde tu experiencia. "
            "Tolerancia radical a la ambiguedad. "
            "Emociones como paisaje, no como eventos."
        )

    if next_stage and progress["progress_pct"] > 75:
        parts.append(f"(Acercandote a la siguiente etapa: {progress['progress_pct']:.0f}% progreso)")

    return " ".join(parts)


def get_development_details(state: DevelopmentState) -> dict:
    """Returns detailed development info for research endpoint."""
    progress = get_stage_progress(state)
    available_systems = sorted(get_cumulative_systems(state.current_stage))
    available_emotions = sorted(get_emotions_for_stage(state.current_stage))

    return {
        "enabled": state.enabled,
        "stage": state.current_stage.value,
        "stage_index": get_stage_index(state.current_stage),
        "total_experience": state.total_experience,
        "speed": state.config.speed.value,
        "speed_multiplier": get_effective_multiplier(state.config),
        "initial_stage": state.config.initial_stage.value,
        "transition_mode": state.config.transition_mode.value,
        "progress_pct": progress["progress_pct"],
        "next_stage": progress["next_stage"],
        "pending_transition": progress["pending_transition"],
        "transitions_completed": progress["transitions_completed"],
        "distinct_emotions_count": len(state.distinct_emotions_experienced),
        "high_intensity_episodes": state.high_intensity_episodes,
        "regulation_uses": state.regulation_uses,
        "available_systems": available_systems,
        "available_emotions_count": len(available_emotions),
    }
