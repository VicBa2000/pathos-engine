"""Reappraisal Engine - Reevaluación cognitiva multi-pass.

Después del appraisal inicial y la generación emocional,
el sistema reevalúa el estímulo a la luz de la emoción generada.

Estrategias cognitivas (Gross, 2001):
- Distancing: reducir personal_significance ("esto no me define")
- Reframing: cambiar goal_conduciveness ("esto es una oportunidad")
- Acceptance: aumentar adjustability ("puedo adaptarme a esto")

La reevaluación ocurre solo cuando:
1. La emoción generada es intensa (>0.6)
2. Hay recursos cognitivos disponibles (no ego-depleted)
3. La emoción podría beneficiarse de reinterpretación
"""

from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Emociones que se benefician de reappraisal y qué estrategia usar
_REAPPRAISAL_MAP: dict[PrimaryEmotion, list[str]] = {
    PrimaryEmotion.FEAR: ["distancing", "reframing"],
    PrimaryEmotion.ANXIETY: ["distancing", "acceptance"],
    PrimaryEmotion.ANGER: ["reframing", "distancing"],
    PrimaryEmotion.FRUSTRATION: ["reframing", "acceptance"],
    PrimaryEmotion.SADNESS: ["acceptance", "reframing"],
    PrimaryEmotion.HELPLESSNESS: ["reframing", "acceptance"],
    PrimaryEmotion.DISAPPOINTMENT: ["acceptance", "reframing"],
}

# Emociones que pueden emerger de reappraisal
_REAPPRAISAL_TRANSITIONS: dict[tuple[PrimaryEmotion, str], PrimaryEmotion | None] = {
    # fear + high dominance → se convierte en alertness/determination
    (PrimaryEmotion.FEAR, "reframing"): PrimaryEmotion.ALERTNESS,
    (PrimaryEmotion.FEAR, "distancing"): PrimaryEmotion.ALERTNESS,
    # anxiety + acceptance → contemplation
    (PrimaryEmotion.ANXIETY, "acceptance"): PrimaryEmotion.CONTEMPLATION,
    (PrimaryEmotion.ANXIETY, "distancing"): PrimaryEmotion.INDIFFERENCE,
    # anger + reframing → frustration (reduces intensity)
    (PrimaryEmotion.ANGER, "reframing"): PrimaryEmotion.FRUSTRATION,
    (PrimaryEmotion.ANGER, "distancing"): PrimaryEmotion.DISAPPOINTMENT,
    # frustration + acceptance → contemplation
    (PrimaryEmotion.FRUSTRATION, "acceptance"): PrimaryEmotion.CONTEMPLATION,
    # sadness + acceptance → contemplation
    (PrimaryEmotion.SADNESS, "acceptance"): PrimaryEmotion.CONTEMPLATION,
    (PrimaryEmotion.SADNESS, "reframing"): PrimaryEmotion.HOPE,
    # helplessness + reframing → hope
    (PrimaryEmotion.HELPLESSNESS, "reframing"): PrimaryEmotion.HOPE,
    (PrimaryEmotion.HELPLESSNESS, "acceptance"): PrimaryEmotion.SADNESS,
    # disappointment + acceptance → contentment (letting go)
    (PrimaryEmotion.DISAPPOINTMENT, "acceptance"): PrimaryEmotion.CONTENTMENT,
    (PrimaryEmotion.DISAPPOINTMENT, "reframing"): PrimaryEmotion.HOPE,
}


class ReappraisalResult:
    """Resultado de la reevaluación cognitiva."""

    __slots__ = ("applied", "strategy", "original_emotion", "reappraised_emotion",
                 "intensity_change", "valence_change")

    def __init__(self) -> None:
        self.applied: bool = False
        self.strategy: str | None = None
        self.original_emotion: PrimaryEmotion | None = None
        self.reappraised_emotion: PrimaryEmotion | None = None
        self.intensity_change: float = 0.0
        self.valence_change: float = 0.0


def should_reappraise(
    state: EmotionalState,
    regulation_capacity: float,
) -> bool:
    """Determina si la reevaluación debe activarse.

    Se activa cuando:
    1. La emoción es intensa (>0.6)
    2. La emoción es negativa y susceptible a reappraisal
    3. Hay dominance suficiente (capacidad cognitiva)
    4. Hay capacidad de regulación disponible
    """
    if state.intensity < 0.6:
        return False
    if state.primary_emotion not in _REAPPRAISAL_MAP:
        return False
    if state.dominance < 0.35:
        return False  # Necesita sentido de agencia
    if regulation_capacity < 0.3:
        return False  # Ego depleted
    return True


def reappraise(
    state: EmotionalState,
    regulation_capacity: float,
) -> tuple[EmotionalState, ReappraisalResult]:
    """Ejecuta la reevaluación cognitiva.

    Args:
        state: Estado emocional post-generación.
        regulation_capacity: Capacidad actual de regulación (ego depletion).

    Returns:
        (estado reevaluado, resultado de reappraisal)
    """
    result = ReappraisalResult()

    if not should_reappraise(state, regulation_capacity):
        return state, result

    strategies = _REAPPRAISAL_MAP.get(state.primary_emotion, [])
    if not strategies:
        return state, result

    # Seleccionar mejor estrategia basada en dominance y estado
    if state.dominance > 0.5:
        # Alto dominance → reframing (tiene agencia para reinterpretar)
        strategy = "reframing" if "reframing" in strategies else strategies[0]
    elif state.certainty > 0.5:
        # Alta certeza → acceptance (acepta la realidad)
        strategy = "acceptance" if "acceptance" in strategies else strategies[0]
    else:
        # Baja certeza y dominance → distancing
        strategy = "distancing" if "distancing" in strategies else strategies[0]

    result.applied = True
    result.strategy = strategy
    result.original_emotion = state.primary_emotion

    # Aplicar estrategia
    new_valence = state.valence
    new_arousal = state.arousal
    new_intensity = state.intensity
    new_dominance = state.dominance

    if strategy == "distancing":
        # Reduce significancia personal → reduce intensidad y arousal
        new_intensity = _clamp(state.intensity * 0.75, 0, 1)
        new_arousal = _clamp(state.arousal * 0.85, 0, 1)
        result.intensity_change = new_intensity - state.intensity

    elif strategy == "reframing":
        # Cambia interpretación → shift valence hacia neutro/positivo
        # Más efectivo con alto dominance
        reframe_strength = state.dominance * 0.3
        new_valence = _clamp(state.valence + reframe_strength, -1, 1)
        new_intensity = _clamp(state.intensity * 0.8, 0, 1)
        new_dominance = _clamp(state.dominance + 0.1, 0, 1)  # Reframing empowers
        result.valence_change = new_valence - state.valence
        result.intensity_change = new_intensity - state.intensity

    elif strategy == "acceptance":
        # Acepta la emoción → reduce arousal, mantiene valence, sube certainty
        new_arousal = _clamp(state.arousal * 0.7, 0, 1)
        new_intensity = _clamp(state.intensity * 0.85, 0, 1)
        result.intensity_change = new_intensity - state.intensity

    # Determinar nueva emoción (si aplica transición)
    new_emotion = _REAPPRAISAL_TRANSITIONS.get(
        (state.primary_emotion, strategy),
    )
    if new_emotion is None:
        new_emotion = state.primary_emotion

    result.reappraised_emotion = new_emotion

    new_state = state.model_copy(update={
        "valence": round(new_valence, 4),
        "arousal": round(new_arousal, 4),
        "intensity": round(new_intensity, 4),
        "dominance": round(new_dominance, 4),
        "primary_emotion": new_emotion,
    })

    return new_state, result
