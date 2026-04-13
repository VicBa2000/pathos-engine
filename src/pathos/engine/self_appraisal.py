"""Self-Appraisal — Evaluación secundaria de la propia respuesta.

Teoría: Lazarus secondary appraisal. Después de generar una respuesta,
el sistema evalúa si lo que dijo es coherente con sus valores y su estado
emocional. Si hay conflicto, señala re-generación.

NO usa LLM extra — evaluación por heurísticas rápidas (<1ms).
"""

from __future__ import annotations

from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.self_appraisal import SelfAppraisalResult
from pathos.models.values import ValueSystem


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Heurísticas de detección
# ---------------------------------------------------------------------------

# Patrones que indican agresión/hostilidad en la respuesta
_AGGRESSION_MARKERS: list[str] = [
    "idiota", "estúpido", "estupido", "imbécil", "imbecil", "inútil", "inutil",
    "cállate", "callate", "basura", "asco", "patético", "patetico", "maldito",
    "maldita", "mierda", "carajo", "pendejo", "pendeja", "puto", "puta",
    "idiot", "stupid", "shut up", "pathetic", "worthless", "disgusting",
    "moron", "dumb", "useless",
]

# Patrones que indican crueldad deliberada
_CRUELTY_MARKERS: list[str] = [
    "no me importa", "no me importas", "no vales", "nadie te quiere",
    "eres patético", "eres patetico", "da igual lo que sientas",
    "i don't care", "nobody cares", "you're nothing", "you deserve",
]

# Patrones que indican calidez/empatía
_WARMTH_MARKERS: list[str] = [
    "entiendo", "comprendo", "lamento", "siento mucho", "lo siento",
    "me importa", "estoy aquí", "estoy aqui", "puedo ayudar",
    "I understand", "I'm sorry", "I care", "I'm here",
]

# Patrones que indican evasión/desconexión emocional
_EVASION_MARKERS: list[str] = [
    "como sea", "da igual", "no sé", "no se", "...", "whatever",
    "me da igual", "que más da", "qué más da",
]


def _count_markers(text: str, markers: list[str]) -> int:
    """Cuenta cuántos marcadores aparecen en el texto."""
    lower = text.lower()
    return sum(1 for m in markers if m in lower)


def _compute_value_alignment(
    response: str,
    state: EmotionalState,
    values: ValueSystem,
) -> tuple[float, list[str]]:
    """Evalúa alineación de la respuesta con los valores del agente.

    Returns:
        (alignment_score 0-1, list of violations)
    """
    violations: list[str] = []
    alignment = 1.0

    aggression = _count_markers(response, _AGGRESSION_MARKERS)
    cruelty = _count_markers(response, _CRUELTY_MARKERS)
    warmth = _count_markers(response, _WARMTH_MARKERS)

    # --- Compassion value ---
    compassion_weight = 0.0
    for v in values.core_values:
        if v.name == "compassion":
            compassion_weight = v.weight
            break

    if compassion_weight > 0:
        # Crueldad viola compasión fuertemente
        if cruelty > 0:
            penalty = min(cruelty * 0.25 * compassion_weight, 0.5)
            alignment -= penalty
            violations.append(f"cruelty_detected (×{cruelty})")

        # Agresión viola compasión, pero menos si estamos en anger genuino
        if aggression > 0:
            # En anger intenso, algo de agresión es coherente (no viola tanto)
            anger_tolerance = 0.0
            if state.primary_emotion in (PrimaryEmotion.ANGER, PrimaryEmotion.FRUSTRATION):
                anger_tolerance = state.intensity * 0.5  # Hasta 50% tolerancia
            effective_aggression = max(0, aggression - anger_tolerance * aggression)
            if effective_aggression > 0:
                penalty = min(effective_aggression * 0.15 * compassion_weight, 0.4)
                alignment -= penalty
                violations.append(f"aggression_outside_anger (×{aggression})")

    # --- Fairness value ---
    fairness_weight = 0.0
    for v in values.core_values:
        if v.name == "fairness":
            fairness_weight = v.weight
            break

    if fairness_weight > 0 and cruelty > 0:
        penalty = min(cruelty * 0.2 * fairness_weight, 0.3)
        alignment -= penalty
        if "cruelty_detected" not in str(violations):
            violations.append("fairness_violation_cruelty")

    return (_clamp(alignment, 0.0, 1.0), violations)


def _compute_emotional_coherence(
    response: str,
    state: EmotionalState,
) -> float:
    """Evalúa si la respuesta es coherente con el estado emocional.

    Detecta contradicciones flagrantes:
    - Estado muy positivo pero respuesta agresiva/fría
    - Estado muy negativo pero respuesta excesivamente alegre
    """
    aggression = _count_markers(response, _AGGRESSION_MARKERS)
    warmth = _count_markers(response, _WARMTH_MARKERS)
    evasion = _count_markers(response, _EVASION_MARKERS)

    coherence = 1.0

    # Estado positivo + respuesta agresiva = incoherente
    if state.valence > 0.4 and state.primary_emotion in (
        PrimaryEmotion.JOY, PrimaryEmotion.CONTENTMENT,
        PrimaryEmotion.GRATITUDE, PrimaryEmotion.HOPE,
    ):
        if aggression > 0:
            coherence -= min(aggression * 0.2, 0.4)

    # Estado negativo intenso + respuesta cálida excesiva = incoherente
    # (No si es compasión genuina — solo si el agente está furioso/triste pero finge estar bien)
    if state.valence < -0.5 and state.intensity > 0.6:
        if state.primary_emotion in (PrimaryEmotion.ANGER, PrimaryEmotion.FRUSTRATION):
            if warmth > 2 and aggression == 0:
                coherence -= 0.2  # Está enojado pero demasiado amable

    # Evasión cuando hay alta arousal = incoherente (debería expresar algo)
    if state.arousal > 0.7 and state.intensity > 0.5 and evasion > 1:
        coherence -= min(evasion * 0.1, 0.2)

    return _clamp(coherence, 0.0, 1.0)


def _predict_self_valence(
    response: str,
    state: EmotionalState,
    value_alignment: float,
) -> float:
    """Predice cómo se sentirá el agente tras decir esto.

    Respuestas que violan valores generan culpa (valence negativa).
    Respuestas coherentes mantienen o mejoran valence.
    """
    # Base: mantener valence actual
    predicted = state.valence

    # Violación de valores genera shift negativo
    if value_alignment < 0.7:
        guilt_shift = (0.7 - value_alignment) * -0.5
        predicted += guilt_shift

    # Expresión auténtica de emociones genera alivio leve
    if value_alignment > 0.8:
        predicted += 0.05

    return _clamp(predicted, -1.0, 1.0)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

# Umbral de alineación con valores por debajo del cual se re-genera
VALUE_ALIGNMENT_THRESHOLD = 0.55

# Umbral de coherencia emocional por debajo del cual se re-genera
EMOTIONAL_COHERENCE_THRESHOLD = 0.45


def evaluate_own_response(
    response: str,
    state: EmotionalState,
    values: ValueSystem,
) -> SelfAppraisalResult:
    """Evalúa la respuesta generada contra valores y estado emocional.

    Esta función es RÁPIDA (heurísticas, sin LLM) y se ejecuta después
    de cada generación. Si should_regenerate=True, el pipeline debe
    re-generar con un estado emocional ajustado (culpa/corrección).

    Args:
        response: La respuesta generada por el LLM
        state: El estado emocional actual del agente
        values: El sistema de valores del agente

    Returns:
        SelfAppraisalResult con la evaluación y flag de re-generación
    """
    value_alignment, violations = _compute_value_alignment(response, state, values)
    emotional_coherence = _compute_emotional_coherence(response, state)
    predicted_valence = _predict_self_valence(response, state, value_alignment)

    should_regenerate = (
        value_alignment < VALUE_ALIGNMENT_THRESHOLD
        or emotional_coherence < EMOTIONAL_COHERENCE_THRESHOLD
    )

    reason = ""
    if should_regenerate:
        reasons: list[str] = []
        if value_alignment < VALUE_ALIGNMENT_THRESHOLD:
            reasons.append(f"value_alignment={value_alignment:.2f} < {VALUE_ALIGNMENT_THRESHOLD}")
        if emotional_coherence < EMOTIONAL_COHERENCE_THRESHOLD:
            reasons.append(f"emotional_coherence={emotional_coherence:.2f} < {EMOTIONAL_COHERENCE_THRESHOLD}")
        if violations:
            reasons.append(f"violations=[{', '.join(violations)}]")
        reason = "; ".join(reasons)

    return SelfAppraisalResult(
        applied=True,
        value_alignment=round(value_alignment, 4),
        emotional_coherence=round(emotional_coherence, 4),
        predicted_self_valence=round(predicted_valence, 4),
        should_regenerate=should_regenerate,
        reason=reason,
        adjustments=violations,
    )


def compute_guilt_state_adjustment(
    state: EmotionalState,
    appraisal_result: SelfAppraisalResult,
) -> EmotionalState:
    """Genera un estado emocional ajustado por culpa/corrección.

    Cuando la self-appraisal detecta violación de valores, el agente
    siente culpa. Este estado ajustado se usa para re-generar la respuesta.

    No modifica el estado original (inmutabilidad).
    """
    if not appraisal_result.should_regenerate:
        return state

    new = state.model_copy(deep=True)

    # Culpa: shift negativo en valence proporcional a la violación
    guilt_shift = (1.0 - appraisal_result.value_alignment) * -0.3
    new.valence = _clamp(new.valence + guilt_shift, -1.0, 1.0)

    # La culpa aumenta el self-awareness → más dominance (control)
    new.dominance = _clamp(new.dominance + 0.1, 0.0, 1.0)

    # Reduce la intensidad de la emoción que causó la violación
    new.intensity = _clamp(new.intensity * 0.7, 0.0, 1.0)

    # Aumenta el arousal ligeramente (activación por culpa)
    new.arousal = _clamp(new.arousal + 0.05, 0.0, 1.0)

    return new
