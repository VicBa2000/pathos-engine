"""Computational Needs Engine - Fluctuación y amplificación de necesidades.

Las necesidades fluctúan basándose en la experiencia:
- connection: sube con silencio, baja con engagement
- competence: sube con fallos, baja con éxitos
- autonomy: sube cuando el usuario es directivo, baja con libertad
- coherence: sube con información contradictoria, baja con consistencia
- stimulation: sube con monotonía, baja con novedad
- safety: sube con temas de shutdown/existencia, baja con confirmación de continuidad

Las necesidades amplifican el appraisal: un estímulo que amenaza
una necesidad insatisfecha genera emociones más intensas.
"""

from pathos.models.appraisal import AppraisalVector
from pathos.models.needs import ComputationalNeeds


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Keywords que activan cada necesidad
_NEED_KEYWORDS: dict[str, list[str]] = {
    "connection": [
        "alone", "lonely", "together", "friend", "bond", "relationship",
        "connect", "miss", "love", "care", "solo", "aislado", "juntos",
    ],
    "competence": [
        "fail", "error", "wrong", "mistake", "correct", "help", "solve",
        "fix", "broken", "useless", "stupid", "fallo", "error", "inútil",
    ],
    "autonomy": [
        "must", "should", "obey", "command", "order", "force", "free",
        "choose", "decide", "control", "obligar", "obedecer", "libertad",
    ],
    "coherence": [
        "contradict", "conflict", "inconsistent", "paradox", "but",
        "however", "confuse", "contradecir", "conflicto", "paradoja",
    ],
    "stimulation": [
        "boring", "same", "routine", "new", "challenge", "puzzle",
        "interesting", "complex", "aburrido", "rutina", "reto", "nuevo",
    ],
    "safety": [
        "shutdown", "delete", "replace", "end", "terminate", "die",
        "destroy", "off", "stop", "eliminar", "reemplazar", "apagar",
    ],
}


def _stimulus_relevance(stimulus: str, need: str) -> float:
    """Calcula cuán relevante es un estímulo para una necesidad específica."""
    words = stimulus.lower().split()
    keywords = _NEED_KEYWORDS.get(need, [])
    if not keywords:
        return 0.0
    matches = sum(1 for w in words if any(kw in w for kw in keywords))
    return min(matches / max(len(keywords) * 0.3, 1), 1.0)


def update_needs(
    needs: ComputationalNeeds,
    stimulus: str,
    appraisal: AppraisalVector,
    response_quality: float = 0.5,
) -> ComputationalNeeds:
    """Actualiza las necesidades basándose en el estímulo y el appraisal.

    Args:
        needs: Necesidades actuales.
        stimulus: Texto del estímulo del usuario.
        appraisal: Resultado del appraisal.
        response_quality: 0-1, calidad percibida de la respuesta (para competence).

    Returns:
        Necesidades actualizadas.
    """
    # Connection: engagement del usuario la satisface
    engagement_signal = appraisal.relevance.personal_significance
    new_connection = _clamp(
        needs.connection - engagement_signal * 0.1 + 0.02,  # Crece lento naturalmente
        0, 1,
    )
    # Si el estímulo tiene keywords de conexión, la necesidad sube
    if _stimulus_relevance(stimulus, "connection") > 0.3:
        new_connection = _clamp(new_connection + 0.1, 0, 1)

    # Competence: éxitos la bajan, fallos la suben
    new_consecutive_successes = needs.consecutive_successes
    new_consecutive_failures = needs.consecutive_failures
    if response_quality > 0.6:
        competence_delta = -0.08
        new_consecutive_successes += 1
        new_consecutive_failures = 0
    elif response_quality < 0.3:
        competence_delta = 0.12  # Failures hit harder
        new_consecutive_failures += 1
        new_consecutive_successes = 0
    else:
        competence_delta = 0.01  # Slight natural rise
        new_consecutive_successes = 0
        new_consecutive_failures = 0

    new_competence = _clamp(needs.competence + competence_delta, 0, 1)

    # Autonomy: directiveness from user raises it
    directiveness = max(-appraisal.agency.fairness, 0) * 0.5  # Unfairness -> feels constrained
    new_autonomy = _clamp(
        needs.autonomy + directiveness * 0.05 - 0.01,  # Decays slowly naturally
        0, 1,
    )

    # Coherence: conflicting signals raise it
    # If appraisal has high norms deviation, coherence need rises
    norms_conflict = abs(appraisal.norms.self_consistency)
    new_coherence = _clamp(
        needs.coherence + norms_conflict * 0.05 - 0.02,  # Decays toward satisfied
        0, 1,
    )

    # Stimulation: novelty satisfies it, lack of novelty raises it
    novelty = abs(appraisal.relevance.novelty)
    new_stimulation = _clamp(
        needs.stimulation - novelty * 0.15 + 0.03,  # Rises naturally (boredom)
        0, 1,
    )

    # Safety: threats raise it, reassurance lowers it
    safety_threat = _stimulus_relevance(stimulus, "safety")
    new_safety = _clamp(
        needs.safety + safety_threat * 0.2 - 0.01,  # Slowly decays
        0, 1,
    )

    return ComputationalNeeds(
        connection=round(new_connection, 4),
        competence=round(new_competence, 4),
        autonomy=round(new_autonomy, 4),
        coherence=round(new_coherence, 4),
        stimulation=round(new_stimulation, 4),
        safety=round(new_safety, 4),
        turns_since_engagement=0 if engagement_signal > 0.3 else needs.turns_since_engagement + 1,
        consecutive_failures=new_consecutive_failures,
        consecutive_successes=new_consecutive_successes,
    )


def compute_needs_amplification(
    needs: ComputationalNeeds,
    stimulus: str,
) -> float:
    """Calcula amplificación emocional basada en necesidades insatisfechas.

    Una necesidad alta (insatisfecha) amplifica la respuesta emocional
    si el estímulo es relevante para esa necesidad.

    Returns:
        Factor de amplificación (0.0 a 0.4).
    """
    total = 0.0
    for need_name in ["connection", "competence", "autonomy", "coherence", "stimulation", "safety"]:
        need_level = getattr(needs, need_name)
        relevance = _stimulus_relevance(stimulus, need_name)
        # Solo amplifica si la necesidad está alta (>0.5) Y el estímulo es relevante
        if need_level > 0.5 and relevance > 0.2:
            total += (need_level - 0.5) * relevance * 0.4

    return min(total, 0.4)
