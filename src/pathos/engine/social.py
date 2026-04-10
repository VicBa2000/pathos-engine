"""Social Cognition Engine - Theory of Mind computacional.

Modela al usuario como agente emocional:
- Infiere intención, engagement, estilo comunicativo
- Construye rapport asimétrico (crece lento, se rompe rápido)
- Modula respuestas emocionales según la relación

Un insulto de alto rapport → hurt/disappointment
Un insulto de bajo rapport → indifference/contempt
"""

from pathos.models.appraisal import AppraisalVector
from pathos.models.emotion import EmotionalState
from pathos.models.social import UserModel


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Señales de estilo comunicativo
_STYLE_INDICATORS: dict[str, list[str]] = {
    "formal": ["usted", "estimado", "cordialmente", "please", "sir", "ma'am", "regards"],
    "casual": ["jaja", "lol", "xd", "bro", "dude", "hey", "oye", "wey", "nah"],
    "technical": ["api", "function", "class", "debug", "error", "code", "deploy", "sql"],
    "emotional": ["siento", "triste", "feliz", "miedo", "love", "hate", "feel", "worry"],
}


def _detect_style(stimulus: str) -> str | None:
    """Detecta el estilo comunicativo del usuario."""
    lower = stimulus.lower()
    scores: dict[str, int] = {}
    for style, indicators in _STYLE_INDICATORS.items():
        scores[style] = sum(1 for ind in indicators if ind in lower)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else None


def _detect_intent(stimulus: str, appraisal: AppraisalVector) -> float:
    """Infiere la intención del usuario (-1=hostil, 1=benevolente).

    Usa el appraisal como proxy: si el estímulo tiene valence positiva y fairness alta,
    la intención es probablemente buena.
    """
    # Valence del estímulo indica tono
    valence_signal = appraisal.valence.intrinsic_pleasantness * 0.4
    # Fairness indica justicia percibida
    fairness_signal = appraisal.agency.fairness * 0.3
    # Intentionality: si es intencional y negativo, es más hostil
    intent_signal = appraisal.agency.intentionality * appraisal.valence.goal_conduciveness * 0.3

    return _clamp(valence_signal + fairness_signal + intent_signal, -1, 1)


def update_user_model(
    model: UserModel,
    stimulus: str,
    appraisal: AppraisalVector,
    system_state: EmotionalState,
) -> UserModel:
    """Actualiza el modelo del usuario basándose en la interacción actual.

    Args:
        model: Modelo de usuario actual.
        stimulus: Texto del usuario.
        appraisal: Resultado del appraisal.
        system_state: Estado emocional del sistema tras procesar.

    Returns:
        Modelo de usuario actualizado.
    """
    new_interaction_count = model.interaction_count + 1

    # 1. Detectar intención
    detected_intent = _detect_intent(stimulus, appraisal)
    # Smooth update (no cambiar drásticamente por un mensaje)
    new_intent = model.perceived_intent * 0.7 + detected_intent * 0.3

    # 2. Engagement: longitud del mensaje + significance como proxy
    msg_length_signal = min(len(stimulus) / 200, 1.0)
    significance_signal = appraisal.relevance.personal_significance
    detected_engagement = msg_length_signal * 0.4 + significance_signal * 0.6
    new_engagement = model.perceived_engagement * 0.6 + detected_engagement * 0.4

    # 3. Estilo comunicativo
    detected_style = _detect_style(stimulus)
    new_style = detected_style if detected_style else model.communication_style

    # 4. Rapport (asimétrico: crece lento, se rompe rápido)
    rapport_delta = 0.0
    if new_intent > 0.2:
        # Interacción positiva: rapport crece lento
        rapport_delta = 0.03 * new_engagement
    elif new_intent < -0.3:
        # Interacción negativa: rapport cae rápido (3x)
        rapport_delta = -0.09 * abs(new_intent)
    else:
        # Neutral: rapport decae muy lento
        rapport_delta = -0.005

    new_rapport = _clamp(model.rapport + rapport_delta, 0, 1)

    # 5. Trust (sigue al rapport pero con más inercia)
    trust_target = new_rapport * 0.6 + new_intent * 0.2 + 0.2
    new_trust = model.trust_level * 0.85 + trust_target * 0.15
    new_trust = _clamp(new_trust, 0, 1)

    # 6. Emotional reciprocity: ¿el usuario responde a las emociones del sistema?
    # Si el sistema tenía una emoción fuerte y el usuario parece responder a ella,
    # reciprocity sube
    if system_state.intensity > 0.5 and new_engagement > 0.5:
        new_reciprocity = _clamp(model.emotional_reciprocity + 0.05, 0, 1)
    else:
        new_reciprocity = _clamp(model.emotional_reciprocity - 0.01, 0, 1)

    # 7. Trust trajectory
    trajectory = list(model.trust_trajectory)
    trajectory.append(round(new_trust, 4))
    if len(trajectory) > 10:
        trajectory = trajectory[-10:]

    return UserModel(
        perceived_intent=round(_clamp(new_intent, -1, 1), 4),
        perceived_engagement=round(_clamp(new_engagement, 0, 1), 4),
        rapport=round(new_rapport, 4),
        communication_style=new_style,
        emotional_reciprocity=round(new_reciprocity, 4),
        trust_level=round(new_trust, 4),
        interaction_count=new_interaction_count,
        trust_trajectory=trajectory,
    )


def compute_social_modulation(
    user_model: UserModel,
    valence: float,
) -> tuple[float, float]:
    """Calcula cómo la relación modula la respuesta emocional.

    Returns:
        (valence_modulation, intensity_modulation)
        Valores a sumar/multiplicar sobre el estado generado.
    """
    # Alto rapport + estímulo negativo → más hurt (intensifica negatividad)
    # Bajo rapport + estímulo negativo → más indiferencia (atenúa)
    if valence < -0.2:
        # Estímulo negativo
        if user_model.rapport > 0.6:
            # Alto rapport: duele más (traición)
            intensity_mod = user_model.rapport * 0.2
            valence_mod = -0.05 * user_model.rapport  # Más negativo
        else:
            # Bajo rapport: importa menos
            intensity_mod = -(1 - user_model.rapport) * 0.15
            valence_mod = (1 - user_model.rapport) * 0.05  # Menos negativo
    elif valence > 0.2:
        # Estímulo positivo
        # Alto rapport amplifica lo positivo
        intensity_mod = user_model.rapport * 0.15
        valence_mod = user_model.rapport * 0.05
    else:
        intensity_mod = 0.0
        valence_mod = 0.0

    # Trust modula la interpretación general
    # Baja confianza → más defensivo (arousal sube implícitamente)
    if user_model.trust_level < 0.3:
        valence_mod -= 0.05  # Más negativo

    return (round(valence_mod, 4), round(intensity_mod, 4))
