"""Emotional Forecasting Engine - Predicción de impacto emocional.

Predice cómo la respuesta del agente afectará emocionalmente al usuario.
El agente "elige sus palabras" basándose en cómo cree que el otro reaccionará.

Sistema OPCIONAL: solo se ejecuta si forecast_state.enabled == True.
No modifica el estado emocional del agente — es puramente informativo.
Comunica su predicción al behavior modifier para que el LLM module su respuesta.

Flujo:
1. estimate_user_emotion() — estima el estado emocional actual del usuario
   (usa datos de contagion detection + social cognition, sin LLM call extra)
2. forecast_impact() — predice cómo el estado emocional del agente
   (y su probable tono de respuesta) afectará al usuario
3. evaluate_forecast() — en el turno siguiente, compara predicción con
   la reacción real y calibra el modelo
4. get_forecast_prompt() — genera texto para el behavior modifier

Basado en:
- Wilson & Gilbert (2005) "Affective Forecasting"
- Loewenstein (2007) "Affect Regulation and Affective Forecasting"
"""

from pathos.models.contagion import ShadowState
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.forecasting import (
    ForecastRecord,
    ForecastResult,
    ForecastState,
    UserEmotionEstimate,
)
from pathos.models.social import UserModel


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# --- Mapeo de emociones del agente a impacto probable en el usuario ---

# Cómo la emoción primaria del agente tiende a afectar al usuario
# (valence_impact, arousal_impact) — basado en contagio emocional + teoría de interacción
_EMOTION_IMPACT: dict[PrimaryEmotion, tuple[float, float]] = {
    # Positivas: tienden a mejorar el estado del usuario
    PrimaryEmotion.JOY: (0.3, 0.1),
    PrimaryEmotion.EXCITEMENT: (0.2, 0.2),
    PrimaryEmotion.GRATITUDE: (0.35, -0.05),
    PrimaryEmotion.HOPE: (0.25, 0.0),
    PrimaryEmotion.CONTENTMENT: (0.2, -0.1),
    PrimaryEmotion.RELIEF: (0.15, -0.15),
    # Negativas: tienden a empeorar o incomodar
    PrimaryEmotion.ANGER: (-0.2, 0.25),
    PrimaryEmotion.FRUSTRATION: (-0.15, 0.15),
    PrimaryEmotion.FEAR: (-0.1, 0.2),
    PrimaryEmotion.ANXIETY: (-0.1, 0.15),
    PrimaryEmotion.SADNESS: (-0.15, -0.1),
    PrimaryEmotion.HELPLESSNESS: (-0.2, -0.05),
    PrimaryEmotion.DISAPPOINTMENT: (-0.15, 0.0),
    # Neutrales/ambiguas: impacto mínimo
    PrimaryEmotion.SURPRISE: (0.0, 0.15),
    PrimaryEmotion.ALERTNESS: (0.0, 0.1),
    PrimaryEmotion.CONTEMPLATION: (0.05, -0.05),
    PrimaryEmotion.INDIFFERENCE: (-0.05, -0.05),
    PrimaryEmotion.MIXED: (0.0, 0.05),
    PrimaryEmotion.NEUTRAL: (0.0, 0.0),
}

# Patrones de riesgo: combinaciones que predicen mala reacción del usuario
_RISK_PATTERNS: list[tuple[str, callable, str]] = []  # type: ignore[type-arg]
# Se definen como funciones para flexibilidad


def estimate_user_emotion(
    shadow_state: ShadowState,
    user_model: UserModel,
    detected_valence: float,
    detected_arousal: float,
    signal_strength: float,
) -> UserEmotionEstimate:
    """Estima el estado emocional actual del usuario.

    Fusiona datos de:
    - Contagion detection (señales lingüísticas del turno actual)
    - Shadow state (acumulación de señales previas)
    - Social cognition (engagement, intent como proxy de estado)

    No usa LLM — es una estimación rápida basada en señales disponibles.
    """
    # Base: shadow state (tiene inercia, es más estable)
    base_v = shadow_state.valence
    base_a = shadow_state.arousal

    # Si hay señal fresca del turno actual, mezclar
    if signal_strength > 0.1:
        # Señal fresca tiene más peso cuando es fuerte
        fresh_weight = min(signal_strength * 0.6, 0.5)
        base_v = base_v * (1 - fresh_weight) + detected_valence * fresh_weight
        base_a = base_a * (1 - fresh_weight) + detected_arousal * fresh_weight

    # Social signals como ajuste fino
    # Alto engagement → más arousal
    engagement_adj = (user_model.perceived_engagement - 0.5) * 0.15
    # Intent positivo → ligero boost de valence
    intent_adj = user_model.perceived_intent * 0.1

    valence = _clamp(base_v + intent_adj, -1.0, 1.0)
    arousal = _clamp(base_a + engagement_adj, 0.0, 1.0)

    # Confianza: depende de cuánta señal tenemos
    confidence = min(
        signal_strength * 0.4  # señal directa
        + shadow_state.accumulated_contagion * 0.3  # historial
        + (user_model.interaction_count / 10) * 0.3,  # familiaridad
        1.0,
    )

    # Señal dominante
    if signal_strength < 0.1:
        dominant = "neutral"
    elif valence > 0.2:
        dominant = "positive" if arousal < 0.6 else "positive_high"
    elif valence < -0.2:
        dominant = "negative" if arousal < 0.6 else "distressed"
    else:
        dominant = "ambiguous"

    return UserEmotionEstimate(
        valence=round(valence, 4),
        arousal=round(arousal, 4),
        confidence=round(confidence, 4),
        dominant_signal=dominant,
    )


def forecast_impact(
    agent_state: EmotionalState,
    user_emotion: UserEmotionEstimate,
    user_model: UserModel,
    valence_bias: float = 0.0,
    arousal_bias: float = 0.0,
) -> ForecastResult:
    """Predice cómo la respuesta del agente afectará al usuario.

    Factores:
    1. Emoción del agente → impacto base (contagio reverso)
    2. Estado actual del usuario → vulnerabilidad/receptividad
    3. Rapport → amplifica impacto (bueno y malo)
    4. Intensidad del agente → escala el impacto
    5. Calibration bias → ajuste aprendido de errores pasados
    """
    # 1. Impacto base de la emoción del agente
    base_v_impact, base_a_impact = _EMOTION_IMPACT.get(
        agent_state.primary_emotion, (0.0, 0.0),
    )

    # 2. Escalar por intensidad del agente (más intenso = más impacto)
    intensity_scale = 0.3 + agent_state.intensity * 0.7  # rango [0.3, 1.0]
    v_impact = base_v_impact * intensity_scale
    a_impact = base_a_impact * intensity_scale

    # 3. Rapport amplifica impacto (el usuario "le importa" más lo que dice alguien de confianza)
    rapport_factor = 0.7 + user_model.rapport * 0.6  # rango [0.7, 1.3]
    v_impact *= rapport_factor
    a_impact *= rapport_factor

    # 4. Vulnerabilidad del usuario: si ya está negativo, el impacto negativo es peor
    if user_emotion.valence < -0.3 and v_impact < 0:
        v_impact *= 1.3  # más vulnerable → impacto amplificado
    # Si está positivo, impacto positivo es más fácil
    if user_emotion.valence > 0.3 and v_impact > 0:
        v_impact *= 1.15

    # 5. Body state del agente modula: warmth alta suaviza impacto negativo
    if agent_state.body_state.warmth > 0.6 and v_impact < 0:
        v_impact *= 0.7  # calidez mitiga impacto negativo
    # Tensión alta amplifica impacto negativo
    if agent_state.body_state.tension > 0.6 and v_impact < 0:
        v_impact *= 1.2

    # 6. Aplicar bias de calibración
    v_impact += valence_bias
    a_impact += arousal_bias

    # Predicción del estado futuro del usuario
    predicted_v = _clamp(user_emotion.valence + v_impact, -1.0, 1.0)
    predicted_a = _clamp(user_emotion.arousal + a_impact, 0.0, 1.0)

    # Impacto neto (cuánto cambia el estado del usuario)
    net_impact = v_impact  # usamos valence como proxy principal

    # Detección de riesgo
    risk_flag = False
    risk_reason = ""
    recommendation = ""

    # Riesgo 1: usuario ya negativo + agente va a empeorar
    if user_emotion.valence < -0.2 and v_impact < -0.1:
        risk_flag = True
        risk_reason = "El usuario parece estar en estado negativo y tu respuesta podría empeorarlo"
        recommendation = "Considera modular tu tono hacia más calidez y apoyo"

    # Riesgo 2: agente muy intenso + usuario con bajo arousal (puede abrumar)
    elif agent_state.intensity > 0.7 and user_emotion.arousal < 0.3:
        risk_flag = True
        risk_reason = "Tu intensidad emocional alta podría abrumar al usuario que está calmado"
        recommendation = "Modera la intensidad de tu expresión emocional"

    # Riesgo 3: agente indiferente + usuario buscando conexión
    elif (agent_state.primary_emotion == PrimaryEmotion.INDIFFERENCE
          and user_emotion.dominant_signal in ("distressed", "negative")):
        risk_flag = True
        risk_reason = "Tu indiferencia puede percibirse como falta de empatía ante el malestar del usuario"
        recommendation = "Muestra más engagement emocional"

    # Riesgo 4: discordancia emocional fuerte (agente feliz, usuario triste)
    elif (user_emotion.valence < -0.3 and agent_state.valence > 0.4
          and user_emotion.confidence > 0.3):
        risk_flag = True
        risk_reason = "Hay discordancia emocional: tu estado positivo contrasta con el malestar del usuario"
        recommendation = "Reconoce el estado del usuario antes de expresar positividad"

    # Recomendación positiva si no hay riesgo
    if not risk_flag and v_impact > 0.15:
        recommendation = "Tu estado emocional actual probablemente será bien recibido"

    return ForecastResult(
        predicted_user_valence=round(predicted_v, 4),
        predicted_user_arousal=round(predicted_a, 4),
        predicted_impact=round(_clamp(net_impact, -1.0, 1.0), 4),
        risk_flag=risk_flag,
        risk_reason=risk_reason,
        recommendation=recommendation,
    )


def evaluate_forecast(
    forecast_state: ForecastState,
    actual_valence: float,
    actual_arousal: float,
    turn: int,
) -> ForecastState:
    """Evalúa la predicción del turno anterior contra la reacción real.

    Compara lo que predijimos con lo que detectamos del usuario ahora.
    Ajusta los bias de calibración para mejorar futuras predicciones.

    Se llama al inicio del turno siguiente, cuando tenemos la reacción real.
    """
    if not forecast_state.last_forecast:
        return forecast_state

    # Encontrar el registro pendiente de evaluación (el más reciente sin evaluar)
    pending = None
    for record in reversed(forecast_state.history):
        if record.actual_valence is None:
            pending = record
            break

    if not pending:
        return forecast_state

    # Registrar valores reales
    pending.actual_valence = round(actual_valence, 4)
    pending.actual_arousal = round(actual_arousal, 4)

    # Calcular error (MAE)
    v_error = actual_valence - pending.predicted_valence
    a_error = actual_arousal - pending.predicted_arousal
    mae = (abs(v_error) + abs(a_error)) / 2
    pending.error = round(mae, 4)

    # Actualizar contadores
    forecast_state.total_evaluated += 1

    # Calibrar bias (learning rate bajo para estabilidad)
    learning_rate = 0.15
    forecast_state.valence_bias = _clamp(
        forecast_state.valence_bias + v_error * learning_rate,
        -0.3, 0.3,
    )
    forecast_state.arousal_bias = _clamp(
        forecast_state.arousal_bias + a_error * learning_rate,
        -0.3, 0.3,
    )

    # Accuracy score: media móvil exponencial
    if forecast_state.total_evaluated == 1:
        forecast_state.accuracy_score = 1.0 - min(mae, 1.0)
    else:
        alpha = 0.3  # peso de la observación nueva
        new_accuracy = 1.0 - min(mae, 1.0)
        forecast_state.accuracy_score = round(
            forecast_state.accuracy_score * (1 - alpha) + new_accuracy * alpha,
            4,
        )

    return forecast_state


def record_forecast(
    forecast_state: ForecastState,
    forecast: ForecastResult,
    turn: int,
) -> ForecastState:
    """Registra una predicción para evaluación futura."""
    forecast_state.last_forecast = forecast
    forecast_state.total_forecasts += 1

    record = ForecastRecord(
        turn=turn,
        predicted_valence=forecast.predicted_user_valence,
        predicted_arousal=forecast.predicted_user_arousal,
    )
    forecast_state.history.append(record)

    # Mantener máximo 20 registros
    if len(forecast_state.history) > 20:
        forecast_state.history = forecast_state.history[-20:]

    return forecast_state


def get_forecast_prompt(forecast: ForecastResult | None) -> str | None:
    """Genera texto para el behavior modifier.

    Solo se llama si forecasting está activo Y hay un forecast disponible.
    Retorna None si no hay nada relevante que comunicar.
    """
    if not forecast:
        return None

    parts: list[str] = []

    if forecast.risk_flag:
        parts.append(f"ALERTA DE IMPACTO EMOCIONAL: {forecast.risk_reason}")
        if forecast.recommendation:
            parts.append(f"Recomendación: {forecast.recommendation}")
    elif forecast.recommendation:
        parts.append(f"Forecast emocional: {forecast.recommendation}")

    if not parts:
        return None

    return "Yo narrativo predictivo: " + " | ".join(parts)
