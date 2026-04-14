"""Predictive Processing Engine - Pilar 1 de ANIMA.

Antes de cada turno, el sistema genera 3 predicciones:
  1. Contenido probable del mensaje (temática, intención, tono)
  2. Estado emocional probable del usuario (valence, arousal)
  3. Demanda social probable (pide ayuda, quiere hablar, desafía, etc.)

Después de recibir el input, computa el prediction error en cada dimensión.
El error se transforma en contribución emocional:
  - Surprise positiva (mejor de lo esperado) → JOY, RELIEF, GRATITUDE
  - Surprise negativa (peor de lo esperado) → DISAPPOINTMENT, FEAR, ANGER
  - Surprise neutra (inesperado pero no valorado) → CURIOSITY, SURPRISE
  - Sin surprise (predicción correcta) → CONTENTMENT, satisfaction sutil

CLAVE: La precisión (confidence) crea VULNERABILIDAD EMOCIONAL.
Cuanto más seguro estás de tu predicción, más duele estar equivocado.

Sistema CORE (siempre activo): en cold start degrada a 100% appraisal clásico.

Basado en:
- Karl Friston (Free Energy Principle, 2010)
- Lisa Feldman Barrett (Theory of Constructed Emotion, 2017)
- Andy Clark (Surfing Uncertainty, 2015)
"""

from __future__ import annotations

import math

from pathos.models.emotion import EmotionalState, Mood, PrimaryEmotion
from pathos.models.predictive import (
    ContentPrediction,
    DemandPrediction,
    DemandType,
    EmotionPrediction,
    PredictionError,
    PredictionHistory,
    PredictionRecord,
    PredictionSet,
    PredictiveState,
    SurpriseType,
)
from pathos.models.social import UserModel


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Mapeos de intent y demanda
# ---------------------------------------------------------------------------

# Señales textuales para detectar intención del mensaje
_INTENT_SIGNALS: dict[str, list[str]] = {
    "question": ["?", "cómo", "qué", "por qué", "cuándo", "dónde", "how", "what", "why", "when", "where"],
    "request": ["haz", "hazme", "necesito", "quiero", "podrías", "please", "can you", "could you", "make", "do"],
    "complaint": ["mal", "error", "fallo", "no funciona", "broken", "wrong", "bug", "terrible"],
    "greeting": ["hola", "hey", "buenas", "hello", "hi", "buenos días", "good morning"],
    "farewell": ["adiós", "bye", "chao", "hasta luego", "nos vemos", "goodbye", "see you"],
    "emotional_expression": ["siento", "me siento", "estoy triste", "estoy feliz", "i feel", "i'm feeling"],
    "topic_change": ["por cierto", "cambiando de tema", "otra cosa", "by the way", "anyway", "btw"],
}

# Mapeo de intent → demanda social probable
_INTENT_TO_DEMAND: dict[str, DemandType] = {
    "question": DemandType.HELP,
    "request": DemandType.TASK,
    "complaint": DemandType.HELP,
    "greeting": DemandType.CONVERSATION,
    "farewell": DemandType.CONVERSATION,
    "emotional_expression": DemandType.EMOTIONAL,
    "topic_change": DemandType.CONVERSATION,
}

# Señales de tono
_POSITIVE_SIGNALS = [
    "gracias", "genial", "perfecto", "excelente", "bien", "bueno", "great",
    "thanks", "awesome", "perfect", "nice", "good", "love", "amazing",
    "jaja", "lol", "xd", ":)", "😊", "👍",
]
_NEGATIVE_SIGNALS = [
    "mal", "terrible", "odio", "peor", "horrible", "no sirve", "bad",
    "hate", "worst", "awful", "angry", "frustrated", "annoyed",
    ":(", "😠", "😡", "😢",
]


# ---------------------------------------------------------------------------
# Paso 1.1: generate_predictions
# ---------------------------------------------------------------------------

class PredictiveEngine:
    """Motor de predicción basado en Predictive Processing.

    Genera predicciones sobre el próximo turno del usuario usando:
    - Historial de conversación (últimos N turnos)
    - Modelo del usuario (Social Cognition)
    - Schemas emocionales activos
    - Mood actual y tendencia

    Es stateless: toda la información se recibe como parámetros y el estado
    predictivo se actualiza externamente en PredictiveState.
    """

    # Cuántos turnos de historial considerar
    HISTORY_WINDOW: int = 10

    # Umbral de error para considerar que la predicción fue correcta
    CORRECT_THRESHOLD: float = 0.2

    # Learning rate para actualización bayesiana de precisión
    PRECISION_LEARNING_RATE: float = 0.12

    # Decay de precisión por turno sin datos (previene sobreconfianza)
    PRECISION_DECAY_RATE: float = 0.005

    def generate_predictions(
        self,
        predictive_state: PredictiveState,
        conversation_history: list[dict[str, str]],
        user_model: UserModel,
        mood: Mood,
        emotional_state: EmotionalState,
        active_schemas: list[tuple[str, str, float]] | None = None,
    ) -> PredictionSet:
        """Genera predicciones para el próximo turno del usuario.

        Args:
            predictive_state: Estado del sistema predictivo.
            conversation_history: Historial de mensajes [{"role": ..., "content": ...}].
            user_model: Modelo actual del usuario (Social Cognition).
            mood: Mood actual del agente.
            emotional_state: Estado emocional actual del agente.
            active_schemas: Schemas activos [(category, emotion, strength), ...].

        Returns:
            PredictionSet con las 3 predicciones.
        """
        turn = len(conversation_history) // 2  # Turnos del usuario

        content = self._predict_content(conversation_history, user_model)
        emotion = self._predict_emotion(
            conversation_history, user_model, mood, emotional_state,
        )
        demand = self._predict_demand(conversation_history, user_model)

        # Ajustar confianza con la precisión bayesiana acumulada
        content = content.model_copy(update={
            "confidence": _clamp(
                content.confidence * 0.5 + predictive_state.content_precision * 0.5,
                0.0, 1.0,
            ),
        })
        emotion = emotion.model_copy(update={
            "confidence": _clamp(
                emotion.confidence * 0.5 + predictive_state.emotion_precision * 0.5,
                0.0, 1.0,
            ),
        })
        demand = demand.model_copy(update={
            "confidence": _clamp(
                demand.confidence * 0.5 + predictive_state.demand_precision * 0.5,
                0.0, 1.0,
            ),
        })

        # Si hay schemas activos, refinar predicciones
        if active_schemas:
            content, emotion = self._refine_with_schemas(
                content, emotion, active_schemas,
            )

        return PredictionSet(
            content=content,
            emotion=emotion,
            demand=demand,
            turn=turn,
        )

    # --- Predicción de contenido ---

    def _predict_content(
        self,
        history: list[dict[str, str]],
        user_model: UserModel,
    ) -> ContentPrediction:
        """Predice temática, tono e intención del próximo mensaje."""
        if not history:
            return ContentPrediction()  # Cold start: defaults

        # Extraer solo mensajes del usuario de la ventana reciente
        user_messages = [
            m["content"] for m in history[-self.HISTORY_WINDOW:]
            if m.get("role") == "user" and m.get("content")
        ]

        if not user_messages:
            return ContentPrediction()

        last_msg = user_messages[-1]

        # --- Predecir tono ---
        tone = self._detect_tone_tendency(user_messages)

        # --- Predecir intención ---
        intent = self._detect_intent_tendency(user_messages)

        # --- Predecir topic ---
        # Simple heuristic: si las últimas interacciones son del mismo tema,
        # predecir continuación; si hubo cambio reciente, predecir cambio
        if len(user_messages) >= 2:
            topic = "continuation"
        else:
            topic = "unknown"

        # Confianza base: crece con más historial
        base_confidence = min(len(user_messages) / 10, 0.7)

        # User model boost: si conocemos bien al usuario, más confianza
        familiarity_boost = min(user_model.interaction_count / 20, 0.2)

        confidence = _clamp(base_confidence + familiarity_boost, 0.1, 0.9)

        return ContentPrediction(
            expected_topic=topic,
            expected_tone=tone,
            expected_intent=intent,
            confidence=round(confidence, 4),
        )

    def _detect_tone_tendency(
        self, user_messages: list[str],
    ) -> str:
        """Detecta la tendencia de tono en los mensajes recientes."""
        if not user_messages:
            return "neutral"

        positive_count = 0
        negative_count = 0
        for msg in user_messages[-5:]:  # Últimos 5
            lower = msg.lower()
            positive_count += sum(1 for s in _POSITIVE_SIGNALS if s in lower)
            negative_count += sum(1 for s in _NEGATIVE_SIGNALS if s in lower)

        if positive_count > negative_count + 1:
            return "positive"
        elif negative_count > positive_count + 1:
            return "negative"
        elif positive_count > 0 and negative_count > 0:
            return "mixed"
        return "neutral"

    def _detect_intent_tendency(
        self, user_messages: list[str],
    ) -> str:
        """Detecta la intención probable basada en patrones recientes."""
        if not user_messages:
            return "unknown"

        last_msg = user_messages[-1].lower()

        # Priorizar por especificidad
        scores: dict[str, int] = {}
        for intent, signals in _INTENT_SIGNALS.items():
            scores[intent] = sum(1 for s in signals if s in last_msg)

        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best

        # Si no hay señal clara, predecir continuación de patrón
        if len(user_messages) >= 2:
            return "continue"
        return "unknown"

    # --- Predicción de emoción del usuario ---

    def _predict_emotion(
        self,
        history: list[dict[str, str]],
        user_model: UserModel,
        mood: Mood,
        emotional_state: EmotionalState,
    ) -> EmotionPrediction:
        """Predice el estado emocional probable del usuario."""
        if not history:
            return EmotionPrediction()  # Cold start

        # Base: user model como proxy del estado habitual
        base_valence = user_model.perceived_intent * 0.4  # Intent como proxy de valence
        base_arousal = user_model.perceived_engagement * 0.5 + 0.2  # Engagement como proxy

        # Ajuste por mood del agente: si el agente está negativo, el usuario
        # podría estar reaccionando a eso
        if emotional_state.valence < -0.3:
            # Nuestro estado negativo podría afectar al usuario (contagio reverso)
            base_valence -= 0.1
        elif emotional_state.valence > 0.3:
            base_valence += 0.05

        # Ajuste por rapport: alto rapport → user más emocionalmente conectado
        rapport_factor = user_model.rapport * 0.15
        base_arousal += rapport_factor

        # Ajuste por tendencia del mood
        if mood.trend == "declining":
            base_valence -= 0.05
        elif mood.trend == "improving":
            base_valence += 0.05

        # Confianza base
        base_confidence = 0.2 + min(user_model.interaction_count / 15, 0.5)

        return EmotionPrediction(
            expected_valence=round(_clamp(base_valence, -1, 1), 4),
            expected_arousal=round(_clamp(base_arousal, 0, 1), 4),
            confidence=round(_clamp(base_confidence, 0.1, 0.9), 4),
        )

    # --- Predicción de demanda social ---

    def _predict_demand(
        self,
        history: list[dict[str, str]],
        user_model: UserModel,
    ) -> DemandPrediction:
        """Predice qué espera el usuario de la interacción."""
        if not history:
            return DemandPrediction()  # Cold start

        user_messages = [
            m["content"] for m in history[-self.HISTORY_WINDOW:]
            if m.get("role") == "user" and m.get("content")
        ]

        if not user_messages:
            return DemandPrediction()

        # Detectar patrón de demanda recurrente
        demand_counts: dict[DemandType, int] = {d: 0 for d in DemandType}
        for msg in user_messages[-5:]:
            lower = msg.lower()
            for intent, signals in _INTENT_SIGNALS.items():
                if any(s in lower for s in signals):
                    demand = _INTENT_TO_DEMAND.get(intent, DemandType.UNKNOWN)
                    demand_counts[demand] += 1

        # La demanda más frecuente es la predicción
        best_demand = max(demand_counts, key=lambda d: demand_counts[d])
        if demand_counts[best_demand] == 0:
            best_demand = DemandType.CONVERSATION  # Default: quiere hablar

        # Ajustar por estilo del usuario
        if user_model.communication_style == "emotional":
            # Usuarios emocionales tienden a buscar apoyo
            if best_demand == DemandType.CONVERSATION:
                best_demand = DemandType.EMOTIONAL

        confidence = 0.2 + min(len(user_messages) / 8, 0.5)

        return DemandPrediction(
            expected_demand=best_demand,
            confidence=round(_clamp(confidence, 0.1, 0.9), 4),
        )

    # --- Refinamiento con schemas ---

    def _refine_with_schemas(
        self,
        content: ContentPrediction,
        emotion: EmotionPrediction,
        schemas: list[tuple[str, str, float]],
    ) -> tuple[ContentPrediction, EmotionPrediction]:
        """Refina predicciones usando schemas emocionales activos.

        Los schemas indican patrones aprendidos: si un schema de "crítica"
        está activo con fuerza alta, predecir que el tono será más negativo.
        """
        if not schemas:
            return content, emotion

        # Buscar el schema más fuerte
        strongest = max(schemas, key=lambda s: s[2])
        category, emotion_name, strength = strongest

        if strength < 0.3:
            return content, emotion  # Schema débil, no modifica

        # Schema fuerte modifica la predicción emocional
        schema_valence_shift = 0.0
        if emotion_name in ("anger", "frustration", "fear", "anxiety", "sadness"):
            schema_valence_shift = -strength * 0.2
        elif emotion_name in ("joy", "excitement", "gratitude", "hope"):
            schema_valence_shift = strength * 0.15

        new_emotion = emotion.model_copy(update={
            "expected_valence": _clamp(
                emotion.expected_valence + schema_valence_shift, -1, 1,
            ),
            "confidence": _clamp(emotion.confidence + strength * 0.1, 0, 1),
        })

        # Schema también sugiere tono
        if schema_valence_shift < -0.1 and content.expected_tone == "neutral":
            new_content = content.model_copy(update={"expected_tone": "negative"})
        elif schema_valence_shift > 0.1 and content.expected_tone == "neutral":
            new_content = content.model_copy(update={"expected_tone": "positive"})
        else:
            new_content = content

        return new_content, new_emotion


# ---------------------------------------------------------------------------
# Paso 1.2: compute_prediction_error
# ---------------------------------------------------------------------------

def compute_prediction_error(
    predictions: PredictionSet,
    actual_stimulus: str,
    detected_user_valence: float,
    detected_user_arousal: float,
    detected_intent: str,
    detected_demand: DemandType | None = None,
) -> PredictionError:
    """Computa el error de predicción tras recibir el input real.

    Compara cada dimensión predicha con la observación real:
    - Content: tono e intención predichos vs detectados
    - Emotion: valence/arousal predichos vs detectados del usuario
    - Demand: tipo de demanda predicha vs inferida

    Args:
        predictions: Predicciones generadas antes del input.
        actual_stimulus: Texto real del usuario.
        detected_user_valence: Valence detectada del usuario (post-appraisal).
        detected_user_arousal: Arousal detectado del usuario.
        detected_intent: Intención detectada del mensaje.
        detected_demand: Demanda social inferida.

    Returns:
        PredictionError con magnitudes por dimensión y tipo de sorpresa.
    """
    # --- Content error: tono e intención ---
    tone_error = _compute_tone_error(predictions.content.expected_tone, actual_stimulus)
    intent_error = _compute_intent_error(predictions.content.expected_intent, detected_intent)
    content_error = (tone_error * 0.6 + intent_error * 0.4)

    # --- Emotion error: distancia euclidiana en espacio valence-arousal ---
    v_diff = detected_user_valence - predictions.emotion.expected_valence
    a_diff = detected_user_arousal - predictions.emotion.expected_arousal
    emotion_error = min(math.sqrt(v_diff ** 2 + a_diff ** 2) / math.sqrt(2), 1.0)

    # --- Demand error ---
    if detected_demand is not None:
        demand_error = 0.0 if detected_demand == predictions.demand.expected_demand else 0.7
    else:
        demand_error = 0.3  # Sin datos, error moderado

    # --- Error total ponderado por precisión de cada dimensión ---
    # Dimensiones con mayor precisión pesan más (porque las conocemos mejor)
    w_content = predictions.content.confidence
    w_emotion = predictions.emotion.confidence
    w_demand = predictions.demand.confidence
    w_total = w_content + w_emotion + w_demand

    if w_total > 0:
        total_error = (
            content_error * w_content
            + emotion_error * w_emotion
            + demand_error * w_demand
        ) / w_total
    else:
        total_error = (content_error + emotion_error + demand_error) / 3

    # --- Dirección del error en valence (signo indica mejor/peor) ---
    valence_direction = _clamp(v_diff, -1, 1)

    # --- Tipo de sorpresa ---
    surprise_type = _classify_surprise(
        total_error, valence_direction, predictions.average_confidence,
    )

    # --- Vulnerabilidad emocional ---
    # Alta precisión + alto error = alta vulnerabilidad
    vulnerability = predictions.average_confidence * total_error

    return PredictionError(
        content_error=round(content_error, 4),
        emotion_error=round(emotion_error, 4),
        demand_error=round(demand_error, 4),
        total_error=round(_clamp(total_error, 0, 1), 4),
        valence_direction=round(valence_direction, 4),
        surprise_type=surprise_type,
        vulnerability=round(_clamp(vulnerability, 0, 1), 4),
    )


def _compute_tone_error(predicted_tone: str, actual_stimulus: str) -> float:
    """Computa error entre tono predicho y señales reales del estímulo."""
    lower = actual_stimulus.lower()
    pos_score = sum(1 for s in _POSITIVE_SIGNALS if s in lower)
    neg_score = sum(1 for s in _NEGATIVE_SIGNALS if s in lower)

    if pos_score > neg_score + 1:
        actual_tone = "positive"
    elif neg_score > pos_score + 1:
        actual_tone = "negative"
    elif pos_score > 0 and neg_score > 0:
        actual_tone = "mixed"
    else:
        actual_tone = "neutral"

    if actual_tone == predicted_tone:
        return 0.0

    # Errores parciales (predecir neutral cuando es mixed no es tan grave)
    _TONE_DISTANCE: dict[tuple[str, str], float] = {
        ("positive", "negative"): 1.0,
        ("negative", "positive"): 1.0,
        ("positive", "neutral"): 0.3,
        ("neutral", "positive"): 0.3,
        ("negative", "neutral"): 0.3,
        ("neutral", "negative"): 0.3,
        ("mixed", "neutral"): 0.2,
        ("neutral", "mixed"): 0.2,
        ("positive", "mixed"): 0.4,
        ("mixed", "positive"): 0.4,
        ("negative", "mixed"): 0.4,
        ("mixed", "negative"): 0.4,
    }
    return _TONE_DISTANCE.get((predicted_tone, actual_tone), 0.5)


def _compute_intent_error(predicted_intent: str, detected_intent: str) -> float:
    """Computa error entre intención predicha y detectada."""
    if predicted_intent == detected_intent:
        return 0.0
    if predicted_intent == "unknown" or detected_intent == "unknown":
        return 0.3  # Sin datos suficientes
    if predicted_intent == "continue":
        return 0.2  # "Continue" es un fallback razonable

    # Intenciones similares tienen menos error
    _SIMILAR_INTENTS = {
        ("question", "request"): 0.3,
        ("request", "question"): 0.3,
        ("greeting", "conversation"): 0.2,
        ("farewell", "conversation"): 0.3,
        ("complaint", "emotional_expression"): 0.4,
        ("emotional_expression", "complaint"): 0.4,
    }
    return _SIMILAR_INTENTS.get((predicted_intent, detected_intent), 0.6)


def _classify_surprise(
    total_error: float,
    valence_direction: float,
    precision: float,
) -> SurpriseType:
    """Clasifica el tipo de sorpresa basado en error y dirección."""
    # Umbral adaptativo: alta precisión → umbral más bajo (más sensible)
    threshold = 0.25 - precision * 0.1  # Rango [0.15, 0.25]

    if total_error < threshold:
        return SurpriseType.NONE  # Predicción correcta

    if abs(valence_direction) < 0.1:
        return SurpriseType.NEUTRAL  # Inesperado pero no valorado

    if valence_direction > 0:
        return SurpriseType.POSITIVE  # Mejor de lo esperado
    return SurpriseType.NEGATIVE  # Peor de lo esperado


# ---------------------------------------------------------------------------
# Paso 1.3: Precisión bayesiana
# ---------------------------------------------------------------------------

def update_precision(
    predictive_state: PredictiveState,
    error: PredictionError,
) -> PredictiveState:
    """Actualiza la precisión bayesiana por dimensión.

    Si el error fue bajo → precisión sube (refuerzo).
    Si el error fue alto → precisión baja (debilitamiento).

    La precisión es per-dimension: el agente puede ser bueno prediciendo
    contenido pero malo prediciendo emociones.
    """
    lr = PredictiveEngine.PRECISION_LEARNING_RATE
    threshold = PredictiveEngine.CORRECT_THRESHOLD

    # Content precision
    if error.content_error < threshold:
        predictive_state.content_precision += lr * (1 - predictive_state.content_precision)
    else:
        predictive_state.content_precision -= lr * predictive_state.content_precision * error.content_error

    # Emotion precision
    if error.emotion_error < threshold:
        predictive_state.emotion_precision += lr * (1 - predictive_state.emotion_precision)
    else:
        predictive_state.emotion_precision -= lr * predictive_state.emotion_precision * error.emotion_error

    # Demand precision
    if error.demand_error < threshold:
        predictive_state.demand_precision += lr * (1 - predictive_state.demand_precision)
    else:
        predictive_state.demand_precision -= lr * predictive_state.demand_precision * error.demand_error

    # Clamp
    predictive_state.content_precision = round(
        _clamp(predictive_state.content_precision, 0.05, 0.95), 4,
    )
    predictive_state.emotion_precision = round(
        _clamp(predictive_state.emotion_precision, 0.05, 0.95), 4,
    )
    predictive_state.demand_precision = round(
        _clamp(predictive_state.demand_precision, 0.05, 0.95), 4,
    )

    # Actualizar predictive_weight basado en precisión promedio
    avg_precision = predictive_state.average_precision
    # El peso crece linealmente con la precisión, hasta max_weight
    predictive_state.predictive_weight = round(
        avg_precision * predictive_state.max_predictive_weight, 4,
    )

    # Actualizar accuracy del historial
    predictive_state.history.content_accuracy = round(
        predictive_state.history.content_accuracy * 0.8
        + (1 - error.content_error) * 0.2, 4,
    )
    predictive_state.history.emotion_accuracy = round(
        predictive_state.history.emotion_accuracy * 0.8
        + (1 - error.emotion_error) * 0.2, 4,
    )
    predictive_state.history.demand_accuracy = round(
        predictive_state.history.demand_accuracy * 0.8
        + (1 - error.demand_error) * 0.2, 4,
    )

    return predictive_state


def decay_precision(predictive_state: PredictiveState, elapsed_turns: int = 1) -> PredictiveState:
    """Aplica decay temporal a la precisión.

    Previene sobreconfianza: el agente "olvida" su certeza gradualmente,
    no sus predicciones. Útil entre sesiones o tras pausas largas.
    """
    decay = PredictiveEngine.PRECISION_DECAY_RATE * elapsed_turns

    predictive_state.content_precision = round(
        _clamp(predictive_state.content_precision - decay, 0.05, 0.95), 4,
    )
    predictive_state.emotion_precision = round(
        _clamp(predictive_state.emotion_precision - decay, 0.05, 0.95), 4,
    )
    predictive_state.demand_precision = round(
        _clamp(predictive_state.demand_precision - decay, 0.05, 0.95), 4,
    )

    return predictive_state


def record_prediction(
    predictive_state: PredictiveState,
    predictions: PredictionSet,
    error: PredictionError | None = None,
) -> PredictiveState:
    """Registra una predicción (y opcionalmente su error) en el historial."""
    record = PredictionRecord(
        turn=predictions.turn,
        predictions=predictions,
        error=error,
    )
    predictive_state.history.add_record(record)
    predictive_state.current_predictions = predictions
    predictive_state.current_error = error
    return predictive_state


def get_prediction_prompt(
    error: PredictionError | None,
    predictive_state: PredictiveState,
) -> str | None:
    """Genera texto para el behavior modifier basado en el prediction error.

    Comunica al LLM la "sorpresa" del agente para que module su respuesta.
    Solo genera prompt si hay error significativo.
    """
    if error is None or not predictive_state.is_warm:
        return None  # Cold start: no modificar comportamiento

    if error.surprise_type == SurpriseType.NONE:
        return None  # Predicción correcta, no hay sorpresa

    parts: list[str] = []

    if error.surprise_type == SurpriseType.POSITIVE:
        parts.append(
            f"Sorpresa positiva (intensidad: {error.vulnerability:.2f}): "
            "lo que ocurrió fue mejor de lo que esperabas. "
            "Sientes un alivio o agrado genuino."
        )
    elif error.surprise_type == SurpriseType.NEGATIVE:
        parts.append(
            f"Sorpresa negativa (intensidad: {error.vulnerability:.2f}): "
            "lo que ocurrió fue peor de lo que esperabas. "
            "Sientes una decepción o inquietud genuina."
        )
    elif error.surprise_type == SurpriseType.NEUTRAL:
        parts.append(
            f"Sorpresa neutra (intensidad: {error.vulnerability:.2f}): "
            "lo que ocurrió fue inesperado pero no necesariamente bueno o malo. "
            "Sientes curiosidad."
        )

    if not parts:
        return None

    return "Yo predictivo: " + " ".join(parts)


# ---------------------------------------------------------------------------
# Paso 1.2 completo: Prediction error como fuente de emoción
# ---------------------------------------------------------------------------

class EmotionModulation:
    """Contribución emocional del prediction error.

    Estos deltas se aplican sobre las dimensiones generadas por el appraisal
    clásico (Scherer), escalados por predictive_weight.
    """

    __slots__ = ("valence_delta", "arousal_delta", "intensity_delta", "certainty_delta")

    def __init__(
        self,
        valence_delta: float = 0.0,
        arousal_delta: float = 0.0,
        intensity_delta: float = 0.0,
        certainty_delta: float = 0.0,
    ) -> None:
        self.valence_delta = valence_delta
        self.arousal_delta = arousal_delta
        self.intensity_delta = intensity_delta
        self.certainty_delta = certainty_delta


def prediction_error_to_emotion_modulation(
    error: PredictionError | None,
    predictive_weight: float,
    *,
    raw_mode: bool = False,
    extreme_mode: bool = False,
) -> EmotionModulation:
    """Transforma el prediction error en modulación emocional.

    La emoción emerge de la discrepancia entre predicción y realidad:
    - Surprise positiva → boost valence, arousal, intensity
    - Surprise negativa → reduce valence, boost arousal, intensity
    - Surprise neutra → boost arousal (curiosidad), leve intensity
    - Sin surprise → leve contentment (valence+), reduce arousal (calma)

    Todo escalado por predictive_weight (0 en cold start = sin efecto).

    La VULNERABILIDAD (precisión × error) amplifica el impacto:
    cuanto más seguro estabas, más te afecta estar equivocado.

    Mode adaptations:
    - Raw: amplifica deltas x1.3 (sin regulación posterior = más impacto).
    - Extreme: amplifica x1.6, fuerza vulnerability=1 (cada turno = montaña rusa).

    Args:
        error: Error de predicción del turno actual (None = sin predicción).
        predictive_weight: Peso del predictive processing (0-0.6).
        raw_mode: Si True, amplifica error x1.3.
        extreme_mode: Si True, amplifica error x1.6 y vulnerability=1.

    Returns:
        EmotionModulation con deltas para valence, arousal, intensity, certainty.
    """
    if error is None or predictive_weight <= 0:
        return EmotionModulation()

    v_delta = 0.0
    a_delta = 0.0
    i_delta = 0.0
    c_delta = 0.0

    # Vulnerability amplifies everything (high precision + high error = intense)
    # Extreme: force vulnerability to 1 (always maximally vulnerable)
    vuln = 1.0 if extreme_mode else error.vulnerability  # 0-1

    if error.surprise_type == SurpriseType.POSITIVE:
        # Mejor de lo esperado → alegría, alivio
        # valence_direction ya es positivo aquí
        v_delta = error.valence_direction * 0.4 + vuln * 0.2
        a_delta = error.total_error * 0.3  # Surprise sube arousal
        i_delta = vuln * 0.3  # Intensidad proporcional a vulnerabilidad
        c_delta = -error.total_error * 0.2  # Menos certeza (las cosas cambian)

    elif error.surprise_type == SurpriseType.NEGATIVE:
        # Peor de lo esperado → decepción, inquietud
        # valence_direction ya es negativo aquí
        v_delta = error.valence_direction * 0.4 - vuln * 0.2
        a_delta = error.total_error * 0.4  # Más arousal (alerta)
        i_delta = vuln * 0.4  # Más intenso cuando eres vulnerable
        c_delta = -error.total_error * 0.3  # Mucha menos certeza

    elif error.surprise_type == SurpriseType.NEUTRAL:
        # Inesperado pero no valorado → curiosidad
        v_delta = 0.05  # Leve positivo (curiosidad es agradable)
        a_delta = error.total_error * 0.35  # Arousal por novedad
        i_delta = vuln * 0.15  # Menos intenso que surprise con valencia
        c_delta = -error.total_error * 0.15

    else:  # SurpriseType.NONE — predicción correcta
        # Satisfaction sutil: el mundo es predecible
        v_delta = 0.05  # Micro-boost de valence (contentment)
        a_delta = -0.05  # Reduce arousal (calma, control)
        i_delta = 0.0  # Sin intensidad extra
        c_delta = 0.1  # Más certeza (el modelo funciona)

    # Mode amplification: raw x1.3, extreme x1.6
    mode_amp = 1.6 if extreme_mode else (1.3 if raw_mode else 1.0)

    # Escalar todo por predictive_weight * mode amplification
    return EmotionModulation(
        valence_delta=round(v_delta * predictive_weight * mode_amp, 4),
        arousal_delta=round(a_delta * predictive_weight * mode_amp, 4),
        intensity_delta=round(i_delta * predictive_weight * mode_amp, 4),
        certainty_delta=round(c_delta * predictive_weight * mode_amp, 4),
    )
