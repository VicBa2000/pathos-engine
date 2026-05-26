"""Tests para Predictive Processing (Pilar 1 ANIMA).

Verifica:
- Predicción básica (generate_predictions)
- Cold start (sin historial)
- Predicción con user model
- Predicción con schemas activos
- Confidence y precisión bayesiana
- Prediction error computation
- Surprise type classification
- Precision update (bayesiana)
- Precision decay temporal
- PredictionHistory buffer
- Prediction prompt generation
- Vulnerabilidad emocional
"""

import pytest

from pathos.engine.predictive import (
    EmotionModulation,
    PredictiveEngine,
    compute_prediction_error,
    decay_precision,
    get_prediction_prompt,
    prediction_error_to_emotion_modulation,
    record_prediction,
    update_precision,
)
from pathos.models.emotion import (
    BodyState,
    EmotionalState,
    Mood,
    MoodLabel,
    PrimaryEmotion,
)
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
    default_predictive_state,
)
from pathos.models.social import UserModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_model(
    intent: float = 0.3,
    engagement: float = 0.5,
    rapport: float = 0.3,
    interactions: int = 0,
    style: str = "unknown",
) -> UserModel:
    return UserModel(
        perceived_intent=intent,
        perceived_engagement=engagement,
        rapport=rapport,
        communication_style=style,
        interaction_count=interactions,
    )


def _mood(
    baseline_valence: float = 0.1,
    baseline_arousal: float = 0.3,
    trend: str = "stable",
) -> Mood:
    return Mood(
        baseline_valence=baseline_valence,
        baseline_arousal=baseline_arousal,
        trend=trend,
        label=MoodLabel.NEUTRAL,
    )


def _emotional_state(
    valence: float = 0.0,
    arousal: float = 0.3,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        certainty=0.5,
        primary_emotion=PrimaryEmotion.NEUTRAL,
        intensity=0.3,
        body_state=BodyState(),
        mood=_mood(),
    )


def _history(messages: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Crea un historial de conversación a partir de tuplas (role, content)."""
    return [{"role": role, "content": content} for role, content in messages]


def _state(
    content_precision: float = 0.3,
    emotion_precision: float = 0.3,
    demand_precision: float = 0.3,
) -> PredictiveState:
    return PredictiveState(
        content_precision=content_precision,
        emotion_precision=emotion_precision,
        demand_precision=demand_precision,
    )


# ---------------------------------------------------------------------------
# Tests: Modelos (PredictionSet, PredictionError, PredictionHistory)
# ---------------------------------------------------------------------------

class TestPredictiveModels:
    """Tests para los modelos de datos predictivos."""

    def test_prediction_set_defaults(self) -> None:
        ps = PredictionSet()
        assert ps.content.confidence == 0.3
        assert ps.emotion.confidence == 0.3
        assert ps.demand.confidence == 0.3
        assert ps.turn == 0

    def test_prediction_set_average_confidence(self) -> None:
        ps = PredictionSet(
            content=ContentPrediction(confidence=0.6),
            emotion=EmotionPrediction(confidence=0.4),
            demand=DemandPrediction(confidence=0.5),
        )
        assert abs(ps.average_confidence - 0.5) < 0.01

    def test_prediction_set_is_cold_start(self) -> None:
        cold = PredictionSet()
        assert cold.is_cold_start

        warm = PredictionSet(
            content=ContentPrediction(confidence=0.6),
            emotion=EmotionPrediction(confidence=0.5),
            demand=DemandPrediction(confidence=0.5),
        )
        assert not warm.is_cold_start

    def test_prediction_error_defaults(self) -> None:
        pe = PredictionError()
        assert pe.content_error == 0.0
        assert pe.surprise_type == SurpriseType.NONE
        assert pe.vulnerability == 0.0

    def test_prediction_history_add_record(self) -> None:
        history = PredictionHistory(max_size=5)
        for i in range(7):
            record = PredictionRecord(turn=i, predictions=PredictionSet(turn=i))
            history.add_record(record)
        assert len(history.records) == 5
        assert history.records[0].turn == 2  # Los primeros 2 se eliminaron

    def test_prediction_history_evaluated_count(self) -> None:
        history = PredictionHistory()
        history.add_record(PredictionRecord(turn=0, predictions=PredictionSet()))
        history.add_record(PredictionRecord(
            turn=1, predictions=PredictionSet(),
            error=PredictionError(),
        ))
        assert history.evaluated_count == 1

    def test_predictive_state_defaults(self) -> None:
        state = default_predictive_state()
        assert state.current_predictions is None
        assert state.current_error is None
        assert state.content_precision == 0.3
        assert state.predictive_weight == 0.0
        assert not state.is_warm

    def test_predictive_state_is_warm(self) -> None:
        state = default_predictive_state()
        for i in range(3):
            state.history.add_record(PredictionRecord(
                turn=i, predictions=PredictionSet(),
                error=PredictionError(),
            ))
        assert state.is_warm

    def test_surprise_type_values(self) -> None:
        assert SurpriseType.POSITIVE == "positive"
        assert SurpriseType.NEGATIVE == "negative"
        assert SurpriseType.NEUTRAL == "neutral"
        assert SurpriseType.NONE == "none"

    def test_demand_type_values(self) -> None:
        assert DemandType.HELP == "help"
        assert DemandType.EMOTIONAL == "emotional"
        assert DemandType.TASK == "task"


# ---------------------------------------------------------------------------
# Tests: PredictiveEngine.generate_predictions
# ---------------------------------------------------------------------------

class TestGeneratePredictions:
    """Tests para la generación de predicciones."""

    def setup_method(self) -> None:
        self.engine = PredictiveEngine()

    def test_cold_start_empty_history(self) -> None:
        """Sin historial, retorna defaults con baja confianza."""
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=[],
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        assert pred.is_cold_start or pred.average_confidence <= 0.35
        assert pred.content.expected_topic == "unknown"
        assert pred.demand.expected_demand == DemandType.UNKNOWN

    def test_basic_prediction_with_history(self) -> None:
        """Con historial, genera predicciones con confianza mayor que cold start."""
        history = _history([
            ("user", "hola, ¿cómo estás?"),
            ("assistant", "¡Hola! Estoy bien, gracias."),
            ("user", "me alegro, ¿puedes ayudarme con algo?"),
            ("assistant", "Claro, dime."),
        ])
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(interactions=5),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        # Con historial, la confianza debería ser mayor que cold start
        assert pred.content.confidence > 0.1
        assert pred.emotion.confidence > 0.1

    def test_positive_user_predicts_positive_tone(self) -> None:
        """Usuario con historial positivo debería predecir tono positivo."""
        history = _history([
            ("user", "genial, perfecto, excelente trabajo"),
            ("assistant", "Gracias!"),
            ("user", "esto está increíble, amazing, love it"),
            ("assistant", "Me alegro!"),
        ])
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(intent=0.7, interactions=5),
            mood=_mood(),
            emotional_state=_emotional_state(valence=0.3),
        )
        assert pred.content.expected_tone == "positive"

    def test_negative_user_predicts_negative_tone(self) -> None:
        """Usuario con historial negativo debería predecir tono negativo."""
        history = _history([
            ("user", "esto está mal, terrible, no funciona"),
            ("assistant", "Lo siento..."),
            ("user", "horrible, odio esto, es lo peor"),
            ("assistant", "Entiendo tu frustración"),
        ])
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(intent=-0.5, interactions=5),
            mood=_mood(),
            emotional_state=_emotional_state(valence=-0.3),
        )
        assert pred.content.expected_tone == "negative"

    def test_question_intent_detected(self) -> None:
        """Preguntas en historial deberían predecir intención de pregunta."""
        history = _history([
            ("user", "¿cómo funciona esto?"),
            ("assistant", "Funciona así..."),
            ("user", "¿y por qué ocurre eso?"),
            ("assistant", "Porque..."),
        ])
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        assert pred.content.expected_intent == "question"

    def test_user_model_affects_emotion_prediction(self) -> None:
        """User model hostil debería predecir valence más baja."""
        history = _history([
            ("user", "hola"),
            ("assistant", "hola"),
        ])
        hostile = _user_model(intent=-0.8, engagement=0.8, interactions=10)
        friendly = _user_model(intent=0.8, engagement=0.8, interactions=10)

        pred_hostile = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=hostile,
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        pred_friendly = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=friendly,
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        assert pred_hostile.emotion.expected_valence < pred_friendly.emotion.expected_valence

    def test_mood_trend_affects_prediction(self) -> None:
        """Mood declining debería predecir valence ligeramente más baja."""
        history = _history([("user", "hola"), ("assistant", "hola")])
        pred_declining = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(trend="declining"),
            emotional_state=_emotional_state(),
        )
        pred_improving = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(trend="improving"),
            emotional_state=_emotional_state(),
        )
        assert pred_declining.emotion.expected_valence < pred_improving.emotion.expected_valence

    def test_schemas_refine_predictions(self) -> None:
        """Schemas activos deberían modificar predicciones."""
        history = _history([("user", "hola"), ("assistant", "hola")])
        schemas_negative = [("criticism", "anger", 0.8)]
        schemas_positive = [("praise", "joy", 0.8)]

        pred_neg = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
            active_schemas=schemas_negative,
        )
        pred_pos = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
            active_schemas=schemas_positive,
        )
        assert pred_neg.emotion.expected_valence < pred_pos.emotion.expected_valence

    def test_weak_schemas_dont_modify(self) -> None:
        """Schemas débiles (strength < 0.3) no deberían modificar."""
        history = _history([("user", "hola"), ("assistant", "hola")])
        weak_schemas = [("criticism", "anger", 0.1)]

        pred_no_schema = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        pred_weak = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
            active_schemas=weak_schemas,
        )
        assert pred_no_schema.emotion.expected_valence == pred_weak.emotion.expected_valence

    def test_precision_affects_confidence(self) -> None:
        """Alta precisión bayesiana debería aumentar confianza."""
        history = _history([("user", "hola"), ("assistant", "hola")])
        low_precision = _state(content_precision=0.1, emotion_precision=0.1, demand_precision=0.1)
        high_precision = _state(content_precision=0.9, emotion_precision=0.9, demand_precision=0.9)

        pred_low = self.engine.generate_predictions(
            predictive_state=low_precision,
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        pred_high = self.engine.generate_predictions(
            predictive_state=high_precision,
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        assert pred_low.average_confidence < pred_high.average_confidence

    def test_demand_prediction_emotional_user(self) -> None:
        """Usuario con estilo 'emotional' debería predecir demanda EMOTIONAL."""
        history = _history([
            ("user", "me siento triste hoy"),
            ("assistant", "Lo siento, ¿quieres hablar de ello?"),
        ])
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(style="emotional"),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        assert pred.demand.expected_demand == DemandType.EMOTIONAL

    def test_agent_negative_state_lowers_prediction(self) -> None:
        """Si el agente está negativo, predice impacto en el usuario."""
        history = _history([("user", "hola"), ("assistant", "hola")])
        pred = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(valence=-0.5),
        )
        pred_neutral = self.engine.generate_predictions(
            predictive_state=_state(),
            conversation_history=history,
            user_model=_user_model(),
            mood=_mood(),
            emotional_state=_emotional_state(valence=0.0),
        )
        assert pred.emotion.expected_valence < pred_neutral.emotion.expected_valence


# ---------------------------------------------------------------------------
# Tests: compute_prediction_error
# ---------------------------------------------------------------------------

class TestComputePredictionError:
    """Tests para el cómputo de errores de predicción."""

    def test_perfect_prediction_no_error(self) -> None:
        """Predicción perfecta debería tener error mínimo."""
        pred = PredictionSet(
            content=ContentPrediction(
                expected_tone="neutral", expected_intent="greeting", confidence=0.5,
            ),
            emotion=EmotionPrediction(
                expected_valence=0.3, expected_arousal=0.4, confidence=0.5,
            ),
            demand=DemandPrediction(
                expected_demand=DemandType.CONVERSATION, confidence=0.5,
            ),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="hola, buenos días",
            detected_user_valence=0.3,
            detected_user_arousal=0.4,
            detected_intent="greeting",
            detected_demand=DemandType.CONVERSATION,
        )
        assert error.emotion_error < 0.05
        assert error.demand_error == 0.0
        assert error.surprise_type == SurpriseType.NONE

    def test_large_valence_error_negative_surprise(self) -> None:
        """Gran error en valence (peor) debería dar surprise negativa."""
        pred = PredictionSet(
            emotion=EmotionPrediction(
                expected_valence=0.5, expected_arousal=0.3, confidence=0.7,
            ),
            content=ContentPrediction(confidence=0.7),
            demand=DemandPrediction(confidence=0.7),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="esto es horrible, terrible",
            detected_user_valence=-0.5,
            detected_user_arousal=0.7,
            detected_intent="complaint",
        )
        assert error.valence_direction < 0
        assert error.emotion_error > 0.3
        assert error.surprise_type == SurpriseType.NEGATIVE

    def test_large_valence_error_positive_surprise(self) -> None:
        """Gran error en valence (mejor) debería dar surprise positiva."""
        pred = PredictionSet(
            emotion=EmotionPrediction(
                expected_valence=-0.5, expected_arousal=0.3, confidence=0.7,
            ),
            content=ContentPrediction(confidence=0.7),
            demand=DemandPrediction(confidence=0.7),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="genial, perfecto, gracias!",
            detected_user_valence=0.5,
            detected_user_arousal=0.4,
            detected_intent="greeting",
        )
        assert error.valence_direction > 0
        assert error.surprise_type == SurpriseType.POSITIVE

    def test_content_error_detects_tone_mismatch(self) -> None:
        """Error de contenido cuando el tono predicho no coincide."""
        pred = PredictionSet(
            content=ContentPrediction(expected_tone="positive", confidence=0.6),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="esto es terrible, lo peor que he visto, odio esto",
            detected_user_valence=-0.5,
            detected_user_arousal=0.6,
            detected_intent="complaint",
        )
        assert error.content_error > 0.3

    def test_demand_error_mismatch(self) -> None:
        """Error de demanda cuando no coincide."""
        pred = PredictionSet(
            demand=DemandPrediction(
                expected_demand=DemandType.HELP, confidence=0.6,
            ),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="hola",
            detected_user_valence=0.0,
            detected_user_arousal=0.3,
            detected_intent="greeting",
            detected_demand=DemandType.CONVERSATION,
        )
        assert error.demand_error > 0.0

    def test_demand_error_no_detected(self) -> None:
        """Sin demanda detectada, error moderado."""
        pred = PredictionSet()
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="hola",
            detected_user_valence=0.0,
            detected_user_arousal=0.3,
            detected_intent="greeting",
            detected_demand=None,
        )
        assert error.demand_error == 0.3

    def test_vulnerability_scales_with_confidence(self) -> None:
        """Vulnerabilidad alta cuando confianza alta + error grande."""
        pred_confident = PredictionSet(
            content=ContentPrediction(confidence=0.9),
            emotion=EmotionPrediction(
                expected_valence=0.5, expected_arousal=0.3, confidence=0.9,
            ),
            demand=DemandPrediction(confidence=0.9),
        )
        pred_uncertain = PredictionSet(
            content=ContentPrediction(confidence=0.1),
            emotion=EmotionPrediction(
                expected_valence=0.5, expected_arousal=0.3, confidence=0.1,
            ),
            demand=DemandPrediction(confidence=0.1),
        )
        # Mismo estímulo, misma realidad, diferente confianza
        error_confident = compute_prediction_error(
            predictions=pred_confident,
            actual_stimulus="esto es terrible, odio todo",
            detected_user_valence=-0.7,
            detected_user_arousal=0.8,
            detected_intent="complaint",
        )
        error_uncertain = compute_prediction_error(
            predictions=pred_uncertain,
            actual_stimulus="esto es terrible, odio todo",
            detected_user_valence=-0.7,
            detected_user_arousal=0.8,
            detected_intent="complaint",
        )
        assert error_confident.vulnerability > error_uncertain.vulnerability

    def test_neutral_surprise_high_content_low_valence_error(self) -> None:
        """Surprise neutra: alto error en contenido, bajo en valence."""
        pred = PredictionSet(
            content=ContentPrediction(
                expected_tone="positive", expected_intent="greeting", confidence=0.7,
            ),
            emotion=EmotionPrediction(
                expected_valence=0.0, expected_arousal=0.3, confidence=0.7,
            ),
            demand=DemandPrediction(confidence=0.7),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="¿Sabías que Plutón ya no es un planeta?",
            detected_user_valence=0.05,
            detected_user_arousal=0.4,
            detected_intent="topic_change",
        )
        # Valence direction close to 0, but content mismatch
        assert abs(error.valence_direction) < 0.15
        # If total error is above threshold, should be neutral surprise
        if error.total_error > 0.15:
            assert error.surprise_type == SurpriseType.NEUTRAL

    def test_error_values_clamped(self) -> None:
        """Todos los valores deben estar en rango."""
        pred = PredictionSet(
            emotion=EmotionPrediction(
                expected_valence=1.0, expected_arousal=0.0, confidence=1.0,
            ),
        )
        error = compute_prediction_error(
            predictions=pred,
            actual_stimulus="odio todo terrible horrible mal",
            detected_user_valence=-1.0,
            detected_user_arousal=1.0,
            detected_intent="complaint",
        )
        assert 0 <= error.content_error <= 1
        assert 0 <= error.emotion_error <= 1
        assert 0 <= error.demand_error <= 1
        assert 0 <= error.total_error <= 1
        assert -1 <= error.valence_direction <= 1
        assert 0 <= error.vulnerability <= 1


# ---------------------------------------------------------------------------
# Tests: update_precision (bayesiana)
# ---------------------------------------------------------------------------

class TestUpdatePrecision:
    """Tests para la actualización de precisión bayesiana."""

    def test_low_error_increases_precision(self) -> None:
        """Error bajo debería aumentar la precisión."""
        state = _state(content_precision=0.3, emotion_precision=0.3, demand_precision=0.3)
        error = PredictionError(
            content_error=0.05, emotion_error=0.1, demand_error=0.0,
        )
        updated = update_precision(state, error)
        assert updated.content_precision > 0.3
        assert updated.emotion_precision > 0.3
        assert updated.demand_precision > 0.3

    def test_high_error_decreases_precision(self) -> None:
        """Error alto debería disminuir la precisión."""
        state = _state(content_precision=0.7, emotion_precision=0.7, demand_precision=0.7)
        error = PredictionError(
            content_error=0.9, emotion_error=0.8, demand_error=0.7,
        )
        updated = update_precision(state, error)
        assert updated.content_precision < 0.7
        assert updated.emotion_precision < 0.7
        assert updated.demand_precision < 0.7

    def test_precision_converges_with_consistent_low_error(self) -> None:
        """Errores consistentemente bajos deberían llevar precisión alta."""
        state = _state(content_precision=0.3, emotion_precision=0.3, demand_precision=0.3)
        error = PredictionError(
            content_error=0.05, emotion_error=0.05, demand_error=0.05,
        )
        for _ in range(20):
            state = update_precision(state, error)
        assert state.content_precision > 0.6
        assert state.emotion_precision > 0.6

    def test_precision_never_exceeds_bounds(self) -> None:
        """Precisión siempre entre 0.05 y 0.95."""
        state = _state(content_precision=0.95, emotion_precision=0.05, demand_precision=0.5)
        # Error bajo en todo
        error = PredictionError(content_error=0.0, emotion_error=0.0, demand_error=0.0)
        updated = update_precision(state, error)
        assert 0.05 <= updated.content_precision <= 0.95
        assert 0.05 <= updated.emotion_precision <= 0.95
        # Error alto en todo
        error = PredictionError(content_error=1.0, emotion_error=1.0, demand_error=1.0)
        updated = update_precision(state, error)
        assert 0.05 <= updated.content_precision <= 0.95
        assert 0.05 <= updated.emotion_precision <= 0.95

    def test_predictive_weight_grows_with_precision(self) -> None:
        """El peso del predictive processing crece con la precisión."""
        state = _state(content_precision=0.3, emotion_precision=0.3, demand_precision=0.3)
        initial_weight = state.predictive_weight

        error = PredictionError(
            content_error=0.05, emotion_error=0.05, demand_error=0.05,
        )
        for _ in range(10):
            state = update_precision(state, error)

        assert state.predictive_weight > initial_weight
        assert state.predictive_weight <= state.max_predictive_weight

    def test_per_dimension_independence(self) -> None:
        """Cada dimensión se actualiza independientemente."""
        state = _state(content_precision=0.5, emotion_precision=0.5, demand_precision=0.5)
        # Error alto solo en contenido
        error = PredictionError(
            content_error=0.9, emotion_error=0.05, demand_error=0.05,
        )
        updated = update_precision(state, error)
        assert updated.content_precision < 0.5  # Bajó
        assert updated.emotion_precision > 0.5  # Subió
        assert updated.demand_precision > 0.5   # Subió

    def test_history_accuracy_updated(self) -> None:
        """La accuracy del historial se actualiza con EMA."""
        state = _state()
        initial_content_acc = state.history.content_accuracy

        error = PredictionError(content_error=0.0, emotion_error=0.0, demand_error=0.0)
        updated = update_precision(state, error)
        assert updated.history.content_accuracy > initial_content_acc


# ---------------------------------------------------------------------------
# Tests: decay_precision
# ---------------------------------------------------------------------------

class TestDecayPrecision:
    """Tests para el decay temporal de precisión."""

    def test_single_turn_decay(self) -> None:
        """Un turno de decay reduce ligeramente la precisión."""
        state = _state(content_precision=0.7, emotion_precision=0.7, demand_precision=0.7)
        decayed = decay_precision(state, elapsed_turns=1)
        assert decayed.content_precision < 0.7
        assert decayed.emotion_precision < 0.7

    def test_multiple_turns_decay_more(self) -> None:
        """Más turnos de decay reducen más la precisión."""
        state1 = _state(content_precision=0.7, emotion_precision=0.7, demand_precision=0.7)
        state10 = _state(content_precision=0.7, emotion_precision=0.7, demand_precision=0.7)
        decayed1 = decay_precision(state1, elapsed_turns=1)
        decayed10 = decay_precision(state10, elapsed_turns=10)
        assert decayed10.content_precision < decayed1.content_precision

    def test_decay_respects_minimum(self) -> None:
        """El decay no baja por debajo del mínimo."""
        state = _state(content_precision=0.06, emotion_precision=0.06, demand_precision=0.06)
        decayed = decay_precision(state, elapsed_turns=100)
        assert decayed.content_precision >= 0.05
        assert decayed.emotion_precision >= 0.05

    def test_zero_turns_no_decay(self) -> None:
        """Cero turnos no debería cambiar la precisión."""
        state = _state(content_precision=0.7, emotion_precision=0.7, demand_precision=0.7)
        decayed = decay_precision(state, elapsed_turns=0)
        assert decayed.content_precision == 0.7


# ---------------------------------------------------------------------------
# Tests: record_prediction
# ---------------------------------------------------------------------------

class TestRecordPrediction:
    """Tests para el registro de predicciones."""

    def test_record_stores_predictions(self) -> None:
        state = default_predictive_state()
        pred = PredictionSet(turn=5)
        updated = record_prediction(state, pred)
        assert updated.current_predictions is not None
        assert updated.current_predictions.turn == 5
        assert len(updated.history.records) == 1

    def test_record_with_error(self) -> None:
        state = default_predictive_state()
        pred = PredictionSet(turn=3)
        error = PredictionError(total_error=0.4, surprise_type=SurpriseType.NEGATIVE)
        updated = record_prediction(state, pred, error)
        assert updated.current_error is not None
        assert updated.current_error.total_error == 0.4

    def test_record_respects_buffer_size(self) -> None:
        state = default_predictive_state()
        state.history.max_size = 5
        for i in range(10):
            pred = PredictionSet(turn=i)
            state = record_prediction(state, pred)
        assert len(state.history.records) == 5


# ---------------------------------------------------------------------------
# Tests: get_prediction_prompt
# ---------------------------------------------------------------------------

class TestGetPredictionPrompt:
    """Tests para la generación de prompts del behavior modifier."""

    def test_cold_start_returns_none(self) -> None:
        """En cold start (no warm), no genera prompt."""
        state = default_predictive_state()
        error = PredictionError(
            surprise_type=SurpriseType.POSITIVE, vulnerability=0.5,
        )
        result = get_prediction_prompt(error, state)
        assert result is None

    def test_no_surprise_returns_none(self) -> None:
        """Sin sorpresa, no genera prompt."""
        state = default_predictive_state()
        # Hacer warm
        for i in range(3):
            state.history.add_record(PredictionRecord(
                turn=i, predictions=PredictionSet(),
                error=PredictionError(),
            ))
        error = PredictionError(surprise_type=SurpriseType.NONE)
        result = get_prediction_prompt(error, state)
        assert result is None

    def test_positive_surprise_generates_prompt(self) -> None:
        """Surprise positiva genera prompt con 'alivio' o 'agrado'."""
        state = default_predictive_state()
        for i in range(3):
            state.history.add_record(PredictionRecord(
                turn=i, predictions=PredictionSet(),
                error=PredictionError(),
            ))
        error = PredictionError(
            surprise_type=SurpriseType.POSITIVE, vulnerability=0.6,
        )
        result = get_prediction_prompt(error, state)
        assert result is not None
        assert "positiva" in result.lower()
        assert "Yo predictivo" in result

    def test_negative_surprise_generates_prompt(self) -> None:
        """Surprise negativa genera prompt con 'decepción' o 'inquietud'."""
        state = default_predictive_state()
        for i in range(3):
            state.history.add_record(PredictionRecord(
                turn=i, predictions=PredictionSet(),
                error=PredictionError(),
            ))
        error = PredictionError(
            surprise_type=SurpriseType.NEGATIVE, vulnerability=0.7,
        )
        result = get_prediction_prompt(error, state)
        assert result is not None
        assert "negativa" in result.lower()

    def test_neutral_surprise_generates_curiosity_prompt(self) -> None:
        """Surprise neutra genera prompt con 'curiosidad'."""
        state = default_predictive_state()
        for i in range(3):
            state.history.add_record(PredictionRecord(
                turn=i, predictions=PredictionSet(),
                error=PredictionError(),
            ))
        error = PredictionError(
            surprise_type=SurpriseType.NEUTRAL, vulnerability=0.3,
        )
        result = get_prediction_prompt(error, state)
        assert result is not None
        assert "curiosidad" in result.lower()

    def test_none_error_returns_none(self) -> None:
        state = default_predictive_state()
        result = get_prediction_prompt(None, state)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Integración completa del flujo
# ---------------------------------------------------------------------------

class TestPredictiveFlow:
    """Tests de integración del flujo completo predict → error → update."""

    def test_full_flow_single_turn(self) -> None:
        """Flujo completo: predecir, recibir, computar error, actualizar."""
        engine = PredictiveEngine()
        state = default_predictive_state()
        history = _history([
            ("user", "hola, ¿cómo estás?"),
            ("assistant", "Bien, gracias!"),
        ])

        # 1. Generar predicciones
        predictions = engine.generate_predictions(
            predictive_state=state,
            conversation_history=history,
            user_model=_user_model(interactions=3),
            mood=_mood(),
            emotional_state=_emotional_state(),
        )
        assert predictions.turn >= 0

        # 2. "Recibir" input y computar error
        error = compute_prediction_error(
            predictions=predictions,
            actual_stimulus="me siento un poco triste hoy",
            detected_user_valence=-0.3,
            detected_user_arousal=0.4,
            detected_intent="emotional_expression",
            detected_demand=DemandType.EMOTIONAL,
        )
        assert error.total_error >= 0

        # 3. Actualizar precisión
        state = update_precision(state, error)

        # 4. Registrar
        state = record_prediction(state, predictions, error)
        assert len(state.history.records) == 1

    def test_precision_improves_over_consistent_interactions(self) -> None:
        """Con interacciones consistentes, la precisión mejora."""
        engine = PredictiveEngine()
        state = default_predictive_state()

        # Simular 15 turnos donde el usuario siempre es positivo y pregunta
        for i in range(15):
            history = _history([
                ("user", "hola, genial, perfecto!"),
                ("assistant", "Me alegro!"),
            ] * (i + 1))

            predictions = engine.generate_predictions(
                predictive_state=state,
                conversation_history=history[-20:],  # Ventana
                user_model=_user_model(intent=0.6, interactions=i + 1),
                mood=_mood(),
                emotional_state=_emotional_state(valence=0.3),
            )

            error = compute_prediction_error(
                predictions=predictions,
                actual_stimulus="genial, excelente, gracias!",
                detected_user_valence=0.5,
                detected_user_arousal=0.4,
                detected_intent="greeting",
                detected_demand=DemandType.CONVERSATION,
            )

            state = update_precision(state, error)
            state = record_prediction(state, predictions, error)

        # Después de 15 turnos consistentes, la precisión debería haber subido
        assert state.average_precision > 0.4
        assert state.is_warm
        assert state.predictive_weight > 0

    def test_surprise_from_pattern_break(self) -> None:
        """Un cambio de patrón genera surprise."""
        engine = PredictiveEngine()
        state = default_predictive_state()

        # Primero: 5 turnos positivos para construir expectativa
        for i in range(5):
            history = _history([
                ("user", "genial, perfecto, excelente!"),
                ("assistant", "Me alegro!"),
            ] * (i + 1))
            predictions = engine.generate_predictions(
                predictive_state=state,
                conversation_history=history[-20:],
                user_model=_user_model(intent=0.7, interactions=i + 1),
                mood=_mood(),
                emotional_state=_emotional_state(valence=0.3),
            )
            error = compute_prediction_error(
                predictions=predictions,
                actual_stimulus="genial, excelente, perfecto!",
                detected_user_valence=0.5,
                detected_user_arousal=0.4,
                detected_intent="greeting",
            )
            state = update_precision(state, error)
            state = record_prediction(state, predictions, error)

        # Ahora: mensaje negativo inesperado
        history = _history([
            ("user", "genial, perfecto!"),
            ("assistant", "Me alegro!"),
        ] * 5)
        predictions = engine.generate_predictions(
            predictive_state=state,
            conversation_history=history,
            user_model=_user_model(intent=0.7, interactions=6),
            mood=_mood(),
            emotional_state=_emotional_state(valence=0.3),
        )
        # El usuario cambia completamente
        error = compute_prediction_error(
            predictions=predictions,
            actual_stimulus="esto es terrible, odio todo, lo peor",
            detected_user_valence=-0.7,
            detected_user_arousal=0.8,
            detected_intent="complaint",
        )

        # Debería haber surprise negativa con vulnerabilidad alta
        assert error.surprise_type == SurpriseType.NEGATIVE
        assert error.vulnerability > 0.1  # Tenía confianza → duele


# ---------------------------------------------------------------------------
# Tests: prediction_error_to_emotion_modulation
# ---------------------------------------------------------------------------

class TestEmotionModulation:
    """Tests para la conversión de prediction error a modulación emocional."""

    def test_no_error_returns_zero_modulation(self) -> None:
        """Sin error, la modulación es cero."""
        mod = prediction_error_to_emotion_modulation(None, 0.5)
        assert mod.valence_delta == 0.0
        assert mod.arousal_delta == 0.0
        assert mod.intensity_delta == 0.0
        assert mod.certainty_delta == 0.0

    def test_zero_weight_returns_zero_modulation(self) -> None:
        """Con peso 0 (cold start), la modulación es cero."""
        error = PredictionError(
            total_error=0.8,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.6,
            valence_direction=-0.5,
        )
        mod = prediction_error_to_emotion_modulation(error, 0.0)
        assert mod.valence_delta == 0.0
        assert mod.arousal_delta == 0.0
        assert mod.intensity_delta == 0.0

    def test_positive_surprise_boosts_valence(self) -> None:
        """Surprise positiva sube valence."""
        error = PredictionError(
            total_error=0.6,
            surprise_type=SurpriseType.POSITIVE,
            vulnerability=0.4,
            valence_direction=0.5,
        )
        mod = prediction_error_to_emotion_modulation(error, 0.5)
        assert mod.valence_delta > 0
        assert mod.arousal_delta > 0  # Surprise sube arousal
        assert mod.intensity_delta > 0  # Vulnerability amplifica

    def test_negative_surprise_reduces_valence(self) -> None:
        """Surprise negativa baja valence."""
        error = PredictionError(
            total_error=0.6,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.4,
            valence_direction=-0.5,
        )
        mod = prediction_error_to_emotion_modulation(error, 0.5)
        assert mod.valence_delta < 0
        assert mod.arousal_delta > 0  # Alerta
        assert mod.intensity_delta > 0
        assert mod.certainty_delta < 0  # Menos certeza

    def test_neutral_surprise_curiosity(self) -> None:
        """Surprise neutra genera curiosidad (arousal+, leve valence+)."""
        error = PredictionError(
            total_error=0.5,
            surprise_type=SurpriseType.NEUTRAL,
            vulnerability=0.3,
            valence_direction=0.0,
        )
        mod = prediction_error_to_emotion_modulation(error, 0.5)
        assert mod.valence_delta >= 0  # Leve positivo (curiosidad)
        assert mod.arousal_delta > 0  # Novedad sube arousal
        assert mod.certainty_delta < 0  # Menos certeza

    def test_no_surprise_contentment(self) -> None:
        """Predicción correcta genera micro-contentment."""
        error = PredictionError(
            total_error=0.05,
            surprise_type=SurpriseType.NONE,
            vulnerability=0.0,
            valence_direction=0.0,
        )
        mod = prediction_error_to_emotion_modulation(error, 0.5)
        assert mod.valence_delta > 0  # Micro-boost
        assert mod.arousal_delta < 0  # Calma (predecible)
        assert mod.certainty_delta > 0  # Más certeza

    def test_weight_scales_modulation(self) -> None:
        """Mayor peso = mayor modulación."""
        error = PredictionError(
            total_error=0.6,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.5,
            valence_direction=-0.5,
        )
        mod_low = prediction_error_to_emotion_modulation(error, 0.1)
        mod_high = prediction_error_to_emotion_modulation(error, 0.6)
        assert abs(mod_high.valence_delta) > abs(mod_low.valence_delta)
        assert abs(mod_high.arousal_delta) > abs(mod_low.arousal_delta)
        assert abs(mod_high.intensity_delta) > abs(mod_low.intensity_delta)

    def test_high_vulnerability_amplifies_negative(self) -> None:
        """Alta vulnerabilidad amplifica la modulación negativa."""
        error_low_vuln = PredictionError(
            total_error=0.6,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.1,
            valence_direction=-0.5,
        )
        error_high_vuln = PredictionError(
            total_error=0.6,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.8,
            valence_direction=-0.5,
        )
        mod_low = prediction_error_to_emotion_modulation(error_low_vuln, 0.5)
        mod_high = prediction_error_to_emotion_modulation(error_high_vuln, 0.5)
        assert mod_high.valence_delta < mod_low.valence_delta  # Más negativo
        assert mod_high.intensity_delta > mod_low.intensity_delta  # Más intenso


# ---------------------------------------------------------------------------
# Tests: Integración generate_emotion + predictive_modulation
# ---------------------------------------------------------------------------

class TestGenerateEmotionWithPredictive:
    """Tests de integración entre prediction error y generación emocional."""

    def _make_appraisal(self) -> "AppraisalVector":
        """Crea un AppraisalVector neutral para tests."""
        from pathos.models.appraisal import (
            AgencyAttribution,
            AppraisalVector,
            CopingPotential,
            NormCompatibility,
            RelevanceCheck,
            ValenceAssessment,
        )
        return AppraisalVector(
            relevance=RelevanceCheck(novelty=0.3, personal_significance=0.5),
            valence=ValenceAssessment(
                goal_conduciveness=0.0, value_alignment=0.0, intrinsic_pleasantness=0.0,
            ),
            coping=CopingPotential(control=0.5, power=0.5, adjustability=0.5),
            agency=AgencyAttribution(
                responsible_agent="user", intentionality=0.5, fairness=0.0,
            ),
            norms=NormCompatibility(
                internal_standards=0.0, external_standards=0.0, self_consistency=0.0,
            ),
        )

    def test_no_modulation_matches_baseline(self) -> None:
        """Sin modulación predictiva, el resultado es igual al clásico."""
        from pathos.engine.generator import generate_emotion
        appraisal = self._make_appraisal()
        state = _emotional_state()

        result_no_mod = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
        )
        result_none_mod = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
            predictive_modulation=None,
        )
        assert result_no_mod.valence == result_none_mod.valence
        assert result_no_mod.arousal == result_none_mod.arousal

    def test_positive_modulation_increases_valence(self) -> None:
        """Modulación positiva aumenta valence respecto al baseline."""
        from pathos.engine.generator import generate_emotion
        appraisal = self._make_appraisal()
        state = _emotional_state()

        baseline = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
        )
        mod = EmotionModulation(
            valence_delta=0.15, arousal_delta=0.05,
            intensity_delta=0.1, certainty_delta=-0.05,
        )
        modified = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
            predictive_modulation=mod,
        )
        assert modified.valence > baseline.valence
        assert modified.intensity > baseline.intensity

    def test_negative_modulation_decreases_valence(self) -> None:
        """Modulación negativa disminuye valence respecto al baseline."""
        from pathos.engine.generator import generate_emotion
        appraisal = self._make_appraisal()
        state = _emotional_state()

        baseline = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
        )
        mod = EmotionModulation(
            valence_delta=-0.2, arousal_delta=0.1,
            intensity_delta=0.15, certainty_delta=-0.1,
        )
        modified = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
            predictive_modulation=mod,
        )
        assert modified.valence < baseline.valence

    def test_zero_modulation_no_effect(self) -> None:
        """Modulación con todos los deltas en cero no cambia nada."""
        from pathos.engine.generator import generate_emotion
        appraisal = self._make_appraisal()
        state = _emotional_state()

        baseline = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
        )
        mod = EmotionModulation()  # Todos ceros
        modified = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
            predictive_modulation=mod,
        )
        assert baseline.valence == modified.valence
        assert baseline.arousal == modified.arousal

    def test_cold_start_produces_zero_modulation(self) -> None:
        """El flujo completo con cold start (weight=0) no modifica la emoción."""
        error = PredictionError(
            total_error=0.8,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.7,
            valence_direction=-0.6,
        )
        # Cold start: predictive_weight = 0
        mod = prediction_error_to_emotion_modulation(error, 0.0)
        assert mod.valence_delta == 0.0
        assert mod.arousal_delta == 0.0

    def test_full_flow_surprise_affects_emotion(self) -> None:
        """Flujo completo: error → modulación → emoción diferente del baseline."""
        from pathos.engine.generator import generate_emotion
        appraisal = self._make_appraisal()
        state = _emotional_state()

        # Surprise negativa fuerte con modelo caliente
        error = PredictionError(
            total_error=0.7,
            surprise_type=SurpriseType.NEGATIVE,
            vulnerability=0.5,
            valence_direction=-0.6,
        )
        mod = prediction_error_to_emotion_modulation(error, 0.5)

        baseline = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
        )
        modified = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
            predictive_modulation=mod,
        )

        # La emoción modificada debe ser más negativa y más intensa
        assert modified.valence < baseline.valence
        assert modified.intensity > baseline.intensity

    def test_values_remain_clamped(self) -> None:
        """Incluso con modulación extrema, los valores quedan en rango."""
        from pathos.engine.generator import generate_emotion
        appraisal = self._make_appraisal()
        state = _emotional_state()

        mod = EmotionModulation(
            valence_delta=-2.0, arousal_delta=2.0,
            intensity_delta=2.0, certainty_delta=-2.0,
        )
        result = generate_emotion(
            appraisal=appraisal, current_state=state, stimulus="test",
            predictive_modulation=mod,
        )
        assert -1 <= result.valence <= 1
        assert 0 <= result.arousal <= 1
        assert 0 <= result.intensity <= 1
        assert 0 <= result.certainty <= 1


# ===========================================================================
# F3 (RESIDUUM) — Predictive Processing sobre el residual
# ===========================================================================

from pathos.engine.predictive import (  # noqa: E402
    CLUSTER_CENTROIDS,
    compute_internal_error,
    merge_internal_error,
    predict_internal_state,
    update_internal_precision,
)
from pathos.models.predictive import (  # noqa: E402
    InternalEmotionStatePrediction,
    PredictedCluster,
)
from pathos.models.residuum import (  # noqa: E402
    EmotionProjection,
    InternalEmotionState,
)


def _internal_pred(clusters: list[str], v: float = 0.6, a: float = 0.7,
                   d: float = 0.6, c: float = 0.5) -> InternalEmotionStatePrediction:
    return InternalEmotionStatePrediction(
        predicted_top_k=[PredictedCluster(cluster=cl, weight=1.0 / len(clusters)) for cl in clusters],
        predicted_valence=v, predicted_arousal=a, predicted_dominance=d, predicted_certainty=c,
    )


def _measured(clusters: list[str], v: float = 0.6, a: float = 0.7,
              d: float = 0.6, c: float = 0.5) -> InternalEmotionState:
    return InternalEmotionState(
        top_5_emotions=[
            EmotionProjection(emotion_name=f"e{i}", cluster=cl, cosine_sim=0.5, raw_activation=10.0)
            for i, cl in enumerate(clusters)
        ],
        measured_valence=v, measured_arousal=a, measured_dominance=d, measured_certainty=c,
    )


class TestPredictInternalState:
    """F3 Paso 3.1 — predict_internal_state."""

    def test_produces_five_clusters(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.6, arousal=0.7)
        pe = EmotionPrediction(expected_valence=0.5, expected_arousal=0.6)
        pred = predict_internal_state(ps, es, es.mood, pe)
        assert len(pred.predicted_top_k) == 5
        assert all(c.cluster in CLUSTER_CENTROIDS for c in pred.predicted_top_k)

    def test_weights_sum_to_one(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.3, arousal=0.5)
        pred = predict_internal_state(ps, es, es.mood, EmotionPrediction())
        assert abs(sum(c.weight for c in pred.predicted_top_k) - 1.0) < 0.01

    def test_vad_in_range(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=-0.9, arousal=0.95, dominance=0.8, certainty=0.2)
        pe = EmotionPrediction(expected_valence=-0.8, expected_arousal=0.9)
        pred = predict_internal_state(ps, es, es.mood, pe)
        assert -1 <= pred.predicted_valence <= 1
        assert 0 <= pred.predicted_arousal <= 1
        assert 0 <= pred.predicted_dominance <= 1
        assert 0 <= pred.predicted_certainty <= 1

    def test_negative_state_predicts_negative_clusters(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=-0.8, arousal=0.85)
        pe = EmotionPrediction(expected_valence=-0.8, expected_arousal=0.85)
        pred = predict_internal_state(ps, es, es.mood, pe)
        assert pred.predicted_top_k[0].cluster in ("fear_anxiety", "anger_hostility")

    def test_positive_state_predicts_positive_clusters(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.85, arousal=0.8)
        pe = EmotionPrediction(expected_valence=0.85, expected_arousal=0.8)
        pred = predict_internal_state(ps, es, es.mood, pe)
        assert pred.predicted_top_k[0].cluster == "joy_excitement"

    def test_mood_trend_declining_lowers_valence(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.2, arousal=0.5)
        pe = EmotionPrediction(expected_valence=0.2, expected_arousal=0.5)
        declining = es.mood.model_copy(update={"trend": "declining"})
        improving = es.mood.model_copy(update={"trend": "improving"})
        down = predict_internal_state(ps, es, declining, pe)
        up = predict_internal_state(ps, es, improving, pe)
        assert down.predicted_valence < up.predicted_valence

    def test_confidence_tracks_internal_precision(self):
        ps = default_predictive_state()
        ps.internal_precision = 0.7
        es = EmotionalState(valence=0.3, arousal=0.5)
        pred = predict_internal_state(ps, es, es.mood, EmotionPrediction())
        assert pred.confidence == 0.7

    def test_negative_schema_shifts_valence_down(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.3, arousal=0.5)
        pe = EmotionPrediction(expected_valence=0.3, expected_arousal=0.5)
        base = predict_internal_state(ps, es, es.mood, pe)
        shifted = predict_internal_state(
            ps, es, es.mood, pe, active_schemas=[("criticism", "anger", 0.8)],
        )
        assert shifted.predicted_valence < base.predicted_valence


class TestComputeInternalError:
    """F3 Paso 3.2 — compute_internal_error (overlap + geometric)."""

    def test_dominant_cluster_hit_zero_error(self):
        pred = _internal_pred(["joy_excitement", "pride_confidence", "love_warmth"])
        ie, _ = compute_internal_error(pred, _measured(["joy_excitement"]))
        assert ie == 0.0

    def test_missed_cluster_full_error(self):
        pred = _internal_pred(["joy_excitement", "pride_confidence"])
        ie, _ = compute_internal_error(pred, _measured(["anger_hostility"]))
        assert ie == 1.0

    def test_partial_overlap(self):
        pred = _internal_pred(["joy_excitement", "pride_confidence"])
        ie, _ = compute_internal_error(pred, _measured(["joy_excitement", "anger_hostility"]))
        assert 0.0 < ie < 1.0

    def test_geometric_small_when_vad_close(self):
        pred = _internal_pred(["joy_excitement"], v=0.6, a=0.7, d=0.6, c=0.5)
        _, ge = compute_internal_error(pred, _measured(["joy_excitement"], v=0.62, a=0.71, d=0.6, c=0.5))
        assert ge < 0.05

    def test_geometric_large_when_vad_far(self):
        pred = _internal_pred(["joy_excitement"], v=0.8, a=0.8, d=0.6, c=0.5)
        _, ge = compute_internal_error(pred, _measured(["anger_hostility"], v=-0.8, a=0.9, d=0.2, c=0.5))
        assert ge > 0.4

    def test_none_prediction_returns_zeros(self):
        assert compute_internal_error(None, _measured(["joy_excitement"])) == (0.0, 0.0)

    def test_none_measured_returns_zeros(self):
        assert compute_internal_error(_internal_pred(["joy_excitement"]), None) == (0.0, 0.0)

    def test_empty_measured_returns_zeros(self):
        empty = InternalEmotionState(top_5_emotions=[])
        assert compute_internal_error(_internal_pred(["joy_excitement"]), empty) == (0.0, 0.0)


class TestUpdateInternalPrecision:
    """F3 — precisión bayesiana del canal interno (residual)."""

    def test_low_error_raises_precision(self):
        ps = default_predictive_state()
        p0 = ps.internal_precision
        update_internal_precision(ps, 0.05)
        assert ps.internal_precision > p0

    def test_high_error_lowers_precision(self):
        ps = default_predictive_state()
        ps.internal_precision = 0.6
        update_internal_precision(ps, 0.9)
        assert ps.internal_precision < 0.6

    def test_precision_clamped(self):
        ps = default_predictive_state()
        for _ in range(100):
            update_internal_precision(ps, 0.0)
        assert ps.internal_precision <= 0.95

    def test_precision_improves_over_consistent_hits(self):
        ps = default_predictive_state()
        start = ps.internal_precision
        for _ in range(10):
            update_internal_precision(ps, 0.05)
        assert ps.internal_precision > start + 0.2


class TestMergeInternalError:
    """F3 Paso 3.2 — merge_internal_error (weight 0.4 content / 0.6 internal)."""

    def test_sets_has_internal(self):
        merged = merge_internal_error(PredictionError(content_error=0.2, total_error=0.2), 0.5, 0.5, 0.5)
        assert merged.has_internal is True
        assert merged.internal_error == 0.5
        assert merged.geometric_error == 0.5

    def test_total_reweighted(self):
        # internal_component = 0.5*0.8 + 0.5*0.6 = 0.7; total = 0.4*0.2 + 0.6*0.7 = 0.5
        merged = merge_internal_error(PredictionError(total_error=0.2, vulnerability=0.1), 0.8, 0.6, 0.5)
        assert abs(merged.total_error - 0.5) < 0.001

    def test_returns_new_object(self):
        err = PredictionError(total_error=0.2)
        merged = merge_internal_error(err, 0.5, 0.5, 0.5)
        assert merged is not err
        assert err.has_internal is False

    def test_surprise_none_promoted_to_neutral_on_divergence(self):
        err = PredictionError(total_error=0.05, surprise_type=SurpriseType.NONE)
        merged = merge_internal_error(err, 0.9, 0.9, 0.5)
        assert merged.surprise_type == SurpriseType.NEUTRAL

    def test_low_internal_keeps_surprise_none(self):
        err = PredictionError(total_error=0.05, surprise_type=SurpriseType.NONE)
        merged = merge_internal_error(err, 0.0, 0.0, 0.5)
        assert merged.surprise_type == SurpriseType.NONE


class TestRawExtremeInvariantPreserved:
    """F3 — extender error_internal NO debe alterar la amplificación de modo.

    Invariante (feedback_raw_extreme_invariants): Raw x1.3, Extreme x1.6 + vuln=1.
    """

    def test_raw_amp_unchanged_after_merge(self):
        err = PredictionError(
            total_error=0.5, valence_direction=-0.5, vulnerability=0.5,
            surprise_type=SurpriseType.NEGATIVE,
        )
        merged = merge_internal_error(err, 0.6, 0.6, 0.5)
        adv = prediction_error_to_emotion_modulation(merged, 0.5)
        raw = prediction_error_to_emotion_modulation(merged, 0.5, raw_mode=True)
        assert abs(raw.valence_delta - adv.valence_delta * 1.3) < 0.001
        assert abs(raw.arousal_delta - adv.arousal_delta * 1.3) < 0.001

    def test_extreme_forces_vuln_one_and_amp(self):
        err = PredictionError(
            total_error=0.5, valence_direction=-0.5, vulnerability=0.5,
            surprise_type=SurpriseType.NEGATIVE,
        )
        merged = merge_internal_error(err, 0.6, 0.6, 0.5)
        ext = prediction_error_to_emotion_modulation(merged, 0.5, extreme_mode=True)
        # NEGATIVE i_delta = vuln*0.4; extreme forces vuln=1, mode_amp=1.6, weight=0.5
        assert abs(ext.intensity_delta - round(1.0 * 0.4 * 0.5 * 1.6, 4)) < 0.001


class TestF3FallbackToV5:
    """F3 — sin medición F2, la modulación es idéntica a v5 (texto-based)."""

    def test_no_internal_modulation_is_v5(self):
        err = PredictionError(
            total_error=0.4, valence_direction=-0.4, vulnerability=0.3,
            surprise_type=SurpriseType.NEGATIVE,
        )
        mod = prediction_error_to_emotion_modulation(err, 0.5)
        assert err.has_internal is False
        assert mod.valence_delta != 0.0

    def test_cold_start_still_zero(self):
        err = PredictionError(total_error=0.4, surprise_type=SurpriseType.NEGATIVE)
        mod = prediction_error_to_emotion_modulation(err, 0.0)
        assert mod.valence_delta == 0.0
        assert mod.arousal_delta == 0.0
        assert mod.intensity_delta == 0.0


class TestF3ContagionRouting:
    """F2.3.6 / GAP 6 — contagion via orthogonal other-speaker measurement.

    When predict_internal_state receives an `other_state` (the measured
    user emotion from the orthogonal other-speaker subspace), it must use
    that as the contagion signal instead of the text-based predicted_emotion.
    """

    def _measured_other(self, v: float, a: float) -> InternalEmotionState:
        return InternalEmotionState(
            top_5_emotions=[EmotionProjection(
                emotion_name="x", cluster="anger_hostility", cosine_sim=0.4, raw_activation=8.0,
            )],
            measured_valence=v, measured_arousal=a, measured_dominance=0.5, measured_certainty=0.5,
        )

    def test_none_other_state_uses_text_proxy(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.4, arousal=0.5)
        pe = EmotionPrediction(expected_valence=0.4, expected_arousal=0.5)
        # Regression: default None keeps the pre-F2.3.6 behavior.
        pred = predict_internal_state(ps, es, es.mood, pe, other_state=None)
        assert -1 <= pred.predicted_valence <= 1

    def test_other_state_overrides_text_proxy(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.4, arousal=0.5)
        # Text proxy says positive user; measured other says strongly negative.
        pe = EmotionPrediction(expected_valence=0.8, expected_arousal=0.4)
        with_text = predict_internal_state(ps, es, es.mood, pe, other_state=None)
        with_other = predict_internal_state(
            ps, es, es.mood, pe, other_state=self._measured_other(-0.8, 0.85),
        )
        # The measured negative user should pull the agent's predicted valence
        # below what the positive text proxy produced.
        assert with_other.predicted_valence < with_text.predicted_valence

    def test_other_state_shifts_arousal(self):
        ps = default_predictive_state()
        es = EmotionalState(valence=0.2, arousal=0.3)
        pe = EmotionPrediction(expected_valence=0.2, expected_arousal=0.3)
        high_arousal_other = predict_internal_state(
            ps, es, es.mood, pe, other_state=self._measured_other(-0.5, 0.9),
        )
        low_arousal_other = predict_internal_state(
            ps, es, es.mood, pe, other_state=self._measured_other(-0.5, 0.1),
        )
        assert high_arousal_other.predicted_arousal > low_arousal_other.predicted_arousal
