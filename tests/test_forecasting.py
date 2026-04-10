"""Tests para Emotional Forecasting.

Verifica:
- Estimación de emoción del usuario
- Predicción de impacto emocional
- Detección de riesgos
- Evaluación de accuracy y calibración
- Registro de predicciones
- Generación de prompts para behavior modifier
- Modularidad: todo funciona solo cuando está habilitado
"""

import pytest

from pathos.engine.forecasting import (
    estimate_user_emotion,
    evaluate_forecast,
    forecast_impact,
    get_forecast_prompt,
    record_forecast,
)
from pathos.models.contagion import ShadowState
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.forecasting import (
    ForecastRecord,
    ForecastResult,
    ForecastState,
    UserEmotionEstimate,
    default_forecast_state,
)
from pathos.models.social import UserModel


# --- Helpers ---

def _shadow(valence: float = 0.0, arousal: float = 0.3, signal: float = 0.0,
            accumulated: float = 0.0) -> ShadowState:
    return ShadowState(
        valence=valence, arousal=arousal,
        signal_strength=signal, accumulated_contagion=accumulated,
    )


def _user_model(intent: float = 0.3, engagement: float = 0.5,
                rapport: float = 0.3, interactions: int = 0) -> UserModel:
    return UserModel(
        perceived_intent=intent,
        perceived_engagement=engagement,
        rapport=rapport,
        interaction_count=interactions,
    )


def _agent_state(
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    valence: float = 0.0, arousal: float = 0.3,
    intensity: float = 0.3, warmth: float = 0.5,
    tension: float = 0.3,
) -> EmotionalState:
    from pathos.models.emotion import BodyState, Mood, MoodLabel
    return EmotionalState(
        valence=valence, arousal=arousal,
        dominance=0.5, certainty=0.5,
        primary_emotion=emotion, intensity=intensity,
        body_state=BodyState(energy=0.5, tension=tension, openness=0.5, warmth=warmth),
        mood=Mood(baseline_valence=0.0, baseline_arousal=0.3, label=MoodLabel.NEUTRAL),
    )


def _user_estimate(valence: float = 0.0, arousal: float = 0.3,
                   confidence: float = 0.3, signal: str = "neutral") -> UserEmotionEstimate:
    return UserEmotionEstimate(
        valence=valence, arousal=arousal,
        confidence=confidence, dominant_signal=signal,
    )


# =========================================================================
# Test: estimate_user_emotion
# =========================================================================

class TestEstimateUserEmotion:
    """Tests para la estimación del estado emocional del usuario."""

    def test_neutral_no_signal(self) -> None:
        """Sin señal, devuelve estimación neutral."""
        result = estimate_user_emotion(
            shadow_state=_shadow(),
            user_model=_user_model(),
            detected_valence=0.0, detected_arousal=0.3,
            signal_strength=0.0,
        )
        assert result.dominant_signal == "neutral"
        assert result.confidence < 0.5

    def test_positive_signal_detected(self) -> None:
        """Señal positiva fuerte detectada → estimación positiva."""
        result = estimate_user_emotion(
            shadow_state=_shadow(valence=0.3, arousal=0.4, signal=0.5),
            user_model=_user_model(),
            detected_valence=0.6, detected_arousal=0.4,
            signal_strength=0.6,
        )
        assert result.valence > 0.2
        assert result.dominant_signal in ("positive", "positive_high")

    def test_negative_signal_detected(self) -> None:
        """Señal negativa fuerte → estimación negativa."""
        result = estimate_user_emotion(
            shadow_state=_shadow(valence=-0.4, arousal=0.5, signal=0.6),
            user_model=_user_model(),
            detected_valence=-0.6, detected_arousal=0.7,
            signal_strength=0.7,
        )
        assert result.valence < -0.2
        assert result.dominant_signal in ("negative", "distressed")

    def test_distressed_signal(self) -> None:
        """Señal negativa con alto arousal → distressed."""
        result = estimate_user_emotion(
            shadow_state=_shadow(valence=-0.5, arousal=0.8, signal=0.7),
            user_model=_user_model(),
            detected_valence=-0.6, detected_arousal=0.8,
            signal_strength=0.7,
        )
        assert result.dominant_signal == "distressed"

    def test_shadow_state_inertia(self) -> None:
        """Shadow state proporciona inercia — no cambia bruscamente."""
        result = estimate_user_emotion(
            shadow_state=_shadow(valence=0.5, arousal=0.4, signal=0.3),
            user_model=_user_model(),
            detected_valence=-0.5, detected_arousal=0.7,
            signal_strength=0.2,  # señal débil
        )
        # Shadow state positivo debería dominar sobre señal débil negativa
        assert result.valence > 0.0

    def test_high_engagement_boosts_arousal(self) -> None:
        """Alto engagement del usuario → boost de arousal estimado."""
        low_eng = estimate_user_emotion(
            shadow_state=_shadow(valence=0.0, arousal=0.3),
            user_model=_user_model(engagement=0.2),
            detected_valence=0.0, detected_arousal=0.3,
            signal_strength=0.0,
        )
        high_eng = estimate_user_emotion(
            shadow_state=_shadow(valence=0.0, arousal=0.3),
            user_model=_user_model(engagement=0.9),
            detected_valence=0.0, detected_arousal=0.3,
            signal_strength=0.0,
        )
        assert high_eng.arousal > low_eng.arousal

    def test_positive_intent_boosts_valence(self) -> None:
        """Intent positivo del usuario → ligero boost de valence."""
        neg_intent = estimate_user_emotion(
            shadow_state=_shadow(),
            user_model=_user_model(intent=-0.5),
            detected_valence=0.0, detected_arousal=0.3,
            signal_strength=0.0,
        )
        pos_intent = estimate_user_emotion(
            shadow_state=_shadow(),
            user_model=_user_model(intent=0.8),
            detected_valence=0.0, detected_arousal=0.3,
            signal_strength=0.0,
        )
        assert pos_intent.valence > neg_intent.valence

    def test_confidence_increases_with_interactions(self) -> None:
        """Más interacciones → mayor confianza en la estimación."""
        new_user = estimate_user_emotion(
            shadow_state=_shadow(signal=0.3, accumulated=0.2),
            user_model=_user_model(interactions=1),
            detected_valence=0.3, detected_arousal=0.4,
            signal_strength=0.3,
        )
        veteran = estimate_user_emotion(
            shadow_state=_shadow(signal=0.3, accumulated=0.2),
            user_model=_user_model(interactions=10),
            detected_valence=0.3, detected_arousal=0.4,
            signal_strength=0.3,
        )
        assert veteran.confidence > new_user.confidence

    def test_confidence_capped_at_1(self) -> None:
        """Confianza nunca excede 1.0."""
        result = estimate_user_emotion(
            shadow_state=_shadow(signal=1.0, accumulated=1.0),
            user_model=_user_model(interactions=100),
            detected_valence=0.5, detected_arousal=0.5,
            signal_strength=1.0,
        )
        assert result.confidence <= 1.0

    def test_valence_clamped(self) -> None:
        """Valence estimada siempre en [-1, 1]."""
        result = estimate_user_emotion(
            shadow_state=_shadow(valence=0.9),
            user_model=_user_model(intent=1.0),
            detected_valence=0.9, detected_arousal=0.5,
            signal_strength=0.9,
        )
        assert -1.0 <= result.valence <= 1.0

    def test_arousal_clamped(self) -> None:
        """Arousal estimada siempre en [0, 1]."""
        result = estimate_user_emotion(
            shadow_state=_shadow(arousal=0.95),
            user_model=_user_model(engagement=1.0),
            detected_valence=0.0, detected_arousal=0.95,
            signal_strength=0.9,
        )
        assert 0.0 <= result.arousal <= 1.0

    def test_ambiguous_signal(self) -> None:
        """Señal con valence cercana a zero → ambiguous."""
        result = estimate_user_emotion(
            shadow_state=_shadow(valence=0.05, arousal=0.4, signal=0.4),
            user_model=_user_model(),
            detected_valence=0.1, detected_arousal=0.4,
            signal_strength=0.4,
        )
        assert result.dominant_signal == "ambiguous"


# =========================================================================
# Test: forecast_impact
# =========================================================================

class TestForecastImpact:
    """Tests para la predicción de impacto emocional."""

    def test_joy_positive_impact(self) -> None:
        """Agente con joy → impacto positivo en usuario."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.6, intensity=0.6),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        assert result.predicted_impact > 0
        assert result.predicted_user_valence > 0

    def test_anger_negative_impact(self) -> None:
        """Agente con anger → impacto negativo en usuario."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.ANGER, valence=-0.5, intensity=0.7),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        assert result.predicted_impact < 0

    def test_neutral_minimal_impact(self) -> None:
        """Agente neutral → impacto mínimo."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.NEUTRAL, intensity=0.2),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        assert abs(result.predicted_impact) < 0.1

    def test_higher_intensity_more_impact(self) -> None:
        """Mayor intensidad del agente → mayor impacto."""
        low = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.4, intensity=0.2),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        high = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.4, intensity=0.9),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        assert abs(high.predicted_impact) > abs(low.predicted_impact)

    def test_rapport_amplifies_impact(self) -> None:
        """Mayor rapport → mayor impacto (positivo o negativo)."""
        low_rapport = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.5, intensity=0.5),
            user_emotion=_user_estimate(),
            user_model=_user_model(rapport=0.1),
        )
        high_rapport = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.5, intensity=0.5),
            user_emotion=_user_estimate(),
            user_model=_user_model(rapport=0.9),
        )
        assert high_rapport.predicted_impact > low_rapport.predicted_impact

    def test_vulnerable_user_amplified_negative(self) -> None:
        """Usuario ya negativo → impacto negativo amplificado."""
        neutral_user = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.FRUSTRATION, valence=-0.3, intensity=0.5),
            user_emotion=_user_estimate(valence=0.0),
            user_model=_user_model(),
        )
        vulnerable_user = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.FRUSTRATION, valence=-0.3, intensity=0.5),
            user_emotion=_user_estimate(valence=-0.5),
            user_model=_user_model(),
        )
        assert vulnerable_user.predicted_impact < neutral_user.predicted_impact

    def test_warmth_mitigates_negative(self) -> None:
        """Alta warmth del agente → mitiga impacto negativo."""
        cold = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.SADNESS, valence=-0.4, intensity=0.5, warmth=0.3),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        warm = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.SADNESS, valence=-0.4, intensity=0.5, warmth=0.8),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        # warm should have less negative impact
        assert warm.predicted_impact > cold.predicted_impact

    def test_tension_amplifies_negative(self) -> None:
        """Alta tensión del agente → amplifica impacto negativo."""
        relaxed = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.ANGER, valence=-0.5, intensity=0.6, tension=0.2),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        tense = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.ANGER, valence=-0.5, intensity=0.6, tension=0.8),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
        )
        assert tense.predicted_impact < relaxed.predicted_impact

    def test_predicted_values_clamped(self) -> None:
        """Valores predichos siempre en rango."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.9, intensity=1.0),
            user_emotion=_user_estimate(valence=0.8, arousal=0.9),
            user_model=_user_model(rapport=1.0),
        )
        assert -1.0 <= result.predicted_user_valence <= 1.0
        assert 0.0 <= result.predicted_user_arousal <= 1.0
        assert -1.0 <= result.predicted_impact <= 1.0

    def test_bias_applied(self) -> None:
        """Bias de calibración se aplica al impacto."""
        no_bias = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.4, intensity=0.5),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
            valence_bias=0.0,
        )
        with_bias = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.4, intensity=0.5),
            user_emotion=_user_estimate(),
            user_model=_user_model(),
            valence_bias=0.1,
        )
        assert with_bias.predicted_user_valence > no_bias.predicted_user_valence


# =========================================================================
# Test: Risk Detection
# =========================================================================

class TestRiskDetection:
    """Tests para la detección de riesgos emocionales."""

    def test_risk_negative_user_negative_agent(self) -> None:
        """Usuario negativo + agente empeora → risk flag."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.ANGER, valence=-0.5, intensity=0.6),
            user_emotion=_user_estimate(valence=-0.3, signal="negative"),
            user_model=_user_model(),
        )
        assert result.risk_flag is True
        assert "negativo" in result.risk_reason.lower() or "empeorar" in result.risk_reason.lower()
        assert result.recommendation != ""

    def test_risk_intense_agent_calm_user(self) -> None:
        """Agente muy intenso + usuario calmado → risk flag."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.EXCITEMENT, valence=0.6, intensity=0.85),
            user_emotion=_user_estimate(arousal=0.15),
            user_model=_user_model(),
        )
        assert result.risk_flag is True
        assert "intensidad" in result.risk_reason.lower() or "abrumar" in result.risk_reason.lower()

    def test_risk_indifferent_to_distressed(self) -> None:
        """Agente indiferente + usuario en distress → risk flag."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.INDIFFERENCE, valence=0.0, intensity=0.3),
            user_emotion=_user_estimate(valence=-0.4, signal="distressed"),
            user_model=_user_model(),
        )
        assert result.risk_flag is True
        assert "indiferencia" in result.risk_reason.lower() or "empatía" in result.risk_reason.lower()

    def test_risk_emotional_discordance(self) -> None:
        """Agente positivo + usuario triste → discordancia."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.7, intensity=0.6),
            user_emotion=_user_estimate(valence=-0.5, confidence=0.5, signal="negative"),
            user_model=_user_model(),
        )
        assert result.risk_flag is True
        assert "discordancia" in result.risk_reason.lower()

    def test_no_risk_positive_interaction(self) -> None:
        """Interacción positiva normal → sin riesgo."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.CONTENTMENT, valence=0.3, intensity=0.3),
            user_emotion=_user_estimate(valence=0.2),
            user_model=_user_model(),
        )
        assert result.risk_flag is False

    def test_positive_recommendation_when_good(self) -> None:
        """Impacto positivo significativo → recomendación positiva."""
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.JOY, valence=0.6, intensity=0.7),
            user_emotion=_user_estimate(valence=0.1),
            user_model=_user_model(rapport=0.5),
        )
        assert result.risk_flag is False
        assert "bien recibido" in result.recommendation.lower()


# =========================================================================
# Test: evaluate_forecast (accuracy tracking + calibration)
# =========================================================================

class TestEvaluateForecast:
    """Tests para la evaluación de accuracy y calibración."""

    def test_evaluate_first_forecast(self) -> None:
        """Primera evaluación establece accuracy score."""
        state = default_forecast_state()
        state.enabled = True
        state.last_forecast = ForecastResult(
            predicted_user_valence=0.3,
            predicted_user_arousal=0.4,
        )
        state.history = [ForecastRecord(turn=1, predicted_valence=0.3, predicted_arousal=0.4)]

        state = evaluate_forecast(state, actual_valence=0.25, actual_arousal=0.35, turn=2)

        assert state.total_evaluated == 1
        assert state.history[0].actual_valence == 0.25
        assert state.history[0].actual_arousal == 0.35
        assert state.history[0].error is not None
        assert state.accuracy_score > 0.5  # predicción bastante buena

    def test_evaluate_updates_bias(self) -> None:
        """Evaluación ajusta bias de calibración."""
        state = default_forecast_state()
        state.enabled = True
        state.last_forecast = ForecastResult(
            predicted_user_valence=0.3,
            predicted_user_arousal=0.4,
        )
        state.history = [ForecastRecord(turn=1, predicted_valence=0.3, predicted_arousal=0.4)]

        # Real: mucho más positivo de lo predicho → bias debería subir
        state = evaluate_forecast(state, actual_valence=0.6, actual_arousal=0.4, turn=2)

        assert state.valence_bias > 0  # corregir hacia arriba

    def test_evaluate_negative_bias_correction(self) -> None:
        """Si predecimos demasiado positivo, bias se corrige hacia abajo."""
        state = default_forecast_state()
        state.enabled = True
        state.last_forecast = ForecastResult(predicted_user_valence=0.5, predicted_user_arousal=0.4)
        state.history = [ForecastRecord(turn=1, predicted_valence=0.5, predicted_arousal=0.4)]

        state = evaluate_forecast(state, actual_valence=-0.1, actual_arousal=0.4, turn=2)

        assert state.valence_bias < 0

    def test_evaluate_no_forecast_noop(self) -> None:
        """Sin forecast previo → no hace nada."""
        state = default_forecast_state()
        original = state.model_copy(deep=True)
        state = evaluate_forecast(state, actual_valence=0.5, actual_arousal=0.5, turn=2)
        assert state.total_evaluated == original.total_evaluated

    def test_evaluate_no_pending_noop(self) -> None:
        """Sin registro pendiente → no hace nada."""
        state = default_forecast_state()
        state.last_forecast = ForecastResult()
        # History vacío o todos evaluados
        state.history = [ForecastRecord(turn=1, predicted_valence=0.3, predicted_arousal=0.4,
                                        actual_valence=0.3, actual_arousal=0.4, error=0.0)]
        state = evaluate_forecast(state, actual_valence=0.5, actual_arousal=0.5, turn=2)
        assert state.total_evaluated == 0

    def test_bias_clamped(self) -> None:
        """Bias nunca excede [-0.3, 0.3]."""
        state = default_forecast_state()
        state.enabled = True
        state.valence_bias = 0.28

        state.last_forecast = ForecastResult(predicted_user_valence=0.0, predicted_user_arousal=0.3)
        state.history = [ForecastRecord(turn=1, predicted_valence=0.0, predicted_arousal=0.3)]

        state = evaluate_forecast(state, actual_valence=0.9, actual_arousal=0.3, turn=2)

        assert state.valence_bias <= 0.3

    def test_accuracy_uses_ema(self) -> None:
        """Accuracy score usa EMA — predicciones recientes pesan más."""
        state = default_forecast_state()
        state.enabled = True
        state.total_evaluated = 5
        state.accuracy_score = 0.8

        state.last_forecast = ForecastResult(predicted_user_valence=0.5, predicted_user_arousal=0.5)
        state.history = [ForecastRecord(turn=6, predicted_valence=0.5, predicted_arousal=0.5)]

        # Predicción perfecta
        state = evaluate_forecast(state, actual_valence=0.5, actual_arousal=0.5, turn=7)

        # accuracy debería subir (EMA con peso 0.3 para nuevo dato)
        assert state.accuracy_score > 0.8


# =========================================================================
# Test: record_forecast
# =========================================================================

class TestRecordForecast:
    """Tests para el registro de predicciones."""

    def test_record_basic(self) -> None:
        """Registra forecast correctamente."""
        state = default_forecast_state()
        forecast = ForecastResult(
            predicted_user_valence=0.3,
            predicted_user_arousal=0.4,
            predicted_impact=0.2,
        )
        state = record_forecast(state, forecast, turn=1)

        assert state.last_forecast == forecast
        assert state.total_forecasts == 1
        assert len(state.history) == 1
        assert state.history[0].turn == 1
        assert state.history[0].predicted_valence == 0.3

    def test_record_increments_count(self) -> None:
        """Cada forecast incrementa el contador."""
        state = default_forecast_state()
        for i in range(5):
            state = record_forecast(state, ForecastResult(), turn=i + 1)
        assert state.total_forecasts == 5

    def test_record_max_history(self) -> None:
        """Historial no excede 20 registros."""
        state = default_forecast_state()
        for i in range(25):
            state = record_forecast(state, ForecastResult(), turn=i + 1)
        assert len(state.history) <= 20

    def test_record_keeps_recent(self) -> None:
        """Al truncar, mantiene los más recientes."""
        state = default_forecast_state()
        for i in range(25):
            state = record_forecast(state, ForecastResult(), turn=i + 1)
        assert state.history[0].turn > 1  # los primeros se eliminaron
        assert state.history[-1].turn == 25


# =========================================================================
# Test: get_forecast_prompt
# =========================================================================

class TestGetForecastPrompt:
    """Tests para la generación de prompts del behavior modifier."""

    def test_none_forecast_returns_none(self) -> None:
        """Sin forecast → None."""
        assert get_forecast_prompt(None) is None

    def test_risk_generates_alert(self) -> None:
        """Con risk → genera alerta."""
        forecast = ForecastResult(
            risk_flag=True,
            risk_reason="El usuario está triste y tu tono puede empeorar",
            recommendation="Sé más cálido",
        )
        prompt = get_forecast_prompt(forecast)
        assert prompt is not None
        assert "ALERTA" in prompt
        assert "triste" in prompt
        assert "cálido" in prompt

    def test_positive_recommendation(self) -> None:
        """Sin riesgo pero con recomendación → la incluye."""
        forecast = ForecastResult(
            risk_flag=False,
            recommendation="Tu estado emocional actual probablemente será bien recibido",
        )
        prompt = get_forecast_prompt(forecast)
        assert prompt is not None
        assert "bien recibido" in prompt

    def test_no_risk_no_recommendation_returns_none(self) -> None:
        """Sin riesgo ni recomendación → None."""
        forecast = ForecastResult()
        assert get_forecast_prompt(forecast) is None


# =========================================================================
# Test: default_forecast_state
# =========================================================================

class TestForecastState:
    """Tests para el estado de forecasting."""

    def test_default_disabled(self) -> None:
        """Por defecto, forecasting está desactivado."""
        state = default_forecast_state()
        assert state.enabled is False

    def test_default_no_history(self) -> None:
        """Por defecto, sin historial."""
        state = default_forecast_state()
        assert len(state.history) == 0
        assert state.total_forecasts == 0
        assert state.total_evaluated == 0

    def test_default_no_bias(self) -> None:
        """Por defecto, sin bias de calibración."""
        state = default_forecast_state()
        assert state.valence_bias == 0.0
        assert state.arousal_bias == 0.0

    def test_enable_toggle(self) -> None:
        """Se puede activar/desactivar."""
        state = default_forecast_state()
        state.enabled = True
        assert state.enabled is True
        state.enabled = False
        assert state.enabled is False


# =========================================================================
# Test: Integration scenarios
# =========================================================================

class TestIntegrationScenarios:
    """Tests de escenarios end-to-end del forecasting."""

    def test_full_cycle_estimate_forecast_evaluate(self) -> None:
        """Ciclo completo: estimar → predecir → evaluar."""
        state = default_forecast_state()
        state.enabled = True

        # Turno 1: estimar usuario + predecir impacto
        user_est = estimate_user_emotion(
            shadow_state=_shadow(valence=-0.2, arousal=0.4, signal=0.4),
            user_model=_user_model(engagement=0.6),
            detected_valence=-0.3, detected_arousal=0.5,
            signal_strength=0.4,
        )
        state.user_emotion = user_est

        forecast = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.GRATITUDE, valence=0.4, intensity=0.5),
            user_emotion=user_est,
            user_model=_user_model(rapport=0.4),
        )
        state = record_forecast(state, forecast, turn=1)

        assert state.total_forecasts == 1
        assert forecast.predicted_impact > 0  # gratitude debería mejorar

        # Turno 2: evaluar con reacción real
        state = evaluate_forecast(state, actual_valence=0.1, actual_arousal=0.35, turn=2)

        assert state.total_evaluated == 1
        assert state.history[0].error is not None

    def test_calibration_improves_over_time(self) -> None:
        """Bias se ajusta consistentemente si hay error sistemático."""
        state = default_forecast_state()
        state.enabled = True

        # Simular 5 turnos donde siempre predecimos menos positivo de lo real
        for turn in range(1, 6):
            forecast = ForecastResult(
                predicted_user_valence=0.1,
                predicted_user_arousal=0.3,
                predicted_impact=0.1,
            )
            state = record_forecast(state, forecast, turn=turn)
            # Real: siempre más positivo
            state = evaluate_forecast(state, actual_valence=0.4, actual_arousal=0.3, turn=turn + 1)

        # Bias debería haberse corregido positivamente
        assert state.valence_bias > 0.05

    def test_disabled_state_produces_no_output(self) -> None:
        """Con forecasting desactivado, no hay output."""
        state = default_forecast_state()
        assert state.enabled is False
        assert state.last_forecast is None
        assert get_forecast_prompt(state.last_forecast) is None

    def test_gratitude_helps_negative_user(self) -> None:
        """Gratitude del agente mejora usuario negativo, sin riesgo."""
        user_est = _user_estimate(valence=-0.15, arousal=0.4)
        result = forecast_impact(
            agent_state=_agent_state(PrimaryEmotion.GRATITUDE, valence=0.5, intensity=0.5, warmth=0.7),
            user_emotion=user_est,
            user_model=_user_model(rapport=0.5),
        )
        assert result.predicted_impact > 0
        assert result.risk_flag is False

    def test_all_emotions_have_impact_mapping(self) -> None:
        """Todas las 19 emociones tienen mapeo de impacto."""
        for emotion in PrimaryEmotion:
            result = forecast_impact(
                agent_state=_agent_state(emotion, intensity=0.5),
                user_emotion=_user_estimate(),
                user_model=_user_model(),
            )
            assert -1.0 <= result.predicted_impact <= 1.0
            assert -1.0 <= result.predicted_user_valence <= 1.0
            assert 0.0 <= result.predicted_user_arousal <= 1.0
