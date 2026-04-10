"""Emotional Forecasting - Modelos de datos.

Predice cómo la respuesta del agente afectará emocionalmente al usuario.
Basado en Wilson & Gilbert (2005) "Affective Forecasting".

Sistema OPCIONAL y modular: solo se ejecuta si está activo en la sesión.
No modifica el estado emocional del agente — es puramente informativo.
"""

from pydantic import BaseModel, Field


class UserEmotionEstimate(BaseModel):
    """Estimación simplificada del estado emocional actual del usuario.

    Derivado de contagion detection (señales lingüísticas) y social cognition
    (intent, engagement). NO es un modelo completo — es lo que el agente
    "cree" que el usuario siente.
    """

    valence: float = Field(default=0.0, ge=-1, le=1, description="Estimación de valence del usuario")
    arousal: float = Field(default=0.3, ge=0, le=1, description="Estimación de arousal del usuario")
    confidence: float = Field(default=0.3, ge=0, le=1, description="Confianza en la estimación")
    dominant_signal: str = Field(default="neutral", description="Señal dominante detectada")


class ForecastResult(BaseModel):
    """Resultado de una predicción de impacto emocional.

    Predice cómo la respuesta del agente (dado su estado emocional actual)
    afectará al usuario en el turno siguiente.
    """

    predicted_user_valence: float = Field(
        default=0.0, ge=-1, le=1,
        description="Valence predicha del usuario tras la respuesta",
    )
    predicted_user_arousal: float = Field(
        default=0.3, ge=0, le=1,
        description="Arousal predicho del usuario tras la respuesta",
    )
    predicted_impact: float = Field(
        default=0.0, ge=-1, le=1,
        description="Impacto emocional neto predicho (-1=muy negativo, 1=muy positivo)",
    )
    risk_flag: bool = Field(
        default=False,
        description="True si se predice impacto negativo significativo",
    )
    risk_reason: str = Field(
        default="",
        description="Razón del riesgo si risk_flag es True",
    )
    recommendation: str = Field(
        default="",
        description="Sugerencia para el behavior modifier",
    )


class ForecastRecord(BaseModel):
    """Registro de una predicción pasada para tracking de accuracy."""

    turn: int = Field(description="Turno en que se hizo la predicción")
    predicted_valence: float = Field(description="Valence predicha")
    predicted_arousal: float = Field(description="Arousal predicho")
    actual_valence: float | None = Field(default=None, description="Valence real observada")
    actual_arousal: float | None = Field(default=None, description="Arousal real observado")
    error: float | None = Field(default=None, description="Error de predicción (MAE)")


class ForecastState(BaseModel):
    """Estado completo del sistema de forecasting por sesión."""

    enabled: bool = Field(default=False, description="Si el forecasting está activo")
    user_emotion: UserEmotionEstimate = Field(default_factory=UserEmotionEstimate)
    last_forecast: ForecastResult | None = Field(default=None)
    history: list[ForecastRecord] = Field(default_factory=list)

    # Calibración del modelo predictivo
    valence_bias: float = Field(
        default=0.0,
        description="Sesgo de calibración en valence (se ajusta con accuracy tracking)",
    )
    arousal_bias: float = Field(
        default=0.0,
        description="Sesgo de calibración en arousal",
    )
    accuracy_score: float = Field(
        default=0.5, ge=0, le=1,
        description="Accuracy acumulada del modelo (0=malo, 1=perfecto)",
    )
    total_forecasts: int = Field(default=0, ge=0)
    total_evaluated: int = Field(default=0, ge=0)


def default_forecast_state() -> ForecastState:
    """Estado de forecasting por defecto (desactivado)."""
    return ForecastState()
