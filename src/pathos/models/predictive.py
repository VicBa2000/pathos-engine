"""Predictive Processing - Modelos de datos.

Pilar 1 de ANIMA: Emociones como errores de predicción.
Basado en Karl Friston (Free Energy Principle, 2010),
Lisa Feldman Barrett (Theory of Constructed Emotion, 2017),
Andy Clark (Surfing Uncertainty, 2015).

El sistema genera predicciones ANTES de recibir el input del usuario.
La emoción emerge de la discrepancia entre predicción y realidad,
ponderada por la precisión (confianza) del modelo predictivo.

Sistema CORE (siempre activo): mejora silenciosamente la calidad
emocional sin requerir activación explícita. En cold start (sin
historial) degrada gracefully a 100% appraisal clásico.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SurpriseType(str, Enum):
    """Tipo de sorpresa derivada del prediction error."""

    POSITIVE = "positive"       # Mejor de lo esperado
    NEGATIVE = "negative"       # Peor de lo esperado
    NEUTRAL = "neutral"         # Inesperado pero no valorado
    NONE = "none"               # Predicción correcta


class DemandType(str, Enum):
    """Tipo de demanda social predicha/detectada."""

    HELP = "help"               # Pide ayuda o consejo
    CONVERSATION = "conversation"  # Quiere hablar / socializar
    CHALLENGE = "challenge"     # Desafía, cuestiona, debate
    EMOTIONAL = "emotional"     # Busca apoyo emocional
    TASK = "task"               # Solicita tarea concreta
    UNKNOWN = "unknown"         # No se puede determinar


# ---------------------------------------------------------------------------
# Predicciones individuales
# ---------------------------------------------------------------------------

class ContentPrediction(BaseModel):
    """Predicción sobre el contenido probable del próximo mensaje.

    No predice texto literal — predice temática, intención y tono.
    """

    expected_topic: str = Field(
        default="unknown",
        description="Tema probable (ej: 'continuación del tema anterior', 'cambio de tema')",
    )
    expected_tone: Literal["positive", "negative", "neutral", "mixed"] = Field(
        default="neutral",
        description="Tono esperado del mensaje",
    )
    expected_intent: Literal[
        "continue", "question", "request", "complaint", "greeting",
        "farewell", "emotional_expression", "topic_change", "unknown",
    ] = Field(
        default="unknown",
        description="Intención probable del usuario",
    )
    confidence: float = Field(
        default=0.3, ge=0, le=1,
        description="Confianza en esta predicción (inicia baja, crece con aciertos)",
    )


class EmotionPrediction(BaseModel):
    """Predicción del estado emocional probable del usuario.

    Basado en la tendencia observada + modelo del usuario.
    """

    expected_valence: float = Field(
        default=0.0, ge=-1, le=1,
        description="Valence esperada del usuario",
    )
    expected_arousal: float = Field(
        default=0.3, ge=0, le=1,
        description="Arousal esperado del usuario",
    )
    confidence: float = Field(
        default=0.3, ge=0, le=1,
        description="Confianza en esta predicción",
    )


class DemandPrediction(BaseModel):
    """Predicción de la demanda social probable.

    ¿Qué espera el usuario de esta interacción?
    """

    expected_demand: DemandType = Field(
        default=DemandType.UNKNOWN,
        description="Tipo de demanda social esperada",
    )
    confidence: float = Field(
        default=0.3, ge=0, le=1,
        description="Confianza en esta predicción",
    )


# ---------------------------------------------------------------------------
# Conjunto de predicciones
# ---------------------------------------------------------------------------

class PredictionSet(BaseModel):
    """Conjunto completo de predicciones para un turno.

    Se genera ANTES de recibir el input del usuario.
    Cada predicción tiene su propia confianza independiente.
    """

    content: ContentPrediction = Field(default_factory=ContentPrediction)
    emotion: EmotionPrediction = Field(default_factory=EmotionPrediction)
    demand: DemandPrediction = Field(default_factory=DemandPrediction)
    turn: int = Field(default=0, ge=0, description="Turno en que se generó")

    @property
    def average_confidence(self) -> float:
        """Confianza promedio del conjunto."""
        return (self.content.confidence + self.emotion.confidence + self.demand.confidence) / 3.0

    @property
    def is_cold_start(self) -> bool:
        """True si todas las confianzas están en valor inicial (sin historial)."""
        return self.average_confidence <= 0.3


# ---------------------------------------------------------------------------
# Errores de predicción
# ---------------------------------------------------------------------------

class PredictionError(BaseModel):
    """Error de predicción computado tras recibir input real.

    La magnitud del error * la precisión = intensidad emocional.
    Cuanto más seguro estabas y más equivocado estás, más intenso el impacto.
    """

    # Errores por dimensión (0 = predicción perfecta)
    content_error: float = Field(
        default=0.0, ge=0, le=1,
        description="Distancia semántica entre contenido predicho y real",
    )
    emotion_error: float = Field(
        default=0.0, ge=0, le=1,
        description="Distancia euclidiana entre emoción predicha y detectada",
    )
    demand_error: float = Field(
        default=0.0, ge=0, le=1,
        description="Error en el tipo de demanda predicha",
    )

    # Error agregado ponderado
    total_error: float = Field(
        default=0.0, ge=0, le=1,
        description="Error ponderado total (promedio ponderado por precisión)",
    )

    # Dirección del error en valence (signo indica mejor/peor de lo esperado)
    valence_direction: float = Field(
        default=0.0, ge=-1, le=1,
        description="Positivo = mejor de lo esperado, negativo = peor",
    )

    # Tipo de sorpresa resultante
    surprise_type: SurpriseType = Field(default=SurpriseType.NONE)

    # Vulnerabilidad emocional (precisión del modelo predictivo)
    vulnerability: float = Field(
        default=0.0, ge=0, le=1,
        description="Alta precisión + alto error = alta vulnerabilidad emocional",
    )


# ---------------------------------------------------------------------------
# Historial de predicciones (rolling buffer)
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
    """Registro de una predicción + outcome para tracking."""

    turn: int = Field(ge=0)
    predictions: PredictionSet
    error: PredictionError | None = Field(
        default=None,
        description="None si aún no se evaluó (turno actual en progreso)",
    )


class PredictionHistory(BaseModel):
    """Buffer circular de predicciones recientes.

    Mantiene las últimas N predicciones para:
    - Computar tendencias de accuracy
    - Detectar patrones (usuario siempre cambia de tema, siempre hostil, etc.)
    - Ajustar precision bayesiana
    """

    records: list[PredictionRecord] = Field(default_factory=list)
    max_size: int = Field(default=50, ge=5, description="Tamaño máximo del buffer")

    # Accuracy tracking por dimensión
    content_accuracy: float = Field(
        default=0.5, ge=0, le=1,
        description="Accuracy acumulada en predicción de contenido",
    )
    emotion_accuracy: float = Field(
        default=0.5, ge=0, le=1,
        description="Accuracy acumulada en predicción de emoción",
    )
    demand_accuracy: float = Field(
        default=0.5, ge=0, le=1,
        description="Accuracy acumulada en predicción de demanda",
    )

    def add_record(self, record: PredictionRecord) -> None:
        """Añade un registro, eliminando el más antiguo si excede el buffer."""
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    @property
    def evaluated_count(self) -> int:
        """Número de predicciones que ya fueron evaluadas."""
        return sum(1 for r in self.records if r.error is not None)

    @property
    def average_accuracy(self) -> float:
        """Accuracy promedio entre las 3 dimensiones."""
        return (self.content_accuracy + self.emotion_accuracy + self.demand_accuracy) / 3.0


# ---------------------------------------------------------------------------
# Estado predictivo por sesión
# ---------------------------------------------------------------------------

class PredictiveState(BaseModel):
    """Estado completo del sistema predictivo por sesión.

    Se añade a SessionState. Siempre activo (CORE), pero degrada
    gracefully a appraisal clásico en cold start.
    """

    # Predicciones del turno actual (generadas ANTES del input)
    current_predictions: PredictionSet | None = Field(
        default=None,
        description="Predicciones del turno actual (None si no se generaron aún)",
    )

    # Error del turno actual (computado DESPUÉS del input)
    current_error: PredictionError | None = Field(
        default=None,
        description="Error del turno actual (None si no se computó aún)",
    )

    # Historial
    history: PredictionHistory = Field(default_factory=PredictionHistory)

    # Precisión bayesiana por dimensión (confianza del modelo predictivo)
    content_precision: float = Field(
        default=0.3, ge=0, le=1,
        description="Precisión bayesiana en contenido (crece con aciertos)",
    )
    emotion_precision: float = Field(
        default=0.3, ge=0, le=1,
        description="Precisión bayesiana en emoción",
    )
    demand_precision: float = Field(
        default=0.3, ge=0, le=1,
        description="Precisión bayesiana en demanda",
    )

    # Peso del predictive processing vs appraisal clásico
    # Empieza en 0 (100% clásico) y crece hasta max_weight con la precisión
    predictive_weight: float = Field(
        default=0.0, ge=0, le=1,
        description="Peso actual del predictive processing (0=todo clásico, 0.6=máximo)",
    )
    max_predictive_weight: float = Field(
        default=0.6, ge=0, le=1,
        description="Peso máximo del predictive processing (60% por defecto)",
    )

    @property
    def average_precision(self) -> float:
        """Precisión promedio del modelo predictivo."""
        return (self.content_precision + self.emotion_precision + self.demand_precision) / 3.0

    @property
    def is_warm(self) -> bool:
        """True si el modelo tiene suficiente historial para ser útil."""
        return self.history.evaluated_count >= 3


def default_predictive_state() -> PredictiveState:
    """Estado predictivo por defecto (cold start)."""
    return PredictiveState()
