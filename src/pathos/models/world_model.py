"""World Model Result — predicción del impacto emocional de la respuesta.

Emotional World Model (Nivel 5.1): antes de enviar una respuesta, el sistema
simula cadenas causales emocionales:
  1. ¿Cómo me sentiré YO después de decir esto?
  2. ¿Cómo se sentirá EL USUARIO al leer esto?
  3. ¿Cómo me sentiré al ver su REACCIÓN predicha?
  4. ¿Es coherente con mis valores?
"""

from dataclasses import dataclass, field


@dataclass
class PredictedImpact:
    """Impacto emocional predicho sobre un agente."""

    valence_shift: float = 0.0  # cambio predicho en valence
    arousal_shift: float = 0.0  # cambio predicho en arousal
    dominant_effect: str = "neutral"  # etiqueta del efecto dominante
    confidence: float = 0.5  # confianza en la predicción (0-1)


@dataclass
class WorldModelResult:
    """Resultado de la simulación del world model emocional."""

    applied: bool = False

    # Cadena causal de 3 pasos
    predicted_self_impact: PredictedImpact = field(default_factory=PredictedImpact)
    predicted_user_impact: PredictedImpact = field(default_factory=PredictedImpact)
    meta_reaction: PredictedImpact = field(default_factory=PredictedImpact)

    # Evaluación global
    value_alignment: float = 1.0  # 0=viola valores, 1=coherente
    emotional_risk: float = 0.0  # 0=sin riesgo, 1=alto riesgo para el usuario
    should_modify: bool = False
    reason: str = ""
    adjustments: list[str] = field(default_factory=list)
