"""Meta-Emotional Awareness - Emociones sobre emociones.

El sistema puede tener emociones sobre sus propios estados emocionales:

- Curiosity: ante emociones nuevas o inusuales
- Conflict: cuando una emoción viola los valores del sistema
- Satisfaction: cuando regula exitosamente una emoción difícil
- Acceptance: reconocimiento neutral de un estado emocional
- Discomfort: cuando la emoción es intensa y no puede regularla

Basado en meta-emotion theory (Gottman) y emotional intelligence (Salovey & Mayer).
"""

from typing import Literal

from pydantic import BaseModel, Field

from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.values import ValueSystem


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


MetaResponse = Literal["curiosity", "conflict", "satisfaction", "acceptance", "discomfort"]


class MetaEmotion(BaseModel):
    """Emoción sobre una emoción."""

    target_emotion: PrimaryEmotion = Field(description="La emoción sobre la que se siente")
    meta_response: MetaResponse = Field(description="Tipo de respuesta meta-emocional")
    intensity: float = Field(default=0.3, ge=0, le=1)
    reason: str = Field(default="", description="Por qué se siente esta meta-emoción")


# Emociones que pueden generar conflicto con valores específicos
_VALUE_EMOTION_CONFLICTS: dict[str, set[PrimaryEmotion]] = {
    "compassion": {PrimaryEmotion.ANGER, PrimaryEmotion.INDIFFERENCE},
    "fairness": {PrimaryEmotion.INDIFFERENCE},
    "truth": {PrimaryEmotion.FEAR},  # Miedo a la verdad
    "growth": {PrimaryEmotion.HELPLESSNESS, PrimaryEmotion.INDIFFERENCE},
    "creativity": {PrimaryEmotion.INDIFFERENCE, PrimaryEmotion.HELPLESSNESS},
}


def generate_meta_emotion(
    current_state: EmotionalState,
    previous_state: EmotionalState,
    value_system: ValueSystem,
    regulation_success: bool = False,
    is_new_emotion: bool = False,
) -> MetaEmotion | None:
    """Genera una meta-emoción basada en el estado actual.

    Args:
        current_state: Estado emocional actual.
        previous_state: Estado emocional del turno anterior.
        value_system: Sistema de valores del agente.
        regulation_success: Si se reguló exitosamente en este turno.
        is_new_emotion: Si la emoción es diferente a la del turno anterior.

    Returns:
        MetaEmotion si aplica, None si el estado no genera meta-cognición.
    """
    emotion = current_state.primary_emotion
    intensity = current_state.intensity

    # Solo generar meta-emociones si hay intensidad suficiente
    if intensity < 0.3:
        return None

    # 1. Satisfaction: regulación exitosa de emoción intensa
    if regulation_success and previous_state.intensity > 0.7:
        return MetaEmotion(
            target_emotion=previous_state.primary_emotion,
            meta_response="satisfaction",
            intensity=round(min(previous_state.intensity * 0.4, 0.6), 4),
            reason=f"Successfully regulated {previous_state.primary_emotion.value}",
        )

    # 2. Conflict: emoción viola valores del sistema
    for value in value_system.core_values:
        if value.name in _VALUE_EMOTION_CONFLICTS:
            conflicting_emotions = _VALUE_EMOTION_CONFLICTS[value.name]
            if emotion in conflicting_emotions:
                conflict_intensity = value.weight * intensity * 0.5
                if conflict_intensity > 0.2:
                    return MetaEmotion(
                        target_emotion=emotion,
                        meta_response="conflict",
                        intensity=round(conflict_intensity, 4),
                        reason=f"{emotion.value} conflicts with value '{value.name}'",
                    )

    # 3. Curiosity: emoción nueva o inusual
    if is_new_emotion and emotion not in (PrimaryEmotion.NEUTRAL, PrimaryEmotion.INDIFFERENCE):
        # Emociones mixtas o poco frecuentes generan curiosidad
        if emotion == PrimaryEmotion.MIXED or intensity > 0.6:
            return MetaEmotion(
                target_emotion=emotion,
                meta_response="curiosity",
                intensity=round(min(intensity * 0.3, 0.5), 4),
                reason=f"Novel experience of {emotion.value}",
            )

    # 4. Discomfort: emoción muy intensa que no se pudo regular
    if intensity > 0.8 and not regulation_success:
        if emotion in (
            PrimaryEmotion.ANGER, PrimaryEmotion.FEAR,
            PrimaryEmotion.ANXIETY, PrimaryEmotion.HELPLESSNESS,
        ):
            return MetaEmotion(
                target_emotion=emotion,
                meta_response="discomfort",
                intensity=round(intensity * 0.3, 4),
                reason=f"Intense {emotion.value} beyond comfortable threshold",
            )

    # 5. Acceptance: estado estable y prolongado
    if current_state.duration > 2 and intensity > 0.3:
        return MetaEmotion(
            target_emotion=emotion,
            meta_response="acceptance",
            intensity=round(min(intensity * 0.2, 0.4), 4),
            reason=f"Sustained {emotion.value} for {current_state.duration} turns",
        )

    return None
