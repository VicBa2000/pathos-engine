"""Calibration Engine - Computa perfiles de calibracion a partir de escenarios.

Flujo:
1. Usuario envia escenarios con respuestas emocionales esperadas
2. Sistema procesa cada escenario (appraisal -> emotion generation)
3. Compara output del sistema vs esperado del usuario
4. Genera CalibrationProfile con offsets/escalas para corregir

El perfil se aplica durante la generacion emocional normal.
"""

from pathos.models.calibration import CalibrationProfile, CalibrationResult
from pathos.models.emotion import EmotionalState


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def compute_calibration_profile(results: list[CalibrationResult]) -> CalibrationProfile:
    """Computa un perfil de calibracion a partir de los resultados de escenarios.

    Calcula offsets y escalas promediando los deltas entre
    respuesta esperada del usuario y respuesta del sistema.

    Args:
        results: Lista de CalibrationResult con deltas ya calculados.

    Returns:
        CalibrationProfile con ajustes.
    """
    if not results:
        return CalibrationProfile()

    n = len(results)

    # Promedio de deltas de valence -> offset directo
    avg_valence_delta = sum(r.valence_delta for r in results) / n

    # Para arousal e intensity, calcular ratio (escala)
    # Si el usuario esperaba mas arousal que el sistema, scale > 1
    arousal_ratios: list[float] = []
    intensity_ratios: list[float] = []

    for r in results:
        if r.system_arousal > 0.05:
            arousal_ratios.append(r.scenario.expected_arousal / r.system_arousal)
        if r.system_intensity > 0.05:
            intensity_ratios.append(r.scenario.expected_intensity / r.system_intensity)

    avg_arousal_scale = (
        sum(arousal_ratios) / len(arousal_ratios) if arousal_ratios else 1.0
    )
    avg_intensity_scale = (
        sum(intensity_ratios) / len(intensity_ratios) if intensity_ratios else 1.0
    )

    # Emotion accuracy
    matches = sum(1 for r in results if r.emotion_match)
    accuracy = matches / n

    return CalibrationProfile(
        valence_offset=_clamp(round(avg_valence_delta * 0.5, 4), -0.5, 0.5),
        arousal_scale=_clamp(round(avg_arousal_scale, 4), 0.5, 2.0),
        intensity_scale=_clamp(round(avg_intensity_scale, 4), 0.5, 2.0),
        scenarios_used=n,
        emotion_accuracy=round(accuracy, 4),
    )


def apply_calibration(
    state: EmotionalState,
    profile: CalibrationProfile,
) -> EmotionalState:
    """Aplica el perfil de calibracion a un estado emocional.

    Modifica valence, arousal e intensity segun los offsets/escalas del perfil.
    No modifica la emocion primaria (se reclasificara despues si cambian las dims).

    Args:
        state: Estado emocional a calibrar.
        profile: Perfil de calibracion.

    Returns:
        Estado emocional calibrado.
    """
    if profile.scenarios_used == 0:
        return state

    new_valence = _clamp(
        round(state.valence + profile.valence_offset, 4), -1, 1,
    )
    new_arousal = _clamp(
        round(state.arousal * profile.arousal_scale, 4), 0, 1,
    )
    new_intensity = _clamp(
        round(state.intensity * profile.intensity_scale, 4), 0, 1,
    )

    return state.model_copy(update={
        "valence": new_valence,
        "arousal": new_arousal,
        "intensity": new_intensity,
    })
