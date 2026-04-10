"""Mood System - Estado emocional prolongado que evoluciona gradualmente.

El mood es diferente de la emocion:
- Emocion: reaccion breve a un estimulo (turnos)
- Mood: estado prolongado basado en patrones acumulados (muchos turnos)

El mood influye en la generacion de emociones (mood congruent bias):
un mood negativo hace mas probable que estimulos ambiguos generen emociones negativas.
"""

from pathos.models.emotion import (
    EmotionalSnapshot,
    EmotionalState,
    Mood,
    MoodLabel,
)

# Cuantos snapshots mantener en el historial
HISTORY_SIZE = 10

# Cuanto peso tiene el promedio emocional reciente sobre el baseline
MOOD_DRIFT_RATE = 0.03

# Mood congruent bias: cuanto sesga el mood la generacion de emociones nuevas
MOOD_CONGRUENCE_STRENGTH = 0.15


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def classify_mood(baseline_valence: float, baseline_arousal: float) -> MoodLabel:
    """Clasifica el mood basado en baseline valence y arousal."""
    if abs(baseline_valence) < 0.15 and abs(baseline_arousal - 0.3) < 0.15:
        return MoodLabel.NEUTRAL
    if baseline_valence > 0.1:
        return MoodLabel.BUOYANT if baseline_arousal > 0.4 else MoodLabel.SERENE
    if baseline_valence < -0.1:
        return MoodLabel.AGITATED if baseline_arousal > 0.4 else MoodLabel.MELANCHOLIC
    return MoodLabel.NEUTRAL


def compute_mood_congruence_bias(mood: Mood) -> tuple[float, float]:
    """Calcula el sesgo que el mood ejerce sobre emociones nuevas.

    Returns:
        (valence_bias, arousal_bias) — valores a sumar al estado emocional generado.
    """
    # El bias es proporcional a la distancia del mood del centro neutral
    valence_bias = (mood.baseline_valence - 0.0) * MOOD_CONGRUENCE_STRENGTH
    arousal_bias = (mood.baseline_arousal - 0.3) * MOOD_CONGRUENCE_STRENGTH
    return (valence_bias, arousal_bias)


def update_mood(mood: Mood, current_state: EmotionalState) -> Mood:
    """Actualiza el mood basado en el estado emocional actual.

    El mood evoluciona gradualmente: el promedio ponderado de las emociones
    recientes tira del baseline lentamente.

    Args:
        mood: Mood actual.
        current_state: Estado emocional del turno actual.

    Returns:
        Mood actualizado.
    """
    # 1. Agregar snapshot al historial
    snapshot = EmotionalSnapshot(
        valence=current_state.valence,
        arousal=current_state.arousal,
        intensity=current_state.intensity,
    )
    history = list(mood.emotional_history)
    history.append(snapshot)
    if len(history) > HISTORY_SIZE:
        history = history[-HISTORY_SIZE:]

    # 2. Calcular promedio ponderado (recientes pesan mas)
    if not history:
        return mood

    total_weight = 0.0
    weighted_valence = 0.0
    weighted_arousal = 0.0
    for i, snap in enumerate(history):
        # Peso exponencial: mas reciente = mas peso
        weight = (i + 1) * snap.intensity  # Emociones intensas pesan mas
        weight = max(weight, 0.1)  # Peso minimo para que todo cuente algo
        total_weight += weight
        weighted_valence += snap.valence * weight
        weighted_arousal += snap.arousal * weight

    avg_valence = weighted_valence / total_weight if total_weight > 0 else 0.0
    avg_arousal = weighted_arousal / total_weight if total_weight > 0 else 0.3

    # 3. Mover baseline gradualmente hacia el promedio emocional
    # Stability resiste el drift, intensidad promedio lo amplifica
    avg_intensity = sum(s.intensity for s in history) / len(history)
    effective_drift = MOOD_DRIFT_RATE * (1 - mood.stability * 0.5) * (0.5 + avg_intensity)

    new_bv = mood.baseline_valence + (avg_valence - mood.baseline_valence) * effective_drift
    new_ba = mood.baseline_arousal + (avg_arousal - mood.baseline_arousal) * effective_drift

    new_bv = _clamp(round(new_bv, 4), -1, 1)
    new_ba = _clamp(round(new_ba, 4), 0, 1)

    # 4. Clasificar el mood
    label = classify_mood(new_bv, new_ba)

    # 5. Determinar trend comparando con historial
    if len(history) >= 3:
        recent_avg = sum(s.valence for s in history[-3:]) / 3
        older_avg = sum(s.valence for s in history[:-3]) / max(len(history) - 3, 1)
        if recent_avg > older_avg + 0.1:
            trend = "improving"
        elif recent_avg < older_avg - 0.1:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = mood.trend

    return Mood(
        baseline_valence=new_bv,
        baseline_arousal=new_ba,
        stability=mood.stability,
        trend=trend,
        label=label,
        extreme_event_count=mood.extreme_event_count,
        turns_since_extreme=mood.turns_since_extreme,
        original_baseline_valence=mood.original_baseline_valence,
        original_baseline_arousal=mood.original_baseline_arousal,
        emotional_history=history,
    )
