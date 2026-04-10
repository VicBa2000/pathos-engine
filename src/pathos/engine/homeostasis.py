"""Homeostasis Emocional - Regulacion del estado entre turnos.

Fase 3 Advanced:
- Baseline shift con sensitizacion (mas eventos extremos = shift mas rapido)
- Stability del mood resiste cambios
- Recovery gradual del baseline cuando no hay eventos extremos
- Limites de shift para evitar extremos permanentes
"""

from datetime import datetime, timezone

from pathos.models.emotion import BodyState, EmotionalState, Mood, PrimaryEmotion


# --- Constantes de decay ---
DECAY_RATE = 0.15              # Por turno, la emocion decae 15% hacia baseline

# --- Constantes de baseline shift ---
BASELINE_SHIFT_THRESHOLD = 0.8  # Intensidad minima para mover baseline
BASE_SHIFT_RATE = 0.02          # Shift base por evento extremo
MAX_SHIFT_RATE = 0.08           # Shift maximo con sensitizacion total
SENSITIZATION_FACTOR = 0.01     # Cuanto aumenta el rate por cada evento extremo previo
MAX_BASELINE_DRIFT = 0.5        # Maximo que el baseline puede alejarse del original

# --- Constantes de recovery ---
RECOVERY_START_TURNS = 5        # Turnos sin evento extremo antes de iniciar recovery
RECOVERY_RATE = 0.005           # Rate de recovery hacia baseline original por turno


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _compute_shift_rate(extreme_count: int, stability: float) -> float:
    """Calcula el rate de shift basado en sensitizacion y estabilidad.

    Mas eventos extremos = shift mas rapido (sensitizacion).
    Mayor stability = mas resistencia al cambio.
    """
    # Sensitizacion: cada evento extremo previo aumenta el rate
    sensitized_rate = BASE_SHIFT_RATE + (extreme_count * SENSITIZATION_FACTOR)
    sensitized_rate = min(sensitized_rate, MAX_SHIFT_RATE)

    # Stability resiste el cambio: stability 1.0 reduce shift a 30%, stability 0.0 no resiste
    resistance = 0.3 + (0.7 * (1 - stability))
    return sensitized_rate * resistance


def _apply_baseline_recovery(mood: Mood) -> Mood:
    """Recupera gradualmente el baseline hacia el original si no hay eventos extremos recientes."""
    if mood.turns_since_extreme < RECOVERY_START_TURNS:
        return mood

    # Recovery proporcional a la distancia del original
    new_bv = _lerp(mood.baseline_valence, mood.original_baseline_valence, RECOVERY_RATE)
    new_ba = _lerp(mood.baseline_arousal, mood.original_baseline_arousal, RECOVERY_RATE)

    # Stability se recupera tambien
    new_stability = _clamp(mood.stability + 0.005, 0, 1)

    return Mood(
        baseline_valence=_clamp(round(new_bv, 4), -1, 1),
        baseline_arousal=_clamp(round(new_ba, 4), 0, 1),
        stability=round(new_stability, 4),
        trend=mood.trend,
        extreme_event_count=mood.extreme_event_count,
        turns_since_extreme=mood.turns_since_extreme,
        original_baseline_valence=mood.original_baseline_valence,
        original_baseline_arousal=mood.original_baseline_arousal,
    )


def regulate(state: EmotionalState, turns_elapsed: int = 1) -> EmotionalState:
    """Aplica regulacion homeostatica entre turnos.

    Args:
        state: Estado emocional actual.
        turns_elapsed: Turnos transcurridos desde la ultima regulacion.

    Returns:
        Estado emocional regulado.
    """
    # 1. Decaimiento natural hacia baseline
    decay = 1 - (DECAY_RATE * turns_elapsed)
    decay = max(decay, 0.1)  # Nunca decae completamente en un turno

    new_valence = _lerp(state.valence, state.mood.baseline_valence, 1 - decay)
    new_arousal = _lerp(state.arousal, state.mood.baseline_arousal, 1 - decay)
    new_intensity = state.intensity * decay

    # 2. Baseline shift avanzado
    new_mood = state.mood.model_copy()
    new_turns_since = new_mood.turns_since_extreme + 1

    if state.intensity > BASELINE_SHIFT_THRESHOLD:
        # Evento extremo detectado
        shift_rate = _compute_shift_rate(new_mood.extreme_event_count, new_mood.stability)

        new_baseline_v = new_mood.baseline_valence + (
            (state.valence - new_mood.baseline_valence) * shift_rate
        )
        new_baseline_a = new_mood.baseline_arousal + (
            (state.arousal - new_mood.baseline_arousal) * shift_rate
        )

        # Limitar drift maximo desde el original
        max_v = new_mood.original_baseline_valence + MAX_BASELINE_DRIFT
        min_v = new_mood.original_baseline_valence - MAX_BASELINE_DRIFT
        max_a = min(new_mood.original_baseline_arousal + MAX_BASELINE_DRIFT, 1.0)
        min_a = max(new_mood.original_baseline_arousal - MAX_BASELINE_DRIFT, 0.0)

        new_mood = Mood(
            baseline_valence=_clamp(round(new_baseline_v, 4), max(min_v, -1), min(max_v, 1)),
            baseline_arousal=_clamp(round(new_baseline_a, 4), min_a, max_a),
            stability=_clamp(round(new_mood.stability - 0.02, 4), 0, 1),
            trend=new_mood.trend,
            extreme_event_count=new_mood.extreme_event_count + 1,
            turns_since_extreme=0,
            original_baseline_valence=new_mood.original_baseline_valence,
            original_baseline_arousal=new_mood.original_baseline_arousal,
        )
    else:
        new_mood = Mood(
            baseline_valence=new_mood.baseline_valence,
            baseline_arousal=new_mood.baseline_arousal,
            stability=new_mood.stability,
            trend=new_mood.trend,
            extreme_event_count=new_mood.extreme_event_count,
            turns_since_extreme=new_turns_since,
            original_baseline_valence=new_mood.original_baseline_valence,
            original_baseline_arousal=new_mood.original_baseline_arousal,
        )

        # 3. Recovery gradual
        new_mood = _apply_baseline_recovery(new_mood)

    # 4. Actualizar trend del mood
    if new_valence > new_mood.baseline_valence + 0.1:
        trend = "improving"
    elif new_valence < new_mood.baseline_valence - 0.1:
        trend = "declining"
    else:
        trend = "stable"
    new_mood = Mood(
        baseline_valence=new_mood.baseline_valence,
        baseline_arousal=new_mood.baseline_arousal,
        stability=new_mood.stability,
        trend=trend,
        extreme_event_count=new_mood.extreme_event_count,
        turns_since_extreme=new_mood.turns_since_extreme,
        original_baseline_valence=new_mood.original_baseline_valence,
        original_baseline_arousal=new_mood.original_baseline_arousal,
    )

    # 5. Body state regulation
    new_energy = _clamp(state.body_state.energy - 0.02 * state.arousal, 0, 1)
    new_tension = _clamp(state.body_state.tension * 0.9, 0, 1)
    new_openness = _lerp(state.body_state.openness, 0.5, 0.1)
    new_warmth = _lerp(state.body_state.warmth, 0.5, 0.1)

    body = BodyState(
        energy=new_energy,
        tension=new_tension,
        openness=_clamp(new_openness, 0, 1),
        warmth=_clamp(new_warmth, 0, 1),
    )

    # 6. Reclasificar emocion si la intensidad decayo mucho
    primary = state.primary_emotion
    secondary = state.secondary_emotion
    if new_intensity < 0.1:
        primary = PrimaryEmotion.NEUTRAL
        secondary = None

    return EmotionalState(
        valence=round(_clamp(new_valence, -1, 1), 4),
        arousal=round(_clamp(new_arousal, 0, 1), 4),
        dominance=round(state.dominance, 4),
        certainty=round(state.certainty, 4),
        primary_emotion=primary,
        secondary_emotion=secondary,
        intensity=round(_clamp(new_intensity, 0, 1), 4),
        body_state=body,
        mood=new_mood,
        duration=state.duration,
        triggered_by=state.triggered_by,
        timestamp=datetime.now(timezone.utc),
    )
