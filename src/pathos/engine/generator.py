"""Emotion Generator - Convierte AppraisalVector en EmotionalState.

Fase 4 (Advanced+):
- Emotional Stack: activación simultánea de todas las emociones
- Co-occurrence rules: inhibición mutua entre emociones incompatibles
- Emergent emotions: combinaciones generan emociones nuevas
- Emotion Dynamics: transiciones no-lineales (ODE)
- Mood congruence bias
- Needs amplification
"""

import math
from datetime import datetime, timezone

from pathos.engine.dynamics import EmotionDynamics
from pathos.engine.mood import compute_mood_congruence_bias, update_mood
from pathos.engine.predictive import EmotionModulation
from pathos.models.coupling import CouplingMatrix
from pathos.models.appraisal import AppraisalVector
from pathos.models.emotion import BodyState, EmotionalState, Mood, PrimaryEmotion


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _lerp(a: float, b: float, t: float) -> float:
    """Interpolacion lineal entre a y b con factor t."""
    return a + (b - a) * t


def compute_valence(appraisal: AppraisalVector) -> float:
    """Calcula valence desde el appraisal (spec formula)."""
    v = appraisal.valence
    raw = (
        v.goal_conduciveness * 0.4
        + v.value_alignment * 0.4
        + v.intrinsic_pleasantness * 0.2
    )
    return _clamp(raw, -1, 1)


def compute_arousal(appraisal: AppraisalVector) -> float:
    """Calcula arousal desde el appraisal."""
    r = appraisal.relevance
    raw = abs(r.novelty) * 0.4 + r.personal_significance * 0.6
    return _clamp(raw, 0, 1)


def compute_dominance(appraisal: AppraisalVector) -> float:
    """Calcula dominance desde el appraisal (5 dimensiones de Scherer).

    Integra coping (control, power, adjustability) y agency (fairness directional).
    Fairness alta → empowerment (dominance sube).
    Fairness baja → victimización (dominance baja).
    """
    c = appraisal.coping
    # Fairness directional: -1 (injusto) → 0, +1 (justo) → 1
    fairness_01 = (appraisal.agency.fairness + 1) / 2
    raw = (
        c.control * 0.35
        + c.power * 0.25
        + fairness_01 * 0.25
        + c.adjustability * 0.15
    )
    return _clamp(raw, 0, 1)


def compute_certainty(appraisal: AppraisalVector) -> float:
    """Calcula certainty desde el appraisal (5 dimensiones de Scherer).

    Integra coping.adjustability y norms.self_consistency.
    Self-consistency alta → más certeza (coherente con quien soy).
    Self-consistency baja → incertidumbre (contradicción identitaria).
    """
    self_con_01 = (appraisal.norms.self_consistency + 1) / 2  # -1..1 → 0..1
    raw = appraisal.coping.adjustability * 0.6 + self_con_01 * 0.4
    return _clamp(raw, 0, 1)


# Prototipos emocionales: (valence, arousal, dominance, certainty)
_EMOTION_PROTOTYPES: dict[PrimaryEmotion, tuple[float, float, float, float]] = {
    # Positivas alta energia
    PrimaryEmotion.JOY:            ( 0.75,  0.65,  0.70,  0.70),
    PrimaryEmotion.EXCITEMENT:     ( 0.70,  0.90,  0.55,  0.25),
    PrimaryEmotion.GRATITUDE:      ( 0.70,  0.40,  0.30,  0.70),
    PrimaryEmotion.HOPE:           ( 0.50,  0.55,  0.35,  0.25),
    # Positivas baja energia
    PrimaryEmotion.CONTENTMENT:    ( 0.55,  0.20,  0.60,  0.80),
    PrimaryEmotion.RELIEF:         ( 0.50,  0.25,  0.45,  0.65),
    # Negativas alta energia
    PrimaryEmotion.ANGER:          (-0.75,  0.80,  0.70,  0.60),
    PrimaryEmotion.FRUSTRATION:    (-0.50,  0.70,  0.40,  0.35),
    PrimaryEmotion.FEAR:           (-0.75,  0.85,  0.15,  0.15),
    PrimaryEmotion.ANXIETY:        (-0.45,  0.65,  0.25,  0.25),
    # Negativas baja energia
    PrimaryEmotion.SADNESS:        (-0.70,  0.20,  0.25,  0.60),
    PrimaryEmotion.HELPLESSNESS:   (-0.75,  0.20,  0.10,  0.15),
    PrimaryEmotion.DISAPPOINTMENT: (-0.50,  0.30,  0.40,  0.55),
    # Neutras / ambiguas
    PrimaryEmotion.SURPRISE:       ( 0.05,  0.85,  0.40,  0.15),
    PrimaryEmotion.ALERTNESS:      (-0.05,  0.70,  0.50,  0.35),
    PrimaryEmotion.CONTEMPLATION:  ( 0.15,  0.25,  0.55,  0.55),
    PrimaryEmotion.INDIFFERENCE:   ( 0.00,  0.10,  0.50,  0.50),
    PrimaryEmotion.NEUTRAL:        ( 0.00,  0.30,  0.50,  0.50),
}

# Pesos para la distancia: valence pesa mas porque es la dimension mas confiable del LLM
_DIM_WEIGHTS = (2.5, 1.0, 0.6, 0.6)


# --- Co-occurrence Rules ---
# Pares de emociones que se inhiben mutuamente (no pueden coexistir fácilmente)
_INHIBITION_PAIRS: set[frozenset[PrimaryEmotion]] = {
    frozenset({PrimaryEmotion.JOY, PrimaryEmotion.SADNESS}),
    frozenset({PrimaryEmotion.CONTENTMENT, PrimaryEmotion.ANXIETY}),
    frozenset({PrimaryEmotion.EXCITEMENT, PrimaryEmotion.HELPLESSNESS}),
    frozenset({PrimaryEmotion.HOPE, PrimaryEmotion.HELPLESSNESS}),
    frozenset({PrimaryEmotion.RELIEF, PrimaryEmotion.FEAR}),
    frozenset({PrimaryEmotion.INDIFFERENCE, PrimaryEmotion.EXCITEMENT}),
}

# Pares de emociones que pueden coexistir (generan emociones emergentes)
_EMERGENT_COMBINATIONS: dict[frozenset[PrimaryEmotion], str] = {
    frozenset({PrimaryEmotion.JOY, PrimaryEmotion.SADNESS}): "bittersweet",
    frozenset({PrimaryEmotion.ANGER, PrimaryEmotion.SADNESS}): "resentment",
    frozenset({PrimaryEmotion.FEAR, PrimaryEmotion.ANGER}): "defiance",
    frozenset({PrimaryEmotion.GRATITUDE, PrimaryEmotion.SADNESS}): "nostalgia",
    frozenset({PrimaryEmotion.HOPE, PrimaryEmotion.FEAR}): "anxious_hope",
    frozenset({PrimaryEmotion.JOY, PrimaryEmotion.GRATITUDE}): "elation",
    frozenset({PrimaryEmotion.ANGER, PrimaryEmotion.FRUSTRATION}): "outrage",
    frozenset({PrimaryEmotion.SADNESS, PrimaryEmotion.FEAR}): "dread",
}


def _weighted_distance(
    v: float, a: float, d: float, c: float,
    proto: tuple[float, float, float, float],
) -> float:
    """Distancia euclidiana ponderada a un prototipo emocional."""
    return sum(
        w * (actual - target) ** 2
        for w, actual, target in zip(_DIM_WEIGHTS, (v, a, d, c), proto)
    )


def compute_emotional_stack(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
) -> dict[str, float]:
    """Calcula el vector de activación para todas las emociones.

    Usa distancia inversa con softmax para producir un vector donde
    todas las emociones tienen una activación, sumando ~1.0.

    Returns:
        Dict {emotion_name: activation_level}
    """
    # Calcular distancias a todos los prototipos
    distances: dict[str, float] = {}
    for emotion, proto in _EMOTION_PROTOTYPES.items():
        if emotion == PrimaryEmotion.MIXED:
            continue
        dist = _weighted_distance(valence, arousal, dominance, certainty, proto)
        distances[emotion.value] = dist

    # Convertir distancias a activaciones (inverse distance softmax)
    # Temperatura controla la "sharpness" de la distribución
    temperature = 0.5
    activations: dict[str, float] = {}
    max_activation = 0.0

    for emotion_name, dist in distances.items():
        # Inverse distance: menor distancia = mayor activación
        activation = math.exp(-dist / temperature)
        activations[emotion_name] = activation
        max_activation = max(max_activation, activation)

    # Normalizar para que sume ~1.0
    total = sum(activations.values())
    if total > 0:
        activations = {k: round(v / total, 4) for k, v in activations.items()}

    # Aplicar co-occurrence inhibition
    activations = _apply_inhibition(activations)

    return activations


def _apply_inhibition(stack: dict[str, float]) -> dict[str, float]:
    """Aplica reglas de inhibición mutua al emotional stack.

    Emociones incompatibles se inhiben entre sí: la más fuerte
    suprime parcialmente a la más débil.
    """
    result = dict(stack)

    for pair in _INHIBITION_PAIRS:
        names = [e.value for e in pair]
        if all(n in result for n in names):
            a_name, b_name = names[0], names[1]
            a_val = result[a_name]
            b_val = result[b_name]

            # La más fuerte suprime a la más débil
            if a_val > b_val:
                result[b_name] = round(b_val * 0.3, 4)  # 70% supresión
            else:
                result[a_name] = round(a_val * 0.3, 4)

    # Re-normalizar
    total = sum(result.values())
    if total > 0:
        result = {k: round(v / total, 4) for k, v in result.items()}

    return result


def detect_emergent_emotions(stack: dict[str, float]) -> list[str]:
    """Detecta emociones emergentes de combinaciones en el stack.

    Cuando dos emociones coexisten con activación suficiente (>0.15 cada una),
    pueden generar una emoción emergente.

    Returns:
        Lista de nombres de emociones emergentes detectadas.
    """
    emergent: list[str] = []
    threshold = 0.15

    for pair, emergent_name in _EMERGENT_COMBINATIONS.items():
        names = [e.value for e in pair]
        if all(n in stack and stack[n] > threshold for n in names):
            emergent.append(emergent_name)

    return emergent


def identify_primary_emotion(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    appraisal: AppraisalVector,
) -> PrimaryEmotion:
    """Mapea dimensiones a emocion primaria via distancia a prototipos."""
    best_emotion = PrimaryEmotion.NEUTRAL
    best_distance = float("inf")

    for emotion, proto in _EMOTION_PROTOTYPES.items():
        if emotion == PrimaryEmotion.MIXED:
            continue
        dist = _weighted_distance(valence, arousal, dominance, certainty, proto)
        if dist < best_distance:
            best_distance = dist
            best_emotion = emotion

    if best_emotion == PrimaryEmotion.FRUSTRATION and appraisal.agency.fairness < -0.3:
        best_emotion = PrimaryEmotion.ANGER

    if best_distance > 1.5:
        best_emotion = PrimaryEmotion.MIXED

    return best_emotion


def identify_secondary_emotion(
    primary: PrimaryEmotion,
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    appraisal: AppraisalVector,
) -> PrimaryEmotion | None:
    """Identifica una emocion secundaria: la segunda mas cercana por prototipo."""
    distances: list[tuple[float, PrimaryEmotion]] = []
    for emotion, proto in _EMOTION_PROTOTYPES.items():
        if emotion in (primary, PrimaryEmotion.MIXED, PrimaryEmotion.NEUTRAL):
            continue
        dist = _weighted_distance(valence, arousal, dominance, certainty, proto)
        distances.append((dist, emotion))

    if not distances:
        return None

    distances.sort()
    second_dist, second_emotion = distances[0]
    return second_emotion if second_dist < 1.5 else None


def compute_intensity(appraisal: AppraisalVector, valence: float, arousal: float) -> float:
    """Calcula la intensidad emocional (con norm_compatibility)."""
    raw = (
        appraisal.relevance.personal_significance
        * max(abs(valence), arousal)
        * (1 + abs(appraisal.norms.internal_standards - 0.5))
    )
    return _clamp(raw, 0, 1)


def compute_body_state(
    valence: float,
    arousal: float,
    dominance: float,
    intensity: float,
    current_body: BodyState,
) -> BodyState:
    """Computa el body state basado en las dimensiones emocionales."""
    energy = _clamp(
        _lerp(current_body.energy, 0.3 + arousal * 0.4 + max(valence, 0) * 0.3, 0.6),
        0, 1,
    )
    tension = _clamp(
        _lerp(current_body.tension, arousal * 0.5 + max(-valence, 0) * 0.5, 0.6),
        0, 1,
    )
    openness = _clamp(
        _lerp(
            current_body.openness,
            0.3 + max(valence, 0) * 0.4 + dominance * 0.3,
            0.6,
        ),
        0, 1,
    )
    warmth = _clamp(
        _lerp(
            current_body.warmth,
            0.3 + max(valence, 0) * 0.5 + (1 - tension) * 0.2,
            0.6,
        ),
        0, 1,
    )

    return BodyState(energy=energy, tension=tension, openness=openness, warmth=warmth)


# Default dynamics instance (can be overridden per session via personality)
_default_dynamics = EmotionDynamics(
    attractor_strength=0.15,
    variability=0.3,
    base_inertia=0.5,
)


def generate_emotion(
    appraisal: AppraisalVector,
    current_state: EmotionalState,
    stimulus: str,
    blend_factor: float = 0.6,
    amplification: float = 0.0,
    emotion_hint: PrimaryEmotion | None = None,
    dynamics: EmotionDynamics | None = None,
    needs_amplification: float = 0.0,
    social_valence_mod: float = 0.0,
    social_intensity_mod: float = 0.0,
    contagion_valence: float = 0.0,
    contagion_arousal: float = 0.0,
    coupling: CouplingMatrix | None = None,
    predictive_modulation: EmotionModulation | None = None,
) -> EmotionalState:
    """Genera un nuevo estado emocional a partir del appraisal y el estado actual.

    Args:
        appraisal: Vector de evaluacion del estimulo.
        current_state: Estado emocional actual.
        stimulus: Texto que disparo la evaluacion.
        blend_factor: 0-1, cuanto pesa el nuevo estado vs inercia (0.6 = 60% nuevo).
        amplification: 0-0.5, amplificacion por memoria emocional.
        emotion_hint: Emocion clasificada directamente por el LLM.
        dynamics: EmotionDynamics instance (None = use default lerp for calibration).
        needs_amplification: 0-0.4, amplificación por necesidades computacionales.
        social_valence_mod: Modulación de valence por social cognition.
        social_intensity_mod: Modulación de intensity por social cognition.
        contagion_valence: Perturbación de contagio emocional en valence.
        contagion_arousal: Perturbación de contagio emocional en arousal.
        coupling: CouplingMatrix para acoplamiento dimensional.
        predictive_modulation: Modulación del Predictive Processing (Pilar 1 ANIMA).

    Returns:
        Nuevo EmotionalState con emotional_stack.
    """
    # Calcular dimensiones desde appraisal
    new_valence = compute_valence(appraisal)
    new_arousal = compute_arousal(appraisal)

    if emotion_hint and emotion_hint in _EMOTION_PROTOTYPES:
        proto = _EMOTION_PROTOTYPES[emotion_hint]
        new_dominance = proto[2]
        new_certainty = proto[3]
    else:
        new_dominance = compute_dominance(appraisal)
        new_certainty = compute_certainty(appraisal)

    # Mood congruence bias
    valence_bias, arousal_bias = compute_mood_congruence_bias(current_state.mood)
    new_valence = _clamp(new_valence + valence_bias, -1, 1)
    new_arousal = _clamp(new_arousal + arousal_bias, 0, 1)

    # Social cognition modulation
    new_valence = _clamp(new_valence + social_valence_mod, -1, 1)

    # Predictive Processing modulation (Pilar 1 ANIMA)
    # El prediction error modifica las dimensiones ANTES de la dinámica ODE,
    # para que el sistema integre la contribución predictiva orgánicamente.
    if predictive_modulation is not None:
        new_valence = _clamp(new_valence + predictive_modulation.valence_delta, -1, 1)
        new_arousal = _clamp(new_arousal + predictive_modulation.arousal_delta, 0, 1)
        new_certainty = _clamp(new_certainty + predictive_modulation.certainty_delta, 0, 1)

    # Apply dynamics or lerp
    if dynamics is not None and blend_factor < 1.0:
        # Use ODE dynamics for non-calibration mode
        valence, arousal, dominance, certainty = dynamics.step_4d(
            current_state.valence, current_state.arousal,
            current_state.dominance, current_state.certainty,
            new_valence, new_arousal, new_dominance, new_certainty,
            current_state.mood.baseline_valence,
            current_state.mood.baseline_arousal,
            current_state.primary_emotion,
            contagion_v=contagion_valence,
            contagion_a=contagion_arousal,
            coupling=coupling,
        )
    else:
        # Classic lerp (used in calibration mode with blend_factor=1.0)
        valence = _clamp(_lerp(current_state.valence, new_valence, blend_factor), -1, 1)
        arousal = _clamp(_lerp(current_state.arousal, new_arousal, blend_factor), 0, 1)
        dominance = _clamp(_lerp(current_state.dominance, new_dominance, blend_factor), 0, 1)
        certainty = _clamp(_lerp(current_state.certainty, new_certainty, blend_factor), 0, 1)

    # Identify emotions
    if emotion_hint:
        primary = emotion_hint
    else:
        primary = identify_primary_emotion(valence, arousal, dominance, certainty, appraisal)
    secondary = identify_secondary_emotion(primary, valence, arousal, dominance, certainty, appraisal)

    # Compute emotional stack
    emotional_stack = compute_emotional_stack(valence, arousal, dominance, certainty)

    # Detect emergent emotions
    emergent = detect_emergent_emotions(emotional_stack)

    # Intensity (with memory + needs amplification + social modulation + predictive)
    intensity = compute_intensity(appraisal, valence, arousal)
    total_amplification = amplification + needs_amplification
    intensity = _clamp(intensity * (1 + total_amplification) + social_intensity_mod, 0, 1)
    if predictive_modulation is not None:
        intensity = _clamp(intensity + predictive_modulation.intensity_delta, 0, 1)

    # Body state
    body = compute_body_state(valence, arousal, dominance, intensity, current_state.body_state)

    # Duration
    duration = current_state.duration + 1 if current_state.primary_emotion == primary else 0

    # Build state
    new_state = EmotionalState(
        valence=round(valence, 4),
        arousal=round(arousal, 4),
        dominance=round(dominance, 4),
        certainty=round(certainty, 4),
        primary_emotion=primary,
        secondary_emotion=secondary,
        intensity=round(intensity, 4),
        emotional_stack=emotional_stack,
        body_state=body,
        mood=current_state.mood,
        duration=duration,
        triggered_by=stimulus[:100],
        timestamp=datetime.now(timezone.utc),
    )
    new_state.mood = update_mood(current_state.mood, new_state)

    return new_state
