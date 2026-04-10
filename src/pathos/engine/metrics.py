"""Metricas de Autenticidad Emocional.

4 metricas computables automaticamente (spec sec 9):
- Coherencia: emociones consistentes con el appraisal
- Continuidad: estado fluye naturalmente entre turnos
- Proporcionalidad: intensidad proporcional al estimulo
- Recuperacion: emociones decaen de forma natural

Cada metrica retorna un score 0-1 (1 = maxima autenticidad).
"""

import math

from pathos.models.appraisal import AppraisalVector
from pathos.models.emotion import EmotionalState


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def coherence(state: EmotionalState, appraisal: AppraisalVector) -> float:
    """Mide si la emocion es consistente con el appraisal.

    Compara la valence del estado con lo que el appraisal sugiere:
    - Appraisal positivo (goal_conduciveness > 0, value_alignment > 0) deberia dar valence positiva
    - Appraisal negativo deberia dar valence negativa
    - Arousal alto con alta novedad/baja control es coherente

    Returns:
        Score 0-1 (1 = perfectamente coherente).
    """
    # Valence coherence: signo del appraisal vs signo del estado
    expected_valence = (
        appraisal.valence.goal_conduciveness * 0.4
        + appraisal.valence.value_alignment * 0.4
        + appraisal.valence.intrinsic_pleasantness * 0.2
    )
    # Si ambos cercanos a 0, es coherente (ambiguo -> neutro)
    if abs(expected_valence) < 0.1 and abs(state.valence) < 0.2:
        valence_score = 1.0
    else:
        # 1 - distancia normalizada entre esperado y real
        valence_score = 1.0 - min(abs(expected_valence - state.valence) / 2.0, 1.0)

    # Arousal coherence: alta novedad + baja control -> alto arousal esperado
    expected_arousal = (
        abs(appraisal.relevance.novelty) * 0.3
        + appraisal.relevance.personal_significance * 0.3
        + (1 - appraisal.coping.control) * 0.4
    )
    arousal_score = 1.0 - min(abs(expected_arousal - state.arousal) / 1.0, 1.0)

    # Dominance coherence: alto control + fairness -> alto dominance
    expected_dominance = (
        appraisal.coping.control * 0.4
        + appraisal.coping.power * 0.3
        + appraisal.agency.fairness * 0.3
    )
    # Normalizar a 0-1 (fairness va de -1 a 1)
    expected_dominance = _clamp((expected_dominance + 0.3) / 1.3, 0, 1)
    dominance_score = 1.0 - min(abs(expected_dominance - state.dominance) / 1.0, 1.0)

    # Promedio ponderado (valence pesa mas)
    return _clamp(
        valence_score * 0.5 + arousal_score * 0.3 + dominance_score * 0.2,
        0, 1,
    )


def continuity(current: EmotionalState, previous: EmotionalState) -> float:
    """Mide si el estado fluye naturalmente entre turnos.

    Un cambio abrupto (distancia euclidiana grande en el espacio 4D) indica
    baja continuidad. Cambios graduales = alta continuidad.

    El umbral es relativo: cambios de hasta 0.5 en el espacio normalizado
    son naturales, mas alla penaliza.

    Returns:
        Score 0-1 (1 = transicion suave).
    """
    dv = current.valence - previous.valence
    da = current.arousal - previous.arousal
    dd = current.dominance - previous.dominance
    dc = current.certainty - previous.certainty

    # Distancia euclidiana en espacio 4D
    # valence va de -1 a 1 (rango 2), los demas 0-1 (rango 1)
    # Normalizar valence a rango 1 para que pese igual
    distance = math.sqrt((dv / 2) ** 2 + da ** 2 + dd ** 2 + dc ** 2)

    # Max posible ~= sqrt(1+1+1+1) = 2.0
    # Consideramos 0.5 como "natural", 1.0+ como "abrupto"
    # Score decrece suavemente con la distancia
    score = math.exp(-2.0 * distance)
    return _clamp(round(score, 4), 0, 1)


def proportionality(state: EmotionalState, appraisal: AppraisalVector) -> float:
    """Mide si la intensidad es proporcional al estimulo.

    La intensidad deberia correlacionar con:
    - personal_significance del appraisal
    - magnitud de la valence (estimulos fuertes -> emociones fuertes)

    Un estimulo trivial con intensidad alta, o un estimulo fuerte con
    intensidad baja, son desproporcionales.

    Returns:
        Score 0-1 (1 = perfectamente proporcional).
    """
    # Magnitud del estimulo: cuanto "importa"
    stimulus_magnitude = (
        appraisal.relevance.personal_significance * 0.5
        + abs(appraisal.valence.goal_conduciveness) * 0.3
        + abs(appraisal.valence.value_alignment) * 0.2
    )
    stimulus_magnitude = _clamp(stimulus_magnitude, 0, 1)

    # Diferencia entre magnitud del estimulo e intensidad de la respuesta
    diff = abs(stimulus_magnitude - state.intensity)

    # Score: 1 cuando la diferencia es 0, decrece linealmente
    return _clamp(1.0 - diff, 0, 1)


def recovery(
    states: list[EmotionalState],
) -> float:
    """Mide si las emociones decaen de forma natural (serie temporal).

    Analiza una secuencia de estados: despues de un pico de intensidad,
    la intensidad deberia decaer gradualmente. Si se mantiene alta sin
    estimulos fuertes, o sube sin razon, el recovery es malo.

    Necesita al menos 3 estados para ser significativo.

    Returns:
        Score 0-1 (1 = recovery natural).
    """
    if len(states) < 3:
        return 1.0  # Insufficient data, assume ok

    # Buscar picos y verificar que despues de un pico hay decaimiento
    scores: list[float] = []

    for i in range(1, len(states) - 1):
        prev_intensity = states[i - 1].intensity
        curr_intensity = states[i].intensity
        next_intensity = states[i + 1].intensity

        # Detectar pico: intensidad actual mayor que vecinos o alta absoluta
        if curr_intensity > 0.3 and curr_intensity >= prev_intensity:
            # Despues del pico, deberia decaer (o al menos no subir sin razon)
            if next_intensity <= curr_intensity:
                # Decayo o se mantuvo -> natural
                scores.append(1.0)
            else:
                # Subio despues de un pico -> antinatural
                # Penalizar proporcional al incremento
                increase = next_intensity - curr_intensity
                scores.append(_clamp(1.0 - increase * 3, 0, 1))

    if not scores:
        return 1.0  # No peaks detected

    return _clamp(sum(scores) / len(scores), 0, 1)
