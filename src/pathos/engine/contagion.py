"""Emotion Contagion Engine - Contagio emocional pre-cognitivo.

Detecta la emocion del usuario via patrones linguisticos y la inyecta
como perturbacion en las dinamicas emocionales del agente.

Diferente de social cognition:
- Social cognition es COGNITIVO (pasa por appraisal, modula respuesta)
- Contagion es PRE-COGNITIVO (afecta directamente el estado emocional)

El contagio funciona en 3 pasos:
1. Detectar la emocion del usuario (sentiment + patrones)
2. Actualizar el shadow state (estado espejo)
3. Computar la perturbacion de contagio para el ODE

Basado en:
- Hatfield et al. (1993) "Emotional Contagion"
- Preston & de Waal (2002) Perception-Action Model
"""

import re

from pathos.models.contagion import ShadowState
from pathos.models.personality import PersonalityProfile


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# --- Lexicon de deteccion emocional ---
# Palabras/patrones con valencia y arousal asociados
# (valence, arousal, weight)
_POSITIVE_HIGH: list[tuple[str, float, float]] = [
    # Alegria/entusiasmo
    ("jaja", 0.7, 0.7), ("haha", 0.7, 0.7), ("jeje", 0.6, 0.5),
    ("!", 0.3, 0.5),  # exclamacion general
    ("increible", 0.8, 0.8), ("amazing", 0.8, 0.8), ("awesome", 0.8, 0.8),
    ("genial", 0.7, 0.7), ("great", 0.7, 0.6), ("love", 0.8, 0.7),
    ("me encanta", 0.8, 0.7), ("fantastico", 0.8, 0.8), ("fantastic", 0.8, 0.8),
    ("excelente", 0.7, 0.6), ("excellent", 0.7, 0.6),
    ("wow", 0.6, 0.8), ("woah", 0.6, 0.8),
    ("gracias", 0.6, 0.3), ("thanks", 0.6, 0.3), ("thank you", 0.6, 0.3),
    (":)", 0.5, 0.3), (":D", 0.7, 0.6), ("xd", 0.5, 0.5),
    ("😊", 0.6, 0.4), ("😄", 0.7, 0.6), ("🎉", 0.7, 0.7),
    ("❤", 0.7, 0.5), ("🥰", 0.8, 0.6),
]

_NEGATIVE_HIGH: list[tuple[str, float, float]] = [
    # Ira/frustracion
    ("mierda", -0.7, 0.8), ("shit", -0.7, 0.8), ("fuck", -0.8, 0.9),
    ("damn", -0.6, 0.7), ("maldita", -0.6, 0.7),
    ("odio", -0.8, 0.8), ("hate", -0.8, 0.8),
    ("no sirve", -0.6, 0.7), ("no funciona", -0.6, 0.7),
    ("estoy harto", -0.7, 0.8), ("fed up", -0.7, 0.8),
    ("me frustra", -0.6, 0.7), ("frustrating", -0.6, 0.7),
    ("ridiculo", -0.5, 0.7), ("ridiculous", -0.5, 0.7),
    ("horrible", -0.7, 0.7), ("terrible", -0.7, 0.6),
    ("😡", -0.7, 0.8), ("🤬", -0.8, 0.9), ("💢", -0.6, 0.7),
]

_NEGATIVE_LOW: list[tuple[str, float, float]] = [
    # Tristeza/desesperanza
    ("triste", -0.7, 0.3), ("sad", -0.7, 0.3),
    ("me siento mal", -0.6, 0.3), ("feel bad", -0.6, 0.3),
    ("solo", -0.5, 0.2), ("alone", -0.5, 0.2), ("lonely", -0.6, 0.2),
    ("no puedo mas", -0.8, 0.3), ("can't anymore", -0.8, 0.3),
    ("deprimido", -0.8, 0.2), ("depressed", -0.8, 0.2),
    ("no tiene sentido", -0.7, 0.2), ("pointless", -0.7, 0.2),
    ("cansado", -0.4, 0.2), ("tired", -0.4, 0.2), ("exhausted", -0.6, 0.2),
    ("decepciona", -0.5, 0.3), ("disappointed", -0.5, 0.3),
    ("😢", -0.6, 0.3), ("😭", -0.7, 0.4), ("💔", -0.7, 0.4),
    (":(", -0.5, 0.3),
]

_FEAR_ANXIETY: list[tuple[str, float, float]] = [
    # Miedo/ansiedad
    ("miedo", -0.6, 0.7), ("afraid", -0.6, 0.7), ("scared", -0.7, 0.8),
    ("preocupa", -0.4, 0.6), ("worried", -0.4, 0.6), ("worry", -0.4, 0.6),
    ("ansiedad", -0.5, 0.7), ("anxiety", -0.5, 0.7), ("anxious", -0.5, 0.7),
    ("nervioso", -0.4, 0.7), ("nervous", -0.4, 0.7),
    ("no se que hacer", -0.5, 0.6), ("don't know what to do", -0.5, 0.6),
    ("panico", -0.8, 0.9), ("panic", -0.8, 0.9),
    ("😰", -0.5, 0.7), ("😨", -0.6, 0.8), ("😱", -0.7, 0.9),
]

# Todas las listas combinadas para busqueda
_ALL_PATTERNS: list[tuple[str, float, float]] = (
    _POSITIVE_HIGH + _NEGATIVE_HIGH + _NEGATIVE_LOW + _FEAR_ANXIETY
)

# Patrones estructurales que amplifican la senal
_CAPS_PATTERN = re.compile(r"[A-Z]{3,}")  # GRITAR
_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")  # noooo, siiii
_ELLIPSIS_PATTERN = re.compile(r"\.{3,}")  # ...
_QUESTION_REPEAT = re.compile(r"\?{2,}")  # ???


def detect_user_emotion(stimulus: str) -> tuple[float, float, float]:
    """Detecta la emocion del usuario via patrones linguisticos.

    No usa LLM — es un proceso rapido y pre-cognitivo.
    Analiza lexicon, estructura, y senales paralingisticas del texto.

    Args:
        stimulus: Texto del usuario.

    Returns:
        (valence, arousal, signal_strength)
        valence: -1 a 1
        arousal: 0 a 1
        signal_strength: 0 a 1 (claridad de la senal)
    """
    if not stimulus or not stimulus.strip():
        return 0.0, 0.3, 0.0

    lower = stimulus.lower()
    total_valence = 0.0
    total_arousal = 0.0
    total_weight = 0.0
    matches = 0

    # 1. Buscar patrones en el lexicon
    for pattern, val, aro in _ALL_PATTERNS:
        if pattern in lower:
            weight = 1.0
            # Patrones mas largos son mas especificos y confiables
            if len(pattern) > 5:
                weight = 1.5
            total_valence += val * weight
            total_arousal += aro * weight
            total_weight += weight
            matches += 1

    # 2. Senales estructurales
    structural_arousal_boost = 0.0

    # MAYUSCULAS = intensidad
    caps_matches = _CAPS_PATTERN.findall(stimulus)
    if caps_matches:
        structural_arousal_boost += 0.15 * min(len(caps_matches), 3)

    # Repeticion de caracteres (noooo, siiiii)
    if _REPEAT_PATTERN.search(stimulus):
        structural_arousal_boost += 0.1

    # Puntos suspensivos = incertidumbre, baja energia
    if _ELLIPSIS_PATTERN.search(stimulus):
        structural_arousal_boost -= 0.1

    # Multiples signos de pregunta = ansiedad/urgencia
    if _QUESTION_REPEAT.search(stimulus):
        structural_arousal_boost += 0.1

    # Multiples exclamaciones = intensidad
    exclamation_count = stimulus.count("!")
    if exclamation_count >= 2:
        structural_arousal_boost += 0.1 * min(exclamation_count - 1, 3)

    # 3. Calcular resultado
    if matches == 0 and abs(structural_arousal_boost) < 0.05:
        # No se detecto nada — senal neutra
        return 0.0, 0.3, 0.0

    if total_weight > 0:
        avg_valence = total_valence / total_weight
        avg_arousal = total_arousal / total_weight
    else:
        avg_valence = 0.0
        avg_arousal = 0.3

    # Aplicar boost estructural al arousal
    avg_arousal = _clamp(avg_arousal + structural_arousal_boost, 0, 1)

    # Signal strength: cuantos patrones matchearon vs longitud del texto
    # Mas matches = senal mas clara
    word_count = max(len(stimulus.split()), 1)
    density = min(matches / (word_count * 0.3), 1.0)
    signal_strength = _clamp(
        density * 0.6 + min(matches / 3, 1.0) * 0.4,
        0, 1,
    )

    # Si hay senales estructurales fuertes, subir signal_strength
    if structural_arousal_boost > 0.1:
        signal_strength = _clamp(signal_strength + 0.15, 0, 1)

    return (
        _clamp(round(avg_valence, 4), -1, 1),
        _clamp(round(avg_arousal, 4), 0, 1),
        _clamp(round(signal_strength, 4), 0, 1),
    )


def update_shadow_state(
    shadow: ShadowState,
    detected_valence: float,
    detected_arousal: float,
    signal_strength: float,
) -> ShadowState:
    """Actualiza el shadow state con la nueva deteccion.

    El shadow state tiene inercia: no cambia drasticamente por un solo mensaje.
    Si no hay senal, decae lentamente hacia neutral.

    Args:
        shadow: Shadow state actual.
        detected_valence: Valencia detectada del usuario.
        detected_arousal: Arousal detectado del usuario.
        signal_strength: Fuerza de la senal (0-1).

    Returns:
        Shadow state actualizado.
    """
    if signal_strength < 0.05:
        # No hay senal significativa — decaer shadow state
        decay_rate = 0.15
        new_valence = shadow.valence * (1 - decay_rate)
        new_arousal = shadow.arousal * (1 - decay_rate) + 0.3 * decay_rate
        new_signal = shadow.signal_strength * (1 - decay_rate)
        new_turns = shadow.turns_since_strong_signal + 1

        return ShadowState(
            valence=round(_clamp(new_valence, -1, 1), 4),
            arousal=round(_clamp(new_arousal, 0, 1), 4),
            signal_strength=round(_clamp(new_signal, 0, 1), 4),
            accumulated_contagion=shadow.accumulated_contagion,
            turns_since_strong_signal=new_turns,
        )

    # Hay senal: mezclar con el shadow state actual
    # La mezcla depende de la fuerza de la senal
    blend = 0.3 + signal_strength * 0.4  # 0.3-0.7 blend hacia lo nuevo
    new_valence = shadow.valence * (1 - blend) + detected_valence * blend
    new_arousal = shadow.arousal * (1 - blend) + detected_arousal * blend
    new_signal = shadow.signal_strength * 0.5 + signal_strength * 0.5

    # Contagion acumulado crece con senales fuertes
    contagion_increment = signal_strength * 0.1
    new_accumulated = _clamp(shadow.accumulated_contagion + contagion_increment, 0, 1)

    return ShadowState(
        valence=round(_clamp(new_valence, -1, 1), 4),
        arousal=round(_clamp(new_arousal, 0, 1), 4),
        signal_strength=round(_clamp(new_signal, 0, 1), 4),
        accumulated_contagion=round(new_accumulated, 4),
        turns_since_strong_signal=0 if signal_strength > 0.3 else shadow.turns_since_strong_signal + 1,
    )


def compute_contagion_perturbation(
    shadow: ShadowState,
    current_valence: float,
    current_arousal: float,
    personality: PersonalityProfile,
    rapport: float = 0.3,
) -> tuple[float, float]:
    """Calcula la perturbacion de contagio para inyectar en el ODE.

    La perturbacion "tira" las dimensiones del agente hacia las del usuario.
    Es pre-cognitiva: no evalua si el contagio es bueno o malo.

    Args:
        shadow: Shadow state actual del usuario.
        current_valence: Valence actual del agente.
        current_arousal: Arousal actual del agente.
        personality: Perfil de personalidad del agente.
        rapport: Nivel de rapport con el usuario (0-1).

    Returns:
        (valence_perturbation, arousal_perturbation)
        Valores a sumar en el ODE step.
    """
    # Sin senal significativa, no hay contagio
    if shadow.signal_strength < 0.1:
        return 0.0, 0.0

    # Susceptibilidad al contagio (de personalidad)
    susceptibility = personality.contagion_susceptibility

    # Rapport modula: mas cercania = mas contagio
    # Minimo 0.3 (incluso con desconocidos hay algo de contagio)
    rapport_factor = 0.3 + rapport * 0.7

    # Fuerza total del contagio
    contagion_strength = (
        shadow.signal_strength
        * susceptibility
        * rapport_factor
    )

    # La perturbacion "tira" hacia el shadow state
    valence_delta = shadow.valence - current_valence
    arousal_delta = shadow.arousal - current_arousal

    # Escalar por fuerza de contagio
    # Factor 0.3 para que el contagio sea sutil, no dominante
    valence_perturbation = valence_delta * contagion_strength * 0.3
    arousal_perturbation = arousal_delta * contagion_strength * 0.3

    return (
        round(_clamp(valence_perturbation, -0.3, 0.3), 4),
        round(_clamp(arousal_perturbation, -0.3, 0.3), 4),
    )
