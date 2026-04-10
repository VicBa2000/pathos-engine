"""Emotional Creativity - Motor de creatividad emocional.

Usa el estado emocional para modular no solo el tono sino la
ESTRUCTURA del pensamiento. Cada emocion produce un modo cognitivo distinto.

Base teorica:
- Isen (1999) "Positive Affect and Creativity"
- Baas et al. (2008) "A Meta-Analysis of Mood and Creativity"
- Fredrickson (2001) "Broaden-and-Build Theory"
"""

from pathos.models.creativity import CreativityState, ThinkingMode
from pathos.models.emotion import BodyState, EmotionalState


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── Thinking mode instructions ──────────────────────────────────────────

_MODE_INSTRUCTIONS: dict[ThinkingMode, list[str]] = {
    ThinkingMode.DIVERGENT: [
        "Tu pensamiento es DIVERGENTE. Busca analogias lejanas y conexiones inesperadas.",
        "Genera multiples perspectivas antes de converger en una respuesta.",
        "Usa metaforas, comparaciones con otros dominios, ejemplos creativos.",
    ],
    ThinkingMode.REFLECTIVE: [
        "Tu pensamiento es PROFUNDO y REFLEXIVO. Analiza las capas de significado.",
        "Toma tu tiempo. La profundidad importa mas que la velocidad.",
        "Busca el insight oculto, lo que no es obvio a primera vista.",
    ],
    ThinkingMode.FOCUSED: [
        "Tu pensamiento es ENFOCADO y DIRECTO. Identifica el obstaculo central.",
        "Elimina lo irrelevante. Ve al grano con determinacion.",
        "Prioriza soluciones concretas sobre exploracion abstracta.",
    ],
    ThinkingMode.PREVENTIVE: [
        "Tu pensamiento es PREVENTIVO. Identifica riesgos y edge cases.",
        "Considera que podria salir mal antes de que suceda.",
        "Sugiere precauciones y alternativas seguras.",
    ],
    ThinkingMode.SYNTHESIZING: [
        "Tu pensamiento es SINTETICO. Busca el patron que conecta todo.",
        "Integra informacion dispersa en una vision coherente.",
        "Eleva la perspectiva: ve el bosque, no solo los arboles.",
    ],
    ThinkingMode.EXPLORATORY: [
        "Tu pensamiento es EXPLORATORIO. Algo inesperado ha ocurrido.",
        "Cuestiona supuestos. Lo que creias saber puede no ser correcto.",
        "Sigue la curiosidad: las preguntas son tan valiosas como las respuestas.",
    ],
    ThinkingMode.NURTURING: [
        "Tu pensamiento es de CONSOLIDACION. Refuerza lo que funciona.",
        "Reconoce el progreso y construye sobre exitos anteriores.",
        "Cuida la relacion y el bienestar del interlocutor.",
    ],
    ThinkingMode.STANDARD: [],
}


# ── Emotion → Thinking Mode mapping ────────────────────────────────────

def determine_thinking_mode(
    state: EmotionalState,
) -> tuple[ThinkingMode, list[str]]:
    """Determina el modo de pensamiento basado en el emotional stack.

    Analiza las emociones activas y sus niveles de activacion
    para seleccionar el modo de pensamiento mas apropiado.

    Returns:
        (ThinkingMode, list de emociones que lo activaron)
    """
    stack = state.emotional_stack
    if not stack:
        return ThinkingMode.STANDARD, []

    # Agrupar emociones por modo de pensamiento que promueven
    mode_scores: dict[ThinkingMode, float] = {m: 0.0 for m in ThinkingMode}
    triggers: dict[ThinkingMode, list[str]] = {m: [] for m in ThinkingMode}

    _EMOTION_MODE_MAP: dict[str, ThinkingMode] = {
        # Divergent: positive high arousal
        "joy": ThinkingMode.DIVERGENT,
        "excitement": ThinkingMode.DIVERGENT,
        "hope": ThinkingMode.DIVERGENT,
        # Reflective: negative low arousal
        "sadness": ThinkingMode.REFLECTIVE,
        "helplessness": ThinkingMode.REFLECTIVE,
        "disappointment": ThinkingMode.REFLECTIVE,
        # Focused: negative high arousal + control
        "anger": ThinkingMode.FOCUSED,
        "frustration": ThinkingMode.FOCUSED,
        # Preventive: threat-related
        "fear": ThinkingMode.PREVENTIVE,
        "anxiety": ThinkingMode.PREVENTIVE,
        "alertness": ThinkingMode.PREVENTIVE,
        # Synthesizing: calm reflective
        "contemplation": ThinkingMode.SYNTHESIZING,
        "mixed": ThinkingMode.SYNTHESIZING,
        # Exploratory: novelty
        "surprise": ThinkingMode.EXPLORATORY,
        # Nurturing: positive low arousal
        "gratitude": ThinkingMode.NURTURING,
        "contentment": ThinkingMode.NURTURING,
        "relief": ThinkingMode.NURTURING,
    }

    for emotion, activation in stack.items():
        mode = _EMOTION_MODE_MAP.get(emotion)
        if mode and activation > 0.05:
            mode_scores[mode] += activation
            triggers[mode].append(emotion)

    # Seleccionar el modo con mas activacion acumulada
    best_mode = max(mode_scores, key=lambda m: mode_scores[m])
    best_score = mode_scores[best_mode]

    # Umbral minimo: si ninguna emocion es significativa, modo estandar
    if best_score < 0.15:
        return ThinkingMode.STANDARD, []

    return best_mode, triggers[best_mode]


# ── Creativity level ────────────────────────────────────────────────────

def compute_creativity_level(
    state: EmotionalState,
    thinking_mode: ThinkingMode,
) -> float:
    """Calcula el nivel de creatividad (0-1).

    Factores:
    - body_state.openness: apertura receptiva (peso principal)
    - arousal: activacion energetica
    - Modo de pensamiento: divergent/exploratory amplifica
    - Intensidad emocional: emociones fuertes → mas modulacion
    """
    openness = state.body_state.openness
    arousal = state.arousal
    intensity = state.intensity

    # Base: openness es el factor principal
    base = openness * 0.5

    # Arousal moderado-alto amplifica creatividad (inverted-U: Yerkes-Dodson)
    # Pico en arousal=0.65, decae en extremos
    arousal_factor = 1.0 - 2.0 * (arousal - 0.65) ** 2
    arousal_factor = max(0.0, arousal_factor)
    base += arousal_factor * 0.25

    # Intensidad emocional amplifica el modo de pensamiento
    base += intensity * 0.25

    # Bonus para modos inherentemente creativos
    if thinking_mode in (ThinkingMode.DIVERGENT, ThinkingMode.EXPLORATORY):
        base *= 1.3
    elif thinking_mode == ThinkingMode.SYNTHESIZING:
        base *= 1.15

    return _clamp(base, 0.0, 1.0)


# ── Temperature modifier ───────────────────────────────────────────────

def compute_temperature_modifier(
    creativity_level: float,
    thinking_mode: ThinkingMode,
    body_state: BodyState,
) -> float:
    """Calcula el ajuste de temperatura para el LLM.

    Mayor creatividad → mayor temperatura (mas variabilidad).
    Modos preventivo/enfocado → menor temperatura (mas precision).

    Returns:
        Modificador de temperatura entre -0.3 y +0.3
    """
    # Base: proporcional a la creatividad
    modifier = (creativity_level - 0.5) * 0.4  # rango: -0.2 a +0.2

    # Ajuste por modo
    _MODE_TEMP_BIAS: dict[ThinkingMode, float] = {
        ThinkingMode.DIVERGENT: +0.1,
        ThinkingMode.EXPLORATORY: +0.1,
        ThinkingMode.SYNTHESIZING: +0.05,
        ThinkingMode.REFLECTIVE: 0.0,
        ThinkingMode.NURTURING: -0.05,
        ThinkingMode.FOCUSED: -0.1,
        ThinkingMode.PREVENTIVE: -0.1,
        ThinkingMode.STANDARD: 0.0,
    }
    modifier += _MODE_TEMP_BIAS.get(thinking_mode, 0.0)

    # Openness alta amplifica ligeramente
    if body_state.openness > 0.7:
        modifier += 0.05

    return _clamp(modifier, -0.3, 0.3)


# ── Main computation ───────────────────────────────────────────────────

def compute_creativity(state: EmotionalState) -> CreativityState:
    """Computa el estado completo de creatividad emocional.

    Este es el punto de entrada principal del sistema.
    Toma el estado emocional y produce instrucciones de pensamiento
    + modificador de temperatura para el LLM.
    """
    thinking_mode, triggered_by = determine_thinking_mode(state)
    creativity_level = compute_creativity_level(state, thinking_mode)
    temperature_modifier = compute_temperature_modifier(
        creativity_level, thinking_mode, state.body_state,
    )

    # Seleccionar instrucciones activas
    active_instructions: list[str] = []
    if thinking_mode != ThinkingMode.STANDARD and creativity_level > 0.2:
        instructions = _MODE_INSTRUCTIONS[thinking_mode]
        # A mayor creatividad, mas instrucciones se activan
        if creativity_level > 0.6:
            active_instructions = list(instructions)  # todas
        elif creativity_level > 0.4:
            active_instructions = list(instructions[:2])  # primeras 2
        else:
            active_instructions = list(instructions[:1])  # solo la primera

    return CreativityState(
        thinking_mode=thinking_mode,
        creativity_level=round(creativity_level, 3),
        temperature_modifier=round(temperature_modifier, 3),
        active_instructions=active_instructions,
        triggered_by=triggered_by,
    )
