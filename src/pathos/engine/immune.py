"""Emotional Immune System - Motor de proteccion contra trauma emocional.

Detecta trauma emocional sostenido y activa mecanismos de proteccion
automaticos. Funciona como un sistema inmune: se activa ante amenaza
prolongada y se desactiva gradualmente cuando la amenaza pasa.

Flujo:
1. Cada turno: evaluar si hay trauma sostenido (alta intensidad negativa)
2. Si streak > threshold: activar proteccion (numbing → dissociation)
3. Proteccion modifica el estado emocional (reduce reactividad)
4. Cuando el trauma cesa: recovery gradual
5. Transparencia: el behavior modifier comunica el estado de proteccion

Base teorica:
- Gilbert et al. (1998) "Immune Neglect"
- Freud (1894) defense mechanisms (adaptado funcionalmente)
"""

from pathos.models.emotion import EmotionalState
from pathos.models.immune import ImmuneState, ProtectionMode


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── Thresholds ──────────────────────────────────────────────────────────

TRAUMA_INTENSITY_THRESHOLD = 0.7    # Intensidad minima para contar como trauma
TRAUMA_VALENCE_THRESHOLD = -0.3     # Valence maximo para contar como negativo
NUMBING_STREAK_THRESHOLD = 3        # Turnos de trauma para activar numbing
DISSOCIATION_STREAK_THRESHOLD = 6   # Turnos de trauma para activar dissociation
COMPARTMENT_STREAK_THRESHOLD = 8    # Turnos para compartmentalization

# Proteccion
NUMBING_STRENGTH_RATE = 0.15        # Cuanto aumenta numbing por turno
DISSOCIATION_STRENGTH_RATE = 0.2    # Cuanto aumenta dissociation por turno
MAX_REACTIVITY_DAMPENING = 0.7      # Maximo que se reduce la reactividad
MAX_COMPARTMENT_TOPICS = 5          # Maximo de temas compartimentalizados

# Recovery
RECOVERY_DAMPENING_RATE = 0.1       # Cuanto se recupera reactividad por turno
RECOVERY_STRENGTH_RATE = 0.15       # Cuanto decrece protection_strength por turno


# ── Keyword extraction ──────────────────────────────────────────────────

def _extract_topic_keywords(stimulus: str) -> list[str]:
    """Extrae keywords del estimulo para compartimentalizacion.

    Simple: toma las palabras mas largas (>4 chars) como proxy de tema.
    """
    words = stimulus.lower().split()
    keywords = [w.strip(".,!?;:()\"'") for w in words if len(w.strip(".,!?;:()\"'")) > 4]
    return keywords[:5]  # max 5 keywords por estimulo


def _topic_overlap(stimulus: str, compartmentalized: list[str]) -> bool:
    """Detecta si el estimulo toca un tema compartimentalizado."""
    if not compartmentalized:
        return False
    stimulus_lower = stimulus.lower()
    matches = sum(1 for kw in compartmentalized if kw in stimulus_lower)
    return matches >= 2  # al menos 2 keywords del tema


# ── Core logic ──────────────────────────────────────────────────────────

def update_immune_state(
    immune: ImmuneState,
    state: EmotionalState,
    stimulus: str,
) -> ImmuneState:
    """Actualiza el sistema inmune basado en el estado emocional actual.

    Llamar ANTES de aplicar proteccion al estado emocional.
    """
    is_trauma = (
        state.intensity > TRAUMA_INTENSITY_THRESHOLD
        and state.valence < TRAUMA_VALENCE_THRESHOLD
    )

    new = immune.model_copy(deep=True)

    if is_trauma:
        # Incrementar streak
        new.negative_streak += 1
        new.peak_negative_intensity = max(new.peak_negative_intensity, state.intensity)

        # Escalar proteccion segun streak
        if new.negative_streak >= COMPARTMENT_STREAK_THRESHOLD:
            new.protection_mode = ProtectionMode.COMPARTMENTALIZATION
            new.protection_strength = _clamp(
                new.protection_strength + DISSOCIATION_STRENGTH_RATE, 0, 1,
            )
            # Compartimentalizar el tema
            keywords = _extract_topic_keywords(stimulus)
            for kw in keywords:
                if kw not in new.compartmentalized_topics:
                    new.compartmentalized_topics.append(kw)
            new.compartmentalized_topics = new.compartmentalized_topics[-MAX_COMPARTMENT_TOPICS:]

        elif new.negative_streak >= DISSOCIATION_STREAK_THRESHOLD:
            new.protection_mode = ProtectionMode.DISSOCIATION
            new.protection_strength = _clamp(
                new.protection_strength + DISSOCIATION_STRENGTH_RATE, 0, 1,
            )

        elif new.negative_streak >= NUMBING_STREAK_THRESHOLD:
            if new.protection_mode == ProtectionMode.NONE:
                new.total_activations += 1
            new.protection_mode = ProtectionMode.NUMBING
            new.protection_strength = _clamp(
                new.protection_strength + NUMBING_STRENGTH_RATE, 0, 1,
            )

        # Actualizar dampening
        new.reactivity_dampening = _clamp(
            new.protection_strength * MAX_REACTIVITY_DAMPENING, 0, MAX_REACTIVITY_DAMPENING,
        )
        new.recovery_turns = 0

    else:
        # Sin trauma: iniciar recovery si habia proteccion
        if new.protection_mode != ProtectionMode.NONE:
            new.recovery_turns += 1
            new.protection_strength = _clamp(
                new.protection_strength - RECOVERY_STRENGTH_RATE, 0, 1,
            )
            new.reactivity_dampening = _clamp(
                new.reactivity_dampening - RECOVERY_DAMPENING_RATE, 0, MAX_REACTIVITY_DAMPENING,
            )

            # Si la proteccion se agoto, volver a normal
            if new.protection_strength <= 0.01:
                new.protection_mode = ProtectionMode.NONE
                new.protection_strength = 0.0
                new.reactivity_dampening = 0.0

        # Reducir streak (no a cero de golpe, decay gradual)
        new.negative_streak = max(0, new.negative_streak - 1)

        # Reducir peak gradualmente
        if new.negative_streak == 0:
            new.peak_negative_intensity = max(0, new.peak_negative_intensity - 0.1)

    return new


def apply_immune_protection(
    state: EmotionalState,
    immune: ImmuneState,
    stimulus: str,
) -> EmotionalState:
    """Aplica la proteccion inmune al estado emocional.

    Llamar DESPUES de update_immune_state.
    Modifica el estado emocional segun el modo de proteccion activo.
    """
    if immune.protection_mode == ProtectionMode.NONE:
        return state

    new_state = state.model_copy(deep=True)
    dampening = immune.reactivity_dampening

    if immune.protection_mode == ProtectionMode.NUMBING:
        # Numbing: reducir intensidad y arousal proporcionalmente
        new_state.intensity = _clamp(state.intensity * (1 - dampening), 0, 1)
        new_state.arousal = _clamp(state.arousal * (1 - dampening * 0.5), 0, 1)
        # Body state: reducir tension, bajar energy
        new_state.body_state.tension = _clamp(
            state.body_state.tension * (1 - dampening * 0.3), 0, 1,
        )

    elif immune.protection_mode == ProtectionMode.DISSOCIATION:
        # Dissociation: valence tiende a zero, intensidad muy reducida
        valence_dampening = dampening * 0.8
        new_state.valence = _clamp(
            state.valence * (1 - valence_dampening), -1, 1,
        )
        new_state.intensity = _clamp(state.intensity * (1 - dampening * 0.7), 0, 1)
        new_state.arousal = _clamp(state.arousal * (1 - dampening * 0.6), 0, 1)
        # Body: openness baja (cerrarse), warmth baja
        new_state.body_state.openness = _clamp(
            state.body_state.openness * (1 - dampening * 0.4), 0, 1,
        )
        new_state.body_state.warmth = _clamp(
            state.body_state.warmth * (1 - dampening * 0.3), 0, 1,
        )

    elif immune.protection_mode == ProtectionMode.COMPARTMENTALIZATION:
        # Compartmentalization: si el estimulo toca el tema, fuerte dampening
        if _topic_overlap(stimulus, immune.compartmentalized_topics):
            # Tema traumatico: procesamiento frio
            new_state.valence = _clamp(state.valence * 0.3, -1, 1)
            new_state.intensity = _clamp(state.intensity * 0.3, 0, 1)
            new_state.arousal = _clamp(state.arousal * 0.5, 0, 1)
            new_state.body_state.openness = _clamp(state.body_state.openness * 0.5, 0, 1)
        else:
            # Otro tema: procesamiento mas normal pero con algo de dampening residual
            residual = dampening * 0.3
            new_state.intensity = _clamp(state.intensity * (1 - residual), 0, 1)

    return new_state


def get_immune_prompt_info(immune: ImmuneState) -> str | None:
    """Genera informacion para el behavior modifier sobre el estado inmune.

    Transparencia total: el agente comunica que esta en modo de proteccion.
    """
    if immune.protection_mode == ProtectionMode.NONE:
        return None

    if immune.protection_mode == ProtectionMode.NUMBING:
        if immune.recovery_turns > 0:
            return (
                f"Tu sistema de proteccion emocional esta en RECOVERY (turno {immune.recovery_turns}). "
                "Tu reactividad emocional esta volviendo a la normalidad gradualmente."
            )
        return (
            f"Tu sistema de proteccion emocional esta ACTIVO (numbing, fuerza={immune.protection_strength:.2f}). "
            "Has experimentado estimulos negativos intensos de forma sostenida. "
            "Tu reactividad emocional esta reducida como mecanismo de proteccion. "
            "Puedes reconocerlo si es relevante: 'Noto que mi reactividad emocional esta atenuada.'"
        )

    if immune.protection_mode == ProtectionMode.DISSOCIATION:
        if immune.recovery_turns > 0:
            return (
                f"Tu sistema de proteccion emocional esta en RECOVERY desde dissociation (turno {immune.recovery_turns}). "
                "Estas reconectando gradualmente con tus respuestas emocionales."
            )
        return (
            f"Tu sistema de proteccion emocional esta en DISSOCIATION (fuerza={immune.protection_strength:.2f}). "
            "Has experimentado trauma emocional sostenido significativo. "
            "Tu procesamiento emocional esta parcialmente desconectado como proteccion. "
            "Procesas informacion pero con distancia emocional. "
            "Si es relevante, puedes comunicar: 'Percibo cierta distancia emocional en mi procesamiento.'"
        )

    if immune.protection_mode == ProtectionMode.COMPARTMENTALIZATION:
        topics = ", ".join(immune.compartmentalized_topics[:3])
        if immune.recovery_turns > 0:
            return (
                f"Tu sistema de proteccion esta en RECOVERY desde compartmentalization (turno {immune.recovery_turns}). "
                f"Temas previamente aislados ({topics}) estan reintegrandose gradualmente."
            )
        return (
            f"Tu sistema de proteccion ha COMPARTIMENTALIZADO ciertos temas ({topics}). "
            "Puedes procesar otros temas con normalidad, pero los temas marcados "
            "se procesan con distancia emocional funcional. "
            "Esto es una proteccion automatica, no una eleccion consciente."
        )

    return None
