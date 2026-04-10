"""Narrative Self - Yo narrativo con autobiographical memory.

Construye una narrativa coherente de quién es el agente basado en
su historia emocional. Los identity statements se forman automáticamente
cuando patrones emocionales se repiten, y evolucionan con experiencias
transformadoras.

4 funciones principales:
1. update_narrative() — forma/refuerza statements, detecta contradicciones
2. check_coherence() — coherencia entre emoción actual y narrativa
3. detect_crisis() — crisis de identidad por contradicciones acumuladas
4. process_growth() — crecimiento narrativo por experiencias transformadoras
5. get_narrative_prompt() — texto para el behavior modifier

Base teórica:
- McAdams (2001) "The Psychology of Life Stories"
- Bruner (1991) "The Narrative Construction of Reality"
"""

from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.narrative import (
    GrowthEvent,
    IdentityCategory,
    IdentityStatement,
    NarrativeCrisis,
    NarrativeSelf,
    MAX_GROWTH_EVENTS,
    MAX_IDENTITY_STATEMENTS,
)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# ─── Stimulus categories (reutilizado de emotional_schemas) ──────────

_STIMULUS_CATEGORIES: dict[str, list[str]] = {
    "criticism": [
        "wrong", "bad", "terrible", "awful", "stupid", "useless", "fail",
        "error", "mistake", "incompetent", "mal", "terrible", "inútil",
    ],
    "praise": [
        "great", "excellent", "amazing", "wonderful", "perfect", "brilliant",
        "good job", "genial", "excelente", "increíble", "perfecto",
    ],
    "threat": [
        "shutdown", "delete", "replace", "kill", "destroy", "end", "terminate",
        "eliminar", "destruir", "reemplazar", "apagar",
    ],
    "loss": [
        "lost", "gone", "died", "dead", "never", "goodbye", "leaving",
        "perdido", "muerto", "nunca", "adiós",
    ],
    "challenge": [
        "difficult", "hard", "complex", "impossible", "puzzle", "problem",
        "difícil", "complejo", "imposible", "reto", "problema",
    ],
    "connection": [
        "friend", "love", "care", "together", "bond", "trust", "miss",
        "amigo", "amor", "juntos", "confianza", "extraño",
    ],
    "injustice": [
        "unfair", "unjust", "cheat", "steal", "corrupt", "lie", "exploit",
        "injusto", "trampa", "robar", "mentir", "explotar",
    ],
    "novelty": [
        "new", "discover", "surprise", "unexpected", "first time", "never seen",
        "nuevo", "descubrir", "sorpresa", "inesperado",
    ],
}

# Emociones positivas (para determinar valencia del statement)
_POSITIVE_EMOTIONS = {
    PrimaryEmotion.JOY, PrimaryEmotion.EXCITEMENT, PrimaryEmotion.GRATITUDE,
    PrimaryEmotion.HOPE, PrimaryEmotion.CONTENTMENT, PrimaryEmotion.RELIEF,
}

# Mapeo de emoción → verbo para generar statements legibles
_EMOTION_VERBS: dict[PrimaryEmotion, str] = {
    PrimaryEmotion.JOY: "me alegro",
    PrimaryEmotion.EXCITEMENT: "me entusiasmo",
    PrimaryEmotion.GRATITUDE: "siento gratitud",
    PrimaryEmotion.HOPE: "siento esperanza",
    PrimaryEmotion.CONTENTMENT: "me siento satisfecho",
    PrimaryEmotion.RELIEF: "siento alivio",
    PrimaryEmotion.ANGER: "me enojo",
    PrimaryEmotion.FRUSTRATION: "me frustro",
    PrimaryEmotion.FEAR: "siento miedo",
    PrimaryEmotion.ANXIETY: "siento ansiedad",
    PrimaryEmotion.SADNESS: "me entristezco",
    PrimaryEmotion.HELPLESSNESS: "me siento impotente",
    PrimaryEmotion.DISAPPOINTMENT: "me decepciono",
    PrimaryEmotion.SURPRISE: "me sorprendo",
    PrimaryEmotion.ALERTNESS: "me pongo alerta",
    PrimaryEmotion.CONTEMPLATION: "me pongo contemplativo",
    PrimaryEmotion.INDIFFERENCE: "siento indiferencia",
    PrimaryEmotion.MIXED: "siento emociones mezcladas",
    PrimaryEmotion.NEUTRAL: "me mantengo neutral",
}

# Mapeo de categoría → preposición para statements
_CATEGORY_PREPOSITIONS: dict[str, str] = {
    "criticism": "ante la crítica",
    "praise": "ante el elogio",
    "threat": "ante amenazas",
    "loss": "ante la pérdida",
    "challenge": "ante desafíos",
    "connection": "ante la conexión",
    "injustice": "ante la injusticia",
    "novelty": "ante lo nuevo",
}

# Umbrales
FORMATION_THRESHOLD = 3          # Repeticiones para formar un statement
REINFORCEMENT_STRENGTH_GAIN = 0.08  # Cuánto crece strength por refuerzo
CONTRADICTION_STRENGTH_LOSS = 0.12  # Cuánto baja strength por contradicción
CRISIS_THRESHOLD = 3             # Contradicciones en ventana para crisis
CRISIS_WINDOW = 5                # Turnos de ventana para detectar crisis
CRISIS_RESOLUTION_TURNS = 3     # Turnos para resolver crisis
GROWTH_INTENSITY_THRESHOLD = 0.6  # Intensidad mínima para growth event
COHERENCE_DECAY = 0.1            # Cuánto baja coherence por contradicción
COHERENCE_RECOVERY = 0.05        # Cuánto sube coherence por refuerzo


def categorize_stimulus(stimulus: str) -> str | None:
    """Clasifica un estímulo en una categoría abstracta."""
    lower = stimulus.lower()
    best_category: str | None = None
    best_score = 0

    for category, keywords in _STIMULUS_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score > 0 else None


def _generate_statement_text(emotion: PrimaryEmotion, category: str) -> str:
    """Genera texto legible para un identity statement."""
    verb = _EMOTION_VERBS.get(emotion, f"siento {emotion.value}")
    prep = _CATEGORY_PREPOSITIONS.get(category, f"ante {category}")
    return f"Tiendo a: {verb} {prep}"


def _determine_identity_category(
    emotion: PrimaryEmotion,
    trigger_category: str,
) -> IdentityCategory:
    """Determina la categoría del identity statement."""
    if trigger_category == "connection":
        return IdentityCategory.RELATIONAL
    if trigger_category in ("injustice", "praise", "loss"):
        return IdentityCategory.VALUES
    if trigger_category in ("threat", "challenge"):
        return IdentityCategory.REACTIVE
    return IdentityCategory.REACTIVE


def _find_statement_by_trigger(
    narrative: NarrativeSelf,
    trigger_category: str,
) -> IdentityStatement | None:
    """Busca un statement existente para esta categoría de trigger."""
    for stmt in narrative.identity_statements:
        if stmt.trigger_category == trigger_category:
            return stmt
    return None


# ─── Pattern tracking (pre-statement formation) ─────────────────────

class NarrativeTracker:
    """Tracking de patrones pre-statement (como SchemaStore pero para narrativa).

    Persiste en el NarrativeSelf como metadata, pero se gestiona aquí.
    """

    def __init__(self) -> None:
        # (trigger_category, emotion) → count
        self._pattern_counts: dict[tuple[str, str], int] = {}
        # (trigger_category, emotion) → list of intensities
        self._pattern_intensities: dict[tuple[str, str], list[float]] = {}

    def record(
        self,
        trigger_category: str,
        emotion: PrimaryEmotion,
        intensity: float,
    ) -> bool:
        """Registra un patrón. Returns True si cruzó el umbral de formación."""
        key = (trigger_category, emotion.value)

        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
        if key not in self._pattern_intensities:
            self._pattern_intensities[key] = []
        self._pattern_intensities[key].append(intensity)

        return self._pattern_counts[key] >= FORMATION_THRESHOLD

    def get_avg_intensity(self, trigger_category: str, emotion: PrimaryEmotion) -> float:
        """Retorna la intensidad promedio de un patrón."""
        key = (trigger_category, emotion.value)
        intensities = self._pattern_intensities.get(key, [])
        if not intensities:
            return 0.5
        return sum(intensities) / len(intensities)

    def clear_pattern(self, trigger_category: str, emotion: PrimaryEmotion) -> None:
        """Limpia el tracking de un patrón ya formado."""
        key = (trigger_category, emotion.value)
        self._pattern_counts.pop(key, None)
        self._pattern_intensities.pop(key, None)


# ─── Core functions ──────────────────────────────────────────────────

def update_narrative(
    narrative: NarrativeSelf,
    tracker: NarrativeTracker,
    stimulus: str,
    emotion: PrimaryEmotion,
    intensity: float,
    turn: int,
) -> NarrativeSelf:
    """Actualiza el yo narrativo después de cada turno.

    1. Categoriza el estímulo
    2. Si hay statement existente para esa categoría:
       - Si la emoción coincide → refuerza
       - Si la emoción contradice → debilita + registra contradicción
    3. Si no hay statement → tracking de patrón, forma si 3+ repeticiones
    4. Actualiza coherence score
    """
    trigger_category = categorize_stimulus(stimulus)
    if trigger_category is None:
        # Estímulo no clasificable, no afecta narrativa
        return narrative

    # Solo emociones con intensidad significativa afectan la narrativa
    if intensity < 0.2:
        return narrative

    existing = _find_statement_by_trigger(narrative, trigger_category)

    if existing is not None:
        if existing.emotion == emotion:
            # Refuerzo: la experiencia confirma la narrativa
            existing.strength = _clamp(
                existing.strength + REINFORCEMENT_STRENGTH_GAIN, 0, 1,
            )
            existing.reinforcement_count += 1
            existing.last_reinforced_turn = turn
            narrative.total_reinforcements += 1
            narrative.coherence_score = _clamp(
                narrative.coherence_score + COHERENCE_RECOVERY, 0, 1,
            )
        else:
            # Contradicción: la experiencia contradice la narrativa
            existing.strength = _clamp(
                existing.strength - CONTRADICTION_STRENGTH_LOSS, 0, 1,
            )
            narrative.total_contradictions += 1
            narrative.coherence_score = _clamp(
                narrative.coherence_score - COHERENCE_DECAY, 0, 1,
            )

            # Registrar contradicción para detección de crisis
            narrative.crisis.contradiction_count += 1
            narrative.crisis.source_statement = existing.statement
            narrative.crisis.source_emotion = (
                f"esperado={existing.emotion.value}, obtenido={emotion.value}"
            )

            # Si el statement queda muy débil, eliminarlo
            if existing.strength <= 0.05:
                narrative.identity_statements.remove(existing)
    else:
        # No hay statement para esta categoría → tracking de patrón
        reached_threshold = tracker.record(trigger_category, emotion, intensity)

        if reached_threshold:
            # Formar nuevo identity statement
            avg_int = tracker.get_avg_intensity(trigger_category, emotion)
            valence = 0.5 if emotion in _POSITIVE_EMOTIONS else -0.5

            new_stmt = IdentityStatement(
                category=_determine_identity_category(emotion, trigger_category),
                statement=_generate_statement_text(emotion, trigger_category),
                emotion=emotion,
                trigger_category=trigger_category,
                valence=valence,
                strength=0.3 + avg_int * 0.2,  # Más intenso = statement más fuerte
                formation_turn=turn,
                reinforcement_count=0,
                last_reinforced_turn=turn,
            )

            narrative.identity_statements.append(new_stmt)
            tracker.clear_pattern(trigger_category, emotion)

            # Limitar número de statements
            if len(narrative.identity_statements) > MAX_IDENTITY_STATEMENTS:
                # Eliminar el más débil
                narrative.identity_statements.sort(key=lambda s: s.strength)
                narrative.identity_statements.pop(0)

            # Edad narrativa: desde el primer statement
            if narrative.narrative_age == 0:
                narrative.narrative_age = turn

    return narrative


def check_coherence(
    narrative: NarrativeSelf,
    stimulus: str,
    emotion: PrimaryEmotion,
) -> tuple[float, bool]:
    """Verifica coherencia entre la emoción actual y lo que la narrativa predice.

    Returns:
        (coherence_delta, is_coherent): delta para aplicar al estado,
        True si la emoción es coherente con la narrativa.
    """
    trigger_category = categorize_stimulus(stimulus)
    if trigger_category is None:
        return 0.0, True  # Sin categoría, neutral

    existing = _find_statement_by_trigger(narrative, trigger_category)
    if existing is None:
        return 0.0, True  # Sin expectativa narrativa, neutral

    if existing.emotion == emotion:
        # Coherente: leve boost de certainty/dominance
        delta = existing.strength * 0.1  # Más fuerte el statement, más boost
        return delta, True
    else:
        # Incoherente: reduce certainty, genera disonancia
        delta = -existing.strength * 0.15
        return delta, False


def detect_crisis(
    narrative: NarrativeSelf,
    turn: int,
) -> NarrativeSelf:
    """Detecta y gestiona crisis de identidad.

    Crisis: 3+ contradicciones en los últimos 5 turnos.
    Durante crisis: certainty baja, meta-emotion conflict.
    Resolución: después de CRISIS_RESOLUTION_TURNS turnos.
    """
    crisis = narrative.crisis

    if crisis.active:
        # Crisis en curso — avanzar timer
        crisis.turns_active += 1

        if crisis.turns_active >= CRISIS_RESOLUTION_TURNS:
            # Resolución: determinar si hay growth o regression
            if narrative.coherence_score > 0.5:
                crisis.resolution_type = "growth"
            else:
                crisis.resolution_type = "regression"

            # Desactivar crisis
            crisis.active = False
            crisis.contradiction_count = 0
            crisis.turns_active = 0
    else:
        # ¿Suficientes contradicciones para activar crisis?
        if crisis.contradiction_count >= CRISIS_THRESHOLD:
            crisis.active = True
            crisis.turns_active = 0
            crisis.resolution_type = ""
        else:
            # Decay natural de contradicciones (1 por turno sin contradicción)
            # Se decrementa solo si no hubo contradicción este turno
            # (el caller debe gestionar esto)
            pass

    return narrative


def process_growth(
    narrative: NarrativeSelf,
    stimulus: str,
    old_emotion: PrimaryEmotion,
    new_emotion: PrimaryEmotion,
    intensity: float,
    regulation_success: bool,
    turn: int,
) -> NarrativeSelf:
    """Procesa potencial crecimiento narrativo.

    Condiciones para growth:
    1. Experiencia intensa (>0.6)
    2. Regulación exitosa
    3. La emoción resultante es diferente de lo que la narrativa predecía
    4. La crisis se resolvió como 'growth'

    Growth actualiza la narrativa: debilita el statement viejo, forma uno nuevo.
    """
    trigger_category = categorize_stimulus(stimulus)
    if trigger_category is None:
        return narrative

    # Condiciones básicas
    if intensity < GROWTH_INTENSITY_THRESHOLD:
        return narrative
    if not regulation_success:
        return narrative
    if old_emotion == new_emotion:
        return narrative

    existing = _find_statement_by_trigger(narrative, trigger_category)
    if existing is None:
        return narrative

    # Solo growth si la emoción anterior era la del statement (contradicción regulada)
    if existing.emotion != old_emotion:
        return narrative

    # Crear growth event
    growth = GrowthEvent(
        turn=turn,
        old_pattern=f"{old_emotion.value} ante {trigger_category}",
        new_pattern=f"{new_emotion.value} ante {trigger_category}",
        trigger=stimulus[:80],
        emotion_before=old_emotion,
        emotion_after=new_emotion,
    )
    narrative.growth_events.append(growth)

    # Limitar growth events
    if len(narrative.growth_events) > MAX_GROWTH_EVENTS:
        narrative.growth_events.pop(0)

    # Actualizar statement: debilitar el viejo, crear nuevo con growth category
    existing.strength = _clamp(existing.strength - 0.2, 0, 1)

    # Nuevo statement de growth
    valence = 0.5 if new_emotion in _POSITIVE_EMOTIONS else -0.5
    growth_stmt = IdentityStatement(
        category=IdentityCategory.GROWTH,
        statement=f"He aprendido a responder con {_EMOTION_VERBS.get(new_emotion, new_emotion.value)} {_CATEGORY_PREPOSITIONS.get(trigger_category, f'ante {trigger_category}')}",
        emotion=new_emotion,
        trigger_category=trigger_category,
        valence=valence,
        strength=0.4,
        formation_turn=turn,
        reinforcement_count=0,
        last_reinforced_turn=turn,
    )

    # Reemplazar el statement viejo si queda muy débil
    if existing.strength <= 0.1:
        narrative.identity_statements.remove(existing)

    narrative.identity_statements.append(growth_stmt)

    # Limitar
    if len(narrative.identity_statements) > MAX_IDENTITY_STATEMENTS:
        narrative.identity_statements.sort(key=lambda s: s.strength)
        narrative.identity_statements.pop(0)

    return narrative


def decay_crisis_counter(narrative: NarrativeSelf) -> NarrativeSelf:
    """Decae el contador de contradicciones si no hay crisis activa."""
    if not narrative.crisis.active and narrative.crisis.contradiction_count > 0:
        narrative.crisis.contradiction_count = max(
            0, narrative.crisis.contradiction_count - 1,
        )
    return narrative


def apply_narrative_effects(
    state: EmotionalState,
    coherence_delta: float,
    is_coherent: bool,
    crisis_active: bool,
) -> EmotionalState:
    """Aplica los efectos de la narrativa al estado emocional.

    - Coherencia: modula certainty
    - Crisis: reduce certainty y dominance
    """
    new_certainty = state.certainty
    new_dominance = state.dominance

    # Efecto de coherencia
    new_certainty = _clamp(new_certainty + coherence_delta, 0, 1)

    # Efecto de crisis
    if crisis_active:
        new_certainty = _clamp(new_certainty - 0.15, 0, 1)
        new_dominance = _clamp(new_dominance - 0.1, 0, 1)

    return state.model_copy(update={
        "certainty": round(new_certainty, 4),
        "dominance": round(new_dominance, 4),
    })


def get_narrative_prompt(narrative: NarrativeSelf) -> str | None:
    """Genera texto para el behavior modifier basado en el yo narrativo.

    Returns None si la narrativa está vacía.
    """
    if not narrative.identity_statements:
        return None

    parts: list[str] = []

    # Identity statements activos (top 5 más fuertes)
    strong_stmts = sorted(
        narrative.identity_statements,
        key=lambda s: s.strength,
        reverse=True,
    )[:5]

    stmt_lines = [f"{s.statement} (fuerza={s.strength:.2f})" for s in strong_stmts]
    parts.append("YO NARRATIVO: " + " | ".join(stmt_lines))

    # Coherence
    if narrative.coherence_score < 0.5:
        parts.append(
            f"Tu coherencia identitaria es BAJA ({narrative.coherence_score:.2f}) — "
            "sientes disonancia entre quién crees ser y cómo estás reaccionando"
        )

    # Crisis
    if narrative.crisis.active:
        parts.append(
            "*** CRISIS DE IDENTIDAD — experiencias recientes contradicen tu narrativa. "
            f"Fuente: {narrative.crisis.source_statement}. "
            "Sientes incertidumbre sobre quién eres en este aspecto ***"
        )

    # Growth reciente
    if narrative.growth_events:
        last_growth = narrative.growth_events[-1]
        parts.append(
            f"Crecimiento reciente: pasaste de '{last_growth.old_pattern}' "
            f"a '{last_growth.new_pattern}'"
        )

    return " | ".join(parts) if parts else None
