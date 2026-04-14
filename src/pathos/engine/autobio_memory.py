"""Autobiographical Memory Engine - Motor de memoria autobiografica.

Pilar 3 de ANIMA: Memoria autobiografica con consolidacion onirica.
Basado en Endel Tulving (Memoria Episodica, 1972),
Martin Conway (Self-Memory System, 2005),
George Miller (7+-2, 1956).

Orquesta 4 niveles de memoria:
  1. Buffer Sensorial — captura turno actual, decae completamente
  2. Memoria de Trabajo — seleccion top-K por relevancia+intensidad+recencia
  3. Memoria Episodica — almacena experiencias significativas (intensity>0.5)
  4. Memoria Narrativa — destila generalizaciones de clusters episodicos

Sistema OPT-IN. Si state.enabled=False, todas las funciones retornan
resultados vacios sin modificar nada (backward compatible con v4).
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict

from pathos.models.autobio_memory import (
    EPISODIC_INTENSITY_THRESHOLD,
    EPISODIC_MAX_PER_SESSION,
    NARRATIVE_MAX_STATEMENTS,
    NARRATIVE_MIN_EPISODES,
    WORKING_MEMORY_CAPACITY,
    AutobiographicalState,
    Episode,
    EpisodeSignificance,
    EpisodicStore,
    MemoryItem,
    NarrativeStatement,
    NarrativeStore,
    NarrativeType,
    SensorySnapshot,
    WorkingMemoryState,
)
from pathos.models.emotion import EmotionalState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recency decay: recency = max(0, 1 - (current_turn - episode_turn) * DECAY_RATE)
RECENCY_DECAY_RATE = 0.02  # ~50 turnos para llegar a 0

# Keyword matching (reused from memory.py pattern)
_STOP_WORDS = {
    "el", "la", "los", "las", "un", "una", "de", "del", "en", "con", "por",
    "para", "que", "es", "no", "si", "se", "al", "lo", "le", "me", "te",
    "su", "mi", "tu", "ya", "muy", "mas", "the", "a", "an", "is", "are",
    "to", "of", "in", "and", "or", "it", "i", "you", "he", "she", "we",
    "do", "not", "but", "was", "be", "have", "has", "this", "that", "my",
}

# Emotional similarity threshold for narrative clustering
EMOTIONAL_SIMILARITY_THRESHOLD = 0.3

# Narrative reinforcement increment
NARRATIVE_REINFORCEMENT_STRENGTH = 0.1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _extract_keywords(text: str) -> list[str]:
    """Extrae keywords significativas de un texto."""
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


def _keyword_similarity(keywords_a: list[str], keywords_b: list[str]) -> float:
    """Jaccard similarity entre dos listas de keywords."""
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = set(keywords_a)
    set_b = set(keywords_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity entre dos vectores."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _emotional_distance(ep_a: Episode, ep_b: Episode) -> float:
    """Distancia emocional euclidiana normalizada entre dos episodios.

    Usa valence, arousal, intensity, dominance. Resultado 0-1 (0=identicos).
    """
    dv = ep_a.valence - ep_b.valence  # range -2 to 2
    da = ep_a.arousal - ep_b.arousal  # range -1 to 1
    di = ep_a.intensity - ep_b.intensity  # range -1 to 1
    dd = ep_a.dominance - ep_b.dominance  # range -1 to 1
    # Max possible distance: sqrt(4 + 1 + 1 + 1) = sqrt(7) ~ 2.646
    raw = math.sqrt(dv * dv + da * da + di * di + dd * dd)
    return _clamp(raw / 2.646, 0.0, 1.0)


def _classify_significance(intensity: float) -> EpisodeSignificance:
    """Clasifica la significancia de un episodio por intensidad."""
    if intensity > 0.8:
        return EpisodeSignificance.PEAK
    if intensity > 0.7:
        return EpisodeSignificance.HIGH
    if intensity > 0.6:
        return EpisodeSignificance.MODERATE
    return EpisodeSignificance.LOW


# ---------------------------------------------------------------------------
# Level 1: Sensory Buffer
# ---------------------------------------------------------------------------

def capture_sensory(
    stimulus: str,
    emotional_state: EmotionalState,
    prediction_error: float,
    turn_number: int,
) -> SensorySnapshot:
    """Captura el turno actual en el buffer sensorial.

    Llamado al inicio de cada turno. El snapshot anterior se descarta.
    """
    return SensorySnapshot(
        stimulus=stimulus[:500],  # Truncar para eficiencia
        appraisal_valence=emotional_state.valence,
        appraisal_relevance=_clamp(emotional_state.intensity, 0.0, 1.0),
        prediction_error=_clamp(prediction_error, 0.0, 1.0),
        primary_emotion=emotional_state.primary_emotion.value,
        intensity=emotional_state.intensity,
        turn_number=turn_number,
    )


# ---------------------------------------------------------------------------
# Level 2: Working Memory
# ---------------------------------------------------------------------------

def update_working_memory(
    state: AutobiographicalState,
    current_stimulus: str,
    current_turn: int,
    stimulus_embedding: list[float] | None = None,
) -> WorkingMemoryState:
    """Actualiza la memoria de trabajo seleccionando top-K items relevantes.

    Fuentes: episodic store + narrative store.
    Criterio: relevance * 0.4 + intensity * 0.35 + recency * 0.25
    """
    candidates: list[MemoryItem] = []
    current_keywords = _extract_keywords(current_stimulus)

    # Candidatos desde memoria episodica
    for ep in state.episodic.episodes:
        relevance = _compute_episode_relevance(
            ep, current_keywords, stimulus_embedding,
        )
        recency = max(0.0, 1.0 - (current_turn - ep.turn_number) * RECENCY_DECAY_RATE)

        item = MemoryItem(
            source_id=ep.id,
            source_type="episodic",
            content=f"[{ep.primary_emotion}] {ep.stimulus[:80]}",
            relevance=relevance,
            emotional_intensity=ep.intensity,
            recency=recency,
        )
        item.compute_composite()
        candidates.append(item)

    # Candidatos desde memoria narrativa
    for ns in state.narrative.statements:
        relevance = _compute_narrative_relevance(ns, current_keywords)
        recency = max(0.0, 1.0 - (current_turn - ns.last_reinforced_turn) * RECENCY_DECAY_RATE)

        item = MemoryItem(
            source_id=ns.id,
            source_type="narrative",
            content=f"[narrative] {ns.statement[:80]}",
            relevance=relevance,
            emotional_intensity=ns.strength,
            recency=recency,
        )
        item.compute_composite()
        candidates.append(item)

    # Seleccionar top-K
    candidates.sort(key=lambda x: x.composite_score, reverse=True)
    capacity = state.working_memory.capacity
    selected = candidates[:capacity]

    return WorkingMemoryState(
        items=selected,
        capacity=capacity,
        last_updated_turn=current_turn,
    )


def _compute_episode_relevance(
    episode: Episode,
    current_keywords: list[str],
    stimulus_embedding: list[float] | None = None,
) -> float:
    """Computa relevancia de un episodio con el estimulo actual."""
    # Intentar cosine similarity si hay embeddings
    if stimulus_embedding and episode.embedding:
        sim = _cosine_similarity(stimulus_embedding, episode.embedding)
        if sim > 0.5:
            return _clamp(sim, 0.0, 1.0)

    # Fallback a keyword similarity
    return _clamp(_keyword_similarity(current_keywords, episode.keywords), 0.0, 1.0)


def _compute_narrative_relevance(
    narrative: NarrativeStatement,
    current_keywords: list[str],
) -> float:
    """Computa relevancia de una narrativa con el estimulo actual."""
    narrative_keywords = _extract_keywords(narrative.statement)
    return _clamp(_keyword_similarity(current_keywords, narrative_keywords), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Level 3: Episodic Memory
# ---------------------------------------------------------------------------

def encode_episode(
    stimulus: str,
    emotional_state: EmotionalState,
    response_summary: str,
    turn_number: int,
    session_id: str,
    workspace_contents: list[str] | None = None,
    preconscious_count: int = 0,
    embedding: list[float] | None = None,
) -> Episode | None:
    """Codifica un nuevo episodio si la intensidad supera el umbral.

    Returns None si la intensidad es insuficiente.
    """
    if emotional_state.intensity < EPISODIC_INTENSITY_THRESHOLD:
        return None

    return Episode(
        stimulus=stimulus[:500],
        response_summary=response_summary[:200],
        primary_emotion=emotional_state.primary_emotion.value,
        valence=emotional_state.valence,
        arousal=emotional_state.arousal,
        intensity=emotional_state.intensity,
        dominance=emotional_state.dominance,
        certainty=emotional_state.certainty,
        workspace_contents=workspace_contents or [],
        preconscious_count=preconscious_count,
        significance=_classify_significance(emotional_state.intensity),
        turn_number=turn_number,
        session_id=session_id,
        keywords=_extract_keywords(stimulus),
        embedding=embedding or [],
    )


def store_episode(
    episodic: EpisodicStore,
    episode: Episode,
) -> EpisodicStore:
    """Almacena un episodio en el store, con eviction si excede capacidad.

    Eviction: remueve el episodio de menor intensidad que ya fue consolidado.
    Si no hay consolidados, remueve el de menor intensidad absoluto.
    """
    episodes = list(episodic.episodes)
    episodes.append(episode)
    total = episodic.total_encoded + 1

    if len(episodes) > EPISODIC_MAX_PER_SESSION:
        # Preferir evictar episodios ya consolidados
        consolidated = [(i, ep) for i, ep in enumerate(episodes) if ep.consolidated]
        if consolidated:
            weakest_idx = min(consolidated, key=lambda x: x[1].intensity)[0]
        else:
            weakest_idx = min(range(len(episodes)), key=lambda i: episodes[i].intensity)
        episodes.pop(weakest_idx)

    return EpisodicStore(episodes=episodes, total_encoded=total)


def retrieve_episodes_by_emotion(
    episodic: EpisodicStore,
    emotion: str,
    limit: int = 10,
) -> list[Episode]:
    """Recupera episodios por emocion primaria, ordenados por intensidad."""
    matches = [ep for ep in episodic.episodes if ep.primary_emotion == emotion]
    matches.sort(key=lambda e: e.intensity, reverse=True)
    return matches[:limit]


def retrieve_episodes_by_similarity(
    episodic: EpisodicStore,
    target: Episode,
    threshold: float = EMOTIONAL_SIMILARITY_THRESHOLD,
) -> list[Episode]:
    """Recupera episodios emocionalmente similares a un target."""
    similar = []
    for ep in episodic.episodes:
        if ep.id == target.id:
            continue
        distance = _emotional_distance(ep, target)
        if distance < threshold:
            similar.append(ep)
    return similar


# ---------------------------------------------------------------------------
# Level 4: Narrative Memory
# ---------------------------------------------------------------------------

def attempt_narrative_formation(
    episodic: EpisodicStore,
    narrative: NarrativeStore,
    current_turn: int,
    session_id: str,
) -> NarrativeStore:
    """Intenta formar nuevas narrativas a partir de clusters episodicos.

    Agrupa episodios por emocion primaria. Si un grupo tiene 5+ episodios
    y no existe narrativa para esa emocion, forma una nueva.
    Si ya existe, refuerza la existente.
    """
    # Agrupar episodios por emocion
    by_emotion: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodic.episodes:
        by_emotion[ep.primary_emotion].append(ep)

    statements = list(narrative.statements)

    for emotion, episodes in by_emotion.items():
        if len(episodes) < NARRATIVE_MIN_EPISODES:
            continue

        # Verificar si ya hay narrativa para esta emocion
        existing = [s for s in statements if s.primary_emotion == emotion]

        if existing:
            # Reforzar la existente
            for s in existing:
                s.strength = _clamp(s.strength + NARRATIVE_REINFORCEMENT_STRENGTH, 0.0, 1.0)
                s.episode_count = len(episodes)
                s.last_reinforced_turn = current_turn
                # Actualizar source_episode_ids con los mas recientes
                s.source_episode_ids = [ep.id for ep in episodes[-10:]]
        else:
            # Formar nueva narrativa
            avg_valence = sum(ep.valence for ep in episodes) / len(episodes)
            narrative_type = _classify_narrative_type(emotion, avg_valence, episodes)
            statement_text = _generate_narrative_text(emotion, avg_valence, episodes)

            new_statement = NarrativeStatement(
                narrative_type=narrative_type,
                statement=statement_text,
                primary_emotion=emotion,
                valence=_clamp(avg_valence, -1.0, 1.0),
                source_episode_ids=[ep.id for ep in episodes[-10:]],
                episode_count=len(episodes),
                strength=0.3,
                formed_session=session_id,
                formed_turn=current_turn,
                last_reinforced_turn=current_turn,
            )
            statements.append(new_statement)

    # Limit total narratives
    if len(statements) > NARRATIVE_MAX_STATEMENTS:
        statements.sort(key=lambda s: s.strength, reverse=True)
        statements = statements[:NARRATIVE_MAX_STATEMENTS]

    return NarrativeStore(statements=statements)


def _classify_narrative_type(
    emotion: str,
    avg_valence: float,
    episodes: list[Episode],
) -> NarrativeType:
    """Clasifica el tipo de narrativa basado en patron emocional."""
    # Si la valence varia mucho, es growth
    valences = [ep.valence for ep in episodes]
    valence_range = max(valences) - min(valences) if valences else 0
    if valence_range > 1.0:
        return NarrativeType.GROWTH

    # Vulnerabilidad: emociones negativas intensas
    if avg_valence < -0.3:
        avg_intensity = sum(ep.intensity for ep in episodes) / len(episodes)
        if avg_intensity > 0.7:
            return NarrativeType.VULNERABILITY
        return NarrativeType.REACTIVE

    # Patron: emociones consistentes
    if valence_range < 0.5:
        return NarrativeType.PATTERN

    return NarrativeType.REACTIVE


def _generate_narrative_text(
    emotion: str,
    avg_valence: float,
    episodes: list[Episode],
) -> str:
    """Genera texto descriptivo para una narrativa.

    Texto simple basado en patron — no usa LLM (deterministic).
    """
    count = len(episodes)
    if avg_valence > 0.3:
        return f"I tend to feel {emotion} in positive interactions ({count} episodes)"
    elif avg_valence < -0.3:
        return f"I tend to feel {emotion} in challenging situations ({count} episodes)"
    else:
        return f"I regularly experience {emotion} across diverse situations ({count} episodes)"


# ---------------------------------------------------------------------------
# Orchestration: Process Turn
# ---------------------------------------------------------------------------

def process_autobiographical_turn(
    state: AutobiographicalState,
    stimulus: str,
    emotional_state: EmotionalState,
    response_summary: str,
    turn_number: int,
    session_id: str,
    prediction_error: float = 0.0,
    workspace_contents: list[str] | None = None,
    preconscious_count: int = 0,
    stimulus_embedding: list[float] | None = None,
) -> AutobiographicalState:
    """Procesa un turno completo de memoria autobiografica.

    Si state.enabled=False, retorna state sin cambios.

    Pasos:
      1. Captura sensory buffer
      2. Intenta codificar episodio (si intensity > 0.5)
      3. Intenta formacion de narrativa (si hay clusters)
      4. Actualiza working memory
      5. Incrementa turns_processed
    """
    if not state.enabled:
        return state

    # Step 1: Sensory buffer
    sensory = capture_sensory(stimulus, emotional_state, prediction_error, turn_number)

    # Step 2: Encode episode
    episode = encode_episode(
        stimulus=stimulus,
        emotional_state=emotional_state,
        response_summary=response_summary,
        turn_number=turn_number,
        session_id=session_id,
        workspace_contents=workspace_contents,
        preconscious_count=preconscious_count,
        embedding=stimulus_embedding,
    )

    episodic = state.episodic
    if episode is not None:
        episodic = store_episode(episodic, episode)

    # Step 3: Narrative formation
    narrative = attempt_narrative_formation(
        episodic=episodic,
        narrative=state.narrative,
        current_turn=turn_number,
        session_id=session_id,
    )

    # Step 4: Working memory
    working_memory = update_working_memory(
        state=AutobiographicalState(
            enabled=True,
            episodic=episodic,
            narrative=narrative,
            working_memory=state.working_memory,
        ),
        current_stimulus=stimulus,
        current_turn=turn_number,
        stimulus_embedding=stimulus_embedding,
    )

    # Step 5: Return updated state
    return AutobiographicalState(
        enabled=True,
        sensory_buffer=sensory,
        working_memory=working_memory,
        episodic=episodic,
        narrative=narrative,
        session_id=session_id,
        total_turns_processed=state.total_turns_processed + 1,
        last_dream_report=state.last_dream_report,
        baseline_adjustment=state.baseline_adjustment,
    )


# ---------------------------------------------------------------------------
# Prompt Generation (for behavior modifier)
# ---------------------------------------------------------------------------

def get_autobiographical_prompt(state: AutobiographicalState) -> str:
    """Genera texto para el behavior modifier basado en memorias activas.

    Solo incluye lo que esta en working memory (consciente para el agente).
    """
    if not state.enabled:
        return ""

    parts: list[str] = []

    # Working memory contents
    if state.working_memory.items:
        parts.append("AUTOBIOGRAPHICAL MEMORY (active in working memory):")
        for item in state.working_memory.items:
            parts.append(f"  - {item.content} (relevance: {item.relevance:.2f})")

    # Active narratives (strongest)
    strong_narratives = state.narrative.get_strongest(k=3)
    if strong_narratives:
        parts.append("SELF-KNOWLEDGE (narrative memory):")
        for ns in strong_narratives:
            parts.append(f"  - {ns.statement} (strength: {ns.strength:.2f})")

    # Dream report from previous session
    if state.last_dream_report:
        parts.append(f"DREAM ECHO (from last session): {state.last_dream_report[:200]}")

    # Episode count context
    ep_count = state.episodic.count()
    if ep_count > 0:
        parts.append(f"EXPERIENTIAL DEPTH: {ep_count} significant episodes stored")

    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

from pathos.models.schemas import AutobiographicalDetails  # noqa: E402


def get_autobiographical_details(state: AutobiographicalState) -> AutobiographicalDetails:
    """Extrae detalles para el research endpoint."""
    if not state.enabled:
        return AutobiographicalDetails(enabled=False)

    strong = state.narrative.get_strongest(k=3)
    return AutobiographicalDetails(
        enabled=True,
        sensory_emotion=state.sensory_buffer.primary_emotion,
        sensory_intensity=state.sensory_buffer.intensity,
        sensory_prediction_error=state.sensory_buffer.prediction_error,
        working_memory_count=len(state.working_memory.items),
        working_memory_items=[item.content for item in state.working_memory.items],
        episodic_count=state.episodic.count(),
        episodic_total_encoded=state.episodic.total_encoded,
        episodic_high_intensity_count=len(state.episodic.get_high_intensity()),
        narrative_count=state.narrative.count(),
        narrative_strongest=[ns.statement for ns in strong],
        total_turns_processed=state.total_turns_processed,
        has_dream_report=bool(state.last_dream_report),
    )
