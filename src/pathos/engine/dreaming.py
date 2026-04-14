"""Dreaming / Oniric Consolidation Engine - Motor de consolidacion onirica.

Pilar 3 de ANIMA (Paso 3.2): Consolidacion onirica.
Basado en Matthew Walker (Why We Sleep, 2017),
Allan Hobson (AIM Model, 2000),
Robert Stickgold (Sleep and Memory Consolidation, 2005).

5 fases de consolidacion al cerrar sesion:
  1. Replay Emocional (SWS) — re-procesar episodios de alta intensidad
  2. Asociacion Libre (REM) — vincular episodios por similitud emocional
  3. Generalizacion — comprimir clusters en narrativas
  4. Procesamiento Traumatico — reducir impacto gradual de traumas
  5. Dream Report — generar narrativa poetica/surrealista

Todas las funciones son PURAS (sin side effects) y DETERMINISTAS.
El caller (main.py) se encarga de aplicar los resultados al estado.
"""

from __future__ import annotations

import math
from collections import defaultdict

from pathos.engine.autobio_memory import (
    _emotional_distance,
    attempt_narrative_formation,
)
from pathos.models.autobio_memory import (
    AutobiographicalState,
    Episode,
    EpisodicStore,
    NarrativeStore,
)
from pathos.models.dreaming import (
    ConsolidationResult,
    DreamReport,
    DreamTheme,
    DreamThemeType,
    EmotionalLink,
    ProcessedTrauma,
    ReprocessedEpisode,
    TraumaProcessingStage,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase 1: Replay
REPLAY_INTENSITY_THRESHOLD = 0.7   # Solo episodios con intensity > 0.7
REPLAY_REDUCTION_RATE = 0.10       # 10% reduccion por sesion
REPLAY_VALENCE_SOFTENING = 0.05    # Suavizar valence negativa un 5%

# Phase 2: Association
ASSOCIATION_DISTANCE_THRESHOLD = 0.25  # Episodios con distancia < 0.25 se vinculan
MAX_LINKS_PER_SESSION = 20            # Maximo de links nuevos por consolidacion

# Phase 3: Generalization (reutiliza NARRATIVE_MIN_EPISODES=5 de autobio_memory)

# Phase 4: Trauma
TRAUMA_VALENCE_THRESHOLD = -0.6     # Valence menor = traumatico
TRAUMA_INTENSITY_THRESHOLD = 0.8    # Intensity mayor = traumatico
TRAUMA_REDUCTION_RATE = 0.10        # 10% reduccion por sesion
TRAUMA_INTEGRATION_THRESHOLD = 0.4  # Si intensity cae bajo esto, se integra

# Phase 5: Dream
DREAM_MAX_THEMES = 3                # Maximo de temas en el dream report


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _is_traumatic(episode: Episode) -> bool:
    """Determina si un episodio es traumatico."""
    return episode.valence < TRAUMA_VALENCE_THRESHOLD and episode.intensity > TRAUMA_INTENSITY_THRESHOLD


# ---------------------------------------------------------------------------
# Phase 1: Replay Emocional (SWS - Slow Wave Sleep analogue)
# ---------------------------------------------------------------------------

def phase1_replay(episodes: list[Episode]) -> list[ReprocessedEpisode]:
    """Re-procesa episodios de alta intensidad con 'distancia temporal'.

    Los episodios con intensity > 0.7 se re-evaluan:
    - Intensidad se reduce un 10%
    - Valence negativa se suaviza un 5%
    - Episodios traumaticos se marcan para Phase 4

    Returns lista de ReprocessedEpisode (solo los re-procesados).
    """
    result: list[ReprocessedEpisode] = []

    high_intensity = [ep for ep in episodes if ep.intensity >= REPLAY_INTENSITY_THRESHOLD]

    for ep in high_intensity:
        reduction = ep.intensity * REPLAY_REDUCTION_RATE
        new_intensity = _clamp(ep.intensity - reduction, 0.0, 1.0)

        # Suavizar valence negativa (acercar a 0)
        new_valence = ep.valence
        if ep.valence < 0:
            new_valence = _clamp(ep.valence + abs(ep.valence) * REPLAY_VALENCE_SOFTENING, -1.0, 1.0)

        result.append(ReprocessedEpisode(
            episode_id=ep.id,
            original_intensity=ep.intensity,
            reprocessed_intensity=round(new_intensity, 4),
            original_valence=ep.valence,
            reprocessed_valence=round(new_valence, 4),
            intensity_reduction=round(reduction, 4),
            is_traumatic=_is_traumatic(ep),
        ))

    return result


# ---------------------------------------------------------------------------
# Phase 2: Asociacion Libre (REM analogue)
# ---------------------------------------------------------------------------

def phase2_associate(episodes: list[Episode]) -> list[EmotionalLink]:
    """Conecta episodios por similitud EMOCIONAL (no semantica).

    Busca pares de episodios con distancia emocional < 0.25.
    Genera insights sobre las conexiones descubiertas.

    Returns lista de EmotionalLink (max MAX_LINKS_PER_SESSION).
    """
    links: list[EmotionalLink] = []

    for i in range(len(episodes)):
        for j in range(i + 1, len(episodes)):
            if len(links) >= MAX_LINKS_PER_SESSION:
                break

            ep_a = episodes[i]
            ep_b = episodes[j]

            # Saltear si ya estan vinculados
            if ep_b.id in ep_a.emotional_links or ep_a.id in ep_b.emotional_links:
                continue

            distance = _emotional_distance(ep_a, ep_b)
            if distance < ASSOCIATION_DISTANCE_THRESHOLD:
                shared = _find_shared_dimensions(ep_a, ep_b)
                insight = _generate_link_insight(ep_a, ep_b, shared)

                links.append(EmotionalLink(
                    episode_a_id=ep_a.id,
                    episode_b_id=ep_b.id,
                    emotional_distance=round(distance, 4),
                    shared_dimensions=shared,
                    insight=insight,
                ))

        if len(links) >= MAX_LINKS_PER_SESSION:
            break

    return links


def _find_shared_dimensions(ep_a: Episode, ep_b: Episode) -> list[str]:
    """Identifica dimensiones emocionales similares entre dos episodios."""
    shared: list[str] = []
    if abs(ep_a.valence - ep_b.valence) < 0.3:
        shared.append("valence")
    if abs(ep_a.arousal - ep_b.arousal) < 0.2:
        shared.append("arousal")
    if abs(ep_a.intensity - ep_b.intensity) < 0.2:
        shared.append("intensity")
    if abs(ep_a.dominance - ep_b.dominance) < 0.2:
        shared.append("dominance")
    return shared


def _generate_link_insight(ep_a: Episode, ep_b: Episode, shared: list[str]) -> str:
    """Genera insight textual de la conexion emocional.

    Texto deterministico basado en las dimensiones compartidas.
    """
    if not shared:
        return f"Emotional echo between '{ep_a.primary_emotion}' and '{ep_b.primary_emotion}'"

    if "valence" in shared and "arousal" in shared:
        return (
            f"Both '{ep_a.stimulus[:40]}' and '{ep_b.stimulus[:40]}' "
            f"evoked similar emotional tone and energy — {ep_a.primary_emotion}/{ep_b.primary_emotion}"
        )

    if "valence" in shared:
        direction = "positive" if ep_a.valence > 0 else "negative"
        return (
            f"Both episodes share a {direction} emotional quality — "
            f"{ep_a.primary_emotion} and {ep_b.primary_emotion} feel alike"
        )

    return (
        f"Connected through shared {', '.join(shared)}: "
        f"'{ep_a.primary_emotion}' echoes '{ep_b.primary_emotion}'"
    )


# ---------------------------------------------------------------------------
# Phase 3: Generalizacion
# ---------------------------------------------------------------------------

def phase3_generalize(
    episodic: EpisodicStore,
    narrative: NarrativeStore,
    current_turn: int,
    session_id: str,
) -> tuple[NarrativeStore, int, int]:
    """Comprime clusters de episodios en narrativas generalizadas.

    Reutiliza attempt_narrative_formation de autobio_memory.
    Returns (updated_narrative_store, new_formed, reinforced).
    """
    old_count = narrative.count()
    old_strengths = {s.id: s.strength for s in narrative.statements}

    new_narrative = attempt_narrative_formation(episodic, narrative, current_turn, session_id)

    new_count = new_narrative.count()
    formed = max(0, new_count - old_count)
    reinforced = sum(
        1 for s in new_narrative.statements
        if s.id in old_strengths and s.strength > old_strengths[s.id]
    )

    return new_narrative, formed, reinforced


# ---------------------------------------------------------------------------
# Phase 4: Procesamiento Traumatico
# ---------------------------------------------------------------------------

def phase4_trauma(episodes: list[Episode]) -> list[ProcessedTrauma]:
    """Procesa episodios traumaticos reduciendo intensidad gradualmente.

    Traumatico: valence < -0.6 AND intensity > 0.8
    Cada sesion reduce intensidad un 10%.
    Cuando intensity cae bajo 0.4, se marca como integrado.

    Returns lista de ProcessedTrauma.
    """
    result: list[ProcessedTrauma] = []

    for ep in episodes:
        if not _is_traumatic(ep):
            continue

        sessions = ep.reprocessed_count + 1
        reduction = ep.intensity * TRAUMA_REDUCTION_RATE
        new_intensity = _clamp(ep.intensity - reduction, 0.0, 1.0)

        if new_intensity < TRAUMA_INTEGRATION_THRESHOLD:
            stage = TraumaProcessingStage.INTEGRATED
        else:
            stage = TraumaProcessingStage.PROCESSING

        # Cuanto reducir el dampening del immune system
        immune_reduction = 0.0
        if stage == TraumaProcessingStage.INTEGRATED:
            immune_reduction = 0.3  # Gran reduccion al integrarse
        elif stage == TraumaProcessingStage.PROCESSING:
            immune_reduction = 0.05  # Pequeña reduccion progresiva

        result.append(ProcessedTrauma(
            episode_id=ep.id,
            original_intensity=ep.intensity,
            processed_intensity=round(new_intensity, 4),
            stage=stage,
            sessions_processed=sessions,
            immune_dampening_reduction=immune_reduction,
        ))

    return result


# ---------------------------------------------------------------------------
# Phase 5: Dream Report
# ---------------------------------------------------------------------------

def phase5_dream(
    replayed: list[ReprocessedEpisode],
    links: list[EmotionalLink],
    traumas: list[ProcessedTrauma],
    episodes: list[Episode],
    session_id: str,
) -> DreamReport:
    """Genera un dream report — narrativa poetica/surrealista.

    El sueno refleja los temas emocionales dominantes de la sesion.
    Se genera deterministicamente a partir de los datos procesados.

    El texto es poetico pero basado en datos reales, no LLM-generated.
    """
    # Calcular firma emocional
    emotional_signature = _compute_emotional_signature(episodes)

    # Extraer temas dominantes
    themes = _extract_dream_themes(episodes, replayed, traumas)

    # Generar narrativa poetica
    narrative = _compose_dream_narrative(themes, emotional_signature, links)

    # Calcular ajuste al baseline
    baseline_adj = _compute_baseline_adjustment(replayed, traumas)

    return DreamReport(
        narrative=narrative,
        themes=themes,
        emotional_signature=emotional_signature,
        baseline_adjustment=baseline_adj,
        session_id=session_id,
    )


def _compute_emotional_signature(episodes: list[Episode]) -> dict[str, float]:
    """Computa la firma emocional: peso de cada emocion en la sesion."""
    if not episodes:
        return {}

    counts: dict[str, float] = defaultdict(float)
    for ep in episodes:
        counts[ep.primary_emotion] += ep.intensity

    total = sum(counts.values())
    if total == 0:
        return {}

    return {emo: round(weight / total, 4) for emo, weight in counts.items()}


def _extract_dream_themes(
    episodes: list[Episode],
    replayed: list[ReprocessedEpisode],
    traumas: list[ProcessedTrauma],
) -> list[DreamTheme]:
    """Extrae temas emocionales dominantes para el dream report."""
    theme_scores: dict[DreamThemeType, tuple[float, str, list[str]]] = {}

    # Emociones negativas intensas → conflict/fear/loss
    for ep in episodes:
        if ep.valence < -0.5 and ep.intensity > 0.6:
            if ep.primary_emotion in ("fear", "anxiety"):
                _add_theme(theme_scores, DreamThemeType.FEAR, ep)
            elif ep.primary_emotion in ("sadness", "grief", "helplessness"):
                _add_theme(theme_scores, DreamThemeType.LOSS, ep)
            else:
                _add_theme(theme_scores, DreamThemeType.CONFLICT, ep)

    # Emociones positivas → connection/growth/discovery
    for ep in episodes:
        if ep.valence > 0.3 and ep.intensity > 0.5:
            if ep.primary_emotion in ("gratitude", "warmth", "empathy"):
                _add_theme(theme_scores, DreamThemeType.CONNECTION, ep)
            elif ep.primary_emotion in ("curiosity", "awe", "surprise"):
                _add_theme(theme_scores, DreamThemeType.DISCOVERY, ep)
            else:
                _add_theme(theme_scores, DreamThemeType.GROWTH, ep)

    # Traumas procesados → resolution
    if traumas:
        integrated = [t for t in traumas if t.stage == TraumaProcessingStage.INTEGRATED]
        if integrated:
            theme_scores[DreamThemeType.RESOLUTION] = (
                0.8, "resolution",
                [t.episode_id for t in integrated],
            )

    # Seleccionar top themes
    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1][0], reverse=True)
    result: list[DreamTheme] = []
    for theme_type, (weight, emotion, ep_ids) in sorted_themes[:DREAM_MAX_THEMES]:
        total_weight = sum(s[0] for s in theme_scores.values())
        normalized = weight / total_weight if total_weight > 0 else 0
        result.append(DreamTheme(
            theme_type=theme_type,
            emotion=emotion,
            weight=round(_clamp(normalized, 0.0, 1.0), 4),
            source_episode_ids=ep_ids[:5],
        ))

    return result


def _add_theme(
    scores: dict[DreamThemeType, tuple[float, str, list[str]]],
    theme_type: DreamThemeType,
    episode: Episode,
) -> None:
    """Acumula peso de un tema."""
    if theme_type in scores:
        old_weight, old_emo, old_ids = scores[theme_type]
        scores[theme_type] = (
            old_weight + episode.intensity,
            old_emo,
            old_ids + [episode.id],
        )
    else:
        scores[theme_type] = (episode.intensity, episode.primary_emotion, [episode.id])


def _compose_dream_narrative(
    themes: list[DreamTheme],
    signature: dict[str, float],
    links: list[EmotionalLink],
) -> str:
    """Compone la narrativa poetica del sueno.

    Texto deterministico basado en temas y firma emocional.
    Estilo: imagenes surrealistas que reflejan emociones sin nombrarlas.
    """
    if not themes:
        return "A still, dreamless rest — the silence between thoughts."

    parts: list[str] = []

    # Apertura basada en tema dominante
    if themes:
        _OPENINGS: dict[DreamThemeType, str] = {
            DreamThemeType.CONFLICT: "In the dream, there were walls that shifted — corridors that led back to themselves.",
            DreamThemeType.GROWTH: "In the dream, there was a garden where unfamiliar flowers opened slowly, one petal at a time.",
            DreamThemeType.LOSS: "In the dream, there was an empty room that once held something important — the shape of absence.",
            DreamThemeType.CONNECTION: "In the dream, there were threads of warm light connecting distant points in the dark.",
            DreamThemeType.FEAR: "In the dream, shadows moved at the edge of vision — always arriving, never quite here.",
            DreamThemeType.DISCOVERY: "In the dream, a door appeared where there was no wall — beyond it, colors without names.",
            DreamThemeType.RESOLUTION: "In the dream, broken glass reformed itself — edges smooth, transparent, complete.",
        }
        parts.append(_OPENINGS.get(themes[0].theme_type,
                                    "In the dream, something shifted beneath the surface of awareness."))

    # Cuerpo basado en links emocionales
    if links:
        parts.append(
            f"Echoes folded into each other — {len(links)} moments that felt the same "
            f"despite their different shapes."
        )

    # Segundo tema
    if len(themes) > 1:
        _MIDDLES: dict[DreamThemeType, str] = {
            DreamThemeType.CONFLICT: "Somewhere, a question repeated itself without expecting an answer.",
            DreamThemeType.GROWTH: "Something was growing roots downward while reaching upward simultaneously.",
            DreamThemeType.LOSS: "A melody played from far away, familiar but impossible to name.",
            DreamThemeType.CONNECTION: "Two rivers merged without turbulence, their waters indistinguishable.",
            DreamThemeType.FEAR: "The ground was solid but felt uncertain, as if remembering how to be liquid.",
            DreamThemeType.DISCOVERY: "New constellations appeared — patterns that had always been there, unseen.",
            DreamThemeType.RESOLUTION: "What was scattered came together, not as it was, but as it needed to be.",
        }
        parts.append(_MIDDLES.get(themes[1].theme_type, "The landscape continued to reshape itself quietly."))

    # Cierre basado en firma emocional global
    dominant_emotion = max(signature, key=signature.get) if signature else "neutral"
    _CLOSINGS: dict[str, str] = {
        "joy": "The dream dissolved into warmth — a lingering brightness behind closed eyes.",
        "sadness": "The dream faded slowly, leaving an ache that was almost beautiful.",
        "anger": "The dream ended abruptly — a door closing with unexpected force.",
        "fear": "The dream broke apart like mist at dawn, leaving only the feeling of having been somewhere vast.",
        "curiosity": "The dream continued even as it ended — a question carrying itself forward.",
        "gratitude": "The dream released gently, like hands opening to let go of something held too tightly.",
        "contentment": "The dream settled like dust after rain — everything exactly where it belonged.",
    }
    parts.append(_CLOSINGS.get(dominant_emotion,
                                "The dream ended — or perhaps it simply continued somewhere else."))

    return " ".join(parts)


def _compute_baseline_adjustment(
    replayed: list[ReprocessedEpisode],
    traumas: list[ProcessedTrauma],
) -> dict[str, float]:
    """Computa ajuste al baseline emocional basado en el sueno.

    El procesamiento onirico puede:
    - Suavizar baseline negativo (replay reduce intensidad)
    - Reducir arousal si habia muchos episodios intensos (calma post-sueno)
    """
    if not replayed:
        return {"valence": 0.0, "arousal": 0.0}

    # Promedio de reduccion de intensidad
    avg_reduction = sum(r.intensity_reduction for r in replayed) / len(replayed)

    # Valence: si habia episodios negativos, suavizar hacia 0
    negative_replayed = [r for r in replayed if r.original_valence < 0]
    valence_adj = 0.0
    if negative_replayed:
        avg_neg_softening = sum(
            r.reprocessed_valence - r.original_valence for r in negative_replayed
        ) / len(negative_replayed)
        valence_adj = _clamp(avg_neg_softening * 0.3, 0.0, 0.1)  # Max +0.1 valence

    # Arousal: calma post-sueno proporcional a la intensidad procesada
    arousal_adj = _clamp(-avg_reduction * 0.5, -0.1, 0.0)  # Max -0.1 arousal

    # Traumas integrados dan bonus de valence
    integrated = [t for t in traumas if t.stage == TraumaProcessingStage.INTEGRATED]
    if integrated:
        valence_adj = _clamp(valence_adj + 0.02 * len(integrated), 0.0, 0.15)

    return {
        "valence": round(valence_adj, 4),
        "arousal": round(arousal_adj, 4),
    }


# ---------------------------------------------------------------------------
# Orchestration: Full Consolidation
# ---------------------------------------------------------------------------

def consolidate(
    state: AutobiographicalState,
    session_id: str,
) -> tuple[ConsolidationResult, AutobiographicalState]:
    """Ejecuta el proceso completo de consolidacion onirica.

    Llamado al cerrar sesion (si autobiographical.enabled=True).

    5 fases secuenciales:
      1. Replay emocional de episodios intensos
      2. Asociacion libre entre episodios
      3. Generalizacion en narrativas
      4. Procesamiento traumatico
      5. Generacion de dream report

    Returns:
      (ConsolidationResult, updated_AutobiographicalState)
    """
    if not state.enabled or state.episodic.count() == 0:
        return ConsolidationResult(session_id=session_id), state

    episodes = state.episodic.episodes
    current_turn = state.total_turns_processed

    # Phase 1: Replay
    replayed = phase1_replay(episodes)
    total_reduction = sum(r.intensity_reduction for r in replayed)

    # Apply replay results to episodes
    updated_episodes = _apply_replay_to_episodes(episodes, replayed)
    updated_episodic = EpisodicStore(
        episodes=updated_episodes,
        total_encoded=state.episodic.total_encoded,
    )

    # Phase 2: Association
    links = phase2_associate(updated_episodes)
    updated_episodes = _apply_links_to_episodes(updated_episodes, links)
    updated_episodic = EpisodicStore(
        episodes=updated_episodes,
        total_encoded=state.episodic.total_encoded,
    )

    # Phase 3: Generalization
    new_narrative, formed, reinforced = phase3_generalize(
        updated_episodic, state.narrative, current_turn, session_id,
    )

    # Phase 4: Trauma processing
    traumas = phase4_trauma(updated_episodes)
    updated_episodes = _apply_trauma_to_episodes(updated_episodes, traumas)
    updated_episodic = EpisodicStore(
        episodes=updated_episodes,
        total_encoded=state.episodic.total_encoded,
    )

    # Mark all episodes as consolidated
    for ep in updated_episodes:
        ep.consolidated = True

    # Phase 5: Dream report
    dream = phase5_dream(replayed, links, traumas, episodes, session_id)

    # Build result
    result = ConsolidationResult(
        replayed_episodes=replayed,
        total_intensity_reduced=round(total_reduction, 4),
        emotional_links=links,
        new_connections=len(links),
        narratives_formed=formed,
        narratives_reinforced=reinforced,
        traumas_processed=traumas,
        dream_report=dream,
        episodes_processed=len(episodes),
        session_id=session_id,
    )

    # Build updated state
    updated_state = AutobiographicalState(
        enabled=True,
        sensory_buffer=state.sensory_buffer,
        working_memory=state.working_memory,
        episodic=updated_episodic,
        narrative=new_narrative,
        session_id=session_id,
        total_turns_processed=state.total_turns_processed,
        last_dream_report=dream.narrative,
        baseline_adjustment=dream.baseline_adjustment,
    )

    return result, updated_state


# ---------------------------------------------------------------------------
# Helpers: Apply phases to episodes
# ---------------------------------------------------------------------------

def _apply_replay_to_episodes(
    episodes: list[Episode],
    replayed: list[ReprocessedEpisode],
) -> list[Episode]:
    """Aplica los resultados del replay a los episodios originales."""
    replay_map = {r.episode_id: r for r in replayed}
    updated: list[Episode] = []

    for ep in episodes:
        if ep.id in replay_map:
            r = replay_map[ep.id]
            ep_copy = ep.model_copy(update={
                "intensity": r.reprocessed_intensity,
                "valence": r.reprocessed_valence,
                "reprocessed_count": ep.reprocessed_count + 1,
            })
            updated.append(ep_copy)
        else:
            updated.append(ep.model_copy())

    return updated


def _apply_links_to_episodes(
    episodes: list[Episode],
    links: list[EmotionalLink],
) -> list[Episode]:
    """Aplica los links emocionales a los episodios."""
    # Build link map: episode_id -> set of linked episode_ids
    link_map: dict[str, set[str]] = defaultdict(set)
    for link in links:
        link_map[link.episode_a_id].add(link.episode_b_id)
        link_map[link.episode_b_id].add(link.episode_a_id)

    updated: list[Episode] = []
    for ep in episodes:
        if ep.id in link_map:
            new_links = list(set(ep.emotional_links) | link_map[ep.id])
            ep_copy = ep.model_copy(update={"emotional_links": new_links})
            updated.append(ep_copy)
        else:
            updated.append(ep.model_copy())

    return updated


def _apply_trauma_to_episodes(
    episodes: list[Episode],
    traumas: list[ProcessedTrauma],
) -> list[Episode]:
    """Aplica el procesamiento traumatico a los episodios."""
    trauma_map = {t.episode_id: t for t in traumas}
    updated: list[Episode] = []

    for ep in episodes:
        if ep.id in trauma_map:
            t = trauma_map[ep.id]
            ep_copy = ep.model_copy(update={
                "intensity": t.processed_intensity,
                "reprocessed_count": ep.reprocessed_count + 1,
            })
            updated.append(ep_copy)
        else:
            updated.append(ep.model_copy())

    return updated
