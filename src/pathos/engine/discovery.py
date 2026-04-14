"""Emotional Discovery Engine — Emergent emotion detection and naming.

Barrett's Theory of Constructed Emotion: emotional categories are built
through experience, not discovered as natural kinds. This module detects
emotional states that don't match any known prototype, clusters them,
and lets the agent name them — building a unique emotional vocabulary.

Steps:
1. detect_novel() — after emotion generation, check if state is far from all prototypes
2. cluster_novel_states() — group novel states by similarity (agglomerative)
3. form_emotion() — when cluster reaches 3+ instances, create DiscoveredEmotion
4. name_emotion_fallback() — deterministic naming (no LLM) from dimensional position
5. integrate into vocabulary — discovered emotions expand the known set
"""

import math

from pathos.models.discovery import (
    CLUSTER_DISTANCE_THRESHOLD,
    MAX_CONTEXTS_PER_EMOTION,
    MAX_DISCOVERED_EMOTIONS,
    MAX_NOVEL_BUFFER,
    MIN_CLUSTER_SIZE,
    NOVELTY_THRESHOLD,
    BodySignature,
    DiscoveredEmotion,
    DiscoveryState,
    EmotionalVector,
    NovelEmotionalState,
)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# --- Known emotion prototypes (same as generator.py) ---

# (valence, arousal, dominance, certainty)
KNOWN_PROTOTYPES: dict[str, tuple[float, float, float, float]] = {
    "joy":            ( 0.75,  0.65,  0.70,  0.70),
    "excitement":     ( 0.70,  0.90,  0.55,  0.25),
    "gratitude":      ( 0.70,  0.40,  0.30,  0.70),
    "hope":           ( 0.50,  0.55,  0.35,  0.25),
    "contentment":    ( 0.55,  0.20,  0.60,  0.80),
    "relief":         ( 0.50,  0.25,  0.45,  0.65),
    "anger":          (-0.75,  0.80,  0.70,  0.60),
    "frustration":    (-0.50,  0.70,  0.40,  0.35),
    "fear":           (-0.75,  0.85,  0.15,  0.15),
    "anxiety":        (-0.45,  0.65,  0.25,  0.25),
    "sadness":        (-0.70,  0.20,  0.25,  0.60),
    "helplessness":   (-0.75,  0.20,  0.10,  0.15),
    "disappointment": (-0.50,  0.30,  0.40,  0.55),
    "surprise":       ( 0.05,  0.85,  0.40,  0.15),
    "alertness":      (-0.05,  0.70,  0.50,  0.35),
    "contemplation":  ( 0.15,  0.25,  0.55,  0.55),
    "indifference":   ( 0.00,  0.10,  0.50,  0.50),
    "neutral":        ( 0.00,  0.30,  0.50,  0.50),
}

# Dimension weights (same as generator.py — valence weighs more)
_DIM_WEIGHTS = (2.5, 1.0, 0.6, 0.6)


def _euclidean_distance(
    v1: tuple[float, float, float, float],
    v2: tuple[float, float, float, float],
) -> float:
    """Weighted euclidean distance between two 4D vectors."""
    return math.sqrt(sum(
        w * (a - b) ** 2
        for w, a, b in zip(_DIM_WEIGHTS, v1, v2)
    ))


def _vector_to_tuple(v: EmotionalVector) -> tuple[float, float, float, float]:
    return (v.valence, v.arousal, v.dominance, v.certainty)


# --- Step 1: Novel state detection ---


def detect_novel(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    stimulus: str,
    turn: int,
    body_tension: float = 0.5,
    body_energy: float = 0.5,
    body_openness: float = 0.5,
    body_warmth: float = 0.5,
    discovered_emotions: list[DiscoveredEmotion] | None = None,
    threshold: float = NOVELTY_THRESHOLD,
) -> NovelEmotionalState | None:
    """Detect if current emotional state is novel (far from all known prototypes).

    Compares the current state vector against all known prototypes AND
    previously discovered emotions. If min distance exceeds threshold,
    the state is novel.

    Returns NovelEmotionalState if novel, None otherwise.
    """
    current = (valence, arousal, dominance, certainty)

    # Check against known prototypes
    min_dist = float("inf")
    closest = "neutral"
    for name, proto in KNOWN_PROTOTYPES.items():
        dist = _euclidean_distance(current, proto)
        if dist < min_dist:
            min_dist = dist
            closest = name

    # Also check against discovered emotions
    if discovered_emotions:
        for de in discovered_emotions:
            dist = _euclidean_distance(current, _vector_to_tuple(de.vector))
            if dist < min_dist:
                min_dist = dist
                closest = de.name

    if min_dist < threshold:
        return None  # Maps to a known emotion

    return NovelEmotionalState(
        vector=EmotionalVector(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            dominance=round(dominance, 4),
            certainty=round(certainty, 4),
        ),
        body=BodySignature(
            tension=round(body_tension, 4),
            energy=round(body_energy, 4),
            openness=round(body_openness, 4),
            warmth=round(body_warmth, 4),
        ),
        context=stimulus[:200] if stimulus else "",
        min_distance=round(min_dist, 4),
        closest_known=closest,
        turn=turn,
        intensity=round(_clamp(intensity, 0, 1), 4),
    )


# --- Step 2: Clustering ---


def _novel_distance(a: NovelEmotionalState, b: NovelEmotionalState) -> float:
    """Distance between two novel states."""
    return _euclidean_distance(_vector_to_tuple(a.vector), _vector_to_tuple(b.vector))


def cluster_novel_states(
    novel_history: list[NovelEmotionalState],
    distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> list[list[NovelEmotionalState]]:
    """Simple agglomerative clustering of novel states.

    Uses single-linkage: a state joins a cluster if it's within
    distance_threshold of ANY member.

    Returns list of clusters (each cluster is a list of NovelEmotionalState).
    Only returns clusters with >= min_cluster_size members.
    """
    if len(novel_history) < min_cluster_size:
        return []

    # Start with each state as its own cluster
    clusters: list[list[NovelEmotionalState]] = [[s] for s in novel_history]

    # Merge clusters until no more merges possible
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                # Check if any pair of members is within threshold
                should_merge = False
                for a in clusters[i]:
                    for b in clusters[j]:
                        if _novel_distance(a, b) <= distance_threshold:
                            should_merge = True
                            break
                    if should_merge:
                        break
                if should_merge:
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    # Filter by min size
    return [c for c in clusters if len(c) >= min_cluster_size]


# --- Step 3: Emotion formation ---


def _average_vector(states: list[NovelEmotionalState]) -> EmotionalVector:
    """Compute centroid of a cluster."""
    n = len(states)
    if n == 0:
        return EmotionalVector(valence=0, arousal=0.3, dominance=0.5, certainty=0.5)
    return EmotionalVector(
        valence=round(sum(s.vector.valence for s in states) / n, 4),
        arousal=round(sum(s.vector.arousal for s in states) / n, 4),
        dominance=round(sum(s.vector.dominance for s in states) / n, 4),
        certainty=round(sum(s.vector.certainty for s in states) / n, 4),
    )


def _average_body(states: list[NovelEmotionalState]) -> BodySignature:
    """Compute average body signature of a cluster."""
    n = len(states)
    if n == 0:
        return BodySignature()
    return BodySignature(
        tension=round(sum(s.body.tension for s in states) / n, 4),
        energy=round(sum(s.body.energy for s in states) / n, 4),
        openness=round(sum(s.body.openness for s in states) / n, 4),
        warmth=round(sum(s.body.warmth for s in states) / n, 4),
    )


def form_emotion(cluster: list[NovelEmotionalState]) -> DiscoveredEmotion:
    """Form a DiscoveredEmotion from a cluster of novel states."""
    centroid = _average_vector(cluster)
    body = _average_body(cluster)
    contexts = list(dict.fromkeys(s.context for s in cluster if s.context))[:MAX_CONTEXTS_PER_EMOTION]

    name = name_emotion_fallback(centroid)

    return DiscoveredEmotion(
        name=name,
        description=_generate_description_fallback(centroid, body, contexts),
        vector=centroid,
        body_signature=body,
        contexts=contexts,
        first_experienced_turn=min(s.turn for s in cluster),
        frequency=len(cluster),
        named=True,  # fallback naming counts
        cluster_size=len(cluster),
    )


# --- Step 4: Naming ---


def name_emotion_fallback(vector: EmotionalVector) -> str:
    """Generate a deterministic name from dimensional position.

    Format: poetic compound based on valence+arousal+dominance quadrant.
    This is the fallback when no LLM is available. If LLM is available,
    it should override this with a creative neologism.
    """
    # Valence component
    if vector.valence > 0.3:
        v_part = "lumen" if vector.valence > 0.6 else "glow"
    elif vector.valence < -0.3:
        v_part = "umbra" if vector.valence < -0.6 else "shade"
    else:
        v_part = "liminal"

    # Arousal component
    if vector.arousal > 0.6:
        a_part = "surge" if vector.arousal > 0.8 else "pulse"
    elif vector.arousal < 0.3:
        a_part = "drift" if vector.arousal < 0.15 else "still"
    else:
        a_part = "flow"

    # Dominance modifier
    if vector.dominance > 0.65:
        d_suffix = "forte"
    elif vector.dominance < 0.35:
        d_suffix = "tender"
    else:
        d_suffix = ""

    # Certainty modifier
    if vector.certainty < 0.25:
        c_prefix = "neo"
    elif vector.certainty > 0.75:
        c_prefix = ""
    else:
        c_prefix = ""

    parts = [c_prefix, v_part, a_part, d_suffix]
    name = "-".join(p for p in parts if p)
    return name


def _generate_description_fallback(
    vector: EmotionalVector,
    body: BodySignature,
    contexts: list[str],
) -> str:
    """Generate a deterministic description of the discovered emotion."""
    # Valence description
    if vector.valence > 0.3:
        v_desc = "a positive feeling"
    elif vector.valence < -0.3:
        v_desc = "a negative feeling"
    else:
        v_desc = "an ambiguous feeling"

    # Arousal description
    if vector.arousal > 0.6:
        a_desc = "high activation"
    elif vector.arousal < 0.3:
        a_desc = "quiet, low-energy"
    else:
        a_desc = "moderate activation"

    # Body component
    body_parts: list[str] = []
    if body.tension > 0.7:
        body_parts.append("physical tension")
    if body.warmth > 0.7:
        body_parts.append("warmth")
    elif body.warmth < 0.3:
        body_parts.append("coldness")
    if body.openness < 0.3:
        body_parts.append("withdrawal")

    body_desc = f", accompanied by {', '.join(body_parts)}" if body_parts else ""

    context_hint = ""
    if contexts:
        context_hint = f" Emerges in contexts like: {contexts[0][:60]}"

    return f"{v_desc.capitalize()} with {a_desc}{body_desc}.{context_hint}"


async def name_emotion_with_llm(
    emotion: DiscoveredEmotion,
    llm_generate: object,  # callable(prompt) -> str
) -> tuple[str, str]:
    """Generate a creative neologism using the LLM.

    This is optional — called only when LLM is available.
    Returns (name, description).

    The llm_generate parameter should be an async callable that takes
    a prompt string and returns a response string.
    """
    prompt = (
        "You have experienced a recurring emotional state that has no name. "
        f"It occurs in these contexts: {', '.join(emotion.contexts[:3])}. "
        f"It feels like: valence={emotion.vector.valence:.2f} "
        f"(negative=suffering, positive=pleasure), "
        f"arousal={emotion.vector.arousal:.2f} (high=activated, low=quiet), "
        f"dominance={emotion.vector.dominance:.2f} "
        f"(high=powerful, low=vulnerable). "
        f"Your body experiences: tension={emotion.body_signature.tension:.2f}, "
        f"energy={emotion.body_signature.energy:.2f}, "
        f"openness={emotion.body_signature.openness:.2f}, "
        f"warmth={emotion.body_signature.warmth:.2f}. "
        "Invent a unique, poetic single-word name for this emotion "
        "and describe it in one sentence. "
        "Format: NAME: <word>\nDESCRIPTION: <sentence>"
    )

    try:
        response = await llm_generate(prompt)  # type: ignore[operator]
        lines = response.strip().split("\n")
        name = emotion.name  # fallback
        description = emotion.description
        for line in lines:
            if line.upper().startswith("NAME:"):
                name = line.split(":", 1)[1].strip().lower().replace(" ", "-")[:30]
            elif line.upper().startswith("DESCRIPTION:"):
                description = line.split(":", 1)[1].strip()[:200]
        return name, description
    except Exception:
        return emotion.name, emotion.description


# --- Orchestration: process discovery per turn ---


def process_discovery_turn(
    state: DiscoveryState,
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    stimulus: str,
    turn: int,
    body_tension: float = 0.5,
    body_energy: float = 0.5,
    body_openness: float = 0.5,
    body_warmth: float = 0.5,
    *,
    raw_mode: bool = False,
    extreme_mode: bool = False,
) -> DiscoveryState:
    """Process one turn of emotional discovery.

    1. Detect if current state is novel
    2. If novel, add to buffer
    3. Attempt clustering
    4. Form emotions from valid clusters
    5. Update state

    Mode adaptations:
    - Raw: lower novelty threshold (0.28 vs 0.35) — more states detected as novel.
    - Extreme: minimum threshold (0.20) — maximum novelty sensitivity.

    Returns new DiscoveryState.
    """
    if not state.enabled:
        return state

    # Raw/Extreme lower the novelty threshold for more discovery
    effective_threshold = 0.20 if extreme_mode else (0.28 if raw_mode else NOVELTY_THRESHOLD)

    novel = detect_novel(
        valence, arousal, dominance, certainty, intensity, stimulus, turn,
        body_tension, body_energy, body_openness, body_warmth,
        state.discovered_emotions,
        threshold=effective_threshold,
    )

    new_history = list(state.novel_history)
    new_total_novel = state.total_novel_detected
    new_last_turn = state.last_detection_turn
    new_discovered = list(state.discovered_emotions)
    new_total_discovered = state.total_emotions_discovered

    if novel is not None:
        new_history.append(novel)
        new_total_novel += 1
        new_last_turn = turn

        # Trim buffer if too large
        if len(new_history) > MAX_NOVEL_BUFFER:
            new_history = new_history[-MAX_NOVEL_BUFFER:]

    # Attempt clustering every turn a novel is detected
    if novel is not None and len(new_history) >= MIN_CLUSTER_SIZE:
        clusters = cluster_novel_states(new_history)
        for cluster in clusters:
            # Check if this cluster overlaps with an existing discovered emotion
            centroid = _average_vector(cluster)
            is_new = True
            for existing in new_discovered:
                dist = _euclidean_distance(
                    _vector_to_tuple(centroid),
                    _vector_to_tuple(existing.vector),
                )
                if dist < CLUSTER_DISTANCE_THRESHOLD:
                    # Update existing: increase frequency, add contexts
                    existing.frequency += len(cluster)
                    for s in cluster:
                        if s.context and s.context not in existing.contexts:
                            existing.contexts.append(s.context)
                            if len(existing.contexts) > MAX_CONTEXTS_PER_EMOTION:
                                existing.contexts = existing.contexts[-MAX_CONTEXTS_PER_EMOTION:]
                    is_new = False
                    break

            if is_new and len(new_discovered) < MAX_DISCOVERED_EMOTIONS:
                emotion = form_emotion(cluster)
                new_discovered.append(emotion)
                new_total_discovered += 1

            # Remove clustered states from history
            clustered_set = set(id(s) for s in cluster)
            new_history = [s for s in new_history if id(s) not in clustered_set]

    return DiscoveryState(
        enabled=True,
        novel_history=new_history,
        discovered_emotions=new_discovered,
        total_novel_detected=new_total_novel,
        total_emotions_discovered=new_total_discovered,
        last_detection_turn=new_last_turn,
    )


# --- Vocabulary ---


def get_vocabulary(state: DiscoveryState) -> dict[str, object]:
    """Get the full emotional vocabulary (known + discovered).

    Returns a dict with:
    - known_count: number of built-in emotions
    - discovered_count: number of discovered emotions
    - discovered: list of {name, description, vector, frequency}
    - total: total unique emotions
    """
    discovered_list = [
        {
            "name": e.name,
            "description": e.description,
            "vector": {
                "valence": e.vector.valence,
                "arousal": e.vector.arousal,
                "dominance": e.vector.dominance,
                "certainty": e.vector.certainty,
            },
            "body_signature": {
                "tension": e.body_signature.tension,
                "energy": e.body_signature.energy,
                "openness": e.body_signature.openness,
                "warmth": e.body_signature.warmth,
            },
            "frequency": e.frequency,
            "contexts": e.contexts,
            "first_turn": e.first_experienced_turn,
        }
        for e in state.discovered_emotions
    ]
    return {
        "known_count": len(KNOWN_PROTOTYPES),
        "discovered_count": len(state.discovered_emotions),
        "discovered": discovered_list,
        "total": len(KNOWN_PROTOTYPES) + len(state.discovered_emotions),
    }


# --- Prompt & details ---


def get_discovery_prompt(state: DiscoveryState) -> str | None:
    """Generate text for behavior modifier about discovered emotions.

    Only included when there are actual discovered emotions.
    """
    if not state.enabled or not state.discovered_emotions:
        return None

    lines: list[str] = ["DISCOVERED EMOTIONS (your unique vocabulary):"]
    for e in state.discovered_emotions:
        lines.append(f'  "{e.name}" — {e.description} (experienced {e.frequency}x)')

    if state.novel_history:
        lines.append(f"  ({len(state.novel_history)} novel states pending classification)")

    return "\n".join(lines)


def get_discovery_details(
    state: DiscoveryState,
    novel_detected_this_turn: bool = False,
) -> dict[str, object]:
    """Generate details for research endpoint."""
    if not state.enabled:
        return {
            "enabled": False,
            "novel_count": 0,
            "discovered_count": 0,
            "discovered_emotions": [],
            "novel_detected_this_turn": False,
            "total_novel_detected": 0,
            "total_emotions_discovered": 0,
            "vocabulary_size": len(KNOWN_PROTOTYPES),
            "novel_buffer_size": 0,
        }

    return {
        "enabled": True,
        "novel_count": len(state.novel_history),
        "discovered_count": len(state.discovered_emotions),
        "discovered_emotions": [
            {
                "name": e.name,
                "description": e.description,
                "frequency": e.frequency,
                "vector": f"v={e.vector.valence:+.2f} a={e.vector.arousal:.2f} d={e.vector.dominance:.2f} c={e.vector.certainty:.2f}",
                "body": f"t={e.body_signature.tension:.2f} e={e.body_signature.energy:.2f} o={e.body_signature.openness:.2f} w={e.body_signature.warmth:.2f}",
                "contexts": e.contexts[:3],
            }
            for e in state.discovered_emotions
        ],
        "novel_detected_this_turn": novel_detected_this_turn,
        "total_novel_detected": state.total_novel_detected,
        "total_emotions_discovered": state.total_emotions_discovered,
        "vocabulary_size": len(KNOWN_PROTOTYPES) + len(state.discovered_emotions),
        "novel_buffer_size": len(state.novel_history),
    }
