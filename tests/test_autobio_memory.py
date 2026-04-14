"""Tests para Autobiographical Memory (Pilar 3 ANIMA) - Paso 3.1.

Verifica:
- Sensory buffer capture
- Episode encoding (threshold, classification)
- Episodic store (capacity, eviction, retrieval)
- Working memory (selection, capacity, composite scoring)
- Narrative formation (clustering, reinforcement, limits)
- Orchestration (process_autobiographical_turn)
- Prompt generation
- AutobiographicalDetails
- OPT-IN behavior (disabled state)
"""

import pytest

from pathos.engine.autobio_memory import (
    _classify_significance,
    _emotional_distance,
    _extract_keywords,
    _keyword_similarity,
    attempt_narrative_formation,
    capture_sensory,
    encode_episode,
    get_autobiographical_details,
    get_autobiographical_prompt,
    process_autobiographical_turn,
    retrieve_episodes_by_emotion,
    retrieve_episodes_by_similarity,
    store_episode,
    update_working_memory,
    RECENCY_DECAY_RATE,
)
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
    default_autobiographical_state,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    valence: float = 0.0,
    arousal: float = 0.3,
    intensity: float = 0.5,
    dominance: float = 0.5,
    certainty: float = 0.5,
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        dominance=dominance,
        certainty=certainty,
        primary_emotion=emotion,
    )


def _episode(
    emotion: str = "joy",
    valence: float = 0.6,
    arousal: float = 0.5,
    intensity: float = 0.7,
    dominance: float = 0.5,
    certainty: float = 0.5,
    turn: int = 1,
    stimulus: str = "test stimulus",
    consolidated: bool = False,
    ep_id: str | None = None,
) -> Episode:
    ep = Episode(
        stimulus=stimulus,
        response_summary="test response",
        primary_emotion=emotion,
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        dominance=dominance,
        certainty=certainty,
        significance=_classify_significance(intensity),
        turn_number=turn,
        session_id="test-session",
        keywords=_extract_keywords(stimulus),
        consolidated=consolidated,
    )
    if ep_id:
        ep.id = ep_id
    return ep


def _autobio_state(enabled: bool = True) -> AutobiographicalState:
    return AutobiographicalState(enabled=enabled, session_id="test-session")


# ===========================================================================
# Test Classes
# ===========================================================================


class TestSensoryBuffer:
    """Tests para Level 1: Sensory Buffer."""

    def test_capture_basic(self):
        es = _state(valence=0.5, intensity=0.6, emotion=PrimaryEmotion.JOY)
        snap = capture_sensory("hello world", es, 0.3, 5)
        assert snap.stimulus == "hello world"
        assert snap.appraisal_valence == 0.5
        assert snap.prediction_error == 0.3
        assert snap.primary_emotion == "joy"
        assert snap.turn_number == 5

    def test_capture_truncates_long_stimulus(self):
        long_text = "x" * 1000
        snap = capture_sensory(long_text, _state(), 0.0, 1)
        assert len(snap.stimulus) == 500

    def test_capture_clamps_prediction_error(self):
        snap = capture_sensory("test", _state(), 1.5, 1)
        assert snap.prediction_error == 1.0

    def test_capture_zero_intensity(self):
        snap = capture_sensory("test", _state(intensity=0.0), 0.0, 0)
        assert snap.intensity == 0.0
        assert snap.appraisal_relevance == 0.0


class TestEpisodeEncoding:
    """Tests para Level 3: Episode encoding."""

    def test_encode_above_threshold(self):
        es = _state(valence=0.7, intensity=0.6, emotion=PrimaryEmotion.JOY)
        ep = encode_episode("great conversation", es, "thanks", 5, "s1")
        assert ep is not None
        assert ep.primary_emotion == "joy"
        assert ep.valence == 0.7
        assert ep.turn_number == 5

    def test_encode_below_threshold_returns_none(self):
        es = _state(intensity=0.3)
        ep = encode_episode("boring", es, "ok", 1, "s1")
        assert ep is None

    def test_encode_at_threshold(self):
        es = _state(intensity=EPISODIC_INTENSITY_THRESHOLD)
        ep = encode_episode("threshold", es, "ok", 1, "s1")
        assert ep is not None

    def test_encode_just_below_threshold(self):
        es = _state(intensity=EPISODIC_INTENSITY_THRESHOLD - 0.01)
        ep = encode_episode("below", es, "ok", 1, "s1")
        assert ep is None

    def test_encode_preserves_workspace_context(self):
        es = _state(intensity=0.8)
        ep = encode_episode("test", es, "ok", 1, "s1",
                           workspace_contents=["appraisal", "schema"],
                           preconscious_count=3)
        assert ep is not None
        assert ep.workspace_contents == ["appraisal", "schema"]
        assert ep.preconscious_count == 3

    def test_encode_extracts_keywords(self):
        es = _state(intensity=0.7)
        ep = encode_episode("Python programming is fascinating", es, "ok", 1, "s1")
        assert ep is not None
        assert "python" in ep.keywords
        assert "programming" in ep.keywords
        assert "fascinating" in ep.keywords

    def test_encode_truncates_stimulus(self):
        es = _state(intensity=0.7)
        ep = encode_episode("x" * 1000, es, "ok", 1, "s1")
        assert ep is not None
        assert len(ep.stimulus) == 500

    def test_encode_with_embedding(self):
        es = _state(intensity=0.7)
        emb = [0.1, 0.2, 0.3]
        ep = encode_episode("test", es, "ok", 1, "s1", embedding=emb)
        assert ep is not None
        assert ep.embedding == [0.1, 0.2, 0.3]


class TestSignificanceClassification:
    """Tests para clasificacion de significancia."""

    def test_low(self):
        assert _classify_significance(0.5) == EpisodeSignificance.LOW
        assert _classify_significance(0.6) == EpisodeSignificance.LOW

    def test_moderate(self):
        assert _classify_significance(0.61) == EpisodeSignificance.MODERATE
        assert _classify_significance(0.7) == EpisodeSignificance.MODERATE

    def test_high(self):
        assert _classify_significance(0.71) == EpisodeSignificance.HIGH
        assert _classify_significance(0.8) == EpisodeSignificance.HIGH

    def test_peak(self):
        assert _classify_significance(0.81) == EpisodeSignificance.PEAK
        assert _classify_significance(1.0) == EpisodeSignificance.PEAK


class TestEpisodicStore:
    """Tests para almacenamiento episodico."""

    def test_store_basic(self):
        store = EpisodicStore()
        ep = _episode()
        new_store = store_episode(store, ep)
        assert new_store.count() == 1
        assert new_store.total_encoded == 1

    def test_store_multiple(self):
        store = EpisodicStore()
        for i in range(5):
            ep = _episode(turn=i)
            store = store_episode(store, ep)
        assert store.count() == 5
        assert store.total_encoded == 5

    def test_eviction_at_capacity(self):
        store = EpisodicStore()
        # Fill to max
        for i in range(EPISODIC_MAX_PER_SESSION):
            ep = _episode(intensity=0.5 + (i % 5) * 0.1, turn=i)
            store = store_episode(store, ep)
        assert store.count() == EPISODIC_MAX_PER_SESSION

        # One more triggers eviction
        ep = _episode(intensity=0.9, turn=EPISODIC_MAX_PER_SESSION)
        store = store_episode(store, ep)
        assert store.count() == EPISODIC_MAX_PER_SESSION

    def test_eviction_prefers_consolidated(self):
        store = EpisodicStore()
        # Add consolidated low-intensity episode
        ep_consolidated = _episode(intensity=0.5, consolidated=True, ep_id="consolidated")
        store = store_episode(store, ep_consolidated)
        # Add unconsolidated high-intensity episode
        ep_fresh = _episode(intensity=0.9, consolidated=False, ep_id="fresh")
        store = store_episode(store, ep_fresh)

        # Fill to max
        for i in range(EPISODIC_MAX_PER_SESSION - 2):
            ep = _episode(intensity=0.6, turn=i + 2)
            store = store_episode(store, ep)

        # One more triggers eviction of consolidated one
        ep_new = _episode(intensity=0.8, turn=999)
        store = store_episode(store, ep_new)
        assert store.count() == EPISODIC_MAX_PER_SESSION
        # The consolidated low-intensity should be evicted
        assert store.get_by_id("consolidated") is None
        assert store.get_by_id("fresh") is not None

    def test_get_by_id(self):
        store = EpisodicStore()
        ep = _episode(ep_id="test123")
        store = store_episode(store, ep)
        assert store.get_by_id("test123") is not None
        assert store.get_by_id("nonexistent") is None

    def test_get_high_intensity(self):
        store = EpisodicStore()
        for intensity in [0.5, 0.6, 0.7, 0.8, 0.9]:
            ep = _episode(intensity=intensity)
            store = store_episode(store, ep)
        high = store.get_high_intensity(threshold=0.7)
        assert len(high) == 3

    def test_get_unconsolidated(self):
        store = EpisodicStore()
        ep1 = _episode(consolidated=True)
        ep2 = _episode(consolidated=False)
        store = store_episode(store, ep1)
        store = store_episode(store, ep2)
        unconsolidated = store.get_unconsolidated()
        assert len(unconsolidated) == 1


class TestEpisodicRetrieval:
    """Tests para recuperacion de episodios."""

    def test_retrieve_by_emotion(self):
        store = EpisodicStore()
        for i in range(5):
            store = store_episode(store, _episode(emotion="joy", intensity=0.5 + i * 0.1, turn=i))
        for i in range(3):
            store = store_episode(store, _episode(emotion="sadness", intensity=0.6, turn=i + 5))

        joy_episodes = retrieve_episodes_by_emotion(store, "joy")
        assert len(joy_episodes) == 5
        # Sorted by intensity (highest first)
        assert joy_episodes[0].intensity >= joy_episodes[-1].intensity

    def test_retrieve_by_emotion_with_limit(self):
        store = EpisodicStore()
        for i in range(10):
            store = store_episode(store, _episode(emotion="joy", turn=i))
        result = retrieve_episodes_by_emotion(store, "joy", limit=3)
        assert len(result) == 3

    def test_retrieve_by_similarity(self):
        store = EpisodicStore()
        target = _episode(valence=0.7, arousal=0.5, intensity=0.8, dominance=0.5)
        similar = _episode(valence=0.6, arousal=0.5, intensity=0.75, dominance=0.5, ep_id="sim")
        different = _episode(valence=-0.8, arousal=0.9, intensity=0.9, dominance=0.2, ep_id="diff")
        store = store_episode(store, similar)
        store = store_episode(store, different)

        results = retrieve_episodes_by_similarity(store, target, threshold=0.3)
        assert any(ep.id == "sim" for ep in results)

    def test_retrieve_by_similarity_excludes_self(self):
        store = EpisodicStore()
        ep = _episode(ep_id="self1")
        store = store_episode(store, ep)
        results = retrieve_episodes_by_similarity(store, ep)
        assert len(results) == 0


class TestEmotionalDistance:
    """Tests para distancia emocional."""

    def test_identical_episodes_zero_distance(self):
        a = _episode(valence=0.5, arousal=0.5, intensity=0.5, dominance=0.5)
        b = _episode(valence=0.5, arousal=0.5, intensity=0.5, dominance=0.5)
        assert _emotional_distance(a, b) == pytest.approx(0.0)

    def test_opposite_episodes_high_distance(self):
        a = _episode(valence=1.0, arousal=1.0, intensity=1.0, dominance=1.0)
        b = _episode(valence=-1.0, arousal=0.0, intensity=0.0, dominance=0.0)
        dist = _emotional_distance(a, b)
        assert dist > 0.8

    def test_distance_is_symmetric(self):
        a = _episode(valence=0.3, arousal=0.7, intensity=0.5)
        b = _episode(valence=-0.2, arousal=0.4, intensity=0.8)
        assert _emotional_distance(a, b) == pytest.approx(_emotional_distance(b, a))

    def test_distance_clamped_0_to_1(self):
        a = _episode(valence=1.0, arousal=1.0, intensity=1.0, dominance=1.0)
        b = _episode(valence=-1.0, arousal=0.0, intensity=0.0, dominance=0.0)
        dist = _emotional_distance(a, b)
        assert 0 <= dist <= 1


class TestWorkingMemory:
    """Tests para Level 2: Working Memory."""

    def test_empty_state_returns_empty_working_memory(self):
        state = _autobio_state()
        wm = update_working_memory(state, "hello", 1)
        assert len(wm.items) == 0

    def test_selects_top_k_from_episodes(self):
        state = _autobio_state()
        # Add more than WORKING_MEMORY_CAPACITY episodes
        episodes = []
        for i in range(10):
            ep = _episode(
                intensity=0.5 + i * 0.05,
                turn=i,
                stimulus=f"episode about topic {i}",
            )
            episodes.append(ep)
        state.episodic = EpisodicStore(episodes=episodes, total_encoded=10)

        wm = update_working_memory(state, "topic", 10)
        assert len(wm.items) <= WORKING_MEMORY_CAPACITY

    def test_capacity_is_respected(self):
        state = _autobio_state()
        state.working_memory.capacity = 3
        episodes = []
        for i in range(10):
            ep = _episode(turn=i, stimulus=f"test {i}")
            episodes.append(ep)
        state.episodic = EpisodicStore(episodes=episodes, total_encoded=10)

        wm = update_working_memory(state, "test", 10)
        assert len(wm.items) <= 3

    def test_composite_score_weights(self):
        item = MemoryItem(
            source_id="t1",
            content="test",
            relevance=1.0,
            emotional_intensity=1.0,
            recency=1.0,
        )
        score = item.compute_composite()
        # 0.4*1 + 0.35*1 + 0.25*1 = 1.0
        assert score == pytest.approx(1.0)

    def test_composite_zero_inputs(self):
        item = MemoryItem(
            source_id="t1",
            content="test",
            relevance=0.0,
            emotional_intensity=0.0,
            recency=0.0,
        )
        assert item.compute_composite() == 0.0

    def test_recency_decay(self):
        state = _autobio_state()
        # Episode from 50 turns ago should have recency 0
        ep_old = _episode(turn=0, stimulus="old topic")
        ep_recent = _episode(turn=49, stimulus="recent topic")
        state.episodic = EpisodicStore(episodes=[ep_old, ep_recent], total_encoded=2)

        wm = update_working_memory(state, "topic", 50)
        if wm.items:
            # Recent should be ranked higher
            recent_items = [it for it in wm.items if "recent" in it.content]
            old_items = [it for it in wm.items if "old" in it.content]
            if recent_items and old_items:
                assert recent_items[0].recency > old_items[0].recency

    def test_includes_narrative_items(self):
        state = _autobio_state()
        ns = NarrativeStatement(
            narrative_type=NarrativeType.PATTERN,
            statement="I tend to feel joy in positive interactions",
            primary_emotion="joy",
            valence=0.6,
            episode_count=5,
            strength=0.8,
            last_reinforced_turn=5,
        )
        state.narrative = NarrativeStore(statements=[ns])

        wm = update_working_memory(state, "positive interaction joy", 6)
        narrative_items = [it for it in wm.items if it.source_type == "narrative"]
        assert len(narrative_items) > 0

    def test_last_updated_turn(self):
        state = _autobio_state()
        wm = update_working_memory(state, "test", 42)
        assert wm.last_updated_turn == 42


class TestNarrativeFormation:
    """Tests para Level 4: Narrative Memory."""

    def test_no_formation_below_threshold(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES - 1):
            store = store_episode(store, _episode(emotion="joy", turn=i))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 10, "s1")
        assert result.count() == 0

    def test_formation_at_threshold(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(emotion="joy", valence=0.6, turn=i))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 10, "s1")
        assert result.count() == 1
        assert result.statements[0].primary_emotion == "joy"

    def test_formation_multiple_emotions(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(emotion="joy", turn=i))
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(emotion="sadness", valence=-0.5, turn=i + 10))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 20, "s1")
        assert result.count() == 2
        emotions = {s.primary_emotion for s in result.statements}
        assert "joy" in emotions
        assert "sadness" in emotions

    def test_reinforcement_existing_narrative(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES + 3):
            store = store_episode(store, _episode(emotion="joy", turn=i))

        # Pre-existing narrative
        ns = NarrativeStatement(
            narrative_type=NarrativeType.PATTERN,
            statement="I feel joy often",
            primary_emotion="joy",
            valence=0.5,
            strength=0.3,
            last_reinforced_turn=0,
        )
        narrative = NarrativeStore(statements=[ns])

        result = attempt_narrative_formation(store, narrative, 20, "s1")
        assert result.count() == 1
        # Strength should increase
        assert result.statements[0].strength > 0.3

    def test_narrative_max_limit(self):
        store = EpisodicStore()
        # Create many different emotions with enough episodes each
        emotions = [f"emotion_{i}" for i in range(NARRATIVE_MAX_STATEMENTS + 5)]
        for emo in emotions:
            for j in range(NARRATIVE_MIN_EPISODES):
                ep = _episode(emotion=emo, turn=j)
                store = store_episode(store, ep)

        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 100, "s1")
        assert result.count() <= NARRATIVE_MAX_STATEMENTS

    def test_narrative_type_vulnerability(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(
                emotion="fear", valence=-0.5, intensity=0.8, turn=i,
            ))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 10, "s1")
        assert result.count() == 1
        assert result.statements[0].narrative_type == NarrativeType.VULNERABILITY

    def test_narrative_type_pattern(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(
                emotion="contentment", valence=0.2, intensity=0.6, turn=i,
            ))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 10, "s1")
        assert result.count() == 1
        assert result.statements[0].narrative_type == NarrativeType.PATTERN

    def test_narrative_text_positive(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(emotion="joy", valence=0.7, turn=i))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 10, "s1")
        assert "positive" in result.statements[0].statement.lower()

    def test_narrative_text_negative(self):
        store = EpisodicStore()
        for i in range(NARRATIVE_MIN_EPISODES):
            store = store_episode(store, _episode(emotion="anger", valence=-0.5, turn=i))
        narrative = NarrativeStore()
        result = attempt_narrative_formation(store, narrative, 10, "s1")
        assert "challenging" in result.statements[0].statement.lower()


class TestNarrativeStore:
    """Tests para NarrativeStore helpers."""

    def test_get_strongest(self):
        statements = [
            NarrativeStatement(narrative_type=NarrativeType.PATTERN, statement="s1",
                             primary_emotion="joy", valence=0.5, strength=0.3),
            NarrativeStatement(narrative_type=NarrativeType.PATTERN, statement="s2",
                             primary_emotion="sadness", valence=-0.3, strength=0.9),
            NarrativeStatement(narrative_type=NarrativeType.PATTERN, statement="s3",
                             primary_emotion="anger", valence=-0.5, strength=0.6),
        ]
        store = NarrativeStore(statements=statements)
        strongest = store.get_strongest(k=2)
        assert len(strongest) == 2
        assert strongest[0].strength == 0.9

    def test_get_by_emotion(self):
        statements = [
            NarrativeStatement(narrative_type=NarrativeType.PATTERN, statement="s1",
                             primary_emotion="joy", valence=0.5),
            NarrativeStatement(narrative_type=NarrativeType.PATTERN, statement="s2",
                             primary_emotion="joy", valence=0.6),
            NarrativeStatement(narrative_type=NarrativeType.PATTERN, statement="s3",
                             primary_emotion="anger", valence=-0.5),
        ]
        store = NarrativeStore(statements=statements)
        joy = store.get_by_emotion("joy")
        assert len(joy) == 2


class TestOrchestration:
    """Tests para process_autobiographical_turn."""

    def test_disabled_returns_unchanged(self):
        state = _autobio_state(enabled=False)
        result = process_autobiographical_turn(
            state, "hello", _state(intensity=0.8, emotion=PrimaryEmotion.JOY),
            "response", 1, "s1",
        )
        assert not result.enabled
        assert result.episodic.count() == 0

    def test_enabled_captures_sensory(self):
        state = _autobio_state()
        result = process_autobiographical_turn(
            state, "hello", _state(intensity=0.3), "response", 1, "s1",
        )
        assert result.sensory_buffer.stimulus == "hello"
        assert result.sensory_buffer.turn_number == 1

    def test_enabled_encodes_episode_above_threshold(self):
        state = _autobio_state()
        result = process_autobiographical_turn(
            state, "important event",
            _state(intensity=0.7, emotion=PrimaryEmotion.JOY),
            "response", 1, "s1",
        )
        assert result.episodic.count() == 1

    def test_enabled_no_episode_below_threshold(self):
        state = _autobio_state()
        result = process_autobiographical_turn(
            state, "boring event", _state(intensity=0.3), "response", 1, "s1",
        )
        assert result.episodic.count() == 0

    def test_increments_turns_processed(self):
        state = _autobio_state()
        result = process_autobiographical_turn(
            state, "test", _state(), "response", 1, "s1",
        )
        assert result.total_turns_processed == 1
        result2 = process_autobiographical_turn(
            result, "test2", _state(), "response", 2, "s1",
        )
        assert result2.total_turns_processed == 2

    def test_preserves_session_id(self):
        state = _autobio_state()
        result = process_autobiographical_turn(
            state, "test", _state(), "resp", 1, "my-session",
        )
        assert result.session_id == "my-session"

    def test_preserves_dream_report(self):
        state = _autobio_state()
        state.last_dream_report = "A dream about the ocean"
        result = process_autobiographical_turn(
            state, "test", _state(), "resp", 1, "s1",
        )
        assert result.last_dream_report == "A dream about the ocean"

    def test_narrative_forms_after_enough_episodes(self):
        state = _autobio_state()
        for i in range(NARRATIVE_MIN_EPISODES + 1):
            state = process_autobiographical_turn(
                state, f"joyful event number {i}",
                _state(intensity=0.7, valence=0.6, emotion=PrimaryEmotion.JOY),
                "happy response", i, "s1",
            )
        assert state.narrative.count() >= 1

    def test_full_flow_multiple_turns(self):
        state = _autobio_state()
        # 10 turns with varying emotions
        emotions_data = [
            (PrimaryEmotion.JOY, 0.7, 0.6),
            (PrimaryEmotion.SADNESS, 0.8, -0.5),
            (PrimaryEmotion.JOY, 0.6, 0.5),
            (PrimaryEmotion.ANGER, 0.9, -0.7),
            (PrimaryEmotion.JOY, 0.7, 0.6),
            (PrimaryEmotion.SADNESS, 0.5, -0.3),
            (PrimaryEmotion.JOY, 0.8, 0.7),
            (PrimaryEmotion.JOY, 0.6, 0.5),
            (PrimaryEmotion.NEUTRAL, 0.4, -0.2),
            (PrimaryEmotion.JOY, 0.7, 0.6),
        ]
        for i, (emo, inten, val) in enumerate(emotions_data):
            state = process_autobiographical_turn(
                state, f"stimulus {i}", _state(intensity=inten, valence=val, emotion=emo),
                f"response {i}", i, "s1",
            )
        assert state.total_turns_processed == 10
        # Episodes stored (only intensity >= 0.5)
        assert state.episodic.count() > 0
        # Working memory has items from episodes
        if state.episodic.count() > 0:
            assert len(state.working_memory.items) > 0


class TestPromptGeneration:
    """Tests para get_autobiographical_prompt."""

    def test_disabled_returns_empty(self):
        state = _autobio_state(enabled=False)
        assert get_autobiographical_prompt(state) == ""

    def test_with_working_memory(self):
        state = _autobio_state()
        state.working_memory.items = [
            MemoryItem(source_id="t1", content="[joy] happy event", relevance=0.8, emotional_intensity=0.7),
        ]
        prompt = get_autobiographical_prompt(state)
        assert "AUTOBIOGRAPHICAL MEMORY" in prompt
        assert "happy event" in prompt

    def test_with_narratives(self):
        state = _autobio_state()
        ns = NarrativeStatement(
            narrative_type=NarrativeType.PATTERN,
            statement="I feel joy in conversation",
            primary_emotion="joy",
            valence=0.5,
            strength=0.8,
        )
        state.narrative = NarrativeStore(statements=[ns])
        prompt = get_autobiographical_prompt(state)
        assert "SELF-KNOWLEDGE" in prompt
        assert "I feel joy" in prompt

    def test_with_dream_report(self):
        state = _autobio_state()
        state.last_dream_report = "Dreamt of flowing water"
        prompt = get_autobiographical_prompt(state)
        assert "DREAM ECHO" in prompt
        assert "flowing water" in prompt

    def test_with_episodes(self):
        state = _autobio_state()
        ep = _episode()
        state.episodic = EpisodicStore(episodes=[ep], total_encoded=1)
        prompt = get_autobiographical_prompt(state)
        assert "EXPERIENTIAL DEPTH" in prompt


class TestAutobiographicalDetails:
    """Tests para research endpoint details."""

    def test_disabled_details(self):
        state = _autobio_state(enabled=False)
        details = get_autobiographical_details(state)
        assert not details.enabled
        assert details.episodic_count == 0

    def test_enabled_details(self):
        state = _autobio_state()
        state.sensory_buffer.primary_emotion = "joy"
        state.sensory_buffer.intensity = 0.7
        ep = _episode()
        state.episodic = EpisodicStore(episodes=[ep], total_encoded=1)
        state.total_turns_processed = 5

        details = get_autobiographical_details(state)
        assert details.enabled
        assert details.sensory_emotion == "joy"
        assert details.episodic_count == 1
        assert details.total_turns_processed == 5


class TestOptInBehavior:
    """Tests para verificar que OPT-IN funciona correctamente."""

    def test_default_is_disabled(self):
        state = default_autobiographical_state()
        assert not state.enabled

    def test_disabled_process_is_noop(self):
        state = default_autobiographical_state()
        result = process_autobiographical_turn(
            state, "intense", _state(intensity=0.9, emotion=PrimaryEmotion.JOY),
            "response", 1, "s1",
        )
        assert result.episodic.count() == 0
        assert result.total_turns_processed == 0
        assert not result.enabled

    def test_enabled_process_works(self):
        state = default_autobiographical_state()
        state.enabled = True
        result = process_autobiographical_turn(
            state, "intense", _state(intensity=0.9, emotion=PrimaryEmotion.JOY),
            "response", 1, "s1",
        )
        assert result.episodic.count() == 1
        assert result.total_turns_processed == 1


class TestKeywordUtilities:
    """Tests para funciones de utilidad de keywords."""

    def test_extract_keywords(self):
        kw = _extract_keywords("The quick brown fox")
        assert "quick" in kw
        assert "brown" in kw
        assert "fox" in kw
        assert "the" not in kw

    def test_extract_keywords_removes_short(self):
        kw = _extract_keywords("I am a big cat")
        assert "big" in kw
        assert "cat" in kw
        assert "am" not in kw

    def test_keyword_similarity_identical(self):
        a = ["python", "programming"]
        assert _keyword_similarity(a, a) == 1.0

    def test_keyword_similarity_disjoint(self):
        assert _keyword_similarity(["python"], ["java"]) == 0.0

    def test_keyword_similarity_partial(self):
        sim = _keyword_similarity(["python", "code"], ["python", "script"])
        assert 0 < sim < 1

    def test_keyword_similarity_empty(self):
        assert _keyword_similarity([], ["python"]) == 0.0
        assert _keyword_similarity(["python"], []) == 0.0


class TestModelHelpers:
    """Tests para model helpers."""

    def test_memory_item_composite_clamped(self):
        item = MemoryItem(
            source_id="t1",
            content="test",
            relevance=1.0,
            emotional_intensity=1.0,
            recency=1.0,
        )
        score = item.compute_composite()
        assert 0 <= score <= 1

    def test_episodic_store_count(self):
        store = EpisodicStore(episodes=[_episode(), _episode()])
        assert store.count() == 2

    def test_narrative_store_count(self):
        ns = NarrativeStatement(
            narrative_type=NarrativeType.PATTERN,
            statement="test",
            primary_emotion="joy",
            valence=0.5,
        )
        store = NarrativeStore(statements=[ns])
        assert store.count() == 1
