"""Tests para Pilar 6: Descubrimiento Emocional Emergente."""

import pytest

from pathos.engine.discovery import (
    KNOWN_PROTOTYPES,
    _average_body,
    _average_vector,
    _euclidean_distance,
    _novel_distance,
    cluster_novel_states,
    detect_novel,
    form_emotion,
    get_discovery_details,
    get_discovery_prompt,
    get_vocabulary,
    name_emotion_fallback,
    process_discovery_turn,
)
from pathos.models.discovery import (
    CLUSTER_DISTANCE_THRESHOLD,
    MAX_NOVEL_BUFFER,
    MIN_CLUSTER_SIZE,
    NOVELTY_THRESHOLD,
    BodySignature,
    DiscoveredEmotion,
    DiscoveryState,
    EmotionalVector,
    NovelEmotionalState,
    default_discovery_state,
)


# --- Helpers ---

def _novel(
    v: float = 0.0, a: float = 0.5, d: float = 0.5, c: float = 0.5,
    context: str = "test", turn: int = 1, intensity: float = 0.5,
    tension: float = 0.5, energy: float = 0.5, openness: float = 0.5, warmth: float = 0.5,
) -> NovelEmotionalState:
    return NovelEmotionalState(
        vector=EmotionalVector(valence=v, arousal=a, dominance=d, certainty=c),
        body=BodySignature(tension=tension, energy=energy, openness=openness, warmth=warmth),
        context=context,
        turn=turn,
        intensity=intensity,
    )


def _enabled_state(**kwargs: object) -> DiscoveryState:
    state = default_discovery_state()
    state.enabled = True
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


# ===============================================================
# Test Models
# ===============================================================

class TestDefaultState:
    def test_disabled_by_default(self) -> None:
        state = default_discovery_state()
        assert state.enabled is False

    def test_empty_buffers(self) -> None:
        state = default_discovery_state()
        assert state.novel_history == []
        assert state.discovered_emotions == []

    def test_zero_counters(self) -> None:
        state = default_discovery_state()
        assert state.total_novel_detected == 0
        assert state.total_emotions_discovered == 0


class TestEmotionalVector:
    def test_valid_range(self) -> None:
        v = EmotionalVector(valence=-0.5, arousal=0.8, dominance=0.3, certainty=0.9)
        assert v.valence == -0.5
        assert v.arousal == 0.8

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            EmotionalVector(valence=2.0, arousal=0.5, dominance=0.5, certainty=0.5)


class TestNovelEmotionalState:
    def test_creation(self) -> None:
        n = _novel(v=-0.3, a=0.9, context="existential question")
        assert n.vector.valence == -0.3
        assert n.context == "existential question"

    def test_intensity_clamped(self) -> None:
        with pytest.raises(Exception):
            NovelEmotionalState(
                vector=EmotionalVector(valence=0, arousal=0.5, dominance=0.5, certainty=0.5),
                body=BodySignature(),
                intensity=1.5,
            )


# ===============================================================
# Test Distance
# ===============================================================

class TestDistance:
    def test_same_point_zero(self) -> None:
        assert _euclidean_distance((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0

    def test_symmetric(self) -> None:
        a = (0.5, 0.3, 0.7, 0.2)
        b = (-0.2, 0.8, 0.1, 0.9)
        assert _euclidean_distance(a, b) == pytest.approx(_euclidean_distance(b, a))

    def test_valence_weighted_more(self) -> None:
        # Same absolute difference but in valence vs arousal
        d_valence = _euclidean_distance((0.5, 0.5, 0.5, 0.5), (0.0, 0.5, 0.5, 0.5))
        d_arousal = _euclidean_distance((0.5, 0.5, 0.5, 0.5), (0.5, 0.0, 0.5, 0.5))
        assert d_valence > d_arousal  # valence weight 2.5 > arousal weight 1.0

    def test_novel_distance_uses_vectors(self) -> None:
        a = _novel(v=0.5, a=0.5, d=0.5, c=0.5)
        b = _novel(v=-0.5, a=0.5, d=0.5, c=0.5)
        dist = _novel_distance(a, b)
        assert dist > 0


# ===============================================================
# Test Novel Detection
# ===============================================================

class TestNovelDetection:
    def test_known_emotion_not_novel(self) -> None:
        # Joy prototype: (0.75, 0.65, 0.70, 0.70) — should not be novel
        result = detect_novel(
            valence=0.75, arousal=0.65, dominance=0.70, certainty=0.70,
            intensity=0.6, stimulus="happy moment", turn=1,
        )
        assert result is None

    def test_close_to_known_not_novel(self) -> None:
        # Slightly off from joy
        result = detect_novel(
            valence=0.72, arousal=0.63, dominance=0.68, certainty=0.68,
            intensity=0.5, stimulus="test", turn=1,
        )
        assert result is None

    def test_far_from_all_is_novel(self) -> None:
        # This point is far from all prototypes
        result = detect_novel(
            valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.7, stimulus="strange feeling", turn=5,
        )
        assert result is not None
        assert result.min_distance >= NOVELTY_THRESHOLD
        assert result.context == "strange feeling"
        assert result.turn == 5

    def test_includes_body_state(self) -> None:
        result = detect_novel(
            valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.5, stimulus="test", turn=1,
            body_tension=0.8, body_energy=0.2,
        )
        assert result is not None
        assert result.body.tension == 0.8
        assert result.body.energy == 0.2

    def test_discovered_emotions_count_as_known(self) -> None:
        # First, create a "discovered emotion" at the novel point
        discovered = [DiscoveredEmotion(
            name="test-emotion",
            vector=EmotionalVector(valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1),
            body_signature=BodySignature(),
        )]
        result = detect_novel(
            valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.5, stimulus="test", turn=1,
            discovered_emotions=discovered,
        )
        assert result is None  # Now it maps to the discovered emotion

    def test_custom_threshold(self) -> None:
        # With very high threshold, everything is "known"
        result = detect_novel(
            valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.5, stimulus="test", turn=1,
            threshold=10.0,
        )
        assert result is None

    def test_context_truncated(self) -> None:
        long_context = "x" * 500
        result = detect_novel(
            valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.5, stimulus=long_context, turn=1,
        )
        assert result is not None
        assert len(result.context) <= 200


# ===============================================================
# Test Clustering
# ===============================================================

class TestClustering:
    def test_empty_input(self) -> None:
        assert cluster_novel_states([]) == []

    def test_too_few_states(self) -> None:
        states = [_novel(v=0.1), _novel(v=0.12)]
        assert cluster_novel_states(states) == []

    def test_single_cluster(self) -> None:
        # Three very similar states
        states = [
            _novel(v=0.3, a=0.1, d=0.9, c=0.1),
            _novel(v=0.32, a=0.12, d=0.88, c=0.12),
            _novel(v=0.28, a=0.08, d=0.92, c=0.08),
        ]
        clusters = cluster_novel_states(states)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_two_clusters(self) -> None:
        # Two groups far apart
        cluster_a = [
            _novel(v=0.3, a=0.1, d=0.9, c=0.1),
            _novel(v=0.32, a=0.12, d=0.88, c=0.12),
            _novel(v=0.28, a=0.08, d=0.92, c=0.08),
        ]
        cluster_b = [
            _novel(v=-0.8, a=0.9, d=0.1, c=0.9),
            _novel(v=-0.78, a=0.88, d=0.12, c=0.88),
            _novel(v=-0.82, a=0.92, d=0.08, c=0.92),
        ]
        clusters = cluster_novel_states(cluster_a + cluster_b)
        assert len(clusters) == 2

    def test_scattered_no_cluster(self) -> None:
        # All far apart
        states = [
            _novel(v=0.9, a=0.1, d=0.1, c=0.1),
            _novel(v=-0.9, a=0.9, d=0.9, c=0.9),
            _novel(v=0.0, a=0.5, d=0.5, c=0.5),
        ]
        clusters = cluster_novel_states(states)
        assert len(clusters) == 0

    def test_min_cluster_size_respected(self) -> None:
        states = [
            _novel(v=0.3, a=0.1, d=0.9, c=0.1),
            _novel(v=0.32, a=0.12, d=0.88, c=0.12),
        ]
        clusters = cluster_novel_states(states, min_cluster_size=2)
        assert len(clusters) == 1

        clusters = cluster_novel_states(states, min_cluster_size=3)
        assert len(clusters) == 0


# ===============================================================
# Test Emotion Formation
# ===============================================================

class TestEmotionFormation:
    def test_basic_formation(self) -> None:
        cluster = [
            _novel(v=0.3, a=0.1, d=0.9, c=0.1, context="ctx1", turn=5),
            _novel(v=0.32, a=0.12, d=0.88, c=0.12, context="ctx2", turn=7),
            _novel(v=0.28, a=0.08, d=0.92, c=0.08, context="ctx3", turn=9),
        ]
        emotion = form_emotion(cluster)
        assert emotion.cluster_size == 3
        assert emotion.frequency == 3
        assert emotion.first_experienced_turn == 5
        assert len(emotion.contexts) == 3
        assert emotion.named is True
        # Centroid should be average
        assert emotion.vector.valence == pytest.approx(0.3, abs=0.01)

    def test_body_signature_averaged(self) -> None:
        cluster = [
            _novel(tension=0.8, energy=0.2, openness=0.5, warmth=0.3),
            _novel(tension=0.6, energy=0.4, openness=0.5, warmth=0.5),
            _novel(tension=0.7, energy=0.3, openness=0.5, warmth=0.4),
        ]
        emotion = form_emotion(cluster)
        assert emotion.body_signature.tension == pytest.approx(0.7, abs=0.01)
        assert emotion.body_signature.energy == pytest.approx(0.3, abs=0.01)

    def test_duplicate_contexts_deduped(self) -> None:
        cluster = [
            _novel(context="same context"),
            _novel(context="same context"),
            _novel(context="different"),
        ]
        emotion = form_emotion(cluster)
        assert len(emotion.contexts) == 2

    def test_has_name(self) -> None:
        cluster = [_novel(), _novel(), _novel()]
        emotion = form_emotion(cluster)
        assert emotion.name != ""
        assert emotion.name != "unnamed"

    def test_has_description(self) -> None:
        cluster = [_novel(), _novel(), _novel()]
        emotion = form_emotion(cluster)
        assert emotion.description != ""


# ===============================================================
# Test Naming
# ===============================================================

class TestNaming:
    def test_positive_high_arousal(self) -> None:
        name = name_emotion_fallback(EmotionalVector(valence=0.8, arousal=0.9, dominance=0.5, certainty=0.5))
        assert "lumen" in name
        assert "surge" in name

    def test_negative_low_arousal(self) -> None:
        name = name_emotion_fallback(EmotionalVector(valence=-0.8, arousal=0.1, dominance=0.5, certainty=0.5))
        assert "umbra" in name
        assert "drift" in name

    def test_liminal_state(self) -> None:
        name = name_emotion_fallback(EmotionalVector(valence=0.0, arousal=0.5, dominance=0.5, certainty=0.5))
        assert "liminal" in name

    def test_high_dominance_forte(self) -> None:
        name = name_emotion_fallback(EmotionalVector(valence=0.5, arousal=0.5, dominance=0.8, certainty=0.5))
        assert "forte" in name

    def test_low_dominance_tender(self) -> None:
        name = name_emotion_fallback(EmotionalVector(valence=0.5, arousal=0.5, dominance=0.2, certainty=0.5))
        assert "tender" in name

    def test_low_certainty_neo(self) -> None:
        name = name_emotion_fallback(EmotionalVector(valence=0.5, arousal=0.5, dominance=0.5, certainty=0.1))
        assert "neo" in name

    def test_different_states_different_names(self) -> None:
        n1 = name_emotion_fallback(EmotionalVector(valence=0.8, arousal=0.9, dominance=0.8, certainty=0.8))
        n2 = name_emotion_fallback(EmotionalVector(valence=-0.8, arousal=0.1, dominance=0.2, certainty=0.1))
        assert n1 != n2


# ===============================================================
# Test Process Discovery Turn
# ===============================================================

class TestProcessDiscoveryTurn:
    def test_disabled_noop(self) -> None:
        state = default_discovery_state()
        new_state = process_discovery_turn(
            state, valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.5, stimulus="test", turn=1,
        )
        assert new_state.total_novel_detected == 0

    def test_detects_novel(self) -> None:
        state = _enabled_state()
        new_state = process_discovery_turn(
            state, valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
            intensity=0.7, stimulus="strange", turn=1,
        )
        assert new_state.total_novel_detected == 1
        assert len(new_state.novel_history) == 1

    def test_known_emotion_ignored(self) -> None:
        state = _enabled_state()
        # Joy prototype
        new_state = process_discovery_turn(
            state, valence=0.75, arousal=0.65, dominance=0.70, certainty=0.70,
            intensity=0.5, stimulus="happy", turn=1,
        )
        assert new_state.total_novel_detected == 0

    def test_cluster_forms_after_3_similar(self) -> None:
        state = _enabled_state()
        # Add 3 similar novel states
        for i in range(3):
            state = process_discovery_turn(
                state,
                valence=0.3 + i * 0.01,
                arousal=0.1 + i * 0.01,
                dominance=0.9 - i * 0.01,
                certainty=0.1 + i * 0.01,
                intensity=0.6,
                stimulus=f"strange feeling {i}",
                turn=i + 1,
            )
        assert state.total_novel_detected == 3
        assert state.total_emotions_discovered == 1
        assert len(state.discovered_emotions) == 1
        assert state.discovered_emotions[0].cluster_size == 3

    def test_buffer_trimmed(self) -> None:
        state = _enabled_state()
        # Add many scattered novel states (no clustering)
        for i in range(MAX_NOVEL_BUFFER + 10):
            state = process_discovery_turn(
                state,
                valence=-1.0 + i * 0.03,  # spread out so no clustering
                arousal=0.1,
                dominance=0.9,
                certainty=0.1,
                intensity=0.5,
                stimulus=f"test {i}",
                turn=i,
            )
        assert len(state.novel_history) <= MAX_NOVEL_BUFFER

    def test_existing_cluster_updated(self) -> None:
        state = _enabled_state()
        # First: form an emotion
        for i in range(3):
            state = process_discovery_turn(
                state, valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
                intensity=0.6, stimulus=f"ctx{i}", turn=i + 1,
            )
        assert state.total_emotions_discovered == 1
        initial_freq = state.discovered_emotions[0].frequency

        # Then: add more in same region — should update existing
        for i in range(3):
            state = process_discovery_turn(
                state, valence=0.31, arousal=0.11, dominance=0.89, certainty=0.11,
                intensity=0.6, stimulus=f"new-ctx{i}", turn=10 + i,
            )
        # Should still be 1 discovered emotion but with higher frequency
        assert state.total_emotions_discovered == 1
        assert state.discovered_emotions[0].frequency >= initial_freq


# ===============================================================
# Test Vocabulary
# ===============================================================

class TestVocabulary:
    def test_known_only(self) -> None:
        state = _enabled_state()
        vocab = get_vocabulary(state)
        assert vocab["known_count"] == len(KNOWN_PROTOTYPES)
        assert vocab["discovered_count"] == 0
        assert vocab["total"] == len(KNOWN_PROTOTYPES)

    def test_with_discovered(self) -> None:
        state = _enabled_state(discovered_emotions=[
            DiscoveredEmotion(
                name="test-emo",
                description="a test emotion",
                vector=EmotionalVector(valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1),
                body_signature=BodySignature(),
                frequency=5,
            ),
        ])
        vocab = get_vocabulary(state)
        assert vocab["discovered_count"] == 1
        assert vocab["total"] == len(KNOWN_PROTOTYPES) + 1
        assert vocab["discovered"][0]["name"] == "test-emo"


# ===============================================================
# Test Prompt
# ===============================================================

class TestPrompt:
    def test_disabled_none(self) -> None:
        state = default_discovery_state()
        assert get_discovery_prompt(state) is None

    def test_no_discoveries_none(self) -> None:
        state = _enabled_state()
        assert get_discovery_prompt(state) is None

    def test_with_discoveries(self) -> None:
        state = _enabled_state(discovered_emotions=[
            DiscoveredEmotion(
                name="glow-drift",
                description="a warm drifting feeling",
                vector=EmotionalVector(valence=0.4, arousal=0.1, dominance=0.5, certainty=0.5),
                body_signature=BodySignature(),
                frequency=3,
            ),
        ])
        prompt = get_discovery_prompt(state)
        assert prompt is not None
        assert "glow-drift" in prompt
        assert "DISCOVERED EMOTIONS" in prompt


# ===============================================================
# Test Details
# ===============================================================

class TestDetails:
    def test_disabled(self) -> None:
        state = default_discovery_state()
        details = get_discovery_details(state)
        assert details["enabled"] is False
        assert details["vocabulary_size"] == len(KNOWN_PROTOTYPES)

    def test_enabled_with_data(self) -> None:
        state = _enabled_state(
            discovered_emotions=[
                DiscoveredEmotion(
                    name="test",
                    description="test desc",
                    vector=EmotionalVector(valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1),
                    body_signature=BodySignature(),
                    frequency=5,
                ),
            ],
            novel_history=[_novel()],
        )
        details = get_discovery_details(state, novel_detected_this_turn=True)
        assert details["enabled"] is True
        assert details["discovered_count"] == 1
        assert details["novel_detected_this_turn"] is True
        assert details["novel_buffer_size"] == 1
        assert details["vocabulary_size"] == len(KNOWN_PROTOTYPES) + 1


# ===============================================================
# Test Serialization
# ===============================================================

class TestSerialization:
    def test_roundtrip_default(self) -> None:
        state = default_discovery_state()
        data = state.model_dump()
        restored = DiscoveryState(**data)
        assert restored.enabled == state.enabled

    def test_roundtrip_with_data(self) -> None:
        state = _enabled_state(
            novel_history=[_novel(v=0.3, a=0.1)],
            discovered_emotions=[
                DiscoveredEmotion(
                    name="test-emo",
                    description="test",
                    vector=EmotionalVector(valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1),
                    body_signature=BodySignature(tension=0.8),
                    contexts=["ctx1", "ctx2"],
                    frequency=5,
                    cluster_size=3,
                ),
            ],
            total_novel_detected=10,
            total_emotions_discovered=1,
        )
        data = state.model_dump()
        restored = DiscoveryState(**data)
        assert restored.enabled is True
        assert len(restored.novel_history) == 1
        assert len(restored.discovered_emotions) == 1
        assert restored.discovered_emotions[0].name == "test-emo"
        assert restored.discovered_emotions[0].body_signature.tension == 0.8
        assert restored.total_novel_detected == 10


# ===============================================================
# Test Averages
# ===============================================================

class TestAverages:
    def test_average_vector(self) -> None:
        states = [
            _novel(v=0.2, a=0.4, d=0.6, c=0.8),
            _novel(v=0.4, a=0.6, d=0.8, c=0.2),
        ]
        avg = _average_vector(states)
        assert avg.valence == pytest.approx(0.3, abs=0.001)
        assert avg.arousal == pytest.approx(0.5, abs=0.001)

    def test_average_body(self) -> None:
        states = [
            _novel(tension=0.8, energy=0.2, openness=0.6, warmth=0.4),
            _novel(tension=0.4, energy=0.6, openness=0.2, warmth=0.8),
        ]
        avg = _average_body(states)
        assert avg.tension == pytest.approx(0.6, abs=0.001)
        assert avg.energy == pytest.approx(0.4, abs=0.001)

    def test_empty_states(self) -> None:
        avg = _average_vector([])
        assert avg.valence == 0.0
        avg_body = _average_body([])
        assert avg_body.tension == 0.5


# ===============================================================
# Test Full Flow
# ===============================================================

class TestFullFlow:
    def test_complete_discovery_flow(self) -> None:
        """Simulate a complete flow: detect, cluster, discover, vocabulary."""
        state = _enabled_state()

        # Phase 1: Add novel states (all similar)
        for i in range(5):
            state = process_discovery_turn(
                state,
                valence=0.3 + i * 0.005,
                arousal=0.1 + i * 0.005,
                dominance=0.9 - i * 0.005,
                certainty=0.1 + i * 0.005,
                intensity=0.6,
                stimulus=f"strange feeling about existence {i}",
                turn=i + 1,
                body_tension=0.7,
                body_warmth=0.3,
            )

        # Should have discovered an emotion
        assert state.total_emotions_discovered >= 1
        assert len(state.discovered_emotions) >= 1

        # Check the discovered emotion has sensible properties
        emotion = state.discovered_emotions[0]
        assert emotion.name != "unnamed"
        assert emotion.description != ""
        assert emotion.frequency >= 3
        assert emotion.vector.valence > 0

        # Check vocabulary
        vocab = get_vocabulary(state)
        assert vocab["discovered_count"] >= 1
        assert vocab["total"] > len(KNOWN_PROTOTYPES)

        # Check prompt
        prompt = get_discovery_prompt(state)
        assert prompt is not None
        assert emotion.name in prompt

    def test_disabled_flow_noop(self) -> None:
        state = default_discovery_state()
        for i in range(10):
            state = process_discovery_turn(
                state, valence=0.3, arousal=0.1, dominance=0.9, certainty=0.1,
                intensity=0.5, stimulus="test", turn=i,
            )
        assert state.total_novel_detected == 0
        assert state.discovered_emotions == []


# ===============================================================
# Test Known Prototypes
# ===============================================================

class TestKnownPrototypes:
    def test_all_18_prototypes(self) -> None:
        assert len(KNOWN_PROTOTYPES) == 18

    def test_matches_generator_prototypes(self) -> None:
        """Verify our prototype copy matches generator.py."""
        # Spot check a few
        assert KNOWN_PROTOTYPES["joy"] == (0.75, 0.65, 0.70, 0.70)
        assert KNOWN_PROTOTYPES["fear"] == (-0.75, 0.85, 0.15, 0.15)
        assert KNOWN_PROTOTYPES["neutral"] == (0.00, 0.30, 0.50, 0.50)
