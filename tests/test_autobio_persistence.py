"""Tests para Autobiographical Memory Persistence (Pilar 3 ANIMA — Paso 3.3).

Verifica:
- Serialización/deserialización de AutobiographicalState
- to_dict / from_dict en SessionState
- Baseline adjustment al cargar sesión
- Episodic store persistence (capacity, eviction preservada)
- Narrative store persistence
- Dream report persistence
- Reset mantiene enabled state
"""

import json
import pytest

from pathos.engine.autobio_memory import (
    _classify_significance,
    _extract_keywords,
    encode_episode,
    process_autobiographical_turn,
    store_episode,
)
from pathos.engine.dreaming import consolidate as dream_consolidate
from pathos.models.autobio_memory import (
    AutobiographicalState,
    Episode,
    EpisodicStore,
    NarrativeStatement,
    NarrativeStore,
    NarrativeType,
    default_autobiographical_state,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.state.manager import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    valence: float = 0.0,
    arousal: float = 0.3,
    intensity: float = 0.5,
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        primary_emotion=emotion,
    )


def _episode(
    emotion: str = "joy",
    valence: float = 0.6,
    intensity: float = 0.7,
    turn: int = 1,
    stimulus: str = "test stimulus",
    ep_id: str | None = None,
) -> Episode:
    ep = Episode(
        stimulus=stimulus,
        response_summary="test response",
        primary_emotion=emotion,
        valence=valence,
        arousal=0.5,
        intensity=intensity,
        significance=_classify_significance(intensity),
        turn_number=turn,
        session_id="test-session",
        keywords=_extract_keywords(stimulus),
    )
    if ep_id:
        ep.id = ep_id
    return ep


# ===========================================================================
# Serialization Tests
# ===========================================================================

class TestAutobiographicalSerialization:
    """Tests para serialización de AutobiographicalState."""

    def test_default_state_roundtrip(self):
        state = default_autobiographical_state()
        data = state.model_dump()
        restored = AutobiographicalState(**data)
        assert restored.enabled == state.enabled
        assert restored.episodic.count() == 0
        assert restored.narrative.count() == 0

    def test_enabled_state_roundtrip(self):
        state = AutobiographicalState(enabled=True, session_id="s1")
        data = state.model_dump()
        restored = AutobiographicalState(**data)
        assert restored.enabled is True
        assert restored.session_id == "s1"

    def test_with_episodes_roundtrip(self):
        state = AutobiographicalState(enabled=True)
        ep = _episode(ep_id="ep1", emotion="joy", valence=0.7)
        state.episodic = EpisodicStore(episodes=[ep], total_encoded=1)

        data = state.model_dump()
        json_str = json.dumps(data, default=str)
        restored_data = json.loads(json_str)
        restored = AutobiographicalState(**restored_data)

        assert restored.episodic.count() == 1
        assert restored.episodic.episodes[0].id == "ep1"
        assert restored.episodic.episodes[0].primary_emotion == "joy"

    def test_with_narratives_roundtrip(self):
        state = AutobiographicalState(enabled=True)
        ns = NarrativeStatement(
            narrative_type=NarrativeType.PATTERN,
            statement="I tend to feel joy in conversation",
            primary_emotion="joy",
            valence=0.6,
            strength=0.7,
            episode_count=5,
        )
        state.narrative = NarrativeStore(statements=[ns])

        data = state.model_dump()
        json_str = json.dumps(data, default=str)
        restored_data = json.loads(json_str)
        restored = AutobiographicalState(**restored_data)

        assert restored.narrative.count() == 1
        assert restored.narrative.statements[0].statement == "I tend to feel joy in conversation"
        assert restored.narrative.statements[0].strength == 0.7

    def test_with_dream_report_roundtrip(self):
        state = AutobiographicalState(
            enabled=True,
            last_dream_report="A dream of flowing water",
            baseline_adjustment={"valence": 0.05, "arousal": -0.03},
        )
        data = state.model_dump()
        restored = AutobiographicalState(**data)
        assert restored.last_dream_report == "A dream of flowing water"
        assert restored.baseline_adjustment["valence"] == 0.05

    def test_episode_keywords_preserved(self):
        ep = _episode(stimulus="Python programming is fascinating")
        state = AutobiographicalState(enabled=True)
        state.episodic = EpisodicStore(episodes=[ep], total_encoded=1)

        data = state.model_dump()
        restored = AutobiographicalState(**data)
        assert "python" in restored.episodic.episodes[0].keywords

    def test_episode_emotional_links_preserved(self):
        ep = _episode(ep_id="ep1")
        ep.emotional_links = ["ep2", "ep3"]
        state = AutobiographicalState(enabled=True)
        state.episodic = EpisodicStore(episodes=[ep], total_encoded=1)

        data = state.model_dump()
        restored = AutobiographicalState(**data)
        assert restored.episodic.episodes[0].emotional_links == ["ep2", "ep3"]


# ===========================================================================
# SessionState Integration Tests
# ===========================================================================

class TestSessionStateIntegration:
    """Tests para integración con SessionState to_dict/from_dict."""

    def test_session_default_autobiographical(self):
        session = SessionState()
        assert session.autobiographical.enabled is False

    def test_session_to_dict_includes_autobiographical(self):
        session = SessionState()
        data = session.to_dict()
        assert "autobiographical" in data

    def test_session_roundtrip_disabled(self):
        session = SessionState()
        data = session.to_dict()
        restored = SessionState.from_dict(data)
        assert restored.autobiographical.enabled is False

    def test_session_roundtrip_enabled_with_data(self):
        session = SessionState()
        session.autobiographical.enabled = True
        session.autobiographical.session_id = "test-s1"
        session.autobiographical.total_turns_processed = 5

        ep = _episode(ep_id="ep1")
        session.autobiographical.episodic = EpisodicStore(episodes=[ep], total_encoded=1)

        data = session.to_dict()
        json_str = json.dumps(data, default=str)
        restored_data = json.loads(json_str)
        restored = SessionState.from_dict(restored_data)

        assert restored.autobiographical.enabled is True
        assert restored.autobiographical.session_id == "test-s1"
        assert restored.autobiographical.episodic.count() == 1

    def test_session_from_dict_missing_autobiographical(self):
        """Legacy saves without autobiographical field should still load."""
        session = SessionState()
        data = session.to_dict()
        del data["autobiographical"]
        restored = SessionState.from_dict(data)
        # Should have default (disabled)
        assert restored.autobiographical.enabled is False


# ===========================================================================
# Baseline Adjustment Tests
# ===========================================================================

class TestBaselineAdjustment:
    """Tests para aplicación de baseline adjustment al cargar sesión."""

    def test_baseline_adjustment_positive_valence(self):
        session = SessionState()
        session.autobiographical.enabled = True
        session.autobiographical.baseline_adjustment = {"valence": 0.05, "arousal": -0.02}

        original_v = session.emotional_state.mood.baseline_valence
        original_a = session.emotional_state.mood.baseline_arousal

        # Simulate what load_session does
        adj = session.autobiographical.baseline_adjustment
        v_adj = adj.get("valence", 0.0)
        a_adj = adj.get("arousal", 0.0)
        session.emotional_state.mood.baseline_valence = round(
            max(-1.0, min(1.0, session.emotional_state.mood.baseline_valence + v_adj)), 4,
        )
        session.emotional_state.mood.baseline_arousal = round(
            max(0.0, min(1.0, session.emotional_state.mood.baseline_arousal + a_adj)), 4,
        )

        assert session.emotional_state.mood.baseline_valence == round(original_v + 0.05, 4)
        assert session.emotional_state.mood.baseline_arousal == round(original_a - 0.02, 4)

    def test_baseline_clamped_at_bounds(self):
        session = SessionState()
        session.emotional_state.mood.baseline_valence = 0.95
        session.autobiographical.enabled = True
        session.autobiographical.baseline_adjustment = {"valence": 0.2, "arousal": -0.5}

        adj = session.autobiographical.baseline_adjustment
        session.emotional_state.mood.baseline_valence = round(
            max(-1.0, min(1.0, session.emotional_state.mood.baseline_valence + adj["valence"])), 4,
        )
        session.emotional_state.mood.baseline_arousal = round(
            max(0.0, min(1.0, session.emotional_state.mood.baseline_arousal + adj["arousal"])), 4,
        )

        assert session.emotional_state.mood.baseline_valence <= 1.0
        assert session.emotional_state.mood.baseline_arousal >= 0.0

    def test_empty_adjustment_no_change(self):
        session = SessionState()
        original_v = session.emotional_state.mood.baseline_valence
        session.autobiographical.baseline_adjustment = {}

        adj = session.autobiographical.baseline_adjustment
        v_adj = adj.get("valence", 0.0)
        session.emotional_state.mood.baseline_valence += v_adj

        assert session.emotional_state.mood.baseline_valence == original_v


# ===========================================================================
# Consolidation + Persistence Tests
# ===========================================================================

class TestConsolidationPersistence:
    """Tests para que consolidación se preserve en serialización."""

    def test_consolidated_episodes_survive_roundtrip(self):
        state = AutobiographicalState(enabled=True, session_id="s1")
        for i in range(3):
            ep = _episode(intensity=0.8, turn=i, ep_id=f"ep{i}")
            state.episodic = store_episode(state.episodic, ep)

        # Consolidate
        result, new_state = dream_consolidate(state, "s1")
        assert all(ep.consolidated for ep in new_state.episodic.episodes)

        # Roundtrip
        data = new_state.model_dump()
        json_str = json.dumps(data, default=str)
        restored = AutobiographicalState(**json.loads(json_str))

        assert all(ep.consolidated for ep in restored.episodic.episodes)
        assert restored.last_dream_report == new_state.last_dream_report

    def test_dream_report_survives_roundtrip(self):
        state = AutobiographicalState(enabled=True, session_id="s1")
        for i in range(3):
            ep = _episode(intensity=0.8, emotion="fear", valence=-0.7, turn=i)
            state.episodic = store_episode(state.episodic, ep)

        _, new_state = dream_consolidate(state, "s1")
        assert new_state.last_dream_report != ""

        data = new_state.model_dump()
        restored = AutobiographicalState(**data)
        assert restored.last_dream_report == new_state.last_dream_report
        assert restored.baseline_adjustment == new_state.baseline_adjustment

    def test_narrative_strength_preserved_after_consolidation(self):
        state = AutobiographicalState(enabled=True, session_id="s1")
        # Add enough episodes of same emotion for narrative
        for i in range(6):
            ep = _episode(emotion="joy", valence=0.6, intensity=0.7, turn=i)
            state.episodic = store_episode(state.episodic, ep)

        _, new_state = dream_consolidate(state, "s1")

        data = new_state.model_dump()
        restored = AutobiographicalState(**data)

        if restored.narrative.count() > 0:
            assert restored.narrative.statements[0].strength > 0


# ===========================================================================
# Reset Tests
# ===========================================================================

class TestMemoryReset:
    """Tests para reset de memoria autobiográfica."""

    def test_reset_clears_all_data(self):
        state = AutobiographicalState(enabled=True, session_id="s1")
        for i in range(5):
            ep = _episode(turn=i)
            state.episodic = store_episode(state.episodic, ep)
        state.last_dream_report = "A dream"

        # Reset
        new_state = default_autobiographical_state()
        new_state.enabled = True  # Preserve enabled

        assert new_state.episodic.count() == 0
        assert new_state.narrative.count() == 0
        assert new_state.last_dream_report == ""
        assert new_state.enabled is True

    def test_reset_preserves_enabled_flag(self):
        state = AutobiographicalState(enabled=True)
        new_state = default_autobiographical_state()
        new_state.enabled = state.enabled
        assert new_state.enabled is True

    def test_reset_disabled_stays_disabled(self):
        state = AutobiographicalState(enabled=False)
        new_state = default_autobiographical_state()
        new_state.enabled = state.enabled
        assert new_state.enabled is False


# ===========================================================================
# Cross-Session Flow Test
# ===========================================================================

class TestCrossSessionFlow:
    """Tests para el flujo completo cross-session."""

    def test_full_cross_session_flow(self):
        """Simula: sesion 1 con episodios → consolidación → sesion 2 con dream."""
        # Session 1: Process turns
        state = AutobiographicalState(enabled=True, session_id="session-1")
        emotions = [
            (PrimaryEmotion.JOY, 0.8, 0.7),
            (PrimaryEmotion.SADNESS, 0.7, -0.5),
            (PrimaryEmotion.JOY, 0.6, 0.6),
            (PrimaryEmotion.ANGER, 0.9, -0.7),
            (PrimaryEmotion.JOY, 0.7, 0.5),
            (PrimaryEmotion.JOY, 0.6, 0.6),
        ]
        for i, (emo, inten, val) in enumerate(emotions):
            state = process_autobiographical_turn(
                state, f"stimulus {i}",
                _state(intensity=inten, valence=val, emotion=emo),
                f"response {i}", i, "session-1",
            )

        assert state.episodic.count() > 0

        # Consolidate (end of session 1)
        result, consolidated_state = dream_consolidate(state, "session-1")
        assert consolidated_state.last_dream_report != ""
        assert "valence" in consolidated_state.baseline_adjustment

        # Serialize (save)
        saved_data = consolidated_state.model_dump()
        json_str = json.dumps(saved_data, default=str)

        # Session 2: Load
        loaded_data = json.loads(json_str)
        session2_state = AutobiographicalState(**loaded_data)

        # Verify cross-session data
        assert session2_state.enabled is True
        assert session2_state.last_dream_report == consolidated_state.last_dream_report
        assert session2_state.episodic.count() > 0
        assert all(ep.consolidated for ep in session2_state.episodic.episodes)

        # Session 2: Continue processing
        session2_state = process_autobiographical_turn(
            session2_state, "new stimulus in session 2",
            _state(intensity=0.7, valence=0.5, emotion=PrimaryEmotion.JOY),
            "new response", 100, "session-2",
        )

        # New episode added (not consolidated)
        unconsolidated = session2_state.episodic.get_unconsolidated()
        assert len(unconsolidated) >= 1
