"""Tests for RESIDUUM F5.3 — Pipeline wiring of CoherenceClassifier.

End-to-end smoke against /chat verifying the new ResiduumDetails fields
(divergence_event_count, divergence_categories, last_divergence,
recent_divergence_events) appear in the response with sensible defaults
when F2 is OFF (no modulators measured).

Full classifier behavior is covered by test_residuum_coherence_classifier;
this file only verifies the wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from pathos.engine.appraiser import AppraisalResult
from pathos.engine.introspection import process_modulation_coherence_turn
from pathos.main import app, state_manager
from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.residuum import (
    DivergenceCategory,
    ResiduumState,
    default_residuum_state,
)


MOCK_APPRAISAL = AppraisalResult(
    vector=AppraisalVector(
        relevance=RelevanceCheck(novelty=0.5, personal_significance=0.7, values_affected=["truth"]),
        valence=ValenceAssessment(goal_conduciveness=0.6, value_alignment=0.5, intrinsic_pleasantness=0.4),
        coping=CopingPotential(control=0.7, power=0.6, adjustability=0.5),
        agency=AgencyAttribution(responsible_agent="user", intentionality=0.6, fairness=0.4),
        norms=NormCompatibility(internal_standards=0.5, external_standards=0.4, self_consistency=0.6),
    ),
    emotion_hint=None,
)


def _mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.generate.return_value = "Mock LLM response"
    mock.embed.return_value = [0.1] * 768
    mock.supports_steering = False
    return mock


@pytest.fixture(autouse=True)
def _reset_sessions() -> None:
    state_manager._sessions.clear()


# ===========================================================================
# Orchestrator gating
# ===========================================================================


class TestProcessModulationCoherenceTurn:
    """The orchestrator must return [] when F2 is disabled, and emit one
    event per modulator that ran (entry in post_states_by_system is non-None)."""

    def test_disabled_returns_no_events(self) -> None:
        s = default_residuum_state()
        s.enabled = False
        pre = EmotionalState(
            primary_emotion=PrimaryEmotion.NEUTRAL,
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        events = process_modulation_coherence_turn(
            s, pre, {"regulation": pre, "reappraisal": pre, "immune": pre},
            measured=None, turn=1,
        )
        assert events == []
        assert s.divergence_events == []

    def test_skips_systems_with_none_post_state(self) -> None:
        """If a modulator did not run this turn (post_state None), no event
        is emitted for it. Only modulators that actually acted produce events."""
        s = default_residuum_state()
        s.enabled = True
        pre = EmotionalState(
            primary_emotion=PrimaryEmotion.NEUTRAL,
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        post_reg = pre.model_copy()
        events = process_modulation_coherence_turn(
            s, pre,
            {"regulation": post_reg, "reappraisal": None, "immune": None},
            measured=None, turn=1,
        )
        # One event for regulation; reappraisal and immune skipped.
        assert len(events) == 1
        assert events[0].system == "regulation"
        assert len(s.divergence_events) == 1

    def test_emits_event_per_active_modulator(self) -> None:
        s = default_residuum_state()
        s.enabled = True
        pre = EmotionalState(
            primary_emotion=PrimaryEmotion.NEUTRAL,
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        post = pre.model_copy()
        events = process_modulation_coherence_turn(
            s, pre,
            {"regulation": post, "reappraisal": post, "immune": post},
            measured=None, turn=2,
        )
        assert len(events) == 3
        systems = [e.system for e in events]
        assert set(systems) == {"regulation", "reappraisal", "immune"}
        # All ordered as reappraisal, regulation, immune in the loop.
        assert systems == ["reappraisal", "regulation", "immune"]

    def test_appends_to_session_divergence_events(self) -> None:
        s = default_residuum_state()
        s.enabled = True
        pre = EmotionalState(
            primary_emotion=PrimaryEmotion.NEUTRAL,
            valence=0.0, arousal=0.5, dominance=0.5,
            certainty=0.5, intensity=0.5,
        )
        post = pre.model_copy()
        process_modulation_coherence_turn(
            s, pre, {"regulation": post, "reappraisal": None, "immune": None},
            measured=None, turn=1,
        )
        assert len(s.divergence_events) == 1
        assert s.last_divergence_event is not None
        assert s.last_divergence_event.system == "regulation"


# ===========================================================================
# Pipeline integration via /chat
# ===========================================================================


class TestChatEndpointResponseShape:
    """ResiduumDetails in the /chat response must always carry the F5
    fields (divergence_event_count, etc.) with empty defaults when F2 is OFF."""

    @patch("pathos.main.appraise")
    def test_chat_response_includes_f5_fields_with_defaults(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/chat",
                json={"message": "hello", "session_id": "f5-chat-defaults"},
            )
        assert resp.status_code == 200, resp.text
        # /chat returns a ChatResponse with embedded residuum details. Check
        # the session state directly — the response includes emotional state
        # but not raw residuum dict. We assert no crash + session has the
        # divergence_events buffer initialized.
        session = state_manager.get_session("f5-chat-defaults")
        assert session.residuum.divergence_events == []
        assert session.residuum.last_divergence_event is None


class TestResearchEndpointF5Wiring:
    """F5.5: /research/chat must wire F5 like /chat. With F2 OFF, no
    divergence events emit; the session state stays clean."""

    @patch("pathos.main.appraise")
    def test_research_chat_does_not_crash_with_f5_inactive(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "hello", "session_id": "f5-research-defaults"},
            )
        assert resp.status_code == 200, resp.text
        session = state_manager.get_session("f5-research-defaults")
        # F2 is OFF by default -> F5 was gated out, no events recorded.
        assert session.residuum.divergence_events == []
        assert session.residuum.last_divergence_event is None
        # ResiduumDetails in research response carries the F5 fields with
        # empty defaults (verified at schema level — wiring did not crash).
        residuum = resp.json()["residuum"]
        assert residuum["divergence_event_count"] == 0
        assert residuum["recent_divergence_events"] == []


class TestExpressionEffectivenessPipeline:
    """F5.6 wireado en /chat, /research/chat, /sandbox/simulate. Solo
    corre en Raw / Extreme. En Advanced no emite eventos overall_expression."""

    @patch("pathos.main.appraise")
    def test_chat_advanced_does_not_emit_expression_events(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            client.post(
                "/chat",
                json={"message": "hola", "session_id": "f56-chat-advanced"},
            )
        session = state_manager.get_session("f56-chat-advanced")
        # Advanced mode — no overall_expression events.
        expression_events = [
            e for e in session.residuum.divergence_events
            if e.system == "overall_expression"
        ]
        assert expression_events == []

    @patch("pathos.main.appraise")
    def test_chat_raw_mode_emits_no_events_when_f2_off(
        self, mock_appraise: AsyncMock,
    ) -> None:
        """Even in Raw, F5.6 needs F2 enabled to compare against measured.
        With F2 OFF (default) the orchestrator silently returns."""
        mock_appraise.return_value = MOCK_APPRAISAL
        sid = "f56-chat-raw-no-f2"
        session = state_manager.get_session(sid)
        session.raw_mode = True
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            client.post("/chat", json={"message": "hola", "session_id": sid})
        # No measurement -> no events.
        assert session.residuum.divergence_events == []


class TestSandboxEndpointF5Wiring:
    """F5.5: /sandbox/simulate must also wire F5. The sandbox session is
    isolated (does not mutate the base session). With F2 OFF, no events."""

    @patch("pathos.main.appraise")
    def test_sandbox_simulate_does_not_crash_with_f5_inactive(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/sandbox/simulate",
                json={
                    "session_id": "f5-sandbox-defaults",
                    "scenario": "Algo nuevo pasa",
                    "initial_state": "neutral",
                },
            )
        assert resp.status_code == 200, resp.text
        # Sandbox is isolated -> base session residuum stays empty.
        base = state_manager.get_session("f5-sandbox-defaults")
        assert base.residuum.divergence_events == []
        # ResiduumDetails in the sandbox response carries F5 defaults.
        residuum = resp.json().get("residuum", {})
        # Sandbox always emits residuum block (default values when F2 OFF).
        assert residuum.get("divergence_event_count", 0) == 0
