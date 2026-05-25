"""Tests for RESIDUUM F4.4 — Pipeline wiring of V2 granular steering.

End-to-end smoke against /research/chat verifying the new SteeringDetails
fields (version, fraction_cap) are populated correctly. We exercise the
"no v2 available" path (default Ollama mock) and confirm:
  - fraction_cap reflects the session mode (default vs Lite vs Raw vs Extreme)
  - version is "none" when steering is gated out (no supports_steering)

Activating the V2 path end-to-end requires a real Transformers provider
with a probe library, which is covered by the unit tests in
test_residuum_steering_hook_v2.py. The integration test here only verifies
the schema wiring, not the V2 forward hook behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from pathos.engine.appraiser import AppraisalResult
from pathos.engine.steering import (
    MAX_STEERING_FRACTION_DEFAULT,
    MAX_STEERING_FRACTION_EXTREME,
    MAX_STEERING_FRACTION_LITE,
    MAX_STEERING_FRACTION_RAW,
)
from pathos.main import app, state_manager
from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
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
    # By default a mocked Ollama-style provider does NOT support steering and
    # has no steerable_model, so the steering hook is gated out -> version=none.
    mock.supports_steering = False
    return mock


@pytest.fixture(autouse=True)
def _reset_sessions() -> None:
    state_manager._sessions.clear()


def _post_research(client: TestClient, session_id: str, payload_extra: dict | None = None) -> dict:
    body: dict = {"message": "Hello", "session_id": session_id}
    if payload_extra:
        body.update(payload_extra)
    resp = client.post("/research/chat", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestSteeringDetailsWiring:
    """SteeringDetails always carries the new F4.4 fields. The values reflect
    the session mode regardless of whether steering actually ran this turn."""

    @patch("pathos.main.appraise")
    def test_default_mode_reports_fraction_cap_default(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            data = _post_research(client, "f44-default")
        steering = data["steering"]
        # No steering ran (no supports_steering, no probe_library) -> version 'none'.
        assert steering["version"] == "none"
        # Default mode -> fraction_cap = 0.10
        assert steering["fraction_cap"] == pytest.approx(MAX_STEERING_FRACTION_DEFAULT)

    @patch("pathos.main.appraise")
    def test_steering_status_propagates_no_vectors_when_neither_v1_nor_v2(
        self, mock_appraise: AsyncMock,
    ) -> None:
        # Engine has no cached vectors AND probe_library is None -> 'no_vectors'.
        mock_appraise.return_value = MOCK_APPRAISAL
        with patch("pathos.main.llm_provider", _mock_llm()):
            with patch("pathos.main.probe_library", None):
                client = TestClient(app, raise_server_exceptions=True)
                data = _post_research(client, "f44-no-vectors")
        # Steering is enabled by default; gate fails at engine.is_ready + no library.
        assert data["steering"]["status"] in {"no_vectors", "provider_unsupported"}
        assert data["steering"]["version"] == "none"

    @patch("pathos.main.appraise")
    def test_fraction_cap_reflects_lite_mode(
        self, mock_appraise: AsyncMock,
    ) -> None:
        # Set session to lite mode AFTER creation so the cap resolution picks it up.
        mock_appraise.return_value = MOCK_APPRAISAL
        sid = "f44-lite"
        # Pre-create session with lite_mode=True.
        session = state_manager.get_session(sid)
        session.lite_mode = True
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            data = _post_research(client, sid)
        assert data["steering"]["fraction_cap"] == pytest.approx(MAX_STEERING_FRACTION_LITE)

    @patch("pathos.main.appraise")
    def test_fraction_cap_reflects_raw_mode(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        sid = "f44-raw"
        session = state_manager.get_session(sid)
        session.raw_mode = True
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            data = _post_research(client, sid)
        assert data["steering"]["fraction_cap"] == pytest.approx(MAX_STEERING_FRACTION_RAW)

    @patch("pathos.main.appraise")
    def test_fraction_cap_reflects_extreme_mode(
        self, mock_appraise: AsyncMock,
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        sid = "f44-extreme"
        session = state_manager.get_session(sid)
        session.extreme_mode = True
        with patch("pathos.main.llm_provider", _mock_llm()):
            client = TestClient(app, raise_server_exceptions=True)
            data = _post_research(client, sid)
        assert data["steering"]["fraction_cap"] == pytest.approx(MAX_STEERING_FRACTION_EXTREME)
