"""Tests para Calibration Mode."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from pathos.engine.appraiser import AppraisalResult
from pathos.engine.calibration import apply_calibration, compute_calibration_profile
from pathos.main import app, state_manager
from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
)
from pathos.models.calibration import (
    CalibrationProfile,
    CalibrationResult,
    CalibrationScenario,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion


# --- Helper factories ---


def _result(
    expected_valence: float = 0.5,
    expected_arousal: float = 0.5,
    expected_intensity: float = 0.5,
    system_valence: float = 0.3,
    system_arousal: float = 0.4,
    system_intensity: float = 0.3,
    expected_emotion: PrimaryEmotion = PrimaryEmotion.JOY,
    system_emotion: PrimaryEmotion = PrimaryEmotion.JOY,
) -> CalibrationResult:
    return CalibrationResult(
        scenario=CalibrationScenario(
            stimulus="test",
            expected_emotion=expected_emotion,
            expected_valence=expected_valence,
            expected_arousal=expected_arousal,
            expected_intensity=expected_intensity,
        ),
        system_emotion=system_emotion,
        system_valence=system_valence,
        system_arousal=system_arousal,
        system_intensity=system_intensity,
        valence_delta=round(expected_valence - system_valence, 4),
        arousal_delta=round(expected_arousal - system_arousal, 4),
        intensity_delta=round(expected_intensity - system_intensity, 4),
        emotion_match=expected_emotion == system_emotion,
    )


def _state(
    valence: float = 0.3,
    arousal: float = 0.4,
    intensity: float = 0.5,
) -> EmotionalState:
    return EmotionalState(valence=valence, arousal=arousal, intensity=intensity)


# --- compute_calibration_profile tests ---


class TestComputeProfile:
    def test_empty_results(self) -> None:
        profile = compute_calibration_profile([])
        assert profile.scenarios_used == 0
        assert profile.valence_offset == 0.0
        assert profile.arousal_scale == 1.0
        assert profile.intensity_scale == 1.0

    def test_single_result_positive_offset(self) -> None:
        """User expects more positive than system -> positive valence offset."""
        result = _result(expected_valence=0.7, system_valence=0.3)
        profile = compute_calibration_profile([result])
        assert profile.valence_offset > 0
        assert profile.scenarios_used == 1

    def test_single_result_negative_offset(self) -> None:
        """User expects more negative than system -> negative valence offset."""
        result = _result(expected_valence=-0.3, system_valence=0.3)
        profile = compute_calibration_profile([result])
        assert profile.valence_offset < 0

    def test_arousal_scale_up(self) -> None:
        """User expects more arousal -> scale > 1."""
        result = _result(expected_arousal=0.8, system_arousal=0.4)
        profile = compute_calibration_profile([result])
        assert profile.arousal_scale > 1.0

    def test_arousal_scale_down(self) -> None:
        """User expects less arousal -> scale < 1."""
        result = _result(expected_arousal=0.3, system_arousal=0.6)
        profile = compute_calibration_profile([result])
        assert profile.arousal_scale < 1.0

    def test_intensity_scale(self) -> None:
        """User expects more intensity -> scale > 1."""
        result = _result(expected_intensity=0.8, system_intensity=0.4)
        profile = compute_calibration_profile([result])
        assert profile.intensity_scale > 1.0

    def test_emotion_accuracy(self) -> None:
        """Accuracy reflects emotion matching."""
        r1 = _result(expected_emotion=PrimaryEmotion.JOY, system_emotion=PrimaryEmotion.JOY)
        r2 = _result(expected_emotion=PrimaryEmotion.ANGER, system_emotion=PrimaryEmotion.SADNESS)
        profile = compute_calibration_profile([r1, r2])
        assert profile.emotion_accuracy == 0.5

    def test_multiple_results_average(self) -> None:
        """Multiple results should average the deltas."""
        r1 = _result(expected_valence=0.6, system_valence=0.4)
        r2 = _result(expected_valence=0.8, system_valence=0.4)
        profile = compute_calibration_profile([r1, r2])
        assert profile.scenarios_used == 2
        assert profile.valence_offset > 0

    def test_offset_clamped(self) -> None:
        """Valence offset should be clamped to [-0.5, 0.5]."""
        result = _result(expected_valence=1.0, system_valence=-1.0)
        profile = compute_calibration_profile([result])
        assert -0.5 <= profile.valence_offset <= 0.5

    def test_scale_clamped(self) -> None:
        """Scales should be clamped to [0.5, 2.0]."""
        result = _result(expected_arousal=1.0, system_arousal=0.1)
        profile = compute_calibration_profile([result])
        assert 0.5 <= profile.arousal_scale <= 2.0


# --- apply_calibration tests ---


class TestApplyCalibration:
    def test_no_calibration(self) -> None:
        """Profile with 0 scenarios should not modify state."""
        state = _state(valence=0.3, arousal=0.4, intensity=0.5)
        profile = CalibrationProfile()
        result = apply_calibration(state, profile)
        assert result.valence == state.valence
        assert result.arousal == state.arousal
        assert result.intensity == state.intensity

    def test_valence_offset_applied(self) -> None:
        state = _state(valence=0.3)
        profile = CalibrationProfile(valence_offset=0.2, scenarios_used=1)
        result = apply_calibration(state, profile)
        assert result.valence == 0.5

    def test_arousal_scale_applied(self) -> None:
        state = _state(arousal=0.4)
        profile = CalibrationProfile(arousal_scale=1.5, scenarios_used=1)
        result = apply_calibration(state, profile)
        assert result.arousal == 0.6

    def test_intensity_scale_applied(self) -> None:
        state = _state(intensity=0.4)
        profile = CalibrationProfile(intensity_scale=1.5, scenarios_used=1)
        result = apply_calibration(state, profile)
        assert result.intensity == 0.6

    def test_values_clamped(self) -> None:
        """Calibrated values should respect ranges."""
        state = _state(valence=0.8, arousal=0.9, intensity=0.9)
        profile = CalibrationProfile(
            valence_offset=0.5, arousal_scale=2.0, intensity_scale=2.0, scenarios_used=1,
        )
        result = apply_calibration(state, profile)
        assert -1 <= result.valence <= 1
        assert 0 <= result.arousal <= 1
        assert 0 <= result.intensity <= 1

    def test_other_fields_preserved(self) -> None:
        """Calibration should not touch dominance, certainty, emotion, etc."""
        state = _state()
        state = state.model_copy(update={
            "dominance": 0.7, "certainty": 0.8,
            "primary_emotion": PrimaryEmotion.JOY,
        })
        profile = CalibrationProfile(valence_offset=0.1, scenarios_used=1)
        result = apply_calibration(state, profile)
        assert result.dominance == 0.7
        assert result.certainty == 0.8
        assert result.primary_emotion == PrimaryEmotion.JOY


# --- Endpoint integration tests ---


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


@pytest.fixture(autouse=True)
def _reset_sessions() -> None:
    state_manager._sessions.clear()


def _mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.generate.return_value = "Mock response"
    mock.embed.return_value = [0.1] * 768
    return mock


class TestCalibrationEndpoints:
    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_submit_scenario(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/calibration/scenario",
                params={"session_id": "cal-1"},
                json={
                    "stimulus": "Someone helped me when I was lost",
                    "expected_emotion": "gratitude",
                    "expected_valence": 0.7,
                    "expected_arousal": 0.4,
                    "expected_intensity": 0.6,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "system_emotion" in data
        assert "valence_delta" in data
        assert "emotion_match" in data

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_apply_profile(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            # Submit scenario
            client.post(
                "/calibration/scenario",
                params={"session_id": "cal-2"},
                json={
                    "stimulus": "Test scenario",
                    "expected_emotion": "joy",
                    "expected_valence": 0.8,
                    "expected_arousal": 0.6,
                    "expected_intensity": 0.7,
                },
            )
            # Apply
            resp = client.post("/calibration/apply", params={"session_id": "cal-2"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenarios_used"] == 1
        assert "valence_offset" in data
        assert "arousal_scale" in data

    def test_apply_without_scenarios_fails(self) -> None:
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/calibration/apply", params={"session_id": "empty"})
        assert resp.status_code == 400

    def test_get_profile_default(self) -> None:
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/calibration/profile/new-session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenarios_used"] == 0
        assert data["valence_offset"] == 0.0

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_reset_calibration(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            # Submit + apply
            client.post(
                "/calibration/scenario",
                params={"session_id": "cal-3"},
                json={
                    "stimulus": "Test",
                    "expected_emotion": "joy",
                    "expected_valence": 0.8,
                    "expected_arousal": 0.6,
                    "expected_intensity": 0.7,
                },
            )
            client.post("/calibration/apply", params={"session_id": "cal-3"})

            # Reset
            resp = client.delete("/calibration/reset/cal-3")
            assert resp.status_code == 200

            # Profile should be default
            resp = client.get("/calibration/profile/cal-3")
            assert resp.json()["scenarios_used"] == 0

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_calibration_affects_chat(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        """After calibration, /chat should produce different emotional output."""
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            sid = "cal-chat"

            # Chat without calibration
            resp1 = client.post("/chat", json={"message": "Test", "session_id": sid})
            state1 = resp1.json()["emotional_state"]

            # Reset session for clean comparison
            client.post(f"/reset/{sid}")

            # Submit scenario: user expects MORE positive
            client.post(
                "/calibration/scenario",
                params={"session_id": sid},
                json={
                    "stimulus": "Test",
                    "expected_emotion": "joy",
                    "expected_valence": 0.9,
                    "expected_arousal": 0.7,
                    "expected_intensity": 0.8,
                },
            )
            client.post("/calibration/apply", params={"session_id": sid})

            # Chat with calibration
            resp2 = client.post("/chat", json={"message": "Test", "session_id": sid})
            state2 = resp2.json()["emotional_state"]

        # Calibrated state should be more positive/intense
        assert state2["valence"] >= state1["valence"]
