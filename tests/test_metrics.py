"""Tests para Metricas de Autenticidad Emocional."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from pathos.engine.appraiser import AppraisalResult
from pathos.engine.metrics import coherence, continuity, proportionality, recovery
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


# --- Helper factories ---


def _appraisal(
    goal: float = 0.5,
    align: float = 0.5,
    pleasant: float = 0.3,
    novelty: float = 0.3,
    significance: float = 0.5,
    control: float = 0.5,
    power: float = 0.5,
    adjustability: float = 0.5,
    fairness: float = 0.3,
) -> AppraisalVector:
    return AppraisalVector(
        relevance=RelevanceCheck(
            novelty=novelty, personal_significance=significance, values_affected=["truth"],
        ),
        valence=ValenceAssessment(
            goal_conduciveness=goal, value_alignment=align, intrinsic_pleasantness=pleasant,
        ),
        coping=CopingPotential(control=control, power=power, adjustability=adjustability),
        agency=AgencyAttribution(responsible_agent="user", intentionality=0.5, fairness=fairness),
        norms=NormCompatibility(internal_standards=0.5, external_standards=0.4, self_consistency=0.5),
    )


def _state(
    valence: float = 0.0,
    arousal: float = 0.3,
    dominance: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.0,
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
) -> EmotionalState:
    return EmotionalState(
        valence=valence, arousal=arousal, dominance=dominance,
        certainty=certainty, intensity=intensity, primary_emotion=emotion,
    )


# --- Coherence tests ---


class TestCoherence:
    def test_positive_appraisal_positive_state(self) -> None:
        """Positive appraisal + positive emotion = high coherence."""
        ap = _appraisal(goal=0.7, align=0.6, pleasant=0.5)
        state = _state(valence=0.5, arousal=0.4, dominance=0.6)
        score = coherence(state, ap)
        assert score > 0.7

    def test_negative_appraisal_negative_state(self) -> None:
        """Negative appraisal + negative emotion = high coherence."""
        ap = _appraisal(goal=-0.7, align=-0.6, pleasant=-0.5)
        state = _state(valence=-0.5, arousal=0.6, dominance=0.3)
        score = coherence(state, ap)
        assert score > 0.7

    def test_mismatch_low_coherence(self) -> None:
        """Positive appraisal + negative emotion = low coherence."""
        ap = _appraisal(goal=0.8, align=0.7, pleasant=0.6)
        state = _state(valence=-0.6, arousal=0.2, dominance=0.3)
        score = coherence(state, ap)
        assert score < 0.6

    def test_neutral_appraisal_neutral_state(self) -> None:
        """Neutral appraisal + neutral state = high coherence."""
        ap = _appraisal(goal=0.05, align=0.0, pleasant=0.0, novelty=0.1, significance=0.1, control=0.5)
        state = _state(valence=0.0, arousal=0.3, dominance=0.5)
        score = coherence(state, ap)
        assert score > 0.8

    def test_score_in_range(self) -> None:
        ap = _appraisal()
        state = _state(valence=0.3, arousal=0.5)
        score = coherence(state, ap)
        assert 0 <= score <= 1


# --- Continuity tests ---


class TestContinuity:
    def test_identical_states_perfect_continuity(self) -> None:
        s1 = _state(valence=0.3, arousal=0.4, dominance=0.5, certainty=0.6)
        s2 = _state(valence=0.3, arousal=0.4, dominance=0.5, certainty=0.6)
        assert continuity(s2, s1) == 1.0

    def test_small_change_high_continuity(self) -> None:
        s1 = _state(valence=0.3, arousal=0.4)
        s2 = _state(valence=0.35, arousal=0.42)
        score = continuity(s2, s1)
        assert score > 0.9

    def test_large_change_low_continuity(self) -> None:
        s1 = _state(valence=-0.8, arousal=0.1, dominance=0.2, certainty=0.2)
        s2 = _state(valence=0.8, arousal=0.9, dominance=0.9, certainty=0.9)
        score = continuity(s2, s1)
        assert score < 0.4

    def test_moderate_change(self) -> None:
        s1 = _state(valence=0.0, arousal=0.3)
        s2 = _state(valence=0.4, arousal=0.5)
        score = continuity(s2, s1)
        assert 0.5 < score < 0.95

    def test_score_in_range(self) -> None:
        s1 = _state(valence=-1, arousal=0, dominance=0, certainty=0)
        s2 = _state(valence=1, arousal=1, dominance=1, certainty=1)
        score = continuity(s2, s1)
        assert 0 <= score <= 1


# --- Proportionality tests ---


class TestProportionality:
    def test_high_stimulus_high_intensity(self) -> None:
        """Strong stimulus + strong response = proportional."""
        ap = _appraisal(significance=0.8, goal=0.7, align=0.6)
        state = _state(intensity=0.7)
        score = proportionality(state, ap)
        assert score > 0.7

    def test_low_stimulus_low_intensity(self) -> None:
        """Weak stimulus + weak response = proportional."""
        ap = _appraisal(significance=0.1, goal=0.1, align=0.1)
        state = _state(intensity=0.1)
        score = proportionality(state, ap)
        assert score > 0.8

    def test_low_stimulus_high_intensity_disproportional(self) -> None:
        """Weak stimulus + strong response = disproportional."""
        ap = _appraisal(significance=0.1, goal=0.1, align=0.1)
        state = _state(intensity=0.8)
        score = proportionality(state, ap)
        assert score < 0.5

    def test_high_stimulus_low_intensity_disproportional(self) -> None:
        """Strong stimulus + weak response = disproportional."""
        ap = _appraisal(significance=0.9, goal=0.8, align=0.7)
        state = _state(intensity=0.1)
        score = proportionality(state, ap)
        assert score < 0.5

    def test_score_in_range(self) -> None:
        ap = _appraisal()
        state = _state(intensity=0.5)
        score = proportionality(state, ap)
        assert 0 <= score <= 1


# --- Recovery tests ---


class TestRecovery:
    def test_natural_decay(self) -> None:
        """Intensity peaks then decays = good recovery."""
        states = [
            _state(intensity=0.2),
            _state(intensity=0.8),
            _state(intensity=0.6),
            _state(intensity=0.4),
        ]
        score = recovery(states)
        assert score == 1.0

    def test_no_decay_after_peak(self) -> None:
        """Intensity peaks then stays high = bad recovery."""
        states = [
            _state(intensity=0.2),
            _state(intensity=0.7),
            _state(intensity=0.8),  # Goes up after peak
            _state(intensity=0.9),  # Still going up
        ]
        score = recovery(states)
        assert score < 1.0

    def test_insufficient_data(self) -> None:
        """Less than 3 states = default to 1.0."""
        assert recovery([_state(), _state()]) == 1.0
        assert recovery([_state()]) == 1.0
        assert recovery([]) == 1.0

    def test_flat_low_intensity(self) -> None:
        """All low intensity = no peaks = perfect recovery."""
        states = [_state(intensity=0.1) for _ in range(5)]
        score = recovery(states)
        assert score == 1.0

    def test_multiple_peaks_with_decay(self) -> None:
        """Multiple peaks that all decay = good recovery."""
        states = [
            _state(intensity=0.2),
            _state(intensity=0.7),
            _state(intensity=0.5),
            _state(intensity=0.3),
            _state(intensity=0.8),
            _state(intensity=0.6),
        ]
        score = recovery(states)
        assert score == 1.0

    def test_score_in_range(self) -> None:
        states = [
            _state(intensity=0.1),
            _state(intensity=0.9),
            _state(intensity=0.95),
        ]
        score = recovery(states)
        assert 0 <= score <= 1


# --- Integration: metrics in research endpoint ---


MOCK_APPRAISAL = AppraisalResult(
    vector=_appraisal(goal=0.6, align=0.5, pleasant=0.4, significance=0.7),
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


class TestMetricsInEndpoint:
    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_includes_metrics(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "Test metrics", "session_id": "metrics-1"},
            )
        data = resp.json()
        metrics = data["authenticity_metrics"]
        assert "coherence" in metrics
        assert "continuity" in metrics
        assert "proportionality" in metrics
        assert "recovery" in metrics
        assert "overall" in metrics
        # All in range
        for key in ["coherence", "continuity", "proportionality", "recovery", "overall"]:
            assert 0 <= metrics[key] <= 1, f"{key} out of range: {metrics[key]}"

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_metrics_overall_is_average(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "Test", "session_id": "metrics-2"},
            )
        m = resp.json()["authenticity_metrics"]
        expected = round((m["coherence"] + m["continuity"] + m["proportionality"] + m["recovery"]) / 4, 4)
        assert abs(m["overall"] - expected) < 0.001

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_continuity_high_on_first_turn(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        """First turn compares to neutral state, should have reasonable continuity."""
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "Hello", "session_id": "metrics-3"},
            )
        # From neutral to mild positive = moderate-high continuity
        assert resp.json()["authenticity_metrics"]["continuity"] > 0.5
