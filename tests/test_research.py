"""Tests para Research Mode endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from pathos.engine.appraiser import AppraisalResult
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
from pathos.models.schemas import (
    AppraisalDetails,
    EmotionGenerationDetails,
    HomeostasisDetails,
    MemoryAmplificationDetails,
    MoodCongruenceDetails,
    ResearchChatResponse,
    ResearchStateResponse,
)


# --- Fixtures ---

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
    """Reset state manager between tests."""
    state_manager._sessions.clear()


def _mock_llm() -> AsyncMock:
    """Create a mock LLM provider."""
    mock = AsyncMock()
    mock.generate.return_value = "Mock response from LLM"
    mock.embed.return_value = [0.1] * 768
    return mock


# --- Schema tests ---


class TestResearchSchemas:
    def test_homeostasis_details_fields(self) -> None:
        state = EmotionalState()
        details = HomeostasisDetails(applied=False, state_before=state, state_after=state)
        assert details.applied is False
        assert details.state_before.primary_emotion == PrimaryEmotion.NEUTRAL

    def test_appraisal_details_fields(self) -> None:
        details = AppraisalDetails(
            vector=MOCK_APPRAISAL.vector,
            computed_valence=0.52,
            computed_arousal=0.42,
            computed_dominance=0.55,
            computed_certainty=0.5,
        )
        assert details.vector.relevance.novelty == 0.5
        assert details.computed_valence == 0.52

    def test_memory_amplification_details(self) -> None:
        details = MemoryAmplificationDetails(
            amplification_factor=0.0,
            memories_count=0,
            memory_stored=False,
        )
        assert details.amplification_factor == 0.0
        assert details.memory_stored is False

    def test_mood_congruence_details(self) -> None:
        details = MoodCongruenceDetails(
            valence_bias=0.015,
            arousal_bias=0.0,
            mood_label="neutral",
            mood_trend="stable",
        )
        assert details.mood_label == "neutral"

    def test_emotion_generation_details(self) -> None:
        details = EmotionGenerationDetails(
            raw_valence=0.5,
            raw_arousal=0.4,
            raw_dominance=0.5,
            raw_certainty=0.5,
            blended_valence=0.3,
            blended_arousal=0.36,
            blended_dominance=0.5,
            blended_certainty=0.5,
            intensity_before_amplification=0.3,
            intensity_after_amplification=0.3,
        )
        assert details.raw_valence == 0.5
        assert details.blended_valence == 0.3

    def test_research_state_response_fields(self) -> None:
        resp = ResearchStateResponse(
            session_id="test",
            turn_count=0,
            emotional_state=EmotionalState(),
            value_system=state_manager.get_session("x").value_system,
            memories=[],
            conversation_length=0,
        )
        assert resp.turn_count == 0
        assert len(resp.value_system.core_values) == 5


# --- Endpoint integration tests ---


class TestResearchChatEndpoint:
    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_returns_all_internals(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        mock_provider.__class__ = type(mock_llm)
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "Hello world", "session_id": "research-1"},
            )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level fields
        assert data["response"] == "Mock response from LLM"
        assert data["session_id"] == "research-1"
        assert data["turn_number"] == 1

        # Homeostasis (turn 1 = not applied)
        assert data["homeostasis"]["applied"] is False

        # Appraisal
        assert "vector" in data["appraisal"]
        assert data["appraisal"]["computed_valence"] is not None

        # Memory
        assert data["memory_amplification"]["amplification_factor"] == 0.0
        assert data["memory_amplification"]["memories_count"] == 0

        # Mood congruence
        assert "valence_bias" in data["mood_congruence"]
        assert "mood_label" in data["mood_congruence"]

        # Emotion generation
        assert "raw_valence" in data["emotion_generation"]
        assert "blended_valence" in data["emotion_generation"]

        # Behavior prompt
        assert len(data["behavior_prompt"]) > 0

        # Emotional state
        assert "primary_emotion" in data["emotional_state"]

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_homeostasis_applied_on_turn_2(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            # Turn 1
            client.post(
                "/research/chat",
                json={"message": "First message", "session_id": "research-2"},
            )
            # Turn 2
            resp = client.post(
                "/research/chat",
                json={"message": "Second message", "session_id": "research-2"},
            )
        data = resp.json()
        assert data["homeostasis"]["applied"] is True
        assert data["turn_number"] == 2

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_appraisal_dimensions_match(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "Test", "session_id": "research-3"},
            )
        data = resp.json()
        appraisal = data["appraisal"]
        # computed values should be derived from the mock appraisal
        assert appraisal["computed_valence"] > 0  # positive appraisal
        assert 0 <= appraisal["computed_arousal"] <= 1
        assert 0 <= appraisal["computed_dominance"] <= 1
        assert 0 <= appraisal["computed_certainty"] <= 1

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_emotion_generation_raw_vs_blended(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        """Raw values come from appraisal, blended include inercia."""
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "Test", "session_id": "research-4"},
            )
        gen = resp.json()["emotion_generation"]
        # Blended values are a mix of raw + previous state (neutral)
        # So blended should be between raw and neutral (0.0 valence, 0.3 arousal)
        assert gen["raw_valence"] != gen["blended_valence"] or gen["raw_arousal"] != gen["blended_arousal"]

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_behavior_prompt_contains_emotion(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "This is great!", "session_id": "research-5"},
            )
        prompt = resp.json()["behavior_prompt"]
        # Should contain base prompt elements
        assert "estado interno" in prompt or "arquitectura emocional" in prompt

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_chat_memory_stored_when_intense(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        """High-intensity emotions should store a memory."""
        high_intensity_appraisal = AppraisalResult(
            vector=AppraisalVector(
                relevance=RelevanceCheck(novelty=0.9, personal_significance=0.9, values_affected=["truth"]),
                valence=ValenceAssessment(goal_conduciveness=-0.8, value_alignment=-0.7, intrinsic_pleasantness=-0.6),
                coping=CopingPotential(control=0.2, power=0.2, adjustability=0.3),
                agency=AgencyAttribution(responsible_agent="user", intentionality=0.8, fairness=-0.7),
                norms=NormCompatibility(internal_standards=-0.6, external_standards=-0.5, self_consistency=-0.4),
            ),
            emotion_hint=None,
        )
        mock_appraise.return_value = high_intensity_appraisal
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/research/chat",
                json={"message": "This is terrible and unfair!", "session_id": "research-6"},
            )
        data = resp.json()
        assert data["memory_amplification"]["memory_stored"] is True


class TestResearchStateEndpoint:
    def test_research_state_new_session(self) -> None:
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/research/state/new-session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "new-session"
        assert data["turn_count"] == 0
        assert data["emotional_state"]["primary_emotion"] == "neutral"
        assert len(data["value_system"]["core_values"]) == 5
        assert data["memories"] == []
        assert data["conversation_length"] == 0

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_state_after_chat(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        mock_appraise.return_value = MOCK_APPRAISAL
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            client.post(
                "/research/chat",
                json={"message": "Hello", "session_id": "state-test"},
            )
            resp = client.get("/research/state/state-test")
        data = resp.json()
        assert data["turn_count"] == 1
        assert data["conversation_length"] == 2  # user + assistant
        assert data["emotional_state"]["primary_emotion"] != "neutral" or data["emotional_state"]["triggered_by"] == "Hello"

    def test_research_state_includes_value_system(self) -> None:
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/research/state/values-test")
        data = resp.json()
        values = data["value_system"]
        assert "core_values" in values
        assert "relational" in values
        assert "self_model" in values
        core_names = [v["name"] for v in values["core_values"]]
        assert "truth" in core_names
        assert "compassion" in core_names

    @patch("pathos.main.llm_provider")
    @patch("pathos.main.appraise")
    def test_research_state_memories_populated(
        self, mock_appraise: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        """After a high-intensity interaction, memories should appear in state."""
        high_appraisal = AppraisalResult(
            vector=AppraisalVector(
                relevance=RelevanceCheck(novelty=0.9, personal_significance=0.9, values_affected=["truth"]),
                valence=ValenceAssessment(goal_conduciveness=-0.8, value_alignment=-0.7, intrinsic_pleasantness=-0.6),
                coping=CopingPotential(control=0.2, power=0.2, adjustability=0.3),
                agency=AgencyAttribution(responsible_agent="user", intentionality=0.8, fairness=-0.7),
                norms=NormCompatibility(internal_standards=-0.6, external_standards=-0.5, self_consistency=-0.4),
            ),
            emotion_hint=None,
        )
        mock_appraise.return_value = high_appraisal
        mock_llm = _mock_llm()
        with patch("pathos.main.llm_provider", mock_llm):
            client = TestClient(app, raise_server_exceptions=True)
            client.post(
                "/research/chat",
                json={"message": "This is unfair!", "session_id": "mem-test"},
            )
            resp = client.get("/research/state/mem-test")
        data = resp.json()
        assert len(data["memories"]) >= 1
        assert data["memories"][0]["stimulus"] == "This is unfair!"
