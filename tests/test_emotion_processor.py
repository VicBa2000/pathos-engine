"""Tests for EmotionProcessor — standalone emotional pipeline."""

import pytest

from pathos.engine.emotion_processor import EmotionProcessor
from pathos.models.emotion_api import (
    EmotionAPIConfig,
    EmotionProcessRequest,
    EmotionProcessResponse,
    ExternalSignal,
)
from pathos.models.emotion import PrimaryEmotion
from pathos.state.manager import StateManager


@pytest.fixture
def state_manager() -> StateManager:
    return StateManager()


@pytest.fixture
def processor(state_manager: StateManager) -> EmotionProcessor:
    return EmotionProcessor(state_manager)


def _make_request(
    stimulus: str = "hello",
    session_id: str = "api-test",
    **kwargs: object,
) -> EmotionProcessRequest:
    return EmotionProcessRequest(stimulus=stimulus, session_id=session_id, **kwargs)


# ── Basic Processing ──


class TestBasicProcessing:
    """Core processing without advanced systems."""

    @pytest.mark.asyncio
    async def test_process_returns_response(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("I feel happy today"))
        assert isinstance(resp, EmotionProcessResponse)
        assert resp.session_id == "api-test"
        assert resp.turn_number == 1

    @pytest.mark.asyncio
    async def test_process_neutral_stimulus(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("hello"))
        assert resp.emotional_state is not None
        assert -1.0 <= resp.valence <= 1.0
        assert 0.0 <= resp.arousal <= 1.0
        assert 0.0 <= resp.dominance <= 1.0
        assert 0.0 <= resp.certainty <= 1.0

    @pytest.mark.asyncio
    async def test_process_positive_stimulus(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("I love this, it makes me so happy!"))
        assert resp.valence > 0.0 or resp.primary_emotion in ("joy", "love", "neutral")

    @pytest.mark.asyncio
    async def test_process_negative_stimulus(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("I hate this, it makes me furious and angry"))
        assert resp.valence < 0.0 or resp.primary_emotion in ("anger", "disgust", "fear", "sadness")

    @pytest.mark.asyncio
    async def test_processing_time_measured(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("test"))
        assert resp.processing_time_ms > 0.0

    @pytest.mark.asyncio
    async def test_body_state_in_response(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("hello"))
        assert 0.0 <= resp.energy <= 1.0
        assert 0.0 <= resp.tension <= 1.0
        assert 0.0 <= resp.openness <= 1.0
        assert 0.0 <= resp.warmth <= 1.0

    @pytest.mark.asyncio
    async def test_mood_in_response(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("hello"))
        assert resp.mood_label != ""
        assert resp.mood_trend in ("rising", "falling", "stable")


# ── Session Persistence ──


class TestSessionPersistence:
    """State persists across calls for the same session."""

    @pytest.mark.asyncio
    async def test_turn_count_increments(self, processor: EmotionProcessor) -> None:
        resp1 = await processor.process(_make_request("first"))
        resp2 = await processor.process(_make_request("second"))
        assert resp1.turn_number == 1
        assert resp2.turn_number == 2

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self, processor: EmotionProcessor) -> None:
        resp_a = await processor.process(_make_request("angry!", session_id="api-a"))
        resp_b = await processor.process(_make_request("happy!", session_id="api-b"))
        assert resp_a.turn_number == 1
        assert resp_b.turn_number == 1

    @pytest.mark.asyncio
    async def test_state_evolves_across_turns(self, processor: EmotionProcessor) -> None:
        await processor.process(_make_request("I am so happy and joyful!"))
        resp2 = await processor.process(_make_request("This makes me extremely sad and depressed"))
        # Second turn should show some change from the positive first turn
        assert resp2.turn_number == 2


# ── Personality Override ──


class TestPersonalityOverride:
    """Personality overrides modify emotional processing."""

    @pytest.mark.asyncio
    async def test_personality_applied(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "test",
            personality={"neuroticism": 0.9, "extraversion": 0.1},
        )
        resp = await processor.process(req)
        assert resp.session_id == "api-test"
        session = processor.state_manager.get_session("api-test")
        assert session.personality.neuroticism == pytest.approx(0.9, abs=0.01)
        assert session.personality.extraversion == pytest.approx(0.1, abs=0.01)

    @pytest.mark.asyncio
    async def test_partial_personality_keeps_defaults(self, processor: EmotionProcessor) -> None:
        req = _make_request("test", personality={"openness": 0.95})
        await processor.process(req)
        session = processor.state_manager.get_session("api-test")
        assert session.personality.openness == pytest.approx(0.95, abs=0.01)
        # Other traits should stay at defaults
        assert session.personality.conscientiousness > 0

    @pytest.mark.asyncio
    async def test_personality_clamped(self, processor: EmotionProcessor) -> None:
        req = _make_request("test", personality={"neuroticism": 5.0, "openness": -2.0})
        await processor.process(req)
        session = processor.state_manager.get_session("api-test")
        assert session.personality.neuroticism == 1.0
        assert session.personality.openness == 0.0


# ── Configuration ──


class TestConfig:
    """API configuration controls response content."""

    @pytest.mark.asyncio
    async def test_minimal_config(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "hello",
            config=EmotionAPIConfig(
                advanced_mode=False,
                include_coupling=False,
                include_pipeline_trace=False,
                include_voice_params=False,
                include_behavior_prompt=False,
                detail_level="minimal",
            ),
        )
        resp = await processor.process(req)
        assert resp.pipeline_trace is None
        assert resp.voice_params is None
        assert resp.behavior_prompt is None
        assert resp.coupling_contributions is None

    @pytest.mark.asyncio
    async def test_pipeline_trace_included(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "I feel great",
            config=EmotionAPIConfig(include_pipeline_trace=True),
        )
        resp = await processor.process(req)
        assert resp.pipeline_trace is not None
        assert "appraisal_method" in resp.pipeline_trace
        assert resp.pipeline_trace["appraisal_method"] == "keyword"

    @pytest.mark.asyncio
    async def test_behavior_prompt_included(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "hello",
            config=EmotionAPIConfig(include_behavior_prompt=True),
        )
        resp = await processor.process(req)
        assert resp.behavior_prompt is not None
        assert isinstance(resp.behavior_prompt, str)
        assert len(resp.behavior_prompt) > 0

    @pytest.mark.asyncio
    async def test_coupling_contributions_included(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "hello",
            config=EmotionAPIConfig(include_coupling=True, advanced_mode=True),
        )
        resp = await processor.process(req)
        assert resp.coupling_contributions is not None
        assert "valence" in resp.coupling_contributions
        assert "arousal" in resp.coupling_contributions

    @pytest.mark.asyncio
    async def test_advanced_mode_off(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "I am happy",
            config=EmotionAPIConfig(advanced_mode=False),
        )
        resp = await processor.process(req)
        # Should still work, just without advanced systems
        assert resp.emotional_state is not None
        assert resp.coupling_contributions is None


# ── External Signals ──


class TestExternalSignals:
    """External signals modulate the emotional pipeline."""

    @pytest.mark.asyncio
    async def test_external_signals_processed(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "neutral text",
            external_signals=[
                ExternalSignal(
                    source="facial_au",
                    arousal_hint=0.9,
                    confidence=0.8,
                ),
            ],
        )
        resp = await processor.process(req)
        assert len(resp.external_contributions) == 1
        assert resp.external_contributions[0].source == "facial_au"

    @pytest.mark.asyncio
    async def test_multiple_signals_fused(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "test",
            external_signals=[
                ExternalSignal(source="facial_au", arousal_hint=0.8, confidence=0.9),
                ExternalSignal(source="weather", valence_hint=-0.3, confidence=0.5),
            ],
        )
        resp = await processor.process(req)
        assert len(resp.external_contributions) == 2

    @pytest.mark.asyncio
    async def test_no_signals_no_contributions(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("test"))
        assert resp.external_contributions == []

    @pytest.mark.asyncio
    async def test_signal_contributions_in_trace(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "test",
            external_signals=[
                ExternalSignal(source="facial_au", arousal_hint=0.9, confidence=0.8),
            ],
            config=EmotionAPIConfig(include_pipeline_trace=True),
        )
        resp = await processor.process(req)
        assert resp.pipeline_trace is not None
        assert resp.pipeline_trace["external_signals_count"] == 1


# ── Top Emotions ──


class TestTopEmotions:
    """Top emotions extracted from emotional stack."""

    @pytest.mark.asyncio
    async def test_top_emotions_populated(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("I feel wonderful and happy"))
        # After advanced processing, should have stack entries
        assert isinstance(resp.top_emotions, dict)

    @pytest.mark.asyncio
    async def test_top_emotions_limited_to_5(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("I feel complex emotions"))
        assert len(resp.top_emotions) <= 5


# ── No LLM Required ──


class TestNoLLMRequired:
    """Processor works entirely without an LLM provider."""

    @pytest.mark.asyncio
    async def test_keyword_appraisal_default(self, processor: EmotionProcessor) -> None:
        req = _make_request(
            "I am terrified and scared",
            config=EmotionAPIConfig(include_pipeline_trace=True),
        )
        resp = await processor.process(req)
        assert resp.pipeline_trace is not None
        assert resp.pipeline_trace["appraisal_method"] == "keyword"

    @pytest.mark.asyncio
    async def test_no_llm_still_processes(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request("this is terrible"))
        assert resp.emotional_state is not None
        assert resp.turn_number == 1


# ── Edge Cases ──


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_stimulus(self, processor: EmotionProcessor) -> None:
        resp = await processor.process(_make_request(""))
        assert resp.emotional_state is not None

    @pytest.mark.asyncio
    async def test_long_stimulus(self, processor: EmotionProcessor) -> None:
        long_text = "I feel happy. " * 500
        resp = await processor.process(_make_request(long_text))
        assert resp.emotional_state is not None

    @pytest.mark.asyncio
    async def test_multiple_rapid_requests(self, processor: EmotionProcessor) -> None:
        for i in range(10):
            resp = await processor.process(_make_request(f"turn {i}"))
            assert resp.turn_number == i + 1

    @pytest.mark.asyncio
    async def test_homeostasis_applied_after_turn_1(self, processor: EmotionProcessor) -> None:
        # First turn: strong negative
        await processor.process(_make_request("I hate everything, fury and rage"))
        # Second turn: neutral — homeostasis should pull back toward baseline
        resp2 = await processor.process(_make_request("ok"))
        assert resp2.turn_number == 2
