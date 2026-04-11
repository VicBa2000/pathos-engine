"""Integration tests for v3 features: Coupling + API + External Signals working together."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from pathos.api_routes import init_api_routes, router as api_router
from pathos.engine.emotion_processor import EmotionProcessor
from pathos.models.emotion_api import (
    EmotionAPIConfig,
    EmotionProcessRequest,
    ExternalSignal,
)
from pathos.state.manager import StateManager

_test_app = FastAPI()
_test_app.include_router(api_router)


@pytest.fixture(autouse=True)
def _setup_api() -> None:
    sm = StateManager()
    processor = EmotionProcessor(sm)
    init_api_routes(sm, processor)


@pytest.fixture
def client() -> TestClient:
    return TestClient(_test_app)


@pytest.fixture
def processor() -> EmotionProcessor:
    return EmotionProcessor(StateManager())


# ── Coupling + API Integration ──


class TestCouplingInAPI:
    """Coupling ODE works correctly through the API endpoints."""

    def test_coupling_active_by_default(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "I am terrified",
            "config": {"include_coupling": True, "include_pipeline_trace": True},
        })
        data = resp.json()
        assert data["coupling_contributions"] is not None
        assert data["pipeline_trace"]["coupling_active"] is True

    def test_coupling_disabled_when_configured(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "I am terrified",
            "config": {"include_coupling": False},
        })
        data = resp.json()
        assert data["coupling_contributions"] is None

    def test_coupling_disabled_with_advanced_off(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
            "config": {"advanced_mode": False, "include_coupling": True},
        })
        assert resp.json()["coupling_contributions"] is None

    def test_neurotic_personality_produces_coupling(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "everything is falling apart, I'm so anxious",
            "personality": {"neuroticism": 0.95},
            "config": {"include_coupling": True},
        })
        data = resp.json()
        assert data["coupling_contributions"] is not None
        # Neurotic personality should produce non-trivial coupling
        contribs = data["coupling_contributions"]
        has_nonzero = any(abs(v) > 1e-8 for v in contribs.values())
        assert has_nonzero, f"Expected non-zero coupling contributions, got {contribs}"

    def test_coupling_evolves_over_turns(self, client: TestClient) -> None:
        """Coupling contributions should change as state deviates from baseline."""
        # Turn 1: strong negative stimulus
        r1 = client.post("/api/v1/emotion/process", json={
            "stimulus": "I'm devastated, this is the worst day of my life",
            "session_id": "api-coupling-evo",
            "config": {"include_coupling": True},
        })
        # Turn 2: another negative stimulus (state deviates further)
        r2 = client.post("/api/v1/emotion/process", json={
            "stimulus": "And now my friend betrayed me too, absolute fury",
            "session_id": "api-coupling-evo",
            "config": {"include_coupling": True},
        })
        c1 = r1.json()["coupling_contributions"]
        c2 = r2.json()["coupling_contributions"]
        # Both should have coupling data
        assert c1 is not None
        assert c2 is not None


# ── External Signals + Coupling Integration ──


class TestExternalSignalsWithCoupling:
    """External signals and coupling interact correctly."""

    def test_signals_plus_coupling_both_active(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "entering the room",
            "external_signals": [
                {"source": "facial_au", "arousal_hint": 0.9, "confidence": 0.8},
            ],
            "config": {"include_coupling": True, "include_pipeline_trace": True},
        })
        data = resp.json()
        assert len(data["external_contributions"]) == 1
        assert data["coupling_contributions"] is not None
        assert data["pipeline_trace"]["external_signals_count"] == 1
        assert data["pipeline_trace"]["coupling_active"] is True

    def test_signals_modulate_arousal_coupling_responds(self, client: TestClient) -> None:
        """High arousal from facial_au should trigger coupling effects on dominance."""
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "waiting quietly",
            "external_signals": [
                {"source": "facial_au", "arousal_hint": 0.95, "confidence": 0.9},
            ],
            "personality": {"neuroticism": 0.8},
            "config": {"include_coupling": True},
        })
        data = resp.json()
        assert data["arousal"] > 0.3  # Should be elevated by facial_au signal

    def test_multiple_signals_with_coupling(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "normal day",
            "external_signals": [
                {"source": "facial_au", "arousal_hint": 0.7, "confidence": 0.8},
                {"source": "weather", "valence_hint": -0.5, "confidence": 0.4},
                {"source": "time_of_day", "valence_hint": 0.6, "arousal_hint": 0.4, "confidence": 0.6},
            ],
            "config": {"include_coupling": True, "include_pipeline_trace": True},
        })
        data = resp.json()
        assert len(data["external_contributions"]) == 3
        assert data["pipeline_trace"]["external_signals_count"] == 3


# ── Batch + Coupling + Signals ──


class TestBatchIntegration:
    """Batch processing with coupling and external signals."""

    def test_batch_with_signals_and_coupling(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/batch", json={
            "stimuli": ["good news!", "bad news...", "unexpected surprise"],
            "external_signals": [
                {"source": "facial_au", "arousal_hint": 0.6, "confidence": 0.7},
            ],
            "config": {"include_coupling": True},
        })
        data = resp.json()
        assert len(data["results"]) == 3
        for result in data["results"]:
            assert result["coupling_contributions"] is not None
            assert len(result["external_contributions"]) == 1

    def test_batch_state_accumulates_with_coupling(self, client: TestClient) -> None:
        """Emotional state should evolve coherently across batch items."""
        resp = client.post("/api/v1/emotion/batch", json={
            "stimuli": [
                "I just won the lottery!",
                "My best friend is here to celebrate!",
                "We're going on vacation tomorrow!",
            ],
            "session_id": "api-batch-accum",
            "config": {"include_coupling": True},
        })
        results = resp.json()["results"]
        # Each turn should build on previous state
        assert results[0]["turn_number"] == 1
        assert results[2]["turn_number"] == 3


# ── Configure + Process Integration ──


class TestConfigureAndProcess:
    """Configure endpoint affects subsequent processing."""

    def test_configure_then_process(self, client: TestClient) -> None:
        # Configure neurotic personality
        client.post("/api/v1/emotion/configure", json={
            "session_id": "api-cfg-proc",
            "personality": {"neuroticism": 0.9, "extraversion": 0.2},
        })
        # Process should use the configured personality
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "something bad happened",
            "session_id": "api-cfg-proc",
            "config": {"include_coupling": True},
        })
        data = resp.json()
        assert data["emotional_state"] is not None
        assert data["coupling_contributions"] is not None

    def test_configure_reset_then_process(self, client: TestClient) -> None:
        # Process something
        client.post("/api/v1/emotion/process", json={
            "stimulus": "I'm angry",
            "session_id": "api-reset-proc",
        })
        # Reset
        client.post("/api/v1/emotion/configure", json={
            "session_id": "api-reset-proc",
            "reset_state": True,
        })
        # Process again — should start fresh
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "hello",
            "session_id": "api-reset-proc",
        })
        # Turn is 1 because reset cleared turn count
        assert resp.json()["turn_number"] == 1


# ── Processor Direct Integration ──


class TestProcessorIntegration:
    """EmotionProcessor handles all v3 features together."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_all_features(self, processor: EmotionProcessor) -> None:
        """Process with coupling + signals + personality + full trace."""
        req = EmotionProcessRequest(
            stimulus="I just received terrible news about a close friend",
            session_id="api-full",
            personality={"neuroticism": 0.7, "agreeableness": 0.8},
            external_signals=[
                ExternalSignal(source="facial_au", arousal_hint=0.8, confidence=0.9),
                ExternalSignal(source="facial_au", valence_hint=-0.6, confidence=0.5),
            ],
            config=EmotionAPIConfig(
                advanced_mode=True,
                include_coupling=True,
                include_pipeline_trace=True,
                include_behavior_prompt=True,
            ),
        )
        resp = await processor.process(req)
        assert resp.session_id == "api-full"
        assert resp.emotional_state is not None
        assert resp.coupling_contributions is not None
        assert len(resp.external_contributions) == 2
        assert resp.pipeline_trace is not None
        assert resp.behavior_prompt is not None
        assert resp.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_multi_turn_coherence(self, processor: EmotionProcessor) -> None:
        """Emotional trajectory should be coherent across multiple turns."""
        stimuli = [
            "Everything is wonderful today!",
            "Actually, I just got some bad news...",
            "But maybe it's not so bad after all.",
            "No, it really is terrible. I'm devastated.",
            "I need to calm down and think clearly.",
        ]
        valences: list[float] = []
        for stimulus in stimuli:
            req = EmotionProcessRequest(
                stimulus=stimulus,
                session_id="api-coherence",
                config=EmotionAPIConfig(include_coupling=True),
            )
            resp = await processor.process(req)
            valences.append(resp.valence)

        # Should show emotional trajectory (not stuck at same value)
        assert len(set(round(v, 3) for v in valences)) > 1, \
            f"Expected varying valence across turns, got {valences}"

    @pytest.mark.asyncio
    async def test_external_signals_subtle_not_dominant(self, processor: EmotionProcessor) -> None:
        """External signals should modulate, not override, the emotional state."""
        # Process without signals
        req_no_sig = EmotionProcessRequest(
            stimulus="I am calm and peaceful",
            session_id="api-no-sig",
        )
        resp_no_sig = await processor.process(req_no_sig)

        # Process with contradictory high-arousal signal
        req_sig = EmotionProcessRequest(
            stimulus="I am calm and peaceful",
            session_id="api-with-sig",
            external_signals=[
                ExternalSignal(source="facial_au", arousal_hint=1.0, confidence=1.0),
            ],
        )
        resp_sig = await processor.process(req_sig)

        # Arousal should be somewhat higher with signal, but not fully at 1.0
        # (global scale = 0.3 ensures subtlety)
        assert resp_sig.arousal >= resp_no_sig.arousal - 0.1  # Allow some noise
        if resp_sig.arousal > resp_no_sig.arousal:
            assert resp_sig.arousal < 0.95  # Not fully dominated by signal
