"""Tests for Emotion API REST endpoints (/api/v1/...)."""

import pytest
from fastapi.testclient import TestClient

from pathos.api_routes import init_api_routes, router as api_router
from pathos.engine.emotion_processor import EmotionProcessor
from pathos.state.manager import StateManager

# We need a minimal FastAPI app for testing
from fastapi import FastAPI

_test_app = FastAPI()
_test_app.include_router(api_router)


@pytest.fixture(autouse=True)
def _setup_api() -> None:
    """Initialize API routes with fresh state for each test."""
    sm = StateManager()
    processor = EmotionProcessor(sm)
    init_api_routes(sm, processor)


@pytest.fixture
def client() -> TestClient:
    return TestClient(_test_app)


# ── Health Check ──


class TestHealthEndpoint:

    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "3.0.0"
        assert data["systems_count"] == 23
        assert data["coupling_enabled"] is True

    def test_health_shows_active_sessions(self, client: TestClient) -> None:
        # Process something to create a session
        client.post("/api/v1/emotion/process", json={"stimulus": "hello"})
        resp = client.get("/api/v1/health")
        assert resp.json()["active_sessions"] >= 1


# ── Process Endpoint ──


class TestProcessEndpoint:

    def test_process_basic(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "I feel happy today",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "emotional_state" in data
        assert data["session_id"].startswith("api-")
        assert data["turn_number"] == 1

    def test_process_with_session_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
            "session_id": "my-session",
        })
        data = resp.json()
        assert data["session_id"] == "api-my-session"

    def test_process_already_prefixed_session(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
            "session_id": "api-existing",
        })
        data = resp.json()
        assert data["session_id"] == "api-existing"

    def test_process_increments_turn(self, client: TestClient) -> None:
        payload = {"stimulus": "hello", "session_id": "api-turns"}
        r1 = client.post("/api/v1/emotion/process", json=payload)
        r2 = client.post("/api/v1/emotion/process", json=payload)
        assert r1.json()["turn_number"] == 1
        assert r2.json()["turn_number"] == 2

    def test_process_with_personality(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
            "personality": {"neuroticism": 0.9},
        })
        assert resp.status_code == 200

    def test_process_with_config(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
            "config": {
                "advanced_mode": False,
                "include_pipeline_trace": True,
            },
        })
        data = resp.json()
        assert data["pipeline_trace"] is not None
        assert data["pipeline_trace"]["appraisal_method"] == "keyword"

    def test_process_with_external_signals(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "neutral",
            "external_signals": [
                {"source": "facial_au", "arousal_hint": 0.9, "confidence": 0.8},
            ],
        })
        data = resp.json()
        assert len(data["external_contributions"]) == 1
        assert data["external_contributions"][0]["source"] == "facial_au"

    def test_process_returns_body_state(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={"stimulus": "hello"})
        data = resp.json()
        assert "energy" in data
        assert "tension" in data
        assert "openness" in data
        assert "warmth" in data

    def test_process_returns_mood(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={"stimulus": "hello"})
        data = resp.json()
        assert "mood_label" in data
        assert "mood_trend" in data

    def test_process_returns_processing_time(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={"stimulus": "hello"})
        assert resp.json()["processing_time_ms"] > 0


# ── Batch Endpoint ──


class TestBatchEndpoint:

    def test_batch_basic(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/batch", json={
            "stimuli": ["I am happy", "Now I am sad", "I feel nothing"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["results"][0]["turn_number"] == 1
        assert data["results"][1]["turn_number"] == 2
        assert data["results"][2]["turn_number"] == 3

    def test_batch_session_continuity(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/batch", json={
            "stimuli": ["happy", "sad"],
            "session_id": "api-batch-test",
        })
        data = resp.json()
        assert data["session_id"] == "api-batch-test"
        # State should persist — check via state endpoint
        state_resp = client.get("/api/v1/emotion/state?session_id=api-batch-test")
        assert state_resp.json()["turn_number"] == 2

    def test_batch_total_time(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/batch", json={
            "stimuli": ["a", "b"],
        })
        assert resp.json()["total_processing_time_ms"] > 0

    def test_batch_with_personality(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/batch", json={
            "stimuli": ["test1", "test2"],
            "personality": {"neuroticism": 0.9},
        })
        assert resp.status_code == 200
        # Personality only applied on first stimulus
        assert len(resp.json()["results"]) == 2


# ── State Endpoint ──


class TestStateEndpoint:

    def test_state_fresh_session(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/state?session_id=api-fresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "api-fresh"
        assert data["turn_number"] == 0

    def test_state_after_processing(self, client: TestClient) -> None:
        client.post("/api/v1/emotion/process", json={
            "stimulus": "I am joyful",
            "session_id": "api-state-test",
        })
        resp = client.get("/api/v1/emotion/state?session_id=api-state-test")
        data = resp.json()
        assert data["turn_number"] == 1
        assert "emotional_state" in data

    def test_state_shows_personality(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/state?session_id=api-pers")
        data = resp.json()
        assert "personality_summary" in data
        assert "openness" in data["personality_summary"]

    def test_state_shows_active_systems(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/state?session_id=api-sys")
        data = resp.json()
        assert "active_systems" in data
        assert "appraisal" in data["active_systems"]

    def test_state_auto_prefixes_session(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/state?session_id=no-prefix")
        assert resp.json()["session_id"] == "api-no-prefix"


# ── Configure Endpoint ──


class TestConfigureEndpoint:

    def test_configure_personality(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/configure", json={
            "session_id": "api-cfg",
            "personality": {"neuroticism": 0.8, "openness": 0.9},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["personality_applied"]["neuroticism"] == pytest.approx(0.8, abs=0.01)
        assert data["personality_applied"]["openness"] == pytest.approx(0.9, abs=0.01)

    def test_configure_values(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/configure", json={
            "session_id": "api-vals",
            "values": {"truth": 0.5},
        })
        data = resp.json()
        assert data["values_applied"] is not None
        assert "truth" in data["values_applied"]

    def test_configure_reset_state(self, client: TestClient) -> None:
        # Process something first
        client.post("/api/v1/emotion/process", json={
            "stimulus": "I am furious",
            "session_id": "api-reset-cfg",
        })
        # Reset
        resp = client.post("/api/v1/emotion/configure", json={
            "session_id": "api-reset-cfg",
            "reset_state": True,
        })
        data = resp.json()
        assert data["state_reset"] is True
        # Verify state is neutral
        state = client.get("/api/v1/emotion/state?session_id=api-reset-cfg")
        assert state.json()["turn_number"] == 0

    def test_configure_auto_prefix(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/configure", json={
            "session_id": "raw-id",
            "personality": {"openness": 0.5},
        })
        assert resp.json()["session_id"] == "api-raw-id"


# ── Reset Endpoint ──


class TestResetEndpoint:

    def test_reset_session(self, client: TestClient) -> None:
        # Process something
        client.post("/api/v1/emotion/process", json={
            "stimulus": "test", "session_id": "api-rst",
        })
        # Reset
        resp = client.post("/api/v1/emotion/reset?session_id=api-rst")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["session_id"] == "api-rst"

    def test_reset_clears_state(self, client: TestClient) -> None:
        client.post("/api/v1/emotion/process", json={
            "stimulus": "hello", "session_id": "api-rst2",
        })
        client.post("/api/v1/emotion/reset?session_id=api-rst2")
        state = client.get("/api/v1/emotion/state?session_id=api-rst2")
        assert state.json()["turn_number"] == 0


# ── Presets Endpoint ──


class TestPresetsEndpoint:

    def test_list_presets(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "presets" in data
        assert len(data["presets"]) >= 4

    def test_preset_structure(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/presets")
        preset = resp.json()["presets"][0]
        assert "name" in preset
        assert "description" in preset
        assert "traits" in preset
        assert "openness" in preset["traits"]

    def test_known_presets_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/emotion/presets")
        names = {p["name"] for p in resp.json()["presets"]}
        assert "balanced" in names
        assert "neurotic" in names
        assert "resilient" in names


# ── Session Isolation (B4 — API session management) ──


class TestSessionManagement:
    """API sessions are isolated from each other and use api- prefix."""

    def test_sessions_are_independent(self, client: TestClient) -> None:
        client.post("/api/v1/emotion/process", json={
            "stimulus": "rage and fury",
            "session_id": "api-independent-a",
        })
        resp_b = client.post("/api/v1/emotion/process", json={
            "stimulus": "peace and calm",
            "session_id": "api-independent-b",
        })
        # Session B should not be affected by session A
        assert resp_b.json()["turn_number"] == 1

    def test_api_prefix_enforced(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
            "session_id": "unprefixed",
        })
        assert resp.json()["session_id"] == "api-unprefixed"

    def test_default_session_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/emotion/process", json={
            "stimulus": "test",
        })
        assert resp.json()["session_id"] == "api-default"
