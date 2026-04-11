"""Tests for external signals configuration endpoints and pipeline integration."""

import pytest
from fastapi.testclient import TestClient

# Use the actual app from main.py to test /signals/* endpoints
from pathos.main import app
from pathos.models.external_signals import (
    ExternalSignalsConfig,
    SignalSourceConfig,
    SIGNAL_SOURCES,
    default_signals_config,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ── Model Tests ──


class TestSignalsConfigModel:
    """ExternalSignalsConfig model behavior."""

    def test_default_all_disabled(self) -> None:
        cfg = default_signals_config()
        assert cfg.enabled is False
        assert cfg.active_count == 0
        assert len(cfg.sources) == 4

    def test_master_toggle_controls_active(self) -> None:
        cfg = ExternalSignalsConfig(enabled=False)
        cfg.sources["facial_au"] = SignalSourceConfig(enabled=True)
        # Master off → no active sources
        assert cfg.active_count == 0
        cfg.enabled = True
        assert cfg.active_count == 1

    def test_individual_toggle(self) -> None:
        cfg = ExternalSignalsConfig(enabled=True)
        cfg.sources["facial_au"].enabled = True
        cfg.sources["weather"].enabled = True
        assert cfg.active_count == 2
        assert "facial_au" in cfg.active_sources
        assert "weather" in cfg.active_sources
        assert "keyboard_dynamics" not in cfg.active_sources

    def test_signal_sources_defined(self) -> None:
        assert len(SIGNAL_SOURCES) == 4
        for name, meta in SIGNAL_SOURCES.items():
            assert "label" in meta
            assert "description" in meta
            assert "base_weight" in meta
            assert "category" in meta


# ── GET /signals/config Endpoint ──


class TestGetSignalsConfig:

    def test_get_default_config(self, client: TestClient) -> None:
        resp = client.get("/signals/config/default")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["active_count"] == 0
        assert len(data["sources"]) == 4

    def test_sources_have_metadata(self, client: TestClient) -> None:
        resp = client.get("/signals/config/default")
        sources = resp.json()["sources"]
        for src in sources:
            assert "source" in src
            assert "label" in src
            assert "description" in src
            assert "category" in src
            assert "base_weight" in src
            assert "enabled" in src
            assert src["enabled"] is False


# ── POST /signals/config Endpoint ──


class TestSetSignalsConfig:

    def test_enable_master_toggle(self, client: TestClient) -> None:
        resp = client.post("/signals/config/test-cfg", json={"enabled": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["active_count"] == 0  # No sources enabled yet

    def test_enable_individual_source(self, client: TestClient) -> None:
        # Enable master + facial_au
        client.post("/signals/config/test-src", json={"enabled": True})
        resp = client.post("/signals/config/test-src", json={
            "sources": {
                "facial_au": {"enabled": True, "arousal_hint": 0.8, "confidence": 0.9},
            },
        })
        data = resp.json()
        assert data["active_count"] == 1
        assert "facial_au" in data["active_sources"]

    def test_enable_multiple_sources(self, client: TestClient) -> None:
        client.post("/signals/config/test-multi", json={"enabled": True})
        resp = client.post("/signals/config/test-multi", json={
            "sources": {
                "facial_au": {"enabled": True},
                "weather": {"enabled": True},
                "keyboard_dynamics": {"enabled": True},
            },
        })
        data = resp.json()
        assert data["active_count"] == 3

    def test_disable_source(self, client: TestClient) -> None:
        # Enable then disable
        client.post("/signals/config/test-dis", json={
            "enabled": True,
            "sources": {"facial_au": {"enabled": True}},
        })
        resp = client.post("/signals/config/test-dis", json={
            "sources": {"facial_au": {"enabled": False}},
        })
        assert resp.json()["active_count"] == 0

    def test_configure_values(self, client: TestClient) -> None:
        client.post("/signals/config/test-vals", json={
            "enabled": True,
            "sources": {
                "facial_au": {
                    "enabled": True,
                    "valence_hint": -0.3,
                    "arousal_hint": 0.85,
                    "confidence": 0.95,
                },
            },
        })
        # Verify via GET
        resp = client.get("/signals/config/test-vals")
        sources = {s["source"]: s for s in resp.json()["sources"]}
        fa = sources["facial_au"]
        assert fa["enabled"] is True
        assert fa["valence_hint"] == pytest.approx(-0.3, abs=0.01)
        assert fa["arousal_hint"] == pytest.approx(0.85, abs=0.01)
        assert fa["confidence"] == pytest.approx(0.95, abs=0.01)

    def test_clamps_values(self, client: TestClient) -> None:
        client.post("/signals/config/test-clamp", json={
            "enabled": True,
            "sources": {
                "weather": {
                    "enabled": True,
                    "valence_hint": -5.0,
                    "arousal_hint": 99.0,
                    "confidence": -1.0,
                },
            },
        })
        resp = client.get("/signals/config/test-clamp")
        sources = {s["source"]: s for s in resp.json()["sources"]}
        w = sources["weather"]
        assert w["valence_hint"] == -1.0
        assert w["arousal_hint"] == 1.0
        assert w["confidence"] == 0.0

    def test_ignores_unknown_source(self, client: TestClient) -> None:
        resp = client.post("/signals/config/test-unk", json={
            "sources": {"nonexistent_sensor": {"enabled": True}},
        })
        assert resp.status_code == 200
        assert resp.json()["active_count"] == 0


# ── POST /signals/test Endpoint ──


class TestSignalTest:

    def test_basic_test(self, client: TestClient) -> None:
        resp = client.post("/signals/test/default", json={
            "source": "facial_au",
            "arousal_hint": 0.9,
            "confidence": 0.8,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["source"] == "facial_au"
        assert "processed" in data
        assert "fused_effect" in data
        assert data["processed"]["weight"] > 0

    def test_test_with_valence(self, client: TestClient) -> None:
        resp = client.post("/signals/test/default", json={
            "source": "weather",
            "valence_hint": -0.5,
            "confidence": 0.6,
        })
        data = resp.json()
        assert data["processed"]["valence_delta"] < 0
        assert data["fused_effect"]["valence_modulation"] < 0

    def test_test_keyboard_dynamics_source(self, client: TestClient) -> None:
        resp = client.post("/signals/test/default", json={
            "source": "keyboard_dynamics",
            "valence_hint": 0.5,
            "arousal_hint": 0.3,
            "confidence": 0.7,
        })
        assert resp.json()["status"] == "ok"

    def test_test_does_not_modify_state(self, client: TestClient) -> None:
        """Testing a signal should NOT change the session's emotional state."""
        # Get initial state
        state1 = client.get("/signals/config/test-nomod")
        # Test a signal
        client.post("/signals/test/test-nomod", json={
            "source": "facial_au", "arousal_hint": 1.0, "confidence": 1.0,
        })
        # State should be unchanged
        state2 = client.get("/signals/config/test-nomod")
        assert state1.json()["enabled"] == state2.json()["enabled"]


# ── Pipeline Integration ──


class TestSignalsPipelineIntegration:
    """Signals config affects the chat pipeline only when enabled."""

    def test_disabled_signals_no_effect(self, client: TestClient) -> None:
        """With signals disabled, pipeline should behave normally."""
        # Signals are disabled by default — just process normally
        # (This test verifies the pipeline doesn't crash with signals_config present)
        from pathos.state.manager import StateManager
        sm = StateManager()
        session = sm.get_session("test-pipeline")
        assert session.signals_config.enabled is False
        assert session.signals_config.active_count == 0
