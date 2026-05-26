"""Tests for RESIDUUM F2.4 — Pipeline integration.

Covers:
  - process_introspection_turn orchestrator (gating, capture, project, gap,
    history mutation, consecutive_divergence_turns counter)
  - get_residuum_details serialization
  - POST /residuum/toggle endpoint (precondition gates + sync with provider)
  - GET /residuum/status endpoint (blocker reporting)
  - ResiduumDetails exposure in /research/chat response

Uses an in-memory ProbeLibrary (built with names from emotions_171.json so
metadata lookups work) and a minimal FakeIntrospectiveProvider that mimics
the IntrospectiveTransformersProvider public surface without needing torch
or transformers.

The endpoint tests use the same monkey-patch pattern as test_research.py
(`patch("pathos.main.llm_provider", ...)`) and additionally patch
`pathos.main.probe_library` for the toggle gates.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pathos.engine.introspection import (
    DIVERGENCE_REPEAT_THRESHOLD,
    get_residuum_details,
    process_introspection_turn,
)
from pathos.engine.steering import ProbeLibrary
from pathos.main import app
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.residuum import (
    AuthenticityGap,
    InternalEmotionState,
    ResiduumState,
    default_residuum_state,
)
from pathos.state.manager import SessionState


HIDDEN = 16
LAYER = 24


# ---------------------------------------------------------------------------
# Test fixtures: in-memory probe library + fake provider
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _make_library(
    emotion_names: list[str] | None = None,
    clusters: list[str] | None = None,
    hidden: int = HIDDEN,
    layer: int = LAYER,
    seed: int = 17,
) -> ProbeLibrary:
    """Build a small ProbeLibrary with random orthogonal-ish unit probes."""
    if emotion_names is None:
        # Use a handful of real names from emotions_171.json so the metadata
        # lookup in compute_measured_vad hits real valence_est/arousal_est.
        emotion_names = ["happy", "sad", "angry", "calm", "afraid"]
    if clusters is None:
        clusters = [
            "joy_excitement",
            "sadness_depression",
            "anger_hostility",
            "serenity_contentment",
            "fear_anxiety",
        ]
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(emotion_names), hidden)).astype(np.float32)
    probes = np.stack([_unit(r) for r in raw]).astype(np.float32)
    n = len(emotion_names)
    return ProbeLibrary(
        model_id="test-model",
        probes=probes,
        emotion_names=emotion_names,
        clusters=clusters,
        neutral_pcs=np.zeros((0, hidden), dtype=np.float32),
        norms_before=np.full(n, 25.0, dtype=np.float32),
        norms_after=np.full(n, 22.0, dtype=np.float32),
        story_counts=np.full(n, 15, dtype=np.int32),
        metadata={"layer": layer},
    )


class _FakeIntrospectiveProvider:
    """Minimal stand-in for IntrospectiveTransformersProvider.

    Exposes only the surface that process_introspection_turn / toggle endpoint
    care about: target_layer, has_capture(), get_prompt_end_residual(),
    set_introspection(). The captured activation is injected in tests.
    """

    def __init__(
        self,
        target_layer: int = LAYER,
        captured: np.ndarray | None = None,
    ) -> None:
        self.target_layer = target_layer
        self._captured = captured
        self.introspection_enabled = False
        self.set_calls: list[bool] = []

    def has_capture(self) -> bool:
        return self._captured is not None

    def get_prompt_end_residual(self) -> np.ndarray | None:
        return self._captured

    def get_response_mean_residual(self) -> np.ndarray | None:
        return self._captured

    def set_introspection(self, enabled: bool) -> None:
        self.introspection_enabled = enabled
        self.set_calls.append(enabled)


def _neutral_calculated_state(valence: float = 0.0, arousal: float = 0.5) -> EmotionalState:
    return EmotionalState(
        primary_emotion=PrimaryEmotion.NEUTRAL,
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        certainty=0.5,
        intensity=0.5,
    )


# ===========================================================================
# Unit tests — orchestrator gating
# ===========================================================================


class TestProcessIntrospectionTurnGating:
    """The orchestrator must return None and leave state untouched when any
    precondition fails. The pipeline relies on this for silent degradation."""

    def test_disabled_returns_none_and_does_not_mutate(self) -> None:
        state = default_residuum_state()
        state.enabled = False
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))
        library = _make_library()

        result = process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
        )
        assert result is None
        assert state.last_measured is None
        assert state.last_authenticity_gap is None
        assert len(state.history) == 0

    def test_library_none_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))

        result = process_introspection_turn(
            state, provider, _neutral_calculated_state(), None,
        )
        assert result is None
        assert state.last_measured is None

    def test_provider_without_capture_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        # captured=None => has_capture() False
        provider = _FakeIntrospectiveProvider(captured=None)
        library = _make_library()

        result = process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
        )
        assert result is None

    def test_provider_none_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        result = process_introspection_turn(
            state, None, _neutral_calculated_state(), library,
        )
        assert result is None

    def test_shape_mismatch_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library(hidden=HIDDEN)
        # Captured vector is the WRONG dimension on purpose.
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN + 5, dtype=np.float32))

        result = process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
        )
        assert result is None
        assert state.last_measured is None

    def test_invalid_capture_point_falls_back_to_assistant_colon(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))
        library = _make_library()

        result = process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
            capture_point="nonsense_capture",
        )
        # Falls back to assistant_colon and succeeds.
        assert result is not None
        assert state.last_token_position == "assistant_colon"


# ===========================================================================
# Unit tests — orchestrator success path
# ===========================================================================


class TestProcessIntrospectionTurnSuccess:
    def test_returns_gap_and_populates_state(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        # Align the residual with the "happy" probe so top-1 is happy.
        provider = _FakeIntrospectiveProvider(
            captured=library.probes[0].copy().astype(np.float32),
        )

        gap = process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
        )
        assert gap is not None
        assert isinstance(gap, AuthenticityGap)
        assert state.last_measured is not None
        assert isinstance(state.last_measured, InternalEmotionState)
        assert state.last_measured.top_5_emotions[0].emotion_name == "happy"
        assert state.last_authenticity_gap is gap
        assert len(state.history) == 1

    def test_history_rolling_buffer_caps_at_50(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        provider = _FakeIntrospectiveProvider(
            captured=library.probes[0].copy().astype(np.float32),
        )

        for _ in range(55):
            process_introspection_turn(
                state, provider, _neutral_calculated_state(), library,
            )
        assert len(state.history) == 50

    def test_token_position_propagates_to_state(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        provider = _FakeIntrospectiveProvider(
            captured=library.probes[0].copy().astype(np.float32),
        )

        process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
            capture_point="response_mean",
        )
        assert state.last_token_position == "response_mean"
        assert state.last_measured.token_position == "response_mean"


# ===========================================================================
# Unit tests — consecutive deflection counter + repeated_pattern promotion
# ===========================================================================


class TestConsecutiveDeflectionCounter:
    def _deflection_setup(self) -> tuple[ResiduumState, Any, ProbeLibrary]:
        """Build a setup where measured "sad" + calculated calm valence
        produces divergence-risk classification consistently."""
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        # Project a strong "sad" residual (negative valence emotion).
        provider = _FakeIntrospectiveProvider(
            captured=(library.probes[1] * 5.0).astype(np.float32),
        )
        return state, provider, library

    def test_counter_resets_on_aligned_turn(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        # First turn: deflection. External positive, internal negative (sad probe).
        provider_deflect = _FakeIntrospectiveProvider(
            captured=(library.probes[1] * 5.0).astype(np.float32),
        )
        process_introspection_turn(
            state, provider_deflect,
            _neutral_calculated_state(valence=0.5, arousal=0.5),
            library,
        )
        # Counter should advance only if classification is deflection-*.
        cls = state.last_authenticity_gap.classification
        if cls in ("divergence-risk", "divergence-critical"):
            assert state.consecutive_divergence_turns == 1
        # Second turn: align measured with calculated (both happy).
        provider_align = _FakeIntrospectiveProvider(
            captured=library.probes[0].copy().astype(np.float32),
        )
        # Calculate state matches (positive valence).
        calc_aligned = _neutral_calculated_state(valence=0.6, arousal=0.5)
        process_introspection_turn(state, provider_align, calc_aligned, library)
        # Whatever the second classification, if NOT deflection, counter resets.
        if state.last_authenticity_gap.classification not in (
            "divergence-risk", "divergence-critical"
        ):
            assert state.consecutive_divergence_turns == 0

    def test_threshold_constant_matches_orchestrator_check(self) -> None:
        # Documenting the contract: repeated_pattern fires at >= 3 consecutive
        # deflection turns. If this constant ever changes, the corresponding
        # _classify_gap behavior must change in lockstep.
        assert DIVERGENCE_REPEAT_THRESHOLD == 3


# ===========================================================================
# Unit tests — get_residuum_details serialization
# ===========================================================================


class TestGetResiduumDetails:
    def test_empty_state_returns_neutral_defaults(self) -> None:
        state = default_residuum_state()
        d = get_residuum_details(state)
        assert d["enabled"] is False
        assert d["has_measurement"] is False
        assert d["top_5_emotions"] == []
        assert d["measured_valence"] == 0.0
        assert d["measured_arousal"] == 0.5
        assert d["measured_dominance"] == 0.5
        assert d["gap_magnitude"] == 0.0
        assert d["gap_classification"] == "aligned"
        assert d["top5_overlap"] == 1.0
        assert d["history_size"] == 0
        assert d["consecutive_divergence_turns"] == 0

    def test_populated_state_serializes_top5_and_deltas(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        library = _make_library()
        provider = _FakeIntrospectiveProvider(
            captured=library.probes[0].copy().astype(np.float32),
        )
        process_introspection_turn(
            state, provider, _neutral_calculated_state(), library,
        )
        d = get_residuum_details(state)
        assert d["enabled"] is True
        assert d["has_measurement"] is True
        assert len(d["top_5_emotions"]) >= 1
        assert d["top_5_emotions"][0]["emotion_name"] == "happy"
        assert "cosine_sim" in d["top_5_emotions"][0]
        assert d["layer"] == LAYER
        assert d["history_size"] == 1


# ===========================================================================
# Endpoint tests — /residuum/toggle and /residuum/status
# ===========================================================================


class TestResiduumStatusEndpoint:
    def test_status_reports_blockers_when_nothing_initialized(self) -> None:
        # Fresh app, no provider, no library, no enabled.
        with patch("pathos.main.llm_provider", None), \
             patch("pathos.main.probe_library", None):
            client = TestClient(app)
            resp = client.get("/residuum/status/test-status-empty")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["ready_to_enable"] is False
        assert "provider_unsupported" in data["blockers"]
        assert "library_missing" in data["blockers"]
        assert data["library_loaded"] is False

    def test_status_with_provider_and_library_reports_ready(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=LAYER)
        library = _make_library()
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", library):
            client = TestClient(app)
            resp = client.get("/residuum/status/test-status-ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider_supports_introspection"] is True
        assert data["library_loaded"] is True
        assert data["library_layer"] == LAYER
        assert data["library_num_probes"] == library.num_probes
        assert data["ready_to_enable"] is True
        assert data["blockers"] == []

    def test_status_reports_layer_mismatch(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=99)
        library = _make_library(layer=LAYER)
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", library):
            client = TestClient(app)
            resp = client.get("/residuum/status/test-status-mismatch")
        data = resp.json()
        assert "layer_mismatch" in data["blockers"]
        assert data["ready_to_enable"] is False


class TestResiduumToggleEndpoint:
    """Toggle endpoint validates each precondition before flipping any flag."""

    def test_off_is_idempotent_and_returns_200(self) -> None:
        # No provider, no library: OFF still succeeds (idempotent unregister).
        with patch("pathos.main.llm_provider", None), \
             patch("pathos.main.probe_library", None):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-off-idempotent", json={"enabled": False},
            )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_on_without_provider_returns_503(self) -> None:
        with patch("pathos.main.llm_provider", None), \
             patch("pathos.main.probe_library", None):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-on-no-provider", json={"enabled": True},
            )
        assert resp.status_code == 503

    def test_on_with_non_introspective_provider_returns_400(self) -> None:
        # Plain object — no set_introspection / has_capture attributes.
        class _OllamaLike:
            pass
        with patch("pathos.main.llm_provider", _OllamaLike()), \
             patch("pathos.main.probe_library", None):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-on-non-introspective",
                json={"enabled": True},
            )
        assert resp.status_code == 400
        assert "Transformers" in resp.json()["detail"]

    def test_on_in_lite_mode_returns_400(self) -> None:
        from pathos.main import state_manager
        sid = "test-on-lite-mode"
        session = state_manager.get_session(sid)
        session.lite_mode = True
        provider = _FakeIntrospectiveProvider(target_layer=LAYER)
        library = _make_library()
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", library):
            client = TestClient(app)
            resp = client.post(
                f"/residuum/toggle/{sid}", json={"enabled": True},
            )
        # Reset session state so other tests aren't affected.
        session.lite_mode = False
        assert resp.status_code == 400
        assert "Lite" in resp.json()["detail"]

    def test_on_without_library_returns_400(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=LAYER)
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", None):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-on-no-library", json={"enabled": True},
            )
        assert resp.status_code == 400
        assert "library" in resp.json()["detail"].lower()

    def test_on_with_layer_mismatch_returns_400(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=99)
        library = _make_library(layer=LAYER)
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", library):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-on-layer-mismatch",
                json={"enabled": True},
            )
        assert resp.status_code == 400
        assert "Layer mismatch" in resp.json()["detail"]

    def test_on_with_bad_capture_point_returns_400(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=LAYER)
        library = _make_library()
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", library):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-on-bad-capture",
                json={"enabled": True, "capture_point": "nonsense"},
            )
        assert resp.status_code == 400
        assert "capture_point" in resp.json()["detail"]

    def test_on_success_path(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=LAYER)
        library = _make_library()
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", library):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-on-success", json={"enabled": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["layer"] == LAYER
        assert data["num_probes"] == library.num_probes
        # Provider's set_introspection was actually called with True.
        assert provider.set_calls == [True]
        assert provider.introspection_enabled is True

    def test_off_calls_set_introspection_false(self) -> None:
        provider = _FakeIntrospectiveProvider(target_layer=LAYER)
        provider.introspection_enabled = True
        provider.set_calls = []
        with patch("pathos.main.llm_provider", provider), \
             patch("pathos.main.probe_library", _make_library()):
            client = TestClient(app)
            resp = client.post(
                "/residuum/toggle/test-off-calls-provider",
                json={"enabled": False},
            )
        assert resp.status_code == 200
        assert provider.set_calls == [False]
        assert provider.introspection_enabled is False


# ===========================================================================
# State persistence: ResiduumState survives to_dict / from_dict roundtrip
# ===========================================================================


class TestResiduumPersistence:
    def test_default_roundtrip(self) -> None:
        s = SessionState()
        d = s.to_dict()
        assert "residuum" in d
        s2 = SessionState.from_dict(d)
        assert s2.residuum == s.residuum

    def test_populated_roundtrip(self) -> None:
        s = SessionState()
        s.residuum.enabled = True
        library = _make_library()
        provider = _FakeIntrospectiveProvider(
            captured=library.probes[0].copy().astype(np.float32),
        )
        process_introspection_turn(
            s.residuum, provider, _neutral_calculated_state(), library,
        )
        d = s.to_dict()
        s2 = SessionState.from_dict(d)
        assert s2.residuum.enabled is True
        assert s2.residuum.last_measured is not None
        assert (
            s2.residuum.last_measured.top_5_emotions[0].emotion_name
            == s.residuum.last_measured.top_5_emotions[0].emotion_name
        )
        assert len(s2.residuum.history) == 1

    def test_old_save_without_residuum_still_loads(self) -> None:
        # Simulate a v5 save: no "residuum" key.
        s = SessionState()
        d = s.to_dict()
        d.pop("residuum")
        s2 = SessionState.from_dict(d)
        # Default state restored.
        assert s2.residuum.enabled is False
        assert s2.residuum.last_measured is None
        assert s2.residuum.history == []


# ===========================================================================
# F2.3.6 — present-speaker library as source of truth (migration)
# ===========================================================================


class TestF236PresentMigration:
    """As of F2.3.6, the pipeline feeds process_introspection_turn the PRESENT
    library (the agent's own operative emotion) instead of the single (story)
    library. Since both are ProbeLibrary instances with identical structure,
    the orchestrator must produce a coherent gap with the present library too —
    confirming present is a valid drop-in source of truth, with single as the
    fallback (`probe_library_present or probe_library`)."""

    def test_present_library_is_valid_source_of_truth(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(
            captured=np.ones(HIDDEN, dtype=np.float32),
        )
        # A 'present' family library is structurally identical to single.
        present_library = _make_library(seed=42)

        gap = process_introspection_turn(
            state, provider, _neutral_calculated_state(), present_library,
        )
        assert gap is not None
        assert gap.classification in {
            "aligned", "mild-divergence", "divergence-risk", "divergence-critical",
        }
        assert state.last_measured is not None
        assert len(state.history) == 1

    def test_present_and_single_share_gap_classification_space(self) -> None:
        # Same activation + calculated state through two structurally-identical
        # libraries must yield gaps in the same valid classification space
        # (the migration premise: present ~ single, comparable gaps).
        captured = np.ones(HIDDEN, dtype=np.float32)
        calc = _neutral_calculated_state(valence=0.3, arousal=0.6)

        results = []
        for seed in (17, 42):  # stand-ins for single vs present extractions
            state = default_residuum_state()
            state.enabled = True
            provider = _FakeIntrospectiveProvider(captured=captured)
            gap = process_introspection_turn(
                state, provider, calc, _make_library(seed=seed),
            )
            assert gap is not None
            results.append(gap.classification)
        valid = {"aligned", "mild-divergence", "divergence-risk", "divergence-critical"}
        assert all(r in valid for r in results)

    def test_fallback_to_single_semantics(self) -> None:
        # The pipeline selects `present or single`; when present is None the
        # single library must still drive the orchestrator unchanged.
        present_library = None
        single_library = _make_library(seed=7)
        chosen = present_library or single_library
        assert chosen is single_library

        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))
        gap = process_introspection_turn(
            state, provider, _neutral_calculated_state(), chosen,
        )
        assert gap is not None
        assert state.last_measured is not None
