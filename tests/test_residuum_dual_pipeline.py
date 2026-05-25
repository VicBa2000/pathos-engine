"""Tests for RESIDUUM F2.3.4 — Dual probes wired to the pipeline.

Covers the runtime side of F2.3 (NPZ extraction itself is exercised in
test_residuum_dual.py):
  - ProbeLibrary.load_family_from_cache: backward-compat, unknown family,
    suffix routing to present/other NPZ.
  - project_dual: shape validation, token_position propagation, returns two
    InternalEmotionState.
  - process_introspection_turn_dual: gating (disabled, libraries None,
    capture missing, shape mismatch), happy path writes the new
    last_measured_present/last_measured_other fields, does NOT touch the
    single-library fields (regression guard for "single remains source of
    truth until F2.3.5+").
  - get_residuum_details: has_dual_measurement flag + present/other fields
    serialized when both states are populated, empty otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathos.engine.introspection import (
    get_residuum_details,
    process_introspection_turn,
    process_introspection_turn_dual,
    project_dual,
)
from pathos.engine.steering import ProbeLibrary
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.residuum import (
    InternalEmotionState,
    ResiduumState,
    default_residuum_state,
)


HIDDEN = 16
LAYER = 24


# ---------------------------------------------------------------------------
# Shared helpers (mirror the style of test_residuum_pipeline.py)
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


_NAMES = ["happy", "sad", "angry", "calm", "afraid"]
_CLUSTERS = [
    "joy_excitement",
    "sadness_depression",
    "anger_hostility",
    "serenity_contentment",
    "fear_anxiety",
]


def _make_library(
    hidden: int = HIDDEN,
    layer: int = LAYER,
    seed: int = 17,
    model_id: str = "test-model",
) -> ProbeLibrary:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(_NAMES), hidden)).astype(np.float32)
    probes = np.stack([_unit(r) for r in raw]).astype(np.float32)
    n = len(_NAMES)
    return ProbeLibrary(
        model_id=model_id,
        probes=probes,
        emotion_names=list(_NAMES),
        clusters=list(_CLUSTERS),
        neutral_pcs=np.zeros((0, hidden), dtype=np.float32),
        norms_before=np.full(n, 25.0, dtype=np.float32),
        norms_after=np.full(n, 22.0, dtype=np.float32),
        story_counts=np.full(n, 15, dtype=np.int32),
        metadata={"layer": layer},
    )


class _FakeIntrospectiveProvider:
    """Minimal IntrospectiveTransformersProvider stand-in."""

    def __init__(
        self,
        target_layer: int = LAYER,
        captured: np.ndarray | None = None,
    ) -> None:
        self.target_layer = target_layer
        self._captured = captured

    def has_capture(self) -> bool:
        return self._captured is not None

    def get_prompt_end_residual(self) -> np.ndarray | None:
        return self._captured

    def get_response_mean_residual(self) -> np.ndarray | None:
        return self._captured


def _neutral_state() -> EmotionalState:
    return EmotionalState(
        primary_emotion=PrimaryEmotion.NEUTRAL,
        valence=0.0,
        arousal=0.5,
        dominance=0.5,
        certainty=0.5,
        intensity=0.5,
    )


def _write_minimal_npz(path: Path, hidden: int = HIDDEN, layer: int = LAYER) -> None:
    """Write a minimal NPZ that ProbeLibrary.load_family_from_cache accepts.

    Only the required fields ('probes', 'emotion_names', 'clusters') are
    written; optional fields default to safe zero arrays.
    """
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((len(_NAMES), hidden)).astype(np.float32)
    probes = np.stack([_unit(r) for r in raw]).astype(np.float32)
    np.savez(
        path,
        probes=probes,
        emotion_names=np.array(_NAMES),
        clusters=np.array(_CLUSTERS),
        metadata=np.array([f'{{"layer": {layer}}}']),
    )


# ===========================================================================
# load_family_from_cache
# ===========================================================================


class TestLoadFamilyFromCache:
    """ProbeLibrary.load_family_from_cache routes by family suffix and stays
    graceful when the NPZ is missing or the family name is unknown."""

    def test_unknown_family_returns_none(self) -> None:
        # Even when the directory exists, an unknown family is graceful.
        assert ProbeLibrary.load_family_from_cache("any:model", "invalid") is None

    def test_missing_npz_returns_none(self, tmp_path: Path) -> None:
        assert ProbeLibrary.load_family_from_cache(
            "qwen3:4b", "single", directory=tmp_path,
        ) is None
        assert ProbeLibrary.load_family_from_cache(
            "qwen3:4b", "present", directory=tmp_path,
        ) is None

    def test_load_from_cache_equivalent_to_family_single(self, tmp_path: Path) -> None:
        # Write only the single-suffix NPZ.
        safe = "qwen3_4b"
        _write_minimal_npz(tmp_path / f"{safe}_171.npz")

        via_legacy = ProbeLibrary.load_from_cache("qwen3:4b", directory=tmp_path)
        via_family = ProbeLibrary.load_family_from_cache(
            "qwen3:4b", "single", directory=tmp_path,
        )
        assert via_legacy is not None
        assert via_family is not None
        assert via_legacy.num_probes == via_family.num_probes
        assert via_legacy.hidden_size == via_family.hidden_size
        assert via_legacy.emotion_names == via_family.emotion_names
        np.testing.assert_allclose(via_legacy.probes, via_family.probes)

    def test_load_family_routes_to_correct_suffix(self, tmp_path: Path) -> None:
        safe = "qwen3_4b"
        # Only 'present' file is present on disk.
        _write_minimal_npz(tmp_path / f"{safe}_171_present.npz")

        present = ProbeLibrary.load_family_from_cache(
            "qwen3:4b", "present", directory=tmp_path,
        )
        other = ProbeLibrary.load_family_from_cache(
            "qwen3:4b", "other", directory=tmp_path,
        )
        single = ProbeLibrary.load_family_from_cache(
            "qwen3:4b", "single", directory=tmp_path,
        )
        assert present is not None
        assert other is None
        assert single is None


# ===========================================================================
# project_dual
# ===========================================================================


class TestProjectDual:
    """project_dual must validate shapes, return two InternalEmotionState with
    propagated token_position, and reuse project_residual semantics."""

    def test_returns_two_states_with_token_position(self) -> None:
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        activation = lib_p.probes[0].copy() + 0.1 * lib_o.probes[2]

        present, other = project_dual(
            activation, lib_p, lib_o, token_position="user_turn_end", k=3,
        )
        assert isinstance(present, InternalEmotionState)
        assert isinstance(other, InternalEmotionState)
        assert present.token_position == "user_turn_end"
        assert other.token_position == "user_turn_end"
        assert present.layer == LAYER
        assert other.layer == LAYER
        # k=3 honored on both sides.
        assert len(present.top_5_emotions) == 3
        assert len(other.top_5_emotions) == 3

    def test_raises_on_hidden_size_mismatch(self) -> None:
        lib_p = _make_library(hidden=HIDDEN, seed=11)
        lib_o = _make_library(hidden=HIDDEN + 4, seed=23)
        activation = np.zeros(HIDDEN, dtype=np.float32)
        with pytest.raises(ValueError, match="hidden_size mismatch"):
            project_dual(activation, lib_p, lib_o)

    def test_raises_on_activation_shape_mismatch(self) -> None:
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        bad = np.zeros(HIDDEN + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="hidden_size"):
            project_dual(bad, lib_p, lib_o)

    def test_default_token_position_is_assistant_colon(self) -> None:
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        present, other = project_dual(
            np.ones(HIDDEN, dtype=np.float32), lib_p, lib_o,
        )
        assert present.token_position == "assistant_colon"
        assert other.token_position == "assistant_colon"

    def test_present_and_other_independent_projections(self) -> None:
        # Build two libraries where the activation exactly equals probe[0] of
        # present and probe[2] of other. Top-1 of each side must hit its own
        # library's emotion.
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        activation = lib_p.probes[0] + lib_o.probes[2]

        present, other = project_dual(activation, lib_p, lib_o, k=1)
        # Strongest cosine on each side must come from its own library.
        assert present.top_5_emotions[0].emotion_name == _NAMES[0]
        assert other.top_5_emotions[0].emotion_name == _NAMES[2]


# ===========================================================================
# process_introspection_turn_dual — gating
# ===========================================================================


class TestProcessIntrospectionTurnDualGating:

    def test_disabled_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = False
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)

        result = process_introspection_turn_dual(state, provider, lib_p, lib_o)
        assert result is None
        assert state.last_measured_present is None
        assert state.last_measured_other is None

    def test_library_present_none_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))
        lib_o = _make_library(seed=23)

        result = process_introspection_turn_dual(state, provider, None, lib_o)
        assert result is None
        assert state.last_measured_present is None
        assert state.last_measured_other is None

    def test_library_other_none_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(captured=np.ones(HIDDEN, dtype=np.float32))
        lib_p = _make_library(seed=11)

        result = process_introspection_turn_dual(state, provider, lib_p, None)
        assert result is None
        assert state.last_measured_present is None

    def test_provider_without_capture_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(captured=None)
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)

        result = process_introspection_turn_dual(state, provider, lib_p, lib_o)
        assert result is None

    def test_shape_mismatch_returns_none(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        provider = _FakeIntrospectiveProvider(
            captured=np.ones(HIDDEN + 3, dtype=np.float32),
        )
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)

        result = process_introspection_turn_dual(state, provider, lib_p, lib_o)
        assert result is None
        assert state.last_measured_present is None
        assert state.last_measured_other is None

    def test_libraries_with_different_hidden_size_return_none_silently(self) -> None:
        # Both libraries match the activation, but they differ between each
        # other — orchestrator should return None instead of raising.
        state = default_residuum_state()
        state.enabled = True
        lib_p = _make_library(hidden=HIDDEN, seed=11)
        lib_o = _make_library(hidden=HIDDEN, seed=23)
        # Force a mismatch by overwriting one library's hidden_size after the
        # fact (the activation still has shape HIDDEN; this exercises the
        # defensive check inside process_introspection_turn_dual).
        lib_o.probes = np.zeros((len(_NAMES), HIDDEN + 2), dtype=np.float32)
        provider = _FakeIntrospectiveProvider(
            captured=np.ones(HIDDEN, dtype=np.float32),
        )

        result = process_introspection_turn_dual(state, provider, lib_p, lib_o)
        assert result is None
        assert state.last_measured_present is None


# ===========================================================================
# process_introspection_turn_dual — happy path + regression
# ===========================================================================


class TestProcessIntrospectionTurnDualSuccess:

    def test_writes_present_and_other_states(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        captured = lib_p.probes[0] + 0.5 * lib_o.probes[1]
        provider = _FakeIntrospectiveProvider(captured=captured)

        result = process_introspection_turn_dual(
            state, provider, lib_p, lib_o, capture_point="assistant_colon",
        )
        assert result is not None
        present, other = result
        assert state.last_measured_present is present
        assert state.last_measured_other is other
        assert present.token_position == "assistant_colon"
        assert other.token_position == "assistant_colon"
        assert len(present.top_5_emotions) == 5
        assert len(other.top_5_emotions) == 5

    def test_does_not_mutate_single_library_fields(self) -> None:
        """Regression: dual orchestrator must NOT touch last_measured /
        last_authenticity_gap / history / consecutive_divergence_turns. Those
        belong to the single-library path until F2.3.5+ migration."""
        state = default_residuum_state()
        state.enabled = True
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        provider = _FakeIntrospectiveProvider(
            captured=np.ones(HIDDEN, dtype=np.float32),
        )

        before_history = list(state.history)
        before_divergence = state.consecutive_divergence_turns
        before_measured = state.last_measured
        before_gap = state.last_authenticity_gap

        result = process_introspection_turn_dual(state, provider, lib_p, lib_o)
        assert result is not None

        # Dual fields populated.
        assert state.last_measured_present is not None
        assert state.last_measured_other is not None
        # Single-library fields untouched.
        assert state.last_measured is before_measured
        assert state.last_authenticity_gap is before_gap
        assert state.history == before_history
        assert state.consecutive_divergence_turns == before_divergence

    def test_invalid_capture_point_falls_back_to_assistant_colon(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        provider = _FakeIntrospectiveProvider(
            captured=np.ones(HIDDEN, dtype=np.float32),
        )

        result = process_introspection_turn_dual(
            state, provider, lib_p, lib_o, capture_point="nonsense",
        )
        assert result is not None
        present, other = result
        assert present.token_position == "assistant_colon"
        assert other.token_position == "assistant_colon"


# ===========================================================================
# Single-path regression: process_introspection_turn unaffected
# ===========================================================================


class TestSinglePathRegressionWithDualLibrariesPresent:
    """process_introspection_turn (single library) must keep working exactly
    as before regardless of whether dual libraries are also loaded. The
    F2.3.4 wiring decision keeps single as the source of truth for the gap
    classification; this is the safety net for that choice."""

    def test_single_path_still_produces_gap_when_dual_unused(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        single = _make_library(seed=7)
        captured = single.probes[0].copy()
        provider = _FakeIntrospectiveProvider(captured=captured)

        gap = process_introspection_turn(
            state, provider, _neutral_state(), single,
        )
        assert gap is not None
        assert state.last_measured is not None
        assert state.last_authenticity_gap is gap
        # Dual fields untouched (we never called the dual orchestrator).
        assert state.last_measured_present is None
        assert state.last_measured_other is None


# ===========================================================================
# get_residuum_details — has_dual_measurement + dual fields
# ===========================================================================


class TestGetResiduumDetailsDual:

    def test_no_dual_measurement_yields_empty_dual_fields(self) -> None:
        state = default_residuum_state()
        details = get_residuum_details(state)
        assert details["has_dual_measurement"] is False
        assert details["present_top_5_emotions"] == []
        assert details["other_top_5_emotions"] == []
        assert details["present_layer"] == -1
        assert details["other_layer"] == -1

    def test_only_single_measurement_keeps_dual_flag_false(self) -> None:
        # Single-library run populates last_measured but NOT the dual fields.
        state = default_residuum_state()
        state.enabled = True
        single = _make_library(seed=7)
        provider = _FakeIntrospectiveProvider(
            captured=single.probes[0].copy(),
        )
        process_introspection_turn(state, provider, _neutral_state(), single)

        details = get_residuum_details(state)
        assert details["has_measurement"] is True
        assert details["has_dual_measurement"] is False
        assert details["present_top_5_emotions"] == []
        assert details["other_top_5_emotions"] == []

    def test_dual_measurement_populates_present_and_other_fields(self) -> None:
        state = default_residuum_state()
        state.enabled = True
        lib_p = _make_library(seed=11)
        lib_o = _make_library(seed=23)
        provider = _FakeIntrospectiveProvider(
            captured=lib_p.probes[0] + 0.4 * lib_o.probes[3],
        )

        process_introspection_turn_dual(state, provider, lib_p, lib_o)

        details = get_residuum_details(state)
        assert details["has_dual_measurement"] is True
        # Top-5 lists populated for both sides.
        assert len(details["present_top_5_emotions"]) == 5
        assert len(details["other_top_5_emotions"]) == 5
        # Each entry has the canonical projection shape used by the frontend.
        first_present = details["present_top_5_emotions"][0]
        assert set(first_present.keys()) == {
            "emotion_name", "cluster", "cosine_sim", "raw_activation",
        }
        assert details["present_layer"] == LAYER
        assert details["other_layer"] == LAYER

    def test_dual_layer_reflects_individual_libraries(self) -> None:
        # Allow present and other to come from different layers; the dict must
        # carry each one independently.
        state = default_residuum_state()
        state.enabled = True
        lib_p = _make_library(seed=11, layer=20)
        lib_o = _make_library(seed=23, layer=28)
        provider = _FakeIntrospectiveProvider(
            captured=np.ones(HIDDEN, dtype=np.float32),
        )
        process_introspection_turn_dual(state, provider, lib_p, lib_o)

        details = get_residuum_details(state)
        assert details["present_layer"] == 20
        assert details["other_layer"] == 28
