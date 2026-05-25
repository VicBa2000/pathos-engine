"""Tests for RESIDUUM F4.3 — SteeringHook V2 (granular 171-probe path).

Covers the additive V2 wiring on SteeringHook:
  - use_v2 property: True only when library + stack + mapping are provided.
  - _compute_raw_vectors:
      * V2 path returns {target_layer: composite_vec}
      * V1 path remains untouched (legacy multilayer composite)
      * V2 with no steerable emotion returns {}
      * V2 with negative layer (malformed library) returns {} with warning
  - apply():
      * V2 path bypasses engine.is_ready (granular composite does not need
        cached 4D vectors); only requires a model with transformer layers.
      * V1 path still requires engine.is_ready.
  - version_used diagnostic flips correctly across paths.

The tests use a FakeTransformersModel with the minimum surface needed to
register forward hooks (model.model.layers list), so we don't need torch
weights or a real LLM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pathos.engine.steering import (
    MAX_STEERING_FRACTION_DEFAULT,
    CachedVectors,
    EmotionalSteeringEngine,
    ProbeLibrary,
    SteeringHook,
    load_stack_to_probe_map,
)


HIDDEN = 16
LAYER = 24
NUM_LAYERS = 36


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_library(
    emotion_names: list[str] | None = None,
    hidden: int = HIDDEN,
    layer: int = LAYER,
    seed: int = 17,
) -> ProbeLibrary:
    if emotion_names is None:
        emotion_names = ["sad", "happy", "joyful", "ecstatic", "angry"]
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(emotion_names), hidden)).astype(np.float32)
    probes = np.stack([_unit(r) for r in raw]).astype(np.float32)
    n = len(emotion_names)
    return ProbeLibrary(
        model_id="test-model",
        probes=probes,
        emotion_names=emotion_names,
        clusters=["c"] * n,
        neutral_pcs=np.zeros((0, hidden), dtype=np.float32),
        norms_before=np.full(n, 10.0, dtype=np.float32),
        norms_after=np.full(n, 9.0, dtype=np.float32),
        story_counts=np.full(n, 15, dtype=np.int32),
        metadata={"layer": layer},
    )


def _make_engine_with_v1_vectors() -> EmotionalSteeringEngine:
    """An engine pre-loaded with synthetic V1 cached vectors so is_ready=True."""
    engine = EmotionalSteeringEngine()
    rng = np.random.default_rng(42)
    vectors: dict[str, dict[int, np.ndarray]] = {}
    for dim in ("valence", "arousal", "dominance", "certainty"):
        layer_vecs: dict[int, np.ndarray] = {}
        for li in (9, 18, 27):  # early, mid, late
            v = rng.standard_normal(HIDDEN).astype(np.float32)
            layer_vecs[li] = _unit(v)
        vectors[dim] = layer_vecs
    engine._cached = CachedVectors(
        model_id="test",
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        vectors=vectors,
    )
    return engine


class _FakeLayer:
    """Minimal transformer layer with a forward-hook register."""

    def __init__(self) -> None:
        self._handles: list = []

    def register_forward_hook(self, fn) -> Any:
        # Return a dummy handle object that has .remove()
        class _Handle:
            def __init__(self_inner) -> None:
                self_inner.removed = False

            def remove(self_inner) -> None:
                self_inner.removed = True
        h = _Handle()
        self._handles.append(h)
        return h


class _FakeModelInner:
    def __init__(self, n_layers: int = NUM_LAYERS) -> None:
        self.layers = [_FakeLayer() for _ in range(n_layers)]


class _FakeTransformersModel:
    """Just enough for _get_model_layers + register_forward_hook to work."""

    def __init__(self, n_layers: int = NUM_LAYERS) -> None:
        self.model = _FakeModelInner(n_layers=n_layers)


# ===========================================================================
# use_v2 property
# ===========================================================================


class TestUseV2Property:
    def test_pure_v1_inputs_is_not_v2(self) -> None:
        engine = EmotionalSteeringEngine()
        hook = SteeringHook(None, engine, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert hook.use_v2 is False

    def test_library_only_is_not_v2(self) -> None:
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 0.5, probe_library=lib,
        )
        assert hook.use_v2 is False

    def test_library_and_stack_only_is_not_v2(self) -> None:
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 0.5,
            probe_library=lib, stack={"sadness": 1.0},
        )
        assert hook.use_v2 is False

    def test_all_three_inputs_makes_v2(self) -> None:
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 0.5,
            probe_library=lib, stack={"sadness": 1.0}, mapping={"sadness": ["sad"]},
        )
        assert hook.use_v2 is True

    def test_empty_stack_still_v2_path(self) -> None:
        # use_v2 cares about wiring, not about whether there's steerable
        # content. An empty stack returns {} in _compute_raw_vectors but the
        # path used is still V2.
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 0.5,
            probe_library=lib, stack={}, mapping={"sadness": ["sad"]},
        )
        assert hook.use_v2 is True


# ===========================================================================
# _compute_raw_vectors routing
# ===========================================================================


class TestComputeRawVectorsRouting:
    def test_v2_returns_single_layer_at_library_target(self) -> None:
        engine = EmotionalSteeringEngine()
        lib = _make_library(layer=24)
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"sadness": 1.0},
            mapping={"sadness": ["sad"]},
            residual_norm=lib.residual_norm_typical,
        )
        vecs = hook._compute_raw_vectors()
        assert list(vecs.keys()) == [24]
        assert vecs[24].shape == (HIDDEN,)

    def test_v2_empty_stack_returns_empty(self) -> None:
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib, stack={}, mapping={"sadness": ["sad"]},
        )
        assert hook._compute_raw_vectors() == {}

    def test_v2_all_empty_mapping_returns_empty(self) -> None:
        # stack with only mixed/neutral whose mapping is empty -> nothing to hook.
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"mixed": 1.0, "neutral": 1.0},
            mapping={"sadness": ["sad"], "mixed": [], "neutral": []},
        )
        assert hook._compute_raw_vectors() == {}

    def test_v2_negative_layer_in_library_returns_empty(self) -> None:
        # Defensive: a malformed library with layer=-1 must not produce hooks.
        engine = EmotionalSteeringEngine()
        lib = _make_library(layer=-1)
        hook = SteeringHook(
            None, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"sadness": 1.0},
            mapping={"sadness": ["sad"]},
        )
        assert hook._compute_raw_vectors() == {}

    def test_v1_unchanged_when_no_v2_inputs(self) -> None:
        # V1 should still emit multi-layer vectors as before. Use an engine
        # with cached 4D vectors so v1 path actually produces output.
        engine = _make_engine_with_v1_vectors()
        hook = SteeringHook(None, engine, 0.7, 0.6, 0.5, 0.5, 1.0)
        vecs = hook._compute_raw_vectors()
        assert set(vecs.keys()) == {9, 18, 27}


# ===========================================================================
# apply() path selection
# ===========================================================================


class TestApplyPathSelection:
    def test_v2_apply_succeeds_even_when_engine_not_ready(self) -> None:
        # The whole point of V2: granular path runs without engine.is_ready.
        engine = EmotionalSteeringEngine()  # no cached vectors -> not ready
        assert engine.is_ready is False
        lib = _make_library()
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        hook = SteeringHook(
            model, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"sadness": 1.0},
            mapping={"sadness": ["sad"]},
            residual_norm=lib.residual_norm_typical,
        )
        ok = hook.apply()
        assert ok is True
        assert hook.version_used == "v2"
        # One hook at the library's layer.
        assert set(hook.vectors_applied.keys()) == {LAYER}
        hook.remove()
        assert hook.version_used == "none"

    def test_v1_apply_requires_engine_ready(self) -> None:
        engine = EmotionalSteeringEngine()  # not ready
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        hook = SteeringHook(model, engine, 0.5, 0.5, 0.5, 0.5, 1.0)
        ok = hook.apply()
        assert ok is False
        # version_used is set inside apply only when we proceeded past readiness;
        # since we returned early, it stays at the constructor default 'none'.
        assert hook.version_used == "none"

    def test_v1_apply_succeeds_when_engine_ready(self) -> None:
        engine = _make_engine_with_v1_vectors()
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        hook = SteeringHook(model, engine, 0.7, 0.6, 0.5, 0.5, 1.0)
        ok = hook.apply()
        assert ok is True
        assert hook.version_used == "v1"
        # All three V1 layers should be hooked.
        assert set(hook.vectors_applied.keys()) == {9, 18, 27}
        hook.remove()

    def test_v2_apply_returns_false_when_nothing_steerable(self) -> None:
        # V2 path but stack is empty -> no hooks registered, returns False.
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        hook = SteeringHook(
            model, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib, stack={}, mapping={"sadness": ["sad"]},
        )
        ok = hook.apply()
        assert ok is False
        # version_used flipped to 'v2' during apply (we reached past readiness),
        # since use_v2 was True even though no hook was registered.
        assert hook.version_used == "v2"

    def test_v2_hook_norm_recorded_correctly(self) -> None:
        engine = EmotionalSteeringEngine()
        lib = _make_library()
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        hook = SteeringHook(
            model, engine, 0.5, 0.5, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"sadness": 1.0},
            mapping={"sadness": ["sad"]},
            residual_norm=lib.residual_norm_typical,
            fraction_cap=MAX_STEERING_FRACTION_DEFAULT,
        )
        hook.apply()
        # Single layer hook, norm > 0 and respects cap.
        assert LAYER in hook.vectors_applied
        norm = hook.vectors_applied[LAYER]
        assert norm > 0
        assert norm <= MAX_STEERING_FRACTION_DEFAULT * lib.residual_norm_typical + 1e-5


# ===========================================================================
# Integration with the real JSON mapping
# ===========================================================================


class TestV2WithRealMapping:
    def test_v2_uses_standard_json_for_advanced_session(self) -> None:
        # Build a library covering 'joy' composite (joyful + happy + ecstatic).
        lib = _make_library(emotion_names=["joyful", "happy", "ecstatic"])
        engine = EmotionalSteeringEngine()
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        mapping = load_stack_to_probe_map("standard")
        hook = SteeringHook(
            model, engine, 0.6, 0.7, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"joy": 1.0},
            mapping=mapping,
            residual_norm=lib.residual_norm_typical,
        )
        ok = hook.apply()
        assert ok is True
        # Expected direction: mean of the 3 joy probes.
        expected_dir = np.mean(
            np.stack([lib.get_probe(p) for p in ("joyful", "happy", "ecstatic")]),
            axis=0,
        )
        expected_unit = expected_dir / np.linalg.norm(expected_dir)
        applied_vec = hook.raw_vectors[LAYER]
        actual_unit = applied_vec / np.linalg.norm(applied_vec)
        assert float(actual_unit @ expected_unit) > 0.9999
        hook.remove()

    def test_v2_uses_restricted_json_for_lite_mapping(self) -> None:
        # RESTRICTED has joy -> ['joyful'] only.
        lib = _make_library(emotion_names=["joyful", "happy"])
        engine = EmotionalSteeringEngine()
        model = _FakeTransformersModel(n_layers=NUM_LAYERS)
        mapping = load_stack_to_probe_map("restricted")
        hook = SteeringHook(
            model, engine, 0.6, 0.7, 0.5, 0.5, 1.0,
            probe_library=lib,
            stack={"joy": 1.0},
            mapping=mapping,
            residual_norm=lib.residual_norm_typical,
        )
        hook.apply()
        joyful = lib.get_probe("joyful")
        applied_vec = hook.raw_vectors[LAYER]
        cos = float(applied_vec @ joyful) / float(np.linalg.norm(applied_vec))
        assert cos > 0.9999
        hook.remove()
