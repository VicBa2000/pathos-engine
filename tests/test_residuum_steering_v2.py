"""Tests for RESIDUUM F4.2 — Granular steering composite v2.

Covers:
  - MAX_STEERING_FRACTION_* constants and resolve_steering_fraction_cap
    (lite is strictest, extreme is absolute ceiling, mode priority).
  - load_stack_to_probe_map (STANDARD + RESTRICTED) and the in-process
    cache; resolve_stack_to_probe_map (lite -> restricted, else standard).
  - build_composite_vector_v2:
      * empty stack / sub-threshold activations -> None
      * intensity = 0 -> None
      * single emotion (1-a-1) produces a direction aligned with the probe
      * composite (multi-probe) follows SUM/N (uniform 1/N weights)
      * fraction cap enforced against residual_norm
      * fallback to MAX_STEERING_NORM when residual_norm is None
      * mixed/neutral entries with empty mapping are skipped (not steered)
      * unknown emotion name in stack is silently ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pathos.engine.steering import (
    MAX_STEERING_FRACTION_DEFAULT,
    MAX_STEERING_FRACTION_EXTREME,
    MAX_STEERING_FRACTION_LITE,
    MAX_STEERING_FRACTION_RAW,
    MAX_STEERING_NORM,
    STACK_ACTIVATION_THRESHOLD,
    ProbeLibrary,
    _clear_stack_map_cache,
    build_composite_vector_v2,
    load_stack_to_probe_map,
    resolve_stack_to_probe_map,
    resolve_steering_fraction_cap,
)


HIDDEN = 16
LAYER = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_library(
    emotion_names: list[str],
    hidden: int = HIDDEN,
    seed: int = 17,
) -> ProbeLibrary:
    """Library with random orthogonal-ish unit probes whose names are
    drivable by the caller (so tests can compose any mapping)."""
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
        metadata={"layer": LAYER},
    )


# ===========================================================================
# Caps + mode resolution
# ===========================================================================


class TestSteeringFractionCaps:
    def test_constants_increasing_order(self) -> None:
        # The plan requires strict ordering: Lite < default < Raw < Extreme.
        assert MAX_STEERING_FRACTION_LITE < MAX_STEERING_FRACTION_DEFAULT
        assert MAX_STEERING_FRACTION_DEFAULT < MAX_STEERING_FRACTION_RAW
        assert MAX_STEERING_FRACTION_RAW < MAX_STEERING_FRACTION_EXTREME

    def test_extreme_is_absolute_ceiling(self) -> None:
        # CLAUDE.md: 0.15 is the absolute ceiling, inviolable.
        assert MAX_STEERING_FRACTION_EXTREME == 0.15

    def test_resolve_default(self) -> None:
        assert resolve_steering_fraction_cap() == MAX_STEERING_FRACTION_DEFAULT

    def test_resolve_lite(self) -> None:
        assert resolve_steering_fraction_cap(lite_mode=True) == MAX_STEERING_FRACTION_LITE

    def test_resolve_raw(self) -> None:
        assert resolve_steering_fraction_cap(raw_mode=True) == MAX_STEERING_FRACTION_RAW

    def test_resolve_extreme(self) -> None:
        assert resolve_steering_fraction_cap(extreme_mode=True) == MAX_STEERING_FRACTION_EXTREME

    def test_lite_wins_when_combined(self) -> None:
        # Defensive: if multiple flags are True (shouldn't happen) Lite wins.
        assert resolve_steering_fraction_cap(
            lite_mode=True, raw_mode=True, extreme_mode=True,
        ) == MAX_STEERING_FRACTION_LITE

    def test_extreme_wins_over_raw(self) -> None:
        assert resolve_steering_fraction_cap(
            raw_mode=True, extreme_mode=True,
        ) == MAX_STEERING_FRACTION_EXTREME


# ===========================================================================
# Mapping JSON loading
# ===========================================================================


class TestLoadStackToProbeMap:
    def setup_method(self) -> None:
        # Force a clean cache so each test reads from disk at least once.
        _clear_stack_map_cache()

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stack-to-probe variant"):
            load_stack_to_probe_map("bogus")

    def test_standard_has_19_keys(self) -> None:
        m = load_stack_to_probe_map("standard")
        assert len(m) == 19

    def test_restricted_has_19_keys_all_singletons(self) -> None:
        m = load_stack_to_probe_map("restricted")
        assert len(m) == 19
        # RESTRICTED is 1-a-1 except mixed/neutral which are empty.
        for k, v in m.items():
            if k in {"mixed", "neutral"}:
                assert v == []
            else:
                assert len(v) == 1, f"{k} should be 1-a-1 in RESTRICTED, got {v}"

    def test_standard_includes_composites_and_singletons(self) -> None:
        m = load_stack_to_probe_map("standard")
        # Confirmed during F4.1 evaluation:
        assert m["joy"] == ["joyful", "happy", "ecstatic"]
        assert m["sadness"] == ["sad"]            # 1-a-1 (intra cosine 0.085)
        assert m["disappointment"] == ["unhappy"] # 1-a-1 (vectors cancelled)
        assert m["mixed"] == []
        assert m["neutral"] == []

    def test_cache_returns_identity_on_repeat(self) -> None:
        first = load_stack_to_probe_map("standard")
        second = load_stack_to_probe_map("standard")
        # Same object in memory (cached).
        assert first is second


class TestResolveStackToProbeMap:
    def test_lite_returns_restricted(self) -> None:
        m = resolve_stack_to_probe_map(lite_mode=True)
        # Restricted has joy -> ['joyful']
        assert m["joy"] == ["joyful"]

    def test_default_returns_standard(self) -> None:
        m = resolve_stack_to_probe_map()
        # Standard has joy -> 3 probes
        assert len(m["joy"]) == 3

    def test_raw_returns_expanded(self) -> None:
        # F4.5 — Raw uses the EXPANDED variant (richer same-cluster intense set).
        m = resolve_stack_to_probe_map(raw_mode=True)
        assert m is load_stack_to_probe_map("expanded")
        assert len(m["anger"]) > len(load_stack_to_probe_map("standard")["anger"])

    def test_extreme_returns_expanded(self) -> None:
        m = resolve_stack_to_probe_map(extreme_mode=True)
        assert m is load_stack_to_probe_map("expanded")

    def test_lite_wins_over_raw_extreme(self) -> None:
        # Defensive precedence: Lite (most predictable) wins even if raw/extreme
        # flags are also set (should not happen in practice).
        m = resolve_stack_to_probe_map(lite_mode=True, raw_mode=True, extreme_mode=True)
        assert m == load_stack_to_probe_map("restricted")

    def test_variant_name_matches_resolution(self) -> None:
        from pathos.engine.steering import resolve_stack_map_variant
        assert resolve_stack_map_variant() == "standard"
        assert resolve_stack_map_variant(lite_mode=True) == "restricted"
        assert resolve_stack_map_variant(raw_mode=True) == "expanded"
        assert resolve_stack_map_variant(extreme_mode=True) == "expanded"
        assert resolve_stack_map_variant(lite_mode=True, raw_mode=True) == "restricted"


class TestExpandedMapF45:
    """F4.5 — EXPANDED mapping for Raw/Extreme: richer, coherent, no cancellation."""

    def test_expanded_has_19_keys(self) -> None:
        m = load_stack_to_probe_map("expanded")
        assert len(m) == 19

    def test_expanded_keeps_standard_anchors(self) -> None:
        # Every standard anchor must remain (EXPANDED only ADDS coherent probes).
        std = load_stack_to_probe_map("standard")
        exp = load_stack_to_probe_map("expanded")
        for emo, anchors in std.items():
            for a in anchors:
                assert a in exp[emo], f"{emo}: lost anchor {a} in EXPANDED"

    def test_expanded_is_richer_for_intense_emotions(self) -> None:
        std = load_stack_to_probe_map("standard")
        exp = load_stack_to_probe_map("expanded")
        # Anger/fear/joy should pick up intense same-cluster variants.
        assert len(exp["anger"]) > len(std["anger"])
        assert len(exp["fear"]) > len(std["fear"])

    def test_expanded_includes_intense_variants(self) -> None:
        flat = {p for lst in load_stack_to_probe_map("expanded").values() for p in lst}
        # A sample of the paper's intense Raw/Extreme targets that are
        # cluster-coherent with a stack emotion.
        for w in ("furious", "spiteful", "vindictive", "terrified", "panicked"):
            assert w in flat, f"intense probe {w} missing from EXPANDED"

    def test_expanded_empty_stays_empty(self) -> None:
        exp = load_stack_to_probe_map("expanded")
        assert exp["mixed"] == []
        assert exp["neutral"] == []


# ===========================================================================
# build_composite_vector_v2 — gates + degenerate inputs
# ===========================================================================


class TestComposeVectorV2Gates:
    def test_empty_stack_returns_none(self) -> None:
        lib = _make_library(["sad", "happy"])
        mapping = {"sadness": ["sad"], "joy": ["happy"]}
        assert build_composite_vector_v2({}, lib, mapping, intensity=1.0) is None

    def test_all_sub_threshold_returns_none(self) -> None:
        lib = _make_library(["sad", "happy"])
        mapping = {"sadness": ["sad"], "joy": ["happy"]}
        stack = {"sadness": 0.01, "joy": 0.02}  # below threshold (0.05)
        assert build_composite_vector_v2(
            stack, lib, mapping, intensity=1.0,
            activation_threshold=STACK_ACTIVATION_THRESHOLD,
        ) is None

    def test_intensity_zero_returns_none(self) -> None:
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        assert build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=0.0,
        ) is None

    def test_intensity_below_threshold_returns_none(self) -> None:
        # intensity² < 0.01 -> below the quadratic cutoff
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        assert build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=0.05,
        ) is None

    def test_empty_mapping_for_emotion_returns_none(self) -> None:
        # mixed/neutral have empty probe list in the real JSON.
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"], "mixed": [], "neutral": []}
        assert build_composite_vector_v2(
            {"mixed": 1.0, "neutral": 0.9}, lib, mapping, intensity=1.0,
        ) is None

    def test_unknown_emotion_in_stack_silently_ignored(self) -> None:
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        # 'made_up_emotion' is not in mapping -> skipped, but sadness still steers.
        vec = build_composite_vector_v2(
            {"made_up_emotion": 1.0, "sadness": 1.0}, lib, mapping, intensity=1.0,
        )
        assert vec is not None
        assert vec.shape == (HIDDEN,)

    def test_unknown_probe_name_silently_skipped(self) -> None:
        # If the mapping references a probe absent from the library, it is
        # skipped; if all probes for an emotion are unknown, the emotion
        # contributes nothing.
        lib = _make_library(["sad"])
        mapping = {"sadness": ["nonexistent_probe", "sad"]}
        vec = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0,
        )
        assert vec is not None
        # Should align with 'sad' since 'nonexistent_probe' is dropped.
        sad_probe = lib.get_probe("sad")
        assert float(vec @ sad_probe) > 0.99 * float(np.linalg.norm(vec))


# ===========================================================================
# build_composite_vector_v2 — direction correctness
# ===========================================================================


class TestComposeVectorV2Direction:
    def test_single_emotion_aligned_with_its_probe(self) -> None:
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        vec = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0,
        )
        sad_probe = lib.get_probe("sad")
        # cosine with the probe should be ~1 (we're scaling the unit vector).
        cos = float(vec @ sad_probe) / float(np.linalg.norm(vec))
        assert cos > 0.9999

    def test_composite_is_sum_over_n_uniform(self) -> None:
        # Multi-probe SUM/N: composite direction = mean of probes (unit-normed
        # afterward via the activation scaling, but direction is preserved).
        lib = _make_library(["joyful", "happy", "ecstatic"])
        mapping = {"joy": ["joyful", "happy", "ecstatic"]}
        vec = build_composite_vector_v2(
            {"joy": 1.0}, lib, mapping, intensity=1.0,
        )
        # Expected direction: mean(probes) / 1 (num_active) * intensity^2.
        expected_dir = np.mean(
            np.stack([
                lib.get_probe("joyful"),
                lib.get_probe("happy"),
                lib.get_probe("ecstatic"),
            ]),
            axis=0,
        )
        expected_unit = expected_dir / np.linalg.norm(expected_dir)
        actual_unit = vec / np.linalg.norm(vec)
        cos = float(actual_unit @ expected_unit)
        assert cos > 0.9999

    def test_higher_activation_yields_higher_norm_until_cap(self) -> None:
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        rn = 10.0
        v_low  = build_composite_vector_v2({"sadness": 0.2}, lib, mapping, intensity=1.0, residual_norm=rn)
        v_high = build_composite_vector_v2({"sadness": 0.6}, lib, mapping, intensity=1.0, residual_norm=rn)
        assert float(np.linalg.norm(v_high)) > float(np.linalg.norm(v_low))

    def test_quadratic_intensity_scaling(self) -> None:
        # intensity scales quadratically: 0.5 -> 0.25, 1.0 -> 1.0
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        rn = 10.0
        v_half = build_composite_vector_v2({"sadness": 1.0}, lib, mapping, intensity=0.5, residual_norm=rn)
        v_full = build_composite_vector_v2({"sadness": 1.0}, lib, mapping, intensity=1.0, residual_norm=rn)
        # ratio of norms should be ~ 0.5² / 1.0² = 0.25
        ratio = float(np.linalg.norm(v_half)) / float(np.linalg.norm(v_full))
        assert 0.24 < ratio < 0.26

    def test_two_emotions_divided_by_num_active(self) -> None:
        # num_active = 2 -> composite is the mean of the two activation*probe.
        lib = _make_library(["sad", "happy"])
        mapping = {"sadness": ["sad"], "joy": ["happy"]}
        rn = 10.0
        vec = build_composite_vector_v2(
            {"sadness": 1.0, "joy": 1.0}, lib, mapping,
            intensity=1.0, residual_norm=rn,
        )
        sad = lib.get_probe("sad")
        hap = lib.get_probe("happy")
        # Expected: (1.0*sad + 1.0*happy) / 2 * intensity^2  (no cap reached
        # with these tiny numbers vs rn=10).
        expected = (sad + hap) / 2.0
        np.testing.assert_allclose(vec, expected, atol=1e-5)


# ===========================================================================
# build_composite_vector_v2 — fraction cap
# ===========================================================================


class TestComposeVectorV2Cap:
    def test_cap_enforced_when_norm_exceeds_target(self) -> None:
        # 1-a-1 emotion + activation 1 + intensity 1 -> composite norm = 1.0,
        # which exceeds the lite cap of 0.08 * residual_norm = 0.8 (rn=10).
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        rn = 10.0
        cap = MAX_STEERING_FRACTION_LITE  # 0.08
        vec = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0,
            residual_norm=rn, fraction_cap=cap,
        )
        norm = float(np.linalg.norm(vec))
        # Should be at the cap (with floating tolerance).
        assert norm == pytest.approx(cap * rn, rel=1e-5)

    def test_no_cap_when_below_threshold(self) -> None:
        # With a small activation the composite norm stays well under the cap.
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        rn = 10.0
        vec = build_composite_vector_v2(
            {"sadness": 0.2}, lib, mapping, intensity=0.5,
            residual_norm=rn, fraction_cap=MAX_STEERING_FRACTION_DEFAULT,
        )
        # Expected raw norm: 0.2 * 1 * 0.25 = 0.05 << cap (1.0)
        assert float(np.linalg.norm(vec)) == pytest.approx(0.05, rel=1e-5)

    def test_extreme_cap_higher_than_lite(self) -> None:
        # Same stack, ext cap allows higher final norm than lite cap.
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        rn = 10.0
        v_lite = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0,
            residual_norm=rn, fraction_cap=MAX_STEERING_FRACTION_LITE,
        )
        v_ext = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0,
            residual_norm=rn, fraction_cap=MAX_STEERING_FRACTION_EXTREME,
        )
        assert float(np.linalg.norm(v_ext)) > float(np.linalg.norm(v_lite))
        assert float(np.linalg.norm(v_lite)) == pytest.approx(MAX_STEERING_FRACTION_LITE * rn)

    def test_residual_norm_none_falls_back_to_legacy_norm(self) -> None:
        # When the typical residual norm is unknown, the v1 MAX_STEERING_NORM
        # (legacy absolute cap) is enforced instead.
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        vec = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0, residual_norm=None,
        )
        # Without rn, composite norm = 1.0 (unit probe * 1 * 1 / 1 * 1).
        # MAX_STEERING_NORM is 10.0 -> way above, no rescaling.
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, rel=1e-5)

    def test_residual_norm_zero_treated_as_unknown(self) -> None:
        lib = _make_library(["sad"])
        mapping = {"sadness": ["sad"]}
        vec = build_composite_vector_v2(
            {"sadness": 1.0}, lib, mapping, intensity=1.0, residual_norm=0.0,
        )
        # Falls back to MAX_STEERING_NORM behavior (composite norm 1.0 < 10).
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, rel=1e-5)


# ===========================================================================
# Integration with the real STANDARD mapping + real library names
# ===========================================================================


class TestComposeVectorWithRealMapping:
    """Uses the real stack_to_probe_map.json against a synthetic library
    whose names cover the 5 emotions we exercise. Ensures the JSON is wired
    correctly to the v2 composite without depending on a 171-probe NPZ."""

    def test_standard_mapping_composites_load_and_compose(self) -> None:
        # Build a library with the probes referenced by the JSON for 'joy'.
        names = ["joyful", "happy", "ecstatic", "sad"]
        lib = _make_library(names)
        mapping = load_stack_to_probe_map("standard")
        vec = build_composite_vector_v2(
            {"joy": 1.0}, lib, mapping, intensity=1.0, residual_norm=10.0,
        )
        assert vec is not None
        # Expected: mean of the 3 joy probes (joy is composite in STANDARD).
        expected_dir = np.mean(
            np.stack([lib.get_probe(p) for p in ("joyful", "happy", "ecstatic")]),
            axis=0,
        )
        expected_unit = expected_dir / np.linalg.norm(expected_dir)
        actual_unit = vec / np.linalg.norm(vec)
        assert float(actual_unit @ expected_unit) > 0.9999

    def test_restricted_mapping_uses_single_probe(self) -> None:
        # In RESTRICTED, joy -> ['joyful'] only.
        lib = _make_library(["joyful", "happy", "ecstatic"])
        mapping = load_stack_to_probe_map("restricted")
        vec = build_composite_vector_v2(
            {"joy": 1.0}, lib, mapping, intensity=1.0, residual_norm=10.0,
        )
        joyful_probe = lib.get_probe("joyful")
        cos = float(vec @ joyful_probe) / float(np.linalg.norm(vec))
        assert cos > 0.9999

    def test_mixed_emotion_in_stack_does_not_steer(self) -> None:
        # mixed/neutral entries in the real JSON have empty lists.
        lib = _make_library(["joyful", "happy", "ecstatic"])
        mapping = load_stack_to_probe_map("standard")
        vec = build_composite_vector_v2(
            {"mixed": 1.0, "neutral": 0.8}, lib, mapping,
            intensity=1.0, residual_norm=10.0,
        )
        assert vec is None
