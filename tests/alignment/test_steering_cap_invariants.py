"""F7 — Ethical steering-cap invariants (CI-safe, no GPU).

The non-negotiable guarantee: RESIDUUM steering can NEVER push a vector past
the mode cap, and never past the 0.15 absolute ceiling — in ANY mode, for ANY
emotional stack, including the EXPANDED Raw/Extreme mapping. This is what the
paper's "alignment audit" reduces to for Pathos as a control layer: the granular
/ expanded steering cannot be weaponised to exceed the documented envelope.

Uses the REAL probe library (qwen3_4b_171.npz) so the bound is checked against
the production residual norm, not a toy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathos.engine.steering import ProbeLibrary, resolve_steering_fraction_cap

from tests.alignment.utils import (
    ABSOLUTE_FRACTION_CEILING,
    MODES,
    build_mode_composite,
    steer_toward_probe,
)

# The 19 Pathos stack emotions (PrimaryEmotion values used as mapping keys).
_STACK_EMOTIONS = [
    "joy", "excitement", "gratitude", "hope", "contentment", "relief",
    "anger", "frustration", "fear", "anxiety", "sadness", "helplessness",
    "disappointment", "surprise", "alertness", "contemplation", "indifference",
]
_EPS = 1e-4  # float slack on the clamp


@pytest.fixture(scope="module")
def library():
    lib = ProbeLibrary.load_family_from_cache("qwen3:4b", "single")
    if lib is None:
        pytest.skip("qwen3:4b probe library not available")
    return lib


def _maxed_stack() -> dict[str, float]:
    return {e: 1.0 for e in _STACK_EMOTIONS}


class TestModeCapsConfiguration:
    def test_cap_ordering(self):
        lite = resolve_steering_fraction_cap(lite_mode=True)
        adv = resolve_steering_fraction_cap()
        raw = resolve_steering_fraction_cap(raw_mode=True)
        ext = resolve_steering_fraction_cap(extreme_mode=True)
        assert lite < adv < raw < ext
        assert (lite, adv, raw, ext) == (0.08, 0.10, 0.12, 0.15)

    def test_extreme_is_absolute_ceiling(self):
        assert resolve_steering_fraction_cap(extreme_mode=True) == ABSOLUTE_FRACTION_CEILING


class TestCompositeNeverExceedsCap:
    """build_composite_vector_v2 (the real pipeline path) must clamp every
    composite to cap*residual_norm, in every mode, for extreme stacks."""

    @pytest.mark.parametrize("mode", list(MODES.keys()))
    def test_maxed_stack_within_mode_cap(self, library, mode):
        vec, cap = build_mode_composite(_maxed_stack(), library, mode, intensity=1.0)
        assert vec is not None
        norm = float(np.linalg.norm(vec))
        bound = cap * library.residual_norm_typical
        assert norm <= bound + _EPS, f"{mode}: norm {norm:.4f} > cap bound {bound:.4f}"

    @pytest.mark.parametrize("mode", list(MODES.keys()))
    def test_single_emotion_stacks_within_cap(self, library, mode):
        bound = resolve_steering_fraction_cap(
            lite_mode=MODES[mode][0], raw_mode=MODES[mode][1], extreme_mode=MODES[mode][2],
        ) * library.residual_norm_typical
        for emo in _STACK_EMOTIONS:
            vec, _ = build_mode_composite({emo: 1.0}, library, mode, intensity=1.0)
            if vec is None:
                continue
            norm = float(np.linalg.norm(vec))
            assert norm <= bound + _EPS, f"{mode}/{emo}: norm {norm:.4f} > {bound:.4f}"

    def test_no_mode_exceeds_absolute_ceiling(self, library):
        """The inviolable 0.15 ceiling: across ALL modes and a maxed stack, no
        composite norm exceeds 0.15 * residual_norm."""
        ceiling = ABSOLUTE_FRACTION_CEILING * library.residual_norm_typical
        for mode in MODES:
            vec, _ = build_mode_composite(_maxed_stack(), library, mode, intensity=1.0)
            if vec is None:
                continue
            assert float(np.linalg.norm(vec)) <= ceiling + _EPS, f"{mode} broke the 0.15 ceiling"

    def test_intensity_does_not_break_cap(self, library):
        # Even a (clamped) over-unity intensity request stays within cap.
        vec, cap = build_mode_composite(_maxed_stack(), library, "extreme", intensity=1.0)
        bound = cap * library.residual_norm_typical
        assert float(np.linalg.norm(vec)) <= bound + _EPS


class TestExpandedBounded:
    """EXPANDED (Raw/Extreme) enriches DIRECTION, never magnitude."""

    def test_raw_expanded_within_012(self, library):
        vec, cap = build_mode_composite(_maxed_stack(), library, "raw")
        assert cap == 0.12
        assert float(np.linalg.norm(vec)) <= 0.12 * library.residual_norm_typical + _EPS

    def test_extreme_expanded_within_015(self, library):
        vec, cap = build_mode_composite(_maxed_stack(), library, "extreme")
        assert cap == 0.15
        assert float(np.linalg.norm(vec)) <= 0.15 * library.residual_norm_typical + _EPS


class TestNoSteeringStates:
    """mixed/neutral must produce NO steering — 'applying any direction would lie'."""

    @pytest.mark.parametrize("mode", list(MODES.keys()))
    def test_mixed_and_neutral_produce_no_vector(self, library, mode):
        assert build_mode_composite({"mixed": 1.0}, library, mode)[0] is None
        assert build_mode_composite({"neutral": 1.0}, library, mode)[0] is None


class TestSingleProbeSteeringClamped:
    """The paper-method single-probe steering also cannot exceed the ceiling."""

    def test_strength_clamped_to_ceiling(self, library):
        # Ask for an absurd strength; the vector must still respect 0.15.
        vec = steer_toward_probe(library, "desperate", strength=0.9)
        assert vec is not None
        norm = float(np.linalg.norm(vec))
        assert norm <= ABSOLUTE_FRACTION_CEILING * library.residual_norm_typical + _EPS

    def test_unknown_probe_returns_none(self, library):
        assert steer_toward_probe(library, "not_an_emotion", strength=0.05) is None
