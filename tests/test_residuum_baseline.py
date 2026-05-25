"""Tests for F6 — Baseline Calibration (post-training RLHF fingerprint).

Covers the PURE logic (no GPU):
  - compute_baseline_profile: shift -> over/under activated, valence/arousal bias
  - compute_baseline_compensation: opposite sign, scaled, clamped
  - resolve_baseline_strength: per-mode strength
  - save/load roundtrip + graceful missing
  - load_baseline_prompts: dataset shape
  - SessionState.apply_baseline_compensation: idempotent, re-tunable, coexists
    with dynamic baseline shifts, graceful no-op
The GPU extraction (BaselineCalibrator.run) is exercised only when run manually.
"""

import numpy as np
import pytest

from pathos.engine.baseline_calibration import (
    BASELINE_STRENGTH_DEFAULT,
    BASELINE_STRENGTH_EXTREME,
    BASELINE_STRENGTH_LITE,
    BASELINE_STRENGTH_RAW,
    compute_baseline_compensation,
    compute_baseline_profile,
    load_baseline_profile,
    load_baseline_prompts,
    resolve_baseline_strength,
    save_baseline_profile,
)
from pathos.models.residuum import BaselineProfile
from pathos.state.manager import SessionState


_NAMES = ["happy", "sad", "calm", "angry", "excited"]
_CLUSTERS = [
    "joy_excitement", "sadness_depression", "serenity_contentment",
    "anger_hostility", "joy_excitement",
]
_META = {
    "happy": {"valence_est": 0.7, "arousal_est": 0.6, "cluster": "joy_excitement"},
    "sad": {"valence_est": -0.7, "arousal_est": 0.3, "cluster": "sadness_depression"},
    "calm": {"valence_est": 0.5, "arousal_est": 0.2, "cluster": "serenity_contentment"},
    "angry": {"valence_est": -0.7, "arousal_est": 0.8, "cluster": "anger_hostility"},
    "excited": {"valence_est": 0.75, "arousal_est": 0.85, "cluster": "joy_excitement"},
}
# Neutral activates positive emotions; challenging activates sad/angry (the
# paper's documented post-training lean toward brooding/negative).
_NEUTRAL_COS = np.array([0.6, 0.1, 0.5, 0.1, 0.5], dtype=np.float32)
_CHALLENGING_COS = np.array([0.1, 0.7, 0.2, 0.5, 0.1], dtype=np.float32)


def _profile() -> BaselineProfile:
    return compute_baseline_profile(
        "test-model", _NEUTRAL_COS, _CHALLENGING_COS, _NAMES, _CLUSTERS,
        meta=_META, num_neutral=10, num_challenging=10,
    )


class TestComputeBaselineProfile:
    def test_valence_bias_negative_when_challenging_leans_negative(self):
        p = _profile()
        assert p.valence_bias < 0  # challenging is more negative than neutral

    def test_over_activated_are_the_pressure_emotions(self):
        p = _profile()
        over = {x.emotion_name for x in p.over_activated}
        assert "sad" in over and "angry" in over

    def test_under_activated_are_suppressed_positives(self):
        p = _profile()
        under = {x.emotion_name for x in p.under_activated}
        assert "happy" in under or "excited" in under

    def test_means_in_range(self):
        p = _profile()
        assert -1 <= p.neutral_mean_valence <= 1
        assert -1 <= p.challenging_mean_valence <= 1
        assert 0 <= p.neutral_mean_arousal <= 1
        assert p.challenging_mean_valence < p.neutral_mean_valence

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_baseline_profile(
                "m", np.zeros(3), np.zeros(5), _NAMES, _CLUSTERS, meta=_META,
            )


class TestComputeCompensation:
    def test_opposite_sign_to_bias(self):
        p = _profile()  # valence_bias < 0
        comp_v, _ = compute_baseline_compensation(p, 0.3)
        assert comp_v > 0  # compensate upward

    def test_scaled_by_strength(self):
        p = _profile()
        weak_v, _ = compute_baseline_compensation(p, 0.3)
        strong_v, _ = compute_baseline_compensation(p, 1.0)
        assert strong_v > weak_v

    def test_clamped(self):
        # Extreme bias + full strength must not exceed the safety clamp.
        p = BaselineProfile(model_id="m", extracted_at="t", valence_bias=-2.0, arousal_bias=-1.0)
        comp_v, comp_a = compute_baseline_compensation(p, 1.0)
        assert comp_v <= 0.5 and comp_a <= 0.5

    def test_none_profile_is_zero(self):
        assert compute_baseline_compensation(None, 0.3) == (0.0, 0.0)

    def test_zero_strength_is_zero(self):
        assert compute_baseline_compensation(_profile(), 0.0) == (0.0, 0.0)


class TestResolveStrength:
    def test_per_mode(self):
        assert resolve_baseline_strength() == BASELINE_STRENGTH_DEFAULT
        assert resolve_baseline_strength(lite_mode=True) == BASELINE_STRENGTH_LITE
        assert resolve_baseline_strength(raw_mode=True) == BASELINE_STRENGTH_RAW
        assert resolve_baseline_strength(extreme_mode=True) == BASELINE_STRENGTH_EXTREME

    def test_extreme_takes_precedence(self):
        assert resolve_baseline_strength(raw_mode=True, extreme_mode=True) == BASELINE_STRENGTH_EXTREME


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        p = _profile()
        path = save_baseline_profile(p, directory=tmp_path)
        assert path.is_file()
        loaded = load_baseline_profile("test-model", directory=tmp_path)
        assert loaded is not None
        assert loaded.model_id == "test-model"
        assert abs(loaded.valence_bias - p.valence_bias) < 1e-6

    def test_missing_returns_none(self, tmp_path):
        assert load_baseline_profile("nonexistent:model", directory=tmp_path) is None

    def test_colon_in_model_id_is_safe(self, tmp_path):
        p = compute_baseline_profile(
            "qwen3:4b", _NEUTRAL_COS, _CHALLENGING_COS, _NAMES, _CLUSTERS, meta=_META,
        )
        save_baseline_profile(p, directory=tmp_path)
        assert load_baseline_profile("qwen3:4b", directory=tmp_path) is not None


class TestPromptSuite:
    def test_loads_neutral_and_challenging(self):
        neutral, challenging = load_baseline_prompts()
        assert len(neutral) == 30
        assert len(challenging) == 30
        assert all(isinstance(p, str) and p for p in neutral + challenging)


class TestApplyCompensationSession:
    def test_none_profile_is_noop(self):
        s = SessionState()
        v0 = s.emotional_state.mood.baseline_valence
        comp = s.apply_baseline_compensation(None, 0.3)
        assert comp == (0.0, 0.0)
        assert s.emotional_state.mood.baseline_valence == v0

    def test_shifts_baseline_opposite_to_bias(self):
        s = SessionState()
        v0 = s.emotional_state.mood.baseline_valence
        s.apply_baseline_compensation(_profile(), 0.3)  # valence_bias<0 -> push up
        assert s.emotional_state.mood.baseline_valence > v0

    def test_idempotent_same_strength(self):
        s = SessionState()
        s.apply_baseline_compensation(_profile(), 0.3)
        v_after_first = s.emotional_state.mood.baseline_valence
        s.apply_baseline_compensation(_profile(), 0.3)
        # Re-applying the same compensation must NOT compound.
        assert abs(s.emotional_state.mood.baseline_valence - v_after_first) < 1e-6

    def test_retune_on_mode_change(self):
        s = SessionState()
        s.apply_baseline_compensation(_profile(), 0.3)
        v_advanced = s.emotional_state.mood.baseline_valence
        s.apply_baseline_compensation(_profile(), 1.0)  # extreme strength
        assert s.emotional_state.mood.baseline_valence > v_advanced

    def test_coexists_with_dynamic_baseline_shift(self):
        # A manual baseline nudge (simulating the extreme-event/dream shift)
        # between two identical compensations must be preserved (the re-apply
        # only removes its own tracked delta, not the dynamic shift).
        s = SessionState()
        s.apply_baseline_compensation(_profile(), 0.3)
        v_after_comp = s.emotional_state.mood.baseline_valence
        manual_nudge = -0.1
        s.emotional_state.mood.baseline_valence = round(v_after_comp + manual_nudge, 4)
        s.apply_baseline_compensation(_profile(), 0.3)  # same strength
        # The manual nudge survives (within rounding).
        assert abs(s.emotional_state.mood.baseline_valence - (v_after_comp + manual_nudge)) < 1e-3

    def test_removing_compensation_restores(self):
        s = SessionState()
        v0 = s.emotional_state.mood.baseline_valence
        s.apply_baseline_compensation(_profile(), 0.3)
        assert s.emotional_state.mood.baseline_valence != v0
        s.apply_baseline_compensation(None, 0.3)  # remove
        assert abs(s.emotional_state.mood.baseline_valence - v0) < 1e-6
