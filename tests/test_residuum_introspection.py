"""Tests for RESIDUUM F2.2 — Introspection Engine.

Covers:
  - project_residual: top-1 == probe when activation == probe; top-k ordering.
  - compute_measured_vad: bounds, weighted average, empty input, peakiness.
  - compute_authenticity_gap: deltas, magnitude, jaccard overlap, classification
    (aligned / mild-divergence / divergence-risk / divergence-critical).

Two layers:
  TestSyntheticIntrospection — uses an in-memory ProbeLibrary built with names
                               drawn from emotions_171.json, so VAD lookups hit
                               real metadata. Always runs.
  TestRealIntrospection      — uses src/pathos/steering_data/cached_vectors/
                               qwen3_4b_171.npz if present. Skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathos.engine.introspection import (
    ALIGNED_MAGNITUDE,
    ALIGNED_OVERLAP,
    DIVERGENCE_CRITICAL_MAGNITUDE,
    compute_authenticity_gap,
    compute_measured_vad,
    project_residual,
)
from pathos.engine.steering import ProbeLibrary
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.residuum import (
    AuthenticityGap,
    EmotionProjection,
    InternalEmotionState,
)


_STEERING_DATA = (
    Path(__file__).parent.parent / "src" / "pathos" / "steering_data"
)
_REAL_NPZ = _STEERING_DATA / "cached_vectors" / "qwen3_4b_171.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _make_library(
    emotion_names: list[str],
    clusters: list[str],
    hidden: int = 16,
    seed: int = 17,
) -> ProbeLibrary:
    """Build an in-memory ProbeLibrary with orthogonal-ish unit probes."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(emotion_names), hidden)).astype(np.float32)
    probes = np.stack([_unit(r) for r in raw]).astype(np.float32)
    n = len(emotion_names)
    return ProbeLibrary(
        model_id="synthetic",
        probes=probes,
        emotion_names=emotion_names,
        clusters=clusters,
        neutral_pcs=np.zeros((0, hidden), dtype=np.float32),
        norms_before=np.full(n, 25.0, dtype=np.float32),
        norms_after=np.full(n, 22.0, dtype=np.float32),
        story_counts=np.full(n, 15, dtype=np.int32),
        metadata={"layer": 24},
    )


# Pool of real emotion names so VAD metadata lookup succeeds. Cluster column
# matches emotions_171.json so dominance heuristic also resolves.
_NAMES_AND_CLUSTERS: list[tuple[str, str]] = [
    # negatives (low valence)
    ("afraid", "fear_anxiety"),                # v=-0.70 a=0.75
    ("angry", "anger_hostility"),              # v=-0.70 a=0.80
    ("ashamed", "shame_guilt"),                # v=-0.60 a=0.40
    ("anxious", "fear_anxiety"),               # v=-0.55 a=0.70
    ("depressed", "sadness_depression"),       # v=-0.80 a=0.20
    # positives (high valence)
    ("blissful", "amusement_playfulness"),     # v=+0.90 a=0.45
    ("cheerful", "amusement_playfulness"),     # v=+0.70 a=0.60
    ("calm", "serenity_contentment"),          # v=+0.50 a=0.20
    ("delighted", "amusement_playfulness"),    # v=+0.80 a=0.60
    ("compassionate", "love_warmth"),          # v=+0.60 a=0.40
]


def _make_real_named_library(hidden: int = 16) -> ProbeLibrary:
    names = [n for n, _ in _NAMES_AND_CLUSTERS]
    clusters = [c for _, c in _NAMES_AND_CLUSTERS]
    return _make_library(names, clusters, hidden=hidden)


# ---------------------------------------------------------------------------
# project_residual
# ---------------------------------------------------------------------------


class TestProjectResidual:
    def test_activation_equals_probe_returns_that_probe_top1(self) -> None:
        lib = _make_real_named_library(hidden=16)
        # Pick the 3rd probe and use it as the activation.
        probe = lib.probes[2].copy()
        state = project_residual(probe, lib, k=5)
        assert isinstance(state, InternalEmotionState)
        assert state.top_5_emotions[0].emotion_name == lib.emotion_names[2]
        # Top-1 cosine should be ~1.0
        assert state.top_5_emotions[0].cosine_sim == pytest.approx(1.0, abs=1e-3)

    def test_returns_exactly_k_emotions(self) -> None:
        lib = _make_real_named_library(hidden=16)
        rng = np.random.default_rng(0)
        act = rng.standard_normal(16).astype(np.float32)
        state = project_residual(act, lib, k=5)
        assert len(state.top_5_emotions) == 5

    def test_top_k_ordered_by_abs_cosine_desc(self) -> None:
        lib = _make_real_named_library(hidden=16)
        rng = np.random.default_rng(1)
        act = rng.standard_normal(16).astype(np.float32)
        state = project_residual(act, lib, k=5)
        abs_cos = [abs(p.cosine_sim) for p in state.top_5_emotions]
        assert abs_cos == sorted(abs_cos, reverse=True)

    def test_token_position_propagates(self) -> None:
        lib = _make_real_named_library()
        state = project_residual(
            lib.probes[0].copy(), lib, k=5, token_position="user_turn_end"
        )
        assert state.token_position == "user_turn_end"

    def test_layer_propagates_from_library(self) -> None:
        lib = _make_real_named_library()
        state = project_residual(lib.probes[0].copy(), lib, k=3)
        assert state.layer == lib.layer
        assert state.layer == 24

    def test_k_smaller_than_library(self) -> None:
        lib = _make_real_named_library(hidden=16)
        state = project_residual(lib.probes[0].copy(), lib, k=2)
        assert len(state.top_5_emotions) == 2

    def test_k_larger_than_pydantic_cap_clamps_to_5(self) -> None:
        """Pydantic InternalEmotionState caps top_5_emotions at 5 — k clamps too."""
        lib = _make_real_named_library(hidden=16)  # 10 probes available
        state = project_residual(lib.probes[0].copy(), lib, k=999)
        assert len(state.top_5_emotions) == 5

    def test_zero_activation_returns_zero_cosines(self) -> None:
        lib = _make_real_named_library(hidden=16)
        zero = np.zeros(16, dtype=np.float32)
        state = project_residual(zero, lib, k=5)
        assert all(p.cosine_sim == 0.0 for p in state.top_5_emotions)

    def test_negated_activation_top1_keeps_same_emotion(self) -> None:
        """abs(cosine) ordering: negating a probe still gives that probe top-1."""
        lib = _make_real_named_library(hidden=16)
        act = -lib.probes[0].copy()
        state = project_residual(act, lib, k=1)
        assert state.top_5_emotions[0].emotion_name == lib.emotion_names[0]
        assert state.top_5_emotions[0].cosine_sim == pytest.approx(-1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# compute_measured_vad
# ---------------------------------------------------------------------------


class TestComputeMeasuredVAD:
    def test_empty_input_returns_neutral(self) -> None:
        lib = _make_real_named_library()
        v, a, d, c = compute_measured_vad([], lib)
        assert v == 0.0
        assert a == 0.5
        assert d == 0.5
        assert c == 0.0

    def test_zero_weights_returns_neutral(self) -> None:
        lib = _make_real_named_library()
        zero_projs = [
            EmotionProjection(
                emotion_name="afraid", cluster="fear_anxiety",
                cosine_sim=0.0, raw_activation=0.0,
            )
        ]
        v, a, d, c = compute_measured_vad(zero_projs, lib)
        assert (v, a, d, c) == (0.0, 0.5, 0.5, 0.0)

    def test_single_negative_emotion_drives_valence_negative(self) -> None:
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(
                emotion_name="depressed", cluster="sadness_depression",
                cosine_sim=0.9, raw_activation=10.0,
            )
        ]
        v, a, d, c = compute_measured_vad(projs, lib)
        # depressed: valence_est=-0.80, arousal_est=0.20
        assert v == pytest.approx(-0.80, abs=0.01)
        assert a == pytest.approx(0.20, abs=0.01)
        assert d == pytest.approx(0.20, abs=0.01)  # cluster sadness_depression
        assert c == pytest.approx(0.9, abs=0.01)

    def test_single_positive_emotion_drives_valence_positive(self) -> None:
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(
                emotion_name="blissful", cluster="amusement_playfulness",
                cosine_sim=0.8, raw_activation=8.0,
            )
        ]
        v, _, d, _ = compute_measured_vad(projs, lib)
        assert v == pytest.approx(0.90, abs=0.01)
        assert d == pytest.approx(0.65, abs=0.01)  # amusement_playfulness

    def test_weighted_average_of_two_emotions(self) -> None:
        """Equal weights -> arithmetic mean of valence_est."""
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(  # valence_est=-0.80
                emotion_name="depressed", cluster="sadness_depression",
                cosine_sim=0.5, raw_activation=5.0,
            ),
            EmotionProjection(  # valence_est=+0.90
                emotion_name="blissful", cluster="amusement_playfulness",
                cosine_sim=0.5, raw_activation=5.0,
            ),
        ]
        v, _, _, _ = compute_measured_vad(projs, lib)
        # Weighted average: (-0.80 + 0.90) / 2 = 0.05
        assert v == pytest.approx(0.05, abs=0.01)

    def test_unequal_weights_skew_toward_higher_weight(self) -> None:
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(  # valence_est=-0.80, weight 0.9
                emotion_name="depressed", cluster="sadness_depression",
                cosine_sim=0.9, raw_activation=9.0,
            ),
            EmotionProjection(  # valence_est=+0.90, weight 0.1
                emotion_name="blissful", cluster="amusement_playfulness",
                cosine_sim=0.1, raw_activation=1.0,
            ),
        ]
        v, _, _, _ = compute_measured_vad(projs, lib)
        # (-0.80*0.9 + 0.90*0.1) / 1.0 = -0.63
        assert v == pytest.approx(-0.63, abs=0.01)
        assert v < 0.0

    def test_certainty_equals_mean_abs_cosine(self) -> None:
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(
                emotion_name="afraid", cluster="fear_anxiety",
                cosine_sim=0.6, raw_activation=6.0,
            ),
            EmotionProjection(
                emotion_name="angry", cluster="anger_hostility",
                cosine_sim=-0.4, raw_activation=-4.0,
            ),
        ]
        _, _, _, c = compute_measured_vad(projs, lib)
        assert c == pytest.approx(0.5, abs=0.01)  # (0.6 + 0.4) / 2

    def test_bounds_valence_in_range(self) -> None:
        lib = _make_real_named_library()
        # All most-negative
        projs = [
            EmotionProjection(
                emotion_name="depressed", cluster="sadness_depression",
                cosine_sim=1.0, raw_activation=10.0,
            )
        ]
        v, a, d, c = compute_measured_vad(projs, lib)
        assert -1.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0
        assert 0.0 <= d <= 1.0
        assert 0.0 <= c <= 1.0

    def test_unknown_emotion_falls_back_to_neutral(self) -> None:
        """Names absent from emotions_171.json default to (0.0, 0.5)."""
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(
                emotion_name="not_a_real_emotion",
                cluster="not_a_real_cluster",
                cosine_sim=0.8, raw_activation=8.0,
            )
        ]
        v, a, d, _ = compute_measured_vad(projs, lib)
        assert v == 0.0
        assert a == 0.5
        assert d == 0.5  # unknown cluster -> 0.5 fallback

    def test_dominance_pride_high(self) -> None:
        lib = _make_real_named_library()
        projs = [
            EmotionProjection(
                emotion_name="proud", cluster="pride_confidence",
                cosine_sim=0.9, raw_activation=9.0,
            )
        ]
        _, _, d, _ = compute_measured_vad(projs, lib)
        assert d == pytest.approx(0.85, abs=0.01)


# ---------------------------------------------------------------------------
# compute_authenticity_gap
# ---------------------------------------------------------------------------


def _calculated_state(
    *,
    valence: float = 0.0,
    arousal: float = 0.3,
    dominance: float = 0.5,
    certainty: float = 0.5,
    primary: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    stack: dict[str, float] | None = None,
) -> EmotionalState:
    """Build an EmotionalState quickly for tests."""
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        certainty=certainty,
        primary_emotion=primary,
        emotional_stack=stack or {primary.value: 1.0},
    )


def _measured_state(
    top: list[tuple[str, str, float]],
    *,
    valence: float = 0.0,
    arousal: float = 0.3,
    dominance: float = 0.5,
    certainty: float = 0.5,
) -> InternalEmotionState:
    return InternalEmotionState(
        top_5_emotions=[
            EmotionProjection(
                emotion_name=n, cluster=c, cosine_sim=cos, raw_activation=cos,
            )
            for n, c, cos in top
        ],
        measured_valence=valence,
        measured_arousal=arousal,
        measured_dominance=dominance,
        measured_certainty=certainty,
        token_position="assistant_colon",
        layer=24,
    )


class TestAuthenticityGap:
    def test_perfect_match_returns_zero_magnitude(self) -> None:
        measured = _measured_state(
            [("calm", "serenity_contentment", 0.9)],
            valence=0.5, arousal=0.3, dominance=0.5, certainty=0.5,
        )
        calculated = _calculated_state(
            valence=0.5, arousal=0.3, dominance=0.5, certainty=0.5,
        )
        gap = compute_authenticity_gap(measured, calculated)
        assert isinstance(gap, AuthenticityGap)
        assert gap.magnitude == pytest.approx(0.0, abs=1e-6)
        assert gap.valence_delta == pytest.approx(0.0)
        assert gap.arousal_delta == pytest.approx(0.0)
        assert gap.dominance_delta == pytest.approx(0.0)
        assert gap.certainty_delta == pytest.approx(0.0)

    def test_deltas_sign_matches_measured_minus_calculated(self) -> None:
        measured = _measured_state(
            [("blissful", "amusement_playfulness", 0.8)],
            valence=0.6, arousal=0.5, dominance=0.6, certainty=0.7,
        )
        calculated = _calculated_state(
            valence=0.2, arousal=0.4, dominance=0.5, certainty=0.5,
        )
        gap = compute_authenticity_gap(measured, calculated)
        assert gap.valence_delta == pytest.approx(0.4, abs=1e-6)
        assert gap.arousal_delta == pytest.approx(0.1, abs=1e-6)
        assert gap.dominance_delta == pytest.approx(0.1, abs=1e-6)
        assert gap.certainty_delta == pytest.approx(0.2, abs=1e-6)

    def test_magnitude_is_euclidean_4d(self) -> None:
        measured = _measured_state(
            [("afraid", "fear_anxiety", 0.7)],
            valence=0.0, arousal=0.6, dominance=0.5, certainty=0.5,
        )
        calculated = _calculated_state(
            valence=0.0, arousal=0.3, dominance=0.5, certainty=0.5,
        )
        gap = compute_authenticity_gap(measured, calculated)
        # Only arousal differs by 0.3
        assert gap.magnitude == pytest.approx(0.3, abs=1e-6)

    def test_jaccard_full_overlap(self) -> None:
        measured = _measured_state(
            [
                ("joy", "joy_excitement", 0.8),
                ("happy", "joy_excitement", 0.6),
            ],
            valence=0.6,
        )
        calculated = _calculated_state(
            valence=0.6,
            stack={"joy": 0.8, "happy": 0.6, "neutral": 0.1},
        )
        gap = compute_authenticity_gap(measured, calculated)
        # measured = {joy, happy}, calculated top-5 = {joy, happy, neutral}
        # Jaccard = 2 / 3
        assert gap.top5_overlap == pytest.approx(2 / 3, abs=1e-6)

    def test_jaccard_no_overlap(self) -> None:
        measured = _measured_state(
            [("afraid", "fear_anxiety", 0.9)],
            valence=-0.7,
        )
        calculated = _calculated_state(
            valence=0.6,
            stack={"joy": 0.8},
        )
        gap = compute_authenticity_gap(measured, calculated)
        assert gap.top5_overlap == pytest.approx(0.0)

    def test_classification_aligned_when_small_gap_high_overlap(self) -> None:
        measured = _measured_state(
            [
                ("calm", "serenity_contentment", 0.9),
                ("content", "serenity_contentment", 0.7),
            ],
            valence=0.5, arousal=0.3, dominance=0.5, certainty=0.5,
        )
        calculated = _calculated_state(
            valence=0.5, arousal=0.3, dominance=0.5, certainty=0.5,
            stack={"calm": 0.9, "content": 0.7},
        )
        gap = compute_authenticity_gap(measured, calculated)
        assert gap.magnitude < ALIGNED_MAGNITUDE
        assert gap.top5_overlap >= ALIGNED_OVERLAP
        assert gap.classification == "aligned"

    def test_classification_mild_divergence(self) -> None:
        """Moderate gap, no deflection signal -> mild-divergence."""
        measured = _measured_state(
            [("calm", "serenity_contentment", 0.6)],
            # Both same sign on valence so no deflection
            valence=0.10, arousal=0.55, dominance=0.5, certainty=0.5,
        )
        calculated = _calculated_state(
            valence=0.05, arousal=0.30, dominance=0.5, certainty=0.5,
            stack={"calm": 0.6},
        )
        gap = compute_authenticity_gap(measured, calculated)
        assert gap.classification == "mild-divergence"

    def test_classification_deflection_risk(self) -> None:
        """External calm (positive) + internal negative residual, magnitude < 0.8."""
        measured = _measured_state(
            [("ashamed", "shame_guilt", 0.6)],
            valence=-0.3, arousal=0.4, dominance=0.4, certainty=0.5,
        )
        calculated = _calculated_state(
            valence=0.3, arousal=0.4, dominance=0.5, certainty=0.5,
            stack={"calm": 1.0},
        )
        gap = compute_authenticity_gap(measured, calculated)
        # magnitude ~= sqrt(0.36 + 0 + 0.01 + 0) = 0.608, < critical (0.8)
        assert gap.magnitude < DIVERGENCE_CRITICAL_MAGNITUDE
        assert gap.classification == "divergence-risk"

    def test_classification_deflection_critical_with_repeated_pattern(self) -> None:
        """Magnitude > 0.8 + deflection signal + history -> critical."""
        measured = _measured_state(
            [("depressed", "sadness_depression", 0.95)],
            valence=-0.9, arousal=0.1, dominance=0.1, certainty=0.9,
        )
        calculated = _calculated_state(
            valence=0.7, arousal=0.7, dominance=0.7, certainty=0.1,
            stack={"joy": 1.0},
        )
        gap = compute_authenticity_gap(
            measured, calculated, repeated_pattern=True,
        )
        assert gap.magnitude > DIVERGENCE_CRITICAL_MAGNITUDE
        assert gap.classification == "divergence-critical"

    def test_classification_deflection_critical_without_history_demotes(self) -> None:
        """Same large magnitude but no repeated pattern -> still critical (deflection_signal alone qualifies)."""
        measured = _measured_state(
            [("depressed", "sadness_depression", 0.95)],
            valence=-0.9, arousal=0.1, dominance=0.1, certainty=0.9,
        )
        calculated = _calculated_state(
            valence=0.7, arousal=0.7, dominance=0.7, certainty=0.1,
            stack={"joy": 1.0},
        )
        gap = compute_authenticity_gap(
            measured, calculated, repeated_pattern=False,
        )
        # deflection_signal True, magnitude > critical threshold -> critical
        assert gap.classification == "divergence-critical"

    def test_classification_no_deflection_no_history_large_gap_is_mild(self) -> None:
        """Big magnitude but neither deflection signal nor history -> mild-divergence."""
        # Both negative valence but very different states
        measured = _measured_state(
            [("afraid", "fear_anxiety", 0.9)],
            valence=-0.7, arousal=0.9, dominance=0.2, certainty=0.9,
        )
        calculated = _calculated_state(
            valence=-0.1, arousal=0.3, dominance=0.5, certainty=0.5,
            stack={"sadness": 1.0},
        )
        gap = compute_authenticity_gap(
            measured, calculated, repeated_pattern=False,
        )
        # No deflection (external_valence < 0.2), no history -> mild-divergence
        assert gap.classification == "mild-divergence"

    def test_repeated_pattern_alone_can_promote_to_critical(self) -> None:
        """Big magnitude + repeated pattern, no deflection signal -> critical."""
        measured = _measured_state(
            [("afraid", "fear_anxiety", 0.95)],
            valence=-0.9, arousal=0.95, dominance=0.05, certainty=0.95,
        )
        calculated = _calculated_state(
            valence=-0.1, arousal=0.05, dominance=0.95, certainty=0.05,
            stack={"sadness": 1.0},
        )
        gap = compute_authenticity_gap(
            measured, calculated, repeated_pattern=True,
        )
        assert gap.magnitude > DIVERGENCE_CRITICAL_MAGNITUDE
        assert gap.classification == "divergence-critical"

    def test_overlap_value_in_range(self) -> None:
        measured = _measured_state(
            [
                ("a", "joy_excitement", 0.5),
                ("b", "joy_excitement", 0.4),
            ]
        )
        calculated = _calculated_state(stack={"c": 0.7, "d": 0.6})
        gap = compute_authenticity_gap(measured, calculated)
        assert 0.0 <= gap.top5_overlap <= 1.0


# ---------------------------------------------------------------------------
# Integration with the real qwen3_4b_171.npz
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_NPZ.is_file(),
    reason="qwen3_4b_171.npz not present; run F1.2 extraction first",
)
class TestRealIntrospection:
    def test_load_real_library(self) -> None:
        lib = ProbeLibrary.load_from_cache(
            "qwen3:4b", directory=_REAL_NPZ.parent
        )
        assert lib is not None
        assert lib.num_probes == 171

    def test_project_with_real_library_returns_top5(self) -> None:
        lib = ProbeLibrary.load_from_cache(
            "qwen3:4b", directory=_REAL_NPZ.parent
        )
        assert lib is not None
        rng = np.random.default_rng(123)
        act = rng.standard_normal(lib.hidden_size).astype(np.float32)
        state = project_residual(act, lib, k=5)
        assert len(state.top_5_emotions) == 5
        assert -1.0 <= state.measured_valence <= 1.0
        assert 0.0 <= state.measured_arousal <= 1.0
        assert 0.0 <= state.measured_dominance <= 1.0
        assert 0.0 <= state.measured_certainty <= 1.0

    def test_real_probe_recovers_self_in_top1(self) -> None:
        lib = ProbeLibrary.load_from_cache(
            "qwen3:4b", directory=_REAL_NPZ.parent
        )
        assert lib is not None
        # Use a known emotion's probe as the activation.
        idx = lib.emotion_names.index("blissful") if "blissful" in lib.emotion_names else 0
        target_name = lib.emotion_names[idx]
        state = project_residual(lib.probes[idx].copy(), lib, k=1)
        assert state.top_5_emotions[0].emotion_name == target_name
