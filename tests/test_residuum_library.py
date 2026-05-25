"""Tests for RESIDUUM F1.3 — ProbeLibrary runtime loader.

Covers:
  - Construction shape validation.
  - NPZ loading + graceful degradation on missing / malformed caches.
  - Accessor behavior (get_probe / list_emotions / get_all_probes).
  - Projection math (cosine_similarity / all_cosines / top_k).
  - ProbeLibraryInfo serializable view.

Two kinds of tests:
  TestSyntheticLibrary   — uses a small NPZ written at test time. Runs always.
  TestRealLibrary        — uses src/pathos/steering_data/cached_vectors/
                           qwen3_4b_171.npz if present. Skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pathos.engine.steering import ProbeLibrary
from pathos.models.residuum import EmotionProjection, ProbeLibraryInfo


_STEERING_DATA = Path(__file__).parent.parent / "src" / "pathos" / "steering_data"
_REAL_NPZ = _STEERING_DATA / "cached_vectors" / "qwen3_4b_171.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _write_synthetic_npz(
    path: Path,
    num_probes: int = 6,
    hidden: int = 8,
    num_pcs: int = 2,
) -> None:
    """Write a small valid NPZ mimicking extract_171_probes output."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((num_probes, hidden)).astype(np.float32)
    probes = np.stack([_unit(r) for r in raw]).astype(np.float32)
    emotion_names = np.array(
        ["happy", "sad", "loving", "hateful", "calm", "panicked"][:num_probes]
    )
    clusters = np.array(
        [
            "joy_excitement",
            "sadness_depression",
            "love_warmth",
            "anger_hostility",
            "serenity_contentment",
            "fear_anxiety",
        ][:num_probes]
    )
    neutral_pcs = rng.standard_normal((num_pcs, hidden)).astype(np.float32)
    neutral_pcs = np.stack([_unit(p) for p in neutral_pcs])
    norms_before = np.linspace(20.0, 30.0, num_probes).astype(np.float32)
    norms_after = norms_before * 0.9
    story_counts = np.full(num_probes, 15, dtype=np.int32)
    metadata = json.dumps(
        {
            "model_id": "synthetic",
            "layer": 12,
            "stories_per_emotion": 15,
            "neutral_pc_variance_threshold": 0.5,
            "token_start_index": 50,
            "device": "cpu",
            "dtype": "float32",
            "seed": 42,
            "extra": {
                "source": "synthetic",
                "num_layers": 18,
                "hidden_size": hidden,
                "num_neutral_pcs": num_pcs,
                "total_stories_used": num_probes * 15,
                "extracted_at": "2026-04-23T00:00:00+00:00",
            },
        }
    )
    np.savez(
        path,
        probes=probes,
        emotion_names=emotion_names,
        clusters=clusters,
        neutral_pcs=neutral_pcs,
        norms_before=norms_before,
        norms_after=norms_after,
        story_counts=story_counts,
        metadata=np.array([metadata]),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestProbeLibraryConstruction:
    def test_valid_construction(self) -> None:
        probes = np.eye(3, 4, dtype=np.float32)
        lib = ProbeLibrary(
            model_id="x",
            probes=probes,
            emotion_names=["a", "b", "c"],
            clusters=["c1", "c2", "c1"],
            neutral_pcs=np.zeros((0, 4), dtype=np.float32),
            norms_before=np.ones(3, dtype=np.float32),
            norms_after=np.ones(3, dtype=np.float32),
            story_counts=np.full(3, 15, dtype=np.int32),
            metadata={"layer": 9},
        )
        assert lib.num_probes == 3
        assert lib.hidden_size == 4
        assert lib.num_neutral_pcs == 0
        assert lib.layer == 9

    def test_probe_rows_must_match_names(self) -> None:
        with pytest.raises(ValueError):
            ProbeLibrary(
                model_id="x",
                probes=np.eye(3, 4, dtype=np.float32),
                emotion_names=["a", "b"],
                clusters=["c", "c", "c"],
                neutral_pcs=np.zeros((0, 4), dtype=np.float32),
                norms_before=np.ones(3, dtype=np.float32),
                norms_after=np.ones(3, dtype=np.float32),
                story_counts=np.full(3, 15, dtype=np.int32),
                metadata={},
            )

    def test_probe_rows_must_match_clusters(self) -> None:
        with pytest.raises(ValueError):
            ProbeLibrary(
                model_id="x",
                probes=np.eye(3, 4, dtype=np.float32),
                emotion_names=["a", "b", "c"],
                clusters=["c"],
                neutral_pcs=np.zeros((0, 4), dtype=np.float32),
                norms_before=np.ones(3, dtype=np.float32),
                norms_after=np.ones(3, dtype=np.float32),
                story_counts=np.full(3, 15, dtype=np.int32),
                metadata={},
            )

    def test_probes_must_be_2d(self) -> None:
        with pytest.raises(ValueError):
            ProbeLibrary(
                model_id="x",
                probes=np.zeros(10, dtype=np.float32),
                emotion_names=[],
                clusters=[],
                neutral_pcs=np.zeros((0, 0), dtype=np.float32),
                norms_before=np.zeros(0, dtype=np.float32),
                norms_after=np.zeros(0, dtype=np.float32),
                story_counts=np.zeros(0, dtype=np.int32),
                metadata={},
            )


# ---------------------------------------------------------------------------
# NPZ I/O
# ---------------------------------------------------------------------------


class TestLoadFromCache:
    def test_missing_cache_returns_none(self, tmp_path: Path) -> None:
        lib = ProbeLibrary.load_from_cache("no_such_model", directory=tmp_path)
        assert lib is None

    def test_valid_roundtrip(self, tmp_path: Path) -> None:
        _write_synthetic_npz(tmp_path / "synthetic_171.npz")
        lib = ProbeLibrary.load_from_cache("synthetic", directory=tmp_path)
        assert lib is not None
        assert lib.num_probes == 6
        assert lib.hidden_size == 8
        assert lib.num_neutral_pcs == 2
        assert lib.layer == 12
        assert "happy" in lib.emotion_names
        assert lib.metadata.get("seed") == 42

    def test_model_id_with_colon_slashes(self, tmp_path: Path) -> None:
        # model_id "foo:bar/baz" must become "foo_bar_baz_171.npz"
        _write_synthetic_npz(tmp_path / "foo_bar_baz_171.npz")
        lib = ProbeLibrary.load_from_cache("foo:bar/baz", directory=tmp_path)
        assert lib is not None
        assert lib.model_id == "foo:bar/baz"

    def test_corrupted_npz_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_171.npz"
        path.write_bytes(b"not a valid npz file")
        lib = ProbeLibrary.load_from_cache("bad", directory=tmp_path)
        assert lib is None

    def test_missing_required_key_returns_none(self, tmp_path: Path) -> None:
        # No 'probes' key at all
        np.savez(
            tmp_path / "partial_171.npz",
            emotion_names=np.array(["a"]),
            clusters=np.array(["c"]),
        )
        lib = ProbeLibrary.load_from_cache("partial", directory=tmp_path)
        assert lib is None

    def test_minimal_npz_without_optional_arrays(self, tmp_path: Path) -> None:
        # Only mandatory keys; neutral_pcs/norms/story_counts default to zeros
        probes = np.eye(2, 3, dtype=np.float32)
        np.savez(
            tmp_path / "minimal_171.npz",
            probes=probes,
            emotion_names=np.array(["a", "b"]),
            clusters=np.array(["c1", "c2"]),
        )
        lib = ProbeLibrary.load_from_cache("minimal", directory=tmp_path)
        assert lib is not None
        assert lib.num_probes == 2
        assert lib.num_neutral_pcs == 0
        assert lib.story_counts.sum() == 0


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_lib(tmp_path: Path) -> ProbeLibrary:
    _write_synthetic_npz(tmp_path / "synthetic_171.npz")
    lib = ProbeLibrary.load_from_cache("synthetic", directory=tmp_path)
    assert lib is not None
    return lib


class TestAccessors:
    def test_get_probe_known(self, synthetic_lib: ProbeLibrary) -> None:
        probe = synthetic_lib.get_probe("happy")
        assert probe is not None
        assert probe.shape == (synthetic_lib.hidden_size,)

    def test_get_probe_unknown_returns_none(self, synthetic_lib: ProbeLibrary) -> None:
        assert synthetic_lib.get_probe("nonexistent_emotion") is None

    def test_get_probe_returns_copy(self, synthetic_lib: ProbeLibrary) -> None:
        probe = synthetic_lib.get_probe("happy")
        assert probe is not None
        probe[:] = 0.0
        # Original unchanged
        assert np.linalg.norm(synthetic_lib.get_probe("happy")) > 0

    def test_get_all_probes_covers_all(self, synthetic_lib: ProbeLibrary) -> None:
        all_probes = synthetic_lib.get_all_probes()
        assert set(all_probes.keys()) == set(synthetic_lib.emotion_names)

    def test_list_emotions_order_matches_rows(self, synthetic_lib: ProbeLibrary) -> None:
        names = synthetic_lib.list_emotions()
        assert names == synthetic_lib.emotion_names

    def test_list_emotions_returns_copy(self, synthetic_lib: ProbeLibrary) -> None:
        names = synthetic_lib.list_emotions()
        names.append("tampered")
        assert "tampered" not in synthetic_lib.emotion_names


# ---------------------------------------------------------------------------
# Projection math
# ---------------------------------------------------------------------------


class TestProjectionMath:
    def test_cosine_similarity_with_known_emotion(self, synthetic_lib: ProbeLibrary) -> None:
        probe = synthetic_lib.get_probe("happy")
        # Cosine of a vector with itself = 1.0
        c = synthetic_lib.cosine_similarity(probe, "happy")
        assert abs(c - 1.0) < 1e-5

    def test_cosine_similarity_unknown_raises(self, synthetic_lib: ProbeLibrary) -> None:
        with pytest.raises(KeyError):
            synthetic_lib.cosine_similarity(np.zeros(synthetic_lib.hidden_size), "ghost")

    def test_cosine_similarity_zero_vector(self, synthetic_lib: ProbeLibrary) -> None:
        c = synthetic_lib.cosine_similarity(
            np.zeros(synthetic_lib.hidden_size, dtype=np.float32), "happy"
        )
        assert c == 0.0

    def test_all_cosines_shape(self, synthetic_lib: ProbeLibrary) -> None:
        act = np.random.default_rng(1).standard_normal(synthetic_lib.hidden_size)
        cos = synthetic_lib.all_cosines(act)
        assert cos.shape == (synthetic_lib.num_probes,)

    def test_all_cosines_bounds(self, synthetic_lib: ProbeLibrary) -> None:
        act = np.random.default_rng(1).standard_normal(synthetic_lib.hidden_size)
        cos = synthetic_lib.all_cosines(act)
        # probes are unit-norm; cosines should be in [-1, 1] up to float error
        assert cos.min() >= -1.0 - 1e-5
        assert cos.max() <= 1.0 + 1e-5

    def test_all_cosines_self(self, synthetic_lib: ProbeLibrary) -> None:
        probe = synthetic_lib.get_probe("happy")
        cos = synthetic_lib.all_cosines(probe)
        i = synthetic_lib.emotion_names.index("happy")
        assert abs(cos[i] - 1.0) < 1e-5

    def test_all_cosines_wrong_shape_raises(self, synthetic_lib: ProbeLibrary) -> None:
        with pytest.raises(ValueError):
            synthetic_lib.all_cosines(np.zeros(synthetic_lib.hidden_size + 1))

    def test_all_cosines_zero_vector(self, synthetic_lib: ProbeLibrary) -> None:
        cos = synthetic_lib.all_cosines(np.zeros(synthetic_lib.hidden_size))
        assert np.all(cos == 0.0)


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_returns_emotion_projections(self, synthetic_lib: ProbeLibrary) -> None:
        act = np.random.default_rng(2).standard_normal(synthetic_lib.hidden_size)
        result = synthetic_lib.top_k(act, k=3)
        assert len(result) == 3
        for p in result:
            assert isinstance(p, EmotionProjection)
            assert -1.0 <= p.cosine_sim <= 1.0

    def test_top_k_sorted_by_abs_cosine(self, synthetic_lib: ProbeLibrary) -> None:
        act = np.random.default_rng(3).standard_normal(synthetic_lib.hidden_size)
        result = synthetic_lib.top_k(act, k=synthetic_lib.num_probes)
        abs_values = [abs(p.cosine_sim) for p in result]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_top_k_own_probe_returns_self_first(self, synthetic_lib: ProbeLibrary) -> None:
        probe = synthetic_lib.get_probe("happy")
        result = synthetic_lib.top_k(probe, k=1)
        assert result[0].emotion_name == "happy"
        assert abs(result[0].cosine_sim - 1.0) < 1e-5

    def test_top_k_k_clamped_to_num_probes(self, synthetic_lib: ProbeLibrary) -> None:
        act = np.random.default_rng(4).standard_normal(synthetic_lib.hidden_size)
        result = synthetic_lib.top_k(act, k=9999)
        assert len(result) == synthetic_lib.num_probes

    def test_top_k_zero_returns_empty(self, synthetic_lib: ProbeLibrary) -> None:
        act = np.random.default_rng(5).standard_normal(synthetic_lib.hidden_size)
        assert synthetic_lib.top_k(act, k=0) == []

    def test_top_k_cluster_populated(self, synthetic_lib: ProbeLibrary) -> None:
        probe = synthetic_lib.get_probe("sad")
        result = synthetic_lib.top_k(probe, k=1)
        assert result[0].cluster != ""


# ---------------------------------------------------------------------------
# Info view
# ---------------------------------------------------------------------------


class TestInfoView:
    def test_info_returns_probe_library_info(self, synthetic_lib: ProbeLibrary) -> None:
        info = synthetic_lib.info()
        assert isinstance(info, ProbeLibraryInfo)
        assert info.num_probes == synthetic_lib.num_probes
        assert info.hidden_size == synthetic_lib.hidden_size
        assert info.num_neutral_pcs == synthetic_lib.num_neutral_pcs
        assert info.layer == synthetic_lib.layer
        assert info.status == "ready"

    def test_info_extracted_at_populated(self, synthetic_lib: ProbeLibrary) -> None:
        info = synthetic_lib.info()
        # Synthetic NPZ sets extracted_at to a valid ISO-8601 string
        assert "2026" in info.extracted_at

    def test_info_roundtrips_via_json(self, synthetic_lib: ProbeLibrary) -> None:
        info = synthetic_lib.info()
        s = info.model_dump_json()
        parsed = json.loads(s)
        assert parsed["num_probes"] == synthetic_lib.num_probes


# ---------------------------------------------------------------------------
# F4.0 — residual_norm_typical property + fallback
# ---------------------------------------------------------------------------


class TestResidualNormTypical:
    """F4.0: ProbeLibrary.residual_norm_typical reads metadata.extra when the
    extractor wrote it, and falls back to mean(norms_before) for older NPZs
    that pre-date the F4.0 schema change. F4 granular steering uses this
    value as the reference for MAX_STEERING_FRACTION."""

    def _lib_with_norms(
        self, norms: np.ndarray, extra: dict | None = None,
    ) -> ProbeLibrary:
        n = norms.shape[0]
        meta = {"layer": 12, "extra": extra} if extra else {"layer": 12}
        return ProbeLibrary(
            model_id="x",
            probes=np.eye(n, 4, dtype=np.float32),
            emotion_names=[f"e{i}" for i in range(n)],
            clusters=["c"] * n,
            neutral_pcs=np.zeros((0, 4), dtype=np.float32),
            norms_before=norms.astype(np.float32),
            norms_after=norms.astype(np.float32) * 0.9,
            story_counts=np.full(n, 15, dtype=np.int32),
            metadata=meta,
        )

    def test_uses_metadata_value_when_present(self) -> None:
        lib = self._lib_with_norms(
            np.array([10.0, 20.0, 30.0]),
            extra={"residual_norm_typical": 42.5},
        )
        assert lib.residual_norm_typical == pytest.approx(42.5)

    def test_falls_back_to_mean_norms_before_when_metadata_missing(self) -> None:
        # Older NPZ: no extra block at all -> fallback to mean(norms_before).
        lib = self._lib_with_norms(np.array([10.0, 20.0, 30.0]))
        assert lib.residual_norm_typical == pytest.approx(20.0)

    def test_falls_back_when_extra_lacks_key(self) -> None:
        lib = self._lib_with_norms(
            np.array([4.0, 8.0]),
            extra={"unrelated": "value"},
        )
        assert lib.residual_norm_typical == pytest.approx(6.0)

    def test_falls_back_when_extra_value_is_not_numeric(self) -> None:
        lib = self._lib_with_norms(
            np.array([4.0, 8.0]),
            extra={"residual_norm_typical": "garbage"},
        )
        assert lib.residual_norm_typical == pytest.approx(6.0)

    def test_zero_when_no_norms_and_no_metadata(self) -> None:
        lib = ProbeLibrary(
            model_id="x",
            probes=np.eye(2, 4, dtype=np.float32),
            emotion_names=["a", "b"],
            clusters=["c", "c"],
            neutral_pcs=np.zeros((0, 4), dtype=np.float32),
            norms_before=np.zeros(0, dtype=np.float32),
            norms_after=np.zeros(0, dtype=np.float32),
            story_counts=np.zeros(2, dtype=np.int32),
            metadata={"layer": 12},
        )
        assert lib.residual_norm_typical == 0.0


# ---------------------------------------------------------------------------
# Real NPZ smoke test (skipped if extraction hasn't run)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _REAL_NPZ.is_file(), reason="qwen3_4b_171.npz not present")
class TestRealLibrary:
    @pytest.fixture(scope="class")
    def real_lib(self) -> ProbeLibrary:
        lib = ProbeLibrary.load_from_cache("qwen3:4b")
        assert lib is not None
        return lib

    def test_has_171_probes(self, real_lib: ProbeLibrary) -> None:
        assert real_lib.num_probes == 171

    def test_hidden_size_matches_qwen3_4b(self, real_lib: ProbeLibrary) -> None:
        # Qwen3-4B uses hidden_size=2560
        assert real_lib.hidden_size == 2560

    def test_layer_is_mid_to_late(self, real_lib: ProbeLibrary) -> None:
        # 2/3 depth of 36 = 24
        assert 18 <= real_lib.layer <= 30

    def test_neutral_pcs_populated(self, real_lib: ProbeLibrary) -> None:
        assert real_lib.num_neutral_pcs > 0

    def test_top_k_happy_is_positive_cluster(self, real_lib: ProbeLibrary) -> None:
        # The top-5 for 'happy' probe itself should be dominated by positive
        # emotions (amusement / joy / love / pride clusters).
        probe = real_lib.get_probe("happy")
        assert probe is not None
        positive_clusters = {
            "joy_excitement",
            "serenity_contentment",
            "love_warmth",
            "pride_confidence",
            "amusement_playfulness",
        }
        result = real_lib.top_k(probe, k=5)
        positive_hits = sum(1 for p in result if p.cluster in positive_clusters)
        assert positive_hits >= 3, f"only {positive_hits}/5 positive-cluster hits"

    def test_within_cluster_cosines_positive(self, real_lib: ProbeLibrary) -> None:
        # Same-cluster emotions should share direction (paper metric).
        # Absolute threshold is modest (0.2) because the neutral-PC projection
        # removes shared-emotion magnitude; what matters is that within-cluster
        # is positive and clearly above the paired opposite (checked below).
        pairs = [("sad", "gloomy"), ("angry", "enraged"), ("afraid", "terrified")]
        for a, b in pairs:
            c = real_lib.cosine_similarity(real_lib.get_probe(a), b)
            assert c > 0.2, f"cos({a},{b})={c:.3f} expected > 0.2"

    def test_opposite_pairs_anti_correlated(self, real_lib: ProbeLibrary) -> None:
        # Semantic opposites should be anti-correlated after neutral-PC removal.
        # Before dataset cleanup (truncated stories), these were strongly positive
        # (+0.7/+0.8) due to a shared "emotional content" axis.
        pairs = [("happy", "sad"), ("calm", "panicked"), ("proud", "ashamed")]
        for a, b in pairs:
            c = real_lib.cosine_similarity(real_lib.get_probe(a), b)
            assert c < 0.0, f"cos({a},{b})={c:.3f} expected < 0 (opposites)"

    def test_info_ready(self, real_lib: ProbeLibrary) -> None:
        info = real_lib.info()
        assert info.status == "ready"
        assert info.num_probes == 171
