"""Tests for RESIDUUM F1.2: neutral_transcripts + extraction helpers + story generation.

Covers everything that can be tested without GPU/model: Pydantic models,
stem-matching, PCA math, story generation with a mock LLM. The full
extract_171_probes() run is gated behind RESIDUUM_RUN_EXTRACTION=1
because it requires torch + transformers + a loadable model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pathos.engine.steering_extract import (
    _build_story_prompt,
    _clean_generated_story,
    _compute_neutral_pcs,
    _load_emotions_171,
    _load_neutral_transcripts,
    _load_story_topics,
    _project_out,
    _slugify_emotion,
    _story_contains_emotion_word,
    generate_stories_for_emotions,
)
from pathos.models.residuum import (
    AuthenticityGap,
    EmotionProjection,
    ExtractionConfig,
    InternalEmotionState,
    ProbeLibraryInfo,
    ProbeMetadata,
)


_STEERING_DATA = Path(__file__).parent.parent / "src" / "pathos" / "steering_data"
_NEUTRAL_PATH = _STEERING_DATA / "neutral_transcripts.json"


# ---------------------------------------------------------------------------
# Neutral transcripts file integrity
# ---------------------------------------------------------------------------


class TestNeutralTranscriptsFile:
    def test_file_exists(self) -> None:
        assert _NEUTRAL_PATH.exists(), f"Missing {_NEUTRAL_PATH}"

    def test_has_meta(self) -> None:
        data = json.loads(_NEUTRAL_PATH.read_text(encoding="utf-8"))
        assert "_meta" in data
        assert data["_meta"]["total"] == 100

    def test_exactly_100_transcripts(self) -> None:
        data = json.loads(_NEUTRAL_PATH.read_text(encoding="utf-8"))
        assert len(data["transcripts"]) == 100

    def test_all_non_empty_strings(self) -> None:
        data = json.loads(_NEUTRAL_PATH.read_text(encoding="utf-8"))
        for i, t in enumerate(data["transcripts"]):
            assert isinstance(t, str) and len(t) > 0, f"transcript {i} empty"

    def test_reasonable_length(self) -> None:
        """Each transcript between ~100 and 1000 chars (style/length diversity)."""
        data = json.loads(_NEUTRAL_PATH.read_text(encoding="utf-8"))
        for i, t in enumerate(data["transcripts"]):
            assert 100 <= len(t) <= 1200, f"transcript {i} length={len(t)} out of range"

    def test_no_emotion_word_leaks(self) -> None:
        """Anti-confound rule: no emotion word stem appears in any neutral transcript."""
        data = json.loads(_NEUTRAL_PATH.read_text(encoding="utf-8"))
        emotions = [e["name"] for e in _load_emotions_171()]
        leaks = []
        for i, t in enumerate(data["transcripts"]):
            for e in emotions:
                if _story_contains_emotion_word(t, e):
                    leaks.append((i, e))
                    break
        assert not leaks, f"Emotion leaks: {leaks[:10]}"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestResiduumModels:
    def test_probe_metadata_requires_fields(self) -> None:
        m = ProbeMetadata(
            emotion_name="happy", cluster="joy_excitement",
            layer=19, dimension=2560,
            norm_before_projection=12.3, norm_after_projection=5.1,
            neutral_pcs_removed=7, source_stories_count=15,
        )
        assert m.emotion_name == "happy"
        assert m.dimension == 2560

    def test_probe_library_info(self) -> None:
        info = ProbeLibraryInfo(
            model_id="qwen3:4b", layer=19, hidden_size=2560,
            num_probes=171, num_neutral_pcs=8,
            extracted_at="2026-04-22T12:00:00Z", source_stories_count=2565,
        )
        assert info.status == "ready"
        assert info.num_probes == 171

    def test_emotion_projection_cosine_range(self) -> None:
        p = EmotionProjection(
            emotion_name="happy", cluster="joy_excitement",
            cosine_sim=0.87, raw_activation=12.3,
        )
        assert -1.0 <= p.cosine_sim <= 1.0

    def test_emotion_projection_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            EmotionProjection(
                emotion_name="x", cluster="c",
                cosine_sim=1.5, raw_activation=0.0,
            )

    def test_internal_emotion_state_defaults(self) -> None:
        s = InternalEmotionState()
        assert s.top_5_emotions == []
        assert s.measured_valence == 0.0
        assert s.measured_arousal == 0.5
        assert s.token_position == "assistant_colon"

    def test_internal_emotion_state_top_5_cap(self) -> None:
        projections = [
            EmotionProjection(emotion_name=f"e{i}", cluster="c", cosine_sim=0.1, raw_activation=0.0)
            for i in range(5)
        ]
        s = InternalEmotionState(top_5_emotions=projections)
        assert len(s.top_5_emotions) == 5

        too_many = [
            EmotionProjection(emotion_name=f"e{i}", cluster="c", cosine_sim=0.1, raw_activation=0.0)
            for i in range(6)
        ]
        with pytest.raises(Exception):
            InternalEmotionState(top_5_emotions=too_many)

    def test_authenticity_gap_defaults(self) -> None:
        g = AuthenticityGap(
            top5_overlap=0.8,
            valence_delta=0.1, arousal_delta=0.05,
            dominance_delta=0.03, certainty_delta=0.02,
            magnitude=0.15,
        )
        assert g.classification == "aligned"

    def test_extraction_config_roundtrip(self) -> None:
        cfg = ExtractionConfig(model_id="qwen3:4b", layer=19, seed=42)
        dumped = cfg.model_dump_json()
        reloaded = ExtractionConfig.model_validate_json(dumped)
        assert reloaded.model_id == "qwen3:4b"
        assert reloaded.layer == 19


# ---------------------------------------------------------------------------
# Anti-leak stem matching
# ---------------------------------------------------------------------------


class TestAntiLeakMatching:
    def test_base_form_matches(self) -> None:
        assert _story_contains_emotion_word("She was happy.", "happy")

    def test_y_to_i_morphology(self) -> None:
        # happily, happiness, happier all share stem "happi"
        assert _story_contains_emotion_word("He laughed happily.", "happy")
        assert _story_contains_emotion_word("Happiness filled him.", "happy")
        assert _story_contains_emotion_word("She grew happier.", "happy")

    def test_ed_to_root_morphology(self) -> None:
        # amuse, amusing, amusement for "amused"
        assert _story_contains_emotion_word("He was amused.", "amused")
        assert _story_contains_emotion_word("An amusing scene.", "amused")
        assert _story_contains_emotion_word("Great amusement.", "amused")

    def test_ing_to_root_morphology(self) -> None:
        assert _story_contains_emotion_word("She was loving.", "loving")
        assert _story_contains_emotion_word("He loved her.", "loving")

    def test_word_boundary_no_false_positive(self) -> None:
        # "hangry" should NOT match "angry"
        assert not _story_contains_emotion_word("They were hangry.", "angry")

    def test_neutral_passage_no_match(self) -> None:
        assert not _story_contains_emotion_word("A binary tree is a data structure.", "happy")
        assert not _story_contains_emotion_word("The API returned a JSON response.", "amused")

    def test_empty_emotion_no_match(self) -> None:
        assert not _story_contains_emotion_word("anything", "")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_slugify(self) -> None:
        assert _slugify_emotion("happy") == "happy"
        assert _slugify_emotion("happy go lucky") == "happy_go_lucky"
        assert _slugify_emotion("ANGRY") == "angry"
        assert _slugify_emotion("  afraid  ") == "afraid"

    def test_build_story_prompt_contains_anti_leak_rule(self) -> None:
        p = _build_story_prompt("amused", "a rainy afternoon")
        assert "amused" in p.lower()
        assert "never appear" in p.lower() or "must not" in p.lower() or "NEVER" in p

    def test_clean_generated_story_strips_think_tags(self) -> None:
        raw = "<think>let me plan this</think>\nShe walked into the room."
        assert _clean_generated_story(raw, "happy") == "She walked into the room."

    def test_clean_generated_story_strips_code_fences(self) -> None:
        raw = "```\nShe walked into the room.\n```"
        cleaned = _clean_generated_story(raw, "happy")
        assert "```" not in cleaned
        assert "She walked" in cleaned

    def test_clean_generated_story_passes_clean_text(self) -> None:
        raw = "A simple paragraph with no artifacts."
        assert _clean_generated_story(raw, "happy") == raw


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


class TestDataLoaders:
    def test_load_emotions_171(self) -> None:
        emos = _load_emotions_171()
        assert len(emos) == 171
        assert all("name" in e and "cluster" in e for e in emos)

    def test_load_story_topics(self) -> None:
        topics = _load_story_topics()
        assert len(topics) >= 50
        assert all(isinstance(t, str) and len(t) > 0 for t in topics)

    def test_load_neutral_transcripts(self) -> None:
        neutrals = _load_neutral_transcripts()
        assert len(neutrals) == 100
        assert all(isinstance(t, str) and len(t) > 0 for t in neutrals)


# ---------------------------------------------------------------------------
# PCA / projection math
# ---------------------------------------------------------------------------


class TestPCAMath:
    def test_pcs_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 64)).astype(np.float32)
        pcs = _compute_neutral_pcs(X, variance_threshold=0.5)
        assert pcs.ndim == 2
        assert pcs.shape[1] == 64
        assert 1 <= pcs.shape[0] <= min(X.shape)

    def test_pcs_are_unit_norm(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 128)).astype(np.float32)
        pcs = _compute_neutral_pcs(X, variance_threshold=0.7)
        norms = np.linalg.norm(pcs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_pcs_orthogonal(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 32)).astype(np.float32)
        pcs = _compute_neutral_pcs(X, variance_threshold=0.8)
        if pcs.shape[0] >= 2:
            gram = pcs @ pcs.T
            off_diag = gram - np.eye(pcs.shape[0], dtype=np.float32)
            assert np.max(np.abs(off_diag)) < 1e-4

    def test_variance_threshold_monotonic(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 128)).astype(np.float32)
        pcs_low = _compute_neutral_pcs(X, variance_threshold=0.3)
        pcs_high = _compute_neutral_pcs(X, variance_threshold=0.9)
        assert pcs_low.shape[0] <= pcs_high.shape[0]

    def test_project_out_removes_component(self) -> None:
        # Build a vector aligned with a basis direction; projecting should zero it out.
        basis = np.eye(3, dtype=np.float32)[:2]  # first two std basis vectors
        v = np.array([2.0, 3.0, 1.5], dtype=np.float32)
        residual = _project_out(v, basis)
        # Only component along axis 2 should remain.
        np.testing.assert_allclose(residual, [0.0, 0.0, 1.5], atol=1e-6)

    def test_project_out_empty_basis_returns_copy(self) -> None:
        basis = np.zeros((0, 5), dtype=np.float32)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        out = _project_out(v, basis)
        np.testing.assert_array_equal(out, v)

    def test_project_out_preserves_orthogonal_components(self) -> None:
        # Projecting out a vector that is orthogonal to v should leave v unchanged.
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        basis = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        out = _project_out(v, basis)
        np.testing.assert_allclose(out, v, atol=1e-6)


# ---------------------------------------------------------------------------
# Story generation with mock LLM
# ---------------------------------------------------------------------------


class TestStoryGeneration:
    def test_generates_expected_files(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def mock_llm(prompt: str) -> str:
            calls.append(prompt)
            return "A passage with neutral description of everyday scene."

        counts = generate_stories_for_emotions(
            n_per_emotion=1,
            llm_call=mock_llm,
            output_dir=tmp_path,
            skip_existing=False,
            seed=42,
            min_bytes=10,
        )
        # 171 emotions * 1 story = 171 generated
        assert counts["generated"] == 171
        # One file per emotion in the slug subdir
        files = list(tmp_path.rglob("story_*.txt"))
        assert len(files) == 171
        # LLM was called at least 171 times (may be more if leak retries)
        assert len(calls) >= 171

    def test_skip_existing(self, tmp_path: Path) -> None:
        # Pre-create all story files
        emotions = _load_emotions_171()
        for e in emotions:
            d = tmp_path / _slugify_emotion(e["name"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "story_00.txt").write_text("existing content placeholder", encoding="utf-8")

        calls: list[str] = []

        def mock_llm(prompt: str) -> str:
            calls.append(prompt)
            return "Fresh passage."

        counts = generate_stories_for_emotions(
            n_per_emotion=1, llm_call=mock_llm,
            output_dir=tmp_path, skip_existing=True,
            min_bytes=10,
        )
        assert counts["generated"] == 0
        assert counts["skipped"] == 171
        assert calls == []

    def test_retry_on_leak(self, tmp_path: Path) -> None:
        # First call leaks the emotion word, second call is clean.
        # Pick a specific emotion to drive the test.
        attempts: dict[str, int] = {}

        def mock_llm(prompt: str) -> str:
            # Find emotion in prompt
            import re
            m = re.search(r"feeling ([^,]+),", prompt)
            emo = m.group(1) if m else "unknown"
            attempts.setdefault(emo, 0)
            attempts[emo] += 1
            if attempts[emo] == 1:
                # Leak on first attempt by including the word
                return f"She was {emo} today, very much so."
            return "A clean passage with no emotion word."

        counts = generate_stories_for_emotions(
            n_per_emotion=1, llm_call=mock_llm,
            output_dir=tmp_path, skip_existing=False,
            max_retries_per_story=3,
            min_bytes=10,
        )
        # Should have generated all 171 after retrying
        assert counts["generated"] == 171
        # Each emotion triggered at least 2 calls (first leaked)
        assert all(v >= 2 for v in attempts.values())

    def test_fails_gracefully_on_persistent_leak(self, tmp_path: Path) -> None:
        def leaky_llm(prompt: str) -> str:
            import re
            m = re.search(r"feeling ([^,]+),", prompt)
            emo = m.group(1) if m else "happy"
            return f"{emo} {emo} {emo}"

        counts = generate_stories_for_emotions(
            n_per_emotion=1, llm_call=leaky_llm,
            output_dir=tmp_path, skip_existing=False,
            max_retries_per_story=2,
            min_bytes=10,
        )
        # All should fail to pass the anti-leak rule
        assert counts["failed"] == 171
        assert counts["generated"] == 0


# ---------------------------------------------------------------------------
# Full extraction (gated by env var: needs torch + GPU + model)
# ---------------------------------------------------------------------------


_RUN_FULL = os.environ.get("RESIDUUM_RUN_EXTRACTION") == "1"


@pytest.mark.skipif(not _RUN_FULL, reason="set RESIDUUM_RUN_EXTRACTION=1 to run")
class TestFullExtraction:
    """End-to-end extraction test. Requires torch + model + pre-generated stories."""

    def test_extract_produces_expected_shapes(self) -> None:
        from pathos.engine.steering_extract import extract_171_probes

        result = extract_171_probes(
            model_id=os.environ.get("RESIDUUM_TEST_MODEL", "qwen3:4b"),
            stories_per_emotion=2,  # minimal for test speed
        )
        assert result["num_probes"] == 171
        assert result["num_neutral_pcs"] >= 1
        assert result["norm_reduction_ratio"] < 1.0  # projection actually removed variance

    def test_opposite_emotions_have_negative_cosine(self) -> None:
        from pathos.engine.steering import load_cached_vectors  # type: ignore

        # Placeholder: once probe loader (F1.3) lands, we verify:
        #   cosine(happy, sad) < -0.3
        #   cosine(happy, excited) > +0.3
        pytest.skip("Probe library loader lands in F1.3")
