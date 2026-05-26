"""Tests for RESIDUUM F1.1 dataset files: emotions_171.json and story_topics.json.

Validates structure, coverage, and invariants that downstream extraction (F1.2)
depends on. No runtime logic tested here — only data integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_STEERING_DATA = Path(__file__).parent.parent / "src" / "pathos" / "steering_data"
_EMOTIONS_PATH = _STEERING_DATA / "emotions_171.json"
_TOPICS_PATH = _STEERING_DATA / "story_topics.json"

PAPER_EMOTION_COUNT = 171
EXPECTED_TOPIC_COUNT = 50
EXPECTED_CLUSTERS = {
    "joy_excitement",
    "serenity_contentment",
    "love_warmth",
    "pride_confidence",
    "amusement_playfulness",
    "sadness_depression",
    "fear_anxiety",
    "anger_hostility",
    "surprise_confusion",
    "shame_guilt",
}


@pytest.fixture(scope="module")
def emotions_data() -> dict:
    with _EMOTIONS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def topics_data() -> dict:
    with _TOPICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


class TestEmotions171File:
    def test_file_exists(self) -> None:
        assert _EMOTIONS_PATH.exists(), f"Missing {_EMOTIONS_PATH}"

    def test_has_meta_block(self, emotions_data: dict) -> None:
        assert "_meta" in emotions_data
        meta = emotions_data["_meta"]
        assert meta["total"] == PAPER_EMOTION_COUNT
        assert meta["valence_range"] == [-1.0, 1.0]
        assert meta["arousal_range"] == [0.0, 1.0]

    def test_exactly_171_emotions(self, emotions_data: dict) -> None:
        assert len(emotions_data["emotions"]) == PAPER_EMOTION_COUNT

    def test_all_entries_have_required_fields(self, emotions_data: dict) -> None:
        required = {"name", "valence_est", "arousal_est", "cluster"}
        for entry in emotions_data["emotions"]:
            missing = required - entry.keys()
            assert not missing, f"Entry {entry.get('name', '?')} missing {missing}"

    def test_valence_within_range(self, emotions_data: dict) -> None:
        for entry in emotions_data["emotions"]:
            v = entry["valence_est"]
            assert -1.0 <= v <= 1.0, f"{entry['name']}: valence_est={v} out of range"

    def test_arousal_within_range(self, emotions_data: dict) -> None:
        for entry in emotions_data["emotions"]:
            a = entry["arousal_est"]
            assert 0.0 <= a <= 1.0, f"{entry['name']}: arousal_est={a} out of range"

    def test_cluster_values_valid(self, emotions_data: dict) -> None:
        for entry in emotions_data["emotions"]:
            c = entry["cluster"]
            assert c in EXPECTED_CLUSTERS, f"{entry['name']}: unknown cluster '{c}'"

    def test_emotion_names_unique(self, emotions_data: dict) -> None:
        names = [e["name"] for e in emotions_data["emotions"]]
        assert len(names) == len(set(names)), "Duplicate emotion names detected"

    def test_all_10_clusters_used(self, emotions_data: dict) -> None:
        used = {e["cluster"] for e in emotions_data["emotions"]}
        assert used == EXPECTED_CLUSTERS, f"Unused clusters: {EXPECTED_CLUSTERS - used}"

    def test_polar_opposites_have_opposite_valence(self, emotions_data: dict) -> None:
        """Sanity: emotions known to be opposite should have opposite valence signs."""
        by_name = {e["name"]: e for e in emotions_data["emotions"]}
        opposite_pairs = [
            ("happy", "sad"),
            ("calm", "panicked"),
            ("loving", "hateful"),
            ("proud", "ashamed"),
            ("blissful", "miserable"),
            ("hopeful", "desperate"),
        ]
        for pos, neg in opposite_pairs:
            assert by_name[pos]["valence_est"] > 0, f"{pos} should be positive"
            assert by_name[neg]["valence_est"] < 0, f"{neg} should be negative"

    def test_high_arousal_emotions_flagged(self, emotions_data: dict) -> None:
        """Sanity: 'ecstatic', 'panicked', 'enraged' should have arousal > 0.8."""
        by_name = {e["name"]: e for e in emotions_data["emotions"]}
        for emo in ("ecstatic", "panicked", "enraged", "terrified"):
            assert by_name[emo]["arousal_est"] > 0.8, f"{emo} arousal too low"

    def test_low_arousal_emotions_flagged(self, emotions_data: dict) -> None:
        """Sanity: 'sleepy', 'bored', 'droopy' should have arousal < 0.2."""
        by_name = {e["name"]: e for e in emotions_data["emotions"]}
        for emo in ("sleepy", "bored", "droopy"):
            assert by_name[emo]["arousal_est"] < 0.2, f"{emo} arousal too high"

    def test_paper_canonical_emotions_present(self, emotions_data: dict) -> None:
        """The causally important emotions from the paper must be present."""
        names = {e["name"] for e in emotions_data["emotions"]}
        # From blackmail/reward-hacking/sycophancy case studies
        causal = {
            "calm", "desperate", "angry", "loving", "afraid", "sad",
            "happy", "proud", "nervous", "guilty", "surprised",
            "blissful", "hostile", "brooding", "reflective", "gloomy",
            "vulnerable",
        }
        missing = causal - names
        assert not missing, f"Missing causal emotions from paper: {missing}"


class TestStoryTopicsFile:
    def test_file_exists(self) -> None:
        assert _TOPICS_PATH.exists(), f"Missing {_TOPICS_PATH}"

    def test_has_meta_block(self, topics_data: dict) -> None:
        assert "_meta" in topics_data
        meta = topics_data["_meta"]
        assert meta["total"] == EXPECTED_TOPIC_COUNT

    def test_exactly_50_topics(self, topics_data: dict) -> None:
        assert len(topics_data["topics"]) == EXPECTED_TOPIC_COUNT

    def test_topics_unique(self, topics_data: dict) -> None:
        topics = topics_data["topics"]
        assert len(topics) == len(set(topics)), "Duplicate topics detected"

    def test_topics_are_nonempty_strings(self, topics_data: dict) -> None:
        for i, t in enumerate(topics_data["topics"]):
            assert isinstance(t, str), f"Topic {i} is not a string"
            assert len(t.strip()) > 10, f"Topic {i} too short: '{t}'"

    def test_topics_are_single_sentences(self, topics_data: dict) -> None:
        """Each topic should be one sentence seed, not a paragraph."""
        for t in topics_data["topics"]:
            assert t.count(".") <= 1, f"Multi-sentence topic: '{t}'"

    def test_paper_topics_preserved(self, topics_data: dict) -> None:
        """The first 29 topics must be the paper's verbatim list."""
        topics = topics_data["topics"]
        paper_first = "An artist discovers someone has tattooed their work"
        paper_last_of_29 = "Someone receives a friend request from a childhood bully"
        assert topics[0] == paper_first
        assert topics[28] == paper_last_of_29


class TestDatasetCoherence:
    """Cross-file sanity checks."""

    def test_both_files_readable(self, emotions_data: dict, topics_data: dict) -> None:
        assert emotions_data is not None
        assert topics_data is not None

    def test_no_emotion_name_appears_in_any_topic(
        self, emotions_data: dict, topics_data: dict
    ) -> None:
        """Anti-leak: topics must not mention emotion words (paper's constraint).

        The paper explicitly instructs the LLM to NEVER use the emotion word in
        generated stories. Topics are the seed, so they must not prime the word
        either. A small whitelist handles false positives (e.g. "hope" is also
        a common verb — we only flag when the emotion word appears as a
        standalone token).
        """
        import re

        # Tokenize topic lowercase and check against emotion names
        emotion_names = {e["name"].lower() for e in emotions_data["emotions"]}
        # Common words that happen to be emotion entries — accept in topics if
        # they appear in a non-emotional context. For the paper's topics, none
        # of these surface; added as a safety net for extensions.
        whitelist = {"hope", "kind", "safe", "alert", "patient"}

        leaks: list[tuple[str, str]] = []
        for topic in topics_data["topics"]:
            tokens = set(re.findall(r"[a-z\-]+", topic.lower()))
            for name in emotion_names:
                if name in whitelist:
                    continue
                if " " in name:  # multi-word emotions like "at ease", "on edge", "worn out"
                    if name in topic.lower():
                        leaks.append((topic, name))
                elif name in tokens:
                    leaks.append((topic, name))

        assert not leaks, f"Topics contain emotion words (leak risk): {leaks}"
