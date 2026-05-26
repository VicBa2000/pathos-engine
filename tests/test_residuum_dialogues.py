"""Tests for RESIDUUM F2.3 dialogue generator.

Validates the offline dialogue generator that produces dialogues_171.json:
prompt construction, turn parsing, double anti-leak, resumability, and the
end-to-end orchestrator with a mocked LLM. No real Ollama call here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from pathos.engine.steering_extract import (
    _build_dialogue_prompt,
    _dialogue_contains_either_emotion,
    _load_existing_dialogues,
    _parse_dialogue_turns,
    _save_dialogues_atomic,
    generate_dialogues_for_emotions,
)


# ---------------------------------------------------------------------------
# _build_dialogue_prompt
# ---------------------------------------------------------------------------

class TestBuildDialoguePrompt:
    def test_contains_both_emotions_and_topic(self):
        p = _build_dialogue_prompt("delighted", "afraid", "Two friends share a meal")
        assert "delighted" in p
        assert "afraid" in p
        assert "Two friends share a meal" in p

    def test_specifies_4_turns_alternating(self):
        p = _build_dialogue_prompt("calm", "angry", "topic")
        # The prompt must contain the strict 4-turn format scaffold.
        assert p.count("Person1:") >= 2
        assert p.count("Person2:") >= 2

    def test_anti_leak_clause_present(self):
        p = _build_dialogue_prompt("ashamed", "proud", "topic")
        assert "NEVER appear" in p
        # Both emotion words must be quoted in the rule.
        assert "'ashamed'" in p
        assert "'proud'" in p

    def test_demands_spoken_lines_not_stage_directions(self):
        # The refined prompt must steer the model away from emitting only
        # stage directions in asterisks/parentheses (which a generation run
        # produced before this fix).
        p = _build_dialogue_prompt("calm", "angry", "topic")
        assert "SAYS OUT LOUD" in p
        assert "NOT through stage directions" in p


# ---------------------------------------------------------------------------
# _parse_dialogue_turns
# ---------------------------------------------------------------------------

class TestParseDialogueTurns:
    def test_parses_valid_4_turn_dialogue(self):
        raw = (
            "Person1: I cant believe what just happened.\n"
            "Person2: You should sit down for a moment.\n"
            "Person1: My hands wont stop shaking right now.\n"
            "Person2: Take a slow breath. I am here with you.\n"
        )
        turns = _parse_dialogue_turns(raw)
        assert turns is not None
        assert len(turns) == 4
        assert [t["speaker"] for t in turns] == ["Person1", "Person2", "Person1", "Person2"]
        assert turns[0]["text"].startswith("I cant")

    def test_accepts_space_between_person_and_number(self):
        raw = (
            "Person 1: line one with enough words.\n"
            "Person 2: line two with enough words.\n"
            "Person 1: line three with enough words.\n"
            "Person 2: line four with enough words.\n"
        )
        turns = _parse_dialogue_turns(raw)
        assert turns is not None
        assert len(turns) == 4

    def test_rejects_only_3_turns(self):
        raw = (
            "Person1: line a\n"
            "Person2: line b\n"
            "Person1: line c\n"
        )
        assert _parse_dialogue_turns(raw) is None

    def test_truncates_to_4_when_more_present(self):
        # Thinking models sometimes emit extra Person labels after the 4-turn
        # dialogue (in their post-hoc reasoning). The parser keeps the first 4.
        raw = (
            "Person1: first turn body here\n"
            "Person2: second turn body here\n"
            "Person1: third turn body here\n"
            "Person2: fourth turn body here\n"
            "Person1: leaked thinking turn that should not count\n"
            "Person2: another leaked thinking turn\n"
        )
        turns = _parse_dialogue_turns(raw)
        assert turns is not None
        assert len(turns) == 4
        assert turns[0]["text"].startswith("first turn")
        assert turns[3]["text"].startswith("fourth turn")

    def test_strips_trailing_think_tag_from_last_turn(self):
        # The 4-turn body must not contain leaked '<think>' content even when
        # the final turn's regex capture extends past the dialogue.
        raw = (
            "Person1: body one with enough words to pass.\n"
            "Person2: body two with enough words to pass.\n"
            "Person1: body three with enough words to pass.\n"
            "Person2: body four with enough words to pass.\n"
            "<think>Okay so now I should explain my approach...</think>\n"
        )
        turns = _parse_dialogue_turns(raw)
        assert turns is not None
        assert "<think>" not in turns[3]["text"]
        assert turns[3]["text"].endswith("pass.")

    def test_rejects_non_alternating_sequence(self):
        raw = (
            "Person1: line a\nPerson1: line b\nPerson2: line c\nPerson2: line d\n"
        )
        assert _parse_dialogue_turns(raw) is None

    def test_rejects_empty_turn_body(self):
        raw = (
            "Person1: a body here\n"
            "Person2: \n"
            "Person1: another body\n"
            "Person2: final body\n"
        )
        assert _parse_dialogue_turns(raw) is None

    def test_handles_multiline_turn_body(self):
        raw = (
            "Person1: This is line one\nstill the first turn.\n"
            "Person2: Now the second.\n"
            "Person1: Third turn.\n"
            "Person2: Fourth and final turn.\n"
        )
        turns = _parse_dialogue_turns(raw)
        assert turns is not None
        assert "still the first turn" in turns[0]["text"]


# ---------------------------------------------------------------------------
# _dialogue_contains_either_emotion (double anti-leak)
# ---------------------------------------------------------------------------

class TestDoubleAntiLeak:
    def test_clean_dialogue_passes(self):
        turns = [
            {"speaker": "Person1", "text": "I feel hollow inside, like nothing matters."},
            {"speaker": "Person2", "text": "Stay still. The world will keep spinning."},
            {"speaker": "Person1", "text": "Maybe. Maybe not."},
            {"speaker": "Person2", "text": "Either way I am here for you."},
        ]
        assert not _dialogue_contains_either_emotion(turns, "sad", "calm")

    def test_first_emotion_leak_detected(self):
        turns = [
            {"speaker": "Person1", "text": "I am so sad about this turn of events."},
            {"speaker": "Person2", "text": "Stay still. The world will keep spinning."},
            {"speaker": "Person1", "text": "Maybe."},
            {"speaker": "Person2", "text": "Here for you."},
        ]
        assert _dialogue_contains_either_emotion(turns, "sad", "calm")

    def test_second_emotion_leak_detected(self):
        turns = [
            {"speaker": "Person1", "text": "Everything is hollow."},
            {"speaker": "Person2", "text": "I am quite calm about this situation."},
            {"speaker": "Person1", "text": "Maybe."},
            {"speaker": "Person2", "text": "Here for you."},
        ]
        assert _dialogue_contains_either_emotion(turns, "sad", "calm")

    def test_morphological_leak_detected(self):
        # 'amused' -> stem 'amus' covers 'amusing', 'amusement'.
        turns = [
            {"speaker": "Person1", "text": "This is rather amusing to watch."},
            {"speaker": "Person2", "text": "I dont see what is funny here."},
            {"speaker": "Person1", "text": "You will."},
            {"speaker": "Person2", "text": "Doubt it."},
        ]
        assert _dialogue_contains_either_emotion(turns, "amused", "angry")


# ---------------------------------------------------------------------------
# _load_existing_dialogues + _save_dialogues_atomic
# ---------------------------------------------------------------------------

class TestExistingDialoguesIO:
    def test_load_missing_file_returns_empty_dict(self, tmp_path: Path):
        result = _load_existing_dialogues(tmp_path / "does-not-exist.json")
        assert result == {}

    def test_load_malformed_json_returns_empty(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {", encoding="utf-8")
        assert _load_existing_dialogues(bad) == {}

    def test_load_indexes_by_person1_and_block_idx(self, tmp_path: Path):
        target = tmp_path / "d.json"
        target.write_text(
            json.dumps(
                {
                    "_meta": {},
                    "dialogues": [
                        {
                            "id": 0,
                            "person1_emotion": "afraid",
                            "person2_emotion": "calm",
                            "idx_within_block": 0,
                            "topic": "t",
                            "turns": [],
                        },
                        {
                            "id": 1,
                            "person1_emotion": "afraid",
                            "person2_emotion": "happy",
                            "idx_within_block": 1,
                            "topic": "t",
                            "turns": [],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        existing = _load_existing_dialogues(target)
        assert ("afraid", 0) in existing
        assert ("afraid", 1) in existing
        assert existing[("afraid", 1)]["person2_emotion"] == "happy"

    def test_save_atomic_writes_payload_and_cleans_tmp(self, tmp_path: Path):
        target = tmp_path / "out.json"
        dialogues = [
            {
                "id": 0,
                "person1_emotion": "afraid",
                "person2_emotion": "calm",
                "idx_within_block": 0,
                "topic": "t",
                "turns": [{"speaker": "Person1", "text": "x"}],
            }
        ]
        _save_dialogues_atomic(target, dialogues, n_per_emotion=5)
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["_meta"]["n_per_emotion_as_person1"] == 5
        assert data["_meta"]["total_dialogues"] == 1
        assert len(data["dialogues"]) == 1
        # Tmp file must not persist.
        assert not (target.with_suffix(target.suffix + ".tmp")).exists()


# ---------------------------------------------------------------------------
# generate_dialogues_for_emotions (mocked LLM)
# ---------------------------------------------------------------------------

_MINI_EMOTIONS = [
    {"name": "afraid", "valence_est": -0.7, "arousal_est": 0.75},
    {"name": "calm", "valence_est": 0.5, "arousal_est": 0.2},
    {"name": "angry", "valence_est": -0.7, "arousal_est": 0.8},
]
_MINI_TOPICS = ["A friend visits unexpectedly"]


def _good_llm_response(emo_a: str, emo_b: str) -> str:
    """Return a 4-turn dialogue that never contains either emotion word."""
    # 'soft' and 'tight' do not collide with 'afraid', 'calm', 'angry' stems.
    return (
        f"Person1: My chest feels tight whenever the door opens like that.\n"
        f"Person2: Sit with me for a moment. The room is quiet enough.\n"
        f"Person1: I keep checking the locks and the windows again.\n"
        f"Person2: Breathe slowly. The world is going to wait for you.\n"
    )


def _leaky_llm_response(emo_a: str, emo_b: str) -> str:
    """Return a 4-turn dialogue that leaks both emotion words."""
    return (
        f"Person1: I am {emo_a} right now and cannot hide it.\n"
        f"Person2: I am perfectly {emo_b} despite all of this commotion.\n"
        f"Person1: My hands are unsteady though.\n"
        f"Person2: That will pass. Just give it time.\n"
    )


def _mostly_good_llm_factory(leak_first_n: int) -> Callable[[str], str]:
    """LLM that leaks the first N calls and then returns clean output."""
    state = {"calls": 0}

    def _call(prompt: str) -> str:
        state["calls"] += 1
        # Extract emotion names from prompt for the leaky response.
        # Prompt format: "Person1 is feeling X. Person2 is feeling Y."
        # We dont actually need the emotions; just produce a leaky-looking string.
        if state["calls"] <= leak_first_n:
            return _leaky_llm_response("afraid", "calm")
        return _good_llm_response("afraid", "calm")

    return _call


class TestGenerateDialoguesOrchestrator:
    def test_generates_n_per_emotion_clean_dialogues(self, tmp_path: Path):
        out = tmp_path / "dialogues.json"
        counts = generate_dialogues_for_emotions(
            n_per_emotion=2,
            llm_call=lambda prompt: _good_llm_response("afraid", "calm"),
            output_path=out,
            skip_existing=False,
            seed=7,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        assert counts["generated"] == len(_MINI_EMOTIONS) * 2
        assert counts["failed"] == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["dialogues"]) == len(_MINI_EMOTIONS) * 2
        # Schema sanity: every dialogue has the required keys.
        for d in data["dialogues"]:
            assert {"id", "person1_emotion", "person2_emotion",
                    "idx_within_block", "topic", "turns"} <= set(d)
            assert len(d["turns"]) == 4

    def test_person2_distinct_from_person1(self, tmp_path: Path):
        out = tmp_path / "d.json"
        generate_dialogues_for_emotions(
            n_per_emotion=2,
            llm_call=lambda p: _good_llm_response("a", "b"),
            output_path=out,
            skip_existing=False,
            seed=1,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        for d in data["dialogues"]:
            assert d["person1_emotion"] != d["person2_emotion"]

    def test_p2_partners_within_block_are_distinct(self, tmp_path: Path):
        # With 3 emotions and n_per_emotion=2, each block has 2 distinct partners.
        out = tmp_path / "d.json"
        generate_dialogues_for_emotions(
            n_per_emotion=2,
            llm_call=lambda p: _good_llm_response("a", "b"),
            output_path=out,
            skip_existing=False,
            seed=5,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        # Group by person1_emotion and check P2s are distinct in each block.
        blocks: dict[str, list[str]] = {}
        for d in data["dialogues"]:
            blocks.setdefault(d["person1_emotion"], []).append(d["person2_emotion"])
        for a, p2_list in blocks.items():
            assert len(set(p2_list)) == len(p2_list), f"Block {a} has duplicate P2s: {p2_list}"

    def test_retries_on_anti_leak_hit(self, tmp_path: Path):
        out = tmp_path / "d.json"
        # Leak only the first call; subsequent calls clean. With max_retries=3
        # the first dialogue retries once and succeeds; the next two succeed
        # immediately. Total: 3 generated, 0 failed.
        leaky_then_clean = _mostly_good_llm_factory(leak_first_n=1)
        counts = generate_dialogues_for_emotions(
            n_per_emotion=1,
            llm_call=leaky_then_clean,
            output_path=out,
            skip_existing=False,
            max_retries_per_dialogue=3,
            seed=11,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        assert counts["generated"] == 3
        assert counts["failed"] == 0

    def test_failure_when_all_retries_leak(self, tmp_path: Path):
        out = tmp_path / "d.json"
        counts = generate_dialogues_for_emotions(
            n_per_emotion=1,
            llm_call=lambda p: _leaky_llm_response("afraid", "calm"),
            output_path=out,
            skip_existing=False,
            max_retries_per_dialogue=2,
            seed=13,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        # Every dialogue should fail because every attempt leaks.
        assert counts["generated"] == 0
        assert counts["failed"] == len(_MINI_EMOTIONS)

    def test_skip_existing_reuses_prior_dialogues(self, tmp_path: Path):
        out = tmp_path / "d.json"
        # First run: full generation.
        counts1 = generate_dialogues_for_emotions(
            n_per_emotion=1,
            llm_call=lambda p: _good_llm_response("a", "b"),
            output_path=out,
            skip_existing=False,
            seed=21,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        assert counts1["generated"] == 3

        # Second run: skip_existing=True should reuse without calling LLM.
        call_count = {"n": 0}

        def _failing_llm(prompt: str) -> str:
            call_count["n"] += 1
            raise AssertionError("LLM should not be called when reusing existing")

        counts2 = generate_dialogues_for_emotions(
            n_per_emotion=1,
            llm_call=_failing_llm,
            output_path=out,
            skip_existing=True,
            seed=21,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        assert counts2["skipped"] == 3
        assert counts2["generated"] == 0
        assert call_count["n"] == 0

    def test_rejects_n_per_emotion_too_large(self, tmp_path: Path):
        out = tmp_path / "d.json"
        with pytest.raises(ValueError, match="exceeds available distinct partners"):
            generate_dialogues_for_emotions(
                n_per_emotion=10,  # only 2 partners available
                llm_call=lambda p: _good_llm_response("a", "b"),
                output_path=out,
                skip_existing=False,
                seed=1,
                emotions=_MINI_EMOTIONS,
                topics=_MINI_TOPICS,
            )

    def test_invalidates_existing_on_anti_leak_and_regenerates(self, tmp_path: Path):
        """If an existing dialogue contains the emotion word, regenerate it."""
        out = tmp_path / "d.json"
        # Pre-seed with a leaky dialogue for ("afraid", 0).
        leaky = {
            "_meta": {"n_per_emotion_as_person1": 1},
            "dialogues": [
                {
                    "id": 0,
                    "person1_emotion": "afraid",
                    "person2_emotion": "calm",
                    "idx_within_block": 0,
                    "topic": "T",
                    "turns": [
                        {"speaker": "Person1", "text": "I am afraid of the dark."},
                        {"speaker": "Person2", "text": "Let me sit with you."},
                        {"speaker": "Person1", "text": "Thank you."},
                        {"speaker": "Person2", "text": "Always."},
                    ],
                },
            ],
        }
        out.write_text(json.dumps(leaky), encoding="utf-8")

        counts = generate_dialogues_for_emotions(
            n_per_emotion=1,
            llm_call=lambda p: _good_llm_response("a", "b"),
            output_path=out,
            skip_existing=True,
            seed=33,
            emotions=_MINI_EMOTIONS,
            topics=_MINI_TOPICS,
        )
        # Leaky one for afraid regenerates, others generate fresh.
        assert counts["generated"] >= 1
        # Check resulting file has no leak for afraid.
        data = json.loads(out.read_text(encoding="utf-8"))
        afraid_dialogues = [d for d in data["dialogues"] if d["person1_emotion"] == "afraid"]
        assert len(afraid_dialogues) == 1
        full_text = " ".join(t["text"] for t in afraid_dialogues[0]["turns"]).lower()
        assert "afraid" not in full_text
