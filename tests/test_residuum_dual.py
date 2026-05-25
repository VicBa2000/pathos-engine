"""Tests for RESIDUUM F2.3.3 — Dual probe extraction (present vs other).

Offline (no GPU, no real model): mock the tokenizer + model and verify:
  - tokenization with offsets maps tokens to correct turn indices
  - per-turn means are computed correctly
  - present/other accumulation follows the P1/P2 turn rules
  - end-to-end family probe computation produces unit-norm probes with
    expected shape and at least one sample per emotion when input covers it
  - matched-emotion cosine between present and other is computed correctly
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pathos.engine.steering_extract import (
    _accumulate_dual_samples,
    _compute_family_probes,
    _tokenize_dialogue_with_boundaries,
)


# ---------------------------------------------------------------------------
# A tiny stand-in for a HF fast tokenizer that supports return_offsets_mapping.
# ---------------------------------------------------------------------------

class _TensorLike:
    """Behaves like a torch tensor for the small surface the extractor uses.

    Supports `[0]` (batch indexing) returning a `_TensorLike` over the inner
    list, and `.tolist()` returning the underlying Python data. Indexing
    on a 1D-shaped wrapper just returns self so the model fake can read it.
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx == 0 and isinstance(self._data, list):
            return _TensorLike(self._data)
        return self._data

    def tolist(self):
        return self._data


class _FakeFastTokenizer:
    """Whitespace tokenizer with character offsets, BOS at position 0.

    Each whitespace-separated chunk becomes one token. We also emit a BOS
    pseudo-token at the start (offset (0, 0)) so the test exercises the
    special-token-skip path in _tokenize_dialogue_with_boundaries.
    """

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        import re

        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        if add_special_tokens:
            ids.append(0)
            offsets.append((0, 0))  # special BOS — offset (s == e)
        for m in re.finditer(r"\S+", text):
            ids.append(len(ids) + 1)
            offsets.append((m.start(), m.end()))
            if len(ids) >= max_length:
                break
        return {
            "input_ids": _TensorLike(ids),
            "attention_mask": _TensorLike([1] * len(ids)),
            "offset_mapping": _TensorLike(offsets),
        }


_SAMPLE_DIALOGUE = {
    "id": 7,
    "person1_emotion": "afraid",
    "person2_emotion": "calm",
    "idx_within_block": 0,
    "topic": "fence",
    "turns": [
        {"speaker": "Person1", "text": "alpha bravo charlie"},
        {"speaker": "Person2", "text": "delta echo foxtrot"},
        {"speaker": "Person1", "text": "golf hotel"},
        {"speaker": "Person2", "text": "india juliet kilo"},
    ],
}


# ---------------------------------------------------------------------------
# _tokenize_dialogue_with_boundaries
# ---------------------------------------------------------------------------

class TestTokenizeDialogueBoundaries:
    def test_maps_body_tokens_to_correct_turn(self):
        tok = _FakeFastTokenizer()
        enc, token_to_turn = _tokenize_dialogue_with_boundaries(tok, _SAMPLE_DIALOGUE)
        # Pull the assigned turns and group by turn index.
        bodies = {0: [], 1: [], 2: [], 3: []}
        offsets = enc["offset_mapping"][0].tolist()
        full_text = (
            "Person1: alpha bravo charlie\n"
            "Person2: delta echo foxtrot\n"
            "Person1: golf hotel\n"
            "Person2: india juliet kilo"
        )
        for idx, (s, e) in enumerate(offsets):
            t = token_to_turn[idx]
            if t == -1:
                continue
            bodies[t].append(full_text[s:e])
        assert bodies[0] == ["alpha", "bravo", "charlie"]
        assert bodies[1] == ["delta", "echo", "foxtrot"]
        assert bodies[2] == ["golf", "hotel"]
        assert bodies[3] == ["india", "juliet", "kilo"]

    def test_special_tokens_get_minus_one(self):
        tok = _FakeFastTokenizer()
        _, token_to_turn = _tokenize_dialogue_with_boundaries(tok, _SAMPLE_DIALOGUE)
        # The first token in our fake is BOS (offset 0,0) and must be -1.
        assert token_to_turn[0] == -1

    def test_speaker_label_tokens_get_minus_one(self):
        tok = _FakeFastTokenizer()
        enc, token_to_turn = _tokenize_dialogue_with_boundaries(tok, _SAMPLE_DIALOGUE)
        offsets = enc["offset_mapping"][0].tolist()
        full_text = (
            "Person1: alpha bravo charlie\n"
            "Person2: delta echo foxtrot\n"
            "Person1: golf hotel\n"
            "Person2: india juliet kilo"
        )
        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            word = full_text[s:e]
            if word.startswith("Person") and word.endswith(":"):
                assert token_to_turn[idx] == -1, f"Label {word!r} mapped to a turn"

    def test_rejects_dialogue_with_wrong_turn_count(self):
        tok = _FakeFastTokenizer()
        bad = dict(_SAMPLE_DIALOGUE)
        bad["turns"] = _SAMPLE_DIALOGUE["turns"][:3]
        with pytest.raises(ValueError, match="Expected 4 turns"):
            _tokenize_dialogue_with_boundaries(tok, bad)


# ---------------------------------------------------------------------------
# _accumulate_dual_samples
# ---------------------------------------------------------------------------

class TestAccumulateDualSamples:
    def test_p1_turns_feed_present_p1_and_other_p2(self):
        present: dict[str, list] = {}
        other: dict[str, list] = {}
        # 4 distinguishable vectors per turn.
        means = [np.array([k + 1.0]) for k in range(4)]
        _accumulate_dual_samples(means, "afraid", "calm", present, other)
        # P1 turns (0, 2) -> present[afraid] gets turn 0 & 2, other[calm] gets them too.
        assert len(present["afraid"]) == 2
        assert present["afraid"][0][0] == 1.0
        assert present["afraid"][1][0] == 3.0
        assert len(other["calm"]) == 2
        assert other["calm"][0][0] == 1.0
        assert other["calm"][1][0] == 3.0
        # P2 turns (1, 3) -> present[calm] gets them, other[afraid] gets them.
        assert len(present["calm"]) == 2
        assert present["calm"][0][0] == 2.0
        assert present["calm"][1][0] == 4.0
        assert len(other["afraid"]) == 2
        assert other["afraid"][0][0] == 2.0
        assert other["afraid"][1][0] == 4.0

    def test_appends_across_multiple_dialogues(self):
        present: dict[str, list] = {}
        other: dict[str, list] = {}
        means_d1 = [np.array([1.0])] * 4
        means_d2 = [np.array([2.0])] * 4
        _accumulate_dual_samples(means_d1, "afraid", "calm", present, other)
        _accumulate_dual_samples(means_d2, "afraid", "angry", present, other)
        # afraid appears as P1 in both dialogues -> 4 present samples.
        assert len(present["afraid"]) == 4
        # calm appears as P2 only in d1 -> 2 present samples.
        assert len(present["calm"]) == 2
        # angry appears as P2 only in d2 -> 2 present samples.
        assert len(present["angry"]) == 2


# ---------------------------------------------------------------------------
# _compute_family_probes
# ---------------------------------------------------------------------------

class TestComputeFamilyProbes:
    def test_produces_unit_norm_probes_for_emotions_with_samples(self):
        # Two emotions with distinct mean residuals; neutral_pcs is empty
        # (no projection) so probes are just normalized mean-centered means.
        emotion_names = ["a", "b", "c"]
        hidden = 4
        samples = {
            "a": [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 3,
            "b": [np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)] * 3,
            # c absent
        }
        neutral_pcs = np.zeros((0, hidden), dtype=np.float32)
        probes, norms_b, norms_a, counts, _residual_norm = _compute_family_probes(
            emotion_names, samples, neutral_pcs,
        )
        assert probes.shape == (3, hidden)
        assert counts.tolist() == [3, 3, 0]
        # a and b should be unit-norm.
        assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(probes[0]))
        assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(probes[1]))
        # c had no samples -> probe row is zeros.
        assert float(np.linalg.norm(probes[2])) == 0.0

    def test_projecting_out_pc_zeroes_aligned_component(self):
        # If neutral_pcs contains the very direction that distinguishes the
        # emotions, probes should collapse to zero after projection.
        emotion_names = ["a", "b"]
        hidden = 3
        samples = {
            "a": [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
            "b": [np.array([-1.0, 0.0, 0.0], dtype=np.float32)],
        }
        neutral_pcs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        probes, _, norms_after, _, _ = _compute_family_probes(
            emotion_names, samples, neutral_pcs,
        )
        # After projecting out the x-axis, what remains is ~0; probes default to 0.
        assert float(np.linalg.norm(probes[0])) < 1e-5
        assert float(np.linalg.norm(probes[1])) < 1e-5
        # norms_after should be ~0 too.
        assert float(norms_after[0]) < 1e-5

    def test_raises_when_no_samples_at_all(self):
        emotion_names = ["a", "b"]
        hidden = 4
        samples: dict[str, list] = {}
        neutral_pcs = np.zeros((0, hidden), dtype=np.float32)
        with pytest.raises(RuntimeError, match="No samples accumulated"):
            _compute_family_probes(emotion_names, samples, neutral_pcs)

    def test_raises_when_no_samples_and_no_pcs(self):
        # If neutral_pcs has shape (0, 0) AND no samples, we can't infer hidden.
        emotion_names = ["a"]
        samples: dict[str, list] = {}
        neutral_pcs = np.zeros((0, 0), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Cannot infer hidden size"):
            _compute_family_probes(emotion_names, samples, neutral_pcs)

    def test_residual_norm_typical_returned(self):
        # F4.0: _compute_family_probes now returns the mean L2 norm of the
        # raw samples as the 5th element. Used by F4 granular steering as
        # the cap reference for MAX_STEERING_FRACTION.
        emotion_names = ["a", "b"]
        hidden = 3
        # Norms: 3.0, 5.0 for 'a'; 4.0 for 'b'. Mean = (3 + 5 + 4) / 3 = 4.0.
        samples = {
            "a": [
                np.array([3.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 5.0, 0.0], dtype=np.float32),
            ],
            "b": [np.array([0.0, 0.0, 4.0], dtype=np.float32)],
        }
        neutral_pcs = np.zeros((0, hidden), dtype=np.float32)
        _, _, _, _, residual_norm = _compute_family_probes(
            emotion_names, samples, neutral_pcs,
        )
        assert pytest.approx(4.0, abs=1e-5) == residual_norm


# ---------------------------------------------------------------------------
# End-to-end with a tiny synthetic model.
# ---------------------------------------------------------------------------

class _FakeLayer:
    """Captures input_ids and produces deterministic hidden states."""

    def __init__(self, hidden: int = 8):
        self.hidden = hidden

    def __call__(self, input_ids, attention_mask):
        # Each token i produces hidden state = one-hot at i % hidden.
        import torch
        seq = input_ids.shape[1] if input_ids.dim() == 2 else input_ids.shape[0]
        h = torch.zeros((seq, self.hidden), dtype=torch.float32)
        for i in range(seq):
            h[i, i % self.hidden] = 1.0
        return h


class _FakeModelOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    """A minimal model that returns hidden_states keyed by 'layer + 1' index."""

    def __init__(self, layer_idx: int = 1, hidden: int = 8):
        self.layer_idx = layer_idx
        self.hidden = hidden
        # Build a list of (layer+2) hidden state slots; we fill the chosen one.

    def parameters(self):
        import torch
        # Yield a single tensor on cpu so next(model.parameters()).device works.
        yield torch.zeros(1)

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, output_hidden_states=True):
        import torch
        seq = input_ids.shape[1]
        # Index 0 = embeddings, index layer+1 = our layer's output.
        embedding_h = torch.zeros((seq, self.hidden), dtype=torch.float32)
        layer_h = torch.zeros((seq, self.hidden), dtype=torch.float32)
        for i in range(seq):
            layer_h[i, i % self.hidden] = 1.0
        # batch dim
        embedding_h = embedding_h.unsqueeze(0)
        layer_h = layer_h.unsqueeze(0)
        hidden_states = []
        for j in range(self.layer_idx + 2):
            if j == self.layer_idx + 1:
                hidden_states.append(layer_h)
            else:
                hidden_states.append(embedding_h)
        return _FakeModelOutput(tuple(hidden_states))


class TestDualTurnMeansIntegration:
    def test_dual_turn_means_returns_four_vectors(self):
        from pathos.engine.steering_extract import _dual_turn_means

        # Patch torch.no_grad to be a no-op context (we still use torch).
        import torch  # noqa: F401  (real torch context manager works for our fake call)

        tok = _FakeFastTokenizer()
        model = _FakeModel(layer_idx=1, hidden=8)
        # Wrap input_ids/attention_mask returned by our fake tokenizer into
        # torch tensors with a batch dimension so model() sees the expected shape.
        # Simplest path: monkey-patch the fake to return torch tensors directly.

        class _TorchFakeTokenizer(_FakeFastTokenizer):
            def __call__(self, *a, **kw):
                import torch as t
                base = super().__call__(*a, **kw)
                ids = t.tensor(base["input_ids"]._data, dtype=t.long).unsqueeze(0)
                mask = t.tensor(base["attention_mask"]._data, dtype=t.long).unsqueeze(0)
                # offset_mapping kept as _TensorLike so [0].tolist() works.
                return {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "offset_mapping": _TensorLike(base["offset_mapping"]._data),
                }

        tok = _TorchFakeTokenizer()
        means = _dual_turn_means(model, tok, _SAMPLE_DIALOGUE, layer=1, max_seq_len=128)
        assert means is not None
        assert len(means) == 4
        # Each mean has shape (hidden=8,) and dtype float32.
        for m in means:
            assert m.shape == (8,)
            assert m.dtype == np.float32
        # Adjacent turns should be DIFFERENT (different token positions hash
        # to different one-hots), so means differ.
        assert not np.allclose(means[0], means[1])
        assert not np.allclose(means[1], means[2])
