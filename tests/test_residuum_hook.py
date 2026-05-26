"""Tests for IntrospectiveTransformersProvider — F2.1 hook infrastructure.

Uses a tiny torch.nn module mock instead of loading a real LLM. The mock
mimics the HF causal-LM structure that `_resolve_layer_module()` walks
(model.model.layers[i]) so the hook attaches in the same way it would on
Qwen/Llama/Mistral.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

# Skip the whole module if torch is unavailable in the test environment.
torch = pytest.importorskip("torch")
torch_nn = pytest.importorskip("torch.nn")


HIDDEN = 8
NUM_LAYERS = 4
TARGET_LAYER = 2


# ---------------------------------------------------------------------------
# Mock model: minimal HF-like structure so _resolve_layer_module() works.
# ---------------------------------------------------------------------------


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor"]:
        # HF decoder layers return a tuple (hidden_states, ...)
        return (self.proj(x),)


class _FakeBackbone(torch.nn.Module):
    def __init__(self, hidden: int = HIDDEN, n_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_FakeDecoderLayer(hidden) for _ in range(n_layers)]
        )


class _FakeCausalLM(torch.nn.Module):
    """Minimal stand-in for AutoModelForCausalLM exposing model.model.layers."""

    def __init__(self, hidden: int = HIDDEN, n_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.model = _FakeBackbone(hidden, n_layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = x
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


def _make_provider(introspection_enabled: bool = True, target_layer: int = TARGET_LAYER):
    """Construct a provider without calling _ensure_loaded.

    We bypass model download by injecting a fake model directly. That keeps
    the test isolated from network/HF/transformers internals.
    """
    from pathos.llm.introspective_provider import IntrospectiveTransformersProvider

    p = IntrospectiveTransformersProvider(
        target_layer=target_layer,
        introspection_enabled=introspection_enabled,
        model_id="qwen3:4b",  # ignored — we replace _model below
    )
    p._model = _FakeCausalLM()
    p._tokenizer = object()
    p._loaded = True
    return p


def _run_forward(p: Any, batch: int = 1, seq: int = 5) -> "torch.Tensor":
    x = torch.randn(batch, seq, HIDDEN)
    with torch.no_grad():
        return p._model(x)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_target_layer_stored(self) -> None:
        p = _make_provider(target_layer=3)
        assert p.target_layer == 3

    def test_introspection_default_enabled(self) -> None:
        p = _make_provider()
        assert p.introspection_enabled is True

    def test_introspection_can_be_disabled_at_init(self) -> None:
        p = _make_provider(introspection_enabled=False)
        assert p.introspection_enabled is False

    def test_no_hook_before_load_or_capture(self) -> None:
        p = _make_provider()
        assert p._hook_handle is None
        assert p.has_capture() is False


# ---------------------------------------------------------------------------
# Hook registration / cleanup
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    def test_register_attaches_to_correct_layer(self) -> None:
        p = _make_provider(target_layer=2)
        p._ensure_hook_registered()
        try:
            assert p._hook_handle is not None
            target = p._model.model.layers[2]
            # PyTorch stores forward hooks in _forward_hooks (OrderedDict)
            assert len(target._forward_hooks) == 1
        finally:
            p._remove_hook()

    def test_remove_clears_handle(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        target = p._model.model.layers[TARGET_LAYER]
        assert len(target._forward_hooks) == 1
        p._remove_hook()
        assert p._hook_handle is None
        assert len(target._forward_hooks) == 0

    def test_register_idempotent(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        first = p._hook_handle
        p._ensure_hook_registered()
        try:
            assert p._hook_handle is first
        finally:
            p._remove_hook()

    def test_unknown_layer_index_raises(self) -> None:
        p = _make_provider(target_layer=999)
        with pytest.raises(RuntimeError):
            p._ensure_hook_registered()

    def test_negative_layer_index_raises(self) -> None:
        p = _make_provider(target_layer=-1)
        with pytest.raises(RuntimeError):
            p._ensure_hook_registered()


# ---------------------------------------------------------------------------
# Capture mechanics
# ---------------------------------------------------------------------------


class TestCapture:
    def test_forward_appends_one_residual(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, batch=1, seq=5)
            assert p.has_capture() is True
            arrs = p.get_all_residuals()
            assert len(arrs) == 1
            assert arrs[0].shape == (1, 5, HIDDEN)
            assert arrs[0].dtype == np.float32
        finally:
            p._remove_hook()

    def test_multiple_forwards_accumulate(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=5)  # prompt
            _run_forward(p, seq=1)  # token 1
            _run_forward(p, seq=1)  # token 2
            assert len(p.get_all_residuals()) == 3
        finally:
            p._remove_hook()

    def test_disabled_provider_does_not_capture(self) -> None:
        p = _make_provider(introspection_enabled=False)
        # No call to _ensure_hook_registered (mirrors generate path).
        _run_forward(p, seq=5)
        assert p.has_capture() is False

    def test_reset_capture_empties_buffer(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=3)
            assert p.has_capture() is True
            p.reset_capture()
            assert p.has_capture() is False
            assert p.get_all_residuals() == []
        finally:
            p._remove_hook()

    def test_hook_does_not_alter_forward_output(self) -> None:
        # Run twice: once with the hook off, once with it on. Output must match.
        p = _make_provider(introspection_enabled=False)
        torch.manual_seed(123)
        x = torch.randn(1, 4, HIDDEN)
        with torch.no_grad():
            no_hook = p._model(x).clone()

        p._ensure_hook_registered()
        try:
            with torch.no_grad():
                with_hook = p._model(x).clone()
        finally:
            p._remove_hook()

        assert torch.allclose(no_hook, with_hook, atol=1e-6)


# ---------------------------------------------------------------------------
# Position accessors
# ---------------------------------------------------------------------------


class TestPositionAccessors:
    def test_prompt_end_returns_last_token_of_first_forward(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=6)
            end = p.get_prompt_end_residual()
            assert end is not None
            assert end.shape == (HIDDEN,)
            # Confirm it equals slice [0, -1, :] of first capture
            np.testing.assert_array_equal(end, p.get_all_residuals()[0][0, -1, :])
        finally:
            p._remove_hook()

    def test_prompt_end_none_when_no_capture(self) -> None:
        p = _make_provider()
        assert p.get_prompt_end_residual() is None

    def test_response_residuals_excludes_prompt(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=5)  # prompt (skipped by accessor)
            _run_forward(p, seq=1)  # token 1
            _run_forward(p, seq=1)  # token 2
            tokens = p.get_response_residuals()
            assert len(tokens) == 2
            for t in tokens:
                assert t.shape == (HIDDEN,)
        finally:
            p._remove_hook()

    def test_response_residuals_empty_when_no_generated_tokens(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=5)  # only the prompt
            assert p.get_response_residuals() == []
        finally:
            p._remove_hook()

    def test_response_mean_is_mean_of_response_tokens(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=5)  # prompt
            _run_forward(p, seq=1)
            _run_forward(p, seq=1)
            mean = p.get_response_mean_residual()
            assert mean is not None
            expected = np.mean(np.stack(p.get_response_residuals(), axis=0), axis=0)
            np.testing.assert_allclose(mean, expected, atol=1e-6)
        finally:
            p._remove_hook()

    def test_response_mean_none_when_no_response(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        try:
            _run_forward(p, seq=5)
            assert p.get_response_mean_residual() is None
        finally:
            p._remove_hook()


# ---------------------------------------------------------------------------
# Runtime toggle
# ---------------------------------------------------------------------------


class TestRuntimeToggle:
    def test_set_introspection_off_removes_hook(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        target = p._model.model.layers[TARGET_LAYER]
        assert len(target._forward_hooks) == 1
        p.set_introspection(False)
        assert len(target._forward_hooks) == 0
        assert p.introspection_enabled is False

    def test_set_introspection_on_registers_hook(self) -> None:
        p = _make_provider(introspection_enabled=False)
        target = p._model.model.layers[TARGET_LAYER]
        assert len(target._forward_hooks) == 0
        p.set_introspection(True)
        try:
            assert len(target._forward_hooks) == 1
            assert p.introspection_enabled is True
        finally:
            p._remove_hook()

    def test_toggle_off_then_capture_is_silent(self) -> None:
        p = _make_provider()
        p._ensure_hook_registered()
        p.set_introspection(False)
        _run_forward(p, seq=5)
        assert p.has_capture() is False
