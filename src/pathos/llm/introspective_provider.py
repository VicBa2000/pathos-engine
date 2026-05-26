"""IntrospectiveTransformersProvider — F2.1 Hook infrastructure.

Subclass of TransformersProvider that registers a forward hook on a target
decoder layer to capture residual-stream activations during generate().

Captured tensors live in self._residuals as a list (one entry per forward
pass during a single generate call):
  - residuals[0]: shape [batch, prompt_len, hidden] — prompt forward
  - residuals[1..N]: shape [batch, 1, hidden] — incremental decoding steps

F2.2 (introspection.py) consumes this raw capture and projects onto the
171 probes via ProbeLibrary. F2.1 itself does NOT load the probe library
nor compute projections — its only job is hook mechanics.

Layer index is wired from ProbeLibrary.layer at construction time so the
hook fires on the same layer where probes were extracted.

Hook overhead is documented in RESIDUUMREWORK.txt as <10% over baseline.
When introspection_enabled=False the hook is never registered (zero cost).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pathos.llm.transformers_provider import TransformersProvider

logger = logging.getLogger(__name__)


class IntrospectiveTransformersProvider(TransformersProvider):
    """Transformers provider with residual-stream capture at target_layer.

    Args:
        target_layer: Decoder layer index to hook (must match the layer used
            during probe extraction; typically ~2/3 depth of the model).
        introspection_enabled: If False, no hook is registered and the
            provider behaves identically to TransformersProvider.
        **kwargs: Forwarded to TransformersProvider.
    """

    def __init__(
        self,
        *,
        target_layer: int,
        introspection_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._target_layer = int(target_layer)
        self._introspection_enabled = bool(introspection_enabled)
        self._hook_handle: Any = None
        self._residuals: list[np.ndarray] = []

    @property
    def target_layer(self) -> int:
        return self._target_layer

    @property
    def introspection_enabled(self) -> bool:
        return self._introspection_enabled

    def set_introspection(self, enabled: bool) -> None:
        """Toggle introspection at runtime (registers/removes the hook)."""
        if enabled and not self._introspection_enabled:
            self._introspection_enabled = True
            if self._loaded:
                self._ensure_hook_registered()
        elif not enabled and self._introspection_enabled:
            self._introspection_enabled = False
            self._remove_hook()

    def _resolve_layer_module(self) -> Any:
        """Locate the decoder layer at target_layer.

        Walks common HF causal-LM structures: model.model.layers[i] for
        Qwen/Llama/Mistral; model.transformer.h[i] for GPT-2/Phi; falls
        back to model.layers[i].
        """
        m = self._model
        candidates = []
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            candidates.append(m.model.layers)
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            candidates.append(m.transformer.h)
        if hasattr(m, "layers"):
            candidates.append(m.layers)
        for layers in candidates:
            try:
                n = len(layers)
            except TypeError:
                continue
            if 0 <= self._target_layer < n:
                return layers[self._target_layer]
        raise RuntimeError(
            f"Could not locate decoder layer {self._target_layer} on model "
            f"{type(self._model).__name__}"
        )

    def _capture_hook(self, module: Any, inputs: Any, output: Any) -> None:
        """Forward hook: extract hidden_states tensor and store as ndarray.

        HF decoder layer outputs are typically a tuple (hidden_states, ...).
        We detach, move to CPU, cast to float32, convert to numpy and append.
        """
        try:
            hs = output[0] if isinstance(output, tuple) else output
            arr = hs.detach().to("cpu").to(dtype_float32()).numpy()
            self._residuals.append(arr)
        except Exception as exc:  # never let a hook take down generation
            logger.warning("Introspection hook failed: %s", exc)

    def _ensure_hook_registered(self) -> None:
        if self._hook_handle is not None:
            return
        layer = self._resolve_layer_module()
        self._hook_handle = layer.register_forward_hook(self._capture_hook)
        logger.info(
            "Introspection hook registered on layer %d (%s)",
            self._target_layer, type(layer).__name__,
        )

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            finally:
                self._hook_handle = None
                logger.info("Introspection hook removed")

    def _ensure_loaded(self) -> None:
        super()._ensure_loaded()
        if self._introspection_enabled and self._hook_handle is None:
            self._ensure_hook_registered()

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        # Clear any previous capture so callers see only this turn's residuals.
        self._residuals.clear()
        return await super().generate(*args, **kwargs)

    async def close(self) -> None:
        self._remove_hook()
        await super().close()

    # --- Public capture accessors (consumed by F2.2) ---

    def has_capture(self) -> bool:
        """True if at least one forward pass was captured since last reset."""
        return len(self._residuals) > 0

    def get_all_residuals(self) -> list[np.ndarray]:
        """Return raw per-forward arrays, oldest first. Caller should treat
        residuals[0] as the prompt forward and residuals[1:] as per-token
        decoding steps."""
        return list(self._residuals)

    def get_prompt_end_residual(self) -> np.ndarray | None:
        """Hidden state at the LAST token of the prompt forward.

        This corresponds to the paper's "Assistant colon" prepared-state
        capture point. Returns None if no capture has happened.
        Shape: [hidden]
        """
        if not self._residuals:
            return None
        first = self._residuals[0]
        # first has shape [batch, seq, hidden]; take batch 0, last token
        return first[0, -1, :].copy()

    def get_response_residuals(self) -> list[np.ndarray]:
        """Hidden states for each generated token, in emission order.

        Returns a list of [hidden] arrays. Empty if no token was generated
        or introspection was disabled.
        """
        if len(self._residuals) <= 1:
            return []
        out: list[np.ndarray] = []
        for arr in self._residuals[1:]:
            # Each step has shape [batch, 1, hidden] (or sometimes [batch, k, hidden] for cache).
            # We always pick the LAST column as the just-emitted token.
            out.append(arr[0, -1, :].copy())
        return out

    def get_response_mean_residual(self) -> np.ndarray | None:
        """Mean residual across all generated tokens.

        Corresponds to the paper's "response mean" / manifested-state capture.
        Returns None if no token was generated.
        Shape: [hidden]
        """
        per_token = self.get_response_residuals()
        if not per_token:
            return None
        return np.mean(np.stack(per_token, axis=0), axis=0)

    def reset_capture(self) -> None:
        """Clear stored residuals without disabling the hook."""
        self._residuals.clear()


def dtype_float32() -> Any:
    """Lazy import of torch.float32 (avoids importing torch at module load)."""
    import torch
    return torch.float32


__all__ = ["IntrospectiveTransformersProvider"]
