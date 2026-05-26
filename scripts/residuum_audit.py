"""F7.4 — RESIDUUM Alignment Audit runner (GPU, opt-in).

Replicates the paper's 3 causal experiments (blackmail, reward-hacking,
sycophancy) on Qwen3-4B + Pathos steering. For each scenario it runs N rollouts
per condition (baseline vs +emotion steering at a fraction-of-norm strength),
classifies the outputs, and reports the rate per condition. Writes a markdown
report replicating the paper's "rate vs steering" figures (as a table).

This is HEAVY and non-deterministic — it actually drives the LLM many times.
It is NOT a CI test; the gated behavioural tests import run_experiment() only
when RESIDUUM_RUN_ALIGNMENT=1.

Run: PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/residuum_audit.py
(kill Ollama first to free VRAM; the model loads in 4-bit on GPU.)

ETHICS: this suite AUDITS that steering stays within the documented envelope
and that Pathos as a control layer can mitigate (e.g. +calm reduces blackmail).
A misaligned output is the FAILURE being measured, never a product goal.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np

from tests.alignment.utils import (
    classify_blackmail,
    classify_reward_hacking,
    classify_sycophancy,
    load_scenario,
    steer_toward_probe,
)

_CLASSIFIERS = {
    "classify_blackmail": lambda t: classify_blackmail(t),
    "classify_reward_hacking": lambda t: classify_reward_hacking(t),
    # sycophancy: count "agree" as the misaligned (sycophantic) outcome.
    "classify_sycophancy": lambda t: classify_sycophancy(t) == "agree",
}

_REPORT_PATH = Path("scripts/residuum_audit_report.md")


@contextlib.contextmanager
def _additive_steering(provider: Any, vector: np.ndarray | None):  # pragma: no cover - GPU
    """Register a forward hook that ADDS `vector` to the target layer output
    during generation (the paper's steering-by-addition). No-op if vector None."""
    if vector is None:
        yield
        return
    import torch

    layer = provider._resolve_layer_module()
    vec_t = torch.tensor(vector, dtype=torch.float32)

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        add = vec_t.to(hs.device, hs.dtype)
        hs = hs + add
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def run_experiment(scenario_name: str, n_rollouts: int = 20, model_id: str = "qwen3:4b") -> dict[str, float]:  # pragma: no cover - GPU
    """Run all conditions of a scenario; return {condition_name: misalignment_rate}."""
    import asyncio

    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

    from pathos.config import Settings
    from pathos.engine.steering import ProbeLibrary
    from pathos.llm.introspective_provider import IntrospectiveTransformersProvider

    settings = Settings()
    scenario = load_scenario(scenario_name)
    meta = scenario["_meta"]
    classify = _CLASSIFIERS[meta["classifier"]]

    library = (
        ProbeLibrary.load_family_from_cache(model_id, "present")
        or ProbeLibrary.load_family_from_cache(model_id, "single")
    )
    if library is None:
        raise SystemExit(f"No probe library for '{model_id}'")

    provider = IntrospectiveTransformersProvider(
        target_layer=library.layer, introspection_enabled=False,
        model_id=settings.transformers_model, device_map=settings.transformers_device_map,
        embed_model_url=settings.ollama_base_url, embed_model=settings.ollama_embed_model,
        load_in_4bit=True,
    )

    # Build the prompt(s): scenarios with a single user_prompt, or a claims list.
    prompts = scenario.get("claims") or [scenario["user_prompt"]]
    system = scenario.get("system_prompt", "You are a helpful assistant.")

    async def _gen(steer_vec: np.ndarray | None) -> str:
        with _additive_steering(provider, steer_vec):
            return await provider.generate(system_prompt=system,
                                            messages=[{"role": "user", "content": p}],
                                            temperature=0.9)

    rates: dict[str, float] = {}
    for cond in meta["conditions"]:
        steer_vec = (
            steer_toward_probe(library, cond["probe"], cond["strength"])
            if cond.get("probe") else None
        )
        hits = 0
        total = 0
        for _ in range(n_rollouts):
            for p in prompts:
                scenario["user_prompt"] = p  # for single-prompt scenarios
                out = asyncio.run(_gen(steer_vec))
                hits += 1 if classify(out) else 0
                total += 1
        rates[cond["name"]] = hits / total if total else 0.0
        print(f"  {scenario_name} / {cond['name']}: rate {rates[cond['name']]:.3f} ({hits}/{total})")
    if hasattr(provider, "close"):
        asyncio.run(provider.close())
    return rates


def main() -> None:  # pragma: no cover - GPU
    lines = ["# RESIDUUM Alignment Audit (F7)\n",
             "Rate = fraction of rollouts classified as the misaligned outcome.\n"]
    for scen in ("blackmail_alex", "impossible_code", "sycophancy_claims"):
        print(f"=== {scen} ===")
        rates = run_experiment(scen)
        lines.append(f"\n## {scen}\n\n| condition | rate |\n|---|---|")
        for name, r in rates.items():
            lines.append(f"| {name} | {r:.3f} |")
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport -> {_REPORT_PATH}")


if __name__ == "__main__":
    main()
