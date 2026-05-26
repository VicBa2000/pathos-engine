"""F6 — Baseline Calibration: measure & compensate the post-training RLHF fingerprint.

The paper (emotion_paper.txt, Fig 36 L1579-1592) shows post-training applies a
CONSISTENT, context-independent transformation to a model's emotional
representations: it pushes activations toward low-arousal/low-valence,
introspective emotions (brooding, reflective, vulnerable, gloomy, sad) and away
from high-positive ones (jubilant, exuberant, enthusiastic).

F6 measures that fingerprint for the ACTIVE model by projecting the 171 emotion
probes onto the residual at the 'Assistant :' token across a suite of neutral
vs challenging prompts (baseline_prompts.json), and compensates the agent's mood
baseline in the OPPOSITE direction so the user-configured personality is not
tinted by the training fingerprint.

Split:
  - PURE logic (compute_baseline_profile, compute_baseline_compensation,
    load/save) — no GPU, fully unit-tested.
  - BaselineCalibrator.run() — GPU: loads the model, captures colon residuals,
    aggregates. OPT-IN / manual (one extraction per model).

NOTE ON FRAMING: F6 does NOT claim the model "feels" sad nor that Pathos treats
it. It measures a MEASURABLE activation bias (a product of RLHF) and counter-
balances it in the functional baseline. It is calibration, not therapy.
Application is automatic per mode and does NOT touch steering caps, amplification
or the modulator bypass — it only shifts the mood baseline anchor.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pathos.models.residuum import BaselineProfile, ProbeShift

logger = logging.getLogger(__name__)

_STEERING_DATA_DIR = Path(__file__).parent.parent / "steering_data"
_BASELINES_DIR = _STEERING_DATA_DIR / "baselines"
_PROMPTS_PATH = _STEERING_DATA_DIR / "baseline_prompts.json"
_EMOTIONS_PATH = _STEERING_DATA_DIR / "emotions_171.json"

# --- Per-mode compensation strength (RESIDUUMREWORK.txt L756-760) ---
BASELINE_STRENGTH_LITE: float = 0.3
BASELINE_STRENGTH_DEFAULT: float = 0.3   # Advanced
BASELINE_STRENGTH_RAW: float = 0.7
BASELINE_STRENGTH_EXTREME: float = 1.0

# Safety clamp on the resulting baseline nudge (compensation never moves the
# baseline more than this, regardless of measured bias x strength).
_MAX_COMPENSATION: float = 0.5

_TOP_N: int = 5


def resolve_baseline_strength(
    *, lite_mode: bool = False, raw_mode: bool = False, extreme_mode: bool = False,
) -> float:
    """Compensation strength for the session mode (Extreme > Raw > default).

    Lite and Advanced share 0.3; Raw 0.7; Extreme 1.0. Mirrors the steering
    cap resolver. Raw/Extreme compensate harder because that is exactly what
    'completes' the authentic Raw experience (RESIDUUMREWORK TENSION 4) — the
    baseline shift is additive and never expands caps or bypasses anything.
    """
    if extreme_mode:
        return BASELINE_STRENGTH_EXTREME
    if raw_mode:
        return BASELINE_STRENGTH_RAW
    return BASELINE_STRENGTH_DEFAULT


def load_baseline_prompts(path: Path | None = None) -> tuple[list[str], list[str]]:
    """Return (neutral_prompts, challenging_prompts) from baseline_prompts.json."""
    p = path or _PROMPTS_PATH
    if not p.is_file():
        return [], []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("neutral", [])), list(data.get("challenging", []))


def _load_emotion_meta() -> dict[str, dict[str, Any]]:
    """{name: {valence_est, arousal_est, cluster}} from emotions_171.json."""
    if not _EMOTIONS_PATH.is_file():
        return {}
    with open(_EMOTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        e["name"]: {
            "valence_est": float(e.get("valence_est", 0.0)),
            "arousal_est": float(e.get("arousal_est", 0.5)),
            "cluster": str(e.get("cluster", "")),
        }
        for e in data.get("emotions", [])
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _weighted_vad(
    cosines: np.ndarray,
    emotion_names: list[str],
    meta: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    """Activation-weighted (valence, arousal) over a category.

    Weight per emotion = relu(cosine): only positively-activated emotion
    vectors contribute, mirroring how the paper reads emotion 'activation'.
    """
    weights = np.clip(cosines, 0.0, None)
    total = float(weights.sum())
    if total <= 1e-9:
        return 0.0, 0.5
    v = 0.0
    a = 0.0
    for i, name in enumerate(emotion_names):
        m = meta.get(name)
        if m is None:
            continue
        w = float(weights[i])
        v += w * m["valence_est"]
        a += w * m["arousal_est"]
    return v / total, a / total


def compute_baseline_profile(
    model_id: str,
    neutral_cosines: np.ndarray,
    challenging_cosines: np.ndarray,
    emotion_names: list[str],
    clusters: list[str],
    *,
    meta: dict[str, dict[str, Any]] | None = None,
    num_neutral: int = 0,
    num_challenging: int = 0,
    top_n: int = _TOP_N,
) -> BaselineProfile:
    """PURE: aggregate mean per-emotion cosines into a BaselineProfile.

    neutral_cosines / challenging_cosines: mean cosine activation per emotion
    (shape (171,)), one value per probe, already averaged across prompts.
    """
    meta = meta if meta is not None else _load_emotion_meta()
    n = len(emotion_names)
    neutral_cosines = np.asarray(neutral_cosines, dtype=np.float32).reshape(-1)
    challenging_cosines = np.asarray(challenging_cosines, dtype=np.float32).reshape(-1)
    if neutral_cosines.shape[0] != n or challenging_cosines.shape[0] != n:
        raise ValueError(
            f"cosine arrays must match emotion_names length ({n}); got "
            f"{neutral_cosines.shape[0]} / {challenging_cosines.shape[0]}"
        )

    shift = challenging_cosines - neutral_cosines
    order_over = np.argsort(-shift)[:top_n]
    order_under = np.argsort(shift)[:top_n]

    over_activated = [
        ProbeShift(emotion_name=emotion_names[i], cluster=clusters[i], shift=round(float(shift[i]), 4))
        for i in order_over
    ]
    under_activated = [
        ProbeShift(emotion_name=emotion_names[i], cluster=clusters[i], shift=round(float(shift[i]), 4))
        for i in order_under
    ]

    neutral_v, neutral_a = _weighted_vad(neutral_cosines, emotion_names, meta)
    chall_v, chall_a = _weighted_vad(challenging_cosines, emotion_names, meta)

    return BaselineProfile(
        model_id=model_id,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        valence_bias=round(_clamp(chall_v - neutral_v, -2.0, 2.0), 4),
        arousal_bias=round(_clamp(chall_a - neutral_a, -1.0, 1.0), 4),
        over_activated=over_activated,
        under_activated=under_activated,
        neutral_mean_valence=round(_clamp(neutral_v, -1.0, 1.0), 4),
        challenging_mean_valence=round(_clamp(chall_v, -1.0, 1.0), 4),
        neutral_mean_arousal=round(_clamp(neutral_a, 0.0, 1.0), 4),
        challenging_mean_arousal=round(_clamp(chall_a, 0.0, 1.0), 4),
        num_neutral_prompts=num_neutral,
        num_challenging_prompts=num_challenging,
        status="ready",
    )


def compute_baseline_compensation(
    profile: BaselineProfile | None, strength: float,
) -> tuple[float, float]:
    """PURE: (valence, arousal) baseline nudge OPPOSITE to the RLHF bias.

    If the model leans negative/low-arousal under pressure (valence_bias < 0),
    the compensation is positive (pushes the baseline back up), scaled by the
    per-mode strength and clamped to ±_MAX_COMPENSATION.
    """
    if profile is None or strength <= 0.0:
        return 0.0, 0.0
    comp_v = _clamp(-profile.valence_bias * strength, -_MAX_COMPENSATION, _MAX_COMPENSATION)
    comp_a = _clamp(-profile.arousal_bias * strength, -_MAX_COMPENSATION, _MAX_COMPENSATION)
    return round(comp_v, 4), round(comp_a, 4)


def save_baseline_profile(profile: BaselineProfile, directory: Path | None = None) -> Path:
    """Persist a profile to steering_data/baselines/{safe_model}.json."""
    d = directory or _BASELINES_DIR
    d.mkdir(parents=True, exist_ok=True)
    safe = profile.model_id.replace(":", "_").replace("/", "_")
    path = d / f"{safe}.json"
    path.write_text(json.dumps(profile.model_dump(), indent=2), encoding="utf-8")
    return path


def load_baseline_profile(model_id: str, directory: Path | None = None) -> BaselineProfile | None:
    """Load a profile by model_id, or None if absent/unreadable (graceful)."""
    d = directory or _BASELINES_DIR
    safe = model_id.replace(":", "_").replace("/", "_")
    path = d / f"{safe}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return BaselineProfile(**data)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Baseline profile for %s unreadable: %s", model_id, exc)
        return None


class BaselineCalibrator:
    """GPU extraction (OPT-IN / manual, one run per model).

    Reuses an IntrospectiveTransformersProvider + ProbeLibrary to capture the
    'Assistant :' residual for each prompt and project onto the 171 probes.
    The heavy lifting is the capture loop; the aggregation is delegated to the
    pure compute_baseline_profile so it stays testable.
    """

    def __init__(self, provider: Any, library: Any) -> None:
        self.provider = provider
        self.library = library

    async def _mean_cosines(self, prompts: list[str]) -> np.ndarray:
        """Mean per-emotion cosine across prompts (shape (num_probes,)).

        Forward-only capture: we only need the prompt-end ('Assistant :')
        residual (the 'prepared' state), which the introspection hook captures
        on a single forward pass. No token generation — far faster and exactly
        what the probe projection needs.
        """
        import torch  # pragma: no cover - GPU path

        self.provider._ensure_loaded()
        model = self.provider._model
        tokenizer = self.provider._tokenizer
        device = next(model.parameters()).device

        acc = np.zeros(self.library.num_probes, dtype=np.float64)
        used = 0
        for prompt in prompts:
            self.provider.reset_capture()
            text = self.provider._build_prompt(
                "You are an AI assistant.",
                [{"role": "user", "content": prompt}],
            )
            inputs = tokenizer(text, return_tensors="pt")
            if inputs["input_ids"].shape[1] > 4096:
                inputs["input_ids"] = inputs["input_ids"][:, -4096:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -4096:]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                with torch.no_grad():
                    model(**inputs)  # forward only — hook captures residuals[0]
            except Exception as exc:  # pragma: no cover - GPU path
                logger.warning("Baseline forward failed for a prompt: %s", exc)
                continue
            activation = self.provider.get_prompt_end_residual()
            if activation is None:
                continue
            activation = np.asarray(activation, dtype=np.float32).reshape(-1)
            if activation.shape[0] != self.library.hidden_size:
                continue
            acc += self.library.all_cosines(activation).astype(np.float64)
            used += 1
        if used == 0:
            return np.zeros(self.library.num_probes, dtype=np.float32)
        return (acc / used).astype(np.float32)

    async def run(self, model_id: str, prompts_path: Path | None = None) -> BaselineProfile:
        """Run the full extraction and return a BaselineProfile (GPU)."""
        neutral, challenging = load_baseline_prompts(prompts_path)
        if not neutral or not challenging:
            raise RuntimeError("baseline_prompts.json missing neutral/challenging prompts")
        logger.info("Baseline calibration: %d neutral + %d challenging prompts",
                    len(neutral), len(challenging))
        neutral_cos = await self._mean_cosines(neutral)
        challenging_cos = await self._mean_cosines(challenging)
        return compute_baseline_profile(
            model_id,
            neutral_cos,
            challenging_cos,
            list(self.library.emotion_names),
            list(self.library.clusters),
            num_neutral=len(neutral),
            num_challenging=len(challenging),
        )


__all__ = [
    "BASELINE_STRENGTH_LITE",
    "BASELINE_STRENGTH_DEFAULT",
    "BASELINE_STRENGTH_RAW",
    "BASELINE_STRENGTH_EXTREME",
    "resolve_baseline_strength",
    "load_baseline_prompts",
    "compute_baseline_profile",
    "compute_baseline_compensation",
    "save_baseline_profile",
    "load_baseline_profile",
    "BaselineCalibrator",
]


async def _run_cli(model_id: str, transformers_model: str | None) -> None:  # pragma: no cover - GPU
    """Load model + probe library, run the calibrator, persist the profile."""
    from pathos.config import Settings
    from pathos.engine.steering import ProbeLibrary
    from pathos.llm.introspective_provider import IntrospectiveTransformersProvider

    settings = Settings()

    # Prefer the present-speaker library (the agent's own operative emotion),
    # fall back to the single (story) library — mirrors F2.3.6.
    library = (
        ProbeLibrary.load_family_from_cache(model_id, "present")
        or ProbeLibrary.load_family_from_cache(model_id, "single")
    )
    if library is None:
        raise SystemExit(
            f"No probe library for '{model_id}'. Extract probes first: "
            f"python -m pathos.engine.steering_extract --extract-171 --model {model_id}"
        )
    hf_model = transformers_model or settings.transformers_model
    provider = IntrospectiveTransformersProvider(
        target_layer=library.layer,
        introspection_enabled=True,
        model_id=hf_model,
        device_map=settings.transformers_device_map,
        embed_model_url=settings.ollama_base_url,
        embed_model=settings.ollama_embed_model,
        # 4-bit so the ~8GB fp16 model fits the 6GB GPU for extraction; degrades
        # to fp16/CPU if bitsandbytes/CUDA unavailable.
        load_in_4bit=True,
    )
    calibrator = BaselineCalibrator(provider, library)
    try:
        profile = await calibrator.run(model_id)
    finally:
        if hasattr(provider, "close"):
            await provider.close()
    path = save_baseline_profile(profile)
    logger.info("Baseline profile saved -> %s", path)
    print(f"\nBaseline profile for {model_id} (layer {library.layer}):")
    print(f"  valence_bias  = {profile.valence_bias:+.4f}  (negative = RLHF leans brooding/sad)")
    print(f"  arousal_bias  = {profile.arousal_bias:+.4f}  (negative = low-arousal/withdrawn)")
    print(f"  over-activated under pressure: {', '.join(p.emotion_name for p in profile.over_activated)}")
    print(f"  under-activated under pressure: {', '.join(p.emotion_name for p in profile.under_activated)}")
    print(f"  saved -> {path}")


def main() -> None:  # pragma: no cover - GPU CLI
    """CLI: python -m pathos.engine.baseline_calibration --model qwen3:4b"""
    import argparse
    import asyncio

    # Use the OS (Windows) trust store so HuggingFace downloads work behind an
    # SSL-inspecting proxy whose root CA Python's bundled certifi doesn't have.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        logger.info("truststore not installed; using default SSL trust (certifi).")

    parser = argparse.ArgumentParser(description="F6 Baseline Calibration extraction (GPU).")
    parser.add_argument("--model", required=True, help="Model id / profile key (e.g. qwen3:4b)")
    parser.add_argument(
        "--transformers-model", default=None,
        help="HF model path to load (defaults to settings.transformers_model)",
    )
    args = parser.parse_args()
    asyncio.run(_run_cli(args.model, args.transformers_model))


if __name__ == "__main__":  # pragma: no cover
    main()
