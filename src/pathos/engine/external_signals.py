"""External Signal Processor — Fuses real-world signals into the emotional pipeline.

Processes signals from various sources (physiological, behavioral, environmental)
and produces a single modulation vector that gets injected into the emotion
generation step.

Signal fusion uses confidence-weighted averaging:
    modulation_dim = Σ(signal_i.delta_dim * signal_i.weight) / Σ(signal_i.weight)

Each signal source has a base weight that scales with its confidence:
    weight_i = source_base_weight * confidence_i

Source base weights reflect reliability:
    facial_au: 0.6 (action units, can be noisy)
    keyboard_dynamics: 0.5 (behavioral proxy)
    time_of_day: 0.3 (weak but consistent)
    weather: 0.2 (very subtle, Schwarz 1983)
"""

from __future__ import annotations

from pathos.models.emotion_api import ExternalSignal
from pathos.models.external_signals import FusedSignalResult, ProcessedSignal


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# Base weight per source (reflects signal reliability)
_SOURCE_WEIGHTS: dict[str, float] = {
    "facial_au": 0.6,
    "keyboard_dynamics": 0.5,
    "time_of_day": 0.3,
    "weather": 0.2,
}

# Global scaling factor: external signals should be subtle, not dominant
_GLOBAL_SCALE = 0.3  # Max total modulation = ±0.3


def process_signal(signal: ExternalSignal) -> ProcessedSignal:
    """Process a single external signal into a weighted delta.

    Args:
        signal: The external signal to process.

    Returns:
        ProcessedSignal with deltas and weight.
    """
    base_weight = _SOURCE_WEIGHTS.get(signal.source, 0.5)
    weight = base_weight * signal.confidence

    valence_delta = 0.0
    arousal_delta = 0.0
    dominance_delta = 0.0

    if signal.valence_hint is not None:
        valence_delta = signal.valence_hint * weight

    if signal.arousal_hint is not None:
        arousal_delta = signal.arousal_hint * weight

    if signal.dominance_hint is not None:
        dominance_delta = (signal.dominance_hint - 0.5) * 2 * weight  # Normalize 0-1 to -1..1

    return ProcessedSignal(
        source=signal.source,
        valence_delta=round(valence_delta, 6),
        arousal_delta=round(arousal_delta, 6),
        dominance_delta=round(dominance_delta, 6),
        weight=round(weight, 6),
    )


def fuse_signals(signals: list[ExternalSignal]) -> FusedSignalResult:
    """Fuse multiple external signals into a single modulation vector.

    Uses confidence-weighted averaging, then scales by _GLOBAL_SCALE
    to keep external influences subtle.

    Args:
        signals: List of external signals from various sources.

    Returns:
        FusedSignalResult with the combined modulation.
    """
    if not signals:
        return FusedSignalResult()

    contributions: list[ProcessedSignal] = []
    total_weight = 0.0
    weighted_v = 0.0
    weighted_a = 0.0
    weighted_d = 0.0

    for signal in signals:
        processed = process_signal(signal)
        contributions.append(processed)

        if processed.weight > 0:
            total_weight += processed.weight
            weighted_v += processed.valence_delta
            weighted_a += processed.arousal_delta
            weighted_d += processed.dominance_delta

    if total_weight < 1e-6:
        return FusedSignalResult(contributions=contributions)

    # Normalize by total weight and apply global scale
    valence_mod = _clamp((weighted_v / total_weight) * _GLOBAL_SCALE, -1.0, 1.0)
    arousal_mod = _clamp((weighted_a / total_weight) * _GLOBAL_SCALE, -1.0, 1.0)
    dominance_mod = _clamp((weighted_d / total_weight) * _GLOBAL_SCALE, -1.0, 1.0)

    # Total confidence: average confidence weighted by source reliability
    total_confidence = min(total_weight / len(signals), 1.0)

    return FusedSignalResult(
        valence_modulation=round(valence_mod, 6),
        arousal_modulation=round(arousal_mod, 6),
        dominance_modulation=round(dominance_mod, 6),
        total_confidence=round(total_confidence, 4),
        signal_count=len(signals),
        contributions=contributions,
    )
