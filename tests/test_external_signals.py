"""Tests for External Signal processing and fusion."""

import pytest

from pathos.models.emotion_api import ExternalSignal
from pathos.models.external_signals import FusedSignalResult, ProcessedSignal
from pathos.engine.external_signals import (
    process_signal,
    fuse_signals,
    _SOURCE_WEIGHTS,
    _GLOBAL_SCALE,
)


class TestProcessSignal:
    """Tests for individual signal processing."""

    def test_basic_facial_au_signal(self) -> None:
        sig = ExternalSignal(
            source="facial_au",
            arousal_hint=0.8,
            confidence=0.9,
        )
        result = process_signal(sig)
        assert result.source == "facial_au"
        assert result.arousal_delta > 0
        assert result.weight == pytest.approx(0.6 * 0.9)

    def test_valence_from_facial_au(self) -> None:
        sig = ExternalSignal(
            source="facial_au",
            valence_hint=-0.6,
            confidence=0.7,
        )
        result = process_signal(sig)
        assert result.valence_delta < 0

    def test_zero_confidence_gives_zero_weight(self) -> None:
        sig = ExternalSignal(
            source="facial_au",
            arousal_hint=0.9,
            confidence=0.0,
        )
        result = process_signal(sig)
        assert result.weight == 0.0
        assert result.arousal_delta == 0.0

    def test_none_hints_give_zero_delta(self) -> None:
        sig = ExternalSignal(source="keyboard_dynamics", confidence=1.0)
        result = process_signal(sig)
        assert result.valence_delta == 0.0
        assert result.arousal_delta == 0.0
        assert result.dominance_delta == 0.0

    def test_unknown_source_uses_default_weight(self) -> None:
        sig = ExternalSignal(
            source="unknown_sensor",
            valence_hint=0.5,
            confidence=1.0,
        )
        result = process_signal(sig)
        assert result.weight == pytest.approx(0.5)  # default

    def test_dominance_hint_normalized(self) -> None:
        """Dominance hint 0-1 should be normalized to -1..1 range internally."""
        sig = ExternalSignal(
            source="keyboard_dynamics",
            dominance_hint=1.0,  # Max dominance
            confidence=1.0,
        )
        result = process_signal(sig)
        assert result.dominance_delta > 0

        sig_low = ExternalSignal(
            source="keyboard_dynamics",
            dominance_hint=0.0,  # Min dominance
            confidence=1.0,
        )
        result_low = process_signal(sig_low)
        assert result_low.dominance_delta < 0


class TestFuseSignals:
    """Tests for multi-signal fusion."""

    def test_empty_signals_returns_default(self) -> None:
        result = fuse_signals([])
        assert result.signal_count == 0
        assert result.valence_modulation == 0.0
        assert result.arousal_modulation == 0.0
        assert not result.has_effect

    def test_single_signal(self) -> None:
        signals = [
            ExternalSignal(source="facial_au", arousal_hint=0.8, confidence=0.9),
        ]
        result = fuse_signals(signals)
        assert result.signal_count == 1
        assert result.arousal_modulation > 0
        assert len(result.contributions) == 1

    def test_multiple_signals_fuse(self) -> None:
        signals = [
            ExternalSignal(source="keyboard_dynamics", arousal_hint=0.9, confidence=0.8),
            ExternalSignal(source="facial_au", valence_hint=-0.5, confidence=0.7),
            ExternalSignal(source="weather", valence_hint=-0.2, confidence=0.6),
        ]
        result = fuse_signals(signals)
        assert result.signal_count == 3
        assert result.arousal_modulation > 0  # from keyboard
        assert result.valence_modulation < 0  # from facial + weather
        assert len(result.contributions) == 3

    def test_modulation_bounded(self) -> None:
        """Even with extreme signals, modulation stays in [-1, 1]."""
        signals = [
            ExternalSignal(source="facial_au", valence_hint=1.0, arousal_hint=1.0, confidence=1.0),
            ExternalSignal(source="keyboard_dynamics", valence_hint=1.0, arousal_hint=1.0, confidence=1.0),
            ExternalSignal(source="time_of_day", valence_hint=1.0, arousal_hint=1.0, confidence=1.0),
        ]
        result = fuse_signals(signals)
        assert -1.0 <= result.valence_modulation <= 1.0
        assert -1.0 <= result.arousal_modulation <= 1.0
        assert -1.0 <= result.dominance_modulation <= 1.0

    def test_global_scale_limits_magnitude(self) -> None:
        """Modulation should be scaled by _GLOBAL_SCALE (0.3) to stay subtle."""
        signals = [
            ExternalSignal(source="facial_au", valence_hint=1.0, confidence=1.0),
        ]
        result = fuse_signals(signals)
        # valence_hint=1.0, weight=0.6, normalized=1.0, scaled=0.3
        assert abs(result.valence_modulation) <= _GLOBAL_SCALE + 0.01

    def test_higher_confidence_has_more_influence(self) -> None:
        """A high-confidence signal should dominate a low-confidence one."""
        signals_high = [
            ExternalSignal(source="keyboard_dynamics", valence_hint=0.8, confidence=1.0),
            ExternalSignal(source="keyboard_dynamics", valence_hint=-0.3, confidence=0.1),
        ]
        result = fuse_signals(signals_high)
        assert result.valence_modulation > 0  # Positive dominates

    def test_facial_au_weighs_more_than_weather(self) -> None:
        """Facial AU (0.6 base) should outweigh weather (0.2 base)."""
        assert _SOURCE_WEIGHTS["facial_au"] > _SOURCE_WEIGHTS["weather"]

        signals = [
            ExternalSignal(source="facial_au", valence_hint=0.5, confidence=0.5),
            ExternalSignal(source="weather", valence_hint=-0.5, confidence=0.5),
        ]
        result = fuse_signals(signals)
        # Facial AU positive should outweigh weather negative
        assert result.valence_modulation > 0

    def test_has_effect_true_when_modulation(self) -> None:
        signals = [
            ExternalSignal(source="facial_au", arousal_hint=0.5, confidence=0.5),
        ]
        result = fuse_signals(signals)
        assert result.has_effect

    def test_all_zero_confidence_no_effect(self) -> None:
        signals = [
            ExternalSignal(source="facial_au", arousal_hint=0.9, confidence=0.0),
            ExternalSignal(source="keyboard_dynamics", valence_hint=0.8, confidence=0.0),
        ]
        result = fuse_signals(signals)
        assert not result.has_effect


class TestEmotionAPIModels:
    """Tests for the Emotion API request/response Pydantic models."""

    def test_external_signal_validation(self) -> None:
        sig = ExternalSignal(source="facial_au", arousal_hint=0.8, confidence=0.9)
        assert sig.source == "facial_au"
        assert sig.valence_hint is None

    def test_external_signal_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            ExternalSignal(source="test", valence_hint=1.5)
        with pytest.raises(Exception):
            ExternalSignal(source="test", confidence=-0.1)

    def test_processed_signal_model(self) -> None:
        ps = ProcessedSignal(source="facial", valence_delta=0.1, weight=0.5)
        assert ps.source == "facial"

    def test_fused_result_default(self) -> None:
        fr = FusedSignalResult()
        assert fr.signal_count == 0
        assert not fr.has_effect

    def test_fused_result_has_effect(self) -> None:
        fr = FusedSignalResult(valence_modulation=0.01)
        assert fr.has_effect
