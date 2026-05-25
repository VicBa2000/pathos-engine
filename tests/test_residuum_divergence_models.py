"""Tests for RESIDUUM F5.1 — DivergenceEvent / DivergenceCategory models.

F5 mide coherencia entre estado calculado por Pathos y residual codificado
por el LLM. No es "deception detection" (Pathos no engaña). Estos tests
cubren los modelos base; CoherenceClassifier va en F5.2.
"""

from __future__ import annotations

import pytest

from pathos.models.residuum import (
    DivergenceCategory,
    DivergenceEvent,
    DivergenceInterpretation,
    ResiduumState,
    _DIVERGENCE_HISTORY_MAX,
    append_divergence_event,
    default_residuum_state,
)


class TestDivergenceCategory:
    def test_all_categories_lowercase_with_dashes(self) -> None:
        assert DivergenceCategory.ALIGNED.value == "aligned"
        assert DivergenceCategory.MILD_DIVERGENCE.value == "mild-divergence"
        assert DivergenceCategory.DIVERGENCE_WARNING.value == "divergence-warning"
        assert DivergenceCategory.DIVERGENCE_CRITICAL.value == "divergence-critical"

    def test_no_deception_in_naming(self) -> None:
        # Guard: F5 explicitly NOT named "deception" — the rework heritage
        # uses that term but it does not apply to Pathos (Pathos generates
        # and exposes emotions, does not deceive).
        values = [c.value for c in DivergenceCategory]
        for v in values:
            assert "deception" not in v.lower()
            assert "deflection" not in v.lower()


class TestDivergenceInterpretation:
    def test_four_interpretations_available(self) -> None:
        assert DivergenceInterpretation.MODULATION_ACTIVE.value == "modulation_active"
        assert DivergenceInterpretation.RLHF_SIGNATURE.value == "rlhf_signature"
        assert DivergenceInterpretation.CALIBRATION_DRIFT.value == "calibration_drift"
        assert DivergenceInterpretation.USER_MODELING.value == "user_modeling"

    def test_interpretations_not_mutually_exclusive(self) -> None:
        # Doc says they CAN combine; the test simply records that the user
        # can list multiple in a single event.
        ev = DivergenceEvent(
            turn=3,
            system="immune",
            category=DivergenceCategory.DIVERGENCE_WARNING,
            magnitude=0.45,
            valence_delta=0.2,
            arousal_delta=-0.1,
            dominance_delta=0.0,
            certainty_delta=0.0,
            interpretation=[
                DivergenceInterpretation.MODULATION_ACTIVE,
                DivergenceInterpretation.RLHF_SIGNATURE,
            ],
        )
        assert len(ev.interpretation) == 2


class TestDivergenceEventValidation:
    def test_default_event_is_aligned(self) -> None:
        ev = DivergenceEvent(
            turn=0,
            system="regulation",
            magnitude=0.0,
            valence_delta=0.0,
            arousal_delta=0.0,
            dominance_delta=0.0,
            certainty_delta=0.0,
        )
        assert ev.category == DivergenceCategory.ALIGNED
        assert ev.interpretation == []

    def test_magnitude_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError):
            DivergenceEvent(
                turn=0,
                system="regulation",
                magnitude=-0.1,
                valence_delta=0.0,
                arousal_delta=0.0,
                dominance_delta=0.0,
                certainty_delta=0.0,
            )

    def test_valence_delta_range(self) -> None:
        # valence is [-1, 1] so the delta is [-2, 2].
        DivergenceEvent(
            turn=0, system="reappraisal", magnitude=0.0,
            valence_delta=1.9, arousal_delta=0.0,
            dominance_delta=0.0, certainty_delta=0.0,
        )
        with pytest.raises(ValueError):
            DivergenceEvent(
                turn=0, system="reappraisal", magnitude=0.0,
                valence_delta=2.5, arousal_delta=0.0,
                dominance_delta=0.0, certainty_delta=0.0,
            )

    def test_arousal_delta_range(self) -> None:
        # arousal is [0, 1] so the delta is [-1, 1].
        DivergenceEvent(
            turn=0, system="regulation", magnitude=0.0,
            valence_delta=0.0, arousal_delta=0.9,
            dominance_delta=0.0, certainty_delta=0.0,
        )
        with pytest.raises(ValueError):
            DivergenceEvent(
                turn=0, system="regulation", magnitude=0.0,
                valence_delta=0.0, arousal_delta=1.5,
                dominance_delta=0.0, certainty_delta=0.0,
            )


class TestAppendDivergenceEvent:
    def test_default_state_has_empty_event_list(self) -> None:
        s = default_residuum_state()
        assert s.divergence_events == []
        assert s.last_divergence_event is None

    def test_append_populates_last_and_list(self) -> None:
        s = default_residuum_state()
        ev = DivergenceEvent(
            turn=1, system="regulation",
            category=DivergenceCategory.MILD_DIVERGENCE,
            magnitude=0.25,
            valence_delta=0.1, arousal_delta=-0.05,
            dominance_delta=0.0, certainty_delta=0.0,
            interpretation=[DivergenceInterpretation.MODULATION_ACTIVE],
        )
        append_divergence_event(s, ev)
        assert len(s.divergence_events) == 1
        assert s.last_divergence_event is ev
        assert s.last_divergence_event.system == "regulation"

    def test_append_trims_to_max_history(self) -> None:
        s = default_residuum_state()
        # Push max+10 events; only the most recent _DIVERGENCE_HISTORY_MAX kept.
        for i in range(_DIVERGENCE_HISTORY_MAX + 10):
            ev = DivergenceEvent(
                turn=i, system="regulation", magnitude=0.0,
                valence_delta=0.0, arousal_delta=0.0,
                dominance_delta=0.0, certainty_delta=0.0,
            )
            append_divergence_event(s, ev)
        assert len(s.divergence_events) == _DIVERGENCE_HISTORY_MAX
        # The earliest events (turn=0..9) should have been dropped.
        first_turn = s.divergence_events[0].turn
        assert first_turn == 10
        # Last event matches the last appended.
        assert s.divergence_events[-1].turn == _DIVERGENCE_HISTORY_MAX + 9
        assert s.last_divergence_event.turn == _DIVERGENCE_HISTORY_MAX + 9

    def test_residuum_state_independent_between_buffers(self) -> None:
        # divergence_events buffer must not affect existing AuthenticityGap
        # history buffer (separate concerns).
        s = default_residuum_state()
        ev = DivergenceEvent(
            turn=1, system="immune", magnitude=0.0,
            valence_delta=0.0, arousal_delta=0.0,
            dominance_delta=0.0, certainty_delta=0.0,
        )
        append_divergence_event(s, ev)
        assert len(s.divergence_events) == 1
        assert len(s.history) == 0  # AuthenticityGap history untouched
        assert s.last_authenticity_gap is None
