"""Tests for Self-Initiated Inquiry — spontaneous emotional reflection."""

import pytest

from pathos.engine.meta import MetaEmotion
from pathos.engine.regulation import RegulationResult
from pathos.engine.self_inquiry import (
    InquiryTrigger,
    SelfInquiry,
    SuggestedBehavior,
    check_self_inquiry,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _make_state(
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.3,
    intensity: float = 0.3,
    duration: int = 1,
) -> EmotionalState:
    """Helper to create EmotionalState for testing."""
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        certainty=0.5,
        primary_emotion=emotion,
        intensity=intensity,
        duration=duration,
        triggered_by="test",
        timestamp="2026-04-09T00:00:00Z",
        emotional_stack={emotion.value: intensity},
        body_state={"energy": 0.5, "tension": 0.3, "openness": 0.5, "warmth": 0.5},
        mood={
            "baseline_valence": 0.1, "baseline_arousal": 0.3,
            "stability": 0.7, "trend": "stable", "label": "neutral",
            "extreme_event_count": 0, "turns_since_extreme": 0,
            "original_baseline_valence": 0.1, "original_baseline_arousal": 0.3,
        },
    )


def _no_regulation() -> RegulationResult:
    """Regulation result with no action."""
    return RegulationResult()


def _breakthrough_regulation() -> RegulationResult:
    """Regulation result with breakthrough."""
    return RegulationResult(breakthrough=True, strategy_used="suppression")


class TestCheckSelfInquiry:
    """Tests for the main check_self_inquiry function."""

    def test_returns_none_on_early_turns(self) -> None:
        """No self-inquiry on turns 0 and 1 — not enough context."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.8, intensity=0.9)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.1)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=1)
        assert result is None

    def test_returns_none_when_calm(self) -> None:
        """No self-inquiry when emotional state is stable and low-intensity."""
        state = _make_state(PrimaryEmotion.NEUTRAL, valence=0.1, intensity=0.2)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.2)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=5)
        assert result is None

    def test_regulation_failure_breakthrough(self) -> None:
        """Breakthrough triggers regulation failure inquiry."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.7, intensity=0.85)
        prev = _make_state(PrimaryEmotion.ANGER, valence=-0.5, intensity=0.75)
        reg = _breakthrough_regulation()
        result = check_self_inquiry(state, prev, None, reg, turn_count=3)
        assert result is not None
        assert result.trigger == InquiryTrigger.REGULATION_FAILURE
        assert result.suggested_behavior == SuggestedBehavior.EXPRESS
        assert "agotada" in result.inquiry_text
        assert result.intensity > 0.3

    def test_emotional_whiplash(self) -> None:
        """Large valence sign change triggers whiplash."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.6, intensity=0.7)
        prev = _make_state(PrimaryEmotion.JOY, valence=0.5, intensity=0.6)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=4)
        assert result is not None
        assert result.trigger == InquiryTrigger.EMOTIONAL_WHIPLASH
        assert result.suggested_behavior == SuggestedBehavior.PAUSE
        assert "joy" in result.inquiry_text.lower()
        assert "anger" in result.inquiry_text.lower()

    def test_whiplash_requires_sign_change(self) -> None:
        """Large delta but same sign does not trigger whiplash."""
        state = _make_state(PrimaryEmotion.JOY, valence=0.9, intensity=0.8)
        prev = _make_state(PrimaryEmotion.CONTENTMENT, valence=0.2, intensity=0.3)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=4)
        # Should not be whiplash (both positive)
        if result is not None:
            assert result.trigger != InquiryTrigger.EMOTIONAL_WHIPLASH

    def test_value_conflict(self) -> None:
        """Meta-emotion conflict triggers value conflict inquiry."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.5, intensity=0.6)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.3)
        meta = MetaEmotion(
            target_emotion=PrimaryEmotion.ANGER,
            meta_response="conflict",
            intensity=0.5,
            reason="anger conflicts with value 'compassion'",
        )
        result = check_self_inquiry(state, prev, meta, _no_regulation(), turn_count=5)
        assert result is not None
        assert result.trigger == InquiryTrigger.VALUE_CONFLICT
        assert result.suggested_behavior == SuggestedBehavior.QUESTION_SELF
        assert "compassion" in result.inquiry_text

    def test_emotional_surge(self) -> None:
        """Intensity jump >0.3 triggers surge."""
        state = _make_state(PrimaryEmotion.FEAR, valence=-0.4, intensity=0.8)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.3)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=3)
        assert result is not None
        assert result.trigger == InquiryTrigger.EMOTIONAL_SURGE
        assert result.suggested_behavior == SuggestedBehavior.PAUSE
        assert "abrupto" in result.inquiry_text

    def test_surge_requires_significant_delta(self) -> None:
        """Small intensity change does not trigger surge."""
        state = _make_state(PrimaryEmotion.CONTEMPLATION, valence=0.3, intensity=0.5)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.3)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=3)
        # Delta is only 0.2, not >0.3
        if result is not None:
            assert result.trigger != InquiryTrigger.EMOTIONAL_SURGE

    def test_novel_emotion(self) -> None:
        """New emotion with high intensity triggers novel inquiry."""
        state = _make_state(PrimaryEmotion.EXCITEMENT, valence=0.6, intensity=0.65)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.1, intensity=0.45)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=5)
        assert result is not None
        assert result.trigger == InquiryTrigger.NOVEL_EMOTION
        assert result.suggested_behavior == SuggestedBehavior.QUESTION_SELF
        assert "nueva" in result.inquiry_text

    def test_novel_requires_high_intensity(self) -> None:
        """New emotion with low intensity does not trigger novel."""
        state = _make_state(PrimaryEmotion.CONTEMPLATION, valence=0.2, intensity=0.4)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.3)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=5)
        # Intensity 0.4 < 0.6 threshold
        assert result is None

    def test_sustained_extreme(self) -> None:
        """Same intense emotion for >3 turns triggers sustained."""
        state = _make_state(PrimaryEmotion.SADNESS, valence=-0.5, intensity=0.7, duration=5)
        prev = _make_state(PrimaryEmotion.SADNESS, valence=-0.5, intensity=0.7, duration=4)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=8)
        assert result is not None
        assert result.trigger == InquiryTrigger.SUSTAINED_EXTREME
        assert result.suggested_behavior == SuggestedBehavior.ACKNOWLEDGE
        assert "5 turnos" in result.inquiry_text

    def test_sustained_requires_same_emotion(self) -> None:
        """Different emotion does not trigger sustained even if long duration."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.6, intensity=0.7, duration=5)
        prev = _make_state(PrimaryEmotion.SADNESS, valence=-0.5, intensity=0.7, duration=4)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=8)
        # Different emotions — may trigger novel but not sustained
        if result is not None:
            assert result.trigger != InquiryTrigger.SUSTAINED_EXTREME


class TestPriorityOrder:
    """Tests that triggers fire in priority order."""

    def test_breakthrough_beats_everything(self) -> None:
        """Regulation failure has highest priority."""
        # Setup conditions for multiple triggers
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.8, intensity=0.9)
        prev = _make_state(PrimaryEmotion.JOY, valence=0.5, intensity=0.3)
        meta = MetaEmotion(
            target_emotion=PrimaryEmotion.ANGER,
            meta_response="conflict",
            intensity=0.5,
            reason="anger conflicts with value 'compassion'",
        )
        reg = _breakthrough_regulation()
        result = check_self_inquiry(state, prev, meta, reg, turn_count=5)
        assert result is not None
        assert result.trigger == InquiryTrigger.REGULATION_FAILURE

    def test_whiplash_beats_conflict(self) -> None:
        """Whiplash has higher priority than value conflict."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.6, intensity=0.7)
        prev = _make_state(PrimaryEmotion.JOY, valence=0.5, intensity=0.6)
        meta = MetaEmotion(
            target_emotion=PrimaryEmotion.ANGER,
            meta_response="conflict",
            intensity=0.5,
            reason="anger conflicts with value 'compassion'",
        )
        result = check_self_inquiry(state, prev, meta, _no_regulation(), turn_count=5)
        assert result is not None
        assert result.trigger == InquiryTrigger.EMOTIONAL_WHIPLASH


class TestIntensityClamping:
    """Tests that inquiry intensities are properly clamped."""

    def test_intensity_within_bounds(self) -> None:
        """All triggers produce intensity between 0 and 1."""
        state = _make_state(PrimaryEmotion.ANGER, valence=-0.9, intensity=1.0)
        prev = _make_state(PrimaryEmotion.JOY, valence=0.9, intensity=0.1)
        reg = _breakthrough_regulation()
        result = check_self_inquiry(state, prev, None, reg, turn_count=3)
        assert result is not None
        assert 0.0 <= result.intensity <= 1.0

    def test_surge_intensity_proportional(self) -> None:
        """Surge intensity scales with the delta."""
        state = _make_state(PrimaryEmotion.FEAR, valence=-0.5, intensity=0.9)
        prev = _make_state(PrimaryEmotion.NEUTRAL, valence=0.0, intensity=0.3)
        result = check_self_inquiry(state, prev, None, _no_regulation(), turn_count=3)
        assert result is not None
        assert result.trigger == InquiryTrigger.EMOTIONAL_SURGE
        # Delta is 0.6, intensity should be 0.6 * 0.8 = 0.48
        assert 0.3 <= result.intensity <= 0.9


class TestSelfInquiryModel:
    """Tests for the SelfInquiry pydantic model."""

    def test_valid_model(self) -> None:
        inquiry = SelfInquiry(
            trigger=InquiryTrigger.EMOTIONAL_SURGE,
            inquiry_text="Test reflection",
            intensity=0.5,
            suggested_behavior=SuggestedBehavior.PAUSE,
        )
        assert inquiry.trigger == InquiryTrigger.EMOTIONAL_SURGE
        assert inquiry.intensity == 0.5

    def test_intensity_clamped_to_range(self) -> None:
        """Intensity must be between 0 and 1."""
        with pytest.raises(Exception):
            SelfInquiry(
                trigger=InquiryTrigger.EMOTIONAL_SURGE,
                inquiry_text="Test",
                intensity=1.5,
                suggested_behavior=SuggestedBehavior.PAUSE,
            )
