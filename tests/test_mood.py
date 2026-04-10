"""Tests para el Mood System (Fase 3)."""

from pathos.engine.mood import (
    HISTORY_SIZE,
    MOOD_CONGRUENCE_STRENGTH,
    classify_mood,
    compute_mood_congruence_bias,
    update_mood,
)
from pathos.models.emotion import (
    EmotionalSnapshot,
    EmotionalState,
    Mood,
    MoodLabel,
    PrimaryEmotion,
)


def _make_mood(
    baseline_valence: float = 0.1,
    baseline_arousal: float = 0.3,
    stability: float = 0.7,
    history: list[EmotionalSnapshot] | None = None,
) -> Mood:
    return Mood(
        baseline_valence=baseline_valence,
        baseline_arousal=baseline_arousal,
        stability=stability,
        emotional_history=history or [],
        original_baseline_valence=0.1,
        original_baseline_arousal=0.3,
    )


def _make_state(
    valence: float = 0.0,
    arousal: float = 0.5,
    intensity: float = 0.5,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        primary_emotion=PrimaryEmotion.JOY,
    )


# --- Tests de MoodLabel ---


class TestClassifyMood:
    def test_neutral(self) -> None:
        assert classify_mood(0.05, 0.3) == MoodLabel.NEUTRAL

    def test_buoyant(self) -> None:
        assert classify_mood(0.4, 0.6) == MoodLabel.BUOYANT

    def test_serene(self) -> None:
        assert classify_mood(0.4, 0.2) == MoodLabel.SERENE

    def test_agitated(self) -> None:
        assert classify_mood(-0.4, 0.6) == MoodLabel.AGITATED

    def test_melancholic(self) -> None:
        assert classify_mood(-0.4, 0.2) == MoodLabel.MELANCHOLIC


# --- Tests de Mood Congruence Bias ---


class TestMoodCongruenceBias:
    def test_neutral_mood_no_bias(self) -> None:
        mood = _make_mood(baseline_valence=0.0, baseline_arousal=0.3)
        v_bias, a_bias = compute_mood_congruence_bias(mood)
        assert abs(v_bias) < 0.01
        assert abs(a_bias) < 0.01

    def test_negative_mood_negative_bias(self) -> None:
        mood = _make_mood(baseline_valence=-0.5)
        v_bias, _ = compute_mood_congruence_bias(mood)
        assert v_bias < 0

    def test_positive_mood_positive_bias(self) -> None:
        mood = _make_mood(baseline_valence=0.5)
        v_bias, _ = compute_mood_congruence_bias(mood)
        assert v_bias > 0

    def test_bias_proportional_to_distance(self) -> None:
        mood_mild = _make_mood(baseline_valence=-0.2)
        mood_strong = _make_mood(baseline_valence=-0.8)
        v_mild, _ = compute_mood_congruence_bias(mood_mild)
        v_strong, _ = compute_mood_congruence_bias(mood_strong)
        assert abs(v_strong) > abs(v_mild)


# --- Tests de Mood Update ---


class TestUpdateMood:
    def test_adds_snapshot_to_history(self) -> None:
        mood = _make_mood()
        state = _make_state(valence=0.5, arousal=0.6, intensity=0.7)
        updated = update_mood(mood, state)
        assert len(updated.emotional_history) == 1
        assert updated.emotional_history[0].valence == 0.5

    def test_history_capped_at_max_size(self) -> None:
        history = [
            EmotionalSnapshot(valence=0.1, arousal=0.3, intensity=0.5)
            for _ in range(HISTORY_SIZE)
        ]
        mood = _make_mood(history=history)
        state = _make_state(valence=0.8)
        updated = update_mood(mood, state)
        assert len(updated.emotional_history) == HISTORY_SIZE
        assert updated.emotional_history[-1].valence == 0.8

    def test_positive_emotions_drift_baseline_positive(self) -> None:
        mood = _make_mood(baseline_valence=0.0)
        # Multiples emociones positivas
        for _ in range(5):
            state = _make_state(valence=0.8, intensity=0.7)
            mood = update_mood(mood, state)
        assert mood.baseline_valence > 0.0

    def test_negative_emotions_drift_baseline_negative(self) -> None:
        mood = _make_mood(baseline_valence=0.0)
        for _ in range(5):
            state = _make_state(valence=-0.8, intensity=0.7)
            mood = update_mood(mood, state)
        assert mood.baseline_valence < 0.0

    def test_stability_slows_drift(self) -> None:
        mood_stable = _make_mood(baseline_valence=0.0, stability=0.9)
        mood_unstable = _make_mood(baseline_valence=0.0, stability=0.1)

        state = _make_state(valence=0.8, intensity=0.7)
        updated_stable = update_mood(mood_stable, state)
        updated_unstable = update_mood(mood_unstable, state)

        assert abs(updated_unstable.baseline_valence) > abs(updated_stable.baseline_valence)

    def test_mood_label_updates(self) -> None:
        mood = _make_mood(baseline_valence=-0.3, baseline_arousal=0.5)
        state = _make_state(valence=-0.8, arousal=0.8, intensity=0.9)
        # Repeated negative high arousal -> agitated
        for _ in range(5):
            mood = update_mood(mood, state)
        assert mood.label == MoodLabel.AGITATED

    def test_trend_improving(self) -> None:
        mood = _make_mood()
        # Start with negative emotions
        for _ in range(3):
            mood = update_mood(mood, _make_state(valence=-0.5, intensity=0.5))
        # Then positive emotions
        for _ in range(3):
            mood = update_mood(mood, _make_state(valence=0.8, intensity=0.7))
        assert mood.trend == "improving"

    def test_trend_declining(self) -> None:
        mood = _make_mood()
        # Start with positive
        for _ in range(3):
            mood = update_mood(mood, _make_state(valence=0.5, intensity=0.5))
        # Then negative
        for _ in range(3):
            mood = update_mood(mood, _make_state(valence=-0.8, intensity=0.7))
        assert mood.trend == "declining"

    def test_intense_emotions_weigh_more(self) -> None:
        mood1 = _make_mood(baseline_valence=0.0)
        mood2 = _make_mood(baseline_valence=0.0)

        # Misma valence, diferente intensidad
        state_mild = _make_state(valence=0.8, intensity=0.2)
        state_intense = _make_state(valence=0.8, intensity=0.9)

        updated1 = update_mood(mood1, state_mild)
        updated2 = update_mood(mood2, state_intense)

        # La emocion mas intensa deberia mover mas el baseline
        assert updated2.baseline_valence > updated1.baseline_valence

    def test_preserves_shift_fields(self) -> None:
        mood = _make_mood()
        mood = Mood(
            **{
                **mood.model_dump(),
                "extreme_event_count": 5,
                "turns_since_extreme": 3,
            }
        )
        state = _make_state()
        updated = update_mood(mood, state)
        assert updated.extreme_event_count == 5
        assert updated.turns_since_extreme == 3


# --- Tests de Rangos ---


class TestMoodRanges:
    def test_baseline_stays_in_range(self) -> None:
        mood = _make_mood(baseline_valence=0.9)
        for _ in range(50):
            state = _make_state(valence=1.0, arousal=1.0, intensity=1.0)
            mood = update_mood(mood, state)
        assert -1 <= mood.baseline_valence <= 1
        assert 0 <= mood.baseline_arousal <= 1

    def test_negative_extreme(self) -> None:
        mood = _make_mood(baseline_valence=-0.9)
        for _ in range(50):
            state = _make_state(valence=-1.0, arousal=0.0, intensity=1.0)
            mood = update_mood(mood, state)
        assert -1 <= mood.baseline_valence <= 1
        assert 0 <= mood.baseline_arousal <= 1
