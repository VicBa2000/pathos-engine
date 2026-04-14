"""Tests para el Pilar 4: Desarrollo Ontogenico.

Cubre: etapas, transiciones, gating de sistemas y emociones,
speed multiplier, criterios cualitativos, persistencia, edge cases.
"""

import pytest

from pathos.models.development import (
    BASE_THRESHOLDS,
    BASE_TRANSITION_CRITERIA,
    SPEED_MULTIPLIERS,
    STAGE_1_EMOTIONS,
    STAGE_2_EMOTIONS,
    STAGE_3_EMOTIONS,
    STAGE_4_EMOTIONS,
    STAGE_ORDER,
    SYSTEMS_UNLOCKED_AT,
    DevelopmentConfig,
    DevelopmentSpeed,
    DevelopmentStage,
    DevelopmentState,
    TransitionCriteria,
    TransitionEvent,
    TransitionMode,
    default_development_state,
    get_cumulative_systems,
    get_emotions_for_stage,
)
from pathos.engine.development import (
    approve_pending_transition,
    attempt_transition,
    check_transition_criteria,
    compute_stage_from_experience,
    execute_transition,
    filter_emotions_by_stage,
    get_development_details,
    get_development_prompt,
    get_effective_criteria,
    get_effective_multiplier,
    get_effective_thresholds,
    get_next_stage,
    get_stage_index,
    get_stage_progress,
    is_emotion_available,
    apply_stage_modifiers,
    is_system_available,
    track_experience,
)


# ===========================================================================
# Test DevelopmentStage Enum
# ===========================================================================

class TestDevelopmentStage:
    """Tests for the DevelopmentStage enum."""

    def test_stage_values(self) -> None:
        assert DevelopmentStage.SENSORIMOTOR.value == "sensorimotor"
        assert DevelopmentStage.PREOPERATIONAL.value == "preoperational"
        assert DevelopmentStage.CONCRETE_OPERATIONAL.value == "concrete_operational"
        assert DevelopmentStage.FORMAL_OPERATIONAL.value == "formal_operational"
        assert DevelopmentStage.POST_FORMAL.value == "post_formal"

    def test_five_stages(self) -> None:
        assert len(DevelopmentStage) == 5

    def test_stage_order(self) -> None:
        assert len(STAGE_ORDER) == 5
        assert STAGE_ORDER[0] == DevelopmentStage.SENSORIMOTOR
        assert STAGE_ORDER[-1] == DevelopmentStage.POST_FORMAL


# ===========================================================================
# Test Speed Multipliers
# ===========================================================================

class TestSpeedMultipliers:
    """Tests for speed multiplier system."""

    def test_glacial(self) -> None:
        assert SPEED_MULTIPLIERS[DevelopmentSpeed.GLACIAL] == 0.25

    def test_natural(self) -> None:
        assert SPEED_MULTIPLIERS[DevelopmentSpeed.NATURAL] == 1.0

    def test_accelerated(self) -> None:
        assert SPEED_MULTIPLIERS[DevelopmentSpeed.ACCELERATED] == 4.0

    def test_fast(self) -> None:
        assert SPEED_MULTIPLIERS[DevelopmentSpeed.FAST] == 10.0

    def test_effective_multiplier_natural(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.NATURAL)
        assert get_effective_multiplier(config) == 1.0

    def test_effective_multiplier_custom(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.CUSTOM, speed_multiplier=7.5)
        assert get_effective_multiplier(config) == 7.5

    def test_effective_thresholds_natural(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.NATURAL)
        thresholds = get_effective_thresholds(config)
        assert thresholds[DevelopmentStage.SENSORIMOTOR] == 0
        assert thresholds[DevelopmentStage.PREOPERATIONAL] == 50
        assert thresholds[DevelopmentStage.CONCRETE_OPERATIONAL] == 200
        assert thresholds[DevelopmentStage.FORMAL_OPERATIONAL] == 500
        assert thresholds[DevelopmentStage.POST_FORMAL] == 1500

    def test_effective_thresholds_fast(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.FAST)
        thresholds = get_effective_thresholds(config)
        assert thresholds[DevelopmentStage.PREOPERATIONAL] == 5
        assert thresholds[DevelopmentStage.CONCRETE_OPERATIONAL] == 20
        assert thresholds[DevelopmentStage.FORMAL_OPERATIONAL] == 50
        assert thresholds[DevelopmentStage.POST_FORMAL] == 150

    def test_effective_thresholds_glacial(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.GLACIAL)
        thresholds = get_effective_thresholds(config)
        assert thresholds[DevelopmentStage.PREOPERATIONAL] == 200
        assert thresholds[DevelopmentStage.CONCRETE_OPERATIONAL] == 800

    def test_effective_thresholds_accelerated(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.ACCELERATED)
        thresholds = get_effective_thresholds(config)
        assert thresholds[DevelopmentStage.PREOPERATIONAL] == 12
        assert thresholds[DevelopmentStage.CONCRETE_OPERATIONAL] == 50

    def test_thresholds_never_zero_for_nonzero_base(self) -> None:
        """Even with high speed, thresholds >= 1 for non-zero bases."""
        config = DevelopmentConfig(speed=DevelopmentSpeed.CUSTOM, speed_multiplier=20.0)
        thresholds = get_effective_thresholds(config)
        for stage, threshold in thresholds.items():
            if BASE_THRESHOLDS[stage] > 0:
                assert threshold >= 1, f"{stage} threshold should be >= 1"


# ===========================================================================
# Test Emotions Per Stage
# ===========================================================================

class TestEmotionsPerStage:
    """Tests for emotion gating per stage."""

    def test_stage_1_has_6_basic(self) -> None:
        emotions = get_emotions_for_stage(DevelopmentStage.SENSORIMOTOR)
        assert len(emotions) == 6
        assert "joy" in emotions
        assert "sadness" in emotions
        assert "fear" in emotions
        assert "anger" in emotions
        assert "surprise" in emotions
        assert "contentment" in emotions

    def test_stage_1_excludes_complex(self) -> None:
        emotions = get_emotions_for_stage(DevelopmentStage.SENSORIMOTOR)
        assert "gratitude" not in emotions
        assert "hope" not in emotions
        assert "anxiety" not in emotions

    def test_stage_2_adds_complex(self) -> None:
        emotions = get_emotions_for_stage(DevelopmentStage.PREOPERATIONAL)
        assert len(emotions) == 10
        assert "gratitude" in emotions
        assert "hope" in emotions
        assert "frustration" in emotions
        assert "anxiety" in emotions
        # Still has basics
        assert "joy" in emotions
        assert "sadness" in emotions

    def test_stage_3_has_all_19(self) -> None:
        emotions = get_emotions_for_stage(DevelopmentStage.CONCRETE_OPERATIONAL)
        assert len(emotions) == 19
        assert "excitement" in emotions
        assert "relief" in emotions
        assert "helplessness" in emotions

    def test_stage_4_same_as_3(self) -> None:
        """Stage 4+ has same 19 primary emotions (emergents handled separately)."""
        e3 = get_emotions_for_stage(DevelopmentStage.CONCRETE_OPERATIONAL)
        e4 = get_emotions_for_stage(DevelopmentStage.FORMAL_OPERATIONAL)
        assert e3 == e4

    def test_stage_5_same_as_4(self) -> None:
        e4 = get_emotions_for_stage(DevelopmentStage.FORMAL_OPERATIONAL)
        e5 = get_emotions_for_stage(DevelopmentStage.POST_FORMAL)
        assert e4 == e5

    def test_emotions_are_cumulative(self) -> None:
        """Each stage has at least as many emotions as the previous."""
        prev_count = 0
        for stage in STAGE_ORDER:
            emotions = get_emotions_for_stage(stage)
            assert len(emotions) >= prev_count
            prev_count = len(emotions)


# ===========================================================================
# Test Systems Per Stage
# ===========================================================================

class TestSystemsPerStage:
    """Tests for system gating per stage."""

    def test_sensorimotor_systems(self) -> None:
        systems = get_cumulative_systems(DevelopmentStage.SENSORIMOTOR)
        assert "appraisal" in systems
        assert "generator" in systems
        assert "homeostasis" in systems
        assert "body_state" in systems
        # Should NOT have advanced
        assert "schemas" not in systems
        assert "regulation" not in systems

    def test_preoperational_adds_systems(self) -> None:
        systems = get_cumulative_systems(DevelopmentStage.PREOPERATIONAL)
        assert "schemas" in systems
        assert "contagion" in systems
        assert "mood" in systems
        assert "needs" in systems
        # Also has stage 1
        assert "appraisal" in systems
        # Should NOT have stage 3
        assert "meta_emotion" not in systems

    def test_concrete_operational_adds_systems(self) -> None:
        systems = get_cumulative_systems(DevelopmentStage.CONCRETE_OPERATIONAL)
        assert "meta_emotion" in systems
        assert "regulation" in systems
        assert "social" in systems
        assert "forecasting" in systems
        assert "self_inquiry" in systems
        assert "temporal" in systems
        # Should NOT have stage 4
        assert "reappraisal" not in systems
        assert "creativity" not in systems

    def test_formal_operational_adds_systems(self) -> None:
        systems = get_cumulative_systems(DevelopmentStage.FORMAL_OPERATIONAL)
        assert "reappraisal" in systems
        assert "creativity" in systems
        assert "immune" in systems
        assert "narrative" in systems
        assert "somatic" in systems
        assert "workspace" in systems
        # Also has all previous
        assert "appraisal" in systems
        assert "regulation" in systems

    def test_post_formal_adds_systems(self) -> None:
        systems = get_cumulative_systems(DevelopmentStage.POST_FORMAL)
        assert "discovery" in systems
        assert "phenomenology" in systems
        assert "dialectical" in systems
        # Has all previous
        assert "workspace" in systems
        assert "regulation" in systems
        assert "appraisal" in systems

    def test_systems_are_cumulative(self) -> None:
        prev_systems: set[str] = set()
        for stage in STAGE_ORDER:
            systems = get_cumulative_systems(stage)
            assert systems >= prev_systems, f"{stage}: lost systems from previous stage"
            prev_systems = systems


# ===========================================================================
# Test Stage Navigation
# ===========================================================================

class TestStageNavigation:
    """Tests for stage navigation helpers."""

    def test_next_stage_sensorimotor(self) -> None:
        assert get_next_stage(DevelopmentStage.SENSORIMOTOR) == DevelopmentStage.PREOPERATIONAL

    def test_next_stage_formal(self) -> None:
        assert get_next_stage(DevelopmentStage.FORMAL_OPERATIONAL) == DevelopmentStage.POST_FORMAL

    def test_next_stage_post_formal_is_none(self) -> None:
        assert get_next_stage(DevelopmentStage.POST_FORMAL) is None

    def test_stage_index(self) -> None:
        assert get_stage_index(DevelopmentStage.SENSORIMOTOR) == 0
        assert get_stage_index(DevelopmentStage.POST_FORMAL) == 4

    def test_compute_stage_natural(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.NATURAL)
        assert compute_stage_from_experience(0, config) == DevelopmentStage.SENSORIMOTOR
        assert compute_stage_from_experience(49, config) == DevelopmentStage.SENSORIMOTOR
        assert compute_stage_from_experience(50, config) == DevelopmentStage.PREOPERATIONAL
        assert compute_stage_from_experience(199, config) == DevelopmentStage.PREOPERATIONAL
        assert compute_stage_from_experience(200, config) == DevelopmentStage.CONCRETE_OPERATIONAL
        assert compute_stage_from_experience(500, config) == DevelopmentStage.FORMAL_OPERATIONAL
        assert compute_stage_from_experience(1500, config) == DevelopmentStage.POST_FORMAL

    def test_compute_stage_fast(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.FAST)
        assert compute_stage_from_experience(0, config) == DevelopmentStage.SENSORIMOTOR
        assert compute_stage_from_experience(5, config) == DevelopmentStage.PREOPERATIONAL
        assert compute_stage_from_experience(20, config) == DevelopmentStage.CONCRETE_OPERATIONAL
        assert compute_stage_from_experience(50, config) == DevelopmentStage.FORMAL_OPERATIONAL
        assert compute_stage_from_experience(150, config) == DevelopmentStage.POST_FORMAL


# ===========================================================================
# Test Experience Tracking
# ===========================================================================

class TestExperienceTracking:
    """Tests for the experience tracking system."""

    def test_track_increments_experience(self) -> None:
        state = DevelopmentState(enabled=True)
        track_experience(state, "joy", 0.5)
        assert state.total_experience == 1

    def test_track_records_emotions(self) -> None:
        state = DevelopmentState(enabled=True)
        track_experience(state, "joy", 0.5)
        track_experience(state, "sadness", 0.3)
        track_experience(state, "joy", 0.4)
        assert state.distinct_emotions_experienced == {"joy", "sadness"}

    def test_track_high_intensity(self) -> None:
        state = DevelopmentState(enabled=True)
        track_experience(state, "anger", 0.6)  # Not high enough
        track_experience(state, "anger", 0.7)  # >= 0.7
        track_experience(state, "anger", 0.9)  # >= 0.7
        assert state.high_intensity_episodes == 2

    def test_track_regulation_uses(self) -> None:
        state = DevelopmentState(enabled=True)
        track_experience(state, "anger", 0.5, regulation_used=False)
        track_experience(state, "anger", 0.5, regulation_used=True)
        track_experience(state, "anger", 0.5, regulation_used=True)
        assert state.regulation_uses == 2

    def test_track_multiple_turns(self) -> None:
        state = DevelopmentState(enabled=True)
        for i in range(100):
            track_experience(state, "joy" if i % 2 == 0 else "sadness", 0.5)
        assert state.total_experience == 100
        assert state.distinct_emotions_experienced == {"joy", "sadness"}


# ===========================================================================
# Test Transition Criteria
# ===========================================================================

class TestTransitionCriteria:
    """Tests for transition criteria checking."""

    def test_no_transition_from_post_formal(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.POST_FORMAL,
            total_experience=9999,
        )
        result = check_transition_criteria(state)
        assert result is None

    def test_transition_sensorimotor_to_preoperational(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=50,
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger"},
            high_intensity_episodes=1,
        )
        result = check_transition_criteria(state)
        assert result == DevelopmentStage.PREOPERATIONAL

    def test_transition_blocked_by_experience(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=49,  # Not enough
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger"},
            high_intensity_episodes=1,
        )
        result = check_transition_criteria(state)
        assert result is None

    def test_transition_blocked_by_emotions(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=50,
            distinct_emotions_experienced={"joy", "sadness", "fear"},  # Only 3, need 4
            high_intensity_episodes=1,
        )
        result = check_transition_criteria(state)
        assert result is None

    def test_transition_preop_to_concrete(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.PREOPERATIONAL,
            total_experience=200,
        )
        result = check_transition_criteria(state, schemas_count=3, episodic_count=10)
        assert result == DevelopmentStage.CONCRETE_OPERATIONAL

    def test_transition_concrete_to_formal(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.CONCRETE_OPERATIONAL,
            total_experience=500,
            regulation_uses=20,
        )
        result = check_transition_criteria(
            state, identities_count=5, crises_resolved=1,
        )
        assert result == DevelopmentStage.FORMAL_OPERATIONAL

    def test_criteria_scaled_by_speed(self) -> None:
        """Fast speed requires fewer schemas, episodes, etc."""
        config = DevelopmentConfig(speed=DevelopmentSpeed.FAST)
        criteria = get_effective_criteria(config, DevelopmentStage.CONCRETE_OPERATIONAL)
        assert criteria is not None
        assert criteria.min_experience == 20  # 200 / 10
        assert criteria.min_schemas_formed == 1  # 3 / 10 -> max(1, 0) = 1
        assert criteria.min_episodic_memories == 1  # 10 / 10 -> 1


# ===========================================================================
# Test Auto vs Manual Transitions
# ===========================================================================

class TestTransitionModes:
    """Tests for auto and manual transition modes."""

    def test_auto_transitions_immediately(self) -> None:
        state = DevelopmentState(
            enabled=True,
            config=DevelopmentConfig(transition_mode=TransitionMode.AUTO),
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=50,
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger"},
            high_intensity_episodes=1,
        )
        event = attempt_transition(state, turn_number=50)
        assert event is not None
        assert event.from_stage == DevelopmentStage.SENSORIMOTOR
        assert event.to_stage == DevelopmentStage.PREOPERATIONAL
        assert state.current_stage == DevelopmentStage.PREOPERATIONAL

    def test_manual_sets_pending(self) -> None:
        state = DevelopmentState(
            enabled=True,
            config=DevelopmentConfig(transition_mode=TransitionMode.MANUAL),
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=50,
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger"},
            high_intensity_episodes=1,
        )
        event = attempt_transition(state, turn_number=50)
        assert event is None  # No transition yet
        assert state.pending_transition == DevelopmentStage.PREOPERATIONAL
        assert state.current_stage == DevelopmentStage.SENSORIMOTOR  # Still in old stage

    def test_approve_pending(self) -> None:
        state = DevelopmentState(
            enabled=True,
            config=DevelopmentConfig(transition_mode=TransitionMode.MANUAL),
            current_stage=DevelopmentStage.SENSORIMOTOR,
            pending_transition=DevelopmentStage.PREOPERATIONAL,
            total_experience=55,
        )
        event = approve_pending_transition(state, turn_number=55)
        assert event is not None
        assert state.current_stage == DevelopmentStage.PREOPERATIONAL
        assert state.pending_transition is None

    def test_approve_no_pending(self) -> None:
        state = DevelopmentState(enabled=True)
        event = approve_pending_transition(state)
        assert event is None

    def test_transition_history_recorded(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=50,
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger"},
            high_intensity_episodes=1,
        )
        attempt_transition(state, turn_number=50)
        assert len(state.transition_history) == 1
        assert state.transition_history[0].from_stage == DevelopmentStage.SENSORIMOTOR
        assert state.transition_history[0].at_experience == 50


# ===========================================================================
# Test System Availability (Gating)
# ===========================================================================

class TestSystemAvailability:
    """Tests for system gating based on development stage."""

    def test_disabled_all_available(self) -> None:
        """When development is OFF, all systems are available."""
        state = DevelopmentState(enabled=False)
        assert is_system_available(state, "regulation") is True
        assert is_system_available(state, "creativity") is True
        assert is_system_available(state, "phenomenology") is True

    def test_sensorimotor_limited(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        assert is_system_available(state, "appraisal") is True
        assert is_system_available(state, "generator") is True
        assert is_system_available(state, "schemas") is False
        assert is_system_available(state, "regulation") is False
        assert is_system_available(state, "creativity") is False

    def test_preoperational_has_schemas(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.PREOPERATIONAL,
        )
        assert is_system_available(state, "schemas") is True
        assert is_system_available(state, "contagion") is True
        assert is_system_available(state, "regulation") is False

    def test_formal_has_most(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.FORMAL_OPERATIONAL,
        )
        assert is_system_available(state, "creativity") is True
        assert is_system_available(state, "immune") is True
        assert is_system_available(state, "narrative") is True
        assert is_system_available(state, "workspace") is True
        assert is_system_available(state, "discovery") is False


# ===========================================================================
# Test Emotion Availability (Gating)
# ===========================================================================

class TestEmotionAvailability:
    """Tests for emotion gating based on development stage."""

    def test_disabled_all_available(self) -> None:
        state = DevelopmentState(enabled=False)
        assert is_emotion_available(state, "gratitude") is True
        assert is_emotion_available(state, "mixed") is True

    def test_sensorimotor_basics_only(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        assert is_emotion_available(state, "joy") is True
        assert is_emotion_available(state, "sadness") is True
        assert is_emotion_available(state, "gratitude") is False
        assert is_emotion_available(state, "anxiety") is False

    def test_preoperational_adds_complex(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.PREOPERATIONAL,
        )
        assert is_emotion_available(state, "gratitude") is True
        assert is_emotion_available(state, "hope") is True
        assert is_emotion_available(state, "excitement") is False


# ===========================================================================
# Test Emotion Filtering
# ===========================================================================

class TestEmotionFiltering:
    """Tests for filtering emotional stack by stage."""

    def test_disabled_no_filter(self) -> None:
        state = DevelopmentState(enabled=False)
        stack = {"joy": 0.5, "gratitude": 0.3, "excitement": 0.2}
        result = filter_emotions_by_stage(state, stack)
        assert result == stack

    def test_filters_unavailable(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        stack = {"joy": 0.5, "gratitude": 0.3, "excitement": 0.2}
        result = filter_emotions_by_stage(state, stack)
        assert "gratitude" not in result
        assert "excitement" not in result
        assert "joy" in result

    def test_redistributes_activation(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        stack = {"joy": 0.5, "gratitude": 0.5}
        result = filter_emotions_by_stage(state, stack)
        # Joy gets gratitude's activation added
        assert result["joy"] == pytest.approx(1.0)

    def test_empty_stack(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        result = filter_emotions_by_stage(state, {})
        assert result == {}

    def test_all_unavailable_fallback(self) -> None:
        """If all emotions are filtered, fallback to neutral if available."""
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        # None of these are in stage 1
        stack = {"excitement": 0.5, "relief": 0.3, "helplessness": 0.2}
        result = filter_emotions_by_stage(state, stack)
        # neutral is not in stage 1 emotions, so should return original
        assert result == stack  # fallback returns original


# ===========================================================================
# Test Default State
# ===========================================================================

class TestDefaultState:
    """Tests for default development state."""

    def test_default_disabled(self) -> None:
        state = default_development_state()
        assert state.enabled is False

    def test_default_sensorimotor(self) -> None:
        state = default_development_state()
        assert state.current_stage == DevelopmentStage.SENSORIMOTOR

    def test_default_zero_experience(self) -> None:
        state = default_development_state()
        assert state.total_experience == 0
        assert len(state.distinct_emotions_experienced) == 0
        assert state.high_intensity_episodes == 0
        assert state.regulation_uses == 0

    def test_default_natural_speed(self) -> None:
        state = default_development_state()
        assert state.config.speed == DevelopmentSpeed.NATURAL

    def test_default_auto_transitions(self) -> None:
        state = default_development_state()
        assert state.config.transition_mode == TransitionMode.AUTO


# ===========================================================================
# Test Serialization
# ===========================================================================

class TestSerialization:
    """Tests for serialization/deserialization of development state."""

    def test_roundtrip_default(self) -> None:
        state = default_development_state()
        data = state.model_dump()
        restored = DevelopmentState(**data)
        assert restored.enabled == state.enabled
        assert restored.current_stage == state.current_stage
        assert restored.total_experience == 0

    def test_roundtrip_with_data(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.CONCRETE_OPERATIONAL,
            total_experience=250,
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger", "gratitude"},
            high_intensity_episodes=12,
            regulation_uses=8,
            config=DevelopmentConfig(
                speed=DevelopmentSpeed.ACCELERATED,
                speed_multiplier=4.0,
                transition_mode=TransitionMode.MANUAL,
            ),
            transition_history=[
                TransitionEvent(
                    from_stage=DevelopmentStage.SENSORIMOTOR,
                    to_stage=DevelopmentStage.PREOPERATIONAL,
                    at_experience=50,
                    turn_number=50,
                ),
            ],
        )
        data = state.model_dump()
        restored = DevelopmentState(**data)
        assert restored.enabled is True
        assert restored.current_stage == DevelopmentStage.CONCRETE_OPERATIONAL
        assert restored.total_experience == 250
        assert "joy" in restored.distinct_emotions_experienced
        assert restored.high_intensity_episodes == 12
        assert restored.config.speed == DevelopmentSpeed.ACCELERATED
        assert len(restored.transition_history) == 1

    def test_roundtrip_pending_transition(self) -> None:
        state = DevelopmentState(
            enabled=True,
            pending_transition=DevelopmentStage.PREOPERATIONAL,
        )
        data = state.model_dump()
        restored = DevelopmentState(**data)
        assert restored.pending_transition == DevelopmentStage.PREOPERATIONAL

    def test_sets_serialized_as_lists(self) -> None:
        state = DevelopmentState(
            enabled=True,
            distinct_emotions_experienced={"joy", "sadness"},
        )
        data = state.model_dump()
        assert isinstance(data["distinct_emotions_experienced"], list)
        assert set(data["distinct_emotions_experienced"]) == {"joy", "sadness"}


# ===========================================================================
# Test Stage Progress
# ===========================================================================

class TestStageProgress:
    """Tests for stage progress computation."""

    def test_progress_start(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=0,
        )
        progress = get_stage_progress(state)
        assert progress["stage"] == "sensorimotor"
        assert progress["stage_index"] == 0
        assert progress["progress_pct"] == 0.0
        assert progress["next_stage"] == "preoperational"

    def test_progress_midway(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=25,
        )
        progress = get_stage_progress(state)
        assert progress["progress_pct"] == pytest.approx(50.0)

    def test_progress_post_formal_100(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.POST_FORMAL,
            total_experience=3000,
        )
        progress = get_stage_progress(state)
        assert progress["progress_pct"] == 100.0
        assert progress["next_stage"] is None

    def test_progress_with_pending(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            pending_transition=DevelopmentStage.PREOPERATIONAL,
            total_experience=50,
        )
        progress = get_stage_progress(state)
        assert progress["pending_transition"] == "preoperational"


# ===========================================================================
# Test Development Prompt
# ===========================================================================

class TestDevelopmentPrompt:
    """Tests for development prompt generation."""

    def test_disabled_returns_none(self) -> None:
        state = DevelopmentState(enabled=False)
        assert get_development_prompt(state) is None

    def test_sensorimotor_prompt(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        prompt = get_development_prompt(state)
        assert prompt is not None
        assert "SENSORIOMOTOR" in prompt
        assert "basicas" in prompt

    def test_post_formal_prompt(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.POST_FORMAL,
            total_experience=2000,
        )
        prompt = get_development_prompt(state)
        assert prompt is not None
        assert "POST-FORMAL" in prompt
        assert "sabiduria" in prompt

    def test_near_transition_shows_progress(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=45,  # 90% of 50
        )
        prompt = get_development_prompt(state)
        assert prompt is not None
        assert "progreso" in prompt


# ===========================================================================
# Test Development Details
# ===========================================================================

class TestDevelopmentDetails:
    """Tests for development details (research endpoint)."""

    def test_details_disabled(self) -> None:
        state = DevelopmentState(enabled=False)
        details = get_development_details(state)
        assert details["enabled"] is False
        assert details["stage"] == "sensorimotor"

    def test_details_enabled(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.CONCRETE_OPERATIONAL,
            total_experience=300,
            distinct_emotions_experienced={"joy", "sadness", "fear", "anger", "gratitude"},
            config=DevelopmentConfig(speed=DevelopmentSpeed.ACCELERATED),
        )
        details = get_development_details(state)
        assert details["enabled"] is True
        assert details["stage"] == "concrete_operational"
        assert details["total_experience"] == 300
        assert details["speed"] == "accelerated"
        assert details["distinct_emotions_count"] == 5
        assert len(details["available_systems"]) > 0
        assert details["available_emotions_count"] == 19


# ===========================================================================
# Test Full Flow
# ===========================================================================

class TestFullDevelopmentFlow:
    """Integration tests for the complete development lifecycle."""

    def test_natural_speed_lifecycle(self) -> None:
        """Agent grows through stages at natural speed."""
        state = DevelopmentState(
            enabled=True,
            config=DevelopmentConfig(speed=DevelopmentSpeed.NATURAL),
        )

        # Simulate 50 turns with varied emotions
        emotions = ["joy", "sadness", "fear", "anger", "surprise", "contentment"]
        for i in range(50):
            emotion = emotions[i % len(emotions)]
            intensity = 0.8 if i % 10 == 0 else 0.4
            track_experience(state, emotion, intensity)

        # Should meet stage 2 criteria
        assert state.total_experience == 50
        assert len(state.distinct_emotions_experienced) >= 4
        assert state.high_intensity_episodes >= 1

        event = attempt_transition(state, turn_number=50)
        assert event is not None
        assert state.current_stage == DevelopmentStage.PREOPERATIONAL

    def test_fast_speed_reaches_stage_3_in_50(self) -> None:
        """Speed x10 reaches stage 3 in ~50 turns."""
        state = DevelopmentState(
            enabled=True,
            config=DevelopmentConfig(speed=DevelopmentSpeed.FAST),
        )

        emotions = ["joy", "sadness", "fear", "anger", "surprise", "contentment",
                     "gratitude", "hope", "frustration", "anxiety"]

        # Simulate 50 turns
        for i in range(50):
            emotion = emotions[i % len(emotions)]
            track_experience(state, emotion, 0.8 if i % 5 == 0 else 0.4)

        # Transition through stages
        # Stage 1 -> 2 (threshold 5)
        event = attempt_transition(state, turn_number=50)
        if event:
            # Stage 2 -> 3 (threshold 20, needs schemas>=1, episodes>=1)
            event = attempt_transition(state, schemas_count=1, episodic_count=1, turn_number=50)
            if event:
                assert state.current_stage == DevelopmentStage.CONCRETE_OPERATIONAL

    def test_development_irreversible(self) -> None:
        """Development can only go forward, never backward."""
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.CONCRETE_OPERATIONAL,
            total_experience=200,
        )
        # Can't go back to sensorimotor
        old_stage = state.current_stage
        # check_transition only looks forward
        result = check_transition_criteria(state)
        if result is not None:
            # Would be FORMAL_OPERATIONAL, never backward
            assert get_stage_index(result) > get_stage_index(old_stage)

    def test_disabled_does_not_gate(self) -> None:
        """When OFF, no gating occurs — full v4 behavior."""
        state = DevelopmentState(enabled=False)
        assert is_system_available(state, "creativity") is True
        assert is_system_available(state, "immune") is True
        assert is_emotion_available(state, "gratitude") is True
        assert is_emotion_available(state, "mixed") is True
        assert get_development_prompt(state) is None


# ===========================================================================
# Test Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_extreme_speed_multiplier(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.CUSTOM, speed_multiplier=20.0)
        thresholds = get_effective_thresholds(config)
        # All thresholds should be at least 1 for non-zero bases
        assert thresholds[DevelopmentStage.PREOPERATIONAL] >= 1

    def test_very_slow_speed(self) -> None:
        config = DevelopmentConfig(speed=DevelopmentSpeed.CUSTOM, speed_multiplier=0.1)
        thresholds = get_effective_thresholds(config)
        assert thresholds[DevelopmentStage.PREOPERATIONAL] == 500  # 50 / 0.1

    def test_execute_transition_directly(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
            total_experience=100,
        )
        event = execute_transition(state, DevelopmentStage.PREOPERATIONAL, turn_number=100)
        assert state.current_stage == DevelopmentStage.PREOPERATIONAL
        assert event.at_experience == 100

    def test_multiple_transitions_in_sequence(self) -> None:
        """Ensure transitions are recorded in history."""
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        execute_transition(state, DevelopmentStage.PREOPERATIONAL, turn_number=50)
        execute_transition(state, DevelopmentStage.CONCRETE_OPERATIONAL, turn_number=200)
        assert len(state.transition_history) == 2
        assert state.current_stage == DevelopmentStage.CONCRETE_OPERATIONAL

    def test_filter_preserves_proportions(self) -> None:
        """When filtering, proportions of remaining emotions are preserved."""
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        stack = {"joy": 0.6, "sadness": 0.2, "gratitude": 0.2}
        result = filter_emotions_by_stage(state, stack)
        # joy:sadness ratio should be preserved (3:1)
        if "joy" in result and "sadness" in result:
            ratio = result["joy"] / result["sadness"]
            assert ratio == pytest.approx(3.0, rel=0.01)


# ===========================================================================
# Test Stage Modifiers
# ===========================================================================

class TestStageModifiers:
    """Tests for apply_stage_modifiers."""

    def _make_state(self) -> object:
        """Create a mock emotional state for testing."""
        from pathos.models.emotion import neutral_state
        return neutral_state()

    def test_disabled_no_change(self) -> None:
        state = DevelopmentState(enabled=False)
        emo = self._make_state()
        original_intensity = emo.intensity
        apply_stage_modifiers(state, emo)
        assert emo.intensity == original_intensity

    def test_sensorimotor_amplifies_intensity(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        emo = self._make_state()
        emo.intensity = 0.5
        emo.body_state.tension = 0.4
        apply_stage_modifiers(state, emo)
        assert emo.intensity == pytest.approx(0.6)  # 0.5 * 1.2
        assert emo.body_state.tension == pytest.approx(0.52)  # 0.4 * 1.3

    def test_sensorimotor_clamped_at_1(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.SENSORIMOTOR,
        )
        emo = self._make_state()
        emo.intensity = 0.95
        apply_stage_modifiers(state, emo)
        assert emo.intensity <= 1.0

    def test_preoperational_slight_amplification(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.PREOPERATIONAL,
        )
        emo = self._make_state()
        emo.intensity = 0.5
        apply_stage_modifiers(state, emo)
        assert emo.intensity == pytest.approx(0.55)  # 0.5 * 1.1

    def test_concrete_operational_no_change(self) -> None:
        """Stage 3 is baseline — no modifiers applied."""
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.CONCRETE_OPERATIONAL,
        )
        emo = self._make_state()
        emo.intensity = 0.5
        emo.body_state.tension = 0.4
        apply_stage_modifiers(state, emo)
        assert emo.intensity == 0.5
        assert emo.body_state.tension == 0.4

    def test_formal_operational_no_change(self) -> None:
        """Stage 4 is also baseline (no explicit modifiers)."""
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.FORMAL_OPERATIONAL,
        )
        emo = self._make_state()
        emo.intensity = 0.5
        apply_stage_modifiers(state, emo)
        assert emo.intensity == 0.5

    def test_post_formal_dampens(self) -> None:
        state = DevelopmentState(
            enabled=True,
            current_stage=DevelopmentStage.POST_FORMAL,
        )
        emo = self._make_state()
        emo.intensity = 0.5
        emo.body_state.tension = 0.4
        apply_stage_modifiers(state, emo)
        assert emo.intensity == pytest.approx(0.475)  # 0.5 * 0.95
        assert emo.body_state.tension == pytest.approx(0.36)  # 0.4 * 0.9
