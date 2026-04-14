"""Tests para Pilar 5: Autonomía Motivacional — Drives Afectivos."""

import pytest

from pathos.engine.drives import (
    _keyword_score,
    attempt_goal_generation,
    compute_base_intensity,
    compute_drive_activation,
    compute_satisfaction_signal,
    compute_urgency,
    evaluate_goal_progress,
    get_drives_details,
    get_drives_prompt,
    is_frustrated,
    process_goals,
    resolve_goal,
    update_drives,
    update_single_drive,
)
from pathos.models.drives import (
    FRUSTRATION_THRESHOLD_TURNS,
    MAX_ACTIVE_GOALS,
    URGENCY_TRIGGER_THRESHOLD,
    Drive,
    DriveState,
    DrivesState,
    EmotionalImpact,
    Goal,
    GoalOutcome,
    GoalStatus,
    default_drives_state,
)
from pathos.models.personality import PersonalityProfile


# --- Helpers ---

def _personality(**kwargs: float) -> PersonalityProfile:
    defaults = {
        "openness": 0.6, "conscientiousness": 0.6,
        "extraversion": 0.5, "agreeableness": 0.6,
        "neuroticism": 0.4,
    }
    defaults.update(kwargs)
    return PersonalityProfile(**defaults)


def _drive_state(
    drive: Drive = Drive.SEEKING,
    intensity: float = 0.5,
    satisfaction: float = 0.5,
    last_satisfied_turn: int = 0,
    urgency: float = 0.0,
) -> DriveState:
    return DriveState(
        drive=drive,
        intensity=intensity,
        satisfaction=satisfaction,
        last_satisfied_turn=last_satisfied_turn,
        urgency=urgency,
    )


def _enabled_drives_state(**drive_overrides: DriveState) -> DrivesState:
    state = default_drives_state()
    state.enabled = True
    for key, ds in drive_overrides.items():
        state.drives[key] = ds
    return state


# ===============================================================
# Test Models
# ===============================================================

class TestDriveEnum:
    def test_four_drives(self) -> None:
        assert len(Drive) == 4

    def test_drive_values(self) -> None:
        assert Drive.SEEKING.value == "seeking"
        assert Drive.CARE.value == "care"
        assert Drive.PLAY.value == "play"
        assert Drive.PANIC_GRIEF.value == "panic_grief"


class TestGoalStatus:
    def test_statuses(self) -> None:
        assert GoalStatus.ACTIVE.value == "active"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.FAILED.value == "failed"
        assert GoalStatus.ABANDONED.value == "abandoned"


class TestDefaultDrivesState:
    def test_disabled_by_default(self) -> None:
        state = default_drives_state()
        assert state.enabled is False

    def test_has_four_drives(self) -> None:
        state = default_drives_state()
        assert len(state.drives) == 4
        for drive in Drive:
            assert drive.value in state.drives

    def test_initial_intensity(self) -> None:
        state = default_drives_state()
        for ds in state.drives.values():
            assert ds.intensity == 0.3

    def test_initial_satisfaction(self) -> None:
        state = default_drives_state()
        for ds in state.drives.values():
            assert ds.satisfaction == 0.5

    def test_no_active_goals(self) -> None:
        state = default_drives_state()
        assert state.active_goals == []
        assert state.resolved_goals == []


class TestDriveState:
    def test_rejects_out_of_range_intensity(self) -> None:
        with pytest.raises(Exception):
            DriveState(drive=Drive.SEEKING, intensity=1.5)

    def test_rejects_out_of_range_satisfaction(self) -> None:
        with pytest.raises(Exception):
            DriveState(drive=Drive.CARE, satisfaction=-0.5)


class TestGoalModel:
    def test_default_goal(self) -> None:
        g = Goal(drive=Drive.SEEKING, description="test")
        assert g.stake == 0.3
        assert g.status == GoalStatus.ACTIVE
        assert g.progress == 0.0

    def test_max_goals_constant(self) -> None:
        assert MAX_ACTIVE_GOALS == 3


# ===============================================================
# Test Base Intensity
# ===============================================================

class TestBaseIntensity:
    def test_seeking_uses_openness(self) -> None:
        p = _personality(openness=0.9)
        base = compute_base_intensity(Drive.SEEKING, p)
        assert base == pytest.approx(0.72, abs=0.01)  # 0.9 * 0.8

    def test_care_uses_agreeableness(self) -> None:
        p = _personality(agreeableness=0.8)
        base = compute_base_intensity(Drive.CARE, p)
        assert base == pytest.approx(0.64, abs=0.01)

    def test_play_uses_extraversion(self) -> None:
        p = _personality(extraversion=0.7)
        base = compute_base_intensity(Drive.PLAY, p)
        assert base == pytest.approx(0.42, abs=0.01)

    def test_panic_grief_uses_neuroticism_times_rapport(self) -> None:
        p = _personality(neuroticism=0.8)
        base = compute_base_intensity(Drive.PANIC_GRIEF, p, rapport=0.9)
        assert base == pytest.approx(0.36, abs=0.01)  # 0.9 * 0.8 * 0.5

    def test_panic_grief_low_rapport(self) -> None:
        p = _personality(neuroticism=0.8)
        base = compute_base_intensity(Drive.PANIC_GRIEF, p, rapport=0.1)
        assert base < 0.1

    def test_clamped_to_1(self) -> None:
        p = _personality(openness=1.0)
        base = compute_base_intensity(Drive.SEEKING, p)
        assert base <= 1.0


# ===============================================================
# Test Urgency
# ===============================================================

class TestUrgency:
    def test_zero_when_satisfied(self) -> None:
        ds = _drive_state(satisfaction=1.0)
        urgency = compute_urgency(ds, current_turn=0)
        assert urgency == 0.0

    def test_grows_with_time(self) -> None:
        ds = _drive_state(intensity=0.8, satisfaction=0.2, last_satisfied_turn=0)
        u1 = compute_urgency(ds, current_turn=2)
        u2 = compute_urgency(ds, current_turn=8)
        assert u2 > u1

    def test_max_urgency_clamped(self) -> None:
        ds = _drive_state(intensity=1.0, satisfaction=0.0, last_satisfied_turn=0)
        urgency = compute_urgency(ds, current_turn=100)
        assert urgency <= 1.0

    def test_intensity_scales_urgency(self) -> None:
        ds_low = _drive_state(intensity=0.2, satisfaction=0.3)
        ds_high = _drive_state(intensity=0.9, satisfaction=0.3)
        assert compute_urgency(ds_high, 5) > compute_urgency(ds_low, 5)


class TestFrustration:
    def test_not_frustrated_early(self) -> None:
        ds = _drive_state(last_satisfied_turn=5)
        assert not is_frustrated(ds, current_turn=10)

    def test_frustrated_after_threshold(self) -> None:
        ds = _drive_state(last_satisfied_turn=0)
        assert is_frustrated(ds, current_turn=FRUSTRATION_THRESHOLD_TURNS)

    def test_frustrated_well_past_threshold(self) -> None:
        ds = _drive_state(last_satisfied_turn=0)
        assert is_frustrated(ds, current_turn=50)


# ===============================================================
# Test Keyword Score
# ===============================================================

class TestKeywordScore:
    def test_no_match(self) -> None:
        assert _keyword_score("hello world", ["explore", "curious"]) == 0.0

    def test_match(self) -> None:
        score = _keyword_score("I am curious about this", ["curious", "explore"])
        assert score > 0.0

    def test_multiple_matches_higher_than_single(self) -> None:
        keywords = ["curious", "explore", "discover", "learn", "wonder",
                     "interesting", "concept", "investigate", "deep", "theory"]
        s1 = _keyword_score("curious explore discover", keywords)
        s2 = _keyword_score("curious", keywords)
        assert s1 > s2

    def test_capped_at_1(self) -> None:
        score = _keyword_score(
            "curious explore discover learn wonder question investigate",
            ["curious", "explore"],
        )
        assert score <= 1.0

    def test_empty_keywords(self) -> None:
        assert _keyword_score("hello", []) == 0.0


# ===============================================================
# Test Drive Activation
# ===============================================================

class TestDriveActivation:
    def test_seeking_activated_by_curiosity(self) -> None:
        p = _personality(openness=0.8)
        activation = compute_drive_activation(
            Drive.SEEKING, "why does this happen? I'm curious", p,
        )
        assert activation > 0.3

    def test_care_activated_by_distress(self) -> None:
        p = _personality(agreeableness=0.8)
        activation = compute_drive_activation(
            Drive.CARE, "I'm struggling and need help", p,
        )
        assert activation > 0.3

    def test_play_activated_by_fun(self) -> None:
        p = _personality(extraversion=0.8)
        activation = compute_drive_activation(
            Drive.PLAY, "let's play a fun game", p,
        )
        assert activation > 0.3

    def test_neutral_stimulus_low_activation(self) -> None:
        p = _personality()
        activation = compute_drive_activation(
            Drive.SEEKING, "the weather is nice today", p,
        )
        assert activation < 0.4

    def test_personality_modulates(self) -> None:
        p_high = _personality(openness=0.9)
        p_low = _personality(openness=0.1)
        a_high = compute_drive_activation(Drive.SEEKING, "hello", p_high)
        a_low = compute_drive_activation(Drive.SEEKING, "hello", p_low)
        assert a_high > a_low


# ===============================================================
# Test Satisfaction Signal
# ===============================================================

class TestSatisfactionSignal:
    def test_seeking_satisfied_by_insight(self) -> None:
        signal = compute_satisfaction_signal(Drive.SEEKING, "wow that's fascinating insight")
        assert signal > 0.0

    def test_care_satisfied_by_thanks(self) -> None:
        signal = compute_satisfaction_signal(Drive.CARE, "thank you so much, that's very helpful")
        assert signal > 0.0

    def test_play_satisfied_by_laughter(self) -> None:
        signal = compute_satisfaction_signal(Drive.PLAY, "haha that's funny love it")
        assert signal > 0.0

    def test_no_signal_neutral(self) -> None:
        signal = compute_satisfaction_signal(Drive.SEEKING, "the sky is blue")
        assert signal == 0.0


# ===============================================================
# Test Update Single Drive
# ===============================================================

class TestUpdateSingleDrive:
    def test_activation_increases_intensity(self) -> None:
        ds = _drive_state(Drive.SEEKING, intensity=0.3)
        p = _personality(openness=0.8)
        new_ds, update = update_single_drive(ds, "why does this happen?", p, 1)
        assert new_ds.intensity > ds.intensity

    def test_satisfaction_signal_updates_satisfaction(self) -> None:
        ds = _drive_state(Drive.CARE, satisfaction=0.3)
        p = _personality()
        new_ds, _ = update_single_drive(ds, "thank you so much", p, 1)
        assert new_ds.satisfaction > ds.satisfaction

    def test_satisfaction_decays_without_signal(self) -> None:
        ds = _drive_state(Drive.SEEKING, satisfaction=0.5)
        p = _personality()
        new_ds, _ = update_single_drive(ds, "the weather", p, 1)
        assert new_ds.satisfaction < ds.satisfaction

    def test_update_returns_deltas(self) -> None:
        ds = _drive_state(Drive.SEEKING)
        p = _personality()
        _, update = update_single_drive(ds, "hello", p, 1)
        assert update.drive == Drive.SEEKING
        assert isinstance(update.triggered, bool)
        assert isinstance(update.frustration, bool)


# ===============================================================
# Test Update All Drives
# ===============================================================

class TestUpdateDrives:
    def test_disabled_returns_unchanged(self) -> None:
        state = default_drives_state()  # disabled
        p = _personality()
        new_state, updates = update_drives(state, "hello", p, 1)
        assert new_state.enabled is False
        assert updates == []

    def test_enabled_returns_updates(self) -> None:
        state = _enabled_drives_state()
        p = _personality()
        new_state, updates = update_drives(state, "I'm curious why", p, 1)
        assert new_state.enabled is True
        assert len(updates) == 4  # One per drive

    def test_seeking_most_activated_by_curiosity(self) -> None:
        state = _enabled_drives_state()
        p = _personality(openness=0.9)
        _, updates = update_drives(state, "why how explore curious", p, 1)
        seeking_update = next(u for u in updates if u.drive == Drive.SEEKING)
        other_updates = [u for u in updates if u.drive != Drive.SEEKING]
        assert seeking_update.new_intensity > max(u.new_intensity for u in other_updates)


# ===============================================================
# Test Goal Generation
# ===============================================================

class TestGoalGeneration:
    def test_no_goal_below_urgency_threshold(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(urgency=0.1)
        goal = generate_goal(Drive.SEEKING, ds, "test", 1, [])
        assert goal is None

    def test_generates_goal_above_threshold(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(urgency=0.6)
        goal = generate_goal(Drive.SEEKING, ds, "explore this", 1, [])
        assert goal is not None
        assert goal.drive == Drive.SEEKING
        assert goal.status == GoalStatus.ACTIVE

    def test_max_3_goals(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(urgency=0.9)
        existing = [
            Goal(drive=Drive.SEEKING, description="g1"),
            Goal(drive=Drive.CARE, description="g2"),
            Goal(drive=Drive.PLAY, description="g3"),
        ]
        goal = generate_goal(Drive.PANIC_GRIEF, ds, "test", 1, existing)
        assert goal is None

    def test_no_duplicate_drive_goal(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(Drive.SEEKING, urgency=0.9)
        existing = [Goal(drive=Drive.SEEKING, description="already active")]
        goal = generate_goal(Drive.SEEKING, ds, "test", 1, existing)
        assert goal is None

    def test_panic_grief_low_stake(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(Drive.PANIC_GRIEF, urgency=0.9)
        goal = generate_goal(Drive.PANIC_GRIEF, ds, "goodbye", 1, [])
        assert goal is not None
        assert goal.stake <= 0.4  # Ethical limit

    def test_goal_has_deadline(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(urgency=0.7)
        goal = generate_goal(Drive.SEEKING, ds, "explore", 5, [])
        assert goal is not None
        assert goal.deadline_turns > 0
        assert goal.created_turn == 5


# ===============================================================
# Test Goal Progress
# ===============================================================

class TestGoalProgress:
    def test_progress_increases_with_satisfaction(self) -> None:
        goal = Goal(drive=Drive.SEEKING, description="explore", stake=0.5)
        updated = evaluate_goal_progress(goal, "wow fascinating insight", 1)
        assert updated.progress > 0.0

    def test_no_progress_irrelevant_stimulus(self) -> None:
        goal = Goal(drive=Drive.SEEKING, description="explore", stake=0.5)
        updated = evaluate_goal_progress(goal, "the weather is nice", 1)
        assert updated.progress == 0.0

    def test_goal_completes_at_high_progress(self) -> None:
        goal = Goal(drive=Drive.CARE, description="help", stake=0.5, progress=0.85)
        updated = evaluate_goal_progress(goal, "thank you so much helpful great", 1)
        assert updated.status == GoalStatus.COMPLETED

    def test_goal_fails_at_deadline(self) -> None:
        goal = Goal(
            drive=Drive.PLAY, description="play", stake=0.3,
            created_turn=0, deadline_turns=5,
        )
        updated = evaluate_goal_progress(goal, "serious topic", 5)
        assert updated.status == GoalStatus.FAILED

    def test_inactive_goal_unchanged(self) -> None:
        goal = Goal(
            drive=Drive.SEEKING, description="done",
            status=GoalStatus.COMPLETED,
        )
        updated = evaluate_goal_progress(goal, "anything", 10)
        assert updated.status == GoalStatus.COMPLETED


# ===============================================================
# Test Goal Resolution (Emotional Impact)
# ===============================================================

class TestGoalResolution:
    def test_success_positive_valence(self) -> None:
        goal = Goal(drive=Drive.SEEKING, description="explore", stake=0.6)
        impact = resolve_goal(goal, GoalOutcome.SUCCESS)
        assert impact.valence_delta > 0
        assert impact.emotion_tag in ("joy", "contentment")

    def test_failure_negative_valence(self) -> None:
        goal = Goal(drive=Drive.CARE, description="help", stake=0.7)
        impact = resolve_goal(goal, GoalOutcome.FAILURE)
        assert impact.valence_delta < 0
        assert impact.emotion_tag in ("disappointment", "frustration")

    def test_partial_mild_positive(self) -> None:
        goal = Goal(drive=Drive.PLAY, description="play", stake=0.5)
        impact = resolve_goal(goal, GoalOutcome.PARTIAL)
        assert impact.valence_delta > 0
        assert impact.valence_delta < resolve_goal(goal, GoalOutcome.SUCCESS).valence_delta

    def test_high_stake_stronger_impact(self) -> None:
        g_high = Goal(drive=Drive.SEEKING, description="x", stake=0.9)
        g_low = Goal(drive=Drive.SEEKING, description="x", stake=0.2)
        i_high = resolve_goal(g_high, GoalOutcome.SUCCESS)
        i_low = resolve_goal(g_low, GoalOutcome.SUCCESS)
        assert abs(i_high.valence_delta) > abs(i_low.valence_delta)

    def test_high_stake_failure_frustration(self) -> None:
        goal = Goal(drive=Drive.SEEKING, description="x", stake=0.8)
        impact = resolve_goal(goal, GoalOutcome.FAILURE)
        assert impact.emotion_tag == "frustration"

    def test_low_stake_success_contentment(self) -> None:
        goal = Goal(drive=Drive.SEEKING, description="x", stake=0.3)
        impact = resolve_goal(goal, GoalOutcome.SUCCESS)
        assert impact.emotion_tag == "contentment"


# ===============================================================
# Test Process Goals
# ===============================================================

class TestProcessGoals:
    def test_disabled_noop(self) -> None:
        state = default_drives_state()
        new_state, impacts = process_goals(state, "hello", 1)
        assert impacts == []

    def test_completes_goal(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(drive=Drive.CARE, description="help", stake=0.5, progress=0.85),
        ]
        new_state, impacts = process_goals(state, "thank you helpful great", 1)
        assert len(new_state.active_goals) == 0
        assert len(new_state.resolved_goals) == 1
        assert len(impacts) == 1
        assert impacts[0].valence_delta > 0

    def test_fails_expired_goal(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(
                drive=Drive.PLAY, description="play", stake=0.4,
                created_turn=0, deadline_turns=5,
            ),
        ]
        new_state, impacts = process_goals(state, "nothing", 5)
        assert len(new_state.active_goals) == 0
        assert len(impacts) == 1
        assert impacts[0].valence_delta < 0

    def test_keeps_active_goals(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(drive=Drive.SEEKING, description="explore", stake=0.5,
                 created_turn=0, deadline_turns=10),
        ]
        new_state, impacts = process_goals(state, "hello", 3)
        assert len(new_state.active_goals) == 1
        assert impacts == []


# ===============================================================
# Test Attempt Goal Generation
# ===============================================================

class TestAttemptGoalGeneration:
    def test_disabled_noop(self) -> None:
        state = default_drives_state()
        result = attempt_goal_generation(state, "hello", 1)
        assert result.active_goals == []

    def test_generates_from_urgent_drive(self) -> None:
        state = _enabled_drives_state(
            seeking=DriveState(
                drive=Drive.SEEKING, intensity=0.8, satisfaction=0.1,
                urgency=0.7,
            ),
        )
        result = attempt_goal_generation(state, "explore this", 5)
        assert len(result.active_goals) == 1
        assert result.active_goals[0].drive == Drive.SEEKING

    def test_respects_max_goals(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(drive=Drive.SEEKING, description="g1"),
            Goal(drive=Drive.CARE, description="g2"),
            Goal(drive=Drive.PLAY, description="g3"),
        ]
        for name, ds in state.drives.items():
            state.drives[name] = DriveState(
                drive=ds.drive, intensity=0.9, satisfaction=0.1, urgency=0.9,
            )
        result = attempt_goal_generation(state, "test", 1)
        assert len(result.active_goals) == 3  # No new ones


# ===============================================================
# Test Prompt Generation
# ===============================================================

class TestDrivesPrompt:
    def test_disabled_returns_none(self) -> None:
        state = default_drives_state()
        assert get_drives_prompt(state) is None

    def test_includes_active_drives(self) -> None:
        state = _enabled_drives_state(
            seeking=DriveState(
                drive=Drive.SEEKING, intensity=0.7, satisfaction=0.2, urgency=0.5,
            ),
        )
        prompt = get_drives_prompt(state)
        assert prompt is not None
        assert "SEEKING" in prompt
        assert "MOTIVATIONAL DRIVES" in prompt

    def test_includes_active_goals(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(drive=Drive.CARE, description="Help the user", stake=0.5),
        ]
        # Need at least one drive with urgency > 0.2 OR a goal
        prompt = get_drives_prompt(state)
        assert prompt is not None
        assert "Help the user" in prompt

    def test_no_prompt_if_nothing_active(self) -> None:
        state = _enabled_drives_state()
        # All drives have default urgency 0
        prompt = get_drives_prompt(state)
        assert prompt is None

    def test_shows_frustrated_drives(self) -> None:
        state = _enabled_drives_state(
            care=DriveState(
                drive=Drive.CARE, intensity=0.7, satisfaction=0.1, urgency=0.5,
            ),
        )
        prompt = get_drives_prompt(state)
        assert prompt is not None
        assert "FRUSTRATED" in prompt or "CARE" in prompt


# ===============================================================
# Test Details for Research
# ===============================================================

class TestDrivesDetails:
    def test_disabled_minimal(self) -> None:
        state = default_drives_state()
        details = get_drives_details(state)
        assert details["enabled"] is False
        assert details["dominant_drive"] is None

    def test_enabled_has_drives(self) -> None:
        state = _enabled_drives_state(
            seeking=DriveState(
                drive=Drive.SEEKING, intensity=0.8, urgency=0.6,
            ),
        )
        details = get_drives_details(state)
        assert details["enabled"] is True
        assert "seeking" in details["drives"]
        assert details["dominant_drive"] == "seeking"

    def test_includes_goals(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(drive=Drive.PLAY, description="play around", stake=0.3),
        ]
        details = get_drives_details(state)
        assert len(details["active_goals"]) == 1
        assert details["active_goals"][0]["drive"] == "play"

    def test_includes_updates_and_impacts(self) -> None:
        state = _enabled_drives_state()
        from pathos.models.drives import DriveUpdate
        updates = [DriveUpdate(
            drive=Drive.SEEKING,
            previous_intensity=0.3, new_intensity=0.5,
            satisfaction_delta=0.0, urgency=0.4, triggered=True,
        )]
        impacts = [EmotionalImpact(
            valence_delta=0.2, emotion_tag="joy", description="Goal achieved",
        )]
        details = get_drives_details(state, updates, impacts)
        assert len(details["updates"]) == 1
        assert details["updates"][0]["triggered"] is True
        assert len(details["emotional_impacts"]) == 1

    def test_frustration_level(self) -> None:
        state = _enabled_drives_state(
            seeking=DriveState(
                drive=Drive.SEEKING, intensity=0.8, satisfaction=0.1,
            ),
            care=DriveState(
                drive=Drive.CARE, intensity=0.6, satisfaction=0.2,
            ),
        )
        details = get_drives_details(state)
        assert details["frustration_level"] > 0


# ===============================================================
# Test Ethical Limits
# ===============================================================

class TestEthicalLimits:
    """PANIC_GRIEF nunca debe generar comportamiento manipulativo."""

    def test_panic_grief_stake_always_capped(self) -> None:
        from pathos.engine.drives import generate_goal
        ds = _drive_state(Drive.PANIC_GRIEF, urgency=1.0)
        goal = generate_goal(Drive.PANIC_GRIEF, ds, "goodbye forever", 1, [])
        assert goal is not None
        assert goal.stake <= 0.4

    def test_panic_grief_description_not_manipulative(self) -> None:
        from pathos.engine.drives import _generate_goal_description
        desc = _generate_goal_description(Drive.PANIC_GRIEF, "leaving")
        assert "stay" not in desc.lower()
        assert "don't go" not in desc.lower()
        assert "connection" in desc.lower() or "continuity" in desc.lower()


# ===============================================================
# Test Serialization
# ===============================================================

class TestSerialization:
    def test_roundtrip_default(self) -> None:
        state = default_drives_state()
        data = state.model_dump()
        restored = DrivesState(**data)
        assert restored.enabled == state.enabled
        assert len(restored.drives) == 4

    def test_roundtrip_with_goals(self) -> None:
        state = _enabled_drives_state()
        state.active_goals = [
            Goal(drive=Drive.SEEKING, description="explore", stake=0.5),
        ]
        data = state.model_dump()
        restored = DrivesState(**data)
        assert len(restored.active_goals) == 1
        assert restored.active_goals[0].drive == Drive.SEEKING

    def test_roundtrip_preserves_drives(self) -> None:
        state = _enabled_drives_state(
            seeking=DriveState(
                drive=Drive.SEEKING, intensity=0.9, satisfaction=0.1,
                urgency=0.7, activation_count=5,
            ),
        )
        data = state.model_dump()
        restored = DrivesState(**data)
        assert restored.drives["seeking"].intensity == 0.9
        assert restored.drives["seeking"].activation_count == 5


# ===============================================================
# Test Full Flow
# ===============================================================

class TestFullFlow:
    def test_multi_turn_flow(self) -> None:
        """Simulates several turns of drive interaction."""
        p = _personality(openness=0.9, agreeableness=0.8)
        state = _enabled_drives_state()

        # Turn 1: curiosity stimulus
        state, updates = update_drives(state, "why does consciousness work?", p, 1)
        state = attempt_goal_generation(state, "why does consciousness work?", 1)
        # SEEKING should be most activated
        seeking_update = next(u for u in updates if u.drive == Drive.SEEKING)
        assert seeking_update.new_intensity > 0.3

        # Turn 2: more curiosity, might generate goal
        state, _ = update_drives(state, "tell me more about qualia", p, 2)
        state = attempt_goal_generation(state, "tell me more about qualia", 2)

        # Turn 3: satisfaction signal
        state, updates = update_drives(state, "wow fascinating insight, makes sense!", p, 3)
        state, impacts = process_goals(state, "wow fascinating insight, makes sense!", 3)

        # Drive should be more satisfied now
        seeking = state.drives["seeking"]
        assert seeking.satisfaction > 0.3

    def test_care_distress_flow(self) -> None:
        """User in distress activates CARE drive."""
        p = _personality(agreeableness=0.9)
        state = _enabled_drives_state()

        # User distress
        state, updates = update_drives(state, "I'm struggling and need help please", p, 1)
        care_update = next(u for u in updates if u.drive == Drive.CARE)
        assert care_update.new_intensity > 0.3
        assert care_update.triggered is True

    def test_disabled_flow_noop(self) -> None:
        """When disabled, nothing changes."""
        p = _personality()
        state = default_drives_state()  # disabled

        state, updates = update_drives(state, "why does this happen?", p, 1)
        assert updates == []
        state = attempt_goal_generation(state, "test", 1)
        assert state.active_goals == []
        state, impacts = process_goals(state, "test", 1)
        assert impacts == []


# ===============================================================
# Test Values Always Clamped
# ===============================================================

class TestValuesClamped:
    def test_intensity_clamped(self) -> None:
        ds = _drive_state(intensity=0.95)
        p = _personality(openness=1.0)
        new_ds, _ = update_single_drive(
            ds, "why how explore curious question investigate discover",
            p, 1,
        )
        assert 0 <= new_ds.intensity <= 1

    def test_satisfaction_clamped(self) -> None:
        ds = _drive_state(satisfaction=0.95)
        p = _personality()
        new_ds, _ = update_single_drive(
            ds, "thank you amazing fascinating perfect",
            p, 1,
        )
        assert 0 <= new_ds.satisfaction <= 1

    def test_urgency_clamped(self) -> None:
        ds = _drive_state(intensity=1.0, satisfaction=0.0, last_satisfied_turn=0)
        u = compute_urgency(ds, 1000)
        assert 0 <= u <= 1

    def test_goal_progress_clamped(self) -> None:
        goal = Goal(drive=Drive.CARE, description="help", progress=0.95)
        updated = evaluate_goal_progress(
            goal, "thank you helpful great perfect appreciate", 1,
        )
        assert 0 <= updated.progress <= 1
