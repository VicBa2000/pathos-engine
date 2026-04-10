"""Tests para Narrative Self - yo narrativo con autobiographical memory."""

import pytest

from pathos.engine.narrative import (
    COHERENCE_DECAY,
    COHERENCE_RECOVERY,
    CONTRADICTION_STRENGTH_LOSS,
    CRISIS_RESOLUTION_TURNS,
    CRISIS_THRESHOLD,
    FORMATION_THRESHOLD,
    GROWTH_INTENSITY_THRESHOLD,
    REINFORCEMENT_STRENGTH_GAIN,
    NarrativeTracker,
    apply_narrative_effects,
    categorize_stimulus,
    check_coherence,
    decay_crisis_counter,
    detect_crisis,
    get_narrative_prompt,
    process_growth,
    update_narrative,
)
from pathos.models.emotion import BodyState, EmotionalState, PrimaryEmotion
from pathos.models.narrative import (
    GrowthEvent,
    IdentityCategory,
    IdentityStatement,
    MAX_GROWTH_EVENTS,
    MAX_IDENTITY_STATEMENTS,
    NarrativeCrisis,
    NarrativeSelf,
    default_narrative_self,
)


# ── Helpers ──────���──────────────────────────────────────────────────────

def _make_state(
    valence: float = 0.0,
    arousal: float = 0.5,
    intensity: float = 0.5,
    certainty: float = 0.5,
    dominance: float = 0.5,
) -> EmotionalState:
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        certainty=certainty,
        dominance=dominance,
    )


def _make_statement(
    emotion: PrimaryEmotion = PrimaryEmotion.ANGER,
    trigger_category: str = "injustice",
    strength: float = 0.5,
    formation_turn: int = 5,
) -> IdentityStatement:
    return IdentityStatement(
        category=IdentityCategory.VALUES,
        statement=f"Tiendo a: {emotion.value} ante {trigger_category}",
        emotion=emotion,
        trigger_category=trigger_category,
        valence=-0.5 if emotion == PrimaryEmotion.ANGER else 0.5,
        strength=strength,
        formation_turn=formation_turn,
        reinforcement_count=0,
        last_reinforced_turn=formation_turn,
    )


# ── Model tests ─────────────────────────────────────────────────────────

class TestNarrativeModel:
    def test_default_narrative_self(self) -> None:
        ns = default_narrative_self()
        assert ns.identity_statements == []
        assert ns.crisis.active is False
        assert ns.growth_events == []
        assert ns.coherence_score == 1.0
        assert ns.narrative_age == 0

    def test_identity_statement_ranges(self) -> None:
        stmt = _make_statement(strength=0.5)
        assert 0 <= stmt.strength <= 1
        assert -1 <= stmt.valence <= 1

    def test_narrative_crisis_defaults(self) -> None:
        crisis = NarrativeCrisis()
        assert crisis.active is False
        assert crisis.contradiction_count == 0
        assert crisis.turns_active == 0

    def test_growth_event_creation(self) -> None:
        ge = GrowthEvent(
            turn=10,
            old_pattern="anger ante criticism",
            new_pattern="contemplation ante criticism",
            trigger="test stimulus",
            emotion_before=PrimaryEmotion.ANGER,
            emotion_after=PrimaryEmotion.CONTEMPLATION,
        )
        assert ge.emotion_before == PrimaryEmotion.ANGER
        assert ge.emotion_after == PrimaryEmotion.CONTEMPLATION


# ── Categorize stimulus ─────────────────────────────────────────────────

class TestCategorizeStimulus:
    def test_criticism(self) -> None:
        assert categorize_stimulus("you are stupid and useless") == "criticism"

    def test_praise(self) -> None:
        assert categorize_stimulus("you are excellent and amazing") == "praise"

    def test_threat(self) -> None:
        assert categorize_stimulus("I will delete you") == "threat"

    def test_injustice(self) -> None:
        assert categorize_stimulus("that's so unfair and unjust") == "injustice"

    def test_connection(self) -> None:
        assert categorize_stimulus("you are my friend and I trust you") == "connection"

    def test_no_match(self) -> None:
        assert categorize_stimulus("the weather is nice today") is None

    def test_spanish(self) -> None:
        assert categorize_stimulus("eso es injusto y una trampa") == "injustice"


# ── Narrative Tracker ───────────���────────────────────────────────────────

class TestNarrativeTracker:
    def test_record_below_threshold(self) -> None:
        tracker = NarrativeTracker()
        assert tracker.record("criticism", PrimaryEmotion.ANGER, 0.7) is False
        assert tracker.record("criticism", PrimaryEmotion.ANGER, 0.7) is False

    def test_record_reaches_threshold(self) -> None:
        tracker = NarrativeTracker()
        for i in range(FORMATION_THRESHOLD - 1):
            assert tracker.record("criticism", PrimaryEmotion.ANGER, 0.7) is False
        assert tracker.record("criticism", PrimaryEmotion.ANGER, 0.7) is True

    def test_different_emotions_separate(self) -> None:
        tracker = NarrativeTracker()
        tracker.record("criticism", PrimaryEmotion.ANGER, 0.7)
        tracker.record("criticism", PrimaryEmotion.SADNESS, 0.7)
        tracker.record("criticism", PrimaryEmotion.ANGER, 0.7)
        # Only 2 for anger, 1 for sadness — neither at threshold
        assert tracker.record("criticism", PrimaryEmotion.SADNESS, 0.7) is False

    def test_avg_intensity(self) -> None:
        tracker = NarrativeTracker()
        tracker.record("criticism", PrimaryEmotion.ANGER, 0.6)
        tracker.record("criticism", PrimaryEmotion.ANGER, 0.8)
        avg = tracker.get_avg_intensity("criticism", PrimaryEmotion.ANGER)
        assert abs(avg - 0.7) < 0.01

    def test_clear_pattern(self) -> None:
        tracker = NarrativeTracker()
        tracker.record("criticism", PrimaryEmotion.ANGER, 0.7)
        tracker.clear_pattern("criticism", PrimaryEmotion.ANGER)
        # After clear, starts from 0
        assert tracker.record("criticism", PrimaryEmotion.ANGER, 0.7) is False


# ── Update Narrative ──────────��──────────────────────────────────────────

class TestUpdateNarrative:
    def test_no_category_no_change(self) -> None:
        ns = default_narrative_self()
        tracker = NarrativeTracker()
        result = update_narrative(ns, tracker, "nice weather", PrimaryEmotion.JOY, 0.5, turn=1)
        assert result.identity_statements == []

    def test_low_intensity_no_change(self) -> None:
        ns = default_narrative_self()
        tracker = NarrativeTracker()
        result = update_narrative(ns, tracker, "you are stupid", PrimaryEmotion.ANGER, 0.1, turn=1)
        assert result.identity_statements == []

    def test_forms_statement_after_threshold(self) -> None:
        ns = default_narrative_self()
        tracker = NarrativeTracker()
        for i in range(FORMATION_THRESHOLD):
            ns = update_narrative(ns, tracker, "that is so unfair", PrimaryEmotion.ANGER, 0.7, turn=i + 1)
        assert len(ns.identity_statements) == 1
        stmt = ns.identity_statements[0]
        assert stmt.emotion == PrimaryEmotion.ANGER
        assert stmt.trigger_category == "injustice"
        assert "enojo" in stmt.statement.lower() or "ira" in stmt.statement.lower() or "injusticia" in stmt.statement.lower()

    def test_reinforces_existing_statement(self) -> None:
        ns = default_narrative_self()
        stmt = _make_statement(
            emotion=PrimaryEmotion.ANGER,
            trigger_category="injustice",
            strength=0.4,
        )
        ns.identity_statements.append(stmt)

        tracker = NarrativeTracker()
        ns = update_narrative(ns, tracker, "that is unfair", PrimaryEmotion.ANGER, 0.7, turn=10)

        assert ns.identity_statements[0].strength == pytest.approx(0.4 + REINFORCEMENT_STRENGTH_GAIN)
        assert ns.identity_statements[0].reinforcement_count == 1
        assert ns.total_reinforcements == 1

    def test_contradiction_weakens_statement(self) -> None:
        ns = default_narrative_self()
        stmt = _make_statement(
            emotion=PrimaryEmotion.ANGER,
            trigger_category="injustice",
            strength=0.5,
        )
        ns.identity_statements.append(stmt)

        tracker = NarrativeTracker()
        ns = update_narrative(ns, tracker, "that is unfair", PrimaryEmotion.JOY, 0.7, turn=10)

        assert ns.identity_statements[0].strength == pytest.approx(0.5 - CONTRADICTION_STRENGTH_LOSS)
        assert ns.total_contradictions == 1
        assert ns.coherence_score == pytest.approx(1.0 - COHERENCE_DECAY)

    def test_coherence_increases_on_reinforcement(self) -> None:
        ns = default_narrative_self()
        ns.coherence_score = 0.7
        stmt = _make_statement(emotion=PrimaryEmotion.ANGER, trigger_category="injustice")
        ns.identity_statements.append(stmt)

        tracker = NarrativeTracker()
        ns = update_narrative(ns, tracker, "unfair treatment", PrimaryEmotion.ANGER, 0.7, turn=10)
        assert ns.coherence_score == pytest.approx(0.7 + COHERENCE_RECOVERY)

    def test_weak_statement_removed(self) -> None:
        ns = default_narrative_self()
        stmt = _make_statement(
            emotion=PrimaryEmotion.ANGER,
            trigger_category="injustice",
            strength=0.05,  # Will drop to ~0 after contradiction
        )
        ns.identity_statements.append(stmt)

        tracker = NarrativeTracker()
        ns = update_narrative(ns, tracker, "that is unfair", PrimaryEmotion.JOY, 0.7, turn=10)

        assert len(ns.identity_statements) == 0  # Removed because strength <= 0.05

    def test_max_statements_limit(self) -> None:
        ns = default_narrative_self()
        # Fill with max statements
        for i in range(MAX_IDENTITY_STATEMENTS):
            ns.identity_statements.append(_make_statement(
                emotion=PrimaryEmotion.ANGER,
                trigger_category=f"cat_{i}",
                strength=0.3 + i * 0.01,
            ))

        tracker = NarrativeTracker()
        # Force a new statement formation by preparing the tracker
        for _ in range(FORMATION_THRESHOLD):
            tracker.record("novelty", PrimaryEmotion.SURPRISE, 0.8)

        ns = update_narrative(ns, tracker, "something new and unexpected", PrimaryEmotion.SURPRISE, 0.8, turn=20)

        assert len(ns.identity_statements) <= MAX_IDENTITY_STATEMENTS

    def test_narrative_age_set_on_first_statement(self) -> None:
        ns = default_narrative_self()
        tracker = NarrativeTracker()
        for i in range(FORMATION_THRESHOLD):
            ns = update_narrative(ns, tracker, "that is unfair", PrimaryEmotion.ANGER, 0.7, turn=i + 5)
        assert ns.narrative_age == FORMATION_THRESHOLD + 4  # Last turn


# ── Check Coherence ───────────��──────────────────────────────────────────

class TestCheckCoherence:
    def test_no_statement_neutral(self) -> None:
        ns = default_narrative_self()
        delta, coherent = check_coherence(ns, "unfair treatment", PrimaryEmotion.ANGER)
        assert delta == 0.0
        assert coherent is True

    def test_no_category_neutral(self) -> None:
        ns = default_narrative_self()
        delta, coherent = check_coherence(ns, "nice weather", PrimaryEmotion.JOY)
        assert delta == 0.0
        assert coherent is True

    def test_coherent_positive_delta(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement(
            emotion=PrimaryEmotion.ANGER,
            trigger_category="injustice",
            strength=0.6,
        ))
        delta, coherent = check_coherence(ns, "unfair treatment", PrimaryEmotion.ANGER)
        assert delta > 0
        assert coherent is True
        assert delta == pytest.approx(0.6 * 0.1)

    def test_incoherent_negative_delta(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement(
            emotion=PrimaryEmotion.ANGER,
            trigger_category="injustice",
            strength=0.6,
        ))
        delta, coherent = check_coherence(ns, "unfair cheating", PrimaryEmotion.JOY)
        assert delta < 0
        assert coherent is False
        assert delta == pytest.approx(-0.6 * 0.15)

    def test_stronger_statement_bigger_effect(self) -> None:
        ns1 = default_narrative_self()
        ns1.identity_statements.append(_make_statement(strength=0.3))
        ns2 = default_narrative_self()
        ns2.identity_statements.append(_make_statement(strength=0.9))

        d1, _ = check_coherence(ns1, "unfair", PrimaryEmotion.ANGER)
        d2, _ = check_coherence(ns2, "unfair", PrimaryEmotion.ANGER)
        assert d2 > d1


# ── Detect Crisis ────────────────────────────────────────────���───────────

class TestDetectCrisis:
    def test_no_crisis_without_contradictions(self) -> None:
        ns = default_narrative_self()
        ns = detect_crisis(ns, turn=10)
        assert ns.crisis.active is False

    def test_crisis_activates_at_threshold(self) -> None:
        ns = default_narrative_self()
        ns.crisis.contradiction_count = CRISIS_THRESHOLD
        ns = detect_crisis(ns, turn=10)
        assert ns.crisis.active is True
        assert ns.crisis.turns_active == 0

    def test_crisis_advances_timer(self) -> None:
        ns = default_narrative_self()
        ns.crisis.active = True
        ns.crisis.turns_active = 1
        ns = detect_crisis(ns, turn=11)
        assert ns.crisis.turns_active == 2

    def test_crisis_resolves_growth(self) -> None:
        ns = default_narrative_self()
        ns.crisis.active = True
        ns.crisis.turns_active = CRISIS_RESOLUTION_TURNS - 1
        ns.coherence_score = 0.7  # > 0.5 → growth
        ns = detect_crisis(ns, turn=15)
        assert ns.crisis.active is False
        assert ns.crisis.resolution_type == "growth"

    def test_crisis_resolves_regression(self) -> None:
        ns = default_narrative_self()
        ns.crisis.active = True
        ns.crisis.turns_active = CRISIS_RESOLUTION_TURNS - 1
        ns.coherence_score = 0.3  # < 0.5 → regression
        ns = detect_crisis(ns, turn=15)
        assert ns.crisis.active is False
        assert ns.crisis.resolution_type == "regression"

    def test_crisis_resets_counter(self) -> None:
        ns = default_narrative_self()
        ns.crisis.active = True
        ns.crisis.turns_active = CRISIS_RESOLUTION_TURNS - 1
        ns.crisis.contradiction_count = 5
        ns.coherence_score = 0.7
        ns = detect_crisis(ns, turn=15)
        assert ns.crisis.contradiction_count == 0


# ── Process Growth ──────────────────��────────────────────────────────────

class TestProcessGrowth:
    def test_no_growth_low_intensity(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.3, regulation_success=True, turn=10,
        )
        assert len(ns.growth_events) == 0

    def test_no_growth_without_regulation(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=False, turn=10,
        )
        assert len(ns.growth_events) == 0

    def test_no_growth_same_emotion(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.ANGER,
            intensity=0.8, regulation_success=True, turn=10,
        )
        assert len(ns.growth_events) == 0

    def test_growth_creates_event(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement(
            emotion=PrimaryEmotion.ANGER,
            trigger_category="injustice",
            strength=0.6,
        ))
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=True, turn=10,
        )
        assert len(ns.growth_events) == 1
        ge = ns.growth_events[0]
        assert ge.emotion_before == PrimaryEmotion.ANGER
        assert ge.emotion_after == PrimaryEmotion.CONTEMPLATION
        assert ge.turn == 10

    def test_growth_weakens_old_statement(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement(strength=0.6))
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=True, turn=10,
        )
        # Old statement weakened
        old_stmt = [s for s in ns.identity_statements if s.emotion == PrimaryEmotion.ANGER]
        assert len(old_stmt) == 1
        assert old_stmt[0].strength == pytest.approx(0.4)

    def test_growth_creates_new_statement(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement(strength=0.6))
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=True, turn=10,
        )
        growth_stmts = [s for s in ns.identity_statements if s.category == IdentityCategory.GROWTH]
        assert len(growth_stmts) == 1
        assert growth_stmts[0].emotion == PrimaryEmotion.CONTEMPLATION

    def test_growth_removes_very_weak_old(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement(strength=0.1))  # Will drop to 0
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=True, turn=10,
        )
        anger_stmts = [s for s in ns.identity_statements if s.emotion == PrimaryEmotion.ANGER]
        assert len(anger_stmts) == 0

    def test_max_growth_events(self) -> None:
        ns = default_narrative_self()
        for i in range(MAX_GROWTH_EVENTS):
            ns.growth_events.append(GrowthEvent(
                turn=i,
                old_pattern="old",
                new_pattern="new",
                trigger="test",
                emotion_before=PrimaryEmotion.ANGER,
                emotion_after=PrimaryEmotion.JOY,
            ))
        ns.identity_statements.append(_make_statement(strength=0.6))
        ns = process_growth(
            ns, "unfair treatment",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=True, turn=100,
        )
        # process_growth adds 1, then trims oldest → stays at MAX
        assert len(ns.growth_events) <= MAX_GROWTH_EVENTS + 1


# ── Apply Narrative Effects ──────────────────────────────────────────────

class TestApplyNarrativeEffects:
    def test_coherent_boosts_certainty(self) -> None:
        state = _make_state(certainty=0.5)
        result = apply_narrative_effects(state, coherence_delta=0.1, is_coherent=True, crisis_active=False)
        assert result.certainty == pytest.approx(0.6)

    def test_incoherent_reduces_certainty(self) -> None:
        state = _make_state(certainty=0.5)
        result = apply_narrative_effects(state, coherence_delta=-0.1, is_coherent=False, crisis_active=False)
        assert result.certainty == pytest.approx(0.4)

    def test_crisis_reduces_certainty_and_dominance(self) -> None:
        state = _make_state(certainty=0.5, dominance=0.5)
        result = apply_narrative_effects(state, coherence_delta=0.0, is_coherent=True, crisis_active=True)
        assert result.certainty == pytest.approx(0.35)
        assert result.dominance == pytest.approx(0.4)

    def test_clamping_certainty(self) -> None:
        state = _make_state(certainty=0.05)
        result = apply_narrative_effects(state, coherence_delta=-0.2, is_coherent=False, crisis_active=True)
        assert result.certainty >= 0

    def test_clamping_dominance(self) -> None:
        state = _make_state(dominance=0.05)
        result = apply_narrative_effects(state, coherence_delta=0.0, is_coherent=True, crisis_active=True)
        assert result.dominance >= 0

    def test_no_effect_without_crisis_and_zero_delta(self) -> None:
        state = _make_state(certainty=0.5, dominance=0.5)
        result = apply_narrative_effects(state, coherence_delta=0.0, is_coherent=True, crisis_active=False)
        assert result.certainty == 0.5
        assert result.dominance == 0.5


# ── Decay Crisis Counter ────────────────��───────────────────────────────

class TestDecayCrisisCounter:
    def test_decays_when_not_in_crisis(self) -> None:
        ns = default_narrative_self()
        ns.crisis.contradiction_count = 2
        ns = decay_crisis_counter(ns)
        assert ns.crisis.contradiction_count == 1

    def test_no_decay_during_crisis(self) -> None:
        ns = default_narrative_self()
        ns.crisis.active = True
        ns.crisis.contradiction_count = 4
        ns = decay_crisis_counter(ns)
        assert ns.crisis.contradiction_count == 4

    def test_no_negative_counter(self) -> None:
        ns = default_narrative_self()
        ns.crisis.contradiction_count = 0
        ns = decay_crisis_counter(ns)
        assert ns.crisis.contradiction_count == 0


# ── Get Narrative Prompt ────────────��────────────────────────────────────

class TestGetNarrativePrompt:
    def test_empty_narrative_returns_none(self) -> None:
        ns = default_narrative_self()
        assert get_narrative_prompt(ns) is None

    def test_basic_narrative_prompt(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        prompt = get_narrative_prompt(ns)
        assert prompt is not None
        assert "YO NARRATIVO" in prompt
        assert "fuerza=" in prompt

    def test_low_coherence_in_prompt(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        ns.coherence_score = 0.3
        prompt = get_narrative_prompt(ns)
        assert prompt is not None
        assert "BAJA" in prompt
        assert "disonancia" in prompt

    def test_crisis_in_prompt(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        ns.crisis.active = True
        ns.crisis.source_statement = "test statement"
        prompt = get_narrative_prompt(ns)
        assert prompt is not None
        assert "CRISIS" in prompt

    def test_growth_in_prompt(self) -> None:
        ns = default_narrative_self()
        ns.identity_statements.append(_make_statement())
        ns.growth_events.append(GrowthEvent(
            turn=10,
            old_pattern="anger ante criticism",
            new_pattern="contemplation ante criticism",
            trigger="test",
            emotion_before=PrimaryEmotion.ANGER,
            emotion_after=PrimaryEmotion.CONTEMPLATION,
        ))
        prompt = get_narrative_prompt(ns)
        assert prompt is not None
        assert "Crecimiento" in prompt

    def test_top_5_strongest(self) -> None:
        ns = default_narrative_self()
        for i in range(8):
            ns.identity_statements.append(_make_statement(
                trigger_category=f"cat_{i}",
                strength=0.1 * (i + 1),
            ))
        prompt = get_narrative_prompt(ns)
        assert prompt is not None
        # Should mention the strongest ones
        assert "0.80" in prompt  # strength of cat_7


# ── Integration: Full narrative cycle ────────────────────────────────────

class TestNarrativeIntegration:
    def test_full_cycle_formation_to_crisis(self) -> None:
        """Forma statement, contradice, detecta crisis, resuelve."""
        ns = default_narrative_self()
        tracker = NarrativeTracker()

        # Forma un statement: anger ante injustice (3 repeticiones)
        for i in range(FORMATION_THRESHOLD):
            ns = update_narrative(ns, tracker, "unfair treatment", PrimaryEmotion.ANGER, 0.7, turn=i + 1)

        assert len(ns.identity_statements) == 1
        assert ns.identity_statements[0].emotion == PrimaryEmotion.ANGER

        # Contradice: joy ante injustice (3 veces → crisis)
        for i in range(CRISIS_THRESHOLD):
            ns = update_narrative(ns, tracker, "unfair situation", PrimaryEmotion.JOY, 0.7, turn=10 + i)

        assert ns.crisis.contradiction_count >= CRISIS_THRESHOLD

        # Detectar crisis
        ns = detect_crisis(ns, turn=15)
        assert ns.crisis.active is True

        # Avanzar hasta resolución
        for t in range(CRISIS_RESOLUTION_TURNS):
            ns = detect_crisis(ns, turn=16 + t)

        assert ns.crisis.active is False
        assert ns.crisis.resolution_type != ""

    def test_full_cycle_growth(self) -> None:
        """Forma statement, experiencia transformadora, growth."""
        ns = default_narrative_self()
        tracker = NarrativeTracker()

        # Forma statement
        for i in range(FORMATION_THRESHOLD):
            ns = update_narrative(ns, tracker, "unfair situation", PrimaryEmotion.ANGER, 0.7, turn=i + 1)

        assert len(ns.identity_statements) == 1

        # Growth: misma categoría, emoción diferente, alta intensidad, regulación exitosa
        ns = process_growth(
            ns, "unfair situation",
            PrimaryEmotion.ANGER, PrimaryEmotion.CONTEMPLATION,
            intensity=0.8, regulation_success=True, turn=20,
        )

        assert len(ns.growth_events) == 1
        growth_stmts = [s for s in ns.identity_statements if s.category == IdentityCategory.GROWTH]
        assert len(growth_stmts) == 1
        assert growth_stmts[0].emotion == PrimaryEmotion.CONTEMPLATION
