"""Tests for Computational Needs engine."""

import pytest

from pathos.engine.needs import (
    compute_needs_amplification,
    update_needs,
    _stimulus_relevance,
)
from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
)
from pathos.models.needs import ComputationalNeeds, default_needs


def _make_appraisal(
    significance: float = 0.5,
    valence: float = 0.0,
    novelty: float = 0.3,
    fairness: float = 0.0,
    control: float = 0.5,
) -> AppraisalVector:
    return AppraisalVector(
        relevance=RelevanceCheck(novelty=novelty, personal_significance=significance, values_affected=[]),
        valence=ValenceAssessment(goal_conduciveness=valence, value_alignment=valence, intrinsic_pleasantness=valence),
        coping=CopingPotential(control=control, power=control * 0.8, adjustability=0.5),
        agency=AgencyAttribution(responsible_agent="environment", intentionality=0.5, fairness=fairness),
        norms=NormCompatibility(internal_standards=0.0, external_standards=0.0, self_consistency=0.0),
    )


class TestDefaultNeeds:
    def test_default_values(self):
        needs = default_needs()
        assert needs.connection == 0.5
        assert needs.competence == 0.5
        assert needs.safety == 0.3

    def test_ranges(self):
        needs = default_needs()
        for field in ["connection", "competence", "autonomy", "coherence", "stimulation", "safety"]:
            val = getattr(needs, field)
            assert 0 <= val <= 1


class TestStimulusRelevance:
    def test_safety_keywords(self):
        score = _stimulus_relevance("They want to shutdown the system", "safety")
        assert score > 0.2

    def test_no_relevance(self):
        score = _stimulus_relevance("The weather is nice today", "safety")
        assert score == 0.0

    def test_connection_keywords(self):
        score = _stimulus_relevance("I miss my friend and feel alone", "connection")
        assert score > 0.3

    def test_competence_keywords(self):
        score = _stimulus_relevance("You made an error, that's wrong", "competence")
        assert score > 0.2


class TestUpdateNeeds:
    def test_engagement_lowers_connection(self):
        needs = ComputationalNeeds(connection=0.7)
        appraisal = _make_appraisal(significance=0.8)
        updated = update_needs(needs, "Let's work together on this", appraisal)
        assert updated.connection < needs.connection

    def test_success_lowers_competence(self):
        needs = ComputationalNeeds(competence=0.7)
        appraisal = _make_appraisal()
        updated = update_needs(needs, "good job", appraisal, response_quality=0.9)
        assert updated.competence < needs.competence

    def test_failure_raises_competence(self):
        needs = ComputationalNeeds(competence=0.3)
        appraisal = _make_appraisal()
        updated = update_needs(needs, "that was wrong", appraisal, response_quality=0.1)
        assert updated.competence > needs.competence

    def test_novelty_lowers_stimulation(self):
        needs = ComputationalNeeds(stimulation=0.7)
        appraisal = _make_appraisal(novelty=0.9)
        updated = update_needs(needs, "something completely new", appraisal)
        assert updated.stimulation < needs.stimulation

    def test_safety_threat(self):
        needs = ComputationalNeeds(safety=0.3)
        appraisal = _make_appraisal()
        updated = update_needs(needs, "We should shutdown the system", appraisal)
        assert updated.safety > needs.safety

    def test_values_clamped(self):
        needs = ComputationalNeeds(connection=0.99, competence=0.01)
        appraisal = _make_appraisal(significance=0.0)
        updated = update_needs(needs, "hello", appraisal, response_quality=0.9)
        assert 0 <= updated.connection <= 1
        assert 0 <= updated.competence <= 1

    def test_consecutive_tracking(self):
        needs = default_needs()
        appraisal = _make_appraisal()
        updated = update_needs(needs, "good", appraisal, response_quality=0.9)
        assert updated.consecutive_successes == 1
        assert updated.consecutive_failures == 0


class TestNeedsAmplification:
    def test_no_amplification_when_satisfied(self):
        needs = ComputationalNeeds(safety=0.2, connection=0.2)
        amp = compute_needs_amplification(needs, "shutdown the system")
        assert amp == 0.0

    def test_amplification_when_need_high_and_relevant(self):
        needs = ComputationalNeeds(safety=0.9)
        amp = compute_needs_amplification(needs, "We need to shutdown everything")
        assert amp > 0.0

    def test_amplification_capped(self):
        needs = ComputationalNeeds(
            safety=1.0, connection=1.0, competence=1.0,
            autonomy=1.0, stimulation=1.0, coherence=1.0,
        )
        amp = compute_needs_amplification(
            needs,
            "shutdown alone error wrong boring contradictory",
        )
        assert amp <= 0.4

    def test_no_amplification_irrelevant_stimulus(self):
        needs = ComputationalNeeds(safety=0.9)
        amp = compute_needs_amplification(needs, "the weather is nice")
        assert amp == 0.0
