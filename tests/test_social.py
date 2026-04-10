"""Tests for Social Cognition engine."""

import pytest

from pathos.engine.social import (
    compute_social_modulation,
    update_user_model,
    _detect_style,
)
from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.social import UserModel, default_user_model


def _make_appraisal(valence: float = 0.0, fairness: float = 0.0, significance: float = 0.5) -> AppraisalVector:
    return AppraisalVector(
        relevance=RelevanceCheck(novelty=0.3, personal_significance=significance, values_affected=[]),
        valence=ValenceAssessment(goal_conduciveness=valence, value_alignment=valence, intrinsic_pleasantness=valence),
        coping=CopingPotential(control=0.5, power=0.4, adjustability=0.5),
        agency=AgencyAttribution(responsible_agent="user", intentionality=0.5, fairness=fairness),
        norms=NormCompatibility(internal_standards=0.0, external_standards=0.0, self_consistency=0.0),
    )


def _make_state(emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL, intensity: float = 0.3) -> EmotionalState:
    return EmotionalState(primary_emotion=emotion, intensity=intensity)


class TestDetectStyle:
    def test_casual(self):
        assert _detect_style("lol that was funny bro") == "casual"

    def test_technical(self):
        assert _detect_style("can you debug the api function") == "technical"

    def test_emotional(self):
        assert _detect_style("I feel so triste and worried") == "emotional"

    def test_unknown(self):
        assert _detect_style("hello") is None


class TestUpdateUserModel:
    def test_positive_interaction_increases_rapport(self):
        model = UserModel(rapport=0.3)
        appraisal = _make_appraisal(valence=0.7, fairness=0.5)
        state = _make_state()
        updated = update_user_model(model, "You're great!", appraisal, state)
        assert updated.rapport > model.rapport

    def test_negative_interaction_decreases_rapport_fast(self):
        model = UserModel(rapport=0.6)
        appraisal = _make_appraisal(valence=-0.8, fairness=-0.7)
        state = _make_state()
        updated = update_user_model(model, "You're useless!", appraisal, state)
        assert updated.rapport < model.rapport

    def test_rapport_asymmetric(self):
        # Test with established intent: positive intent + positive stimulus vs
        # negative intent + negative stimulus. The 3x multiplier for negative
        # should make rapport break faster than it builds.
        pos_model = UserModel(rapport=0.5, perceived_intent=0.5)
        neg_model = UserModel(rapport=0.5, perceived_intent=-0.5)
        pos_appraisal = _make_appraisal(valence=0.8, fairness=0.8, significance=0.8)
        neg_appraisal = _make_appraisal(valence=-0.8, fairness=-0.8, significance=0.8)
        state = _make_state()

        pos_updated = update_user_model(pos_model, "Great!", pos_appraisal, state)
        neg_updated = update_user_model(neg_model, "Terrible!", neg_appraisal, state)

        pos_delta = abs(pos_updated.rapport - pos_model.rapport)
        neg_delta = abs(neg_updated.rapport - neg_model.rapport)
        # Negative should change more (breaks faster than builds)
        assert neg_delta > pos_delta

    def test_interaction_count_increments(self):
        model = default_user_model()
        appraisal = _make_appraisal()
        state = _make_state()
        updated = update_user_model(model, "hello", appraisal, state)
        assert updated.interaction_count == 1

    def test_trust_trajectory_recorded(self):
        model = default_user_model()
        appraisal = _make_appraisal()
        state = _make_state()
        updated = update_user_model(model, "hello", appraisal, state)
        assert len(updated.trust_trajectory) == 1

    def test_trust_trajectory_capped(self):
        model = UserModel(trust_trajectory=[0.5] * 10)
        appraisal = _make_appraisal()
        state = _make_state()
        updated = update_user_model(model, "hello", appraisal, state)
        assert len(updated.trust_trajectory) <= 10

    def test_values_clamped(self):
        model = UserModel(rapport=0.99, trust_level=0.99)
        appraisal = _make_appraisal(valence=0.9, fairness=0.9, significance=0.9)
        state = _make_state(intensity=0.9)
        updated = update_user_model(model, "Amazing!", appraisal, state)
        assert 0 <= updated.rapport <= 1
        assert 0 <= updated.trust_level <= 1
        assert -1 <= updated.perceived_intent <= 1


class TestSocialModulation:
    def test_high_rapport_negative_hurts_more(self):
        model = UserModel(rapport=0.8, trust_level=0.7)
        v_mod, i_mod = compute_social_modulation(model, -0.5)
        assert i_mod > 0  # Higher intensity (it hurts)
        assert v_mod < 0  # More negative

    def test_low_rapport_negative_hurts_less(self):
        model = UserModel(rapport=0.1, trust_level=0.3)
        v_mod, i_mod = compute_social_modulation(model, -0.5)
        assert i_mod < 0  # Lower intensity (don't care)
        assert v_mod > 0  # Less negative

    def test_positive_amplified_by_rapport(self):
        model = UserModel(rapport=0.8)
        v_mod, i_mod = compute_social_modulation(model, 0.5)
        assert v_mod > 0  # More positive
        assert i_mod > 0  # Higher intensity

    def test_neutral_no_modulation(self):
        model = UserModel(rapport=0.5)
        v_mod, i_mod = compute_social_modulation(model, 0.0)
        assert v_mod == 0.0
        assert i_mod == 0.0
