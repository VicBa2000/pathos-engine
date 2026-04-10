"""Tests for Personality Profile model."""

import pytest

from pathos.models.personality import PersonalityProfile, default_personality


class TestDefaultPersonality:
    def test_default_values(self):
        p = default_personality()
        assert p.openness == 0.6
        assert p.neuroticism == 0.4
        assert p.extraversion == 0.5

    def test_all_in_range(self):
        p = default_personality()
        for field in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert 0 <= getattr(p, field) <= 1


class TestDerivedProperties:
    def test_variability_increases_with_neuroticism(self):
        low_n = PersonalityProfile(neuroticism=0.1)
        high_n = PersonalityProfile(neuroticism=0.9)
        assert high_n.variability > low_n.variability

    def test_regulation_capacity_decreases_with_neuroticism(self):
        low_n = PersonalityProfile(neuroticism=0.1)
        high_n = PersonalityProfile(neuroticism=0.9)
        assert low_n.regulation_capacity_base > high_n.regulation_capacity_base

    def test_empathy_weight_increases_with_agreeableness(self):
        low_a = PersonalityProfile(agreeableness=0.1)
        high_a = PersonalityProfile(agreeableness=0.9)
        assert high_a.empathy_weight > low_a.empathy_weight

    def test_norm_weight_increases_with_conscientiousness(self):
        low_c = PersonalityProfile(conscientiousness=0.1)
        high_c = PersonalityProfile(conscientiousness=0.9)
        assert high_c.norm_weight > low_c.norm_weight

    def test_arousal_baseline_increases_with_extraversion(self):
        low_e = PersonalityProfile(extraversion=0.1)
        high_e = PersonalityProfile(extraversion=0.9)
        assert high_e.arousal_baseline > low_e.arousal_baseline

    def test_derived_values_in_range(self):
        # Test edge cases
        for n in [0.0, 0.5, 1.0]:
            for e in [0.0, 0.5, 1.0]:
                for c in [0.0, 0.5, 1.0]:
                    p = PersonalityProfile(neuroticism=n, extraversion=e, conscientiousness=c)
                    assert 0 <= p.variability <= 1
                    assert 0 <= p.regulation_capacity_base <= 1
                    assert 0.1 <= p.arousal_baseline <= 0.7
                    assert 0 <= p.inertia_base <= 0.8


class TestPersonalityEdgeCases:
    def test_extreme_neurotic(self):
        p = PersonalityProfile(neuroticism=1.0, conscientiousness=0.0)
        assert p.variability > 0.5
        assert p.regulation_capacity_base < 0.5

    def test_extreme_stable(self):
        p = PersonalityProfile(neuroticism=0.0, conscientiousness=1.0)
        assert p.variability < 0.5
        assert p.regulation_capacity_base > 0.5
