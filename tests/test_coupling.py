"""Tests for CouplingMatrix — cross-dimensional ODE interaction coefficients."""

import pytest

from pathos.models.coupling import (
    CouplingMatrix,
    coupling_from_personality,
    default_coupling,
    preset_default_coupling,
    preset_neurotic_coupling,
    preset_resilient_coupling,
    preset_volatile_coupling,
)


class TestCouplingMatrixModel:
    """Tests for the CouplingMatrix Pydantic model."""

    def test_default_is_zero(self) -> None:
        m = CouplingMatrix()
        assert m.is_zero

    def test_as_matrix_shape(self) -> None:
        m = CouplingMatrix()
        mat = m.as_matrix()
        assert len(mat) == 4
        for row in mat:
            assert len(row) == 4

    def test_diagonal_is_zero(self) -> None:
        m = preset_default_coupling()
        mat = m.as_matrix()
        for i in range(4):
            assert mat[i][i] == 0.0

    def test_coefficients_in_range(self) -> None:
        m = preset_volatile_coupling()
        mat = m.as_matrix()
        for row in mat:
            for val in row:
                assert -0.5 <= val <= 0.5

    def test_non_zero_preset(self) -> None:
        m = preset_default_coupling()
        assert not m.is_zero

    def test_field_validation_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            CouplingMatrix(alpha_v_a=0.6)
        with pytest.raises(Exception):
            CouplingMatrix(alpha_a_v=-0.6)


class TestCouplingContribution:
    """Tests for get_coupling_contribution()."""

    def test_zero_deviations_give_zero_contribution(self) -> None:
        m = preset_default_coupling()
        c_v, c_a, c_d, c_c = m.get_coupling_contribution(0.0, 0.0, 0.0, 0.0)
        assert c_v == 0.0
        assert c_a == 0.0
        assert c_d == 0.0
        assert c_c == 0.0

    def test_zero_matrix_gives_zero_contribution(self) -> None:
        m = default_coupling()
        c_v, c_a, c_d, c_c = m.get_coupling_contribution(0.5, 0.3, -0.2, 0.1)
        assert c_v == 0.0
        assert c_a == 0.0
        assert c_d == 0.0
        assert c_c == 0.0

    def test_negative_valence_increases_arousal(self) -> None:
        """When valence is below attractor (dev_v < 0), arousal should increase.
        alpha_a_v is negative, so: alpha_a_v * dev_v = (-) * (-) = positive arousal push.
        """
        m = preset_default_coupling()
        _, c_a, _, _ = m.get_coupling_contribution(
            dev_v=-0.5, dev_a=0.0, dev_d=0.0, dev_c=0.0,
        )
        assert c_a > 0, "Negative valence deviation should push arousal up"

    def test_positive_valence_decreases_arousal(self) -> None:
        """Positive valence deviation should slightly reduce arousal push."""
        m = preset_default_coupling()
        _, c_a, _, _ = m.get_coupling_contribution(
            dev_v=0.5, dev_a=0.0, dev_d=0.0, dev_c=0.0,
        )
        assert c_a < 0, "Positive valence deviation should push arousal down"

    def test_high_arousal_reduces_dominance(self) -> None:
        """alpha_d_a is negative: high arousal erodes sense of control."""
        m = preset_default_coupling()
        _, _, c_d, _ = m.get_coupling_contribution(
            dev_v=0.0, dev_a=0.5, dev_d=0.0, dev_c=0.0,
        )
        assert c_d < 0, "High arousal should push dominance down"

    def test_negative_valence_reduces_certainty(self) -> None:
        """alpha_c_v is positive: when valence drops (dev_v < 0),
        certainty drops proportionally."""
        m = preset_default_coupling()
        _, _, _, c_c = m.get_coupling_contribution(
            dev_v=-0.5, dev_a=0.0, dev_d=0.0, dev_c=0.0,
        )
        assert c_c < 0, "Negative valence should push certainty down"

    def test_high_dominance_increases_certainty(self) -> None:
        """alpha_c_d is positive: feeling in control reinforces certainty."""
        m = preset_default_coupling()
        _, _, _, c_c = m.get_coupling_contribution(
            dev_v=0.0, dev_a=0.0, dev_d=0.5, dev_c=0.0,
        )
        assert c_c > 0, "High dominance should push certainty up"

    def test_uncertainty_increases_arousal(self) -> None:
        """alpha_a_c is negative: low certainty (dev_c < 0) increases arousal.
        (-) * (-) = positive push on arousal.
        """
        m = preset_default_coupling()
        _, c_a, _, _ = m.get_coupling_contribution(
            dev_v=0.0, dev_a=0.0, dev_d=0.0, dev_c=-0.5,
        )
        assert c_a > 0, "Uncertainty should push arousal up"

    def test_positive_dominance_is_pleasant(self) -> None:
        """alpha_v_d is positive: high dominance is mildly pleasant."""
        m = preset_default_coupling()
        c_v, _, _, _ = m.get_coupling_contribution(
            dev_v=0.0, dev_a=0.0, dev_d=0.5, dev_c=0.0,
        )
        assert c_v > 0, "High dominance should push valence up slightly"

    def test_contributions_are_proportional_to_deviation(self) -> None:
        """Double the deviation should give double the contribution."""
        m = preset_default_coupling()
        _, c_a_small, _, _ = m.get_coupling_contribution(
            dev_v=-0.2, dev_a=0.0, dev_d=0.0, dev_c=0.0,
        )
        _, c_a_large, _, _ = m.get_coupling_contribution(
            dev_v=-0.4, dev_a=0.0, dev_d=0.0, dev_c=0.0,
        )
        assert abs(c_a_large - 2 * c_a_small) < 1e-5


class TestCouplingFromPersonality:
    """Tests for coupling_from_personality() generation."""

    def test_returns_coupling_matrix(self) -> None:
        m = coupling_from_personality(0.5, 0.5, 0.5, 0.5, 0.5)
        assert isinstance(m, CouplingMatrix)

    def test_all_coefficients_in_range(self) -> None:
        """Even with extreme personality values, coefficients stay in [-0.5, 0.5]."""
        for o in [0.0, 0.5, 1.0]:
            for c in [0.0, 0.5, 1.0]:
                for e in [0.0, 0.5, 1.0]:
                    for a in [0.0, 0.5, 1.0]:
                        for n in [0.0, 0.5, 1.0]:
                            m = coupling_from_personality(o, c, e, a, n, 0.5)
                            mat = m.as_matrix()
                            for row in mat:
                                for val in row:
                                    assert -0.5 <= val <= 0.5, (
                                        f"Out of range: {val} for O={o} C={c} E={e} A={a} N={n}"
                                    )

    def test_neurotic_has_stronger_v_to_a_coupling(self) -> None:
        """High neuroticism should produce stronger V->A destabilization."""
        m_low = coupling_from_personality(0.5, 0.5, 0.5, 0.5, 0.1)
        m_high = coupling_from_personality(0.5, 0.5, 0.5, 0.5, 0.9)
        # alpha_a_v is negative; more neurotic = more negative (stronger)
        assert m_high.alpha_a_v < m_low.alpha_a_v

    def test_conscientiousness_dampens_coupling(self) -> None:
        """High conscientiousness should produce weaker coupling magnitudes."""
        m_low_c = coupling_from_personality(0.5, 0.1, 0.5, 0.5, 0.5)
        m_high_c = coupling_from_personality(0.5, 0.9, 0.5, 0.5, 0.5)
        # Compare absolute magnitudes of a destabilizing coefficient
        assert abs(m_high_c.alpha_d_a) < abs(m_low_c.alpha_d_a)

    def test_high_reactivity_amplifies(self) -> None:
        """High emotional_reactivity should amplify coupling magnitudes."""
        m_low = coupling_from_personality(0.5, 0.5, 0.5, 0.5, 0.5, 0.1)
        m_high = coupling_from_personality(0.5, 0.5, 0.5, 0.5, 0.5, 0.9)
        assert abs(m_high.alpha_a_v) > abs(m_low.alpha_a_v)

    def test_openness_reduces_uncertainty_coupling(self) -> None:
        """High openness should reduce certainty-related coupling (tolerates ambiguity)."""
        m_low_o = coupling_from_personality(0.1, 0.5, 0.5, 0.5, 0.5)
        m_high_o = coupling_from_personality(0.9, 0.5, 0.5, 0.5, 0.5)
        # alpha_c_v (valence->certainty) should be weaker with high openness
        assert abs(m_high_o.alpha_c_v) < abs(m_low_o.alpha_c_v)

    def test_extraversion_moderates_arousal_valence(self) -> None:
        """High extraversion reduces the arousal->valence penalty
        (extraverts thrive on activation)."""
        m_intro = coupling_from_personality(0.5, 0.5, 0.1, 0.5, 0.5)
        m_extra = coupling_from_personality(0.5, 0.5, 0.9, 0.5, 0.5)
        # alpha_v_a is negative; with extraversion it should be less negative
        assert m_extra.alpha_v_a > m_intro.alpha_v_a

    def test_non_zero_for_non_extreme_personality(self) -> None:
        """A normal personality should produce non-zero coupling."""
        m = coupling_from_personality(0.5, 0.5, 0.5, 0.5, 0.5)
        assert not m.is_zero


class TestPresets:
    """Tests for preset coupling matrices."""

    def test_default_coupling_is_zero(self) -> None:
        assert default_coupling().is_zero

    def test_preset_default_is_non_zero(self) -> None:
        assert not preset_default_coupling().is_zero

    def test_preset_neurotic_is_non_zero(self) -> None:
        assert not preset_neurotic_coupling().is_zero

    def test_preset_resilient_is_non_zero(self) -> None:
        assert not preset_resilient_coupling().is_zero

    def test_preset_volatile_is_non_zero(self) -> None:
        assert not preset_volatile_coupling().is_zero

    def test_neurotic_stronger_than_resilient(self) -> None:
        """Neurotic preset should have stronger destabilizing couplings."""
        n = preset_neurotic_coupling()
        r = preset_resilient_coupling()
        # alpha_a_v: negative for both, neurotic should be more negative
        assert n.alpha_a_v < r.alpha_a_v

    def test_volatile_stronger_than_default(self) -> None:
        """Volatile preset should have stronger coupling than default."""
        v = preset_volatile_coupling()
        d = preset_default_coupling()
        # Sum of absolute values
        v_total = sum(abs(x) for row in v.as_matrix() for x in row)
        d_total = sum(abs(x) for row in d.as_matrix() for x in row)
        assert v_total > d_total

    def test_resilient_weakest_coupling(self) -> None:
        """Resilient preset should have weakest overall coupling magnitude."""
        presets = [
            preset_default_coupling(),
            preset_neurotic_coupling(),
            preset_resilient_coupling(),
            preset_volatile_coupling(),
        ]
        totals = [sum(abs(x) for row in p.as_matrix() for x in row) for p in presets]
        resilient_total = totals[2]
        assert resilient_total == min(totals)

    def test_all_presets_in_range(self) -> None:
        for preset_fn in [
            preset_default_coupling,
            preset_neurotic_coupling,
            preset_resilient_coupling,
            preset_volatile_coupling,
        ]:
            m = preset_fn()
            mat = m.as_matrix()
            for row in mat:
                for val in row:
                    assert -0.5 <= val <= 0.5
