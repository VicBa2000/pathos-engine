"""Coupling Matrix — Cross-dimensional interaction coefficients.

Models how the 4 emotional dimensions (Valence, Arousal, Dominance, Certainty)
influence each other through coupled ODEs. Based on psychological evidence:

  - Negative valence increases arousal (distress activates the system)
  - High arousal reduces dominance (overwhelm erodes sense of control)
  - Negative valence reduces certainty (negativity breeds doubt)
  - High dominance increases certainty (feeling in control = more predictable)
  - Low certainty increases arousal (uncertainty is activating)
  - Negative valence reduces dominance (feeling bad = feeling less capable)

The coupling matrix is generated from PersonalityProfile:
  - High neuroticism amplifies destabilizing couplings
  - High conscientiousness dampens cross-dimensional noise
  - High emotional_reactivity amplifies all couplings
  - High openness slightly increases tolerance to uncertainty

Each coefficient alpha_ij represents how dimension j's deviation from its
attractor influences dimension i's rate of change:

    dX_i/dt = ... + alpha_ij * (X_j - attractor_j) + ...

Positive alpha = excitatory coupling (j pulls i in same direction)
Negative alpha = inhibitory coupling (j pushes i in opposite direction)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class CouplingMatrix(BaseModel):
    """4x4 coupling coefficients between emotional dimensions.

    Naming convention: alpha_{target}_{source}
    Example: alpha_v_a = how arousal deviation affects valence rate of change.

    All coefficients are in [-0.5, 0.5]. Typical magnitude is 0.02-0.15.
    Zero means no coupling (dimensions are independent, legacy behavior).
    """

    # --- Valence is influenced by: ---
    alpha_v_a: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Arousal -> Valence. Negative: high arousal pushes valence down (overwhelm).",
    )
    alpha_v_d: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Dominance -> Valence. Positive: feeling in control is mildly pleasant.",
    )
    alpha_v_c: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Certainty -> Valence. Positive: predictability is mildly comforting.",
    )

    # --- Arousal is influenced by: ---
    alpha_a_v: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Valence -> Arousal. Negative: negative valence increases arousal (distress activates).",
    )
    alpha_a_d: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Dominance -> Arousal. Negative: high dominance reduces arousal (calm control).",
    )
    alpha_a_c: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Certainty -> Arousal. Negative: uncertainty increases arousal (vigilance).",
    )

    # --- Dominance is influenced by: ---
    alpha_d_v: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Valence -> Dominance. Positive: positive mood enhances sense of control.",
    )
    alpha_d_a: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Arousal -> Dominance. Negative: high arousal erodes sense of control.",
    )
    alpha_d_c: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Certainty -> Dominance. Positive: certainty reinforces dominance.",
    )

    # --- Certainty is influenced by: ---
    alpha_c_v: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Valence -> Certainty. Positive: positive valence increases certainty.",
    )
    alpha_c_a: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Arousal -> Certainty. Negative: high arousal disrupts certainty.",
    )
    alpha_c_d: float = Field(
        default=0.0, ge=-0.5, le=0.5,
        description="Dominance -> Certainty. Positive: dominance reinforces certainty.",
    )

    def as_matrix(self) -> list[list[float]]:
        """Return the 4x4 matrix as nested lists.

        Row order: [V, A, D, C].  Column order: [V, A, D, C].
        Diagonal is always 0 (self-coupling is handled by the attractor term).
        """
        return [
            [0.0,           self.alpha_v_a, self.alpha_v_d, self.alpha_v_c],
            [self.alpha_a_v, 0.0,           self.alpha_a_d, self.alpha_a_c],
            [self.alpha_d_v, self.alpha_d_a, 0.0,           self.alpha_d_c],
            [self.alpha_c_v, self.alpha_c_a, self.alpha_c_d, 0.0          ],
        ]

    def get_coupling_contribution(
        self,
        dev_v: float,
        dev_a: float,
        dev_d: float,
        dev_c: float,
    ) -> tuple[float, float, float, float]:
        """Compute the coupling contribution for each dimension.

        Args:
            dev_v: Valence deviation from attractor (current - attractor).
            dev_a: Arousal deviation from attractor.
            dev_d: Dominance deviation from attractor.
            dev_c: Certainty deviation from attractor.

        Returns:
            (coupling_v, coupling_a, coupling_d, coupling_c) — additive terms
            to include in each dimension's ODE.
        """
        c_v = self.alpha_v_a * dev_a + self.alpha_v_d * dev_d + self.alpha_v_c * dev_c
        c_a = self.alpha_a_v * dev_v + self.alpha_a_d * dev_d + self.alpha_a_c * dev_c
        c_d = self.alpha_d_v * dev_v + self.alpha_d_a * dev_a + self.alpha_d_c * dev_c
        c_c = self.alpha_c_v * dev_v + self.alpha_c_a * dev_a + self.alpha_c_d * dev_d
        return (
            round(c_v, 6),
            round(c_a, 6),
            round(c_d, 6),
            round(c_c, 6),
        )

    @property
    def is_zero(self) -> bool:
        """True if all coefficients are zero (no coupling, legacy behavior)."""
        return all(
            getattr(self, f"alpha_{t}_{s}") == 0.0
            for t in "vadc"
            for s in "vadc"
            if t != s
        )


def coupling_from_personality(
    openness: float,
    conscientiousness: float,
    extraversion: float,
    agreeableness: float,
    neuroticism: float,
    emotional_reactivity: float = 0.5,
) -> CouplingMatrix:
    """Generate a CouplingMatrix from Big Five personality traits.

    Psychological rationale for each coefficient:

    V->A (alpha_a_v): Negative valence activates the system (fight-or-flight).
        Amplified by neuroticism (more reactive to negativity).
        Dampened by conscientiousness (emotional discipline).

    A->V (alpha_v_a): Very high arousal is unpleasant (overwhelm).
        Amplified by neuroticism. Dampened by extraversion (thrives on activation).

    A->D (alpha_d_a): High arousal erodes sense of control.
        Amplified by neuroticism. Dampened by conscientiousness.

    V->D (alpha_d_v): Positive valence enhances sense of control.
        Amplified by extraversion (positive mood = empowerment).

    V->C (alpha_c_v): Negative valence breeds doubt and uncertainty.
        Amplified by neuroticism. Dampened by openness (tolerates ambiguity).

    D->C (alpha_c_d): Feeling in control increases certainty.
        Amplified by conscientiousness (structured = predictable).

    C->A (alpha_a_c): Uncertainty increases arousal (vigilance).
        Amplified by neuroticism. Dampened by openness.

    D->A (alpha_a_d): High dominance calms the system.
        Amplified by conscientiousness.

    C->V (alpha_v_c): Certainty is mildly comforting.
        Small effect, amplified by conscientiousness.

    V->D, A->C, D->V are smaller secondary effects.
    """
    # Global reactivity scale: how strongly dimensions cross-talk
    reactivity = 0.5 + emotional_reactivity * 0.5  # 0.5-1.0

    # Neuroticism amplifies destabilizing couplings
    n_amp = 0.7 + neuroticism * 0.6  # 0.7-1.3

    # Conscientiousness dampens coupling (self-discipline)
    c_damp = 1.0 - conscientiousness * 0.3  # 0.7-1.0

    # Openness increases tolerance (reduces uncertainty-driven coupling)
    o_tol = 1.0 - openness * 0.25  # 0.75-1.0

    # Extraversion modulates arousal-related couplings
    e_mod = 0.8 + extraversion * 0.4  # 0.8-1.2

    # --- Compute coefficients ---

    # Valence -> Arousal: negative valence activates (NEGATIVE coefficient)
    # When V < attractor, dev_v is negative, so negative alpha makes arousal go UP
    alpha_a_v = _clamp(-0.08 * n_amp * reactivity * c_damp, -0.5, 0.5)

    # Arousal -> Valence: high arousal is unpleasant (NEGATIVE coefficient)
    alpha_v_a = _clamp(-0.05 * n_amp * reactivity * (1.1 - extraversion * 0.3), -0.5, 0.5)

    # Arousal -> Dominance: high arousal erodes control (NEGATIVE)
    alpha_d_a = _clamp(-0.07 * n_amp * reactivity * c_damp, -0.5, 0.5)

    # Valence -> Dominance: positive mood enhances control (POSITIVE)
    alpha_d_v = _clamp(0.06 * e_mod * reactivity * c_damp, -0.5, 0.5)

    # Valence -> Certainty: negativity breeds doubt (POSITIVE — same direction)
    # When V goes down (dev_v negative), certainty goes down too
    alpha_c_v = _clamp(0.06 * n_amp * reactivity * o_tol, -0.5, 0.5)

    # Dominance -> Certainty: control = predictability (POSITIVE)
    alpha_c_d = _clamp(0.05 * (0.8 + conscientiousness * 0.4) * reactivity, -0.5, 0.5)

    # Certainty -> Arousal: uncertainty activates (NEGATIVE)
    # Low certainty (dev_c negative) → negative * negative = positive arousal push
    alpha_a_c = _clamp(-0.06 * n_amp * reactivity * o_tol, -0.5, 0.5)

    # Dominance -> Arousal: control calms (NEGATIVE)
    alpha_a_d = _clamp(-0.04 * (0.8 + conscientiousness * 0.4) * reactivity, -0.5, 0.5)

    # Certainty -> Valence: certainty is mildly comforting (POSITIVE)
    alpha_v_c = _clamp(0.03 * (0.8 + conscientiousness * 0.4) * reactivity, -0.5, 0.5)

    # Dominance -> Valence: feeling in control is pleasant (POSITIVE)
    alpha_v_d = _clamp(0.04 * e_mod * reactivity, -0.5, 0.5)

    # Arousal -> Certainty: high arousal disrupts certainty (NEGATIVE)
    alpha_c_a = _clamp(-0.05 * n_amp * reactivity * o_tol, -0.5, 0.5)

    # Certainty -> Dominance: certainty reinforces dominance (POSITIVE)
    alpha_d_c = _clamp(0.04 * (0.8 + conscientiousness * 0.4) * reactivity, -0.5, 0.5)

    return CouplingMatrix(
        alpha_v_a=round(alpha_v_a, 4),
        alpha_v_d=round(alpha_v_d, 4),
        alpha_v_c=round(alpha_v_c, 4),
        alpha_a_v=round(alpha_a_v, 4),
        alpha_a_d=round(alpha_a_d, 4),
        alpha_a_c=round(alpha_a_c, 4),
        alpha_d_v=round(alpha_d_v, 4),
        alpha_d_a=round(alpha_d_a, 4),
        alpha_d_c=round(alpha_d_c, 4),
        alpha_c_v=round(alpha_c_v, 4),
        alpha_c_a=round(alpha_c_a, 4),
        alpha_c_d=round(alpha_c_d, 4),
    )


def default_coupling() -> CouplingMatrix:
    """Default coupling matrix (all zeros — legacy independent behavior)."""
    return CouplingMatrix()


# --- Personality presets ---

def preset_default_coupling() -> CouplingMatrix:
    """Balanced personality coupling (default Big Five values)."""
    return coupling_from_personality(
        openness=0.6, conscientiousness=0.6, extraversion=0.5,
        agreeableness=0.6, neuroticism=0.4, emotional_reactivity=0.5,
    )


def preset_neurotic_coupling() -> CouplingMatrix:
    """High neuroticism: strong destabilizing cross-talk."""
    return coupling_from_personality(
        openness=0.4, conscientiousness=0.3, extraversion=0.3,
        agreeableness=0.5, neuroticism=0.85, emotional_reactivity=0.8,
    )


def preset_resilient_coupling() -> CouplingMatrix:
    """Low neuroticism, high conscientiousness: dampened, stable coupling."""
    return coupling_from_personality(
        openness=0.6, conscientiousness=0.8, extraversion=0.6,
        agreeableness=0.7, neuroticism=0.15, emotional_reactivity=0.3,
    )


def preset_volatile_coupling() -> CouplingMatrix:
    """High reactivity, low conscientiousness: amplified cross-talk."""
    return coupling_from_personality(
        openness=0.7, conscientiousness=0.2, extraversion=0.7,
        agreeableness=0.4, neuroticism=0.7, emotional_reactivity=0.9,
    )
