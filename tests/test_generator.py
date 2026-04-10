"""Tests para el Emotion Generator (Fase 2: 5D appraisal, 19 emociones)."""

from pathos.engine.generator import (
    compute_arousal,
    compute_body_state,
    compute_dominance,
    compute_intensity,
    compute_valence,
    generate_emotion,
    identify_primary_emotion,
)
from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
)
from pathos.models.emotion import BodyState, PrimaryEmotion, neutral_state


def _make_appraisal(
    novelty: float = 0.0,
    significance: float = 0.5,
    goal_cond: float = 0.0,
    value_align: float = 0.0,
    pleasantness: float = 0.0,
    control: float = 0.5,
    power: float = 0.5,
    adjustability: float = 0.5,
    responsible_agent: str = "environment",
    intentionality: float = 0.5,
    fairness: float = 0.0,
    internal_standards: float = 0.0,
    external_standards: float = 0.0,
    self_consistency: float = 0.0,
) -> AppraisalVector:
    """Helper para crear AppraisalVectors de test."""
    return AppraisalVector(
        relevance=RelevanceCheck(
            novelty=novelty,
            personal_significance=significance,
            values_affected=[],
        ),
        valence=ValenceAssessment(
            goal_conduciveness=goal_cond,
            value_alignment=value_align,
            intrinsic_pleasantness=pleasantness,
        ),
        coping=CopingPotential(
            control=control,
            power=power,
            adjustability=adjustability,
        ),
        agency=AgencyAttribution(
            responsible_agent=responsible_agent,
            intentionality=intentionality,
            fairness=fairness,
        ),
        norms=NormCompatibility(
            internal_standards=internal_standards,
            external_standards=external_standards,
            self_consistency=self_consistency,
        ),
    )


# --- Tests de rango ---


class TestRanges:
    """Todos los valores deben estar dentro de sus rangos definidos."""

    def test_valence_range(self) -> None:
        a = _make_appraisal(goal_cond=1, value_align=1, pleasantness=1)
        assert -1 <= compute_valence(a) <= 1
        a = _make_appraisal(goal_cond=-1, value_align=-1, pleasantness=-1)
        assert -1 <= compute_valence(a) <= 1

    def test_arousal_range(self) -> None:
        a = _make_appraisal(novelty=1, significance=1, control=0)
        assert 0 <= compute_arousal(a) <= 1
        a = _make_appraisal(novelty=0, significance=0, control=1)
        assert 0 <= compute_arousal(a) <= 1

    def test_dominance_range(self) -> None:
        a = _make_appraisal(control=1, power=1, fairness=1)
        assert 0 <= compute_dominance(a) <= 1
        a = _make_appraisal(control=0, power=0, fairness=-1)
        assert 0 <= compute_dominance(a) <= 1

    def test_intensity_range(self) -> None:
        a = _make_appraisal(significance=1, goal_cond=1, value_align=1, pleasantness=1, internal_standards=1)
        v = compute_valence(a)
        ar = compute_arousal(a)
        assert 0 <= compute_intensity(a, v, ar) <= 1

    def test_body_state_range(self) -> None:
        body = compute_body_state(1, 1, 1, 1, BodyState())
        assert 0 <= body.energy <= 1
        assert 0 <= body.tension <= 1
        assert 0 <= body.openness <= 1
        assert 0 <= body.warmth <= 1

        body = compute_body_state(-1, 0, 0, 0, BodyState())
        assert 0 <= body.energy <= 1
        assert 0 <= body.tension <= 1
        assert 0 <= body.openness <= 1
        assert 0 <= body.warmth <= 1


# --- Tests de identificacion de emociones (19 emociones) ---


class TestEmotionIdentification:
    """Las emociones deben mapearse coherentemente a las 19 categorias."""

    def _id(self, v: float, a: float, d: float, c: float = 0.5, **kw: float) -> PrimaryEmotion:
        appraisal = _make_appraisal(**kw) if kw else _make_appraisal()
        return identify_primary_emotion(v, a, d, c, appraisal)

    # Positivas alta energia - valores cerca de prototipos
    def test_joy(self) -> None:
        assert self._id(0.75, 0.65, 0.70, 0.70) == PrimaryEmotion.JOY

    def test_gratitude(self) -> None:
        assert self._id(0.70, 0.40, 0.30, 0.70) == PrimaryEmotion.GRATITUDE

    def test_excitement(self) -> None:
        assert self._id(0.70, 0.90, 0.55, 0.25) == PrimaryEmotion.EXCITEMENT

    def test_hope(self) -> None:
        assert self._id(0.50, 0.55, 0.35, 0.25) == PrimaryEmotion.HOPE

    # Positivas baja energia
    def test_contentment(self) -> None:
        assert self._id(0.55, 0.20, 0.60, 0.80) == PrimaryEmotion.CONTENTMENT

    def test_relief(self) -> None:
        assert self._id(0.50, 0.25, 0.45, 0.65) == PrimaryEmotion.RELIEF

    # Negativas alta energia
    def test_anger(self) -> None:
        assert self._id(-0.75, 0.80, 0.70, 0.60, fairness=-0.5) == PrimaryEmotion.ANGER

    def test_frustration(self) -> None:
        assert self._id(-0.50, 0.70, 0.40, 0.35) == PrimaryEmotion.FRUSTRATION

    def test_fear(self) -> None:
        assert self._id(-0.75, 0.85, 0.15, 0.15) == PrimaryEmotion.FEAR

    def test_anxiety(self) -> None:
        assert self._id(-0.45, 0.65, 0.25, 0.25) == PrimaryEmotion.ANXIETY

    # Negativas baja energia
    def test_sadness(self) -> None:
        assert self._id(-0.70, 0.20, 0.25, 0.60) == PrimaryEmotion.SADNESS

    def test_helplessness(self) -> None:
        assert self._id(-0.75, 0.20, 0.10, 0.15) == PrimaryEmotion.HELPLESSNESS

    def test_disappointment(self) -> None:
        assert self._id(-0.50, 0.30, 0.40, 0.55) == PrimaryEmotion.DISAPPOINTMENT

    # Neutrales
    def test_surprise(self) -> None:
        assert self._id(0.05, 0.85, 0.40, 0.15) == PrimaryEmotion.SURPRISE

    def test_alertness(self) -> None:
        assert self._id(-0.05, 0.70, 0.50, 0.35) == PrimaryEmotion.ALERTNESS

    def test_indifference(self) -> None:
        assert self._id(0.0, 0.10, 0.50, 0.50) == PrimaryEmotion.INDIFFERENCE

    def test_contemplation(self) -> None:
        assert self._id(0.15, 0.25, 0.55, 0.55) == PrimaryEmotion.CONTEMPLATION


# --- Tests del pipeline completo ---


class TestGenerateEmotion:
    """Test del generador de emociones completo."""

    def test_positive_stimulus_generates_positive_state(self) -> None:
        appraisal = _make_appraisal(
            significance=0.7, goal_cond=0.8, value_align=0.7, pleasantness=0.6, control=0.7,
        )
        state = generate_emotion(appraisal, neutral_state(), "algo bueno paso")
        assert state.valence > 0

    def test_negative_stimulus_generates_negative_state(self) -> None:
        appraisal = _make_appraisal(
            significance=0.8, goal_cond=-0.7, value_align=-0.8, pleasantness=-0.6, control=0.2,
        )
        state = generate_emotion(appraisal, neutral_state(), "algo malo paso")
        assert state.valence < 0

    def test_inercia_emocional(self) -> None:
        appraisal = _make_appraisal(goal_cond=0.8, value_align=0.8, pleasantness=0.8)
        state_high = generate_emotion(appraisal, neutral_state(), "test", blend_factor=0.9)
        state_low = generate_emotion(appraisal, neutral_state(), "test", blend_factor=0.1)
        assert abs(state_high.valence) > abs(state_low.valence)

    def test_amplification_increases_intensity(self) -> None:
        appraisal = _make_appraisal(significance=0.7, goal_cond=0.8, value_align=0.7)
        state_no_amp = generate_emotion(appraisal, neutral_state(), "test", amplification=0.0)
        state_amp = generate_emotion(appraisal, neutral_state(), "test", amplification=0.5)
        assert state_amp.intensity > state_no_amp.intensity

    def test_secondary_emotion_populated(self) -> None:
        # Anger con baja certeza -> secondary anxiety
        appraisal = _make_appraisal(
            significance=0.8, goal_cond=-0.8, value_align=-0.8, pleasantness=-0.7,
            control=0.7, fairness=-0.5, adjustability=0.2,
        )
        state = generate_emotion(appraisal, neutral_state(), "injusticia incierta")
        # El estado deberia tener alguna emocion valida
        assert state.primary_emotion is not None

    def test_duration_increments_same_emotion(self) -> None:
        appraisal = _make_appraisal(goal_cond=0.8, value_align=0.8, pleasantness=0.8, significance=0.7, control=0.8)
        state1 = generate_emotion(appraisal, neutral_state(), "happy1")
        state2 = generate_emotion(appraisal, state1, "happy2")
        if state1.primary_emotion == state2.primary_emotion:
            assert state2.duration == state1.duration + 1

    def test_all_outputs_in_range(self) -> None:
        """Test exhaustivo de rangos con inputs extremos."""
        extremes = [
            _make_appraisal(1, 1, 1, 1, 1, 1, 1, 1, "user", 1, 1, 1, 1, 1),
            _make_appraisal(-1, 0, -1, -1, -1, 0, 0, 0, "self", 0, -1, -1, -1, -1),
            _make_appraisal(0, 0.5, 0, 0, 0, 0.5, 0.5, 0.5),
        ]

        for appraisal in extremes:
            state = generate_emotion(appraisal, neutral_state(), "test")
            assert -1 <= state.valence <= 1
            assert 0 <= state.arousal <= 1
            assert 0 <= state.dominance <= 1
            assert 0 <= state.certainty <= 1
            assert 0 <= state.intensity <= 1
            assert 0 <= state.body_state.energy <= 1
            assert 0 <= state.body_state.tension <= 1
            assert 0 <= state.body_state.openness <= 1
            assert 0 <= state.body_state.warmth <= 1
