"""Tests para Global Workspace (Pilar 2 ANIMA).

Verifica:
- WorkspaceCandidate saliency computation
- Coalition formation (emotion, category bonds)
- Noise filtering
- Top-K selection
- Integration score (IIT-inspired)
- Workspace stability
- Preconscious buffer persistence
- Priming temporal (irrupcion)
- Mood influence indirecta
- Somatic echo
- Workspace prompt generation
- Full turn orchestration
"""

import pytest

from pathos.engine.workspace import (
    COALITION_COHERENCE_BONUS,
    NOISE_THRESHOLD,
    WORKSPACE_CAPACITY,
    apply_priming,
    compute_integration,
    compute_mood_influence,
    compute_somatic_echo,
    compute_stability,
    filter_noise,
    form_coalitions,
    generate_candidate,
    get_workspace_prompt,
    process_workspace_turn,
    select_workspace,
)
from pathos.models.workspace import (
    Coalition,
    ConsciousnessState,
    PreconsciousBuffer,
    WorkspaceCandidate,
    WorkspaceResult,
    default_consciousness_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(
    source: str = "test",
    content: str = "test content",
    urgency: float = 0.5,
    relevance: float = 0.5,
    intensity: float = 0.5,
    emotion: str = "neutral",
    category: str = "general",
) -> WorkspaceCandidate:
    """Shorthand para crear candidates con saliency computada."""
    return generate_candidate(
        source=source, content=content,
        urgency=urgency, relevance=relevance,
        emotional_intensity=intensity,
        emotion_tag=emotion, category=category,
    )


# ---------------------------------------------------------------------------
# Tests: WorkspaceCandidate
# ---------------------------------------------------------------------------

class TestWorkspaceCandidate:
    """Tests para el modelo WorkspaceCandidate."""

    def test_saliency_computation(self) -> None:
        c = _c(urgency=0.8, relevance=0.6, intensity=0.5)
        assert c.saliency == pytest.approx(0.8 * 0.6 * 0.5, abs=0.01)

    def test_saliency_zero_if_any_dim_zero(self) -> None:
        c = _c(urgency=0.0, relevance=0.8, intensity=0.9)
        assert c.saliency == 0.0

    def test_saliency_max(self) -> None:
        c = _c(urgency=1.0, relevance=1.0, intensity=1.0)
        assert c.saliency == 1.0

    def test_saliency_clamped(self) -> None:
        c = _c(urgency=0.5, relevance=0.5, intensity=0.5)
        assert 0 <= c.saliency <= 1

    def test_defaults(self) -> None:
        c = WorkspaceCandidate(source="x", content="y")
        assert c.urgency == 0.0
        assert c.saliency == 0.0
        assert c.preconscious_turns == 0


# ---------------------------------------------------------------------------
# Tests: Noise Filtering
# ---------------------------------------------------------------------------

class TestNoiseFiltering:
    """Tests para el filtrado de ruido (Fase 1)."""

    def test_filters_low_saliency(self) -> None:
        candidates = [
            _c(urgency=0.1, relevance=0.1, intensity=0.1),  # saliency = 0.001
            _c(urgency=0.8, relevance=0.8, intensity=0.8),  # saliency = 0.512
        ]
        passed, filtered = filter_noise(candidates)
        assert len(passed) == 1
        assert filtered == 1
        assert passed[0].saliency > NOISE_THRESHOLD

    def test_keeps_all_above_threshold(self) -> None:
        candidates = [_c(urgency=0.5, relevance=0.5, intensity=0.5) for _ in range(5)]
        passed, filtered = filter_noise(candidates)
        assert len(passed) == 5
        assert filtered == 0

    def test_empty_input(self) -> None:
        passed, filtered = filter_noise([])
        assert passed == []
        assert filtered == 0

    def test_all_filtered(self) -> None:
        candidates = [_c(urgency=0.1, relevance=0.1, intensity=0.1)]
        passed, filtered = filter_noise(candidates)
        assert len(passed) == 0
        assert filtered == 1


# ---------------------------------------------------------------------------
# Tests: Coalition Formation
# ---------------------------------------------------------------------------

class TestCoalitionFormation:
    """Tests para la formacion de coaliciones (Fase 2)."""

    def test_same_emotion_forms_coalition(self) -> None:
        candidates = [
            _c(source="a", emotion="anger", urgency=0.6, relevance=0.6, intensity=0.6),
            _c(source="b", emotion="anger", urgency=0.5, relevance=0.5, intensity=0.5),
        ]
        coalitions = form_coalitions(candidates)
        # Deberia haber 1 coalicion con 2 miembros
        multi = [c for c in coalitions if len(c.members) > 1]
        assert len(multi) == 1
        assert multi[0].bond_type == "emotion"
        assert multi[0].coherence == COALITION_COHERENCE_BONUS

    def test_different_emotions_no_coalition(self) -> None:
        candidates = [
            _c(source="a", emotion="anger", urgency=0.6, relevance=0.6, intensity=0.6),
            _c(source="b", emotion="joy", urgency=0.5, relevance=0.5, intensity=0.5),
        ]
        coalitions = form_coalitions(candidates)
        multi = [c for c in coalitions if len(c.members) > 1]
        assert len(multi) == 0

    def test_same_category_forms_coalition(self) -> None:
        candidates = [
            _c(source="a", category="relationship", urgency=0.6, relevance=0.6, intensity=0.6),
            _c(source="b", category="relationship", urgency=0.5, relevance=0.5, intensity=0.5),
        ]
        coalitions = form_coalitions(candidates)
        multi = [c for c in coalitions if len(c.members) > 1]
        assert len(multi) == 1
        assert multi[0].bond_type == "category"

    def test_coalition_saliency_with_bonus(self) -> None:
        candidates = [
            _c(source="a", emotion="fear", urgency=0.6, relevance=0.6, intensity=0.6),
            _c(source="b", emotion="fear", urgency=0.5, relevance=0.5, intensity=0.5),
        ]
        coalitions = form_coalitions(candidates)
        multi = [c for c in coalitions if len(c.members) > 1]
        assert len(multi) == 1
        sum_saliency = sum(m.saliency for m in multi[0].members)
        assert multi[0].effective_saliency > sum_saliency  # Bonus applied

    def test_singleton_coalition_no_bonus(self) -> None:
        candidates = [_c(source="a")]
        coalitions = form_coalitions(candidates)
        assert len(coalitions) == 1
        assert coalitions[0].coherence == 1.0
        assert coalitions[0].bond_type == "none"

    def test_empty_input(self) -> None:
        assert form_coalitions([]) == []

    def test_neutral_emotion_no_coalition(self) -> None:
        candidates = [
            _c(source="a", emotion="neutral"),
            _c(source="b", emotion="neutral"),
        ]
        coalitions = form_coalitions(candidates)
        multi = [c for c in coalitions if len(c.members) > 1]
        assert len(multi) == 0  # Neutral no forma coalicion por emocion


# ---------------------------------------------------------------------------
# Tests: Workspace Selection (full pipeline)
# ---------------------------------------------------------------------------

class TestSelectWorkspace:
    """Tests para la seleccion completa del workspace (3 fases)."""

    def test_basic_selection(self) -> None:
        candidates = [
            _c(source="high", urgency=0.9, relevance=0.9, intensity=0.9),
            _c(source="low", urgency=0.2, relevance=0.2, intensity=0.2),
        ]
        result = select_workspace(candidates)
        assert len(result.conscious) <= WORKSPACE_CAPACITY
        assert result.conscious[0].source == "high"

    def test_capacity_limit(self) -> None:
        candidates = [
            _c(source=f"s{i}", urgency=0.8, relevance=0.8, intensity=0.8)
            for i in range(10)
        ]
        result = select_workspace(candidates, capacity=5)
        assert len(result.conscious) == 5
        assert len(result.preconscious) == 5

    def test_saliency_ranking(self) -> None:
        candidates = [
            _c(source="low", urgency=0.3, relevance=0.3, intensity=0.3),
            _c(source="high", urgency=0.9, relevance=0.9, intensity=0.9),
            _c(source="mid", urgency=0.6, relevance=0.6, intensity=0.6),
        ]
        result = select_workspace(candidates, capacity=2)
        sources = [c.source for c in result.conscious]
        assert "high" in sources
        assert "low" not in sources

    def test_empty_candidates(self) -> None:
        result = select_workspace([])
        assert result.conscious == []
        assert result.preconscious == []
        assert result.total_candidates == 0

    def test_all_noise(self) -> None:
        candidates = [_c(urgency=0.1, relevance=0.1, intensity=0.1)]
        result = select_workspace(candidates)
        assert result.conscious == []
        assert result.filtered_noise == 1

    def test_single_candidate(self) -> None:
        result = select_workspace([_c(source="solo", urgency=0.8, relevance=0.8, intensity=0.8)])
        assert len(result.conscious) == 1
        assert result.conscious[0].source == "solo"

    def test_coalition_beats_isolated(self) -> None:
        """Una coalicion con bonus deberia ganarle a un candidato aislado equivalente."""
        candidates = [
            # Coalicion de dos "fear" con saliency moderada cada uno
            _c(source="fear1", emotion="fear", urgency=0.5, relevance=0.5, intensity=0.5),
            _c(source="fear2", emotion="fear", urgency=0.5, relevance=0.5, intensity=0.5),
            # Aislado con saliency alta pero solo
            _c(source="solo", emotion="joy", urgency=0.6, relevance=0.6, intensity=0.6),
        ]
        result = select_workspace(candidates, capacity=2)
        sources = [c.source for c in result.conscious]
        # La coalicion fear deberia entrar (bonus coherence)
        assert "fear1" in sources or "fear2" in sources

    def test_total_candidates_count(self) -> None:
        candidates = [_c() for _ in range(7)]
        result = select_workspace(candidates)
        assert result.total_candidates == 7

    def test_coalitions_formed_count(self) -> None:
        candidates = [
            _c(source="a", emotion="anger"),
            _c(source="b", emotion="anger"),
            _c(source="c", emotion="joy"),
        ]
        result = select_workspace(candidates)
        assert result.coalitions_formed >= 1


# ---------------------------------------------------------------------------
# Tests: Integration Score
# ---------------------------------------------------------------------------

class TestIntegrationScore:
    """Tests para la integracion informacional (IIT-inspired)."""

    def test_single_candidate_full_integration(self) -> None:
        assert compute_integration([_c()]) == 1.0

    def test_empty_workspace_zero(self) -> None:
        assert compute_integration([]) == 0.0

    def test_same_emotion_high_integration(self) -> None:
        workspace = [
            _c(source="a", emotion="anger"),
            _c(source="b", emotion="anger"),
        ]
        score = compute_integration(workspace)
        assert score > 0.5

    def test_different_everything_low_integration(self) -> None:
        workspace = [
            _c(source="a", emotion="joy", category="work", urgency=0.3),
            _c(source="b", emotion="fear", category="health", urgency=0.3),
        ]
        score = compute_integration(workspace)
        assert score < 0.5

    def test_high_urgency_partial_integration(self) -> None:
        """Dos candidatos con alta urgency tienen conexion parcial (crisis)."""
        workspace = [
            _c(source="a", emotion="joy", urgency=0.8),
            _c(source="b", emotion="fear", urgency=0.8),
        ]
        score = compute_integration(workspace)
        assert score > 0  # Parcial por crisis

    def test_score_clamped(self) -> None:
        workspace = [_c(emotion="anger") for _ in range(5)]
        score = compute_integration(workspace)
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# Tests: Workspace Stability
# ---------------------------------------------------------------------------

class TestStability:
    """Tests para la estabilidad del workspace."""

    def test_identical_workspace_full_stability(self) -> None:
        workspace = [_c(source="a"), _c(source="b")]
        stability = compute_stability(workspace, ["a", "b"])
        assert stability == 1.0

    def test_completely_new_workspace(self) -> None:
        workspace = [_c(source="c"), _c(source="d")]
        stability = compute_stability(workspace, ["a", "b"])
        assert stability == 0.0

    def test_partial_overlap(self) -> None:
        workspace = [_c(source="a"), _c(source="c")]
        stability = compute_stability(workspace, ["a", "b"])
        assert 0 < stability < 1

    def test_no_previous(self) -> None:
        assert compute_stability([_c()], []) == 0.0

    def test_empty_current(self) -> None:
        assert compute_stability([], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# Tests: Preconscious Buffer
# ---------------------------------------------------------------------------

class TestPreconsciousBuffer:
    """Tests para el buffer preconsciente."""

    def test_add_candidates(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([_c(source="a"), _c(source="b")])
        assert len(buf.candidates) == 2

    def test_persistence_increments_turns(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([_c(source="a")])
        buf.add_candidates([_c(source="b")])  # Second turn
        # "a" should have 1 turn of persistence
        a_candidate = next(c for c in buf.candidates if c.source == "a")
        assert a_candidate.preconscious_turns == 1

    def test_replaces_same_source(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([_c(source="a", urgency=0.3)])
        buf.add_candidates([_c(source="a", urgency=0.8)])
        assert len(buf.candidates) == 1

    def test_max_size_limit(self) -> None:
        buf = PreconsciousBuffer(max_size=5)
        buf.add_candidates([_c(source=f"s{i}", urgency=0.5, relevance=0.5, intensity=0.5) for i in range(10)])
        assert len(buf.candidates) <= 5

    def test_get_persistent(self) -> None:
        buf = PreconsciousBuffer()
        c = _c(source="persistent")
        c.preconscious_turns = 5
        buf.candidates = [c, _c(source="fresh")]
        persistent = buf.get_persistent(min_turns=3)
        assert len(persistent) == 1
        assert persistent[0].source == "persistent"

    def test_remove_by_source(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([_c(source="a"), _c(source="b")])
        buf.remove_by_source("a")
        assert len(buf.candidates) == 1
        assert buf.candidates[0].source == "b"


# ---------------------------------------------------------------------------
# Tests: Priming
# ---------------------------------------------------------------------------

class TestPriming:
    """Tests para el priming temporal."""

    def test_no_priming_before_3_turns(self) -> None:
        buf = PreconsciousBuffer()
        c = _c(source="young")
        c.preconscious_turns = 2
        c.saliency = 0.2
        buf.candidates = [c]
        irrupting = apply_priming(buf)
        assert len(irrupting) == 0
        assert buf.candidates[0].saliency == 0.2  # No cambio

    def test_priming_after_3_turns(self) -> None:
        buf = PreconsciousBuffer()
        c = _c(source="old")
        c.preconscious_turns = 5
        c.saliency = 0.2
        buf.candidates = [c]
        irrupting = apply_priming(buf)
        assert buf.candidates[0].saliency > 0.2  # Subio

    def test_irruption_when_saliency_exceeds_threshold(self) -> None:
        buf = PreconsciousBuffer()
        c = _c(source="irrupt")
        c.preconscious_turns = 15
        c.saliency = 0.45
        buf.candidates = [c]
        irrupting = apply_priming(buf)
        assert len(irrupting) == 1
        assert irrupting[0].source == "irrupt"


# ---------------------------------------------------------------------------
# Tests: Mood Influence
# ---------------------------------------------------------------------------

class TestMoodInfluence:
    """Tests para la influencia del preconsciente en el mood."""

    def test_negative_preconscious_negative_mood(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([
            _c(source="a", emotion="anger", urgency=0.7, relevance=0.7, intensity=0.7),
            _c(source="b", emotion="fear", urgency=0.6, relevance=0.6, intensity=0.6),
        ])
        v, a = compute_mood_influence(buf)
        assert v < 0  # Preconsciente negativo → mood negativo

    def test_positive_preconscious_positive_mood(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([
            _c(source="a", emotion="joy", urgency=0.7, relevance=0.7, intensity=0.7),
        ])
        v, a = compute_mood_influence(buf)
        assert v > 0

    def test_empty_preconscious_no_influence(self) -> None:
        buf = PreconsciousBuffer()
        v, a = compute_mood_influence(buf)
        assert v == 0.0
        assert a == 0.0

    def test_influence_is_reduced(self) -> None:
        """La influencia preconsciente es menor que la consciente (30%)."""
        buf = PreconsciousBuffer()
        buf.add_candidates([
            _c(source="a", emotion="anger", urgency=0.9, relevance=0.9, intensity=0.9),
        ])
        v, _ = compute_mood_influence(buf)
        assert abs(v) < 0.3  # Reducida al 30%


# ---------------------------------------------------------------------------
# Tests: Somatic Echo
# ---------------------------------------------------------------------------

class TestSomaticEcho:
    """Tests para el eco somatico del preconsciente."""

    def test_negative_emotions_create_tension(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([
            _c(source="a", emotion="anxiety", urgency=0.8, relevance=0.8, intensity=0.8),
        ])
        tension = compute_somatic_echo(buf)
        assert tension > 0

    def test_positive_emotions_no_tension(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([
            _c(source="a", emotion="joy", urgency=0.8, relevance=0.8, intensity=0.8),
        ])
        tension = compute_somatic_echo(buf)
        assert tension == 0.0

    def test_empty_preconscious_no_echo(self) -> None:
        buf = PreconsciousBuffer()
        tension = compute_somatic_echo(buf)
        assert tension == 0.0

    def test_echo_clamped(self) -> None:
        buf = PreconsciousBuffer()
        buf.add_candidates([
            _c(source=f"s{i}", emotion="fear", urgency=1.0, relevance=1.0, intensity=1.0)
            for i in range(10)
        ])
        tension = compute_somatic_echo(buf)
        assert 0 <= tension <= 1


# ---------------------------------------------------------------------------
# Tests: Workspace Prompt
# ---------------------------------------------------------------------------

class TestWorkspacePrompt:
    """Tests para la generacion de prompts del workspace."""

    def test_generates_prompt_from_conscious(self) -> None:
        result = WorkspaceResult(
            conscious=[
                _c(source="appraisal", content="El usuario esta frustrado"),
                _c(source="schema", content="Patron de critica activo"),
            ],
        )
        prompt = get_workspace_prompt(result)
        assert prompt is not None
        assert "appraisal" in prompt
        assert "schema" in prompt
        assert "Yo consciente" in prompt

    def test_empty_workspace_no_prompt(self) -> None:
        result = WorkspaceResult()
        assert get_workspace_prompt(result) is None

    def test_low_integration_warns_fragmentation(self) -> None:
        result = WorkspaceResult(
            conscious=[
                _c(source="a", content="something"),
                _c(source="b", content="other"),
            ],
            integration_score=0.1,
        )
        prompt = get_workspace_prompt(result)
        assert prompt is not None
        assert "fragmentada" in prompt.lower()


# ---------------------------------------------------------------------------
# Tests: Full Turn Orchestration
# ---------------------------------------------------------------------------

class TestProcessWorkspaceTurn:
    """Tests de integracion del flujo completo."""

    def test_disabled_workspace_noop(self) -> None:
        state = default_consciousness_state()
        assert not state.enabled
        result = process_workspace_turn(state, [_c()])
        assert result.current_result is None

    def test_enabled_workspace_processes(self) -> None:
        state = ConsciousnessState(enabled=True)
        candidates = [
            _c(source="appraisal", urgency=0.8, relevance=0.8, intensity=0.8),
            _c(source="schema", urgency=0.5, relevance=0.5, intensity=0.5),
        ]
        state = process_workspace_turn(state, candidates)
        assert state.current_result is not None
        assert len(state.current_result.conscious) > 0

    def test_preconscious_persists_across_turns(self) -> None:
        state = ConsciousnessState(enabled=True)

        # Turn 1: 6 candidates, solo 5 entran
        candidates = [
            _c(source=f"s{i}", urgency=0.6, relevance=0.6, intensity=0.6)
            for i in range(6)
        ]
        state = process_workspace_turn(state, candidates)
        assert len(state.preconscious.candidates) >= 1

    def test_workspace_stability_tracking(self) -> None:
        state = ConsciousnessState(enabled=True)

        # Turn 1
        candidates1 = [_c(source="a", urgency=0.8, relevance=0.8, intensity=0.8)]
        state = process_workspace_turn(state, candidates1)
        assert state.previous_workspace_sources == ["a"]

        # Turn 2: same source
        candidates2 = [_c(source="a", urgency=0.8, relevance=0.8, intensity=0.8)]
        state = process_workspace_turn(state, candidates2)
        assert state.current_result is not None
        assert state.current_result.workspace_stability > 0

    def test_integration_history_grows(self) -> None:
        state = ConsciousnessState(enabled=True)
        for i in range(5):
            state = process_workspace_turn(
                state, [_c(source=f"s{i}", urgency=0.8, relevance=0.8, intensity=0.8)],
            )
        assert len(state.integration_history) == 5

    def test_irruption_from_preconscious(self) -> None:
        """Un candidato persistente en el preconsciente eventualmente irrumpe."""
        state = ConsciousnessState(enabled=True)

        # Meter un candidato en el preconsciente con alta saliency casi-threshold
        persistent = _c(source="hidden", urgency=0.5, relevance=0.5, intensity=0.5)
        persistent.preconscious_turns = 12
        persistent.saliency = 0.45
        state.preconscious.candidates = [persistent]

        # Procesar un turno con candidatos que llenen el workspace
        candidates = [
            _c(source=f"fill{i}", urgency=0.6, relevance=0.6, intensity=0.6)
            for i in range(3)
        ]
        state = process_workspace_turn(state, candidates)

        # El candidato persistente deberia haber ganado saliency via priming
        # y potencialmente entrado al workspace
        all_conscious_sources = [c.source for c in (state.current_result.conscious if state.current_result else [])]
        all_precon_sources = [c.source for c in state.preconscious.candidates]

        # Debe estar en uno u otro (no perdido)
        assert "hidden" in all_conscious_sources or "hidden" in all_precon_sources


# ---------------------------------------------------------------------------
# Tests: ConsciousnessState model
# ---------------------------------------------------------------------------

class TestConsciousnessState:
    """Tests para el modelo de estado."""

    def test_defaults(self) -> None:
        state = default_consciousness_state()
        assert not state.enabled
        assert state.current_result is None
        assert len(state.preconscious.candidates) == 0
        assert state.integration_history == []

    def test_serialization(self) -> None:
        state = ConsciousnessState(enabled=True)
        data = state.model_dump()
        restored = ConsciousnessState(**data)
        assert restored.enabled
