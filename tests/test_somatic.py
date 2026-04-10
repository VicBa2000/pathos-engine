"""Tests para Somatic Markers - marcadores emocionales en decisiones."""

import pytest

from pathos.engine.somatic import (
    categorize_stimulus,
    compute_somatic_bias,
    evaluate_user_reaction,
    register_pending_decision,
)
from pathos.models.somatic import SomaticMarker, SomaticMarkerStore, default_somatic_store


class TestCategorizeStimulus:
    """Tests para categorizacion de estimulos."""

    def test_criticism(self) -> None:
        cat, kw = categorize_stimulus("This is terrible and wrong")
        assert cat == "criticism"
        assert len(kw) > 0

    def test_praise(self) -> None:
        cat, kw = categorize_stimulus("Amazing work, excellent job!")
        assert cat == "praise"

    def test_threat(self) -> None:
        cat, kw = categorize_stimulus("I want to delete and destroy everything")
        assert cat == "threat"

    def test_challenge(self) -> None:
        cat, kw = categorize_stimulus("This is a difficult and complex problem")
        assert cat == "challenge"

    def test_connection(self) -> None:
        cat, kw = categorize_stimulus("I love you, you're my friend")
        assert cat == "connection"

    def test_injustice(self) -> None:
        cat, kw = categorize_stimulus("That's unfair, they cheat and lie")
        assert cat == "injustice"

    def test_novelty(self) -> None:
        cat, kw = categorize_stimulus("I made a new discovery, what a surprise!")
        assert cat == "novelty"

    def test_no_category(self) -> None:
        cat, kw = categorize_stimulus("The weather is nice today")
        assert cat is None
        assert kw == []

    def test_spanish(self) -> None:
        cat, _ = categorize_stimulus("Esto es terrible e inútil")
        assert cat == "criticism"


class TestRegisterPendingDecision:
    """Tests para registro de decisiones pendientes."""

    def test_registers_category(self) -> None:
        store = default_somatic_store()
        updated = register_pending_decision(store, "This problem is really difficult")
        assert updated.pending_category == "challenge"
        assert len(updated.pending_keywords) > 0

    def test_no_category_clears_pending(self) -> None:
        store = SomaticMarkerStore(pending_category="challenge", pending_keywords=["difficult"])
        updated = register_pending_decision(store, "What time is it?")
        assert updated.pending_category is None

    def test_preserves_markers(self) -> None:
        marker = SomaticMarker(stimulus_category="praise", valence_tag=0.5, strength=0.6)
        store = SomaticMarkerStore(markers=[marker])
        updated = register_pending_decision(store, "This is a challenge")
        assert len(updated.markers) == 1


class TestEvaluateUserReaction:
    """Tests para evaluacion de reaccion del usuario."""

    def test_no_pending_no_change(self) -> None:
        store = default_somatic_store()
        updated = evaluate_user_reaction(store, 0.5, 1)
        assert len(updated.markers) == 0

    def test_positive_reaction_creates_positive_marker(self) -> None:
        store = SomaticMarkerStore(
            pending_category="challenge",
            pending_keywords=["difficult"],
        )
        updated = evaluate_user_reaction(store, 0.6, 1)
        assert len(updated.markers) == 1
        assert updated.markers[0].valence_tag > 0
        assert updated.markers[0].stimulus_category == "challenge"

    def test_negative_reaction_creates_negative_marker(self) -> None:
        store = SomaticMarkerStore(
            pending_category="criticism",
            pending_keywords=["wrong"],
        )
        updated = evaluate_user_reaction(store, -0.7, 1)
        assert len(updated.markers) == 1
        assert updated.markers[0].valence_tag < 0

    def test_weak_reaction_no_marker(self) -> None:
        store = SomaticMarkerStore(
            pending_category="praise",
            pending_keywords=["great"],
        )
        updated = evaluate_user_reaction(store, 0.05, 1)  # Too weak
        assert len(updated.markers) == 0

    def test_reinforces_existing_marker(self) -> None:
        existing = SomaticMarker(
            stimulus_category="challenge",
            valence_tag=0.4,
            strength=0.5,
            reinforcement_count=2,
        )
        store = SomaticMarkerStore(
            markers=[existing],
            pending_category="challenge",
            pending_keywords=["difficult"],
        )
        updated = evaluate_user_reaction(store, 0.6, 3)
        marker = updated.markers[0]
        assert marker.strength > 0.5  # Reinforced
        assert marker.reinforcement_count == 3

    def test_contradictory_evidence_weakens(self) -> None:
        existing = SomaticMarker(
            stimulus_category="challenge",
            valence_tag=0.5,
            strength=0.6,
            reinforcement_count=3,
        )
        store = SomaticMarkerStore(
            markers=[existing],
            pending_category="challenge",
            pending_keywords=["difficult"],
        )
        updated = evaluate_user_reaction(store, -0.5, 4)
        marker = updated.markers[0]
        assert marker.strength < 0.6  # Weakened

    def test_clears_pending_after_eval(self) -> None:
        store = SomaticMarkerStore(
            pending_category="praise",
            pending_keywords=["great"],
        )
        updated = evaluate_user_reaction(store, 0.5, 1)
        assert updated.pending_category is None

    def test_max_markers_limit(self) -> None:
        markers = [
            SomaticMarker(
                stimulus_category=f"cat_{i}",
                valence_tag=0.3,
                strength=0.3,
            )
            for i in range(15)
        ]
        store = SomaticMarkerStore(
            markers=markers,
            pending_category="novelty",
            pending_keywords=["new"],
        )
        updated = evaluate_user_reaction(store, 0.8, 1)
        assert len(updated.markers) <= 15


class TestComputeSomaticBias:
    """Tests para calculo de bias somatico."""

    def test_no_markers_no_bias(self) -> None:
        store = default_somatic_store()
        bias, reason = compute_somatic_bias(store, "This is a challenge")
        assert bias == 0.0
        assert reason is None

    def test_no_matching_category(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(stimulus_category="praise", valence_tag=0.5, strength=0.6)]
        )
        bias, reason = compute_somatic_bias(store, "This is a challenge")
        assert bias == 0.0

    def test_positive_marker_positive_bias(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(stimulus_category="challenge", valence_tag=0.6, strength=0.7)]
        )
        bias, reason = compute_somatic_bias(store, "This problem is really difficult")
        assert bias > 0

    def test_negative_marker_negative_bias(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(stimulus_category="criticism", valence_tag=-0.7, strength=0.8)]
        )
        bias, reason = compute_somatic_bias(store, "You're wrong and terrible at this")
        assert bias < 0
        assert reason is not None  # Gut feeling triggered

    def test_weak_marker_no_bias(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(stimulus_category="challenge", valence_tag=0.5, strength=0.1)]
        )
        bias, _ = compute_somatic_bias(store, "This is a difficult problem")
        assert bias == 0.0  # Too weak to activate

    def test_bias_clamped(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(stimulus_category="threat", valence_tag=-1.0, strength=1.0)]
        )
        bias, _ = compute_somatic_bias(store, "I'll delete and destroy everything")
        assert -0.2 <= bias <= 0.2

    def test_no_category_no_bias(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(stimulus_category="challenge", valence_tag=0.5, strength=0.7)]
        )
        bias, _ = compute_somatic_bias(store, "The weather is nice")
        assert bias == 0.0

    def test_gut_feeling_on_strong_negative(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(
                stimulus_category="criticism",
                valence_tag=-0.6,
                strength=0.7,
                reinforcement_count=4,
            )]
        )
        _, reason = compute_somatic_bias(store, "This is terrible, you're wrong")
        assert reason is not None
        assert "negativo" in reason
        assert "criticism" in reason

    def test_gut_feeling_on_strong_positive(self) -> None:
        store = SomaticMarkerStore(
            markers=[SomaticMarker(
                stimulus_category="praise",
                valence_tag=0.6,
                strength=0.7,
                reinforcement_count=3,
            )]
        )
        _, reason = compute_somatic_bias(store, "Amazing work, excellent!")
        assert reason is not None
        assert "positivo" in reason


class TestSomaticIntegration:
    """Tests de integracion del pipeline de somatic markers."""

    def test_full_cycle_formation(self) -> None:
        """Ciclo completo: decision -> reaccion -> marcador -> bias."""
        store = default_somatic_store()

        # Turn 1: Agent responds to a challenge
        store = register_pending_decision(store, "This problem is really difficult")
        assert store.pending_category == "challenge"

        # Turn 2: User reacts positively
        store = evaluate_user_reaction(store, 0.6, 2)
        assert len(store.markers) == 1
        assert store.markers[0].valence_tag > 0

        # Turn 3: Similar challenge appears — should have positive bias
        bias, _ = compute_somatic_bias(store, "Another complex and difficult task")
        assert bias > 0

    def test_marker_reversal(self) -> None:
        """Un marcador puede revertirse con evidencia contraria repetida."""
        store = SomaticMarkerStore(
            markers=[SomaticMarker(
                stimulus_category="challenge",
                valence_tag=0.5,
                strength=0.5,
                reinforcement_count=2,
            )]
        )

        # Repeated negative reactions weaken/reverse the marker
        for turn in range(3, 7):
            store = SomaticMarkerStore(
                markers=store.markers,
                pending_category="challenge",
                pending_keywords=["difficult"],
            )
            store = evaluate_user_reaction(store, -0.6, turn)

        marker = store.markers[0]
        assert marker.strength < 0.5 or marker.valence_tag < 0.5

    def test_multiple_categories(self) -> None:
        """Se forman marcadores independientes por categoria."""
        store = default_somatic_store()

        # Form challenge marker (positive)
        store = SomaticMarkerStore(markers=store.markers, pending_category="challenge", pending_keywords=["difficult"])
        store = evaluate_user_reaction(store, 0.6, 1)

        # Form criticism marker (negative)
        store = SomaticMarkerStore(markers=store.markers, pending_category="criticism", pending_keywords=["wrong"])
        store = evaluate_user_reaction(store, -0.7, 2)

        assert len(store.markers) == 2

        # Challenge should have positive bias
        bias_c, _ = compute_somatic_bias(store, "This is a difficult problem")
        assert bias_c > 0

        # Criticism should have negative bias
        bias_cr, _ = compute_somatic_bias(store, "You're wrong about this")
        assert bias_cr < 0


class TestDefaultSomaticStore:
    """Tests para el store por defecto."""

    def test_empty_defaults(self) -> None:
        s = default_somatic_store()
        assert len(s.markers) == 0
        assert s.pending_category is None
        assert s.pending_keywords == []
