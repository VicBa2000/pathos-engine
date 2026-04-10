"""Tests for Emotional Schemas engine."""

import pytest

from pathos.engine.emotional_schemas import SchemaStore, EmotionalSchema
from pathos.models.emotion import PrimaryEmotion


class TestStimulusCategorization:
    def test_criticism(self):
        store = SchemaStore()
        assert store.categorize_stimulus("That was a terrible mistake") == "criticism"

    def test_praise(self):
        store = SchemaStore()
        assert store.categorize_stimulus("That was excellent work!") == "praise"

    def test_threat(self):
        store = SchemaStore()
        assert store.categorize_stimulus("We should shutdown the system") == "threat"

    def test_unknown(self):
        store = SchemaStore()
        assert store.categorize_stimulus("the sky is blue") is None


class TestSchemaFormation:
    def test_no_schema_before_threshold(self):
        store = SchemaStore()
        result = store.record_pattern("terrible mistake", PrimaryEmotion.ANGER, 0.7)
        assert result is None
        result = store.record_pattern("awful error", PrimaryEmotion.ANGER, 0.7)
        assert result is None

    def test_schema_forms_at_threshold(self):
        store = SchemaStore()
        store.record_pattern("terrible mistake", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("awful error", PrimaryEmotion.ANGER, 0.8)
        schema = store.record_pattern("bad wrong useless", PrimaryEmotion.ANGER, 0.6)
        assert schema is not None
        assert schema.trigger_category == "criticism"
        assert schema.typical_emotion == PrimaryEmotion.ANGER

    def test_schema_typical_intensity(self):
        store = SchemaStore()
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.6)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.8)
        schema = store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.7)
        assert schema is not None
        # Average of 0.6, 0.8, 0.7 = 0.7
        assert 0.6 <= schema.typical_intensity <= 0.8


class TestSchemaReinforcement:
    def test_reinforcement_same_emotion(self):
        store = SchemaStore()
        # Form schema
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.7)
        # Reinforce
        initial_strength = store.schemas[0].reinforcement_strength
        store.record_pattern("wrong mistake", PrimaryEmotion.ANGER, 0.8)
        assert store.schemas[0].reinforcement_strength > initial_strength

    def test_weakening_different_emotion(self):
        store = SchemaStore()
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.7)
        initial_strength = store.schemas[0].reinforcement_strength
        store.record_pattern("wrong mistake", PrimaryEmotion.SADNESS, 0.5)
        assert store.schemas[0].reinforcement_strength < initial_strength


class TestSchemaPriming:
    def test_priming_with_schema(self):
        store = SchemaStore()
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.7)
        emotion, amp = store.check_priming("another mistake was terrible")
        assert emotion == PrimaryEmotion.ANGER
        assert amp > 0.0

    def test_no_priming_without_schema(self):
        store = SchemaStore()
        emotion, amp = store.check_priming("a nice day")
        assert emotion is None
        assert amp == 0.0

    def test_amplification_capped(self):
        store = SchemaStore()
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.7)
        store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.7)
        _, amp = store.check_priming("terrible mistake")
        assert amp <= 0.4


class TestMaladaptiveDetection:
    def test_maladaptive_after_repeated_intense_negative(self):
        store = SchemaStore()
        # Form schema
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.8)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.8)
        store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.8)
        # Reinforce past threshold
        for _ in range(5):
            store.record_pattern(f"wrong mistake", PrimaryEmotion.ANGER, 0.8)
        assert store.schemas[0].adaptive is False

    def test_positive_schema_stays_adaptive(self):
        store = SchemaStore()
        store.record_pattern("excellent work", PrimaryEmotion.JOY, 0.8)
        store.record_pattern("great job amazing", PrimaryEmotion.JOY, 0.8)
        store.record_pattern("wonderful perfect", PrimaryEmotion.JOY, 0.8)
        for _ in range(5):
            store.record_pattern("brilliant excellent", PrimaryEmotion.JOY, 0.8)
        assert store.schemas[0].adaptive is True

    def test_maladaptive_amplifies_more(self):
        store = SchemaStore()
        store.record_pattern("terrible", PrimaryEmotion.ANGER, 0.8)
        store.record_pattern("awful", PrimaryEmotion.ANGER, 0.8)
        store.record_pattern("bad error", PrimaryEmotion.ANGER, 0.8)
        _, normal_amp = store.check_priming("terrible mistake")

        for _ in range(5):
            store.record_pattern(f"wrong mistake", PrimaryEmotion.ANGER, 0.8)
        _, maladaptive_amp = store.check_priming("terrible mistake")
        assert maladaptive_amp > normal_amp
