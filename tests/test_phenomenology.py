"""Tests para Pilar 7: Fenomenología Computacional — Qualia Funcionales."""

import math

import pytest

from pathos.engine.phenomenology import (
    _get_quadrant,
    _hsl_to_rgb,
    _hue_from_valence,
    _select_template,
    _TEXTURE_TEMPLATES,
    _SOUND_TEMPLATES,
    _MOVEMENT_TEMPLATES,
    _TEMPORALITY_TEMPLATES,
    _METAPHOR_TEMPLATES,
    _RAW_TEXTURE_TEMPLATES,
    _RAW_METAPHOR_TEMPLATES,
    _EXTREME_TEXTURE_TEMPLATES,
    _EXTREME_METAPHOR_TEMPLATES,
    compare_qualia,
    emotion_to_color,
    generate_profile,
    generate_template_fields,
    get_llm_prompt_for_qualia,
    get_phenomenology_details,
    get_phenomenology_prompt,
    get_qualia_evolution,
    get_qualia_history_summary,
    parse_llm_qualia_response,
    process_phenomenology_turn,
    record_qualia,
)
from pathos.models.phenomenology import (
    MAX_QUALIA_RECORDS_PER_EMOTION,
    MAX_TRACKED_EMOTIONS,
    PhenomenologicalProfile,
    PhenomenologyState,
    QualiaHistory,
    QualiaRecord,
    default_phenomenology_state,
)


# --- Helpers ---

def _enabled_state() -> PhenomenologyState:
    state = default_phenomenology_state()
    state.enabled = True
    return state


def _make_profile(**kwargs) -> PhenomenologicalProfile:
    defaults = dict(
        emotion_name="joy", valence=0.7, arousal=0.6, dominance=0.7,
        certainty=0.7, intensity=0.8, turn=1,
    )
    defaults.update(kwargs)
    return generate_profile(**defaults)


# ===============================================================
# Test Models
# ===============================================================

class TestPhenomenologicalProfile:
    def test_default_values(self):
        p = PhenomenologicalProfile()
        assert p.color_r == 128
        assert p.color_g == 128
        assert p.color_b == 128
        assert p.weight == 0.5
        assert p.temperature == 0.5
        assert p.emotion_name == "neutral"
        assert p.generated_by_llm is False

    def test_field_constraints(self):
        p = PhenomenologicalProfile(color_r=255, color_g=0, color_b=0, weight=1.0, temperature=0.0)
        assert p.color_r == 255
        assert p.weight == 1.0
        assert p.temperature == 0.0

    def test_invalid_color_rejected(self):
        with pytest.raises(Exception):
            PhenomenologicalProfile(color_r=256)
        with pytest.raises(Exception):
            PhenomenologicalProfile(color_r=-1)

    def test_invalid_weight_rejected(self):
        with pytest.raises(Exception):
            PhenomenologicalProfile(weight=1.5)

    def test_serialization_roundtrip(self):
        p = PhenomenologicalProfile(
            color_r=200, color_g=100, color_b=50,
            weight=0.3, temperature=0.8,
            texture="warm silk", sound="humming",
            movement="expansion", temporality="time stretches",
            metaphor="sunrise after storm",
            emotion_name="joy", turn=5, intensity=0.7,
            generated_by_llm=True,
        )
        data = p.model_dump()
        p2 = PhenomenologicalProfile(**data)
        assert p2.texture == "warm silk"
        assert p2.generated_by_llm is True
        assert p2.color_r == 200


class TestQualiaRecord:
    def test_default_values(self):
        r = QualiaRecord(emotion_name="sadness")
        assert r.turn == 0
        assert r.metaphor == ""
        assert r.intensity == 0.5

    def test_serialization(self):
        r = QualiaRecord(emotion_name="fear", turn=10, metaphor="cliff edge", intensity=0.9)
        data = r.model_dump()
        r2 = QualiaRecord(**data)
        assert r2.metaphor == "cliff edge"


class TestQualiaHistory:
    def test_empty_history(self):
        h = QualiaHistory(emotion_name="joy")
        assert h.count == 0

    def test_count_tracks_records(self):
        h = QualiaHistory(
            emotion_name="joy",
            records=[QualiaRecord(emotion_name="joy", turn=i) for i in range(5)],
        )
        assert h.count == 5


class TestPhenomenologyState:
    def test_default_state(self):
        s = default_phenomenology_state()
        assert s.enabled is False
        assert s.current_profile is None
        assert s.qualia_histories == {}
        assert s.total_profiles_generated == 0

    def test_serialization_roundtrip(self):
        s = _enabled_state()
        s.total_profiles_generated = 10
        s.qualia_histories["joy"] = QualiaHistory(
            emotion_name="joy",
            records=[QualiaRecord(emotion_name="joy", turn=1, metaphor="light")],
        )
        data = s.model_dump()
        s2 = PhenomenologyState(**data)
        assert s2.enabled is True
        assert s2.total_profiles_generated == 10
        assert "joy" in s2.qualia_histories
        assert s2.qualia_histories["joy"].records[0].metaphor == "light"


# ===============================================================
# Test Color Generation
# ===============================================================

class TestHueFromValence:
    def test_negative_valence_cold_hue(self):
        hue = _hue_from_valence(-1.0)
        assert hue == 240  # blue

    def test_positive_valence_warm_hue(self):
        hue = _hue_from_valence(1.0)
        assert hue == 0  # red-gold

    def test_neutral_valence_green(self):
        hue = _hue_from_valence(0.0)
        assert hue == 60  # yellow-green

    def test_interpolation(self):
        hue = _hue_from_valence(-0.5)
        assert hue == 200  # steel blue

    def test_clamped_beyond_range(self):
        hue_low = _hue_from_valence(-2.0)
        hue_high = _hue_from_valence(2.0)
        assert hue_low == 240
        assert hue_high == 0


class TestHSLToRGB:
    def test_pure_red(self):
        r, g, b = _hsl_to_rgb(0, 1.0, 0.5)
        assert r == 255
        assert g == 0
        assert b == 0

    def test_pure_green(self):
        r, g, b = _hsl_to_rgb(120, 1.0, 0.5)
        assert r == 0
        assert g == 255
        assert b == 0

    def test_black(self):
        r, g, b = _hsl_to_rgb(0, 0.0, 0.0)
        assert r == 0 and g == 0 and b == 0

    def test_white(self):
        r, g, b = _hsl_to_rgb(0, 0.0, 1.0)
        assert r == 255 and g == 255 and b == 255

    def test_values_clamped(self):
        r, g, b = _hsl_to_rgb(60, 1.0, 0.5)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


class TestEmotionToColor:
    def test_positive_high_arousal_warm(self):
        r, g, b = emotion_to_color(0.8, 0.9, 0.7)
        # Should be warm-ish (high red component)
        assert r > g  # warm colors have more red

    def test_negative_low_arousal_cool(self):
        r, g, b = emotion_to_color(-0.8, 0.2, 0.3)
        # Should be cool (higher blue)
        assert b > r

    def test_neutral_state(self):
        r, g, b = emotion_to_color(0.0, 0.3, 0.5)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_different_states_different_colors(self):
        c1 = emotion_to_color(0.8, 0.9, 0.9)
        c2 = emotion_to_color(-0.8, 0.1, 0.2)
        assert c1 != c2

    def test_intensity_affects_lightness(self):
        _, _, b_low = emotion_to_color(-0.5, 0.5, 0.1)
        _, _, b_high = emotion_to_color(-0.5, 0.5, 0.9)
        # Higher intensity should produce brighter colors overall
        # (harder to test component-wise, but sum should differ)
        r_low, g_low, b_low = emotion_to_color(0.0, 0.5, 0.1)
        r_high, g_high, b_high = emotion_to_color(0.0, 0.5, 0.9)
        assert (r_high + g_high + b_high) > (r_low + g_low + b_low)


# ===============================================================
# Test Quadrant Classification
# ===============================================================

class TestQuadrantClassification:
    def test_positive_high(self):
        assert _get_quadrant(0.5, 0.7) == "pos_high"

    def test_positive_low(self):
        assert _get_quadrant(0.5, 0.3) == "pos_low"

    def test_negative_high(self):
        assert _get_quadrant(-0.5, 0.7) == "neg_high"

    def test_negative_low(self):
        assert _get_quadrant(-0.5, 0.3) == "neg_low"

    def test_neutral(self):
        assert _get_quadrant(0.0, 0.2) == "neutral"

    def test_near_zero_high_arousal_not_neutral(self):
        assert _get_quadrant(0.1, 0.6) != "neutral"


# ===============================================================
# Test Template Generation
# ===============================================================

class TestTemplateGeneration:
    def test_all_fields_populated(self):
        fields = generate_template_fields(0.5, 0.7, turn=0)
        assert fields["texture"] != ""
        assert fields["sound"] != ""
        assert fields["movement"] != ""
        assert fields["temporality"] != ""
        assert fields["metaphor"] != ""

    def test_different_turns_vary(self):
        f0 = generate_template_fields(0.5, 0.7, turn=0)
        f1 = generate_template_fields(0.5, 0.7, turn=1)
        # At least some fields should differ (templates have 3 options)
        different = sum(1 for k in f0 if f0[k] != f1[k])
        assert different >= 1

    def test_raw_mode_uses_raw_templates(self):
        fields = generate_template_fields(-0.5, 0.7, turn=0, mode="raw")
        # Raw templates exist for this quadrant
        raw_textures = _RAW_TEXTURE_TEMPLATES.get("neg_high", [])
        assert fields["texture"] in raw_textures

    def test_extreme_mode_uses_extreme_templates(self):
        fields = generate_template_fields(0.5, 0.7, turn=0, mode="extreme")
        extreme_textures = _EXTREME_TEXTURE_TEMPLATES.get("pos_high", [])
        assert fields["texture"] in extreme_textures

    def test_normal_mode_uses_base_templates(self):
        fields = generate_template_fields(0.5, 0.7, turn=0, mode="normal")
        base_textures = _TEXTURE_TEMPLATES.get("pos_high", [])
        assert fields["texture"] in base_textures

    def test_all_quadrants_have_templates(self):
        for quadrant in ("pos_high", "pos_low", "neg_high", "neg_low", "neutral"):
            assert quadrant in _TEXTURE_TEMPLATES
            assert quadrant in _SOUND_TEMPLATES
            assert quadrant in _MOVEMENT_TEMPLATES
            assert quadrant in _TEMPORALITY_TEMPLATES
            assert quadrant in _METAPHOR_TEMPLATES

    def test_all_raw_quadrants_have_templates(self):
        for quadrant in ("pos_high", "pos_low", "neg_high", "neg_low", "neutral"):
            assert quadrant in _RAW_TEXTURE_TEMPLATES
            assert quadrant in _RAW_METAPHOR_TEMPLATES

    def test_all_extreme_quadrants_have_templates(self):
        for quadrant in ("pos_high", "pos_low", "neg_high", "neg_low", "neutral"):
            assert quadrant in _EXTREME_TEXTURE_TEMPLATES
            assert quadrant in _EXTREME_METAPHOR_TEMPLATES


class TestSelectTemplate:
    def test_deterministic_with_same_turn(self):
        t1 = _select_template(_TEXTURE_TEMPLATES, "pos_high", 5)
        t2 = _select_template(_TEXTURE_TEMPLATES, "pos_high", 5)
        assert t1 == t2

    def test_wraps_around(self):
        options = _TEXTURE_TEMPLATES["pos_high"]
        for i in range(len(options)):
            t = _select_template(_TEXTURE_TEMPLATES, "pos_high", i)
            assert t == options[i]
        # Wrap
        t_wrap = _select_template(_TEXTURE_TEMPLATES, "pos_high", len(options))
        assert t_wrap == options[0]

    def test_missing_quadrant_falls_back(self):
        t = _select_template(_TEXTURE_TEMPLATES, "nonexistent", 0)
        # Should fallback to neutral
        assert t in _TEXTURE_TEMPLATES["neutral"]


# ===============================================================
# Test Profile Generation
# ===============================================================

class TestProfileGeneration:
    def test_basic_generation(self):
        p = generate_profile(
            emotion_name="joy", valence=0.7, arousal=0.6,
            dominance=0.7, certainty=0.7, intensity=0.8, turn=1,
        )
        assert p.emotion_name == "joy"
        assert p.turn == 1
        assert 0 <= p.color_r <= 255
        assert p.weight == pytest.approx(0.3, abs=0.01)  # 1 - 0.7
        assert p.texture != ""
        assert p.generated_by_llm is False

    def test_weight_from_dominance(self):
        p_high = generate_profile("test", 0, 0.5, 0.9, 0.5, 0.5)
        p_low = generate_profile("test", 0, 0.5, 0.1, 0.5, 0.5)
        assert p_high.weight < p_low.weight  # high dominance = low weight

    def test_temperature_from_body_warmth(self):
        p_cold = generate_profile("test", 0, 0.5, 0.5, 0.5, 0.5, body_warmth=0.1)
        p_hot = generate_profile("test", 0, 0.5, 0.5, 0.5, 0.5, body_warmth=0.9)
        assert p_cold.temperature < p_hot.temperature

    def test_llm_fields_override_templates(self):
        llm = {"texture": "custom texture", "metaphor": "custom metaphor",
               "sound": "", "movement": "", "temporality": ""}
        p = generate_profile("joy", 0.7, 0.6, 0.7, 0.7, 0.8, llm_fields=llm)
        assert p.texture == "custom texture"
        assert p.metaphor == "custom metaphor"
        assert p.generated_by_llm is True

    def test_empty_llm_fields_use_templates(self):
        llm = {"texture": "", "metaphor": "", "sound": "", "movement": "", "temporality": ""}
        p = generate_profile("joy", 0.7, 0.6, 0.7, 0.7, 0.8, llm_fields=llm)
        assert p.generated_by_llm is False  # all empty = template

    def test_none_llm_fields_use_templates(self):
        p = generate_profile("joy", 0.7, 0.6, 0.7, 0.7, 0.8, llm_fields=None)
        assert p.generated_by_llm is False

    def test_raw_mode(self):
        p = generate_profile("anger", -0.7, 0.8, 0.7, 0.6, 0.9, mode="raw")
        raw_textures = _RAW_TEXTURE_TEMPLATES.get("neg_high", [])
        assert p.texture in raw_textures

    def test_extreme_mode(self):
        p = generate_profile("anger", -0.7, 0.8, 0.7, 0.6, 0.9, mode="extreme")
        extreme_textures = _EXTREME_TEXTURE_TEMPLATES.get("neg_high", [])
        assert p.texture in extreme_textures

    def test_intensity_clamped(self):
        p = generate_profile("test", 0, 0.5, 0.5, 0.5, 1.5)
        assert p.intensity <= 1.0
        p2 = generate_profile("test", 0, 0.5, 0.5, 0.5, -0.5)
        assert p2.intensity >= 0.0


# ===============================================================
# Test LLM Prompt Generation
# ===============================================================

class TestLLMPrompt:
    def test_basic_prompt(self):
        prompt = get_llm_prompt_for_qualia(
            "joy", 0.7, 0.6, 0.7, 0.8, 0.3, 0.6,
        )
        assert "joy" in prompt
        assert "TEXTURE:" in prompt
        assert "METAPHOR:" in prompt

    def test_raw_mode_instruction(self):
        prompt = get_llm_prompt_for_qualia(
            "anger", -0.7, 0.8, 0.7, 0.9, 0.8, 0.3, mode="raw",
        )
        assert "Visceral" in prompt

    def test_extreme_mode_instruction(self):
        prompt = get_llm_prompt_for_qualia(
            "fear", -0.8, 0.9, 0.2, 0.9, 0.9, 0.2, mode="extreme",
        )
        assert "Maximum intensity" in prompt

    def test_history_included(self):
        prompt = get_llm_prompt_for_qualia(
            "joy", 0.7, 0.6, 0.7, 0.8, 0.3, 0.6,
            qualia_history_summary="Turn 1: 'warm light'",
        )
        assert "EVOLUTION" in prompt
        assert "warm light" in prompt

    def test_context_included(self):
        prompt = get_llm_prompt_for_qualia(
            "joy", 0.7, 0.6, 0.7, 0.8, 0.3, 0.6,
            context="user shared good news",
        )
        assert "good news" in prompt


class TestParseLLMResponse:
    def test_valid_response(self):
        response = (
            "TEXTURE: warm silk against the chest\n"
            "SOUND: a gentle hum rising\n"
            "MOVEMENT: expanding from center\n"
            "TEMPORALITY: time slows down\n"
            "METAPHOR: sunrise after a long night"
        )
        fields = parse_llm_qualia_response(response)
        assert fields["texture"] == "warm silk against the chest"
        assert fields["sound"] == "a gentle hum rising"
        assert fields["metaphor"] == "sunrise after a long night"

    def test_missing_fields_empty(self):
        response = "TEXTURE: just this\nMETAPHOR: and this"
        fields = parse_llm_qualia_response(response)
        assert fields["texture"] == "just this"
        assert fields["metaphor"] == "and this"
        assert fields["sound"] == ""
        assert fields["movement"] == ""

    def test_empty_response(self):
        fields = parse_llm_qualia_response("")
        assert all(v == "" for v in fields.values())

    def test_case_insensitive(self):
        response = "texture: lower case\nMetaphor: Mixed case"
        fields = parse_llm_qualia_response(response)
        assert fields["texture"] == "lower case"
        assert fields["metaphor"] == "Mixed case"


# ===============================================================
# Test Qualia Recording
# ===============================================================

class TestQualiaRecording:
    def test_record_creates_history(self):
        state = _enabled_state()
        profile = _make_profile(emotion_name="joy")
        record_qualia(state, profile)
        assert "joy" in state.qualia_histories
        assert state.qualia_histories["joy"].count == 1
        assert state.total_profiles_generated == 1

    def test_multiple_records_same_emotion(self):
        state = _enabled_state()
        for i in range(5):
            p = _make_profile(emotion_name="joy", turn=i)
            record_qualia(state, p)
        assert state.qualia_histories["joy"].count == 5
        assert state.total_profiles_generated == 5

    def test_multiple_emotions(self):
        state = _enabled_state()
        record_qualia(state, _make_profile(emotion_name="joy"))
        record_qualia(state, _make_profile(emotion_name="sadness", valence=-0.7))
        assert state.total_unique_emotions_profiled == 2

    def test_rolling_buffer(self):
        state = _enabled_state()
        for i in range(MAX_QUALIA_RECORDS_PER_EMOTION + 10):
            record_qualia(state, _make_profile(emotion_name="joy", turn=i))
        assert state.qualia_histories["joy"].count == MAX_QUALIA_RECORDS_PER_EMOTION

    def test_eviction_when_max_emotions_reached(self):
        state = _enabled_state()
        # Fill to max
        for i in range(MAX_TRACKED_EMOTIONS):
            emo = f"emotion_{i}"
            record_qualia(state, _make_profile(emotion_name=emo))
        assert len(state.qualia_histories) == MAX_TRACKED_EMOTIONS

        # Add one more — should evict the least recorded
        record_qualia(state, _make_profile(emotion_name="new_emotion"))
        assert len(state.qualia_histories) == MAX_TRACKED_EMOTIONS
        assert "new_emotion" in state.qualia_histories


# ===============================================================
# Test Qualia Evolution
# ===============================================================

class TestQualiaEvolution:
    def test_empty_evolution(self):
        state = _enabled_state()
        assert get_qualia_evolution(state, "joy") == []

    def test_returns_records_in_order(self):
        state = _enabled_state()
        for i in range(3):
            record_qualia(state, _make_profile(emotion_name="joy", turn=i))
        records = get_qualia_evolution(state, "joy")
        assert len(records) == 3
        assert records[0].turn == 0
        assert records[2].turn == 2

    def test_history_summary_empty(self):
        state = _enabled_state()
        assert get_qualia_history_summary(state, "joy") == ""

    def test_history_summary_content(self):
        state = _enabled_state()
        for i in range(5):
            p = generate_profile("joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=i)
            record_qualia(state, p)
        summary = get_qualia_history_summary(state, "joy", max_entries=2)
        assert "Turn" in summary
        # Should have 2 entries
        assert summary.count("Turn") == 2


class TestCompareQualia:
    def test_not_enough_history(self):
        state = _enabled_state()
        result = compare_qualia(state, "joy")
        assert "Not enough" in result

    def test_single_record(self):
        state = _enabled_state()
        record_qualia(state, _make_profile(emotion_name="joy", turn=1))
        result = compare_qualia(state, "joy")
        assert "Not enough" in result

    def test_comparison_with_history(self):
        state = _enabled_state()
        # Early: low intensity
        p1 = generate_profile("joy", 0.3, 0.3, 0.5, 0.5, 0.3, turn=1)
        record_qualia(state, p1)
        # Late: high intensity
        p2 = generate_profile("joy", 0.8, 0.8, 0.5, 0.5, 0.9, turn=50)
        record_qualia(state, p2)

        result = compare_qualia(state, "joy", early_turn=0, late_turn=100)
        assert "intensified" in result
        assert "turn 1" in result
        assert "to 50" in result

    def test_softened_comparison(self):
        state = _enabled_state()
        p1 = generate_profile("sadness", -0.7, 0.5, 0.3, 0.5, 0.9, turn=1)
        record_qualia(state, p1)
        p2 = generate_profile("sadness", -0.3, 0.3, 0.5, 0.5, 0.3, turn=50)
        record_qualia(state, p2)
        result = compare_qualia(state, "sadness")
        assert "softened" in result


# ===============================================================
# Test Prompt Generation
# ===============================================================

class TestPhenomenologyPrompt:
    def test_disabled_returns_empty(self):
        state = default_phenomenology_state()
        assert get_phenomenology_prompt(state) == ""

    def test_enabled_no_profile_returns_empty(self):
        state = _enabled_state()
        assert get_phenomenology_prompt(state) == ""

    def test_basic_prompt_content(self):
        state = _enabled_state()
        state.current_profile = _make_profile()
        prompt = get_phenomenology_prompt(state)
        assert "PHENOMENOLOGICAL EXPERIENCE" in prompt
        assert "Color" in prompt
        assert "Weight" in prompt
        assert "Temperature" in prompt

    def test_prompt_includes_texture(self):
        state = _enabled_state()
        state.current_profile = _make_profile()
        prompt = get_phenomenology_prompt(state)
        assert "Texture:" in prompt

    def test_prompt_includes_metaphor(self):
        state = _enabled_state()
        state.current_profile = _make_profile()
        prompt = get_phenomenology_prompt(state)
        assert "Core experience:" in prompt

    def test_history_note_appears(self):
        state = _enabled_state()
        # Build some history
        for i in range(5):
            process_phenomenology_turn(
                state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=i,
            )
        prompt = get_phenomenology_prompt(state)
        assert "experienced joy" in prompt
        assert "evolved" in prompt


# ===============================================================
# Test Details for Research Endpoint
# ===============================================================

class TestPhenomenologyDetails:
    def test_disabled_details(self):
        state = default_phenomenology_state()
        details = get_phenomenology_details(state)
        assert details["enabled"] is False
        assert details["current_profile"] is None

    def test_enabled_with_profile(self):
        state = _enabled_state()
        state.current_profile = _make_profile()
        details = get_phenomenology_details(state)
        assert details["enabled"] is True
        assert details["current_profile"] is not None
        assert "color" in details["current_profile"]
        assert "metaphor" in details["current_profile"]

    def test_qualia_evolution_tracked(self):
        state = _enabled_state()
        process_phenomenology_turn(state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=1)
        process_phenomenology_turn(state, "sadness", -0.7, 0.2, 0.3, 0.6, 0.6, turn=2)
        details = get_phenomenology_details(state)
        assert details["unique_emotions_profiled"] == 2
        assert details["total_profiles_generated"] == 2
        assert "joy" in details["qualia_evolution"]
        assert "sadness" in details["qualia_evolution"]


# ===============================================================
# Test Orchestration
# ===============================================================

class TestProcessPhenomenologyTurn:
    def test_generates_and_records(self):
        state = _enabled_state()
        profile = process_phenomenology_turn(
            state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=1,
        )
        assert profile.emotion_name == "joy"
        assert state.current_profile is profile
        assert state.total_profiles_generated == 1
        assert "joy" in state.qualia_histories

    def test_multiple_turns(self):
        state = _enabled_state()
        for i in range(10):
            process_phenomenology_turn(
                state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=i,
            )
        assert state.total_profiles_generated == 10
        assert state.qualia_histories["joy"].count == 10

    def test_different_emotions(self):
        state = _enabled_state()
        process_phenomenology_turn(state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=1)
        process_phenomenology_turn(state, "fear", -0.7, 0.9, 0.2, 0.2, 0.9, turn=2)
        assert state.total_unique_emotions_profiled == 2

    def test_with_llm_fields(self):
        state = _enabled_state()
        llm = {"texture": "custom", "metaphor": "unique", "sound": "", "movement": "", "temporality": ""}
        profile = process_phenomenology_turn(
            state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8,
            turn=1, llm_fields=llm,
        )
        assert profile.generated_by_llm is True
        assert profile.texture == "custom"

    def test_mode_propagation(self):
        state = _enabled_state()
        profile = process_phenomenology_turn(
            state, "anger", -0.7, 0.8, 0.7, 0.6, 0.9,
            turn=1, mode="raw",
        )
        raw_textures = _RAW_TEXTURE_TEMPLATES.get("neg_high", [])
        assert profile.texture in raw_textures


# ===============================================================
# Test Full Flow
# ===============================================================

class TestFullFlow:
    def test_complete_session(self):
        """Simulate a multi-turn session with evolving emotions."""
        state = _enabled_state()

        # Turn 1: joy
        p1 = process_phenomenology_turn(state, "joy", 0.7, 0.6, 0.7, 0.7, 0.8, turn=1)
        assert p1.color_r > p1.color_b  # warm color

        # Turn 2: sadness
        p2 = process_phenomenology_turn(state, "sadness", -0.7, 0.2, 0.3, 0.6, 0.6, turn=2)
        assert p2.weight > p1.weight  # sadness heavier (lower dominance)

        # Turn 3: joy again
        p3 = process_phenomenology_turn(state, "joy", 0.8, 0.7, 0.6, 0.8, 0.9, turn=3)

        # Check history
        joy_records = get_qualia_evolution(state, "joy")
        assert len(joy_records) == 2  # turns 1 and 3
        assert joy_records[0].turn == 1
        assert joy_records[1].turn == 3

        # Check comparison
        comparison = compare_qualia(state, "joy")
        assert "joy" in comparison

        # Check details
        details = get_phenomenology_details(state)
        assert details["total_profiles_generated"] == 3
        assert details["unique_emotions_profiled"] == 2

        # Check prompt
        prompt = get_phenomenology_prompt(state)
        assert "PHENOMENOLOGICAL" in prompt

    def test_values_always_clamped(self):
        """Ensure all values stay in range regardless of input."""
        state = _enabled_state()
        p = process_phenomenology_turn(
            state, "extreme", 2.0, 2.0, -1.0, -1.0, 3.0,
            body_tension=2.0, body_warmth=-1.0, turn=1,
        )
        assert 0 <= p.color_r <= 255
        assert 0 <= p.color_g <= 255
        assert 0 <= p.color_b <= 255
        assert 0.0 <= p.weight <= 1.0
        assert 0.0 <= p.temperature <= 1.0
        assert 0.0 <= p.intensity <= 1.0
