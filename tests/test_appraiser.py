"""Tests para el Appraisal Module (enfoque hibrido: emotion + valence + arousal)."""

from pathos.engine.appraiser import _build_appraisal, _parse_response


# --- Tests de parsing ---


def test_parse_clean_json() -> None:
    raw = '{"emotion": "anger", "valence": -0.8, "arousal": 0.9}'
    result = _parse_response(raw)
    assert result["emotion"] == "anger"
    assert result["valence"] == -0.8
    assert result["arousal"] == 0.9


def test_parse_markdown_wrapped() -> None:
    raw = '```json\n{"emotion": "joy", "valence": 0.5, "arousal": 0.3}\n```'
    result = _parse_response(raw)
    assert result["emotion"] == "joy"
    assert result["valence"] == 0.5


def test_parse_with_thinking_tags() -> None:
    raw = '<think>Let me analyze...</think>{"emotion": "fear", "valence": -0.5, "arousal": 0.7}'
    result = _parse_response(raw)
    assert result["emotion"] == "fear"
    assert result["valence"] == -0.5


def test_parse_clamps_values() -> None:
    raw = '{"emotion": "joy", "valence": 2.0, "arousal": -0.5}'
    result = _parse_response(raw)
    assert result["valence"] == 1.0
    assert result["arousal"] == 0.0


def test_parse_invalid_json_returns_neutral() -> None:
    raw = "This is not JSON at all"
    result = _parse_response(raw)
    assert result["emotion"] == "neutral"
    assert result["valence"] == 0.0
    assert result["arousal"] == 0.3


def test_parse_unknown_emotion_defaults_neutral() -> None:
    raw = '{"emotion": "rage", "valence": -0.9, "arousal": 0.9}'
    result = _parse_response(raw)
    assert result["emotion"] == "neutral"


def test_parse_missing_emotion_defaults_neutral() -> None:
    raw = '{"valence": 0.5, "arousal": 0.3}'
    result = _parse_response(raw)
    assert result["emotion"] == "neutral"


def test_parse_extra_text_around_json() -> None:
    raw = 'Here: {"emotion": "sadness", "valence": -0.6, "arousal": 0.2} done'
    result = _parse_response(raw)
    assert result["emotion"] == "sadness"
    assert result["valence"] == -0.6


# --- Tests de build_appraisal ---


def test_build_positive_emotion() -> None:
    result = _build_appraisal("joy", 0.8, 0.6)
    assert result.valence.goal_conduciveness == 0.8
    assert result.valence.intrinsic_pleasantness == 0.8
    assert result.coping.control == 0.7  # joy default


def test_build_negative_emotion() -> None:
    result = _build_appraisal("anger", -0.8, 0.85)
    assert result.valence.goal_conduciveness == -0.8
    assert result.coping.control == 0.4  # anger default
    assert result.agency.fairness == -0.8  # anger default


def test_build_neutral_emotion() -> None:
    result = _build_appraisal("neutral", 0.0, 0.1)
    assert abs(result.valence.goal_conduciveness) < 0.01
    assert result.coping.control == 0.5  # neutral default


def test_build_unknown_emotion_uses_defaults() -> None:
    result = _build_appraisal("unknown_emotion", 0.0, 0.5)
    assert result.coping.control == 0.5  # fallback default


def test_build_all_values_in_range() -> None:
    """All expanded values should be within their defined ranges."""
    emotions = ["joy", "anger", "fear", "sadness", "excitement", "neutral"]
    for emotion in emotions:
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for a in [0.0, 0.5, 1.0]:
                result = _build_appraisal(emotion, v, a)
                assert -1 <= result.relevance.novelty <= 1
                assert 0 <= result.relevance.personal_significance <= 1
                assert -1 <= result.valence.goal_conduciveness <= 1
                assert 0 <= result.coping.control <= 1
                assert 0 <= result.coping.power <= 1
                assert 0 <= result.coping.adjustability <= 1
                assert -1 <= result.agency.fairness <= 1
