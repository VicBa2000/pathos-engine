"""Tests for Emotion API request/response models."""

import pytest

from pathos.models.emotion_api import (
    EmotionAPIConfig,
    EmotionBatchRequest,
    EmotionConfigureRequest,
    EmotionConfigureResponse,
    EmotionHealthResponse,
    EmotionPresetsResponse,
    EmotionProcessRequest,
    EmotionProcessResponse,
    EmotionStateResponse,
    ExternalSignal,
    ExternalSignalContribution,
    PersonalityPreset,
)


class TestEmotionProcessRequest:
    def test_minimal_request(self) -> None:
        req = EmotionProcessRequest(stimulus="Hello world")
        assert req.stimulus == "Hello world"
        assert req.session_id == "api-default"
        assert req.personality is None
        assert req.external_signals == []

    def test_full_request(self) -> None:
        req = EmotionProcessRequest(
            stimulus="I feel great today!",
            session_id="api-test-123",
            personality={"openness": 0.8, "neuroticism": 0.3},
            external_signals=[
                ExternalSignal(source="facial_au", arousal_hint=0.6, confidence=0.9),
            ],
            config=EmotionAPIConfig(
                advanced_mode=True,
                include_coupling=True,
                include_voice_params=True,
                detail_level="full",
            ),
        )
        assert req.session_id == "api-test-123"
        assert len(req.external_signals) == 1
        assert req.config.detail_level == "full"

    def test_stimulus_max_length(self) -> None:
        with pytest.raises(Exception):
            EmotionProcessRequest(stimulus="x" * 10001)

    def test_session_id_pattern(self) -> None:
        with pytest.raises(Exception):
            EmotionProcessRequest(stimulus="hi", session_id="invalid id with spaces")


class TestEmotionBatchRequest:
    def test_basic_batch(self) -> None:
        req = EmotionBatchRequest(
            stimuli=["Hello", "How are you?", "That's great!"],
        )
        assert len(req.stimuli) == 3

    def test_empty_batch_rejected(self) -> None:
        with pytest.raises(Exception):
            EmotionBatchRequest(stimuli=[])

    def test_max_batch_size(self) -> None:
        with pytest.raises(Exception):
            EmotionBatchRequest(stimuli=["x"] * 51)

    def test_batch_with_signals(self) -> None:
        req = EmotionBatchRequest(
            stimuli=["a", "b"],
            external_signals=[
                ExternalSignal(source="time_of_day", valence_hint=0.1, confidence=0.8),
            ],
        )
        assert len(req.external_signals) == 1


class TestEmotionConfigureRequest:
    def test_personality_only(self) -> None:
        req = EmotionConfigureRequest(
            personality={"neuroticism": 0.8},
        )
        assert req.personality == {"neuroticism": 0.8}
        assert req.values is None
        assert req.reset_state is False

    def test_values_only(self) -> None:
        req = EmotionConfigureRequest(
            values={"truth": 0.9, "compassion": 0.7},
        )
        assert req.values is not None

    def test_reset_state(self) -> None:
        req = EmotionConfigureRequest(reset_state=True)
        assert req.reset_state is True


class TestEmotionAPIConfig:
    def test_defaults(self) -> None:
        cfg = EmotionAPIConfig()
        assert cfg.advanced_mode is True
        assert cfg.include_coupling is True
        assert cfg.include_voice_params is False
        assert cfg.detail_level == "standard"

    def test_invalid_detail_level(self) -> None:
        with pytest.raises(Exception):
            EmotionAPIConfig(detail_level="invalid")

    def test_valid_detail_levels(self) -> None:
        for level in ["minimal", "standard", "full"]:
            cfg = EmotionAPIConfig(detail_level=level)
            assert cfg.detail_level == level


class TestResponseModels:
    def test_process_response_minimal(self) -> None:
        from pathos.models.emotion import neutral_state
        resp = EmotionProcessResponse(
            session_id="api-test",
            turn_number=1,
            emotional_state=neutral_state(),
        )
        assert resp.session_id == "api-test"
        assert resp.primary_emotion == ""

    def test_health_response(self) -> None:
        health = EmotionHealthResponse()
        assert health.status == "ok"
        assert health.version == "3.0.0"
        assert "facial_au" in health.external_signals_supported
        assert "weather" in health.external_signals_supported

    def test_presets_response(self) -> None:
        resp = EmotionPresetsResponse(
            presets=[
                PersonalityPreset(
                    name="neurotic",
                    description="High neuroticism, low conscientiousness",
                    traits={"neuroticism": 0.85, "conscientiousness": 0.3},
                ),
            ]
        )
        assert len(resp.presets) == 1
        assert resp.presets[0].name == "neurotic"

    def test_configure_response(self) -> None:
        resp = EmotionConfigureResponse(
            session_id="api-test",
            personality_applied={"openness": 0.8},
        )
        assert resp.session_id == "api-test"

    def test_state_response(self) -> None:
        from pathos.models.emotion import neutral_state
        resp = EmotionStateResponse(
            session_id="api-test",
            turn_number=5,
            emotional_state=neutral_state(),
            personality_summary={"neuroticism": 0.4},
            active_systems=["coupling", "contagion", "narrative"],
        )
        assert resp.turn_number == 5
        assert "coupling" in resp.active_systems

    def test_external_signal_contribution(self) -> None:
        c = ExternalSignalContribution(
            source="facial_au",
            arousal_delta=0.05,
            weight_applied=0.54,
        )
        assert c.source == "facial_au"
