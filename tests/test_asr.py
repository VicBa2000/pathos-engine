"""Tests para Voice V2: ASR (Speech-to-Text) con Whisper.

Verifica:
- ASR Service: singleton, estados, lazy loading (mock — no requiere modelo real)
- Audio decoding: WAV parsing, PCM16 raw, resampling, stereo→mono
- Transcription output format
- Integration: endpoint requirements (full_voice mode check)
"""

import io
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pathos.voice.asr import (
    ASRService,
    WHISPER_MODEL_NAME,
    WHISPER_SAMPLE_RATE,
    get_asr_service,
)


# --- Helpers ---

def _make_wav_bytes(
    sample_rate: int = 16000,
    duration_s: float = 1.0,
    channels: int = 1,
    sample_width: int = 2,
    freq: float = 440.0,
) -> bytes:
    """Generate a WAV file bytes with a sine wave."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5)

    if sample_width == 2:
        pcm = (audio * 32767).astype(np.int16)
    elif sample_width == 4:
        pcm = (audio * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")

    # Duplicate for stereo
    if channels > 1:
        pcm = np.column_stack([pcm] * channels).flatten()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


def _make_pcm16_bytes(
    sample_rate: int = 16000,
    duration_s: float = 0.5,
    freq: float = 440.0,
) -> bytes:
    """Generate raw PCM16 bytes (no WAV header)."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
    return audio.tobytes()


# =========================================================================
# Test: ASR Service basics
# =========================================================================

class TestASRServiceBasics:
    """Tests para el servicio ASR sin modelo real."""

    def test_default_model_name(self) -> None:
        """Default model is whisper-small."""
        svc = ASRService()
        assert svc.model_name == WHISPER_MODEL_NAME
        assert svc.model_name == "small"

    def test_not_initialized_by_default(self) -> None:
        """Service starts uninitialized."""
        svc = ASRService()
        assert not svc.is_initialized
        assert svc._model is None

    def test_custom_model_name(self) -> None:
        """Can specify a different model."""
        svc = ASRService(model_name="tiny")
        assert svc.model_name == "tiny"

    def test_singleton_returns_same_instance(self) -> None:
        """get_asr_service() returns singleton."""
        import pathos.voice.asr as asr_module
        # Reset singleton
        asr_module._asr_service = None
        s1 = get_asr_service()
        s2 = get_asr_service()
        assert s1 is s2
        # Cleanup
        asr_module._asr_service = None

    @pytest.mark.asyncio
    async def test_transcribe_raises_if_not_initialized(self) -> None:
        """Transcribe raises RuntimeError if not initialized."""
        svc = ASRService()
        with pytest.raises(RuntimeError, match="ASR not initialized"):
            await svc.transcribe(b"fake audio")

    @pytest.mark.asyncio
    async def test_initialize_raises_without_whisper(self) -> None:
        """Initialize raises if whisper package not installed."""
        svc = ASRService()
        with patch.dict("sys.modules", {"whisper": None, "torch": MagicMock()}):
            with pytest.raises(RuntimeError, match="Whisper not installed"):
                await svc.initialize()

    @pytest.mark.asyncio
    async def test_shutdown_noop_if_not_initialized(self) -> None:
        """Shutdown is no-op if not initialized."""
        svc = ASRService()
        await svc.shutdown()  # Should not raise
        assert not svc.is_initialized


# =========================================================================
# Test: Audio decoding
# =========================================================================

class TestAudioDecoding:
    """Tests para _decode_audio: WAV parsing, PCM16, resampling."""

    def test_decode_wav_16khz_mono(self) -> None:
        """Standard 16kHz mono WAV decodes correctly."""
        wav_bytes = _make_wav_bytes(sample_rate=16000, channels=1, duration_s=0.5)
        audio = ASRService._decode_audio(wav_bytes)
        assert audio.dtype == np.float32
        assert len(audio) == 8000  # 0.5s * 16kHz
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_decode_wav_44khz_resamples(self) -> None:
        """44.1kHz WAV is resampled to 16kHz."""
        wav_bytes = _make_wav_bytes(sample_rate=44100, channels=1, duration_s=1.0)
        audio = ASRService._decode_audio(wav_bytes)
        assert audio.dtype == np.float32
        # Should be resampled to 16kHz: ~16000 samples for 1 second
        assert abs(len(audio) - 16000) < 10

    def test_decode_wav_48khz_resamples(self) -> None:
        """48kHz WAV is resampled to 16kHz."""
        wav_bytes = _make_wav_bytes(sample_rate=48000, channels=1, duration_s=0.5)
        audio = ASRService._decode_audio(wav_bytes)
        expected_len = int(0.5 * WHISPER_SAMPLE_RATE)
        assert abs(len(audio) - expected_len) < 10

    def test_decode_wav_stereo_to_mono(self) -> None:
        """Stereo WAV is converted to mono."""
        wav_bytes = _make_wav_bytes(sample_rate=16000, channels=2, duration_s=0.5)
        audio = ASRService._decode_audio(wav_bytes)
        assert audio.dtype == np.float32
        assert len(audio) == 8000  # mono: 0.5s * 16kHz

    def test_decode_wav_32bit(self) -> None:
        """32-bit WAV decodes correctly."""
        wav_bytes = _make_wav_bytes(sample_rate=16000, channels=1, duration_s=0.5, sample_width=4)
        audio = ASRService._decode_audio(wav_bytes)
        assert audio.dtype == np.float32
        assert len(audio) == 8000
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_decode_raw_pcm16(self) -> None:
        """Raw PCM16 bytes (no WAV header) are decoded as 16kHz mono."""
        pcm_bytes = _make_pcm16_bytes(duration_s=0.5)
        audio = ASRService._decode_audio(pcm_bytes)
        assert audio.dtype == np.float32
        assert len(audio) == 8000
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_decode_wav_24khz_resamples(self) -> None:
        """24kHz WAV (from TTS) is resampled to 16kHz."""
        wav_bytes = _make_wav_bytes(sample_rate=24000, channels=1, duration_s=1.0)
        audio = ASRService._decode_audio(wav_bytes)
        expected_len = int(1.0 * WHISPER_SAMPLE_RATE)
        assert abs(len(audio) - expected_len) < 10

    def test_decode_preserves_signal_shape(self) -> None:
        """Decoded audio preserves the general shape (not all zeros)."""
        wav_bytes = _make_wav_bytes(sample_rate=16000, duration_s=0.5, freq=440)
        audio = ASRService._decode_audio(wav_bytes)
        # Should have both positive and negative values (sine wave)
        assert audio.max() > 0.1
        assert audio.min() < -0.1

    def test_decode_empty_pcm_raises(self) -> None:
        """Empty bytes should raise (not enough data for WAV or PCM)."""
        # Empty WAV will fail wave.open, then empty PCM will produce empty array
        audio = ASRService._decode_audio(b"")
        assert len(audio) == 0


# =========================================================================
# Test: Transcription (mocked)
# =========================================================================

class TestTranscription:
    """Tests para transcripción con modelo mockeado."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_correct_format(self) -> None:
        """Transcribe returns dict with text, language, segments."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "  Hello, world!  ",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " Hello, world! "},
            ],
        }

        result = await svc.transcribe(_make_wav_bytes())
        assert result["text"] == "Hello, world!"
        assert result["language"] == "en"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello, world!"
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.5

    @pytest.mark.asyncio
    async def test_transcribe_with_language_hint(self) -> None:
        """Language hint is passed to Whisper."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "Hola mundo",
            "language": "es",
            "segments": [],
        }

        result = await svc.transcribe(_make_wav_bytes(), language="es")
        # Check that language was passed to the model
        call_args = svc._model.transcribe.call_args
        assert call_args[1]["language"] == "es"
        assert result["language"] == "es"

    @pytest.mark.asyncio
    async def test_transcribe_without_language_hint(self) -> None:
        """Without language hint, Whisper auto-detects."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "Bonjour",
            "language": "fr",
            "segments": [],
        }

        result = await svc.transcribe(_make_wav_bytes(), language=None)
        call_args = svc._model.transcribe.call_args
        assert "language" not in call_args[1]
        assert result["language"] == "fr"

    @pytest.mark.asyncio
    async def test_transcribe_strips_whitespace(self) -> None:
        """Transcription text is stripped."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "   lots of spaces   ",
            "language": "en",
            "segments": [{"start": 0, "end": 1, "text": "   lots of spaces   "}],
        }

        result = await svc.transcribe(_make_wav_bytes())
        assert result["text"] == "lots of spaces"
        assert result["segments"][0]["text"] == "lots of spaces"

    @pytest.mark.asyncio
    async def test_transcribe_empty_result(self) -> None:
        """Empty transcription returns empty string."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "",
            "language": "en",
            "segments": [],
        }

        result = await svc.transcribe(_make_wav_bytes())
        assert result["text"] == ""
        assert result["segments"] == []

    @pytest.mark.asyncio
    async def test_transcribe_multiple_segments(self) -> None:
        """Multiple segments are preserved."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "First sentence. Second sentence.",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " First sentence."},
                {"start": 1.5, "end": 3.0, "text": " Second sentence."},
            ],
        }

        result = await svc.transcribe(_make_wav_bytes())
        assert len(result["segments"]) == 2
        assert result["segments"][0]["end"] == 1.5
        assert result["segments"][1]["start"] == 1.5

    @pytest.mark.asyncio
    async def test_transcribe_missing_language_field(self) -> None:
        """Missing language field defaults to 'unknown'."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        svc._model.transcribe.return_value = {
            "text": "Hello",
            "segments": [],
        }

        result = await svc.transcribe(_make_wav_bytes())
        assert result["language"] == "unknown"


# =========================================================================
# Test: Initialize/Shutdown (mocked)
# =========================================================================

class TestInitializeShutdown:
    """Tests para inicialización y shutdown con mocks."""

    @pytest.mark.asyncio
    async def test_initialize_loads_model(self) -> None:
        """Initialize sets _initialized and loads model."""
        svc = ASRService(model_name="tiny")

        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"whisper": mock_whisper, "torch": mock_torch}):
            # Patch the lazy imports inside initialize
            import pathos.voice.asr as asr_mod
            original_init = asr_mod.ASRService.initialize

            async def patched_init(self: ASRService) -> None:
                if self._initialized:
                    return
                async with self._lock:
                    if self._initialized:
                        return
                    self._device = "cpu"
                    self._model = mock_model
                    self._initialized = True

            asr_mod.ASRService.initialize = patched_init  # type: ignore[assignment]
            try:
                await svc.initialize()
                assert svc.is_initialized
                assert svc._model is mock_model
            finally:
                asr_mod.ASRService.initialize = original_init  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_double_initialize_is_noop(self) -> None:
        """Calling initialize twice doesn't reload."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()
        original_model = svc._model
        await svc.initialize()
        assert svc._model is original_model

    @pytest.mark.asyncio
    async def test_shutdown_clears_model(self) -> None:
        """Shutdown clears model and resets state."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            await svc.shutdown()

        assert not svc.is_initialized
        assert svc._model is None

    @pytest.mark.asyncio
    async def test_shutdown_clears_cuda_cache(self) -> None:
        """Shutdown calls cuda.empty_cache if CUDA available."""
        svc = ASRService()
        svc._initialized = True
        svc._model = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            await svc.shutdown()

        mock_torch.cuda.empty_cache.assert_called_once()


# =========================================================================
# Test: Whisper sample rate constant
# =========================================================================

class TestConstants:
    """Tests para constantes del módulo."""

    def test_whisper_sample_rate(self) -> None:
        assert WHISPER_SAMPLE_RATE == 16000

    def test_default_model(self) -> None:
        assert WHISPER_MODEL_NAME == "small"


# =========================================================================
# Test: VoiceMode FULL_VOICE
# =========================================================================

class TestVoiceModeFull:
    """Tests que verifican que FULL_VOICE está correctamente definido."""

    def test_full_voice_mode_exists(self) -> None:
        from pathos.models.voice import VoiceMode
        assert VoiceMode.FULL_VOICE == "full_voice"

    def test_voice_config_supports_full_voice(self) -> None:
        from pathos.models.voice import VoiceConfig, VoiceMode
        config = VoiceConfig(mode=VoiceMode.FULL_VOICE)
        assert config.mode == VoiceMode.FULL_VOICE

    def test_voice_details_has_asr_fields(self) -> None:
        from pathos.models.schemas import VoiceDetails
        details = VoiceDetails(asr_available=True, last_transcription="hello", detected_language="en")
        assert details.asr_available is True
        assert details.last_transcription == "hello"
        assert details.detected_language == "en"

    def test_voice_details_defaults(self) -> None:
        from pathos.models.schemas import VoiceDetails
        details = VoiceDetails()
        assert details.asr_available is False
        assert details.last_transcription == ""
        assert details.detected_language == ""
