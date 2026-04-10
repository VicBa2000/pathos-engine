"""ASR Service - Whisper wrapper for speech-to-text.

Carga y gestiona el modelo Whisper-small de OpenAI para transcripción de voz.
El texto transcrito se alimenta al pipeline emocional normal.

LAZY LOADING: el modelo solo se carga cuando se activa full_voice mode.
Si voice mode está en text_only o voice_out, no consume VRAM adicional.

Hardware target: GTX 1660 Super 6GB
- qwen3:4b (Ollama): ~2.5GB VRAM
- Kokoro TTS: ~1GB VRAM
- Whisper-small: ~0.5GB VRAM
- Total: ~4GB — cabe en 6GB

Audio input: WAV o raw PCM16, 16kHz mono (Whisper standard).
"""

import asyncio
import io
import logging
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Whisper constants
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz
WHISPER_MODEL_NAME = "small"  # ~0.5GB VRAM


class ASRService:
    """Servicio de ASR con OpenAI Whisper.

    Lazy loading: el modelo solo se carga al llamar initialize().
    Thread-safe: usa locks para evitar transcripción concurrente en GPU.
    """

    def __init__(self, model_name: str = WHISPER_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._device: str = "cpu"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Carga el modelo Whisper en GPU.

        Solo se llama cuando el usuario activa full_voice mode.
        Bloquea hasta que el modelo esté listo.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info("Loading Whisper model: %s", self.model_name)

            try:
                import torch
                import whisper
            except ImportError as e:
                raise RuntimeError(
                    "Whisper not installed. Run: pip install openai-whisper"
                ) from e

            # Detect device
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            # Load model
            self._model = whisper.load_model(
                self.model_name, device=self._device,
            )

            self._initialized = True
            logger.info(
                "Whisper model '%s' loaded on %s", self.model_name, self._device,
            )

    async def shutdown(self) -> None:
        """Libera el modelo y la VRAM."""
        if not self._initialized:
            return

        async with self._lock:
            import torch

            del self._model
            self._model = None
            self._initialized = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: WAV file bytes or raw PCM16 audio at 16kHz mono.
            language: Optional language hint (e.g., "en", "es"). If None, Whisper auto-detects.

        Returns:
            Dict with keys:
            - text: transcribed text (str)
            - language: detected language code (str)
            - segments: list of segments with timestamps (list[dict])
        """
        if not self._initialized:
            raise RuntimeError("ASR not initialized. Call initialize() first.")

        async with self._lock:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._transcribe_sync, audio_bytes, language,
                ),
                timeout=60.0,
            )

    def _transcribe_sync(
        self, audio_bytes: bytes, language: str | None,
    ) -> dict[str, Any]:
        """Transcripción síncrona (se ejecuta en thread pool)."""
        # Convert bytes to numpy float32 array
        audio_np = self._decode_audio(audio_bytes)

        # Build transcribe options
        options: dict[str, Any] = {}
        if language:
            options["language"] = language

        # Transcribe
        result = self._model.transcribe(audio_np, **options)

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> Any:
        """Decode audio bytes to float32 numpy array at 16kHz.

        Supports:
        - WAV files (any sample rate — resampled to 16kHz if needed)
        - WebM, MP3, OGG, and other formats via ffmpeg (whisper.load_audio)
        - Raw PCM16 audio (assumed 16kHz mono) as last resort
        """
        # Try WAV first (fast, no ffmpeg needed)
        try:
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

            # Convert to float32
            if sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Mono conversion
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            # Resample to 16kHz if needed
            if framerate != WHISPER_SAMPLE_RATE:
                duration = len(audio) / framerate
                target_len = int(duration * WHISPER_SAMPLE_RATE)
                indices = np.linspace(0, len(audio) - 1, target_len)
                audio = np.interp(indices, np.arange(len(audio)), audio)

            return audio.astype(np.float32)

        except (wave.Error, EOFError):
            pass

        # Try ffmpeg via whisper.load_audio (handles WebM, MP3, OGG, etc.)
        try:
            import os
            import tempfile

            import whisper

            # Detect extension from magic bytes
            ext = ".webm"
            if audio_bytes[:4] == b"OggS":
                ext = ".ogg"
            elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
                ext = ".mp3"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            try:
                audio = whisper.load_audio(tmp_path)
                return audio
            finally:
                os.unlink(tmp_path)

        except Exception:
            logger.warning("Audio decode via ffmpeg/whisper failed, falling back to raw PCM16", exc_info=True)

        # Last resort: assume raw PCM16 at 16kHz mono
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio


# Singleton global (lazy)
_asr_service: ASRService | None = None


def get_asr_service() -> ASRService:
    """Obtiene la instancia global del ASR service."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
