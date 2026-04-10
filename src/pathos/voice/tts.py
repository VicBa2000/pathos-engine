"""TTS Router - Switch inteligente entre Kokoro y Parler-TTS.

Kokoro (default): rapido, ligero (~350MB VRAM), multi-idioma.
Parler-TTS (opcional): emociones complejas via descripcion textual, solo EN, ~2.5GB VRAM.

Logica de seleccion:
- Kokoro: emociones basicas, cualquier idioma, intensidad < 0.6
- Parler: emociones complejas/mixtas/emergentes, ingles, intensidad >= 0.6
- Fallback: si Parler falla o no esta cargado, usa Kokoro

Audio output: 24kHz, PCM16, mono WAV (ambos backends normalizados)
"""

import logging

from pathos.models.voice import TTSBackend, VoiceParams
from pathos.voice.tts_kokoro import KokoroTTSService, get_kokoro_service
from pathos.voice.tts_parler import ParlerTTSService, get_parler_service

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2


class TTSService:
    """Router TTS que delega a Kokoro o Parler-TTS segun parametros."""

    def __init__(self) -> None:
        self._kokoro: KokoroTTSService = get_kokoro_service()
        self._parler: ParlerTTSService = get_parler_service()
        self._parler_init_failed: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._kokoro.is_initialized

    async def initialize(self) -> None:
        """Inicializa Kokoro (backend principal). Parler se carga on-demand."""
        await self._kokoro.initialize(lang_code="a")  # English por defecto

    async def shutdown(self) -> None:
        """Libera ambos backends."""
        await self._kokoro.shutdown()
        if self._parler.is_initialized:
            await self._parler.shutdown()

    @property
    def parler_available(self) -> bool:
        """True si Parler-TTS esta disponible (inicializado o no ha fallado)."""
        return self._parler.is_initialized or not self._parler_init_failed

    @property
    def parler_initialized(self) -> bool:
        """True si Parler-TTS esta cargado en memoria."""
        return self._parler.is_initialized

    async def generate_speech(self, text: str, params: VoiceParams) -> bytes:
        """Genera audio delegando al backend apropiado.

        Si params.backend == PARLER y Parler esta disponible, usa Parler.
        En cualquier otro caso, usa Kokoro.
        Si Parler falla, fallback a Kokoro.
        """
        if params.backend == TTSBackend.PARLER and params.parler_description:
            if self._parler.is_initialized:
                try:
                    logger.info("Generating with Parler-TTS (desc: %s...)", params.parler_description[:80])
                    return await self._parler.generate_speech(text, params)
                except Exception as e:
                    logger.warning("Parler-TTS generation failed (%s), falling back to Kokoro", e)
            elif not self._parler_init_failed:
                try:
                    logger.info("Initializing Parler-TTS on demand...")
                    await self._parler.initialize()
                    logger.info("Parler-TTS initialized, generating speech...")
                    return await self._parler.generate_speech(text, params)
                except Exception as e:
                    logger.warning("Parler-TTS not available (%s), using Kokoro for all voices", e)
                    self._parler_init_failed = True
            else:
                logger.info("Parler-TTS previously failed to init, using Kokoro fallback")

        # Asegurar pipeline Kokoro para el idioma de la voz
        # Voice key puede ser blend "af_heart,af_bella" — usar primera voz
        first_voice = params.voice_key.split(",")[0] if params.voice_key else "af_heart"
        voice_lang_char = first_voice[0] if first_voice else "a"
        if voice_lang_char not in self._kokoro._pipelines:
            await self._kokoro.initialize(voice_lang_char)

        return await self._kokoro.generate_speech(text, params)

    def list_available_voices(self) -> list[str]:
        """Lista voces de Kokoro."""
        return self._kokoro.list_available_voices()


# Singleton global (lazy)
_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    """Obtiene la instancia global del TTS router."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
