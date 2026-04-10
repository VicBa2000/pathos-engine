"""Kokoro TTS Backend - Lightweight emotional TTS.

Kokoro es un modelo TTS de 82M parametros (~350MB VRAM).
Soporta multiples idiomas: EN, ES, FR, IT, JA, KO, PT, HI, ZH.
Control emocional via: seleccion de voz + speed + stage directions.

Audio: 24kHz, float32 -> PCM16 mono WAV
"""

import asyncio
import io
import logging
import wave
from typing import Any

from pathos.models.voice import KOKORO_LANG_CODES, VoiceParams

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


class KokoroTTSService:
    """Servicio TTS con Kokoro.

    Lazy loading: el modelo solo se carga al llamar initialize().
    Thread-safe: usa lock para serializar generacion.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}  # lang_code -> KPipeline
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, lang_code: str = "a") -> None:
        """Carga el pipeline de Kokoro para un idioma.

        Args:
            lang_code: Codigo Kokoro ('a'=US EN, 'e'=ES, etc.)
        """
        if lang_code in self._pipelines:
            return

        async with self._lock:
            if lang_code in self._pipelines:
                return

            logger.info("Loading Kokoro pipeline for lang_code='%s'", lang_code)
            pipe = await asyncio.get_event_loop().run_in_executor(
                None, self._load_pipeline, lang_code,
            )
            self._pipelines[lang_code] = pipe
            self._initialized = True
            logger.info("Kokoro pipeline ready for lang_code='%s'", lang_code)

    @staticmethod
    def _load_pipeline(lang_code: str) -> Any:
        """Carga sincrona del pipeline (thread pool)."""
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise RuntimeError(
                "Kokoro not installed. Run: pip install kokoro soundfile"
            ) from e

        return KPipeline(lang_code=lang_code)

    def _get_pipeline(self, voice_key: str) -> Any:
        """Obtiene el pipeline correcto basado en el voice_key.

        El primer caracter del voice_key indica el idioma en Kokoro.
        """
        lang_char = voice_key[0] if voice_key else "a"

        if lang_char in self._pipelines:
            return self._pipelines[lang_char]

        # Fallback a cualquier pipeline cargado
        if self._pipelines:
            fallback = next(iter(self._pipelines.values()))
            logger.warning(
                "No pipeline for lang '%s', using fallback", lang_char,
            )
            return fallback

        raise RuntimeError("No Kokoro pipeline loaded. Call initialize() first.")

    async def ensure_pipeline_for_lang(self, language: str) -> None:
        """Asegura que el pipeline para un idioma este cargado."""
        kokoro_code = KOKORO_LANG_CODES.get(language, "a")
        if kokoro_code not in self._pipelines:
            await self.initialize(kokoro_code)

    async def generate_speech(self, text: str, params: VoiceParams) -> bytes:
        """Genera audio WAV completo.

        Args:
            text: Texto a sintetizar (puede incluir stage direction prepended).
            params: Parametros de voz (voice_key, speed).

        Returns:
            bytes: WAV file completo (24kHz, PCM16, mono)
        """
        if not self._initialized:
            raise RuntimeError("Kokoro TTS not initialized. Call initialize() first.")

        if len(text) > 5000:
            logger.warning("TTS input truncated from %d to 5000 chars", len(text))
            text = text[:5000]

        async with self._lock:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._generate_sync, text, params,
                ),
                timeout=60.0,
            )

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Limpia texto para Kokoro. Strip markdown, Unicode, emojis."""
        import re

        # Remover stage direction brackets
        text = re.sub(r'\[speaking[^\]]*\]\s*', '', text)

        # --- Markdown cleanup ---
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        # Italic: *text* or _text_
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)
        # Strikethrough: ~~text~~
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        # Code blocks and inline code
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        # Headers: # ## ###
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Bullet points: - * > at start of line
        text = re.sub(r'^[\-\*>]\s+', '', text, flags=re.MULTILINE)
        # Numbered lists: 1. 2. etc
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # Links: [text](url) -> text
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
        # HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # --- Unicode normalization ---
        # Smart quotes -> ASCII
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        # Dashes -> spoken form
        text = text.replace('\u2014', ', ').replace('\u2013', '-')
        # Ellipsis
        text = text.replace('\u2026', '...')
        # Bullet char
        text = text.replace('\u2022', ',')

        # Remover todo fuera de ASCII imprimible
        text = re.sub(r'[^\x20-\x7E]', ' ', text)

        # Limpiar residuos de markdown
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', ' ', text)

        # Colapsar espacios y newlines
        text = re.sub(r'\s+', ' ', text).strip()

        return text if text else "..."

    @staticmethod
    def _post_process_emotion(audio: Any, params: VoiceParams) -> Any:
        """Post-procesa el audio para aplicar emocion audible.

        Tecnicas aplicadas (en orden):
        1. Pitch shift via resample (cambia tono base)
        2. Pitch contour: variacion sutil de pitch a lo largo del audio
           para evitar monotonia (ondulacion natural del habla)
        3. Dynamic range: expansion para emociones fuertes, compresion para suaves
        4. Volume: mas fuerte para ira, mas suave para tristeza
        5. Tremolo: vibracion nerviosa para miedo/ansiedad
        """
        import numpy as np
        from scipy.signal import resample

        # 1. Pitch shift via resample (cambia pitch + duración)
        # La duración se compensa porque _effective_speed ya ajustó el speed
        if abs(params.pitch_semitones) > 0.1:
            factor = 2 ** (params.pitch_semitones / 12.0)
            n_out = int(len(audio) / factor)
            audio = resample(audio, n_out).astype(np.float32)

        # 2. Pitch contour — variacion MUY sutil de pitch a lo largo de la frase
        # Simula entonacion natural (sube al inicio, baja al final).
        # Mantenido muy suave para evitar artefactos con el resample.
        pitch_contour_amount = abs(params.pitch_semitones) * 0.06
        if pitch_contour_amount > 0.01:
            n = len(audio)
            contour = np.ones(n, dtype=np.float32)
            third = n // 3
            if third > 0:
                rise = np.linspace(1.0 - pitch_contour_amount, 1.0, third)
                contour[:third] = rise
                fall = np.linspace(1.0, 1.0 - pitch_contour_amount * 0.5, third)
                contour[-third:] = fall
            # Aplicar como modulacion de amplitud suave (no micro-resample)
            # Esto da una ligera ilusion de entonacion sin artefactos
            audio = audio * contour

        # 3. Dynamic range — expansion/compresion MUY suave
        # Solo un toque para dar mas presencia a emociones fuertes
        if params.volume > 1.05:
            threshold = 0.3
            mask = np.abs(audio) > threshold
            audio[mask] *= 1.0 + (params.volume - 1.0) * 0.15
        elif params.volume < 0.85:
            threshold = 0.5
            mask = np.abs(audio) > threshold
            audio[mask] *= 0.95

        # 4. Volume
        if abs(params.volume - 1.0) > 0.02:
            audio = audio * params.volume

        # 5. Tremolo (vibracion de volumen para fear/anxiety)
        if params.tremolo > 0.01:
            t = np.arange(len(audio)) / SAMPLE_RATE
            # Frecuencia variable: empieza lento, se acelera (ansiedad creciente)
            freq = 5.5 + 1.5 * np.sin(2 * np.pi * 0.3 * t)
            phase = np.cumsum(freq) / SAMPLE_RATE * 2 * np.pi
            tremolo_wave = 1.0 + params.tremolo * np.sin(phase)
            audio = (audio * tremolo_wave).astype(np.float32)

        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _effective_speed(params: VoiceParams) -> float:
        """Calcula el speed efectivo compensando el pitch shift.

        Pitch shift via resample cambia la duración por factor 1/f.
        Para mantener la duración deseada (según speed emocional),
        generamos a un speed ajustado que se cancela con el resample.

        Ejemplo: pitch -3st (factor=0.84) acorta audio a 84%.
                 Si queremos speed=0.7 (lento), generamos a speed=0.7*0.84=0.59
                 El resample expande a 1/0.84=119%, resultado: ~0.7x original.
        """
        if abs(params.pitch_semitones) < 0.1:
            return params.speed
        factor = 2 ** (params.pitch_semitones / 12.0)
        # El resample divide por factor, asi que pre-multiplicamos
        return params.speed * factor

    def _generate_sync(self, text: str, params: VoiceParams) -> bytes:
        """Generacion sincrona (thread pool).

        Estrategia robusta: genera oración por oración.
        Si una oración falla, la skipea y continúa con las demás.
        Post-procesa con pitch shift, volume y tremolo emocional.
        """
        import re

        import numpy as np

        pipeline = self._get_pipeline(params.voice_key)
        clean_text = self._sanitize_text(text)

        # Dividir en oraciones para generar de forma robusta
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        if not sentences:
            sentences = [clean_text]

        audio_chunks: list[Any] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                gen_speed = self._effective_speed(params)
                for _gs, _ps, audio in pipeline(
                    sentence,
                    voice=params.voice_key,
                    speed=gen_speed,
                ):
                    audio_chunks.append(audio)
            except TypeError:
                logger.warning("G2P skipping sentence: %s", sentence[:60])
                continue

        if not audio_chunks:
            raise RuntimeError("Kokoro: all sentences failed G2P")

        combined = np.concatenate([
            chunk.numpy() if hasattr(chunk, "numpy") else chunk
            for chunk in audio_chunks
        ])

        # Aplicar emoción al audio (pitch, volume, tremolo)
        combined = self._post_process_emotion(combined, params)

        return self._float32_to_wav(combined)

    @staticmethod
    def _float32_to_wav(audio: Any) -> bytes:
        """Convierte float32 array a WAV bytes (24kHz, PCM16, mono)."""
        import numpy as np

        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        return buf.getvalue()

    async def shutdown(self) -> None:
        """Libera pipelines."""
        async with self._lock:
            self._pipelines.clear()
            self._initialized = False
            logger.info("Kokoro TTS pipelines released")

    def list_available_voices(self) -> list[str]:
        """Lista voces disponibles de los pipelines cargados."""
        voices: list[str] = []
        for pipe in self._pipelines.values():
            if hasattr(pipe, "voices"):
                voices.extend(pipe.voices)
        return voices


# Singleton global (lazy)
_kokoro_service: KokoroTTSService | None = None


def get_kokoro_service() -> KokoroTTSService:
    """Obtiene la instancia global del servicio Kokoro."""
    global _kokoro_service
    if _kokoro_service is None:
        _kokoro_service = KokoroTTSService()
    return _kokoro_service
