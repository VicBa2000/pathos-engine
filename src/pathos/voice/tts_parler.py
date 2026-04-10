"""Parler-TTS Backend - Emotion-rich TTS via text descriptions.

Parler-TTS mini (~880M params, ~2.5GB VRAM) genera voz controlada
por descripciones textuales libres. Solo soporta ingles.

Uso: para emociones complejas, mixtas o emergentes donde Kokoro
no puede expresar el matiz emocional deseado.

Audio: 44.1kHz, float32 -> PCM16 mono WAV (resampled a 24kHz para compatibilidad)
"""

import asyncio
import io
import logging
import wave
from typing import Any

from pathos.models.voice import VoiceParams

logger = logging.getLogger(__name__)

# Output resampled a 24kHz para consistencia con Kokoro
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

MODEL_ID = "parler-tts/parler-tts-mini-v1"


class ParlerTTSService:
    """Servicio TTS con Parler-TTS.

    Lazy loading: solo se carga si se solicita explicitamente.
    Mas pesado que Kokoro (~2.5GB VRAM), usar solo para emociones complejas en ingles.
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._native_sample_rate: int = 44100
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Carga el modelo Parler-TTS en GPU."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info("Loading Parler-TTS model: %s", self.model_id)
            await asyncio.get_event_loop().run_in_executor(None, self._load_model)
            self._initialized = True
            logger.info(
                "Parler-TTS loaded on %s (native sr=%d)",
                self._device, self._native_sample_rate,
            )

    def _load_model(self) -> None:
        """Carga sincrona (thread pool)."""
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Parler-TTS not installed. Run: pip install parler-tts"
            ) from e

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id,
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._native_sample_rate = self._model.config.sampling_rate

    async def generate_speech(self, text: str, params: VoiceParams) -> bytes:
        """Genera audio WAV con descripcion emocional.

        Args:
            text: Texto a sintetizar.
            params: Debe tener parler_description con la descripcion emocional.

        Returns:
            bytes: WAV file (24kHz, PCM16, mono) — resampled desde 44.1kHz
        """
        if not self._initialized:
            raise RuntimeError("Parler-TTS not initialized. Call initialize() first.")

        if len(text) > 5000:
            logger.warning("TTS input truncated from %d to 5000 chars", len(text))
            text = text[:5000]

        description = params.parler_description
        if not description:
            description = "A person speaks clearly in a natural voice."

        async with self._lock:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._generate_sync, text, description,
                ),
                timeout=60.0,
            )

    @staticmethod
    def _clean_text_for_parler(text: str) -> str:
        """Limpia texto de anotaciones de Kokoro que Parler no entiende.

        Remueve: [word](+N) stress annotations, excessive ellipsis.
        """
        import re
        # Remove Kokoro stress annotations: [word](+N) -> word
        text = re.sub(r'\[([^\]]+)\]\(\+\d+\)', r'\1', text)
        # Remove stage directions if any leaked through
        text = re.sub(r'\[speaking[^\]]*\]\s*', '', text)
        # Collapse excessive ellipsis (emotional pauses) to single period
        text = re.sub(r'\.{3,}', '.', text)
        # Clean double periods
        text = re.sub(r'\.{2,}', '.', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else "Hello."

    # Parler mini: max_length=2580 audio codes ÷ 86 fps = ~30s de audio.
    # Solo chunkeamos si el texto excede ~120 tokens T5 (~30s de habla).
    _MAX_TOKENS_SINGLE_PASS = 120

    def _generate_sync(self, text: str, description: str) -> bytes:
        """Generacion sincrona (thread pool).

        Genera todo el texto en UNA sola llamada para evitar variacion de voz.
        Solo divide en chunks si el texto es extremadamente largo (>30s).
        """
        import re

        import numpy as np
        import torch

        clean_text = self._clean_text_for_parler(text)

        logger.info("Parler generating: desc='%s...', text='%s...'",
                     description[:80], clean_text[:80])

        # Seed fijo por turno para reproducibilidad
        seed = hash(clean_text[:20]) % (2**31)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Pre-tokenizar descripcion
        desc_encoding = self._tokenizer(description, return_tensors="pt")
        desc_input_ids = desc_encoding.input_ids.to(self._device)
        desc_attention_mask = desc_encoding.attention_mask.to(self._device)

        # Verificar si cabe en una sola generacion
        text_tokens = len(self._tokenizer.encode(clean_text))

        if text_tokens <= self._MAX_TOKENS_SINGLE_PASS:
            # Una sola generacion — sin variacion de voz
            logger.info("Parler: single pass (%d tokens)", text_tokens)
            audio = self._generate_chunk(clean_text, desc_input_ids, desc_attention_mask)
        else:
            # Texto muy largo: dividir en chunks grandes con crossfade
            logger.info("Parler: text too long (%d tokens), chunking", text_tokens)
            sentences = re.split(r'(?<=[.!?])\s+', clean_text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 2]
            if not sentences:
                sentences = [clean_text]

            chunks = self._group_sentences(sentences)
            logger.info("Parler: %d sentences -> %d chunks", len(sentences), len(chunks))

            audio_chunks: list[Any] = []
            for chunk_text in chunks:
                try:
                    chunk_audio = self._generate_chunk(
                        chunk_text, desc_input_ids, desc_attention_mask,
                    )
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.warning("Parler skipping chunk (%s): %s", e, chunk_text[:60])
                    continue

            if not audio_chunks:
                raise RuntimeError("Parler-TTS: all chunks failed")

            audio = self._crossfade_chunks(audio_chunks, self._native_sample_rate)

        # Resample de 44.1kHz a 24kHz para compatibilidad
        if self._native_sample_rate != OUTPUT_SAMPLE_RATE:
            audio = self._resample(audio, self._native_sample_rate, OUTPUT_SAMPLE_RATE)

        return self._float32_to_wav(audio)

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Agrupa oraciones en chunks grandes (~100 tokens, ~25s audio)."""
        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            token_count = len(self._tokenizer.encode(candidate))

            if token_count > self._MAX_TOKENS_SINGLE_PASS and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = candidate

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _generate_chunk(
        self,
        text: str,
        desc_input_ids: Any,
        desc_attention_mask: Any,
    ) -> Any:
        """Genera audio para un chunk de texto.

        Parametros de generacion optimizados para expresividad emocional:
        - max_new_tokens: suficiente para no cortar frases (~30s)
        - min_new_tokens: evita generaciones demasiado cortas
        - temperature 1.0: variacion natural en la voz
        """
        import numpy as np
        import torch

        prompt_encoding = self._tokenizer(text, return_tensors="pt")
        prompt_input_ids = prompt_encoding.input_ids.to(self._device)
        prompt_attention_mask = prompt_encoding.attention_mask.to(self._device)

        # Estimar tokens de audio necesarios: ~86 audio frames por segundo,
        # ~15 caracteres por segundo de habla → chars * 86/15 ≈ chars * 6
        estimated_audio_tokens = min(len(text) * 6, 2580)
        min_tokens = max(estimated_audio_tokens // 3, 10)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=desc_input_ids,
                attention_mask=desc_attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                # Evitar cortes: generar suficientes tokens de audio
                max_new_tokens=estimated_audio_tokens,
                min_new_tokens=min_tokens,
                # Sampling para naturalidad
                do_sample=True,
                temperature=1.0,
            )

        return generation.cpu().numpy().squeeze().astype(np.float32)

    @staticmethod
    def _crossfade_chunks(chunks: list[Any], sample_rate: int) -> Any:
        """Concatena chunks con crossfade de 80ms para suavizar transiciones."""
        import numpy as np

        if len(chunks) == 1:
            return chunks[0]

        fade_samples = int(sample_rate * 0.08)  # 80ms crossfade

        result = chunks[0]
        for chunk in chunks[1:]:
            if len(result) < fade_samples or len(chunk) < fade_samples:
                result = np.concatenate([result, chunk])
                continue

            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            overlap = result[-fade_samples:] * fade_out + chunk[:fade_samples] * fade_in

            result = np.concatenate([
                result[:-fade_samples],
                overlap,
                chunk[fade_samples:],
            ])

        return result

    @staticmethod
    def _resample(audio: Any, src_rate: int, dst_rate: int) -> Any:
        """Resample lineal simple."""
        import numpy as np

        duration = len(audio) / src_rate
        target_len = int(duration * dst_rate)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

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
            wf.setframerate(OUTPUT_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        return buf.getvalue()

    async def shutdown(self) -> None:
        """Libera modelo y VRAM."""
        if not self._initialized:
            return

        async with self._lock:
            import torch

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._initialized = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Parler-TTS model unloaded")


# Singleton global (lazy)
_parler_service: ParlerTTSService | None = None


def get_parler_service() -> ParlerTTSService:
    """Obtiene la instancia global del servicio Parler-TTS."""
    global _parler_service
    if _parler_service is None:
        _parler_service = ParlerTTSService()
    return _parler_service
