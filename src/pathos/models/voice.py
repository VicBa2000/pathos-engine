"""Voice Models - Configuracion y parametros de voz.

Modelos para el sistema de voz de Pathos:
- VoiceMode: text-only | voice-out | full-voice
- TTSBackend: kokoro | parler
- VoiceConfig: configuracion de la sesion de voz
- VoiceParams: parametros generados por turno (emotional state -> voice params)
- VoicePreset: catalogo de voces disponibles
"""

from enum import Enum

from pydantic import BaseModel, Field


class VoiceMode(str, Enum):
    """Modos de voz disponibles."""

    TEXT_ONLY = "text_only"       # Sin voz (default)
    VOICE_OUT = "voice_out"      # El agente habla (V1: TTS)
    FULL_VOICE = "full_voice"    # Agente habla + escucha (V1+V2: TTS+ASR)


class TTSBackend(str, Enum):
    """Backend de TTS disponible."""

    KOKORO = "kokoro"    # Rapido, ligero, multi-idioma
    PARLER = "parler"    # Control emocional por descripcion textual (solo EN)


class VoicePreset(BaseModel):
    """Un preset de voz disponible."""

    key: str = Field(description="Clave del preset (ej: 'af_heart')")
    language: str = Field(description="Codigo de idioma (ej: 'en')")
    name: str = Field(description="Nombre del hablante (ej: 'Heart')")
    gender: str = Field(description="'man' o 'woman'")
    backend: TTSBackend = Field(default=TTSBackend.KOKORO, description="Backend TTS")
    requires_espeak: bool = Field(default=False, description="Necesita espeak-ng instalado")


# Catalogo de voces Kokoro
# Naming: 1st letter = lang (a=US EN, b=UK EN, e=ES, j=JA, z=ZH, f=FR, h=HI, i=IT, p=PT-BR, k=KO)
#          2nd letter = gender (f=female, m=male)
AVAILABLE_VOICES: list[VoicePreset] = [
    # American English
    VoicePreset(key="af_heart", language="en", name="Heart", gender="woman"),
    VoicePreset(key="af_bella", language="en", name="Bella", gender="woman"),
    VoicePreset(key="af_nicole", language="en", name="Nicole", gender="woman"),
    VoicePreset(key="af_sarah", language="en", name="Sarah", gender="woman"),
    VoicePreset(key="af_sky", language="en", name="Sky", gender="woman"),
    VoicePreset(key="am_adam", language="en", name="Adam", gender="man"),
    VoicePreset(key="am_michael", language="en", name="Michael", gender="man"),
    # British English
    VoicePreset(key="bf_emma", language="en", name="Emma", gender="woman"),
    VoicePreset(key="bf_isabella", language="en", name="Isabella", gender="woman"),
    VoicePreset(key="bm_george", language="en", name="George", gender="man"),
    VoicePreset(key="bm_lewis", language="en", name="Lewis", gender="man"),
    # Spanish (requires espeak-ng)
    VoicePreset(key="ef_dora", language="es", name="Dora", gender="woman", requires_espeak=True),
    VoicePreset(key="em_alex", language="es", name="Alex", gender="man", requires_espeak=True),
    # French (requires espeak-ng)
    VoicePreset(key="ff_siwis", language="fr", name="Siwis", gender="woman", requires_espeak=True),
    # Italian (requires espeak-ng)
    VoicePreset(key="if_sara", language="it", name="Sara", gender="woman", requires_espeak=True),
    VoicePreset(key="im_nicola", language="it", name="Nicola", gender="man", requires_espeak=True),
    # Portuguese (requires espeak-ng)
    VoicePreset(key="pf_dora", language="pt", name="Dora", gender="woman", requires_espeak=True),
    VoicePreset(key="pm_alex", language="pt", name="Alex", gender="man", requires_espeak=True),
    # Hindi (requires espeak-ng)
    VoicePreset(key="hf_alpha", language="hi", name="Alpha", gender="woman", requires_espeak=True),
    VoicePreset(key="hm_omega", language="hi", name="Omega", gender="man", requires_espeak=True),
    # Chinese (requires espeak-ng)
    VoicePreset(key="zf_xiaobei", language="zh", name="Xiaobei", gender="woman", requires_espeak=True),
    VoicePreset(key="zm_yunjian", language="zh", name="Yunjian", gender="man", requires_espeak=True),
    # Parler-TTS voices (English only, description-driven)
    VoicePreset(key="parler_woman", language="en", name="Parler Woman", gender="woman", backend=TTSBackend.PARLER),
    VoicePreset(key="parler_man", language="en", name="Parler Man", gender="man", backend=TTSBackend.PARLER),
]

# Lookup rapido por key
VOICE_CATALOG: dict[str, VoicePreset] = {v.key: v for v in AVAILABLE_VOICES}

# Voces por defecto por idioma (Kokoro)
DEFAULT_VOICE_BY_LANG: dict[str, str] = {
    "en": "af_heart",
    "es": "em_alex",
    "fr": "ff_siwis",
    "it": "im_nicola",
    "pt": "pm_alex",
    "hi": "hm_omega",
    "zh": "zm_yunjian",
}

# Mapeo de lang code a Kokoro lang_code
KOKORO_LANG_CODES: dict[str, str] = {
    "en": "a",   # American English
    "es": "e",   # Spanish
    "fr": "f",   # French
    "it": "i",   # Italian
    "ja": "j",   # Japanese
    "ko": "k",   # Korean
    "pt": "p",   # Brazilian Portuguese
    "hi": "h",   # Hindi
    "zh": "z",   # Chinese
}


class VoiceParams(BaseModel):
    """Parametros de voz generados para un turno especifico.

    El Voice Parameter Generator transforma el emotional_state del agente
    en estos parametros que controlan la sintesis de voz.
    """

    voice_key: str = Field(
        default="af_heart",
        description="Preset de voz a usar",
    )
    speed: float = Field(
        default=1.0, ge=0.5, le=2.0,
        description="Velocidad del habla (1.0 = normal)",
    )
    pitch_semitones: float = Field(
        default=0.0, ge=-4.0, le=4.0,
        description="Pitch shift en semitonos (- = grave, + = agudo)",
    )
    volume: float = Field(
        default=1.0, ge=0.5, le=1.5,
        description="Multiplicador de volumen (< 1 = susurro, > 1 = fuerte)",
    )
    tremolo: float = Field(
        default=0.0, ge=0.0, le=0.15,
        description="Intensidad del tremolo (vibración nerviosa, 0 = ninguno)",
    )
    stage_direction: str = Field(
        default="",
        description="Direccion escenica inyectada en el texto (ej: '[speaking warmly]')",
    )
    backend: TTSBackend = Field(
        default=TTSBackend.KOKORO,
        description="Backend TTS a usar",
    )
    parler_description: str = Field(
        default="",
        description="Descripcion textual para Parler-TTS (solo si backend=parler)",
    )


class VoiceConfig(BaseModel):
    """Configuracion de voz por sesion."""

    mode: VoiceMode = Field(
        default=VoiceMode.TEXT_ONLY,
        description="Modo de voz activo",
    )
    default_voice: str = Field(
        default="af_heart",
        description="Voz por defecto para la sesion",
    )
    language: str = Field(
        default="en",
        description="Idioma principal de la sesion",
    )
    auto_speak: bool = Field(
        default=True,
        description="Si generar audio automaticamente con cada respuesta",
    )
    sample_rate: int = Field(
        default=24000,
        description="Sample rate del audio (24kHz Kokoro, 44.1kHz Parler)",
    )
    tts_backend: TTSBackend = Field(
        default=TTSBackend.KOKORO,
        description="Backend TTS preferido",
    )


def default_voice_config() -> VoiceConfig:
    """Configuracion de voz por defecto (text-only)."""
    return VoiceConfig()


def _check_espeak() -> bool:
    """Detecta si espeak-ng esta instalado en el sistema."""
    import shutil
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return True
    # Windows: check common install path
    from pathlib import Path
    for p in [Path("C:/Program Files/eSpeak NG/espeak-ng.exe"),
              Path("C:/Program Files (x86)/eSpeak NG/espeak-ng.exe")]:
        if p.exists():
            return True
    return False


_espeak_available: bool | None = None


def is_espeak_available() -> bool:
    """Cached check for espeak-ng availability."""
    global _espeak_available
    if _espeak_available is None:
        _espeak_available = _check_espeak()
    return _espeak_available


def get_available_voices() -> list[VoicePreset]:
    """Returns only voices that can actually work on this system."""
    espeak = is_espeak_available()
    return [v for v in AVAILABLE_VOICES if not v.requires_espeak or espeak]
