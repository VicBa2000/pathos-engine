"""Tests para Voice V1: TTS Emocional (Kokoro + Parler-TTS).

Verifica:
- Voice Parameter Generator: stage directions, speed, backend selection
- Voice models: VoiceConfig, VoiceParams, presets, TTSBackend
- Text preparation con stage directions
- Parler description generation
- Backend selection logic
- Rangos de todos los parametros para todas las emociones
- TTS Service: inicializacion, estados (mock — no requiere modelo real)
"""

import pytest

from pathos.models.emotion import BodyState, EmotionalState, Mood, MoodLabel, PrimaryEmotion
from pathos.models.voice import (
    AVAILABLE_VOICES,
    DEFAULT_VOICE_BY_LANG,
    KOKORO_LANG_CODES,
    TTSBackend,
    VOICE_CATALOG,
    VoiceConfig,
    VoiceMode,
    VoiceParams,
    VoicePreset,
    default_voice_config,
)
from pathos.voice.params import (
    _EMOTION_DIRECTIONS,
    _PARLER_EMOTION_DESC,
    apply_stress_annotations,
    compute_parler_description,
    compute_speed,
    compute_stage_direction,
    compute_voice_blend,
    generate_voice_params,
    inject_emotional_pauses,
    prepare_text_for_tts,
    should_use_parler,
)


# --- Helpers ---

def _state(
    emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.3,
    dominance: float = 0.5,
    certainty: float = 0.5,
    intensity: float = 0.3,
    warmth: float = 0.5,
    openness: float = 0.5,
    energy: float = 0.5,
    tension: float = 0.3,
    secondary: PrimaryEmotion | None = None,
) -> EmotionalState:
    return EmotionalState(
        valence=valence, arousal=arousal,
        dominance=dominance, certainty=certainty,
        primary_emotion=emotion, secondary_emotion=secondary,
        intensity=intensity,
        body_state=BodyState(energy=energy, tension=tension, openness=openness, warmth=warmth),
        mood=Mood(baseline_valence=0.0, baseline_arousal=0.3, label=MoodLabel.NEUTRAL),
    )


# =========================================================================
# Test: Voice Models
# =========================================================================

class TestVoiceModels:
    """Tests para modelos de datos de voz."""

    def test_voice_modes(self) -> None:
        """Los 3 modos de voz existen."""
        assert VoiceMode.TEXT_ONLY == "text_only"
        assert VoiceMode.VOICE_OUT == "voice_out"
        assert VoiceMode.FULL_VOICE == "full_voice"

    def test_tts_backends(self) -> None:
        """Los 2 backends TTS existen."""
        assert TTSBackend.KOKORO == "kokoro"
        assert TTSBackend.PARLER == "parler"

    def test_default_config_is_text_only(self) -> None:
        """Default: text-only (voz desactivada)."""
        config = default_voice_config()
        assert config.mode == VoiceMode.TEXT_ONLY
        assert config.sample_rate == 24000
        assert config.tts_backend == TTSBackend.KOKORO

    def test_available_voices_exist(self) -> None:
        """Voces disponibles en catalogo."""
        assert len(AVAILABLE_VOICES) > 10

    def test_voice_catalog_lookup(self) -> None:
        """Lookup por key funciona."""
        heart = VOICE_CATALOG.get("af_heart")
        assert heart is not None
        assert heart.language == "en"
        assert heart.gender == "woman"

    def test_spanish_voices(self) -> None:
        """Voces en espanol disponibles."""
        sp_voices = [v for v in AVAILABLE_VOICES if v.language == "es"]
        assert len(sp_voices) >= 2

    def test_default_voice_per_language(self) -> None:
        """Cada idioma tiene voz por defecto."""
        assert "en" in DEFAULT_VOICE_BY_LANG
        assert "es" in DEFAULT_VOICE_BY_LANG
        for key in DEFAULT_VOICE_BY_LANG.values():
            assert key in VOICE_CATALOG

    def test_kokoro_lang_codes(self) -> None:
        """Mapeo de idiomas a codigos Kokoro."""
        assert KOKORO_LANG_CODES["en"] == "a"
        assert KOKORO_LANG_CODES["es"] == "e"
        assert KOKORO_LANG_CODES["ja"] == "j"

    def test_voice_params_defaults(self) -> None:
        """VoiceParams tiene defaults razonables."""
        params = VoiceParams()
        assert params.speed == 1.0
        assert params.backend == TTSBackend.KOKORO
        assert params.voice_key == "af_heart"
        assert params.stage_direction == ""

    def test_voice_params_speed_range(self) -> None:
        """VoiceParams speed se clampea a rango."""
        params = VoiceParams(speed=0.5)
        assert params.speed >= 0.5
        params = VoiceParams(speed=2.0)
        assert params.speed <= 2.0

    def test_parler_voices_in_catalog(self) -> None:
        """Voces Parler estan en el catalogo."""
        parler_voices = [v for v in AVAILABLE_VOICES if v.backend == TTSBackend.PARLER]
        assert len(parler_voices) >= 2
        for v in parler_voices:
            assert v.language == "en"  # Parler solo EN


# =========================================================================
# Test: Stage Directions
# =========================================================================

class TestStageDirections:
    """Tests para la generacion de directivas escenicas."""

    def test_all_emotions_have_direction(self) -> None:
        """Todas las 19 emociones tienen direccion (neutral vacia)."""
        for emotion in PrimaryEmotion:
            assert emotion in _EMOTION_DIRECTIONS

    def test_neutral_no_direction(self) -> None:
        """Emocion neutral -> sin direccion."""
        state = _state(PrimaryEmotion.NEUTRAL, intensity=0.5)
        assert compute_stage_direction(state) == ""

    def test_joy_direction(self) -> None:
        """Joy con intensidad -> direccion con warmth."""
        state = _state(PrimaryEmotion.JOY, intensity=0.6)
        direction = compute_stage_direction(state)
        assert "warmth" in direction.lower() or "smile" in direction.lower()

    def test_sadness_direction(self) -> None:
        """Sadness -> direccion suave."""
        state = _state(PrimaryEmotion.SADNESS, intensity=0.5)
        direction = compute_stage_direction(state)
        assert "softly" in direction.lower() or "sadness" in direction.lower()

    def test_anger_direction(self) -> None:
        """Anger -> direccion firme."""
        state = _state(PrimaryEmotion.ANGER, intensity=0.7)
        direction = compute_stage_direction(state)
        assert "firm" in direction.lower() or "intensity" in direction.lower()

    def test_low_intensity_no_direction(self) -> None:
        """Intensidad muy baja -> sin direccion."""
        state = _state(PrimaryEmotion.JOY, intensity=0.1)
        assert compute_stage_direction(state) == ""

    def test_secondary_emotion_combined(self) -> None:
        """Con secondary emotion e intensidad alta -> direccion combinada."""
        state = _state(
            PrimaryEmotion.JOY, intensity=0.7,
            secondary=PrimaryEmotion.GRATITUDE,
        )
        direction = compute_stage_direction(state)
        assert len(direction) > 0
        assert "undertones" in direction.lower() or "warmth" in direction.lower()

    def test_secondary_same_as_primary_no_duplicate(self) -> None:
        """Si secondary == primary, no duplica."""
        state = _state(PrimaryEmotion.JOY, intensity=0.7, secondary=PrimaryEmotion.JOY)
        direction = compute_stage_direction(state)
        assert "undertones" not in direction.lower()


# =========================================================================
# Test: Speed
# =========================================================================

class TestSpeed:
    """Tests para el calculo de velocidad del habla."""

    def test_neutral_speed_near_default(self) -> None:
        """Estado neutral -> speed cercano a 1.0."""
        state = _state(arousal=0.3)
        speed = compute_speed(state)
        assert 0.85 <= speed <= 1.15

    def test_high_arousal_faster(self) -> None:
        """Alto arousal -> habla mas rapida."""
        speed_low = compute_speed(_state(arousal=0.1))
        speed_high = compute_speed(_state(arousal=0.9))
        assert speed_high > speed_low

    def test_sadness_slower(self) -> None:
        """Sadness con bajo arousal -> mas lento."""
        state = _state(PrimaryEmotion.SADNESS, valence=-0.5, arousal=0.2, intensity=0.5)
        speed = compute_speed(state)
        assert speed < 1.0

    def test_excitement_faster(self) -> None:
        """Excitement -> mas rapido."""
        state = _state(PrimaryEmotion.EXCITEMENT, arousal=0.8, intensity=0.7)
        speed = compute_speed(state)
        assert speed > 1.0

    def test_speed_clamped(self) -> None:
        """Speed siempre en [0.6, 1.5]."""
        for emotion in PrimaryEmotion:
            for arousal in [0.0, 0.5, 1.0]:
                state = _state(emotion, arousal=arousal, valence=-0.5)
                speed = compute_speed(state)
                assert 0.8 <= speed <= 1.3, f"speed={speed} for {emotion}, arousal={arousal}"


# =========================================================================
# Test: Parler Description
# =========================================================================

class TestParlerDescription:
    """Tests para la generacion de descripciones Parler-TTS."""

    def test_all_emotions_have_description(self) -> None:
        """Todas las 19 emociones tienen descripcion Parler."""
        for emotion in PrimaryEmotion:
            assert emotion in _PARLER_EMOTION_DESC

    def test_joy_description(self) -> None:
        """Joy -> descripcion con joy/warm."""
        state = _state(PrimaryEmotion.JOY, intensity=0.6)
        desc = compute_parler_description(state)
        assert "joy" in desc.lower() or "warm" in desc.lower()

    def test_high_intensity_adds_modifier(self) -> None:
        """Intensidad alta -> agrega 'strong emotional intensity'."""
        state = _state(PrimaryEmotion.ANGER, intensity=0.8)
        desc = compute_parler_description(state)
        assert "strong" in desc.lower()

    def test_low_intensity_adds_subtle(self) -> None:
        """Intensidad baja -> agrega whispering/subtle."""
        state = _state(PrimaryEmotion.JOY, intensity=0.2)
        desc = compute_parler_description(state)
        assert "whisper" in desc.lower() or "subtle" in desc.lower()

    def test_secondary_emotion_mixed(self) -> None:
        """Secondary emotion -> hints/undertone."""
        state = _state(PrimaryEmotion.JOY, intensity=0.6, secondary=PrimaryEmotion.SADNESS)
        desc = compute_parler_description(state)
        assert "undertone" in desc.lower() or "hints" in desc.lower() or "sadness" in desc.lower()


# =========================================================================
# Test: Backend Selection
# =========================================================================

class TestBackendSelection:
    """Tests para la logica de seleccion Kokoro vs Parler."""

    def test_non_english_always_kokoro(self) -> None:
        """Idioma no ingles -> siempre Kokoro."""
        state = _state(PrimaryEmotion.MIXED, intensity=0.8)
        assert should_use_parler(state, "es") is False
        assert should_use_parler(state, "ja") is False

    def test_simple_emotion_kokoro(self) -> None:
        """Emocion simple en ingles -> Kokoro."""
        state = _state(PrimaryEmotion.JOY, intensity=0.5)
        assert should_use_parler(state, "en") is False

    def test_complex_emotion_high_intensity_parler(self) -> None:
        """Emocion compleja + alta intensidad en ingles -> Parler."""
        state = _state(PrimaryEmotion.MIXED, intensity=0.7)
        assert should_use_parler(state, "en") is True

    def test_mixed_with_secondary_parler(self) -> None:
        """Secondary emotion fuerte en ingles -> Parler."""
        state = _state(PrimaryEmotion.JOY, intensity=0.7, secondary=PrimaryEmotion.SADNESS)
        assert should_use_parler(state, "en") is True

    def test_low_intensity_complex_stays_kokoro(self) -> None:
        """Emocion compleja pero baja intensidad -> Kokoro."""
        state = _state(PrimaryEmotion.MIXED, intensity=0.3)
        assert should_use_parler(state, "en") is False


# =========================================================================
# Test: generate_voice_params (full pipeline)
# =========================================================================

class TestGenerateVoiceParams:
    """Tests para la generacion completa de parametros."""

    def test_generates_all_fields(self) -> None:
        """Genera todos los campos de VoiceParams."""
        state = _state(PrimaryEmotion.JOY, intensity=0.6, arousal=0.7)
        params = generate_voice_params(state)
        assert isinstance(params, VoiceParams)
        # Voice key puede ser blend (af_heart,af_bella) o simple
        assert "af_" in params.voice_key
        assert params.speed > 0
        assert params.backend in (TTSBackend.KOKORO, TTSBackend.PARLER)
        assert len(params.stage_direction) > 0

    def test_custom_voice(self) -> None:
        """Respeta la voz por defecto pasada."""
        params = generate_voice_params(_state(), default_voice="em_alex")
        assert params.voice_key == "em_alex"

    def test_language_auto_selects_voice(self) -> None:
        """Idioma detectado auto-selecciona voz."""
        params = generate_voice_params(_state(), detected_language="es")
        assert params.voice_key == "em_alex"

    def test_all_emotions_produce_valid_params(self) -> None:
        """Todas las 19 emociones producen parametros validos."""
        for emotion in PrimaryEmotion:
            state = _state(emotion, intensity=0.5, arousal=0.5)
            params = generate_voice_params(state)
            assert 0.5 <= params.speed <= 2.0
            assert params.backend in (TTSBackend.KOKORO, TTSBackend.PARLER)

    def test_extreme_states_valid_params(self) -> None:
        """Estados extremos producen parametros validos (no overflow)."""
        extreme_high = _state(
            PrimaryEmotion.EXCITEMENT, valence=1.0, arousal=1.0,
            dominance=1.0, certainty=1.0, intensity=1.0,
            warmth=1.0, openness=1.0,
        )
        extreme_low = _state(
            PrimaryEmotion.HELPLESSNESS, valence=-1.0, arousal=0.0,
            dominance=0.0, certainty=0.0, intensity=0.0,
            warmth=0.0, openness=0.0,
        )
        for state in [extreme_high, extreme_low]:
            params = generate_voice_params(state)
            assert 0.5 <= params.speed <= 2.0

    def test_parler_selected_for_complex_en(self) -> None:
        """Emociones complejas en ingles -> Parler seleccionado."""
        state = _state(PrimaryEmotion.MIXED, intensity=0.8)
        params = generate_voice_params(state, detected_language="en")
        assert params.backend == TTSBackend.PARLER
        assert len(params.parler_description) > 0

    def test_kokoro_for_spanish(self) -> None:
        """Espanol -> siempre Kokoro."""
        state = _state(PrimaryEmotion.MIXED, intensity=0.8)
        params = generate_voice_params(state, detected_language="es")
        assert params.backend == TTSBackend.KOKORO


# =========================================================================
# Test: prepare_text_for_tts
# =========================================================================

class TestPrepareText:
    """Tests para la preparacion del texto para TTS."""

    def test_no_state_no_changes(self) -> None:
        """Sin state -> texto sin cambios."""
        assert prepare_text_for_tts("Hello", "") == "Hello"

    def test_no_direction_prepend_for_kokoro(self) -> None:
        """Kokoro: no prepende stage directions (las ignora)."""
        result = prepare_text_for_tts("Hello", "[speaking warmly]")
        assert "[speaking" not in result
        assert "Hello" in result

    def test_with_state_applies_annotations(self) -> None:
        """Con state de alta intensidad -> aplica anotaciones."""
        state = _state(PrimaryEmotion.JOY, intensity=0.8, valence=0.7)
        result = prepare_text_for_tts("I'm so happy today!", "", state=state)
        assert "happy" in result  # Texto preservado

    def test_preserves_core_text(self) -> None:
        """El texto core se preserva."""
        state = _state(PrimaryEmotion.NEUTRAL, intensity=0.1)
        text = "This is a message"
        result = prepare_text_for_tts(text, "", state=state)
        assert "This is a message" in result


class TestVoiceBlending:
    """Tests para la mezcla de voces por emocion."""

    def test_neutral_no_blend(self) -> None:
        """Neutral con baja intensidad -> sin blend."""
        state = _state(PrimaryEmotion.NEUTRAL, intensity=0.2)
        voice = compute_voice_blend(state, "af_heart")
        assert "," not in voice

    def test_joy_high_intensity_blends(self) -> None:
        """Joy con alta intensidad -> blend de voces."""
        state = _state(PrimaryEmotion.JOY, intensity=0.7)
        voice = compute_voice_blend(state, "af_heart")
        assert "," in voice  # Comma-separated blend

    def test_low_intensity_no_blend(self) -> None:
        """Baja intensidad -> sin blend independientemente de emocion."""
        state = _state(PrimaryEmotion.ANGER, intensity=0.2)
        voice = compute_voice_blend(state, "af_heart")
        assert "," not in voice

    def test_non_american_no_blend(self) -> None:
        """Voces no americanas no se blendean."""
        state = _state(PrimaryEmotion.JOY, intensity=0.8)
        voice = compute_voice_blend(state, "em_alex")
        assert voice == "em_alex"

    def test_british_no_blend(self) -> None:
        """Voces britanicas no se blendean."""
        state = _state(PrimaryEmotion.EXCITEMENT, intensity=0.9)
        voice = compute_voice_blend(state, "bf_emma")
        assert voice == "bf_emma"

    def test_male_voice_uses_male_blends(self) -> None:
        """Voces masculinas usan blend masculino."""
        state = _state(PrimaryEmotion.ANGER, intensity=0.7)
        voice = compute_voice_blend(state, "am_adam")
        assert "am_" in voice

    def test_all_emotions_produce_valid_blend(self) -> None:
        """Todas las emociones producen blends validos."""
        for emotion in PrimaryEmotion:
            state = _state(emotion, intensity=0.7)
            voice = compute_voice_blend(state, "af_heart")
            # Debe ser una voz valida o blend
            parts = voice.split(",")
            for p in parts:
                assert p.startswith("af_") or p.startswith("am_") or p == ""


class TestStressAnnotations:
    """Tests para anotaciones de enfasis."""

    def test_low_intensity_no_annotations(self) -> None:
        """Intensidad baja -> sin anotaciones."""
        state = _state(PrimaryEmotion.JOY, intensity=0.2, valence=0.5)
        result = apply_stress_annotations("I love this", state)
        assert "(+" not in result

    def test_positive_emphasis(self) -> None:
        """Palabras positivas con valence positivo -> enfasis."""
        state = _state(PrimaryEmotion.JOY, intensity=0.8, valence=0.7)
        result = apply_stress_annotations("This is really wonderful", state)
        assert "(+" in result

    def test_negative_emphasis(self) -> None:
        """Palabras negativas con valence negativo -> enfasis."""
        state = _state(PrimaryEmotion.ANGER, intensity=0.8, valence=-0.7)
        result = apply_stress_annotations("I hate this terrible thing", state)
        assert "(+" in result

    def test_max_annotations_limited(self) -> None:
        """Maximo 3 anotaciones por texto."""
        state = _state(PrimaryEmotion.JOY, intensity=0.9, valence=0.8)
        text = "really very truly extremely absolutely completely totally"
        result = apply_stress_annotations(text, state)
        assert result.count("(+") <= 3

    def test_high_intensity_higher_stress(self) -> None:
        """Intensidad > 0.7 -> stress level +2."""
        state = _state(PrimaryEmotion.JOY, intensity=0.8, valence=0.7)
        result = apply_stress_annotations("This is really great", state)
        if "(+" in result:
            assert "(+2)" in result


class TestEmotionalPauses:
    """Tests para pausas emocionales."""

    def test_low_intensity_no_pauses(self) -> None:
        """Baja intensidad -> sin pausas extra."""
        state = _state(PrimaryEmotion.SADNESS, intensity=0.2)
        text = "Hello, how are you"
        result = inject_emotional_pauses(text, state)
        assert result == text

    def test_sadness_adds_pauses(self) -> None:
        """Sadness con alta intensidad -> reemplaza comas con elipsis."""
        state = _state(PrimaryEmotion.SADNESS, intensity=0.6)
        text = "I feel, so tired, and empty"
        result = inject_emotional_pauses(text, state)
        assert "..." in result

    def test_anxiety_adds_hesitation(self) -> None:
        """Anxiety -> agrega hesitacion."""
        state = _state(PrimaryEmotion.ANXIETY, intensity=0.7)
        text = "I think, maybe, we should go"
        result = inject_emotional_pauses(text, state)
        assert "..." in result

    def test_relief_adds_sigh(self) -> None:
        """Relief -> agrega suspiro al inicio."""
        state = _state(PrimaryEmotion.RELIEF, intensity=0.6)
        text = "Thank goodness"
        result = inject_emotional_pauses(text, state)
        assert result.startswith("...")


# =========================================================================
# Test: TTS Service (unit tests — sin modelo real)
# =========================================================================

class TestTTSService:
    """Tests del servicio TTS sin cargar modelo."""

    def test_service_not_initialized_by_default(self) -> None:
        """El servicio no esta inicializado por defecto (lazy)."""
        from pathos.voice.tts import TTSService
        service = TTSService()
        assert service.is_initialized is False

    def test_list_available_voices(self) -> None:
        """Lista voces (vacia sin modelo cargado)."""
        from pathos.voice.tts import TTSService
        service = TTSService()
        voices = service.list_available_voices()
        assert isinstance(voices, list)

    def test_singleton_pattern(self) -> None:
        """get_tts_service retorna la misma instancia."""
        from pathos.voice.tts import get_tts_service
        s1 = get_tts_service()
        s2 = get_tts_service()
        assert s1 is s2

    def test_kokoro_service_singleton(self) -> None:
        """get_kokoro_service retorna la misma instancia."""
        from pathos.voice.tts_kokoro import get_kokoro_service
        s1 = get_kokoro_service()
        s2 = get_kokoro_service()
        assert s1 is s2

    def test_parler_service_singleton(self) -> None:
        """get_parler_service retorna la misma instancia."""
        from pathos.voice.tts_parler import get_parler_service
        s1 = get_parler_service()
        s2 = get_parler_service()
        assert s1 is s2


# =========================================================================
# Test: Integration scenarios
# =========================================================================

class TestVoiceIntegration:
    """Tests de escenarios de integracion."""

    def test_emotional_journey_params(self) -> None:
        """Una secuencia emocional produce parametros que cambian coherentemente."""
        # Neutral -> Joy -> Anger -> Sadness
        emotions = [
            (PrimaryEmotion.NEUTRAL, 0.1, 0.3),
            (PrimaryEmotion.JOY, 0.7, 0.7),
            (PrimaryEmotion.ANGER, 0.8, 0.8),
            (PrimaryEmotion.SADNESS, 0.5, 0.2),
        ]
        params_list = []
        for emotion, intensity, arousal in emotions:
            state = _state(emotion, intensity=intensity, arousal=arousal)
            params_list.append(generate_voice_params(state))

        # Joy: direction con warmth, speed normal-alto
        assert "warmth" in params_list[1].stage_direction.lower() or "smile" in params_list[1].stage_direction.lower()

        # Anger: alta intensidad, speed alto
        assert params_list[2].speed > params_list[0].speed

        # Sadness: bajo arousal, speed bajo
        assert params_list[3].speed < params_list[2].speed

    def test_text_only_mode_no_voice_needed(self) -> None:
        """En text_only, no se necesita nada de voice."""
        config = default_voice_config()
        assert config.mode == VoiceMode.TEXT_ONLY

    def test_voice_out_mode(self) -> None:
        """voice_out mode se configura correctamente."""
        config = VoiceConfig(mode=VoiceMode.VOICE_OUT, default_voice="em_alex", language="es")
        assert config.mode == VoiceMode.VOICE_OUT
        assert config.default_voice == "em_alex"

    def test_full_tts_text_preparation(self) -> None:
        """Pipeline completo: estado -> params -> texto preparado."""
        state = _state(PrimaryEmotion.EXCITEMENT, intensity=0.8, arousal=0.9, valence=0.7)
        params = generate_voice_params(state)
        prepared = prepare_text_for_tts("I'm so happy to help you!", params.stage_direction, state=state)
        # Stage directions NO se prependen para Kokoro
        assert "[speaking" not in prepared
        # Texto core preservado
        assert "happy" in prepared
        assert "help" in prepared
