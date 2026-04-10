"""Voice Parameter Generator - emotional_state -> VoiceParams.

Transforma el estado emocional del agente en parametros que controlan
la sintesis de voz. Estrategias:

1. SPEED: arousal alto -> habla mas rapida, bajo -> mas lenta
2. VOICE BLENDING: mezcla de voces Kokoro por emocion (style vector interpolation)
3. STRESS ANNOTATIONS: enfasis en palabras clave via [word](+N) syntax de Kokoro
4. PARLER DESCRIPTIONS: descripcion textual de la voz (para Parler-TTS)
5. BACKEND SELECTION: Kokoro para basicas, Parler para complejas (EN only)
6. VOICE PRESET: mapeo por idioma detectado
7. EMOTIONAL PAUSES: insercion de pausas en puntuacion segun emocion
"""

import logging
import re

from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.voice import DEFAULT_VOICE_BY_LANG, TTSBackend, VoiceParams

logger = logging.getLogger(__name__)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# --- Voice Blending por emocion ---
# Kokoro soporta mezcla de voces via "voice1,voice2" que interpola style vectors.
# Cada voz tiene prosodia y timbre distintos baked-in:
# - af_heart: calida, suave (buena para joy, gratitude, contentment)
# - af_bella: expresiva, dramatica (buena para excitement, surprise)
# - af_sarah: clara, seria (buena para anger, alertness)
# - af_sky: eterea, suave (buena para sadness, contemplation)
# - af_nicole: energica, bright (buena para hope, relief)
# - am_adam: grave, firme (buena para anger, dominance)
# - am_michael: calido, profundo (buena para contemplation, sadness)

# Mapeo: emocion -> (voz_primaria, voz_secundaria, ratio_blend)
# El ratio indica cuanto de la voz secundaria mezclar (0 = solo primaria)
# Solo se activa cuando la intensidad > threshold
_EMOTION_VOICE_BLENDS: dict[PrimaryEmotion, tuple[str, str, float]] = {
    # Positivas: heart (calida) + bella (expresiva)
    PrimaryEmotion.JOY: ("af_heart", "af_bella", 0.3),
    PrimaryEmotion.EXCITEMENT: ("af_bella", "af_nicole", 0.4),
    PrimaryEmotion.GRATITUDE: ("af_heart", "af_sky", 0.2),
    PrimaryEmotion.HOPE: ("af_nicole", "af_heart", 0.3),
    PrimaryEmotion.CONTENTMENT: ("af_heart", "af_sky", 0.3),
    PrimaryEmotion.RELIEF: ("af_sky", "af_heart", 0.3),
    # Negativas
    PrimaryEmotion.ANGER: ("af_sarah", "af_bella", 0.3),
    PrimaryEmotion.FRUSTRATION: ("af_sarah", "af_nicole", 0.2),
    PrimaryEmotion.FEAR: ("af_sky", "af_bella", 0.2),
    PrimaryEmotion.ANXIETY: ("af_sky", "af_nicole", 0.3),
    PrimaryEmotion.SADNESS: ("af_sky", "af_heart", 0.2),
    PrimaryEmotion.HELPLESSNESS: ("af_sky", "af_sarah", 0.2),
    PrimaryEmotion.DISAPPOINTMENT: ("af_sky", "af_sarah", 0.15),
    # Neutras
    PrimaryEmotion.SURPRISE: ("af_bella", "af_nicole", 0.4),
    PrimaryEmotion.ALERTNESS: ("af_nicole", "af_sarah", 0.3),
    PrimaryEmotion.CONTEMPLATION: ("af_sky", "af_heart", 0.3),
    PrimaryEmotion.INDIFFERENCE: ("af_sarah", "af_sky", 0.15),
    PrimaryEmotion.MIXED: ("af_heart", "af_sky", 0.4),
    PrimaryEmotion.NEUTRAL: ("af_heart", "", 0.0),
}

# Mapeo equivalente para voces masculinas
_EMOTION_VOICE_BLENDS_MALE: dict[PrimaryEmotion, tuple[str, str, float]] = {
    PrimaryEmotion.JOY: ("am_michael", "am_adam", 0.2),
    PrimaryEmotion.EXCITEMENT: ("am_adam", "am_michael", 0.3),
    PrimaryEmotion.GRATITUDE: ("am_michael", "am_adam", 0.15),
    PrimaryEmotion.HOPE: ("am_michael", "am_adam", 0.2),
    PrimaryEmotion.CONTENTMENT: ("am_michael", "am_adam", 0.15),
    PrimaryEmotion.RELIEF: ("am_michael", "am_adam", 0.2),
    PrimaryEmotion.ANGER: ("am_adam", "am_michael", 0.3),
    PrimaryEmotion.FRUSTRATION: ("am_adam", "am_michael", 0.2),
    PrimaryEmotion.FEAR: ("am_michael", "am_adam", 0.2),
    PrimaryEmotion.ANXIETY: ("am_michael", "am_adam", 0.25),
    PrimaryEmotion.SADNESS: ("am_michael", "am_adam", 0.15),
    PrimaryEmotion.HELPLESSNESS: ("am_michael", "am_adam", 0.2),
    PrimaryEmotion.DISAPPOINTMENT: ("am_michael", "am_adam", 0.15),
    PrimaryEmotion.SURPRISE: ("am_adam", "am_michael", 0.3),
    PrimaryEmotion.ALERTNESS: ("am_adam", "am_michael", 0.25),
    PrimaryEmotion.CONTEMPLATION: ("am_michael", "am_adam", 0.2),
    PrimaryEmotion.INDIFFERENCE: ("am_adam", "am_michael", 0.1),
    PrimaryEmotion.MIXED: ("am_michael", "am_adam", 0.3),
    PrimaryEmotion.NEUTRAL: ("am_michael", "", 0.0),
}

_BLEND_INTENSITY_THRESHOLD = 0.3  # Debajo de esta intensidad, no blendear


# --- Stage Directions (legacy, para metadata/research display) ---
# Nota: Kokoro NO interpreta stage directions. Se mantienen solo para
# informacion en el research panel y para Parler-TTS.
_EMOTION_DIRECTIONS: dict[PrimaryEmotion, str] = {
    PrimaryEmotion.JOY: "[speaking with warmth and a smile]",
    PrimaryEmotion.EXCITEMENT: "[speaking with enthusiasm and energy]",
    PrimaryEmotion.GRATITUDE: "[speaking with sincere gratitude and warmth]",
    PrimaryEmotion.HOPE: "[speaking with gentle optimism]",
    PrimaryEmotion.CONTENTMENT: "[speaking calmly and peacefully]",
    PrimaryEmotion.RELIEF: "[speaking with a relieved sigh]",
    PrimaryEmotion.ANGER: "[speaking with firm intensity]",
    PrimaryEmotion.FRUSTRATION: "[speaking with visible frustration]",
    PrimaryEmotion.FEAR: "[speaking with nervous tension]",
    PrimaryEmotion.ANXIETY: "[speaking with worried hesitation]",
    PrimaryEmotion.SADNESS: "[speaking softly with sadness]",
    PrimaryEmotion.HELPLESSNESS: "[speaking with a heavy, defeated tone]",
    PrimaryEmotion.DISAPPOINTMENT: "[speaking with quiet disappointment]",
    PrimaryEmotion.SURPRISE: "[speaking with genuine surprise]",
    PrimaryEmotion.ALERTNESS: "[speaking with sharp attention]",
    PrimaryEmotion.CONTEMPLATION: "[speaking thoughtfully and slowly]",
    PrimaryEmotion.INDIFFERENCE: "[speaking flatly]",
    PrimaryEmotion.MIXED: "[speaking with conflicted emotion]",
    PrimaryEmotion.NEUTRAL: "",
}

_MIN_INTENSITY_FOR_DIRECTION = 0.2


# --- Stress annotation keywords ---
# Palabras emocionales comunes que reciben enfasis (+1 o +2) segun la emocion
_POSITIVE_EMPHASIS_WORDS = {
    "love", "happy", "glad", "wonderful", "amazing", "great", "fantastic",
    "beautiful", "excellent", "perfect", "brilliant", "awesome", "incredible",
    "thanks", "thank", "grateful", "appreciate",
}
_NEGATIVE_EMPHASIS_WORDS = {
    "hate", "terrible", "awful", "horrible", "worst", "never", "nothing",
    "angry", "furious", "scared", "afraid", "worried", "sorry", "sad",
    "hurt", "pain", "wrong", "bad", "failed", "lost", "alone",
}
_INTENSITY_EMPHASIS_WORDS = {
    "very", "really", "so", "extremely", "absolutely", "completely",
    "totally", "utterly", "deeply", "truly", "incredibly",
}


# --- Parler-TTS emotion descriptions ---
# Descripciones textuales ricas para control emocional fino

# Descripciones optimizadas para Parler-TTS mini.
# Parler responde mejor a: adjetivos acusticos concretos (pitch, pace, tone,
# breathiness, volume) en 1-2 frases cortas. Evitar abstracciones.
_PARLER_EMOTION_DESC: dict[PrimaryEmotion, str] = {
    PrimaryEmotion.JOY: "A female speaker with a warm, bright voice and a slightly high pitch delivers a happy, upbeat speech",
    PrimaryEmotion.EXCITEMENT: "A female speaker with a high-pitched, fast-paced, energetic voice delivers an animated and enthusiastic speech",
    PrimaryEmotion.GRATITUDE: "A female speaker with a warm, soft, sincere voice delivers a heartfelt and gentle speech",
    PrimaryEmotion.HOPE: "A female speaker with a gentle, slightly rising pitch delivers an optimistic and encouraging speech",
    PrimaryEmotion.CONTENTMENT: "A female speaker with a calm, relaxed, low-energy voice delivers a peaceful and satisfied speech",
    PrimaryEmotion.RELIEF: "A female speaker with a breathy, relaxing voice and falling pitch delivers a relieved speech",
    PrimaryEmotion.ANGER: "A female speaker with a loud, firm, low-pitched voice delivers an intense and forceful speech",
    PrimaryEmotion.FRUSTRATION: "A female speaker with a tense, clipped, slightly loud voice delivers a frustrated and impatient speech",
    PrimaryEmotion.FEAR: "A female speaker with a quiet, shaky, high-pitched voice delivers a trembling and fearful speech",
    PrimaryEmotion.ANXIETY: "A female speaker with a hesitant, unsteady, slightly fast voice delivers a nervous and worried speech",
    PrimaryEmotion.SADNESS: "A female speaker with a slow, soft, low-pitched voice delivers a melancholic and sorrowful speech",
    PrimaryEmotion.HELPLESSNESS: "A female speaker with a flat, monotone, low-energy voice delivers a heavy and defeated speech",
    PrimaryEmotion.DISAPPOINTMENT: "A female speaker with a quiet, falling pitch delivers a disappointed and subdued speech",
    PrimaryEmotion.SURPRISE: "A female speaker with a rising pitch and wide vocal range delivers a surprised and astonished speech",
    PrimaryEmotion.ALERTNESS: "A female speaker with a sharp, clear, slightly fast voice delivers an attentive and focused speech",
    PrimaryEmotion.CONTEMPLATION: "A female speaker with a slow, measured, thoughtful voice delivers a reflective and deliberate speech",
    PrimaryEmotion.INDIFFERENCE: "A female speaker with a flat, monotone voice and low energy delivers a disinterested speech",
    PrimaryEmotion.MIXED: "A female speaker with a voice that shifts between high and low pitch delivers an emotionally conflicted speech",
    PrimaryEmotion.NEUTRAL: "A female speaker with a clear, natural voice at a moderate pace delivers a conversational speech",
}

# Emociones que se benefician de Parler (complejas, mixed, alta intensidad)
_COMPLEX_EMOTIONS: set[PrimaryEmotion] = {
    PrimaryEmotion.MIXED,
    PrimaryEmotion.HELPLESSNESS,
    PrimaryEmotion.ANXIETY,
    PrimaryEmotion.FEAR,
    PrimaryEmotion.RELIEF,
}


def compute_stage_direction(state: EmotionalState) -> str:
    """Genera la direccion escenica basada en el estado emocional.

    Nota: Kokoro no interpreta stage directions. Estas se usan para:
    - Metadata en el research panel
    - Parler-TTS (que si las interpreta)
    """
    if state.intensity < _MIN_INTENSITY_FOR_DIRECTION:
        return ""

    direction = _EMOTION_DIRECTIONS.get(state.primary_emotion, "")

    if state.secondary_emotion and state.secondary_emotion != state.primary_emotion:
        secondary_dir = _EMOTION_DIRECTIONS.get(state.secondary_emotion, "")
        if secondary_dir and state.intensity > 0.5:
            primary_part = direction.rstrip("]")
            secondary_clean = secondary_dir.lstrip("[").rstrip("]")
            if "with" in secondary_clean:
                secondary_tone = secondary_clean.split("with", 1)[1].strip()
                direction = f"{primary_part}, with undertones of {secondary_tone}]"

    return direction


def compute_voice_blend(state: EmotionalState, default_voice: str) -> str:
    """Computa la voz o mezcla de voces para Kokoro basada en emocion.

    Kokoro soporta interpolacion de style vectors via "voice1,voice2".
    Distintas voces tienen prosodia diferente (calida, energica, grave, etc.)
    Mapeamos emociones a combinaciones que producen la prosodia deseada.

    Solo aplica para voces EN americanas (af_*, am_*). Para otros idiomas
    o voces britanicas, retorna la voz sin modificar.
    """
    # Solo blendear voces americanas EN
    if not default_voice.startswith(("af_", "am_")):
        return default_voice

    is_male = default_voice.startswith("am_")
    blend_map = _EMOTION_VOICE_BLENDS_MALE if is_male else _EMOTION_VOICE_BLENDS

    blend = blend_map.get(state.primary_emotion)
    if not blend:
        return default_voice

    primary, secondary, ratio = blend

    # No blendear si la intensidad es baja
    if state.intensity < _BLEND_INTENSITY_THRESHOLD or ratio == 0.0 or not secondary:
        return primary

    # Escalar el blend por intensidad (mas intenso = mas blend)
    effective_ratio = ratio * min(state.intensity, 1.0)
    if effective_ratio < 0.1:
        return primary

    # Kokoro usa comma-separated voices para blend
    return f"{primary},{secondary}"


def apply_stress_annotations(text: str, state: EmotionalState) -> str:
    """Aplica stress annotations de Kokoro a palabras emocionales.

    Kokoro soporta [word](+N) para aumentar enfasis en palabras.
    Seleccionamos palabras emocionalmente significativas y les damos
    enfasis proporcional a la intensidad.

    Solo aplica con intensidad > 0.4 para evitar sobre-enfatizar texto neutro.
    """
    if state.intensity < 0.4:
        return text

    # Determinar nivel de stress y palabras a enfatizar
    stress_level = 1 if state.intensity < 0.7 else 2

    # Seleccionar que tipo de palabras enfatizar segun emocion
    target_words: set[str] = set()
    if state.valence > 0.2:
        target_words = _POSITIVE_EMPHASIS_WORDS
    elif state.valence < -0.2:
        target_words = _NEGATIVE_EMPHASIS_WORDS

    # Siempre enfatizar intensificadores
    target_words = target_words | _INTENSITY_EMPHASIS_WORDS

    if not target_words:
        return text

    # Aplicar anotaciones (max 3 por texto para no saturar)
    annotated_count = 0
    max_annotations = 3

    def _annotate_word(match: re.Match[str]) -> str:
        nonlocal annotated_count
        word = match.group(0)
        if annotated_count >= max_annotations:
            return word
        if word.lower().rstrip(".,!?;:") in target_words:
            annotated_count += 1
            # Preservar puntuacion al final
            core = word.rstrip(".,!?;:")
            trailing = word[len(core):]
            return f"[{core}](+{stress_level}){trailing}"
        return word

    return re.sub(r'\b\S+\b[.,!?;:]*', _annotate_word, text)


def inject_emotional_pauses(text: str, state: EmotionalState) -> str:
    """Inyecta pausas emocionales via puntuacion.

    Emociones lentas (sadness, contemplation) -> pausas mas largas en comas/puntos.
    Emociones rapidas (excitement, anger) -> menos pausas.
    Hesitacion (anxiety, fear) -> agrega "..." entre frases.
    """
    if state.intensity < 0.3:
        return text

    # Sadness, helplessness, contemplation: pausas mas largas (doble coma -> ellipsis)
    if state.primary_emotion in (
        PrimaryEmotion.SADNESS, PrimaryEmotion.HELPLESSNESS,
        PrimaryEmotion.CONTEMPLATION, PrimaryEmotion.DISAPPOINTMENT,
    ) and state.intensity > 0.4:
        # Insertar pausas extra despues de comas (Kokoro respeta puntuacion)
        text = re.sub(r',\s', '... ', text, count=2)

    # Anxiety, fear: hesitacion via "..." en algunas comas
    elif state.primary_emotion in (
        PrimaryEmotion.ANXIETY, PrimaryEmotion.FEAR,
    ) and state.intensity > 0.5:
        # Reemplazar algunas comas con hesitacion
        parts = text.split(', ', 2)
        if len(parts) > 1:
            text = '... '.join(parts[:2])
            if len(parts) > 2:
                text += ', ' + parts[2]

    # Relief: suspiro via "..." al inicio
    elif state.primary_emotion == PrimaryEmotion.RELIEF and state.intensity > 0.5:
        text = "... " + text

    return text


def compute_speed(state: EmotionalState) -> float:
    """Computa velocidad del habla basada en arousal y emocion.

    Alto arousal -> habla mas rapida (excitement, anger, fear)
    Bajo arousal -> habla mas lenta (sadness, contemplation)
    Rango: [0.85, 1.25] — sutil para que combine bien con pitch/blend.
    Nota: _effective_speed multiplica speed × pitch_factor, asi que
    el speed real puede ser mayor/menor que estos valores.
    """
    base = 1.0

    # Arousal: mas activacion -> mas rapido (moderado)
    arousal_adj = (state.arousal - 0.3) * 0.35  # [-0.105, 0.245]

    # Intensidad amplifica el efecto (pero menos agresivo)
    intensity_mult = 0.6 + state.intensity * 0.4  # [0.6, 1.0]
    arousal_adj *= intensity_mult

    # Valence negativa + baja energia -> un poco mas lento
    if state.valence < -0.3 and state.arousal < 0.4:
        arousal_adj -= 0.06

    # Emociones con ajustes especificos (reducidos)
    emotion_adj: dict[PrimaryEmotion, float] = {
        PrimaryEmotion.EXCITEMENT: 0.1,
        PrimaryEmotion.ANGER: 0.08,
        PrimaryEmotion.FEAR: 0.08,
        PrimaryEmotion.JOY: 0.05,
        PrimaryEmotion.ANXIETY: 0.06,
        PrimaryEmotion.SADNESS: -0.06,
        PrimaryEmotion.CONTEMPLATION: -0.06,
        PrimaryEmotion.HELPLESSNESS: -0.08,
        PrimaryEmotion.CONTENTMENT: -0.03,
    }
    arousal_adj += emotion_adj.get(state.primary_emotion, 0.0)

    speed = base + arousal_adj
    return round(_clamp(speed, 0.85, 1.25), 2)


def compute_parler_description(state: EmotionalState) -> str:
    """Genera descripcion concisa para Parler-TTS usando el estado emocional.

    Parler-TTS mini funciona mejor con descripciones cortas (~100-150 chars)
    que usan adjetivos acusticos concretos. Descripciones muy largas
    diluyen la instruccion y el modelo pierde foco.

    Estrategia: base (emocion primaria) + max 2 modificadores mas impactantes.
    """
    # Base: emocion primaria (ya incluye pitch, pace, tone)
    desc = _PARLER_EMOTION_DESC.get(state.primary_emotion, _PARLER_EMOTION_DESC[PrimaryEmotion.NEUTRAL])

    # Elegir max 2 modificadores que mas impactan la prosodia
    modifiers: list[str] = []

    # 1. Intensidad (el modificador mas importante)
    if state.intensity > 0.75:
        modifiers.append("with very strong intensity")
    elif state.intensity < 0.25:
        modifiers.append("very subtly, almost whispering")

    # 2. Body state: el rasgo mas extremo
    body = state.body_state
    extremes: list[tuple[float, str]] = [
        (body.tension, "with a tight, strained voice" if body.tension > 0.65 else ""),
        (1.0 - body.energy, "with a quiet, low-energy voice" if body.energy < 0.3 else ""),
        (body.warmth, "with a warm, caring tone" if body.warmth > 0.7 else ""),
        (1.0 - body.warmth, "with a cold, distant tone" if body.warmth < 0.25 else ""),
    ]
    # Tomar solo el body modifier mas extremo
    best_body = max(extremes, key=lambda x: x[0])
    if best_body[1]:
        modifiers.append(best_body[1])

    # 3. Secondary emotion (solo si es fuerte y diferente)
    if (state.secondary_emotion
            and state.secondary_emotion != state.primary_emotion
            and state.intensity > 0.5
            and len(modifiers) < 2):
        _SECONDARY_HINTS: dict[PrimaryEmotion, str] = {
            PrimaryEmotion.JOY: "with hints of warmth",
            PrimaryEmotion.SADNESS: "with an undertone of sadness",
            PrimaryEmotion.ANGER: "with a hint of anger",
            PrimaryEmotion.FEAR: "with a touch of fear",
            PrimaryEmotion.ANXIETY: "with nervous undertones",
            PrimaryEmotion.HOPE: "with a glimmer of hope",
            PrimaryEmotion.GRATITUDE: "with grateful undertones",
            PrimaryEmotion.SURPRISE: "with a note of surprise",
        }
        hint = _SECONDARY_HINTS.get(state.secondary_emotion)
        if hint:
            modifiers.append(hint)

    # Construir descripcion final (max ~150 chars)
    if modifiers:
        desc += ", " + ", ".join(modifiers[:2])

    return desc


def should_use_parler(
    state: EmotionalState,
    detected_language: str | None,
    user_backend: TTSBackend | None = None,
) -> bool:
    """Decide si usar Parler-TTS en vez de Kokoro.

    Si el usuario selecciono manualmente un backend (user_backend), se respeta
    esa eleccion — Parler para todas las emociones si es ingles.
    Si no, usa routing automatico: Parler solo para emociones complejas.
    """
    lang = detected_language or "en"

    # Parler solo soporta ingles
    if lang != "en":
        return False

    # Si el usuario selecciono Parler manualmente, usarlo para todo
    if user_backend == TTSBackend.PARLER:
        return True

    # Si el usuario selecciono Kokoro manualmente, nunca Parler
    if user_backend == TTSBackend.KOKORO:
        return False

    # Routing automatico: emociones complejas con alta intensidad
    if state.primary_emotion in _COMPLEX_EMOTIONS and state.intensity > 0.5:
        return True

    # Emociones con secondary emotion fuerte (mezcla emocional)
    if (state.secondary_emotion
            and state.secondary_emotion != state.primary_emotion
            and state.intensity > 0.6):
        return True

    return False


# --- Language detection ---

_LANG_KEYWORDS: dict[str, set[str]] = {
    "es": {
        "hola", "que", "como", "estas", "esta", "soy", "eres", "que",
        "como", "por", "para", "pero", "tambien", "bien", "mal",
        "tengo", "tienes", "tiene", "quiero", "puedo", "puedes", "dime",
        "siento", "gracias", "bueno", "buena", "hoy", "ayer", "manana",
        "vida", "mundo", "todo", "nada", "algo", "mucho", "poco", "muy",
        "buenos", "buenas", "dias", "noches", "tardes",
        "ser", "hacer", "decir", "pensar", "creo", "si", "no", "verdad",
        "amor", "tiempo", "dia", "siempre", "nunca", "cuando",
        "donde", "porque", "aunque", "entonces", "despues",
        "ahora", "aqui", "necesito", "ayuda", "favor", "perdon",
        "triste", "feliz", "miedo", "enojado", "contento",
    },
    "fr": {
        "je", "suis", "est", "les", "des", "une", "pas", "vous", "nous",
        "avec", "pour", "dans", "qui", "que", "sur", "mais", "tout",
        "tres", "bien", "merci", "bonjour", "comment", "pourquoi", "parce",
        "aussi", "encore", "jamais", "toujours", "oui", "non", "ici",
    },
    "it": {
        "sono", "sei", "siamo", "questo", "questa", "quello", "quella",
        "come", "stai", "bene", "male", "grazie", "buongiorno", "ciao",
        "perche", "anche", "sempre", "mai", "tutto", "niente",
        "molto", "poco", "piu", "della", "dello", "degli",
    },
    "pt": {
        "eu", "sou", "voce", "como", "esta", "bem", "mal",
        "obrigado", "obrigada", "sim", "nao", "porque", "tambem",
        "muito", "pouco", "tudo", "nada", "sempre", "nunca",
        "aqui", "onde", "quando", "mais", "menos", "bom", "mau",
    },
    "ja": {
        "の", "は", "が", "を", "に", "で", "と", "も", "か", "ね",
        "です", "ます", "した", "する", "こと", "もの", "ない", "ある",
        "これ", "それ", "あの", "この", "どう", "なに", "なぜ",
    },
    "ko": {
        "은", "는", "이", "가", "를", "에", "서", "도", "와", "과",
        "입니다", "습니다", "있다", "없다", "하다", "되다", "않다",
    },
}


def detect_language(text: str) -> str:
    """Detecta el idioma del texto usando heuristica de palabras."""
    if not text or not text.strip():
        return "en"

    for ch in text:
        if "\u3040" <= ch <= "\u309f" or "\u30a0" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff":
            return "ja"
        if "\uac00" <= ch <= "\ud7af":
            return "ko"

    words = set(text.lower().split())
    scores: dict[str, int] = {}
    for lang, keywords in _LANG_KEYWORDS.items():
        score = len(words & keywords)
        if score > 0:
            scores[lang] = score

    if scores:
        best_lang = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best_lang

    if len(text.split()) >= 4:
        try:
            from langdetect import detect
            lang = detect(text)
            if lang in DEFAULT_VOICE_BY_LANG:
                return lang
        except Exception:
            pass

    return "en"


def generate_voice_params(
    state: EmotionalState,
    default_voice: str = "af_heart",
    detected_language: str | None = None,
    user_backend: TTSBackend | None = None,
) -> VoiceParams:
    """Genera todos los parametros de voz para un turno.

    Funcion principal del Voice Parameter Generator.
    Toma el estado emocional completo y produce VoiceParams.
    Decide automaticamente si usar Kokoro o Parler-TTS (o respeta eleccion del usuario).
    Aplica voice blending para expresividad real via style vector interpolation.

    Args:
        state: Estado emocional actual del agente.
        default_voice: Voz por defecto del usuario.
        detected_language: Idioma detectado del texto.
        user_backend: Backend seleccionado manualmente por el usuario (None = auto).
    """
    # Usar la voz que el usuario eligió. Solo auto-seleccionar por idioma
    # si la voz actual no matchea el idioma detectado (ej: voz EN pero texto ES)
    voice_key = default_voice
    if detected_language and detected_language in DEFAULT_VOICE_BY_LANG:
        # Verificar si la voz actual es del mismo idioma que el texto
        voice_lang_char = default_voice[0] if default_voice else "a"
        lang_codes = {"a": "en", "b": "en", "e": "es", "f": "fr", "i": "it",
                      "j": "ja", "k": "ko", "p": "pt", "h": "hi", "z": "zh"}
        voice_lang = lang_codes.get(voice_lang_char, "en")
        if voice_lang != detected_language:
            voice_key = DEFAULT_VOICE_BY_LANG[detected_language]
            logger.info("Voice auto-switched: %s (detected language: %s)", voice_key, detected_language)

    direction = compute_stage_direction(state)
    speed = compute_speed(state)

    # Decidir backend (respetando eleccion manual del usuario)
    use_parler = should_use_parler(state, detected_language, user_backend=user_backend)
    backend = TTSBackend.PARLER if use_parler else TTSBackend.KOKORO
    parler_desc = compute_parler_description(state) if use_parler else ""

    # Voice blending (solo Kokoro, solo voces EN americanas)
    if backend == TTSBackend.KOKORO:
        voice_key = compute_voice_blend(state, voice_key)

    # --- Pitch shift (semitonos) ---
    # BALANCEADO: valores moderados porque el pitch shift via resample
    # se ACUMULA con speed, voice blend, y pitch contour.
    # Max real: ~1.5st positivo, ~1.2st negativo (a intensidad 1.0)
    # Mas alla de eso suena artificial/chipmunk.
    pitch_map: dict[PrimaryEmotion, float] = {
        PrimaryEmotion.SADNESS: -1.0,
        PrimaryEmotion.HELPLESSNESS: -1.2,
        PrimaryEmotion.DISAPPOINTMENT: -0.6,
        PrimaryEmotion.CONTEMPLATION: -0.3,
        PrimaryEmotion.CONTENTMENT: 0.2,
        PrimaryEmotion.INDIFFERENCE: -0.3,
        PrimaryEmotion.EXCITEMENT: 1.0,
        PrimaryEmotion.JOY: 0.8,
        PrimaryEmotion.ANGER: -0.6,
        PrimaryEmotion.FEAR: 0.8,
        PrimaryEmotion.ANXIETY: 0.6,
        PrimaryEmotion.SURPRISE: 1.2,
        PrimaryEmotion.FRUSTRATION: -0.3,
        PrimaryEmotion.GRATITUDE: 0.3,
        PrimaryEmotion.HOPE: 0.4,
        PrimaryEmotion.RELIEF: -0.2,
        PrimaryEmotion.ALERTNESS: 0.3,
    }
    target_pitch = pitch_map.get(state.primary_emotion, 0.0)
    pitch = round(target_pitch * state.intensity, 2)
    pitch = _clamp(pitch, -1.5, 1.5)

    # --- Volume ---
    # BALANCEADO: rangos mas estrechos para evitar saturacion.
    # Dynamic range expansion en post-proceso ya amplifica la diferencia.
    volume_map: dict[PrimaryEmotion, float] = {
        PrimaryEmotion.SADNESS: 0.78,
        PrimaryEmotion.HELPLESSNESS: 0.75,
        PrimaryEmotion.DISAPPOINTMENT: 0.82,
        PrimaryEmotion.CONTEMPLATION: 0.85,
        PrimaryEmotion.INDIFFERENCE: 0.82,
        PrimaryEmotion.ANGER: 1.15,
        PrimaryEmotion.EXCITEMENT: 1.12,
        PrimaryEmotion.FEAR: 0.88,
        PrimaryEmotion.ANXIETY: 0.9,
        PrimaryEmotion.JOY: 1.08,
        PrimaryEmotion.SURPRISE: 1.1,
        PrimaryEmotion.FRUSTRATION: 1.08,
    }
    target_vol = volume_map.get(state.primary_emotion, 1.0)
    volume = round(1.0 + (target_vol - 1.0) * state.intensity, 2)
    volume = _clamp(volume, 0.7, 1.2)

    # --- Tremolo (vibración nerviosa) ---
    # Solo para fear/anxiety/helplessness con alta intensidad
    tremolo = 0.0
    if state.primary_emotion in (PrimaryEmotion.FEAR, PrimaryEmotion.ANXIETY) and state.intensity > 0.4:
        tremolo = round(0.06 * state.intensity, 3)
    elif state.primary_emotion == PrimaryEmotion.HELPLESSNESS and state.intensity > 0.5:
        tremolo = round(0.04 * state.intensity, 3)
    tremolo = _clamp(tremolo, 0.0, 0.08)

    return VoiceParams(
        voice_key=voice_key,
        speed=speed,
        pitch_semitones=pitch,
        volume=volume,
        tremolo=tremolo,
        stage_direction=direction,
        backend=backend,
        parler_description=parler_desc,
    )


def prepare_text_for_tts(
    text: str,
    stage_direction: str,
    state: EmotionalState | None = None,
    backend: TTSBackend = TTSBackend.KOKORO,
) -> str:
    """Prepara el texto para TTS con mejoras de expresividad.

    Para Kokoro: aplica stress annotations [word](+N) y pausas emocionales.
    Para Parler-TTS: texto limpio sin anotaciones (Parler usa la descripcion
    para controlar emocion, no markup en el texto).
    """
    if backend == TTSBackend.PARLER:
        # Parler no necesita transformaciones — controla emocion via description.
        # Retornar texto limpio.
        return text

    # Kokoro: aplicar mejoras de expresividad en el texto
    if state is not None:
        text = inject_emotional_pauses(text, state)
        text = apply_stress_annotations(text, state)

    return text
