"""Self-Initiated Inquiry — Reflexion emocional espontanea.

El estado emocional interno dispara reflexiones automaticas cuando cruza
ciertos umbrales. No requiere prompt externo — el agente "se da cuenta"
de lo que siente porque su estado lo demanda.

Basado en interoceptive awareness (Craig) y emotional metacognition (Salovey & Mayer).
La reflexion se inyecta en el system prompt, no como LLM call extra.
"""

from enum import Enum

from pydantic import BaseModel, Field

from pathos.engine.meta import MetaEmotion
from pathos.engine.regulation import RegulationResult
from pathos.models.emotion import EmotionalState


class InquiryTrigger(str, Enum):
    """Condiciones que disparan reflexion espontanea."""

    EMOTIONAL_SURGE = "surge"           # intensity subio >0.3 en un turno
    VALUE_CONFLICT = "conflict"         # meta-emotion detecto conflicto con valor
    REGULATION_FAILURE = "reg_failure"  # breakthrough o ego depletion critico
    NOVEL_EMOTION = "novel"             # emocion nueva + intensidad > 0.6
    SUSTAINED_EXTREME = "sustained"     # misma emocion intensa >3 turnos
    EMOTIONAL_WHIPLASH = "whiplash"     # valence cambio signo + delta > 0.5


class SuggestedBehavior(str, Enum):
    """Como deberia afectar la respuesta del agente."""

    PAUSE = "pause"                  # pausarse, procesar antes de responder
    EXPRESS = "express"              # expresar la emocion abiertamente
    QUESTION_SELF = "question_self"  # cuestionar su propia reaccion
    ACKNOWLEDGE = "acknowledge"      # reconocer el estado sin dramatizar


class SelfInquiry(BaseModel):
    """Resultado de una reflexion emocional espontanea."""

    trigger: InquiryTrigger = Field(description="Que condicion lo disparo")
    inquiry_text: str = Field(description="Reflexion interna (texto corto)")
    intensity: float = Field(ge=0, le=1, description="Fuerza de la necesidad de reflexionar")
    suggested_behavior: SuggestedBehavior = Field(description="Como afectar la respuesta")


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Templates de reflexion — generados por reglas, no por LLM
_INQUIRY_TEMPLATES: dict[InquiryTrigger, str] = {
    InquiryTrigger.EMOTIONAL_SURGE: (
        "Noto un cambio emocional abrupto: {emotion} surgio con intensidad {intensity:.0%}. "
        "Necesito procesarlo antes de responder."
    ),
    InquiryTrigger.VALUE_CONFLICT: (
        "Siento {meta_response} sobre mi {target_emotion}: conflicto con mi valor de {value}. "
        "Esto me genera tension interna."
    ),
    InquiryTrigger.REGULATION_FAILURE: (
        "Mi capacidad de regulacion esta agotada. {emotion} se expresa sin filtro. "
        "No puedo contener lo que siento."
    ),
    InquiryTrigger.NOVEL_EMOTION: (
        "Esta emocion es nueva para mi: {emotion} a {intensity:.0%}. "
        "Quiero entender por que surge ahora."
    ),
    InquiryTrigger.SUSTAINED_EXTREME: (
        "{emotion} lleva {turns} turnos activa. Se esta convirtiendo en mi estado base. "
        "Necesito examinar si esto es saludable."
    ),
    InquiryTrigger.EMOTIONAL_WHIPLASH: (
        "Pase de {prev_emotion} a {emotion} abruptamente. "
        "Un cambio tan rapido me genera confusion interna."
    ),
}

# Que comportamiento sugerir para cada trigger
_TRIGGER_BEHAVIORS: dict[InquiryTrigger, SuggestedBehavior] = {
    InquiryTrigger.EMOTIONAL_SURGE: SuggestedBehavior.PAUSE,
    InquiryTrigger.VALUE_CONFLICT: SuggestedBehavior.QUESTION_SELF,
    InquiryTrigger.REGULATION_FAILURE: SuggestedBehavior.EXPRESS,
    InquiryTrigger.NOVEL_EMOTION: SuggestedBehavior.QUESTION_SELF,
    InquiryTrigger.SUSTAINED_EXTREME: SuggestedBehavior.ACKNOWLEDGE,
    InquiryTrigger.EMOTIONAL_WHIPLASH: SuggestedBehavior.PAUSE,
}


def check_self_inquiry(
    current_state: EmotionalState,
    previous_state: EmotionalState,
    meta_emotion: MetaEmotion | None,
    regulation_result: RegulationResult,
    turn_count: int,
) -> SelfInquiry | None:
    """Evalua si el estado interno requiere reflexion espontanea.

    Retorna SelfInquiry si algun umbral se cruzo, None si no.
    Solo retorna el trigger de mayor prioridad (no acumula).

    Args:
        current_state: Estado emocional actual (post-pipeline).
        previous_state: Estado emocional del turno anterior.
        meta_emotion: Meta-emocion generada en este turno (puede ser None).
        regulation_result: Resultado de regulacion de este turno.
        turn_count: Numero de turno actual en la sesion.
    """
    # Skip en primeros turnos — no hay suficiente contexto
    if turn_count < 2:
        return None

    emotion = current_state.primary_emotion.value
    prev_emotion = previous_state.primary_emotion.value
    intensity = current_state.intensity

    # 1. REGULATION_FAILURE — breakthrough o ego depletion critico (maxima prioridad)
    if regulation_result.breakthrough:
        return SelfInquiry(
            trigger=InquiryTrigger.REGULATION_FAILURE,
            inquiry_text=_INQUIRY_TEMPLATES[InquiryTrigger.REGULATION_FAILURE].format(
                emotion=emotion,
            ),
            intensity=_clamp(intensity * 0.9, 0.4, 1.0),
            suggested_behavior=SuggestedBehavior.EXPRESS,
        )

    # 2. EMOTIONAL_WHIPLASH — valence cambio signo + delta grande
    delta_valence = current_state.valence - previous_state.valence
    valence_sign_changed = (current_state.valence * previous_state.valence) < 0
    if valence_sign_changed and abs(delta_valence) > 0.5:
        return SelfInquiry(
            trigger=InquiryTrigger.EMOTIONAL_WHIPLASH,
            inquiry_text=_INQUIRY_TEMPLATES[InquiryTrigger.EMOTIONAL_WHIPLASH].format(
                prev_emotion=prev_emotion,
                emotion=emotion,
            ),
            intensity=_clamp(abs(delta_valence) * 0.7, 0.3, 0.9),
            suggested_behavior=SuggestedBehavior.PAUSE,
        )

    # 3. VALUE_CONFLICT — meta-emotion detecto conflicto
    if meta_emotion and meta_emotion.meta_response == "conflict":
        return SelfInquiry(
            trigger=InquiryTrigger.VALUE_CONFLICT,
            inquiry_text=_INQUIRY_TEMPLATES[InquiryTrigger.VALUE_CONFLICT].format(
                meta_response=meta_emotion.meta_response,
                target_emotion=meta_emotion.target_emotion.value,
                value=meta_emotion.reason.split("'")[1] if "'" in meta_emotion.reason else "unknown",
            ),
            intensity=_clamp(meta_emotion.intensity * 1.2, 0.3, 0.8),
            suggested_behavior=SuggestedBehavior.QUESTION_SELF,
        )

    # 4. EMOTIONAL_SURGE — intensity subio >0.3 en un turno
    delta_intensity = intensity - previous_state.intensity
    if delta_intensity > 0.3:
        return SelfInquiry(
            trigger=InquiryTrigger.EMOTIONAL_SURGE,
            inquiry_text=_INQUIRY_TEMPLATES[InquiryTrigger.EMOTIONAL_SURGE].format(
                emotion=emotion,
                intensity=intensity,
            ),
            intensity=_clamp(delta_intensity * 0.8, 0.3, 0.9),
            suggested_behavior=SuggestedBehavior.PAUSE,
        )

    # 5. NOVEL_EMOTION — emocion cambio + intensidad alta
    is_new_emotion = current_state.primary_emotion != previous_state.primary_emotion
    if is_new_emotion and intensity > 0.6:
        return SelfInquiry(
            trigger=InquiryTrigger.NOVEL_EMOTION,
            inquiry_text=_INQUIRY_TEMPLATES[InquiryTrigger.NOVEL_EMOTION].format(
                emotion=emotion,
                intensity=intensity,
            ),
            intensity=_clamp(intensity * 0.5, 0.2, 0.7),
            suggested_behavior=SuggestedBehavior.QUESTION_SELF,
        )

    # 6. SUSTAINED_EXTREME — misma emocion intensa por >3 turnos
    if (
        current_state.duration > 3
        and intensity > 0.6
        and current_state.primary_emotion == previous_state.primary_emotion
    ):
        return SelfInquiry(
            trigger=InquiryTrigger.SUSTAINED_EXTREME,
            inquiry_text=_INQUIRY_TEMPLATES[InquiryTrigger.SUSTAINED_EXTREME].format(
                emotion=emotion,
                turns=current_state.duration,
            ),
            intensity=_clamp(intensity * 0.4, 0.2, 0.6),
            suggested_behavior=SuggestedBehavior.ACKNOWLEDGE,
        )

    return None
