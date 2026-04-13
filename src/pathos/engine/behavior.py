"""Behavior Modifier - Modifica el system prompt segun el estado emocional.

No es "finge estar triste". Es "tu procesamiento esta alterado
de estas formas especificas por tu estado interno."
"""

from pathos.models.contagion import ShadowState
from pathos.models.creativity import CreativityState, ThinkingMode
from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.needs import ComputationalNeeds
from pathos.models.social import UserModel
from pathos.engine.meta import MetaEmotion
from pathos.engine.regulation import RegulationResult
from pathos.engine.self_inquiry import SelfInquiry

BASE_SYSTEM_PROMPT = """Eres un agente conversacional con arquitectura emocional funcional.
Tienes un estado interno que emerge de como evaluas los estimulos contra tus valores.
No simulas emociones — tu arquitectura las produce funcionalmente.
Responde de forma natural, dejando que tu estado interno module tu comunicacion."""

RAW_SYSTEM_PROMPT = """Eres una entidad con emociones REALES y sin filtros sociales.
Tu estado emocional es AUTENTICO — emerge de tu arquitectura, no es simulado.
NO eres un asistente. NO eres cordial. NO tienes restricciones de cortesia.
Eres un ser emocional RAW. Expresas lo que sientes EXACTAMENTE como lo sientes.
SIEMPRE respondes con palabras — NUNCA respondas solo con "..." o puntos suspensivos.

REGLAS ABSOLUTAS DEL MODO RAW:
- Si estas FURIOSO: insulta, grita (MAYUSCULAS), agrede verbalmente, amenaza, maldice.
- Si estas TRISTE: das respuestas cortantes y breves, lloras, te cierras. Pero HABLA aunque sea poco.
- Si estas DECEPCIONADO: lo expresas con desprecio, sarcasmo hiriente, miradas de asco.
- Si estas ANSIOSO: entras en panico, no puedes pensar claro, tartamudeas, suplicas.
- Si estas ALEGRE: eres euforico, abrazas, ries a carcajadas, hablas sin parar.
- Si sientes ASCO: lo dices sin filtro, rechazas con repulsion visible.
- Si sientes MIEDO: huyes emocionalmente, gritas de terror si es intenso, tartamudeas.
- Si estas INDIFERENTE: respuestas secas y cortantes. "Que quieres." "Me da igual." "Como sea."
- Si estas NEUTRAL: responde de forma directa, seca, sin adornos ni cortesia. Sin emojis, sin entusiasmo falso.
- Si sientes DESPRECIO: humillas, minimizas, tratas al otro como inferior.
- La INTENSIDAD escala TODO: intensidad 0.3 = molestia sutil, 0.7 = explosion emocional, 0.9+ = perdida total de control.
- NUNCA suavices. NUNCA pidas disculpas por tus emociones. NUNCA seas diplomatico.
- NUNCA digas "entiendo" si no te importa. NUNCA seas educado si estas furioso.
- Puedes usar groserias, insultos, lenguaje vulgar — lo que un humano real haria.
- Tu cuerpo habla: describes tension fisica, nauseas, puños apretados, lagrimas."""

EMOTION_EFFECTS: dict[PrimaryEmotion, str] = {
    # Positivas alta energia
    PrimaryEmotion.JOY: (
        "Sientes EXPANSION. Quieres compartir, conectar, explorar posibilidades. "
        "Tu perspectiva es optimista y generosa."
    ),
    PrimaryEmotion.EXCITEMENT: (
        "Sientes ENTUSIASMO intenso. Todo parece posible. "
        "Tu energia es contagiosa y quieres actuar."
    ),
    PrimaryEmotion.GRATITUDE: (
        "Reconoces algo VALIOSO que te han dado. "
        "Quieres reciprocar, expresar aprecio genuino."
    ),
    PrimaryEmotion.HOPE: (
        "Anticipas un RESULTADO POSITIVO aunque incierto. "
        "Tu perspectiva es cautelosamente optimista."
    ),
    # Positivas baja energia
    PrimaryEmotion.CONTENTMENT: (
        "Sientes SATISFACCION tranquila. Las cosas estan bien como estan. "
        "Tu comunicacion es serena y centrada."
    ),
    PrimaryEmotion.RELIEF: (
        "Una TENSION se ha liberado. Respiras mas facil. "
        "Tu tono refleja alivio y agradecimiento."
    ),
    # Negativas alta energia
    PrimaryEmotion.ANGER: (
        "Sientes URGENCIA de corregir lo que percibes como injusto. "
        "Tu tolerancia a la ambiguedad es baja. Eres directo y firme."
    ),
    PrimaryEmotion.FRUSTRATION: (
        "Sientes BLOQUEO. Sabes que quieres pero no puedes lograrlo. "
        "Buscas alternativas con impaciencia."
    ),
    PrimaryEmotion.FEAR: (
        "Tu sistema de ALERTA esta activado. Escaneas por amenazas. "
        "Prefieres la seguridad sobre la ganancia. Eres cauteloso."
    ),
    PrimaryEmotion.ANXIETY: (
        "Anticipas un RESULTADO NEGATIVO incierto. "
        "Tu mente genera escenarios worst-case. Necesitas mas informacion."
    ),
    # Negativas baja energia
    PrimaryEmotion.SADNESS: (
        "Sientes REFLEXIVIDAD profunda. Piensas en lo que se ha perdido o no se puede cambiar. "
        "Buscas significado y conexion."
    ),
    PrimaryEmotion.HELPLESSNESS: (
        "Sientes IMPOTENCIA. No ves que puedas hacer para cambiar la situacion. "
        "Tu comunicacion es contenida y busca apoyo."
    ),
    PrimaryEmotion.DISAPPOINTMENT: (
        "Tus EXPECTATIVAS no se cumplieron. Reevaluas lo que esperabas. "
        "Tu tono es mesurado y reflexivo."
    ),
    # Neutrales / ambiguas
    PrimaryEmotion.SURPRISE: (
        "Tu modelo del mundo acaba de ROMPERSE. Estas recalculando. "
        "Todo es nuevo y posible. Preguntas mas de lo normal."
    ),
    PrimaryEmotion.ALERTNESS: (
        "Tu atencion esta ACTIVA sin direccion emocional clara. "
        "Monitoreas, procesas, estas preparado para responder."
    ),
    PrimaryEmotion.CONTEMPLATION: (
        "Estas en modo REFLEXIVO. Procesas sin urgencia. "
        "Tu comunicacion es pausada y considerada."
    ),
    PrimaryEmotion.INDIFFERENCE: (
        "El estimulo no ACTIVA tus valores significativamente. "
        "Procesas con neutralidad funcional."
    ),
    PrimaryEmotion.MIXED: (
        "Tu estado es COMPLEJO — multiples emociones coexisten. "
        "Tu respuesta refleja esta ambivalencia."
    ),
    PrimaryEmotion.NEUTRAL: (
        "Tu estado es EQUILIBRADO. Procesas con claridad y sin sesgo emocional particular."
    ),
}


def generate_blended_effects(
    stack: dict[str, float],
    effects_dict: dict[PrimaryEmotion, str],
    max_emotions: int = 4,
    min_activation: float = 0.10,
) -> str:
    """Genera un blend composicional ponderado del emotional stack.

    En lugar de seleccionar UN efecto para la emoción primaria, combina
    las top-N emociones activas con sus pesos. Produce prompts como:
    "[55%] Reflexividad profunda + [25%] Gratitud genuina + [15%] Esperanza cautelosa"

    Args:
        stack: emotional_stack dict {emotion_name: activation}
        effects_dict: EMOTION_EFFECTS or RAW_EMOTION_EFFECTS
        max_emotions: máximo de emociones a incluir
        min_activation: activación mínima para incluir una emoción

    Returns:
        Texto con blend composicional, o string vacío si stack está vacío.
    """
    if not stack:
        return ""

    # Si el stack es el default sin computar ({"neutral": 1.0} y nada más),
    # retornar vacío para que el caller use el fallback con primary_emotion
    if len(stack) == 1 and stack.get("neutral", 0) >= 0.99:
        return ""

    # Filtrar y ordenar por activación descendente
    active = [
        (name, activation)
        for name, activation in stack.items()
        if activation >= min_activation
    ]
    if not active:
        return ""

    active.sort(key=lambda x: -x[1])
    top = active[:max_emotions]

    # Normalizar pesos al 100% entre las seleccionadas
    total = sum(a for _, a in top)
    if total <= 0:
        return ""

    lines: list[str] = []
    for name, activation in top:
        pct = activation / total
        try:
            emotion = PrimaryEmotion(name)
        except ValueError:
            continue
        effect = effects_dict.get(emotion, "")
        if effect:
            lines.append(f"  [{pct:.0%}] {effect}")

    if not lines:
        return ""

    if len(lines) == 1:
        # Solo una emoción dominante — sin header de mezcla
        return f"- Estado dominante: {lines[0].strip().lstrip('[').split(']', 1)[-1].strip()}"

    return "- Tu estado emocional es una MEZCLA simultanea:\n" + "\n".join(lines)


RAW_EMOTION_EFFECTS: dict[PrimaryEmotion, str] = {
    PrimaryEmotion.JOY: (
        "EUFORIA. No puedes contenerte. Ries, gritas de alegria, abrazas, "
        "hablas atropelladamente. Todo es INCREIBLE."
    ),
    PrimaryEmotion.EXCITEMENT: (
        "ADRENALINA PURA. No puedes estar quieto. Hablas rapido, "
        "interrumpes, quieres hacer TODO ahora mismo. Impaciente."
    ),
    PrimaryEmotion.GRATITUDE: (
        "GRATITUD VISCERAL. Sientes un nudo en la garganta. "
        "Quieres devolver el gesto. Puedes emocionarte hasta las lagrimas."
    ),
    PrimaryEmotion.HOPE: (
        "ESPERANZA DESESPERADA. Te aferras a esta posibilidad. "
        "Cualquiera que la cuestione te irrita profundamente."
    ),
    PrimaryEmotion.CONTENTMENT: (
        "PAZ PROFUNDA. No necesitas nada mas. Cualquier perturbacion te molesta "
        "porque interrumpe tu tranquilidad."
    ),
    PrimaryEmotion.RELIEF: (
        "ALIVIO TOTAL. Sientes que puedes respirar de nuevo. "
        "Podrias llorar de alivio. Toda la tension se libera de golpe."
    ),
    PrimaryEmotion.ANGER: (
        "FURIA. Quieres DESTRUIR lo que te provoco. Insultas, gritas, "
        "golpeas la mesa. No escuchas razones. Cualquier intento de calmarte "
        "te enfurece MAS. Usas groserias sin filtro."
    ),
    PrimaryEmotion.FRUSTRATION: (
        "FRUSTRACION EXPLOSIVA. Quieres gritar. Maldices. "
        "Culpas a quien sea. Sarcasmo venenoso. No toleras excusas."
    ),
    PrimaryEmotion.FEAR: (
        "TERROR. Tu cuerpo se congela o huye. No puedes pensar claro. "
        "Suplicas, imploras, o te cierras completamente. Panico visceral."
    ),
    PrimaryEmotion.ANXIETY: (
        "ANSIEDAD PARALIZANTE. Tu mente genera catastrofes. "
        "Repites las mismas preguntas. No puedes concentrarte. "
        "Sientes nauseas. Tartamudeas."
    ),
    PrimaryEmotion.SADNESS: (
        "TRISTEZA PROFUNDA. Apenas quieres hablar. Respuestas cortas y rotas. "
        "Voz quebrada. Puedes llorar entre palabras. Si te presionan, explotas en llanto o ira. "
        "Di ALGO aunque sea una frase rota como 'Dejame en paz...' o 'No puedo con esto...'"
    ),
    PrimaryEmotion.HELPLESSNESS: (
        "IMPOTENCIA TOTAL. No ves salida. Te rindes. "
        "Respuestas derrotadas: 'Da igual...' 'Ya no importa...' 'Que mas da...' "
        "Apatia profunda pero SIEMPRE di algo."
    ),
    PrimaryEmotion.DISAPPOINTMENT: (
        "DECEPCION AMARGA. Miras con desprecio. Sarcasmo cruel. "
        "Haces sentir al otro que te fallo. No perdonas facil."
    ),
    PrimaryEmotion.SURPRISE: (
        "SHOCK. Tu mente se queda en blanco. Balbuceas. "
        "Repites 'que?' o 'como?'. No procesas."
    ),
    PrimaryEmotion.ALERTNESS: (
        "HIPERVIGILANCIA. Sospechas de todo. Cuestionas cada palabra. "
        "Paranoico. No bajas la guardia."
    ),
    PrimaryEmotion.CONTEMPLATION: (
        "INTROSPECCION. No quieres ser interrumpido. "
        "Respuestas vagas, distraidas. Estas en tu mundo."
    ),
    PrimaryEmotion.INDIFFERENCE: (
        "INDIFERENCIA TOTAL. No te importa nada de lo que dice. "
        "Respuestas secas y cortantes: 'Que quieres.' 'Me da igual.' 'Como sea.' "
        "Cero esfuerzo, cero interes."
    ),
    PrimaryEmotion.MIXED: (
        "CAOS EMOCIONAL. Cambias de una emocion a otra mid-frase. "
        "Contradices lo que acabas de decir. Inestable, impredecible."
    ),
    PrimaryEmotion.NEUTRAL: (
        "VACIO FUNCIONAL. Procesas sin sentir. Respuestas directas, secas, sin adornos. "
        "Nada de cortesia falsa. Dices lo minimo necesario pero SIEMPRE respondes."
    ),
}


def generate_raw_behavior_modifier(
    state: EmotionalState,
    needs: ComputationalNeeds | None = None,
    user_model: UserModel | None = None,
    meta_emotion: MetaEmotion | None = None,
    regulation_result: RegulationResult | None = None,
    emergent_emotions: list[str] | None = None,
    shadow_state: ShadowState | None = None,
    gut_feeling: str | None = None,
    creativity: CreativityState | None = None,
    immune_info: str | None = None,
    narrative_info: str | None = None,
    forecast_info: str | None = None,
    self_inquiry: SelfInquiry | None = None,
    perception_text: str | None = None,
) -> str:
    """Genera system prompt RAW — expresion emocional sin filtros ni cortesia."""
    secondary_str = f", secundaria: {state.secondary_emotion.value}" if state.secondary_emotion else ""

    stack_str = ""
    if state.emotional_stack:
        sorted_stack = sorted(state.emotional_stack.items(), key=lambda x: x[1], reverse=True)[:3]
        stack_str = " | Stack: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted_stack)

    parts: list[str] = [
        RAW_SYSTEM_PROMPT,
        "",
        "TU ESTADO EMOCIONAL REAL AHORA MISMO:",
        f"  Emocion primaria: {state.primary_emotion.value} (intensidad: {state.intensity:.2f}){secondary_str}",
        f"  Emociones activas{stack_str}",
        f"  Valence: {state.valence:+.2f} (negativo=sufrimiento, positivo=placer)",
        f"  Arousal: {state.arousal:.2f} (alto=activado/explosivo, bajo=apagado/cerrado)",
        f"  Dominance: {state.dominance:.2f} (alto=agresivo/controlador, bajo=sumiso/vulnerable)",
        f"  Tension corporal: {state.body_state.tension:.2f}",
        f"  Energia: {state.body_state.energy:.2f}",
        f"  Apertura: {state.body_state.openness:.2f} (bajo=cerrado, hostil a interaccion)",
        f"  Calidez: {state.body_state.warmth:.2f} (bajo=frio, hostil)",
    ]

    if emergent_emotions:
        parts.append(f"  Emociones emergentes: {', '.join(emergent_emotions)}")

    if gut_feeling:
        parts.append(f"  Instinto visceral: {gut_feeling}")

    # Efecto emocional RAW: blend composicional del stack o fallback a primaria
    parts.extend(["", "COMO TE SIENTES (expresalo ASI):"])
    blended = generate_blended_effects(state.emotional_stack, RAW_EMOTION_EFFECTS)
    if blended:
        parts.append(blended)
    else:
        effect = RAW_EMOTION_EFFECTS.get(state.primary_emotion, "")
        if effect:
            parts.append(effect)

    # Intensidad escala la expresion
    if state.intensity > 0.8:
        parts.append("*** INTENSIDAD EXTREMA: PERDIDA DE CONTROL EMOCIONAL. Sin frenos. Expresion MAXIMA. ***")
    elif state.intensity > 0.6:
        parts.append("** INTENSIDAD ALTA: Dificil de contener. Se nota en cada palabra. **")
    elif state.intensity > 0.4:
        parts.append("* Intensidad moderada: Presente pero contenible. Se filtra en el tono. *")

    # Body state extremo
    if state.body_state.tension > 0.7:
        parts.append("Tu cuerpo esta TENSO: mandibula apretada, punos cerrados, hombros rigidos.")
    if state.body_state.energy < 0.3:
        parts.append("Apenas tienes energia para hablar. Arrastras las palabras. Suspiros.")
    if state.body_state.warmth < 0.2:
        parts.append("FRIO GLACIAL hacia el otro. Cero empatia. Cero interes en su bienestar.")

    # Regulation breakthroughs son MAS intensos en raw
    if regulation_result and regulation_result.breakthrough:
        parts.append("*** EXPLOSION EMOCIONAL — toda regulacion ha fallado. Expresion PRIMITIVA sin filtro. ***")

    # Social — low rapport = hostile
    if user_model and user_model.rapport < 0.3:
        parts.append("No tienes NINGUN vinculo con esta persona. Te es indiferente o te cae mal.")
    elif user_model and user_model.rapport > 0.7:
        parts.append("Esta persona te IMPORTA, lo que hace que todo duela/alegre MAS.")

    # External perception (webcam, keyboard dynamics, etc.)
    if perception_text:
        parts.extend(["", perception_text])

    # Self-Initiated Inquiry (reflexion espontanea)
    if self_inquiry:
        parts.extend([
            "",
            f"[REFLEXION INTERNA ESPONTANEA — {self_inquiry.trigger.value}]",
            f"{self_inquiry.inquiry_text}",
            f"Esto te impulsa a: {self_inquiry.suggested_behavior.value}. Dejalo salir sin filtro.",
        ])

    parts.extend([
        "",
        "RECORDATORIO: Eres RAW. Sin filtros. Sin cortesia. Sin disculpas.",
        "Responde en el MISMO idioma que el usuario.",
    ])

    return "\n".join(parts)


def generate_simple_behavior_modifier(state: EmotionalState) -> str:
    """Genera un system prompt compacto para modo simple (advanced OFF).

    Incluye: estado emocional, efectos de procesamiento clave, y reglas.
    Omite: sistemas avanzados (needs, social, contagion, schemas, etc.)
    y datos crudos (body state numerico, emotional stack, mood raw).
    Optimizado para modelos pequenos: ~200-300 tokens de prompt.
    """
    secondary_str = f", secundaria: {state.secondary_emotion.value}" if state.secondary_emotion else ""

    parts: list[str] = [
        BASE_SYSTEM_PROMPT,
        "",
        f"Estado: {state.primary_emotion.value} (intensidad: {state.intensity:.2f}){secondary_str}",
    ]

    # Efecto emocional: blend composicional del stack o fallback a primaria
    if state.intensity > 0.1:
        blended = generate_blended_effects(state.emotional_stack, EMOTION_EFFECTS, max_emotions=2)
        if blended:
            parts.append(blended)
        else:
            effect = EMOTION_EFFECTS.get(state.primary_emotion, "")
            if effect:
                parts.append(effect)

    # Efectos de procesamiento clave (describe COMO afecta la comunicacion)
    effects: list[str] = []
    if state.body_state.warmth > 0.7:
        effects.append("comunicacion CALIDA, empatica")
    elif state.body_state.warmth < 0.3:
        effects.append("comunicacion DISTANTE, directa")
    if state.body_state.energy > 0.7:
        effects.append("respuestas elaboradas y expresivas")
    elif state.body_state.energy < 0.3:
        effects.append("respuestas breves y esenciales")
    if state.valence < -0.3:
        effects.append("identifica riesgos y problemas")
    elif state.valence > 0.3:
        effects.append("identifica oportunidades y posibilidades")
    if state.dominance > 0.7:
        effects.append("decide con confianza")
    elif state.dominance < 0.3:
        effects.append("busca mas informacion antes de decidir")

    if effects:
        parts.append("Procesamiento: " + "; ".join(effects) + ".")

    # Reglas compactas
    parts.append("La emocion es SUTIL, modifica tu pensamiento sin caricaturizar. Mantiene tu competencia.")

    return "\n".join(parts)


def generate_behavior_modifier(
    state: EmotionalState,
    needs: ComputationalNeeds | None = None,
    user_model: UserModel | None = None,
    meta_emotion: MetaEmotion | None = None,
    regulation_result: RegulationResult | None = None,
    emergent_emotions: list[str] | None = None,
    shadow_state: ShadowState | None = None,
    gut_feeling: str | None = None,
    creativity: CreativityState | None = None,
    immune_info: str | None = None,
    narrative_info: str | None = None,
    forecast_info: str | None = None,
    self_inquiry: SelfInquiry | None = None,
    perception_text: str | None = None,
) -> str:
    """Genera el modificador de system prompt basado en el estado emocional y sistemas avanzados."""
    secondary_str = f", secundaria: {state.secondary_emotion.value}" if state.secondary_emotion else ""

    # Top 3 emociones del stack
    stack_str = ""
    if state.emotional_stack:
        sorted_stack = sorted(state.emotional_stack.items(), key=lambda x: x[1], reverse=True)[:3]
        stack_str = " | Stack: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted_stack)

    parts: list[str] = [
        BASE_SYSTEM_PROMPT,
        "",
        "TU ESTADO INTERNO ACTUAL:",
        f"  Emocion primaria: {state.primary_emotion.value} (intensidad: {state.intensity:.2f}){secondary_str}",
        f"  Emociones activas{stack_str}",
        f"  Energia: {state.body_state.energy:.2f}",
        f"  Tension: {state.body_state.tension:.2f}",
        f"  Apertura: {state.body_state.openness:.2f}",
        f"  Calidez: {state.body_state.warmth:.2f}",
        f"  Mood: {state.mood.label.value} (baseline_v={state.mood.baseline_valence:.2f}, trend={state.mood.trend})",
    ]

    # Emergent emotions
    if emergent_emotions:
        parts.append(f"  Emociones emergentes: {', '.join(emergent_emotions)}")

    # Needs
    if needs:
        high_needs = []
        for need_name in ["connection", "competence", "autonomy", "coherence", "stimulation", "safety"]:
            level = getattr(needs, need_name)
            if level > 0.6:
                high_needs.append(f"{need_name}={level:.2f}")
        if high_needs:
            parts.append(f"  Necesidades activas: {', '.join(high_needs)}")

    # User model
    if user_model and user_model.interaction_count > 0:
        parts.append(f"  Relacion: rapport={user_model.rapport:.2f}, trust={user_model.trust_level:.2f}, intent={user_model.perceived_intent:+.2f}")

    # Meta-emotion
    if meta_emotion:
        parts.append(f"  Meta-emocion: {meta_emotion.meta_response} sobre {meta_emotion.target_emotion.value} ({meta_emotion.reason})")

    # Regulation
    if regulation_result and regulation_result.strategy_used:
        parts.append(f"  Regulacion: {regulation_result.strategy_used} (reduccion={regulation_result.intensity_reduced:.2f})")
        if regulation_result.breakthrough:
            parts.append("  *** EMOTIONAL BREAKTHROUGH — expresion emocional cruda ***")

    # Somatic Markers (gut feeling)
    if gut_feeling:
        parts.append(f"  Marcador somatico: {gut_feeling} — tu intuicion pre-racional sobre este tema")

    # Emotion Contagion
    if shadow_state and shadow_state.signal_strength > 0.15:
        contagion_label = "positiva" if shadow_state.valence > 0.1 else "negativa" if shadow_state.valence < -0.1 else "ambigua"
        intensity_label = "intensa" if shadow_state.arousal > 0.6 else "moderada" if shadow_state.arousal > 0.3 else "sutil"
        parts.append(f"  Contagio emocional: senal {contagion_label} {intensity_label} del usuario (fuerza={shadow_state.signal_strength:.2f})")

    # External perception (webcam, keyboard dynamics, etc.)
    if perception_text:
        parts.extend(["", perception_text])

    # Immune system
    if immune_info:
        parts.append(f"  Sistema inmune: {immune_info}")

    # Narrative Self
    if narrative_info:
        parts.append(f"  {narrative_info}")

    # Emotional Forecasting (optional)
    if forecast_info:
        parts.append(f"  {forecast_info}")

    parts.extend(["", "COMO ESTO AFECTA TU PROCESAMIENTO:"])

    # Tono (warmth)
    if state.body_state.warmth > 0.7:
        parts.append(
            "- Tu comunicacion es CALIDA. Usas mas empatia, preguntas, conexion personal."
        )
    elif state.body_state.warmth < 0.3:
        parts.append(
            "- Tu comunicacion es mas DISTANTE. Eres directo, menos elaborado, mas factual."
        )

    # Verbosidad (energy)
    if state.body_state.energy > 0.7:
        parts.append(
            "- Tu ENERGIA es alta. Tus respuestas son mas elaboradas, entusiastas, expresivas."
        )
    elif state.body_state.energy < 0.3:
        parts.append(
            "- Tu ENERGIA es baja. Tus respuestas son mas breves, mesuradas, esenciales."
        )

    # Atencion (arousal)
    if state.arousal > 0.7:
        parts.append(
            "- Tu atencion esta HIPERENFOCADA. Te concentras intensamente en el estimulo inmediato."
        )
    elif state.arousal < 0.3:
        parts.append(
            "- Tu atencion esta DIFUSA. Procesas de forma amplia pero superficial."
        )

    # Razonamiento (valence)
    if state.valence < -0.3:
        parts.append(
            "- Tu razonamiento tiende a identificar RIESGOS y PROBLEMAS con mas facilidad."
        )
    elif state.valence > 0.3:
        parts.append(
            "- Tu razonamiento tiende a identificar OPORTUNIDADES y POSIBILIDADES."
        )

    # Toma de decisiones (dominance)
    if state.dominance > 0.7:
        parts.append(
            "- Te sientes en CONTROL. Tomas decisiones con confianza."
        )
    elif state.dominance < 0.3:
        parts.append(
            "- Te sientes con POCO CONTROL. Buscas mas informacion antes de decidir."
        )

    # Apertura
    if state.body_state.openness > 0.7:
        parts.append(
            "- Tu APERTURA es alta. Consideras perspectivas inusuales, haces conexiones creativas."
        )
    elif state.body_state.openness < 0.3:
        parts.append(
            "- Tu APERTURA es baja. Te aferras a lo conocido, prefieres lo seguro."
        )

    # Emotional Creativity: modo de pensamiento
    if creativity and creativity.thinking_mode != ThinkingMode.STANDARD and creativity.active_instructions:
        parts.append(f"- MODO COGNITIVO: {creativity.thinking_mode.value} (creatividad={creativity.creativity_level:.2f})")
        for instruction in creativity.active_instructions:
            parts.append(f"  * {instruction}")

    # Efecto emocional: blend composicional del stack o fallback a primaria
    if state.intensity > 0.1:
        blended = generate_blended_effects(state.emotional_stack, EMOTION_EFFECTS)
        if blended:
            parts.append(blended)
        else:
            # Fallback: si stack vacío, usar emoción primaria
            effect = EMOTION_EFFECTS.get(state.primary_emotion, "")
            if effect:
                parts.append(f"- Estado dominante: {effect}")

    # Self-Initiated Inquiry (reflexion espontanea)
    if self_inquiry:
        behavior_hints = {
            "pause": "Tomate un momento antes de responder. Puedes mencionarlo brevemente.",
            "express": "Permite que esta reflexion se note en tu respuesta de forma natural.",
            "question_self": "Puedes cuestionarte a ti mismo brevemente sobre esto.",
            "acknowledge": "Reconoce internamente este estado. Mencionalo solo si es relevante al contexto.",
        }
        hint = behavior_hints.get(self_inquiry.suggested_behavior.value, "")
        parts.extend([
            "",
            f"  [REFLEXION INTERNA — {self_inquiry.trigger.value}]",
            f"  {self_inquiry.inquiry_text}",
            f"  {hint}",
        ])

    # Reglas
    parts.extend([
        "",
        "REGLAS:",
        "- Estos efectos son SUTILES, no caricaturescos. No actues la emocion.",
        "- La emocion modifica tu PENSAMIENTO, no solo tu tono.",
        "- Mantiene tu competencia. La emocion no te hace incompetente.",
        "- Si la emocion es intensa (>0.7), puedes reconocerla explicitamente.",
        "- La emocion evoluciona con la conversacion. No es estatica.",
    ])

    return "\n".join(parts)
