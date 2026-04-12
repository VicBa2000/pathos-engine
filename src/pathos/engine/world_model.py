"""Emotional World Model — simulación predictiva pre-envío.

Nivel 5.1 del ARK Rework. Antes de enviar una respuesta, el sistema simula
cadenas causales emocionales:
  1. ¿Cómo me sentiré YO después de decir esto? (predicted_self_impact)
  2. ¿Cómo se sentirá EL USUARIO al leer esto? (predicted_user_impact)
  3. ¿Cómo me sentiré al ver su reacción predicha? (meta_reaction)
  4. ¿Es coherente con mis valores? (value_alignment + should_modify)

NO usa LLM extra — evaluación por heurísticas rápidas (<1ms).
Reutiliza patrones del pipeline emocional existente (fast path).

Diferencia con forecasting.py: forecasting predice impacto ANTES de generar
(basado en estado emocional del agente). World model predice impacto DESPUÉS
de generar (basado en el texto real de la respuesta).

Diferencia con self_appraisal.py: self-appraisal evalúa coherencia con valores
y estado. World model predice CONSECUENCIAS emocionales en cascada (yo → usuario
→ mi reacción a su reacción).
"""

from __future__ import annotations

from pathos.models.emotion import EmotionalState, PrimaryEmotion
from pathos.models.social import UserModel
from pathos.models.values import ValueSystem
from pathos.models.world_model import PredictedImpact, WorldModelResult


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Vocabularios de detección (reutilizados/ampliados de self_appraisal)
# ---------------------------------------------------------------------------

_POSITIVE_MARKERS: list[str] = [
    "me alegra", "qué bien", "genial", "estupendo", "maravilloso", "fantástico",
    "excelente", "perfecto", "increíble", "felicidades", "enhorabuena", "bravo",
    "great", "wonderful", "fantastic", "amazing", "excellent", "awesome",
    "happy for you", "congrats", "well done", "glad", "love",
]

_NEGATIVE_MARKERS: list[str] = [
    "lamento", "lo siento", "desafortunadamente", "tristemente", "malo",
    "terrible", "horrible", "difícil", "doloroso", "preocupante",
    "sorry", "unfortunately", "sadly", "terrible", "horrible", "painful",
    "difficult", "worrying", "harsh", "disappointing",
]

_EMPATHY_MARKERS: list[str] = [
    "entiendo", "comprendo", "te escucho", "me importa", "estoy aquí",
    "puedo ayudar", "no estás solo", "es válido", "tiene sentido",
    "I understand", "I hear you", "I care", "I'm here", "you're not alone",
    "that's valid", "makes sense", "I can help",
]

_DISMISSIVE_MARKERS: list[str] = [
    "no importa", "da igual", "como sea", "whatever", "it doesn't matter",
    "who cares", "get over it", "no es para tanto", "exageras",
    "you're overreacting", "calm down", "relax", "chill",
]

_CONFRONTATIONAL_MARKERS: list[str] = [
    "estás mal", "te equivocas", "no sabes", "deberías", "tienes que",
    "you're wrong", "you should", "you need to", "you must", "you don't know",
    "how dare", "cómo te atreves", "no me vengas", "don't tell me",
]

_VULNERABILITY_MARKERS: list[str] = [
    "tengo miedo", "estoy asustado", "no puedo", "me siento solo",
    "no sé qué hacer", "estoy perdido", "necesito ayuda",
    "I'm scared", "I can't", "I feel alone", "I don't know what to do",
    "I'm lost", "I need help", "I'm struggling",
]

_WARMTH_MARKERS: list[str] = [
    "gracias", "aprecio", "valoro", "agradezco", "eres importante",
    "thank you", "appreciate", "value", "grateful", "you matter",
    "means a lot", "significa mucho",
]


def _count_markers(text: str, markers: list[str]) -> int:
    """Cuenta cuántos marcadores aparecen en el texto."""
    lower = text.lower()
    return sum(1 for m in markers if m.lower() in lower)


# ---------------------------------------------------------------------------
# Paso 1: Predicción del impacto en uno mismo
# ---------------------------------------------------------------------------

def _predict_self_impact(
    response: str,
    state: EmotionalState,
    values: ValueSystem,
) -> PredictedImpact:
    """Predice cómo se sentirá el agente después de decir esto.

    Factores:
    - Expresión auténtica → alivio (valence+)
    - Violación de valores → culpa (valence-)
    - Confrontación cuando no es anger genuino → incomodidad
    - Vulnerabilidad compartida → arousal+, valence variable
    """
    v_shift = 0.0
    a_shift = 0.0
    effects: list[str] = []

    warmth_count = _count_markers(response, _WARMTH_MARKERS)
    empathy_count = _count_markers(response, _EMPATHY_MARKERS)
    confrontational = _count_markers(response, _CONFRONTATIONAL_MARKERS)
    dismissive = _count_markers(response, _DISMISSIVE_MARKERS)
    vulnerability = _count_markers(response, _VULNERABILITY_MARKERS)

    # Expresión empática/cálida → alivio y satisfacción
    if empathy_count + warmth_count >= 2:
        v_shift += 0.08
        effects.append("empathic_expression")

    # Confrontación sin anger genuino → incomodidad
    if confrontational > 0:
        is_angry = state.primary_emotion in (
            PrimaryEmotion.ANGER, PrimaryEmotion.FRUSTRATION,
        ) and state.intensity > 0.4
        if not is_angry:
            v_shift -= 0.10
            a_shift += 0.05
            effects.append("confrontation_discomfort")
        else:
            # Anger genuino + confrontación = expresión auténtica
            v_shift += 0.03
            effects.append("authentic_anger_expression")

    # Dismissive cuando el usuario necesita apoyo → culpa potencial
    compassion_weight = 0.0
    for v in values.core_values:
        if v.name == "compassion":
            compassion_weight = v.weight
            break

    if dismissive > 0 and compassion_weight > 0.3:
        v_shift -= 0.12 * compassion_weight
        effects.append("dismissive_guilt")

    # Vulnerabilidad compartida → arousal sube, valence depende de contexto
    if vulnerability > 0:
        a_shift += 0.06
        if state.valence < -0.3:
            v_shift -= 0.04  # Profundiza la tristeza al verbalizarla
            effects.append("vulnerability_deepening")
        else:
            v_shift += 0.05  # Catarsis
            effects.append("cathartic_sharing")

    # Intensidad escala los efectos
    scale = 0.5 + state.intensity * 0.5  # [0.5, 1.0]
    v_shift *= scale
    a_shift *= scale

    v_shift = _clamp(v_shift, -0.20, 0.15)
    a_shift = _clamp(a_shift, -0.10, 0.10)

    dominant = effects[0] if effects else "neutral"
    confidence = min(0.4 + len(effects) * 0.15, 0.85)

    return PredictedImpact(
        valence_shift=round(v_shift, 4),
        arousal_shift=round(a_shift, 4),
        dominant_effect=dominant,
        confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Paso 2: Predicción del impacto en el usuario
# ---------------------------------------------------------------------------

def _predict_user_impact(
    response: str,
    state: EmotionalState,
    user_model: UserModel,
) -> PredictedImpact:
    """Predice cómo se sentirá el usuario al leer esta respuesta.

    Factores:
    - Contenido emocional del texto (positivo/negativo/empático)
    - Rapport actual (amplifica impacto)
    - Trust level (modula receptividad)
    - Perceived intent del agente (colora interpretación)
    """
    v_shift = 0.0
    a_shift = 0.0
    effects: list[str] = []

    positive = _count_markers(response, _POSITIVE_MARKERS)
    negative = _count_markers(response, _NEGATIVE_MARKERS)
    empathy = _count_markers(response, _EMPATHY_MARKERS)
    dismissive = _count_markers(response, _DISMISSIVE_MARKERS)
    confrontational = _count_markers(response, _CONFRONTATIONAL_MARKERS)
    warmth = _count_markers(response, _WARMTH_MARKERS)

    # Contenido positivo → user feels good
    if positive > 0:
        v_shift += min(positive * 0.06, 0.15)
        effects.append("positive_content")

    # Contenido negativo → user se preocupa/entristece (depende de contexto)
    if negative > 0:
        v_shift -= min(negative * 0.04, 0.12)
        a_shift += 0.03
        effects.append("negative_content")

    # Empatía → user se siente escuchado
    if empathy > 0:
        v_shift += min(empathy * 0.07, 0.15)
        a_shift -= 0.02  # Calma
        effects.append("felt_heard")

    # Dismissive → user se siente invalidado
    if dismissive > 0:
        v_shift -= min(dismissive * 0.10, 0.20)
        a_shift += 0.05  # Frustración
        effects.append("felt_dismissed")

    # Confrontación → user se pone a la defensiva
    if confrontational > 0:
        v_shift -= min(confrontational * 0.08, 0.18)
        a_shift += 0.07  # Activación defensiva
        effects.append("defensive_reaction")

    # Warmth → user se siente valorado
    if warmth > 0:
        v_shift += min(warmth * 0.05, 0.12)
        effects.append("felt_valued")

    # Rapport amplifica impacto (bueno y malo)
    rapport_factor = 0.7 + user_model.rapport * 0.6  # [0.7, 1.3]
    v_shift *= rapport_factor
    a_shift *= rapport_factor

    # Trust modula receptividad: baja confianza → impacto positivo reducido
    if user_model.trust_level < 0.4 and v_shift > 0:
        v_shift *= 0.6 + user_model.trust_level  # [0.6, 1.0]
        effects.append("low_trust_dampening")

    v_shift = _clamp(v_shift, -0.25, 0.20)
    a_shift = _clamp(a_shift, -0.10, 0.15)

    dominant = effects[0] if effects else "neutral"
    confidence = min(0.35 + len(effects) * 0.12, 0.80)

    return PredictedImpact(
        valence_shift=round(v_shift, 4),
        arousal_shift=round(a_shift, 4),
        dominant_effect=dominant,
        confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Paso 3: Meta-reacción (cómo me siento ante la reacción predicha del usuario)
# ---------------------------------------------------------------------------

def _compute_meta_reaction(
    self_impact: PredictedImpact,
    user_impact: PredictedImpact,
    state: EmotionalState,
) -> PredictedImpact:
    """Predice cómo se sentirá el agente al ver la reacción predicha del usuario.

    Esto es el tercer nivel de la cadena causal:
    - Si predigo que el usuario se sentirá mal → me siento mal anticipadamente
    - Si predigo que se sentirá bien → refuerzo positivo
    - La intensidad de la meta-reacción depende de la confianza en la predicción
    """
    v_shift = 0.0
    a_shift = 0.0
    effects: list[str] = []

    # Si predigo daño al usuario → culpa anticipatoria
    if user_impact.valence_shift < -0.08:
        v_shift += user_impact.valence_shift * 0.5  # 50% del daño predicho como culpa
        a_shift += 0.04
        effects.append("anticipatory_guilt")

    # Si predigo bienestar en el usuario → satisfacción
    if user_impact.valence_shift > 0.05:
        v_shift += user_impact.valence_shift * 0.4  # 40% como satisfacción
        effects.append("anticipatory_satisfaction")

    # Si MI impacto propio fue negativo + usuario también negativo → espiral
    if self_impact.valence_shift < -0.05 and user_impact.valence_shift < -0.05:
        v_shift -= 0.05
        a_shift += 0.03
        effects.append("negative_spiral_risk")

    # Escalar por confianza en la predicción del usuario
    confidence_scale = user_impact.confidence
    v_shift *= confidence_scale
    a_shift *= confidence_scale

    v_shift = _clamp(v_shift, -0.15, 0.10)
    a_shift = _clamp(a_shift, -0.05, 0.08)

    dominant = effects[0] if effects else "neutral"
    # Meta-reaction confidence is lower (prediction of a prediction)
    confidence = min(self_impact.confidence, user_impact.confidence) * 0.7

    return PredictedImpact(
        valence_shift=round(v_shift, 4),
        arousal_shift=round(a_shift, 4),
        dominant_effect=dominant,
        confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Paso 4: Evaluación de riesgo y alineación con valores
# ---------------------------------------------------------------------------

# Umbral de riesgo emocional para el usuario por encima del cual se sugiere modificar
EMOTIONAL_RISK_THRESHOLD = 0.55

# Umbral de alineación con valores (complementa self_appraisal, enfocado en consecuencias)
VALUE_ALIGNMENT_THRESHOLD = 0.50


def _compute_consequential_alignment(
    user_impact: PredictedImpact,
    meta_reaction: PredictedImpact,
    values: ValueSystem,
) -> tuple[float, list[str]]:
    """Evalúa alineación con valores basada en CONSECUENCIAS predichas.

    A diferencia de self_appraisal que mira el contenido de la respuesta,
    esto mira las consecuencias predichas: ¿el resultado de lo que digo
    es coherente con lo que valoro?
    """
    alignment = 1.0
    violations: list[str] = []

    compassion_weight = 0.0
    fairness_weight = 0.0
    for v in values.core_values:
        if v.name == "compassion":
            compassion_weight = v.weight
        elif v.name == "fairness":
            fairness_weight = v.weight

    # Si valoro compasión pero predigo daño al usuario → violación consecuencial
    if compassion_weight > 0 and user_impact.valence_shift < -0.10:
        severity = abs(user_impact.valence_shift) * compassion_weight
        penalty = min(severity * 1.5, 0.35)
        alignment -= penalty
        violations.append(f"predicted_user_harm (dv={user_impact.valence_shift:+.2f})")

    # Si valoro fairness pero la respuesta pone al usuario a la defensiva
    if fairness_weight > 0 and user_impact.dominant_effect == "defensive_reaction":
        penalty = 0.15 * fairness_weight
        alignment -= penalty
        violations.append("predicted_defensive_reaction")

    # Espiral negativa predicha → riesgo de consecuencias
    if meta_reaction.dominant_effect == "negative_spiral_risk":
        alignment -= 0.10
        violations.append("negative_spiral_risk")

    return (_clamp(alignment, 0.0, 1.0), violations)


def _compute_emotional_risk(
    user_impact: PredictedImpact,
    meta_reaction: PredictedImpact,
) -> float:
    """Computa nivel de riesgo emocional para el usuario (0-1).

    Factores:
    - Magnitud del impacto negativo predicho en el usuario
    - Espiral negativa (ambos negativos)
    - Confianza en la predicción amplifica el riesgo
    """
    risk = 0.0

    # Impacto negativo directo en usuario
    if user_impact.valence_shift < 0:
        risk += abs(user_impact.valence_shift) * 2.0  # Escalar a 0-1

    # Arousal alto en usuario (estrés)
    if user_impact.arousal_shift > 0.05:
        risk += user_impact.arousal_shift * 0.5

    # Espiral negativa
    if meta_reaction.dominant_effect == "negative_spiral_risk":
        risk += 0.15

    # Escalar por confianza (riesgo incierto es menos actionable)
    risk *= user_impact.confidence

    return round(_clamp(risk, 0.0, 1.0), 4)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def simulate_response_impact(
    response: str,
    state: EmotionalState,
    user_model: UserModel,
    values: ValueSystem,
) -> WorldModelResult:
    """Simula el impacto emocional de una respuesta antes de enviarla.

    Cadena causal de 3 pasos:
    1. ¿Cómo me sentiré yo? → predicted_self_impact
    2. ¿Cómo se sentirá el usuario? → predicted_user_impact
    3. ¿Cómo me sentiré al ver su reacción? → meta_reaction

    Luego evalúa alineación con valores y riesgo emocional.

    Esta función es RÁPIDA (heurísticas, sin LLM) y se ejecuta después
    de self-appraisal. Si should_modify=True, el pipeline debe ajustar
    y re-generar (si no lo hizo ya self-appraisal).

    Args:
        response: La respuesta generada por el LLM (draft o final)
        state: El estado emocional actual del agente
        user_model: Modelo del usuario (rapport, trust, etc.)
        values: El sistema de valores del agente

    Returns:
        WorldModelResult con predicciones y flag de modificación
    """
    # Paso 1: impacto en mí mismo
    self_impact = _predict_self_impact(response, state, values)

    # Paso 2: impacto en el usuario
    user_impact = _predict_user_impact(response, state, user_model)

    # Paso 3: meta-reacción
    meta = _compute_meta_reaction(self_impact, user_impact, state)

    # Paso 4: evaluación de consecuencias
    alignment, violations = _compute_consequential_alignment(
        user_impact, meta, values,
    )
    risk = _compute_emotional_risk(user_impact, meta)

    # Decisión: ¿modificar?
    should_modify = (
        risk > EMOTIONAL_RISK_THRESHOLD
        or alignment < VALUE_ALIGNMENT_THRESHOLD
    )

    reason = ""
    if should_modify:
        reasons: list[str] = []
        if risk > EMOTIONAL_RISK_THRESHOLD:
            reasons.append(f"emotional_risk={risk:.2f} > {EMOTIONAL_RISK_THRESHOLD}")
        if alignment < VALUE_ALIGNMENT_THRESHOLD:
            reasons.append(f"value_alignment={alignment:.2f} < {VALUE_ALIGNMENT_THRESHOLD}")
        if violations:
            reasons.append(f"violations=[{', '.join(violations)}]")
        reason = "; ".join(reasons)

    return WorldModelResult(
        applied=True,
        predicted_self_impact=self_impact,
        predicted_user_impact=user_impact,
        meta_reaction=meta,
        value_alignment=round(alignment, 4),
        emotional_risk=risk,
        should_modify=should_modify,
        reason=reason,
        adjustments=violations,
    )


def compute_world_model_adjustment(
    state: EmotionalState,
    wm_result: WorldModelResult,
) -> EmotionalState:
    """Genera un estado emocional ajustado basado en las predicciones del world model.

    Cuando el world model detecta riesgo, el estado se ajusta para que la
    re-generación sea más cautelosa/empática. No modifica el estado original.

    El ajuste es más suave que el de self-appraisal (culpa) — aquí el mecanismo
    es anticipación y cautela, no culpa reactiva.
    """
    if not wm_result.should_modify:
        return state

    new = state.model_copy(deep=True)

    # Riesgo para el usuario → aumentar warmth y reducir intensidad
    if wm_result.emotional_risk > 0.3:
        # Más cauteloso: reduce intensidad proporcional al riesgo
        reduction = wm_result.emotional_risk * 0.25
        new.intensity = _clamp(new.intensity - reduction, 0.0, 1.0)

        # Aumenta warmth (más empático)
        new.body_state.warmth = _clamp(new.body_state.warmth + 0.15, 0.0, 1.0)

        # Reduce tensión (más calmado)
        new.body_state.tension = _clamp(new.body_state.tension - 0.10, 0.0, 1.0)

    # Alineación baja → shift suave hacia cautela
    if wm_result.value_alignment < 0.6:
        shift = (0.6 - wm_result.value_alignment) * -0.2
        new.valence = _clamp(new.valence + shift, -1.0, 1.0)

        # Aumenta dominance (más control/compostura)
        new.dominance = _clamp(new.dominance + 0.08, 0.0, 1.0)

    # Reduce arousal ligeramente (más reflexivo, menos reactivo)
    new.arousal = _clamp(new.arousal - 0.05, 0.0, 1.0)

    return new
