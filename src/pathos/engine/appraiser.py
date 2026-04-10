"""Appraisal Module - Evalua estimulos contra los valores del agente.

Enfoque hibrido para modelos pequenos (4B):
- El LLM clasifica la emocion por nombre (su fortaleza)
- El LLM da valence y arousal (2 numeros simples)
- El codigo construye el AppraisalVector para behavior modifier/body state
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pathos.models.appraisal import (
    AgencyAttribution,
    AppraisalVector,
    CopingPotential,
    NormCompatibility,
    RelevanceCheck,
    ValenceAssessment,
)
from pathos.models.emotion import PrimaryEmotion

if TYPE_CHECKING:
    from pathos.llm.base import LLMProvider
    from pathos.models.values import ValueSystem

# Emotion names for the prompt
_EMOTION_NAMES = [e.value for e in PrimaryEmotion]

APPRAISAL_SYSTEM = """You classify emotions from stimuli. Output ONLY a JSON object. No other text.

Emotions: joy, excitement, gratitude, hope, contentment, relief, anger, frustration, fear, anxiety, sadness, helplessness, disappointment, surprise, alertness, contemplation, indifference, mixed, neutral"""

APPRAISAL_USER = """Classify: "{stimulus}"

{{"emotion":"<name>","valence":<-1.0 to 1.0>,"arousal":<0.0 to 1.0>}}

"Someone stole credit for your work" -> {{"emotion":"anger","valence":-0.8,"arousal":0.85}}
"You won a surprise lottery!" -> {{"emotion":"excitement","valence":0.9,"arousal":0.95}}
"A friend helped you all night" -> {{"emotion":"gratitude","valence":0.7,"arousal":0.3}}
"You got promoted for your hard work" -> {{"emotion":"joy","valence":0.8,"arousal":0.6}}
"You hear footsteps at night" -> {{"emotion":"fear","valence":-0.7,"arousal":0.9}}
"A loved one passed away" -> {{"emotion":"sadness","valence":-0.9,"arousal":0.2}}
"All work done, nothing pending" -> {{"emotion":"contentment","valence":0.6,"arousal":0.1}}
"The weather is the same" -> {{"emotion":"neutral","valence":0.0,"arousal":0.1}}

ONLY the JSON."""


@dataclass
class AppraisalResult:
    """Result of appraising a stimulus: AppraisalVector + direct emotion classification."""

    vector: AppraisalVector
    emotion_hint: PrimaryEmotion | None = None


# Control and fairness defaults per emotion category (for AppraisalVector construction)
_EMOTION_DEFAULTS: dict[str, tuple[float, float]] = {
    # (control, fairness)
    "joy":            (0.7,  0.7),
    "excitement":     (0.3,  0.5),
    "gratitude":      (0.2,  0.9),
    "hope":           (0.3,  0.3),
    "contentment":    (0.9,  0.5),
    "relief":         (0.5,  0.5),
    "anger":          (0.4, -0.8),
    "frustration":    (0.35, -0.4),
    "fear":           (0.1,  0.0),
    "anxiety":        (0.2,  0.0),
    "sadness":        (0.1,  0.0),
    "helplessness":   (0.05, -0.1),
    "disappointment": (0.3, -0.3),
    "surprise":       (0.4,  0.1),
    "alertness":      (0.5,  0.0),
    "contemplation":  (0.6,  0.3),
    "indifference":   (0.5,  0.0),
    "mixed":          (0.3,  0.0),
    "neutral":        (0.5,  0.0),
}


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convierte a float de forma segura."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_response(raw: str) -> dict[str, object]:
    """Parsea la respuesta del LLM.

    Espera: {"emotion": "name", "valence": X, "arousal": X}
    """
    text = raw.strip()

    # Remover thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remover markdown
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Extraer JSON
    brace_start = text.find("{")
    if brace_start > 0:
        text = text[brace_start:]
    brace_end = text.rfind("}")
    if brace_end >= 0:
        text = text[: brace_end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"emotion": "neutral", "valence": 0.0, "arousal": 0.3}

    # Parse emotion name
    emotion_raw = str(data.get("emotion", "neutral")).lower().strip()
    if emotion_raw not in _EMOTION_DEFAULTS:
        emotion_raw = "neutral"

    return {
        "emotion": emotion_raw,
        "valence": _clamp(_safe_float(data.get("valence", 0.0)), -1, 1),
        "arousal": _clamp(_safe_float(data.get("arousal", 0.3)), 0, 1),
    }


def _build_appraisal(emotion: str, valence: float, arousal: float) -> AppraisalVector:
    """Construye AppraisalVector desde emocion + valence + arousal.

    El vector se usa para behavior modifier y body state.
    La clasificacion emocional viene directa del LLM (emotion_hint).
    """
    v = valence
    a = arousal
    c, f = _EMOTION_DEFAULTS.get(emotion, (0.5, 0.0))

    # Significance y novelty desde arousal
    significance = _clamp(a / 0.6 * 0.8, 0, 1)
    novelty = _clamp((a - significance * 0.6) / 0.4, 0, 1)

    return AppraisalVector(
        relevance=RelevanceCheck(
            novelty=_clamp(novelty, -1, 1),
            personal_significance=significance,
            values_affected=[],
        ),
        valence=ValenceAssessment(
            goal_conduciveness=v,
            value_alignment=v,
            intrinsic_pleasantness=v,
        ),
        coping=CopingPotential(
            control=c,
            power=_clamp(c * 0.8, 0, 1),
            adjustability=_clamp(c * 0.5 + (1 - a) * 0.4 + 0.1, 0, 1),
        ),
        agency=AgencyAttribution(
            responsible_agent="environment",
            intentionality=0.5,
            fairness=f,
        ),
        norms=NormCompatibility(
            internal_standards=_clamp(f * 0.7, -1, 1),
            external_standards=_clamp(f * 0.6, -1, 1),
            self_consistency=0.0,
        ),
    )


def _format_values(value_system: ValueSystem) -> str:
    """Formatea los valores del agente para el prompt."""
    lines = []
    for v in value_system.core_values:
        lines.append(f"- {v.name} (peso: {v.weight}): {v.description}")
    return "\n".join(lines)


def _resolve_emotion(name: str) -> PrimaryEmotion | None:
    """Convierte string a PrimaryEmotion, None si no es valido."""
    try:
        return PrimaryEmotion(name)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Keyword-based appraisal for Lite Mode (no LLM call)
# ---------------------------------------------------------------------------
# Each entry: (keyword_fragment, emotion, valence, arousal)
# Fragments are matched with `in` against the lowercased stimulus.
_KEYWORD_EMOTION_MAP: list[tuple[str, str, float, float]] = [
    # --- Joy / Positive ---
    ("feliz", "joy", 0.7, 0.6),
    ("contento", "joy", 0.6, 0.5),
    ("alegr", "joy", 0.7, 0.6),
    ("happy", "joy", 0.7, 0.6),
    ("glad", "joy", 0.6, 0.5),
    ("jaja", "joy", 0.5, 0.5),
    ("jeje", "joy", 0.4, 0.4),
    ("haha", "joy", 0.5, 0.5),
    ("lol", "joy", 0.5, 0.5),
    # --- Excitement ---
    ("emocionad", "excitement", 0.8, 0.85),
    ("increible", "excitement", 0.8, 0.8),
    ("genial", "excitement", 0.7, 0.7),
    ("excited", "excitement", 0.8, 0.85),
    ("amazing", "excitement", 0.8, 0.85),
    ("awesome", "excitement", 0.7, 0.8),
    ("wow", "surprise", 0.3, 0.8),
    # --- Gratitude ---
    ("gracias", "gratitude", 0.6, 0.3),
    ("agradec", "gratitude", 0.7, 0.3),
    ("thanks", "gratitude", 0.6, 0.3),
    ("grateful", "gratitude", 0.7, 0.3),
    ("appreciate", "gratitude", 0.6, 0.3),
    # --- Hope ---
    ("espero", "hope", 0.4, 0.4),
    ("ojala", "hope", 0.5, 0.4),
    ("esperanza", "hope", 0.5, 0.4),
    ("hope", "hope", 0.5, 0.4),
    ("hopeful", "hope", 0.5, 0.4),
    # --- Contentment ---
    ("tranquil", "contentment", 0.5, 0.15),
    ("paz", "contentment", 0.5, 0.1),
    ("relajad", "contentment", 0.5, 0.15),
    ("peaceful", "contentment", 0.5, 0.1),
    ("calm", "contentment", 0.5, 0.15),
    ("relaxed", "contentment", 0.5, 0.15),
    # --- Relief ---
    ("alivio", "relief", 0.5, 0.3),
    ("aliviad", "relief", 0.5, 0.3),
    ("por fin", "relief", 0.5, 0.35),
    ("relief", "relief", 0.5, 0.3),
    ("relieved", "relief", 0.5, 0.3),
    ("finally", "relief", 0.4, 0.35),
    # --- Anger ---
    ("odio", "anger", -0.8, 0.85),
    ("furios", "anger", -0.85, 0.9),
    ("enojad", "anger", -0.7, 0.8),
    ("rabia", "anger", -0.8, 0.85),
    ("maldita", "anger", -0.7, 0.8),
    ("hate", "anger", -0.8, 0.85),
    ("furious", "anger", -0.85, 0.9),
    ("angry", "anger", -0.7, 0.8),
    # --- Frustration ---
    ("frustra", "frustration", -0.6, 0.7),
    ("harto", "frustration", -0.6, 0.7),
    ("no sirve", "frustration", -0.5, 0.6),
    ("no funciona", "frustration", -0.5, 0.6),
    ("frustrated", "frustration", -0.6, 0.7),
    ("annoying", "frustration", -0.5, 0.6),
    ("fed up", "frustration", -0.6, 0.7),
    ("stuck", "frustration", -0.5, 0.6),
    # --- Fear ---
    ("miedo", "fear", -0.7, 0.8),
    ("terror", "fear", -0.8, 0.9),
    ("asusta", "fear", -0.7, 0.8),
    ("afraid", "fear", -0.7, 0.8),
    ("scared", "fear", -0.7, 0.8),
    ("terrif", "fear", -0.8, 0.9),
    # --- Anxiety ---
    ("ansiedad", "anxiety", -0.5, 0.7),
    ("nervios", "anxiety", -0.5, 0.7),
    ("preocupa", "anxiety", -0.5, 0.6),
    ("panico", "anxiety", -0.7, 0.9),
    ("anxious", "anxiety", -0.5, 0.7),
    ("nervous", "anxiety", -0.5, 0.7),
    ("worried", "anxiety", -0.5, 0.6),
    ("panic", "anxiety", -0.7, 0.9),
    # --- Sadness ---
    ("triste", "sadness", -0.7, 0.25),
    ("llor", "sadness", -0.7, 0.3),
    ("deprimid", "sadness", -0.8, 0.2),
    ("solo", "sadness", -0.5, 0.2),
    ("sad", "sadness", -0.7, 0.25),
    ("crying", "sadness", -0.7, 0.3),
    ("depressed", "sadness", -0.8, 0.2),
    ("lonely", "sadness", -0.6, 0.2),
    # --- Helplessness ---
    ("no puedo mas", "helplessness", -0.8, 0.3),
    ("no se que hacer", "helplessness", -0.7, 0.35),
    ("impotente", "helplessness", -0.8, 0.3),
    ("helpless", "helplessness", -0.8, 0.3),
    ("powerless", "helplessness", -0.8, 0.3),
    # --- Disappointment ---
    ("decepciona", "disappointment", -0.5, 0.35),
    ("esperaba mas", "disappointment", -0.5, 0.3),
    ("disappointed", "disappointment", -0.5, 0.35),
    ("let down", "disappointment", -0.5, 0.3),
    # --- Surprise ---
    ("sorpres", "surprise", 0.1, 0.8),
    ("no me esperaba", "surprise", 0.1, 0.8),
    ("de repente", "surprise", 0.0, 0.75),
    ("surprise", "surprise", 0.1, 0.8),
    ("unexpected", "surprise", 0.0, 0.75),
    ("suddenly", "surprise", 0.0, 0.7),
    # --- Alertness ---
    ("cuidado", "alertness", -0.2, 0.7),
    ("atencion", "alertness", 0.0, 0.6),
    ("alert", "alertness", 0.0, 0.6),
    ("careful", "alertness", -0.1, 0.6),
    # --- Contemplation ---
    ("pienso", "contemplation", 0.0, 0.3),
    ("reflexion", "contemplation", 0.1, 0.3),
    ("pregunto", "contemplation", 0.0, 0.35),
    ("thinking", "contemplation", 0.0, 0.3),
    ("wonder", "contemplation", 0.1, 0.35),
    # --- Emoji boosters ---
    ("\U0001f600", "joy", 0.6, 0.5),         # 😀
    ("\U0001f602", "joy", 0.7, 0.6),         # 😂
    ("\U0001f622", "sadness", -0.6, 0.3),    # 😢
    ("\U0001f62d", "sadness", -0.7, 0.4),    # 😭
    ("\U0001f621", "anger", -0.7, 0.8),      # 😡
    ("\U0001f92c", "anger", -0.8, 0.9),      # 🤬
    ("\U0001f631", "fear", -0.7, 0.85),      # 😱
    ("\u2764", "gratitude", 0.6, 0.3),       # ❤
    ("\U0001f494", "sadness", -0.6, 0.3),    # 💔
    ("\U0001f525", "excitement", 0.7, 0.8),  # 🔥
]


def appraise_lite(stimulus: str) -> AppraisalResult:
    """Keyword-based appraisal — no LLM call, <1ms.

    Scans the stimulus for keyword fragments and accumulates votes
    per emotion. Returns the same AppraisalResult as the LLM version.
    """
    text = stimulus.lower()

    # Accumulate votes: emotion -> (total_valence, total_arousal, count)
    votes: dict[str, list[float]] = {}
    for fragment, emotion, valence, arousal in _KEYWORD_EMOTION_MAP:
        if fragment in text:
            if emotion not in votes:
                votes[emotion] = [0.0, 0.0, 0.0]
            votes[emotion][0] += valence
            votes[emotion][1] += arousal
            votes[emotion][2] += 1.0

    if not votes:
        emotion_name = "neutral"
        valence = 0.0
        arousal = 0.3
    else:
        # Pick emotion with most keyword hits
        emotion_name = max(votes, key=lambda e: votes[e][2])
        v_sum, a_sum, count = votes[emotion_name]
        valence = _clamp(v_sum / count, -1.0, 1.0)
        arousal = _clamp(a_sum / count, 0.0, 1.0)

    vector = _build_appraisal(emotion_name, valence, arousal)
    emotion_hint = _resolve_emotion(emotion_name)
    return AppraisalResult(vector=vector, emotion_hint=emotion_hint)


async def appraise(
    stimulus: str,
    value_system: ValueSystem,
    llm: LLMProvider,
    think: bool = True,
) -> AppraisalResult:
    """Evalua un estimulo usando el LLM.

    Retorna AppraisalResult con:
    - vector: AppraisalVector para behavior modifier/body state
    - emotion_hint: emocion clasificada directamente por el LLM
    """
    user_msg = APPRAISAL_USER.format(stimulus=stimulus)

    response = await llm.generate(
        system_prompt=APPRAISAL_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        think=think,
    )

    parsed = _parse_response(response)
    emotion_name = str(parsed["emotion"])
    valence = float(parsed["valence"])
    arousal = float(parsed["arousal"])

    vector = _build_appraisal(emotion_name, valence, arousal)
    emotion_hint = _resolve_emotion(emotion_name)

    return AppraisalResult(vector=vector, emotion_hint=emotion_hint)
