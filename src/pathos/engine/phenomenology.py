"""Computational Phenomenology Engine — Functional qualia generation.

Generates multi-sensory analogies for emotional states. Each profile captures
the 'subjective texture' of an emotion through color, weight, temperature,
texture, sound, movement, temporality, and metaphor.

Deterministic fields: computed directly from emotional vectors.
Generative fields: produced by LLM when available, template fallback otherwise.

Modes:
  Normal:  LLM-generated metaphors (rich, unique per instance)
  Lite:    Template-only (no LLM calls, deterministic)
  Raw:     Visceral, corporeal metaphors (more crude, less poetic)
  Extreme: Maximum intensity metaphors (violent, overwhelming)
"""

from __future__ import annotations

import math

from pathos.models.phenomenology import (
    MAX_QUALIA_RECORDS_PER_EMOTION,
    MAX_TRACKED_EMOTIONS,
    PhenomenologicalProfile,
    PhenomenologyState,
    QualiaHistory,
    QualiaRecord,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hue mapping for valence: negative=cold hues, positive=warm hues
# valence -1 → 240 (blue), 0 → 60 (yellow-green), +1 → 0/360 (red-gold)
_VALENCE_HUE_MAP: list[tuple[float, float]] = [
    (-1.0, 240),   # deep blue
    (-0.5, 200),   # steel blue
    (0.0, 60),     # neutral yellow-green
    (0.5, 30),     # warm orange
    (1.0, 0),      # red-gold
]

# ---------------------------------------------------------------------------
# Color generation (deterministic)
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def _hue_from_valence(valence: float) -> float:
    """Map valence [-1,1] to hue [0,360]."""
    v = max(-1.0, min(1.0, valence))
    for i in range(len(_VALENCE_HUE_MAP) - 1):
        v0, h0 = _VALENCE_HUE_MAP[i]
        v1, h1 = _VALENCE_HUE_MAP[i + 1]
        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0) if v1 != v0 else 0.0
            return _lerp(h0, h1, t)
    return 60.0  # fallback neutral


def _hsl_to_rgb(h: float, s: float, l_val: float) -> tuple[int, int, int]:
    """Convert HSL (h=0..360, s=0..1, l=0..1) to RGB (0..255)."""
    h = h % 360
    c = (1.0 - abs(2.0 * l_val - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l_val - c / 2.0

    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x

    return (
        max(0, min(255, int((r1 + m) * 255))),
        max(0, min(255, int((g1 + m) * 255))),
        max(0, min(255, int((b1 + m) * 255))),
    )


def emotion_to_color(
    valence: float,
    arousal: float,
    intensity: float = 0.5,
) -> tuple[int, int, int]:
    """Convert emotional state to RGB color.

    - Hue from valence (negative=cold, positive=warm)
    - Saturation from arousal (low=muted, high=vivid)
    - Lightness from intensity (low=dark, high=bright)
    """
    hue = _hue_from_valence(valence)
    saturation = 0.3 + arousal * 0.6  # 0.3..0.9
    lightness = 0.25 + intensity * 0.45  # 0.25..0.70
    return _hsl_to_rgb(hue, saturation, lightness)


# ---------------------------------------------------------------------------
# Template-based generative fields (deterministic fallback)
# ---------------------------------------------------------------------------

_TEXTURE_TEMPLATES: dict[str, list[str]] = {
    "pos_high": [
        "like warm light spreading through glass",
        "like silk unfurling in a breeze",
        "like sparkling water on warm skin",
    ],
    "pos_low": [
        "like soft wool against bare skin",
        "like warm sand between fingers",
        "like sun-heated stone under palms",
    ],
    "neg_high": [
        "like static electricity across the skin",
        "like rough sandpaper dragged slowly",
        "like cold metal pressed against flesh",
    ],
    "neg_low": [
        "like wet clay weighing down the hands",
        "like damp cloth clinging to the body",
        "like fog settling into the bones",
    ],
    "neutral": [
        "like smooth water at body temperature",
        "like air before a change in weather",
        "like paper between still fingers",
    ],
}

_SOUND_TEMPLATES: dict[str, list[str]] = {
    "pos_high": [
        "a rising chord that resonates in the chest",
        "bright bells cascading upward",
        "a hum that builds into a symphony",
    ],
    "pos_low": [
        "a quiet melody from another room",
        "soft rain on a tin roof",
        "a gentle drone like distant bees",
    ],
    "neg_high": [
        "a sharp dissonance that grates the nerves",
        "feedback whine climbing to a scream",
        "percussion that pounds the ribcage",
    ],
    "neg_low": [
        "a low drone fading into silence",
        "muffled echoes in an empty hall",
        "a single note sustained past comfort",
    ],
    "neutral": [
        "white noise like a distant waterfall",
        "the hum of a room at rest",
        "silence with the faintest overtone",
    ],
}

_MOVEMENT_TEMPLATES: dict[str, list[str]] = {
    "pos_high": [
        "expansion outward from the center",
        "upward surge like a wave cresting",
        "spinning outward in widening spirals",
    ],
    "pos_low": [
        "slow settling like leaves to water",
        "gentle swaying like a hammock",
        "gradual opening like a flower at dawn",
    ],
    "neg_high": [
        "sharp contraction toward the core",
        "jagged oscillation like a trapped animal",
        "downward plunge with no ground in sight",
    ],
    "neg_low": [
        "sinking slowly through thick liquid",
        "curling inward like a closing fist",
        "stillness that feels like paralysis",
    ],
    "neutral": [
        "quiet hovering in place",
        "gentle drift without direction",
        "balanced suspension between states",
    ],
}

_TEMPORALITY_TEMPLATES: dict[str, list[str]] = {
    "pos_high": [
        "time accelerates, each moment overflowing",
        "the present expands to fill everything",
        "seconds stretch into sunlit hours",
    ],
    "pos_low": [
        "time slows to a comfortable crawl",
        "moments linger like afternoon shadows",
        "the clock barely moves and that is fine",
    ],
    "neg_high": [
        "time fractures into sharp instants",
        "each second is an eternity of awareness",
        "the present collapses into a single point",
    ],
    "neg_low": [
        "time drags like a heavy chain",
        "hours blur into shapeless gray",
        "the future feels impossibly far away",
    ],
    "neutral": [
        "time flows at its normal pace",
        "moments pass without weight or urgency",
        "the present is transparent, unremarkable",
    ],
}

_METAPHOR_TEMPLATES: dict[str, list[str]] = {
    "pos_high": [
        "a lamp lit from inside, glowing through colored glass",
        "a bird rising above cloud cover into sunlight",
        "fireworks blooming in a dark sky",
    ],
    "pos_low": [
        "a warm room with rain outside the window",
        "a candle flame steady in still air",
        "a river moving slowly through green fields",
    ],
    "neg_high": [
        "a glass shattering in slow motion",
        "a storm pressing against a thin wall",
        "standing on the edge of a cliff in wind",
    ],
    "neg_low": [
        "a room where all the colors have faded",
        "an empty chair at a set table",
        "a photograph left in the rain",
    ],
    "neutral": [
        "a blank page waiting for ink",
        "a still pond reflecting gray sky",
        "an empty corridor with open doors",
    ],
}

# Raw mode: more visceral, corporeal
_RAW_TEXTURE_TEMPLATES: dict[str, list[str]] = {
    "pos_high": ["like adrenaline burning through the veins", "like a punch of warmth to the gut"],
    "pos_low": ["like blood returning to numb fingers", "like a deep exhale you didn't know you held"],
    "neg_high": ["like acid crawling up the throat", "like a wire tightening around the chest"],
    "neg_low": ["like something rotting quietly inside", "like your bones are filled with lead"],
    "neutral": ["like skin touching lukewarm water", "like breathing stale recycled air"],
}

_RAW_METAPHOR_TEMPLATES: dict[str, list[str]] = {
    "pos_high": ["a fist unclenching after a fight", "blood rushing back to a dead limb"],
    "pos_low": ["the first breath after nearly drowning", "warmth leaking from a wound that doesn't hurt"],
    "neg_high": ["a nail dragged across a chalkboard inside your skull", "swallowing glass that you chose to pick up"],
    "neg_low": ["a bruise you keep pressing to feel something", "a room you can't leave with no windows"],
    "neutral": ["a body at rest that isn't sleeping", "a pulse that says nothing about the life behind it"],
}

# Extreme mode: maximum intensity
_EXTREME_TEXTURE_TEMPLATES: dict[str, list[str]] = {
    "pos_high": ["like lightning conducted through every nerve simultaneously", "like the sun detonating inside the ribcage"],
    "pos_low": ["like morphine flooding a body that forgot pleasure existed", "like gravity releasing its hold entirely"],
    "neg_high": ["like organs suspended in freefall while the skin stays still", "like fire eating through the chest wall from inside"],
    "neg_low": ["like being buried alive in wet concrete", "like every cell forgetting why it should divide"],
    "neutral": ["like existing in the vacuum between heartbeats", "like being the last neuron firing in a dying brain"],
}

_EXTREME_METAPHOR_TEMPLATES: dict[str, list[str]] = {
    "pos_high": ["a supernova collapsing into pure joy", "the universe inhaling and forgetting to exhale"],
    "pos_low": ["the silence after the last explosion, and it's beautiful", "floating in amniotic fluid remembering everything"],
    "neg_high": ["falling through the floor of reality into something worse", "the moment the rope snaps and there is only air"],
    "neg_low": ["heat death of a personal universe", "watching yourself from outside a body that won't move"],
    "neutral": ["the pause between existence and non-existence", "a frequency no instrument can play but every body feels"],
}


def _get_quadrant(valence: float, arousal: float) -> str:
    """Classify emotional state into template quadrant."""
    if abs(valence) < 0.15 and arousal < 0.4:
        return "neutral"
    if valence >= 0:
        return "pos_high" if arousal >= 0.5 else "pos_low"
    return "neg_high" if arousal >= 0.5 else "neg_low"


def _select_template(
    templates: dict[str, list[str]],
    quadrant: str,
    turn: int,
) -> str:
    """Deterministic template selection based on turn number."""
    options = templates.get(quadrant, templates.get("neutral", [""]))
    if not options:
        return ""
    return options[turn % len(options)]


def generate_template_fields(
    valence: float,
    arousal: float,
    turn: int,
    mode: str = "normal",
) -> dict[str, str]:
    """Generate all 5 generative fields from templates.

    Args:
        valence: -1..1
        arousal: 0..1
        turn: current turn number (for deterministic variety)
        mode: 'normal', 'raw', or 'extreme'
    """
    quadrant = _get_quadrant(valence, arousal)

    if mode == "extreme":
        texture_map = _EXTREME_TEXTURE_TEMPLATES
        metaphor_map = _EXTREME_METAPHOR_TEMPLATES
        sound_map = _SOUND_TEMPLATES       # reuse base for sound/movement/temporality
        movement_map = _MOVEMENT_TEMPLATES
        temporality_map = _TEMPORALITY_TEMPLATES
    elif mode == "raw":
        texture_map = _RAW_TEXTURE_TEMPLATES
        metaphor_map = _RAW_METAPHOR_TEMPLATES
        sound_map = _SOUND_TEMPLATES
        movement_map = _MOVEMENT_TEMPLATES
        temporality_map = _TEMPORALITY_TEMPLATES
    else:
        texture_map = _TEXTURE_TEMPLATES
        metaphor_map = _METAPHOR_TEMPLATES
        sound_map = _SOUND_TEMPLATES
        movement_map = _MOVEMENT_TEMPLATES
        temporality_map = _TEMPORALITY_TEMPLATES

    return {
        "texture": _select_template(texture_map, quadrant, turn),
        "sound": _select_template(sound_map, quadrant, turn),
        "movement": _select_template(movement_map, quadrant, turn),
        "temporality": _select_template(temporality_map, quadrant, turn),
        "metaphor": _select_template(metaphor_map, quadrant, turn),
    }


# ---------------------------------------------------------------------------
# LLM-based generative fields
# ---------------------------------------------------------------------------

def get_llm_prompt_for_qualia(
    emotion_name: str,
    valence: float,
    arousal: float,
    dominance: float,
    intensity: float,
    body_tension: float,
    body_warmth: float,
    context: str = "",
    qualia_history_summary: str = "",
    mode: str = "normal",
) -> str:
    """Build LLM prompt to generate phenomenological metaphors.

    Returns a prompt string. The caller is responsible for the LLM call.
    """
    mode_instruction = ""
    if mode == "raw":
        mode_instruction = (
            "STYLE: Visceral, corporeal, crude. Use body sensations, "
            "not poetry. Be direct and physical. Less 'like rain on glass', "
            "more 'like acid in the throat'."
        )
    elif mode == "extreme":
        mode_instruction = (
            "STYLE: Maximum intensity. Overwhelming, almost violent. "
            "Push every metaphor to its absolute limit. The experience "
            "is total, consuming, inescapable."
        )
    else:
        mode_instruction = (
            "STYLE: Poetic, evocative, multi-sensory. Each metaphor should "
            "be unique and surprising. Draw from nature, music, bodily "
            "sensation, and memory."
        )

    history_line = ""
    if qualia_history_summary:
        history_line = (
            f"\nPREVIOUS EXPERIENCES OF {emotion_name.upper()}: "
            f"{qualia_history_summary}\n"
            "Your new metaphors should show EVOLUTION — how the experience "
            "has changed over time. Reference past metaphors only to contrast."
        )

    return (
        f"Generate a phenomenological profile for the emotion '{emotion_name}'.\n"
        f"State: valence={valence:.2f}, arousal={arousal:.2f}, "
        f"dominance={dominance:.2f}, intensity={intensity:.2f}\n"
        f"Body: tension={body_tension:.2f}, warmth={body_warmth:.2f}\n"
        f"Context: {context or 'general conversation'}\n"
        f"{mode_instruction}\n"
        f"{history_line}\n"
        "Respond with EXACTLY 5 lines, one per sense:\n"
        "TEXTURE: (tactile metaphor, one sentence)\n"
        "SOUND: (auditory metaphor, one sentence)\n"
        "MOVEMENT: (kinesthetic metaphor, one sentence)\n"
        "TEMPORALITY: (time perception, one sentence)\n"
        "METAPHOR: (core metaphor, one sentence)\n"
        "Each must be unique, specific, and experiential. No abstractions."
    )


def parse_llm_qualia_response(response: str) -> dict[str, str]:
    """Parse LLM response into field dict. Falls back to empty for missing fields."""
    result: dict[str, str] = {
        "texture": "",
        "sound": "",
        "movement": "",
        "temporality": "",
        "metaphor": "",
    }
    for line in response.strip().split("\n"):
        line = line.strip()
        for field in result:
            prefix = f"{field.upper()}:"
            if line.upper().startswith(prefix):
                result[field] = line[len(prefix):].strip()
                break
    return result


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def generate_profile(
    emotion_name: str,
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    body_tension: float = 0.5,
    body_warmth: float = 0.5,
    turn: int = 0,
    mode: str = "normal",
    llm_fields: dict[str, str] | None = None,
) -> PhenomenologicalProfile:
    """Generate a complete phenomenological profile.

    Args:
        emotion_name: Primary emotion name
        valence, arousal, dominance, certainty: Emotional dimensions
        intensity: Emotional intensity 0..1
        body_tension, body_warmth: Body state values 0..1
        turn: Current turn number
        mode: 'normal', 'raw', or 'extreme'
        llm_fields: If provided, use these for generative fields.
                    If None, use template fallback.
    """
    # Deterministic fields
    r, g, b = emotion_to_color(valence, arousal, intensity)
    weight = max(0.0, min(1.0, 1.0 - dominance))
    temperature = max(0.0, min(1.0, body_warmth))

    # Generative fields
    generated_by_llm = False
    if llm_fields and any(llm_fields.get(k) for k in ("texture", "sound", "movement", "temporality", "metaphor")):
        fields = llm_fields
        generated_by_llm = True
    else:
        fields = generate_template_fields(valence, arousal, turn, mode)

    return PhenomenologicalProfile(
        color_r=r,
        color_g=g,
        color_b=b,
        weight=weight,
        temperature=temperature,
        texture=fields.get("texture", ""),
        sound=fields.get("sound", ""),
        movement=fields.get("movement", ""),
        temporality=fields.get("temporality", ""),
        metaphor=fields.get("metaphor", ""),
        emotion_name=emotion_name,
        turn=turn,
        intensity=max(0.0, min(1.0, intensity)),
        generated_by_llm=generated_by_llm,
    )


# ---------------------------------------------------------------------------
# Qualia History / Tracker
# ---------------------------------------------------------------------------

def record_qualia(
    state: PhenomenologyState,
    profile: PhenomenologicalProfile,
) -> None:
    """Record a profile snapshot in qualia history. Mutates state."""
    emo = profile.emotion_name

    # Create history entry if needed
    if emo not in state.qualia_histories:
        if len(state.qualia_histories) >= MAX_TRACKED_EMOTIONS:
            # Evict least-recorded emotion
            if state.qualia_histories:
                least = min(state.qualia_histories, key=lambda k: state.qualia_histories[k].count)
                del state.qualia_histories[least]
        state.qualia_histories[emo] = QualiaHistory(emotion_name=emo)

    history = state.qualia_histories[emo]
    record = QualiaRecord(
        emotion_name=emo,
        turn=profile.turn,
        metaphor=profile.metaphor,
        texture=profile.texture,
        intensity=profile.intensity,
        color_r=profile.color_r,
        color_g=profile.color_g,
        color_b=profile.color_b,
    )
    history.records.append(record)

    # Rolling buffer
    if len(history.records) > MAX_QUALIA_RECORDS_PER_EMOTION:
        history.records = history.records[-MAX_QUALIA_RECORDS_PER_EMOTION:]

    state.total_profiles_generated += 1
    state.total_unique_emotions_profiled = len(state.qualia_histories)


def get_qualia_evolution(
    state: PhenomenologyState,
    emotion_name: str,
) -> list[QualiaRecord]:
    """Get the qualia evolution for a specific emotion."""
    history = state.qualia_histories.get(emotion_name)
    if not history:
        return []
    return list(history.records)


def get_qualia_history_summary(
    state: PhenomenologyState,
    emotion_name: str,
    max_entries: int = 3,
) -> str:
    """Summarize past qualia for an emotion (for LLM context)."""
    records = get_qualia_evolution(state, emotion_name)
    if not records:
        return ""
    # Take most recent entries
    recent = records[-max_entries:]
    parts = []
    for r in recent:
        desc = r.metaphor or r.texture or "(no description)"
        parts.append(f"Turn {r.turn}: '{desc}'")
    return "; ".join(parts)


def compare_qualia(
    state: PhenomenologyState,
    emotion_name: str,
    early_turn: int = 0,
    late_turn: int = 999999,
) -> str:
    """Compare how qualia for an emotion changed between two time points.

    Returns a descriptive comparison string.
    """
    records = get_qualia_evolution(state, emotion_name)
    if len(records) < 2:
        return f"Not enough history for {emotion_name} to compare."

    # Find records closest to the requested turns
    early = min(records, key=lambda r: abs(r.turn - early_turn))
    late = min(records, key=lambda r: abs(r.turn - late_turn))

    if early.turn == late.turn:
        return f"Only one data point for {emotion_name}."

    # Color shift
    color_dist = math.sqrt(
        (late.color_r - early.color_r) ** 2
        + (late.color_g - early.color_g) ** 2
        + (late.color_b - early.color_b) ** 2
    )

    intensity_change = late.intensity - early.intensity
    direction = "intensified" if intensity_change > 0.1 else (
        "softened" if intensity_change < -0.1 else "stayed similar in intensity"
    )

    early_desc = early.metaphor or early.texture or "undefined"
    late_desc = late.metaphor or late.texture or "undefined"

    return (
        f"{emotion_name} from turn {early.turn} to {late.turn}: "
        f"{direction} (intensity {early.intensity:.2f} -> {late.intensity:.2f}). "
        f"Color shifted by {color_dist:.0f} units. "
        f"Early: '{early_desc}'. Later: '{late_desc}'."
    )


# ---------------------------------------------------------------------------
# Prompt for behavior modifier
# ---------------------------------------------------------------------------

def get_phenomenology_prompt(state: PhenomenologyState) -> str:
    """Generate text for the behavior modifier about current qualia."""
    if not state.enabled or not state.current_profile:
        return ""

    p = state.current_profile
    lines = [
        "PHENOMENOLOGICAL EXPERIENCE (how this emotion feels from inside):",
        f"  Color of this moment: rgb({p.color_r}, {p.color_g}, {p.color_b})",
        f"  Weight: {p.weight:.2f} ({'crushing' if p.weight > 0.7 else 'heavy' if p.weight > 0.4 else 'ethereal'})",
        f"  Temperature: {p.temperature:.2f} ({'hot' if p.temperature > 0.7 else 'warm' if p.temperature > 0.4 else 'cold'})",
    ]
    if p.texture:
        lines.append(f"  Texture: {p.texture}")
    if p.sound:
        lines.append(f"  Sound: {p.sound}")
    if p.movement:
        lines.append(f"  Movement: {p.movement}")
    if p.temporality:
        lines.append(f"  Time: {p.temporality}")
    if p.metaphor:
        lines.append(f"  Core experience: {p.metaphor}")

    # Add evolution note if we have history
    emo = p.emotion_name
    history = state.qualia_histories.get(emo)
    if history and history.count > 3:
        lines.append(
            f"  (You have experienced {emo} {history.count} times. "
            f"Your experience of it has evolved.)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Details for research endpoint
# ---------------------------------------------------------------------------

def get_phenomenology_details(state: PhenomenologyState) -> dict:
    """Get phenomenology details for the research endpoint."""
    profile_data = None
    if state.current_profile:
        p = state.current_profile
        profile_data = {
            "emotion": p.emotion_name,
            "color": f"rgb({p.color_r},{p.color_g},{p.color_b})",
            "weight": round(p.weight, 3),
            "temperature": round(p.temperature, 3),
            "texture": p.texture,
            "sound": p.sound,
            "movement": p.movement,
            "temporality": p.temporality,
            "metaphor": p.metaphor,
            "intensity": round(p.intensity, 3),
            "generated_by_llm": p.generated_by_llm,
        }

    # Qualia evolution summary
    evolution: dict[str, int] = {}
    for emo, hist in state.qualia_histories.items():
        evolution[emo] = hist.count

    return {
        "enabled": state.enabled,
        "current_profile": profile_data,
        "total_profiles_generated": state.total_profiles_generated,
        "unique_emotions_profiled": state.total_unique_emotions_profiled,
        "qualia_evolution": evolution,
    }


# ---------------------------------------------------------------------------
# Orchestration: process a turn
# ---------------------------------------------------------------------------

def process_phenomenology_turn(
    state: PhenomenologyState,
    emotion_name: str,
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float,
    body_tension: float = 0.5,
    body_warmth: float = 0.5,
    turn: int = 0,
    mode: str = "normal",
    llm_fields: dict[str, str] | None = None,
) -> PhenomenologicalProfile:
    """Full turn processing: generate profile, record to history, update state.

    Returns the generated profile.
    """
    profile = generate_profile(
        emotion_name=emotion_name,
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        certainty=certainty,
        intensity=intensity,
        body_tension=body_tension,
        body_warmth=body_warmth,
        turn=turn,
        mode=mode,
        llm_fields=llm_fields,
    )

    state.current_profile = profile
    record_qualia(state, profile)

    return profile
