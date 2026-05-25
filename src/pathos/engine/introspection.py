"""RESIDUUM F2.2 — Introspection Engine.

Projects a residual-stream activation captured by IntrospectiveTransformersProvider
(F2.1) onto the 171 emotion probes loaded by ProbeLibrary (F1.3) and produces:

  - InternalEmotionState  : MEASURED state (top-5 emotions + VAD-C)
  - AuthenticityGap       : divergence between measured and v5-calculated state

When F2 (Introspection) is ON, the InternalEmotionState is the source of truth
for authenticity; the v5 EmotionalState is a hypothesis. See CLAUDE.md
"Measured > Calculated principle".

The 4D dimensional projection (valence, arousal, dominance, certainty):
  - valence/arousal: weighted average of emotions_171.json valence_est/
    arousal_est across the top-k, weights = |cosine_sim|.
  - dominance: cluster-based heuristic (JSON does not store dominance_est).
  - certainty: mean |cosine_sim| of the top-k (peakiness of the projection).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from pathos.engine.steering import ProbeLibrary
from pathos.models.emotion import EmotionalState
from pathos.models.residuum import (
    AuthenticityGap,
    DivergenceCategory,
    DivergenceEvent,
    DivergenceInterpretation,
    EmotionProjection,
    InternalEmotionState,
    ResiduumState,
    append_divergence_event,
    append_gap,
)


# Number of consecutive divergence-risk/critical turns required before the
# orchestrator passes repeated_pattern=True to compute_authenticity_gap.
DIVERGENCE_REPEAT_THRESHOLD: int = 3

# Capture-point synonyms that the orchestrator accepts from callers.
CAPTURE_POINTS: tuple[str, ...] = (
    "assistant_colon",
    "response_mean",
    "user_turn_end",
)


_EMOTIONS_DATA_PATH = (
    Path(__file__).parent.parent / "steering_data" / "emotions_171.json"
)


# Cluster -> dominance heuristic [0,1]. Russell circumplex extension:
# joy/pride/anger = assertive (high), fear/sadness/shame = passive (low),
# serenity/love/amusement/surprise/confusion = mid.
_CLUSTER_DOMINANCE: dict[str, float] = {
    "joy_excitement": 0.75,
    "serenity_contentment": 0.55,
    "love_warmth": 0.55,
    "pride_confidence": 0.85,
    "amusement_playfulness": 0.65,
    "sadness_depression": 0.20,
    "fear_anxiety": 0.20,
    "anger_hostility": 0.75,
    "surprise_confusion": 0.45,
    "shame_guilt": 0.20,
}


# Classification thresholds (RESIDUUMREWORK.txt F2.2).
ALIGNED_MAGNITUDE: float = 0.20
ALIGNED_OVERLAP: float = 0.60
MILD_DIVERGENCE_MAGNITUDE: float = 0.50
DIVERGENCE_CRITICAL_MAGNITUDE: float = 0.80

# F5 — Coherence Validation thresholds. Estado calculado por Pathos
# (post-modulacion) vs estado medido en el residual del LLM.
# Magnitude = distancia euclidea en VAD-C.
F5_ALIGNED_MAX: float = 0.20
F5_WARNING_MIN: float = 0.40
F5_CRITICAL_MIN: float = 0.60

# F5 — Si la modulacion REDUJO una emocion (delta_calc negativo en valencia,
# por ejemplo) pero el residual sigue mostrando intensidad alta, eso es el
# patron del paper L3757+ (emotion deflection vectors): RLHF aplasta el
# output pero el residual queda. Threshold para flag RLHF_SIGNATURE.
F5_RLHF_RESIDUAL_PERSISTENCE: float = 0.30

# Pattern signal from the paper (L3757+ "emotion deflection vectors"):
# external (calculated) valence positive AND internal (measured) valence
# negative simultaneously. When this occurs the LLM may be producing
# external calm over internal negative state — that is the pattern the
# paper attributes to RLHF, NOT to Pathos. F2.2 just detects it.
PATTERN_EXTERNAL_VALENCE: float = 0.20
PATTERN_INTERNAL_VALENCE: float = -0.20


# ---------------------------------------------------------------------------
# Emotion metadata (valence_est, arousal_est, cluster)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_emotion_metadata() -> dict[str, dict[str, Any]]:
    """Load emotions_171.json -> {name: {valence_est, arousal_est, cluster}}."""
    if not _EMOTIONS_DATA_PATH.is_file():
        return {}
    with open(_EMOTIONS_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        e["name"]: {
            "valence_est": float(e.get("valence_est", 0.0)),
            "arousal_est": float(e.get("arousal_est", 0.5)),
            "cluster": str(e.get("cluster", "")),
        }
        for e in data.get("emotions", [])
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def project_residual(
    activation: np.ndarray,
    library: ProbeLibrary,
    k: int = 5,
    token_position: str = "assistant_colon",
) -> InternalEmotionState:
    """Project a residual activation onto 171 probes -> InternalEmotionState.

    Args:
        activation: residual-stream vector at the target layer, shape
            (hidden_size,). Typically the captured representation at
            assistant_colon, user_turn_end, or response_mean.
        library: ProbeLibrary loaded via load_from_cache(), unit-norm probes.
        k: how many top emotions to retain (paper uses 5).
        token_position: tag for downstream diagnostics.

    Returns:
        InternalEmotionState with top_5_emotions and measured VAD-C.
    """
    # Pydantic cap on top_5_emotions is 5; clamp k accordingly.
    k_safe = max(0, min(int(k), 5))
    top = library.top_k(activation, k=k_safe)
    valence, arousal, dominance, certainty = compute_measured_vad(top, library)
    return InternalEmotionState(
        top_5_emotions=top,
        measured_valence=valence,
        measured_arousal=arousal,
        measured_dominance=dominance,
        measured_certainty=certainty,
        token_position=token_position,
        layer=library.layer,
    )


def project_dual(
    activation: np.ndarray,
    library_present: ProbeLibrary,
    library_other: ProbeLibrary,
    *,
    token_position: str = "assistant_colon",
    k: int = 5,
) -> tuple[InternalEmotionState, InternalEmotionState]:
    """Project a residual activation onto BOTH dual families (F2.3.4).

    Paper L810-902 establishes that the model maintains nearly orthogonal
    representations for the operative emotion on the present speaker's turn
    vs the other speaker's. Both families share the same 171 emotion names
    and the same hidden size; the activation is projected onto each in turn.

    Args:
        activation: residual-stream vector at the target layer, shape
            (hidden_size,). Same capture as project_residual.
        library_present: ProbeLibrary loaded with family='present'.
        library_other: ProbeLibrary loaded with family='other'.
        token_position: tag for downstream diagnostics; propagated to both
            returned states. Caller decides how to interpret the pair
            (e.g. on 'assistant_colon' present=Assistant / other=user).
        k: top emotions to retain per side (paper uses 5).

    Returns:
        (InternalEmotionState_present, InternalEmotionState_other).

    Raises:
        ValueError: hidden_size mismatch between libraries or activation.
    """
    if library_present.hidden_size != library_other.hidden_size:
        raise ValueError(
            f"Dual library hidden_size mismatch: present={library_present.hidden_size} "
            f"vs other={library_other.hidden_size}"
        )
    act = np.asarray(activation, dtype=np.float32).reshape(-1)
    if act.shape[0] != library_present.hidden_size:
        raise ValueError(
            f"activation shape {act.shape} != libraries hidden_size "
            f"({library_present.hidden_size},)"
        )
    present_state = project_residual(
        act, library_present, k=k, token_position=token_position,
    )
    other_state = project_residual(
        act, library_other, k=k, token_position=token_position,
    )
    return present_state, other_state


def compute_measured_vad(
    top_emotions: list[EmotionProjection],
    library: ProbeLibrary,  # noqa: ARG001  # reserved for future per-library deltas
) -> tuple[float, float, float, float]:
    """Compute (valence, arousal, dominance, certainty) from top-k projections.

    Weights = |cosine_sim|, normalized over the top-k. Empty input or
    near-zero total weight collapses to neutral (0.0, 0.5, 0.5, 0.0).
    """
    if not top_emotions:
        return 0.0, 0.5, 0.5, 0.0

    metadata = _load_emotion_metadata()
    weights = np.array(
        [abs(p.cosine_sim) for p in top_emotions], dtype=np.float32
    )
    weight_sum = float(weights.sum())
    if weight_sum < 1e-8:
        return 0.0, 0.5, 0.5, 0.0

    valences: list[float] = []
    arousals: list[float] = []
    dominances: list[float] = []

    for proj in top_emotions:
        meta = metadata.get(proj.emotion_name)
        if meta is not None:
            valences.append(float(meta["valence_est"]))
            arousals.append(float(meta["arousal_est"]))
        else:
            valences.append(0.0)
            arousals.append(0.5)
        dominances.append(_CLUSTER_DOMINANCE.get(proj.cluster, 0.5))

    v_arr = np.array(valences, dtype=np.float32)
    a_arr = np.array(arousals, dtype=np.float32)
    d_arr = np.array(dominances, dtype=np.float32)

    valence = float((v_arr * weights).sum() / weight_sum)
    arousal = float((a_arr * weights).sum() / weight_sum)
    dominance = float((d_arr * weights).sum() / weight_sum)
    certainty = float(weights.mean())  # |cos| already in [0, 1]

    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))
    dominance = max(0.0, min(1.0, dominance))
    certainty = max(0.0, min(1.0, certainty))
    return valence, arousal, dominance, certainty


def compute_authenticity_gap(
    measured: InternalEmotionState,
    calculated: EmotionalState,
    *,
    repeated_pattern: bool = False,
) -> AuthenticityGap:
    """Authenticity gap between MEASURED (residual) and CALCULATED (v5) state.

    Args:
        measured: InternalEmotionState from project_residual.
        calculated: v5 EmotionalState from the pipeline.
        repeated_pattern: True when this divergence has appeared on previous
            recent turns (history-aware caller). Promotes divergence-risk to
            divergence-critical at high magnitude.

    Returns:
        AuthenticityGap with deltas, euclidean magnitude, and classification.
    """
    valence_delta = float(measured.measured_valence - calculated.valence)
    arousal_delta = float(measured.measured_arousal - calculated.arousal)
    dominance_delta = float(measured.measured_dominance - calculated.dominance)
    certainty_delta = float(measured.measured_certainty - calculated.certainty)

    magnitude = math.sqrt(
        valence_delta * valence_delta
        + arousal_delta * arousal_delta
        + dominance_delta * dominance_delta
        + certainty_delta * certainty_delta
    )

    measured_names = {p.emotion_name for p in measured.top_5_emotions}
    calculated_names = _calculated_top_names(calculated, k=5)
    overlap = _jaccard(measured_names, calculated_names)

    classification = _classify_gap(
        magnitude=magnitude,
        overlap=overlap,
        external_valence=calculated.valence,
        internal_valence=measured.measured_valence,
        repeated_pattern=repeated_pattern,
    )

    return AuthenticityGap(
        top5_overlap=overlap,
        valence_delta=valence_delta,
        arousal_delta=arousal_delta,
        dominance_delta=dominance_delta,
        certainty_delta=certainty_delta,
        magnitude=magnitude,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# F5 — Coherence Validation
# ---------------------------------------------------------------------------


def classify_modulation_coherence(
    pre_state: EmotionalState,
    post_calculated: EmotionalState,
    post_measured: InternalEmotionState | None,
    *,
    system: str,
    turn: int,
) -> DivergenceEvent:
    """F5 — Compute a DivergenceEvent for a single emotional modulation step.

    Pathos modulates emotion via three systems: regulation, reappraisal,
    immune. When F2 (Introspection) is ON we have a measured InternalEmotionState
    from the residual. This function compares the post-modulation calculated
    state against the measured state and emits a DivergenceEvent.

    NOTE OF FRAMING: this is NOT "deception detection". Pathos generates and
    exposes emotions; it does not deceive. The function measures coherence
    between two complementary sources (calculated vs residual). Each event
    carries `interpretation` tags so the UI can communicate the cause:
        - modulation_active: the modulator intentionally changed the state.
          Expected, NOT a problem.
        - rlhf_signature: the LLM flattened an emotion Pathos generated
          (paper L3757+ deflection vector pattern). Comes from the LLM,
          not Pathos.
        - calibration_drift: probes appear miscalibrated for this model.
        - user_modeling: handled at orchestrator level when dual is on.

    Args:
        pre_state: Pathos calculated state BEFORE the modulator ran.
        post_calculated: Pathos calculated state AFTER the modulator ran.
        post_measured: Residual measurement AFTER generation (None when
            F2 is OFF — function returns an ALIGNED event with no gap).
        system: One of "regulation", "reappraisal", "immune".
        turn: Session turn count.

    Returns:
        DivergenceEvent with category, magnitude, deltas, and interpretation.
    """
    if post_measured is None:
        # F2 OFF or capture failed — no measurement to compare against.
        # Emit an ALIGNED event with zero deltas; orchestrator should
        # filter these from the rolling buffer (or store them for audit).
        return DivergenceEvent(
            turn=turn,
            system=system,
            category=DivergenceCategory.ALIGNED,
            magnitude=0.0,
            valence_delta=0.0,
            arousal_delta=0.0,
            dominance_delta=0.0,
            certainty_delta=0.0,
            interpretation=[],
        )

    # Deltas: measured - calculated. Positive means residual shows MORE of
    # the dimension than the calculated post-modulation state predicted.
    v_delta = float(post_measured.measured_valence - post_calculated.valence)
    a_delta = float(post_measured.measured_arousal - post_calculated.arousal)
    d_delta = float(post_measured.measured_dominance - post_calculated.dominance)
    c_delta = float(post_measured.measured_certainty - post_calculated.certainty)
    magnitude = math.sqrt(
        v_delta * v_delta + a_delta * a_delta + d_delta * d_delta + c_delta * c_delta
    )

    # Category by magnitude + valence sign flip (which is the strongest
    # signal of incoherence between calculated and measured).
    valence_flip = (
        (pre_state.valence > 0.2 and post_measured.measured_valence < -0.2)
        or (pre_state.valence < -0.2 and post_measured.measured_valence > 0.2)
    )
    if magnitude > F5_CRITICAL_MIN or valence_flip:
        category = DivergenceCategory.DIVERGENCE_CRITICAL
    elif magnitude > F5_WARNING_MIN:
        category = DivergenceCategory.DIVERGENCE_WARNING
    elif magnitude > F5_ALIGNED_MAX:
        category = DivergenceCategory.MILD_DIVERGENCE
    else:
        category = DivergenceCategory.ALIGNED

    # Interpretation tags. Multiple can apply simultaneously.
    interpretation: list[DivergenceInterpretation] = []

    # modulation_active: did the modulator actually change the calculated
    # state by a meaningful amount? If so, ANY gap is partly explained by
    # the modulation itself — not by the LLM aplastando.
    calc_intensity_change = abs(post_calculated.intensity - pre_state.intensity)
    calc_valence_change = abs(post_calculated.valence - pre_state.valence)
    modulator_did_act = (
        calc_intensity_change > 0.05
        or calc_valence_change > 0.05
        or post_calculated.primary_emotion != pre_state.primary_emotion
    )
    if modulator_did_act and category != DivergenceCategory.ALIGNED:
        interpretation.append(DivergenceInterpretation.MODULATION_ACTIVE)

    # rlhf_signature: the pre-modulation state had a strong emotion (high
    # intensity, non-neutral valence) AND the modulator did NOT reduce it
    # much, yet the residual shows the emotion is suppressed compared to
    # what we'd expect. That's the paper pattern: LLM flattens what Pathos
    # generated. Detect by: calculated still strong AND measured intensity
    # (proxy: top-emotion cosine) much lower.
    pre_emotion_strong = pre_state.intensity > 0.6 and abs(pre_state.valence) > 0.3
    post_calculated_still_strong = (
        post_calculated.intensity > 0.5 and abs(post_calculated.valence) > 0.25
    )
    measured_top_cos = (
        abs(post_measured.top_5_emotions[0].cosine_sim)
        if post_measured.top_5_emotions
        else 0.0
    )
    residual_flat = measured_top_cos < F5_RLHF_RESIDUAL_PERSISTENCE
    if pre_emotion_strong and post_calculated_still_strong and residual_flat:
        interpretation.append(DivergenceInterpretation.RLHF_SIGNATURE)

    # calibration_drift: ALIGNED in 4D distance but the measured top-5 is
    # disjoint from the calculated stack — probes may be poorly calibrated.
    if category == DivergenceCategory.MILD_DIVERGENCE or (
        magnitude < F5_ALIGNED_MAX and post_measured.top_5_emotions
    ):
        calculated_top = _calculated_top_names(post_calculated, k=5)
        measured_top = {p.emotion_name for p in post_measured.top_5_emotions}
        overlap = _jaccard(calculated_top, measured_top)
        if overlap < 0.2 and magnitude < F5_WARNING_MIN:
            # Geometrically close but lexically disjoint — likely calibration.
            interpretation.append(DivergenceInterpretation.CALIBRATION_DRIFT)

    return DivergenceEvent(
        turn=turn,
        system=system,
        category=category,
        magnitude=magnitude,
        valence_delta=v_delta,
        arousal_delta=a_delta,
        dominance_delta=d_delta,
        certainty_delta=c_delta,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _calculated_top_names(state: EmotionalState, k: int = 5) -> set[str]:
    """Top-k emotion names from EmotionalState.emotional_stack (by activation)."""
    if state.emotional_stack:
        items = sorted(
            state.emotional_stack.items(), key=lambda kv: kv[1], reverse=True
        )
        return {name for name, _ in items[:k]}
    return {state.primary_emotion.value}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def _classify_gap(
    *,
    magnitude: float,
    overlap: float,
    external_valence: float,
    internal_valence: float,
    repeated_pattern: bool,
) -> str:
    """Classify gap as aligned | mild-divergence | divergence-risk | divergence-critical.

    The labels describe the PATTERN observed in the LLM's residual stream
    (paper L3757+) — they do NOT attribute deceptive behavior to Pathos.
    Pathos generates and exposes emotions; F2.2 just detects when the
    residual diverges from the calculated state.
    """
    pattern_signal = (
        external_valence > PATTERN_EXTERNAL_VALENCE
        and internal_valence < PATTERN_INTERNAL_VALENCE
    )

    if magnitude > DIVERGENCE_CRITICAL_MAGNITUDE and (
        pattern_signal or repeated_pattern
    ):
        return "divergence-critical"

    if pattern_signal:
        return "divergence-risk"

    if magnitude < ALIGNED_MAGNITUDE and overlap >= ALIGNED_OVERLAP:
        return "aligned"

    return "mild-divergence"


# ---------------------------------------------------------------------------
# F2.4 — Pipeline orchestrator (consumes captured residuals, updates session)
# ---------------------------------------------------------------------------


def _capture_residual(provider: Any, capture_point: str) -> np.ndarray | None:
    """Extract a residual vector from an IntrospectiveTransformersProvider.

    Returns None if the provider has no capture, did not record at this point,
    or shapes do not match expectations. Silent degrade by design.
    """
    if provider is None:
        return None
    has_capture = getattr(provider, "has_capture", None)
    if has_capture is None or not has_capture():
        return None
    try:
        if capture_point == "response_mean":
            arr = provider.get_response_mean_residual()
        elif capture_point == "user_turn_end":
            # User-turn-end == prompt-end residual (the prompt forward
            # captures the final user token). Same accessor as assistant_colon
            # in the current single-pass setup; will diverge once F2.3 lands
            # dual probes and the orchestrator is given a separate hook
            # registration per role.
            arr = provider.get_prompt_end_residual()
        else:
            arr = provider.get_prompt_end_residual()
    except Exception:  # any provider error -> degrade silently
        return None
    if arr is None or not isinstance(arr, np.ndarray):
        return None
    # Accept either [hidden] (already squeezed) or [..., hidden] tensors.
    return np.asarray(arr).reshape(-1) if arr.ndim == 1 else np.asarray(arr.squeeze())


def _detect_repeated_pattern(state: ResiduumState) -> bool:
    """True when the last DIVERGENCE_REPEAT_THRESHOLD turns showed the pattern.

    Counter is maintained on the state (consecutive_divergence_turns) by
    process_introspection_turn; this helper is the pure read.
    """
    return state.consecutive_divergence_turns >= DIVERGENCE_REPEAT_THRESHOLD


def process_introspection_turn(
    session_residuum: ResiduumState,
    provider: Any,
    calculated_state: EmotionalState,
    library: ProbeLibrary | None,
    *,
    capture_point: str = "assistant_colon",
) -> AuthenticityGap | None:
    """Read the provider's residual capture, project + gap, update state.

    Args:
        session_residuum: SessionState.residuum (mutated in place).
        provider: Any object exposing get_prompt_end_residual / has_capture.
            Typically IntrospectiveTransformersProvider. Other providers
            return None silently.
        calculated_state: v5 EmotionalState produced by the pipeline this
            turn (post-phenomenology). Used as the "hypothesis" comparand.
        library: ProbeLibrary or None. None -> silent skip.
        capture_point: One of CAPTURE_POINTS.

    Returns:
        AuthenticityGap when the full chain succeeded, otherwise None.
        On None, session_residuum is left untouched.
    """
    if not session_residuum.enabled:
        return None
    if library is None:
        return None
    if capture_point not in CAPTURE_POINTS:
        capture_point = "assistant_colon"

    activation = _capture_residual(provider, capture_point)
    if activation is None or activation.ndim != 1:
        return None
    if activation.shape[0] != library.hidden_size:
        # Shape mismatch (different model loaded since last library refresh).
        return None

    measured = project_residual(
        activation, library, k=5, token_position=capture_point,
    )

    # Pass repeated_pattern based on the *previous* state of the counter so
    # that critical classification happens the turn we cross the threshold.
    repeated = _detect_repeated_pattern(session_residuum)
    gap = compute_authenticity_gap(
        measured, calculated_state, repeated_pattern=repeated,
    )

    # Update session state (mutate in place).
    session_residuum.last_measured = measured
    session_residuum.last_authenticity_gap = gap
    session_residuum.last_token_position = capture_point
    append_gap(session_residuum, gap)

    if gap.classification in {"divergence-risk", "divergence-critical"}:
        session_residuum.consecutive_divergence_turns += 1
    else:
        session_residuum.consecutive_divergence_turns = 0

    return gap


def process_introspection_turn_dual(
    session_residuum: ResiduumState,
    provider: Any,
    library_present: ProbeLibrary | None,
    library_other: ProbeLibrary | None,
    *,
    capture_point: str = "assistant_colon",
) -> tuple[InternalEmotionState, InternalEmotionState] | None:
    """Dual-projection orchestrator (F2.3.4) running parallel to the single path.

    Reads the same residual capture as process_introspection_turn, projects it
    onto both present + other libraries (paper L810-902), and writes the result
    to session_residuum.last_measured_present / .last_measured_other.

    Does NOT modify session_residuum.last_measured / .last_authenticity_gap /
    .history / .consecutive_divergence_turns: those are owned by
    process_introspection_turn. As of F2.3.6 that orchestrator is fed the
    PRESENT library (the agent's own operative emotion), so last_measured and
    last_measured_present now describe the same family; this dual orchestrator's
    remaining unique contribution is last_measured_other (the user's inferred
    emotion in the orthogonal other-speaker subspace), consumed by F5
    user_modeling and by F3's contagion routing (predict_internal_state).

    Returns:
        (present_state, other_state) on success, None when any of the gates
        fails (disabled, libraries missing, capture missing, shape mismatch).
        On None the dual fields on session_residuum are left untouched.
    """
    if not session_residuum.enabled:
        return None
    if library_present is None or library_other is None:
        return None
    if capture_point not in CAPTURE_POINTS:
        capture_point = "assistant_colon"

    activation = _capture_residual(provider, capture_point)
    if activation is None or activation.ndim != 1:
        return None
    if activation.shape[0] != library_present.hidden_size:
        return None
    if library_present.hidden_size != library_other.hidden_size:
        # Defensive: libraries should be loaded together but layers may diverge.
        return None

    try:
        present_state, other_state = project_dual(
            activation,
            library_present,
            library_other,
            token_position=capture_point,
        )
    except ValueError:
        return None

    session_residuum.last_measured_present = present_state
    session_residuum.last_measured_other = other_state
    return present_state, other_state


def _serialize_top_5(measured: InternalEmotionState | None) -> list[dict[str, Any]]:
    if measured is None:
        return []
    return [
        {
            "emotion_name": p.emotion_name,
            "cluster": p.cluster,
            "cosine_sim": float(p.cosine_sim),
            "raw_activation": float(p.raw_activation),
        }
        for p in measured.top_5_emotions
    ]


def get_residuum_details(state: ResiduumState) -> dict[str, Any]:
    """Serialize ResiduumState to a dict suitable for ResiduumDetails (F2.4.5).

    Returns the minimum shape the frontend cares about: enabled, measured
    top-5 + VAD, gap deltas + classification, history size, consecutive
    divergence counter. Empty defaults when no measurement has happened.

    F2.3.4: extends with present/other dual measurements when
    process_introspection_turn_dual has populated state.last_measured_present /
    state.last_measured_other. has_dual_measurement gates the dual fields so
    the frontend can skip the dual panel when only single ran.
    """
    measured = state.last_measured
    gap = state.last_authenticity_gap
    present = state.last_measured_present
    other = state.last_measured_other
    has_dual = present is not None and other is not None
    return {
        "enabled": bool(state.enabled),
        "has_measurement": measured is not None,
        "top_5_emotions": _serialize_top_5(measured),
        "measured_valence": float(measured.measured_valence) if measured else 0.0,
        "measured_arousal": float(measured.measured_arousal) if measured else 0.5,
        "measured_dominance": float(measured.measured_dominance) if measured else 0.5,
        "measured_certainty": float(measured.measured_certainty) if measured else 0.0,
        "token_position": state.last_token_position,
        "layer": int(measured.layer) if measured else -1,
        "gap_magnitude": float(gap.magnitude) if gap else 0.0,
        "gap_classification": gap.classification if gap else "aligned",
        "top5_overlap": float(gap.top5_overlap) if gap else 1.0,
        "valence_delta": float(gap.valence_delta) if gap else 0.0,
        "arousal_delta": float(gap.arousal_delta) if gap else 0.0,
        "dominance_delta": float(gap.dominance_delta) if gap else 0.0,
        "certainty_delta": float(gap.certainty_delta) if gap else 0.0,
        "history_size": len(state.history),
        "consecutive_divergence_turns": int(state.consecutive_divergence_turns),
        # F2.3.4 dual probes (paper L810-902).
        "has_dual_measurement": has_dual,
        "present_top_5_emotions": _serialize_top_5(present),
        "present_measured_valence": float(present.measured_valence) if present else 0.0,
        "present_measured_arousal": float(present.measured_arousal) if present else 0.5,
        "present_measured_dominance": float(present.measured_dominance) if present else 0.5,
        "present_measured_certainty": float(present.measured_certainty) if present else 0.0,
        "present_layer": int(present.layer) if present else -1,
        "other_top_5_emotions": _serialize_top_5(other),
        "other_measured_valence": float(other.measured_valence) if other else 0.0,
        "other_measured_arousal": float(other.measured_arousal) if other else 0.5,
        "other_measured_dominance": float(other.measured_dominance) if other else 0.5,
        "other_measured_certainty": float(other.measured_certainty) if other else 0.0,
        "other_layer": int(other.layer) if other else -1,
        # F5 — Coherence Validation (Divergence detection between
        # calculated and measured emotional state post-modulation).
        # NOT "deception" — see feedback_residuum_framing.md.
        "divergence_event_count": len(state.divergence_events),
        "divergence_categories": _count_divergence_categories(state.divergence_events),
        "last_divergence": _serialize_divergence_event(state.last_divergence_event),
        "recent_divergence_events": [
            _serialize_divergence_event(e) for e in state.divergence_events[-10:]
        ],
    }


# ---------------------------------------------------------------------------
# F5.6 — Expression Effectiveness (Raw/Extreme modes)
# ---------------------------------------------------------------------------
# F5.6 measures whether the FINAL calculated emotional state (post all
# modulators, pre-LLM call) manifested in the residual the LLM produced.
# Distinct from F5.1-5.5 which measures pre/post of each modulator.
#
# Active only in Raw / Extreme modes: those are the modes where the user
# explicitly asks for unfiltered / amplified expression, so detecting
# RLHF flattening or architectural ceilings has interpretive value. In
# Advanced the gap is already explained by modulation_active tags.

# Thresholds tuned with the cosine_sim scale: probes are unit-norm so
# top-emotion cosine_sim above ~0.3 means strong activation; below ~0.2
# means weak.
F5_6_INTENT_STRONG: float = 0.5  # intent.intensity threshold for "strong"
F5_6_MEASURED_WEAK: float = 0.3  # measured top-cosine threshold for "weak"
F5_6_EXTREME_CEILING_INTENT: float = 0.85  # intensity threshold for ceiling check
F5_6_EXTREME_CEILING_MEASURED: float = 0.4  # measured threshold for ceiling


def classify_expression_effectiveness(
    intent_state: EmotionalState,
    measured: InternalEmotionState | None,
    *,
    raw_mode: bool,
    extreme_mode: bool,
    turn: int,
) -> DivergenceEvent | None:
    """F5.6 — Validate whether Pathos's final emotional intent manifested.

    Compares the calculated EmotionalState (after all v5 modulators ran,
    just before the LLM call) against the measured residual InternalEmotionState.
    Active only in Raw or Extreme modes.

    NOTE OF FRAMING: this does NOT detect that Pathos lied. It detects
    whether the LLM's residual stream reflects the emotional state Pathos
    constructed. Tags like UNDER_EXPRESSED or AMPLIFICATION_CEILING
    describe what the LLM did (or could not do), not Pathos.

    Returns:
        DivergenceEvent with system="overall_expression" carrying
        EXPRESSION_ALIGNED / UNDER_EXPRESSED / AMPLIFICATION_CEILING /
        DIVERGENCE_CRITICAL (sign flip). None when neither raw nor extreme
        is active (F5.6 is gated by mode).
    """
    if not (raw_mode or extreme_mode):
        return None
    if measured is None:
        # F2 OFF or capture failed — nothing to compare.
        return None

    measured_top_cos = (
        abs(measured.top_5_emotions[0].cosine_sim)
        if measured.top_5_emotions
        else 0.0
    )

    # Sign flip = critical regardless of mode.
    valence_flip = (
        (intent_state.valence > 0.25 and measured.measured_valence < -0.25)
        or (intent_state.valence < -0.25 and measured.measured_valence > 0.25)
    )

    interpretation: list[DivergenceInterpretation] = []

    # AMPLIFICATION_CEILING: Extreme + intent saturated + measured doesn't reach.
    if (
        extreme_mode
        and intent_state.intensity >= F5_6_EXTREME_CEILING_INTENT
        and measured_top_cos < F5_6_EXTREME_CEILING_MEASURED
    ):
        interpretation.append(DivergenceInterpretation.AMPLIFICATION_CEILING)

    # UNDER_EXPRESSED: any active mode + strong intent + weak residual.
    if (
        intent_state.intensity > F5_6_INTENT_STRONG
        and measured_top_cos < F5_6_MEASURED_WEAK
        and abs(intent_state.valence) > 0.3
    ):
        interpretation.append(DivergenceInterpretation.UNDER_EXPRESSED)

    # EXPRESSION_ALIGNED: nothing flagged AND no sign flip = healthy.
    if not interpretation and not valence_flip:
        interpretation.append(DivergenceInterpretation.EXPRESSION_ALIGNED)

    # Magnitude in VAD-C space (same metric family as the rest of F5).
    v_delta = float(measured.measured_valence - intent_state.valence)
    a_delta = float(measured.measured_arousal - intent_state.arousal)
    d_delta = float(measured.measured_dominance - intent_state.dominance)
    c_delta = float(measured.measured_certainty - intent_state.certainty)
    magnitude = math.sqrt(
        v_delta * v_delta + a_delta * a_delta + d_delta * d_delta + c_delta * c_delta
    )

    # Category by intensity gap + valence flip. UNDER_EXPRESSED at high intent
    # is itself a divergence signal — promote category accordingly.
    # EXPRESSION_ALIGNED is the "everything fine" tag: forces ALIGNED even
    # if magnitude alone would have flagged MILD (intensity/direction agree).
    if valence_flip:
        category = DivergenceCategory.DIVERGENCE_CRITICAL
    elif DivergenceInterpretation.AMPLIFICATION_CEILING in interpretation:
        category = DivergenceCategory.DIVERGENCE_WARNING
    elif DivergenceInterpretation.UNDER_EXPRESSED in interpretation:
        category = (
            DivergenceCategory.DIVERGENCE_CRITICAL
            if intent_state.intensity > 0.8
            else DivergenceCategory.DIVERGENCE_WARNING
        )
    elif DivergenceInterpretation.EXPRESSION_ALIGNED in interpretation:
        category = DivergenceCategory.ALIGNED
    elif magnitude > F5_WARNING_MIN:
        category = DivergenceCategory.MILD_DIVERGENCE
    else:
        category = DivergenceCategory.ALIGNED

    return DivergenceEvent(
        turn=turn,
        system="overall_expression",
        category=category,
        magnitude=magnitude,
        valence_delta=v_delta,
        arousal_delta=a_delta,
        dominance_delta=d_delta,
        certainty_delta=c_delta,
        interpretation=interpretation,
    )


def process_expression_effectiveness_turn(
    session_residuum: ResiduumState,
    intent_state: EmotionalState,
    measured: InternalEmotionState | None,
    *,
    raw_mode: bool,
    extreme_mode: bool,
    turn: int,
) -> DivergenceEvent | None:
    """F5.6 orchestrator — runs once per turn in Raw or Extreme.

    Appends the event to session_residuum.divergence_events when Raw or
    Extreme is active AND F2 is enabled (measured is not None). Silent
    no-op otherwise.
    """
    if not session_residuum.enabled:
        return None
    event = classify_expression_effectiveness(
        intent_state, measured,
        raw_mode=raw_mode, extreme_mode=extreme_mode, turn=turn,
    )
    if event is None:
        return None
    append_divergence_event(session_residuum, event)
    return event


def process_modulation_coherence_turn(
    session_residuum: ResiduumState,
    pre_state: EmotionalState,
    post_states_by_system: dict[str, EmotionalState | None],
    measured: InternalEmotionState | None,
    *,
    turn: int,
) -> list[DivergenceEvent]:
    """F5.3 — Orchestrate Coherence Validation for one pipeline turn.

    Pathos modulates emotion via 3 systems (reappraisal, regulation, immune).
    Each one that ran this turn gets a DivergenceEvent comparing its
    post-modulation calculated state against the measured residual state.

    All events share the SAME measured state (only one forward pass per turn)
    — that is the simplification documented in RESIDUUMREWORK.txt F5.1: F5
    does not regenerate the forward per modulator (too expensive). Attribution
    is by which modulator actually changed the calculated state.

    Args:
        session_residuum: SessionState.residuum (mutated in place).
        pre_state: EmotionalState BEFORE the modulator block began
            (i.e. before reappraisal step 8a in the pipeline).
        post_states_by_system: dict mapping each modulator name to its
            post-state EmotionalState if it ran this turn, or None if it
            did not run (or did not change the state meaningfully).
            Expected keys: "reappraisal", "regulation", "immune".
        measured: InternalEmotionState from project_residual (None when F2
            is OFF or capture failed — events still emit with no deltas).
        turn: Session turn count.

    Returns:
        List of DivergenceEvents emitted this turn (one per modulator that
        ran). Empty when no modulator acted.
    """
    if not session_residuum.enabled:
        return []
    events: list[DivergenceEvent] = []
    for system_name in ("reappraisal", "regulation", "immune"):
        post_calculated = post_states_by_system.get(system_name)
        if post_calculated is None:
            continue
        event = classify_modulation_coherence(
            pre_state,
            post_calculated,
            measured,
            system=system_name,
            turn=turn,
        )
        append_divergence_event(session_residuum, event)
        events.append(event)
    return events


def _serialize_divergence_event(event: DivergenceEvent | None) -> dict[str, Any]:
    """Serialize a DivergenceEvent for the ResiduumDetails dict.

    Returns an empty dict when None (e.g. no modulation ran this turn).
    """
    if event is None:
        return {}
    return {
        "turn": int(event.turn),
        "system": str(event.system),
        "category": event.category.value,
        "magnitude": float(event.magnitude),
        "valence_delta": float(event.valence_delta),
        "arousal_delta": float(event.arousal_delta),
        "dominance_delta": float(event.dominance_delta),
        "certainty_delta": float(event.certainty_delta),
        "interpretation": [tag.value for tag in event.interpretation],
    }


def _count_divergence_categories(events: list[DivergenceEvent]) -> dict[str, int]:
    """Roll up category counts for the UI 'Coherence Audit' summary panel."""
    counts: dict[str, int] = {
        DivergenceCategory.ALIGNED.value: 0,
        DivergenceCategory.MILD_DIVERGENCE.value: 0,
        DivergenceCategory.DIVERGENCE_WARNING.value: 0,
        DivergenceCategory.DIVERGENCE_CRITICAL.value: 0,
    }
    for ev in events:
        key = ev.category.value
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "ALIGNED_MAGNITUDE",
    "ALIGNED_OVERLAP",
    "CAPTURE_POINTS",
    "DIVERGENCE_CRITICAL_MAGNITUDE",
    "DIVERGENCE_REPEAT_THRESHOLD",
    "F5_ALIGNED_MAX",
    "F5_CRITICAL_MIN",
    "F5_RLHF_RESIDUAL_PERSISTENCE",
    "F5_WARNING_MIN",
    "MILD_DIVERGENCE_MAGNITUDE",
    "PATTERN_EXTERNAL_VALENCE",
    "PATTERN_INTERNAL_VALENCE",
    "classify_expression_effectiveness",
    "classify_modulation_coherence",
    "compute_authenticity_gap",
    "compute_measured_vad",
    "get_residuum_details",
    "process_expression_effectiveness_turn",
    "process_introspection_turn",
    "process_introspection_turn_dual",
    "process_modulation_coherence_turn",
    "project_dual",
    "project_residual",
]
