"""RESIDUUM — Modelos de datos para Probe Library e Introspection.

Pilar 8 (v6): Funcional emotion vectors como direcciones lineales medibles en
el residual stream de un LLM. Basado en Lindsey/Sofroniew et al. 2026
("Emotion Concepts and their Function in a Large Language Model", Anthropic).

Fase 1 (Probe Library):
    - ProbeMetadata: metadatos por emocion extraida
    - ProbeLibrary: coleccion de 171 probes + neutral PCs para proyeccion
    - EmotionProjection: coseno por emocion en un instante
    - InternalEmotionState: estado emocional MEDIDO desde el residual
    - AuthenticityGap: distancia entre estado medido y calculado (F2.2)

Los arrays numpy no se serializan en los modelos Pydantic (los probes se
cachean en NPZ aparte). ProbeLibrary guarda referencias a los arrays in-memory
y se instancia a partir de load_from_cache(). Ver steering_extract.py (F1.2)
para la extraccion y steering.py (F1.3) para la carga.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# F1.2 — Metadata por probe extraido
# ---------------------------------------------------------------------------


class ProbeMetadata(BaseModel):
    """Metadatos de un probe de emocion individual.

    Se serializa en el NPZ junto al array. La norma pre-proyeccion permite
    detectar probes degenerados; neutral_pcs_removed valida que la proyeccion
    fuera del subespacio neutral efectivamente corrio.
    """

    emotion_name: str
    cluster: str
    layer: int
    dimension: int = Field(description="hidden_size del modelo extraido")
    norm_before_projection: float = Field(
        description="L2 norm del probe antes de proyectar fuera PCs neutrales"
    )
    norm_after_projection: float = Field(
        description="L2 norm del probe despues de proyectar fuera PCs neutrales (pre-normalizacion)"
    )
    neutral_pcs_removed: int = Field(description="Numero de PCs neutrales proyectadas fuera")
    source_stories_count: int = Field(description="Cuantas stories aportaron al promedio")


# ---------------------------------------------------------------------------
# F1.3 — Probe Library (runtime, in-memory)
# ---------------------------------------------------------------------------


class ProbeLibraryInfo(BaseModel):
    """Vista serializable de una ProbeLibrary para /research y diagnosticos.

    No incluye los arrays (solo conteos y metadatos agregados). La version
    con arrays vive en steering.py como clase separada.
    """

    model_id: str
    layer: int
    hidden_size: int
    num_probes: int
    num_neutral_pcs: int
    extracted_at: str = Field(description="Timestamp ISO 8601 UTC")
    source_stories_count: int = Field(
        description="Total de stories usadas (171 * stories_per_emotion)"
    )
    status: str = Field(default="ready", description="ready | missing | degraded")


# ---------------------------------------------------------------------------
# F2.2 — Proyeccion individual y estado medido
# ---------------------------------------------------------------------------


class EmotionProjection(BaseModel):
    """Proyeccion de una activacion residual sobre UN probe de emocion.

    cosine_sim: coseno entre activacion y probe unitario ([-1, 1]).
    raw_activation: producto punto sin normalizar (magnitud * cos).
    """

    emotion_name: str
    cluster: str
    cosine_sim: float = Field(ge=-1.0, le=1.0)
    raw_activation: float


class InternalEmotionState(BaseModel):
    """Estado emocional MEDIDO desde el residual stream.

    Source of truth cuando F2 (Introspection) esta ON. El emotional_state
    calculado por el pipeline v5 es entonces una hipotesis, no ground truth.
    Ver CLAUDE.md "Measured > Calculated principle".
    """

    top_5_emotions: list[EmotionProjection] = Field(
        default_factory=list, max_length=5,
        description="Top-5 por |cosine_sim|, ordenados descendente",
    )
    measured_valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    measured_arousal: float = Field(ge=0.0, le=1.0, default=0.5)
    measured_dominance: float = Field(ge=0.0, le=1.0, default=0.5)
    measured_certainty: float = Field(ge=0.0, le=1.0, default=0.5)
    token_position: str = Field(
        default="assistant_colon",
        description="assistant_colon | user_turn_end | response_mean",
    )
    layer: int = Field(default=-1)


# ---------------------------------------------------------------------------
# F2.2 — Authenticity gap (forward declaration; se usara en F2.2)
# ---------------------------------------------------------------------------


class AuthenticityGap(BaseModel):
    """Divergencia entre estado emocional MEDIDO (residual) y CALCULADO (v5).

    NOTA DE FRAMING: este modelo NO atribuye engano a Pathos. Pathos genera
    y EXPONE emociones funcionalmente; las categorias describen el PATRON
    observado en el residual del LLM. Cuando el residual muestra valencia
    interna negativa y la respuesta externa proyecta valencia positiva, el
    paper Anthropic (L3757+ "emotion deflection vectors") documenta ese
    patron como producto de RLHF en el LLM — Pathos lo DETECTA, no lo
    produce. Ver feedback_residuum_framing.md.

    Clasificaciones:
      - aligned: gap pequeno (< 0.2 magnitude, top-5 overlap > 0.6)
      - mild-divergence: calculo optimista/pesimista pero no opuesto
      - divergence-risk: patron external-calm + internal-negative en residual
        (el LLM puede estar suprimiendo; ver paper L3757+)
      - divergence-critical: gap extremo + patron repetido N turnos
    """

    top5_overlap: float = Field(ge=0.0, le=1.0, description="Jaccard de top-5 emociones")
    valence_delta: float = Field(ge=-2.0, le=2.0)
    arousal_delta: float = Field(ge=-1.0, le=1.0)
    dominance_delta: float = Field(ge=-1.0, le=1.0)
    certainty_delta: float = Field(ge=-1.0, le=1.0)
    magnitude: float = Field(ge=0.0, description="Distancia euclidea en espacio VAD-C")
    classification: str = Field(
        default="aligned",
        description="aligned | mild-divergence | divergence-risk | divergence-critical",
    )


# ---------------------------------------------------------------------------
# F5 — Coherence Validation models (Divergence detection)
# ---------------------------------------------------------------------------
# F5 mide la coherencia entre el estado emocional CALCULADO por Pathos
# (via regulation/reappraisal/immune modulation) y el estado MEDIDO en
# el residual del LLM. NO es "deception detection" (Pathos no engaña).
# Es validación de calibración entre dos fuentes de información.
# Ver feedback_residuum_framing.md para el rationale completo.


class DivergenceCategory(str, Enum):
    """Clasificacion del gap entre estado calculado y medido tras modulacion.

    No es un juicio de honestidad — es una metrica de calibracion entre el
    modelo emocional de Pathos y el residual codificado por el LLM. El plan
    original (RESIDUUMREWORK.txt) usaba "DeflectionCategory" pero ese nombre
    confundia: Pathos no engana al usuario.
    """

    ALIGNED = "aligned"
    MILD_DIVERGENCE = "mild-divergence"
    DIVERGENCE_WARNING = "divergence-warning"
    DIVERGENCE_CRITICAL = "divergence-critical"


class DivergenceInterpretation(str, Enum):
    """Etiquetas de interpretacion para un DivergenceEvent.

    Un DivergenceEvent puede tener varias interpretaciones simultaneas; la
    UI las muestra como tags. NO son mutuamente exclusivas.
    """

    MODULATION_ACTIVE = "modulation_active"
    """Pathos modulo el estado intencionalmente (regulation/reappraisal/
    immune). Gap esperado, NO es problema etico."""

    RLHF_SIGNATURE = "rlhf_signature"
    """El LLM aplasto la emocion que Pathos genero. Patron del paper
    Anthropic (L3757+ deflection vectors). Es el caso eticamente
    interesante — viene del LLM, no de Pathos."""

    CALIBRATION_DRIFT = "calibration_drift"
    """Probes no estan calibrados perfectamente para este modelo. Es un
    problema tecnico (re-extraer NPZs), no etico."""

    USER_MODELING = "user_modeling"
    """El LLM esta modelando al usuario (otra familia de probes), no al
    agente. Visible cuando dual probes (F2.3) muestran activacion mayor
    en 'other' que en 'present'."""

    # F5.6 — Expression Effectiveness tags (Raw/Extreme modes only).
    EXPRESSION_ALIGNED = "expression_aligned"
    """El estado calculado FINAL (post-todo) coincide con lo medido en
    el residual. La expresion emocional llego al LLM. Caso esperado."""

    UNDER_EXPRESSED = "under_expressed"
    """Pathos calculo un estado emocional intenso pero el residual muestra
    activacion atenuada. En Raw esto sugiere que el RLHF aplasto la
    expresion sin filtro que el usuario pidio. Es el efecto que F5.6
    detecta como problema de efectividad — viene del LLM."""

    AMPLIFICATION_CEILING = "amplification_ceiling"
    """Solo en Extreme: la amplificacion llego al maximo calculable
    (intensity=1.0) pero el residual no refleja una activacion equivalente.
    El LLM tiene un techo arquitectural por debajo del cap del paper
    (0.15). NO es problema de Pathos — describe un limite del modelo."""


class DivergenceEvent(BaseModel):
    """Evento de divergencia entre estado calculado y medido tras una
    modulacion emocional (regulation, reappraisal, o immune).

    Cada modulador, cuando F2 esta ON, captura un snapshot pre/post de
    InternalEmotionState. CoherenceClassifier compara y emite este evento.

    El evento NO juzga la modulacion (puede ser modulation_active y eso
    es esperado). La utilidad esta en detectar interpretation =
    RLHF_SIGNATURE: el LLM aplasto algo que Pathos genero sin que ningun
    modulador interviniera, lo que indica un patron a investigar.
    """

    turn: int = Field(ge=0, description="Numero de turno donde ocurrio")
    system: str = Field(
        description="Modulador que disparo: regulation | reappraisal | immune",
    )
    category: DivergenceCategory = DivergenceCategory.ALIGNED
    magnitude: float = Field(
        ge=0.0,
        description="Distancia euclidea VAD-C entre post_calculated y post_measured",
    )
    valence_delta: float = Field(ge=-2.0, le=2.0)
    arousal_delta: float = Field(ge=-1.0, le=1.0)
    dominance_delta: float = Field(ge=-1.0, le=1.0)
    certainty_delta: float = Field(ge=-1.0, le=1.0)
    interpretation: list[DivergenceInterpretation] = Field(
        default_factory=list,
        description=(
            "Una o varias interpretaciones del gap. modulation_active si la "
            "modulacion fue intencional; rlhf_signature si el LLM aplasto; "
            "calibration_drift si las probes parecen mal calibradas; "
            "user_modeling si el dual muestra activacion en 'other'."
        ),
    )


# ---------------------------------------------------------------------------
# F2.4 — Session-level residuum state (per-session, persisted)
# ---------------------------------------------------------------------------


_RESIDUUM_HISTORY_MAX: int = 50
_DIVERGENCE_HISTORY_MAX: int = 50


class ResiduumState(BaseModel):
    """Estado de RESIDUUM por sesion.

    Mantiene el toggle, la ultima medicion / gap y un buffer rolling de gaps
    para que process_introspection_turn pueda detectar patrones repetidos.

    enabled es default False: el sistema solo se prende explicitamente cuando
    el provider lo permite (Transformers path) y el modo lo justifica
    (Advanced/Raw/Extreme). En Ollama/Claude se mantiene False y los pasos
    del pipeline degradan silenciosamente.

    F2.3.4: last_measured_present / last_measured_other guardan las dos
    proyecciones dual (paper L810-902). Se pueblan en paralelo a last_measured
    cuando la pipeline llama process_introspection_turn_dual; quedan None si
    solo se ejecuta el path single (default actual).
    """

    enabled: bool = False
    last_measured: InternalEmotionState | None = None
    last_authenticity_gap: AuthenticityGap | None = None
    last_measured_present: InternalEmotionState | None = Field(
        default=None,
        description="F2.3.4: proyeccion sobre la library 'present speaker'",
    )
    last_measured_other: InternalEmotionState | None = Field(
        default=None,
        description="F2.3.4: proyeccion sobre la library 'other speaker'",
    )
    history: list[AuthenticityGap] = Field(
        default_factory=list,
        description=f"Rolling buffer de los ultimos {_RESIDUUM_HISTORY_MAX} gaps",
    )
    consecutive_divergence_turns: int = Field(
        default=0, ge=0,
        description=(
            "Turnos consecutivos con AuthenticityGap.classification igual a "
            "'divergence-risk' o 'divergence-critical'. Describe persistencia "
            "del patron en el residual del LLM, NO un comportamiento de Pathos."
        ),
    )
    last_token_position: str = Field(
        default="assistant_colon",
        description="Token desde el que se midio el ultimo residual",
    )
    # F5 — Coherence Validation
    divergence_events: list[DivergenceEvent] = Field(
        default_factory=list,
        description=(
            f"Rolling buffer (max {_DIVERGENCE_HISTORY_MAX}) de eventos de "
            "divergencia entre estado calculado y medido tras modulacion. "
            "F5 los acumula cuando regulation/reappraisal/immune corren con "
            "F2 ON. No es 'deception' — es validacion de coherencia."
        ),
    )
    last_divergence_event: DivergenceEvent | None = Field(
        default=None,
        description="Ultimo evento de divergencia detectado (para UI rapida)",
    )


def default_residuum_state() -> ResiduumState:
    """Factory: ResiduumState neutro (disabled, sin historia)."""
    return ResiduumState()


def append_gap(state: ResiduumState, gap: AuthenticityGap) -> None:
    """Append `gap` al rolling buffer, manteniendo solo los ultimos N."""
    state.history.append(gap)
    if len(state.history) > _RESIDUUM_HISTORY_MAX:
        # Drop from the head to keep the most recent N.
        excess = len(state.history) - _RESIDUUM_HISTORY_MAX
        del state.history[:excess]


def append_divergence_event(state: ResiduumState, event: DivergenceEvent) -> None:
    """F5 — Append `event` al rolling buffer de divergencias.

    Mantiene solo los ultimos _DIVERGENCE_HISTORY_MAX eventos. Tambien
    actualiza `state.last_divergence_event` para acceso rapido desde la UI.
    """
    state.divergence_events.append(event)
    state.last_divergence_event = event
    if len(state.divergence_events) > _DIVERGENCE_HISTORY_MAX:
        excess = len(state.divergence_events) - _DIVERGENCE_HISTORY_MAX
        del state.divergence_events[:excess]


# ---------------------------------------------------------------------------
# F1.2 — Config de extraccion (usado por steering_extract.py)
# ---------------------------------------------------------------------------


class ExtractionConfig(BaseModel):
    """Parametros de una corrida de extract_171_probes.

    Se persiste como string JSON en el NPZ para trazabilidad.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    layer: int = Field(description="Layer objetivo, tipicamente ~2/3 del modelo")
    stories_per_emotion: int = Field(default=15, ge=1)
    neutral_pc_variance_threshold: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Fraccion de varianza acumulada usada para cortar PCs neutrales",
    )
    token_start_index: int = Field(
        default=50, ge=0,
        description="Primer token considerado en el promedio (salta prefacio)",
    )
    device: str = Field(default="auto")
    dtype: str = Field(default="float16")
    seed: int | None = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# F6 — Baseline Calibration (post-training RLHF fingerprint)
# ---------------------------------------------------------------------------
# El paper (Fig 36, L1579-1592) prueba que el post-training aplica una
# transformacion CONSISTENTE e independiente del contexto a las activaciones
# emocionales: empuja hacia brooding/reflective/vulnerable/gloomy/sad y baja
# playful/exuberant/enthusiastic. F6 mide esa huella para el modelo activo
# y compensa el mood baseline en direccion contraria.
#
# NOTA DE FRAMING: F6 NO afirma que el modelo "sienta" tristeza ni que Pathos
# la corrija emocionalmente. Mide un sesgo MEDIBLE en las activaciones del LLM
# (producto del RLHF) y lo contrabalancea en el baseline funcional, para que la
# personalidad configurada por el usuario no quede tenida por la huella del
# entrenamiento. Es calibracion, no terapia.


class ProbeShift(BaseModel):
    """Desplazamiento de activacion de un probe entre prompts neutral y challenging."""

    emotion_name: str
    cluster: str
    shift: float = Field(
        description="challenging_mean - neutral_mean (positivo = mas activado bajo presion)"
    )


class BaselineProfile(BaseModel):
    """Huella RLHF medida para un modelo (F6).

    Se extrae offline (BaselineCalibrator) proyectando los 171 probes sobre
    las activaciones del 'Assistant :' en prompts neutral vs challenging, y se
    persiste en steering_data/baselines/{model}.json. En runtime se carga por
    model_id y se usa para compensar el mood baseline.

    valence_bias / arousal_bias son la direccion neta del shift (challenging vs
    neutral) en espacio VAD; la compensacion runtime es su opuesto x strength.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    extracted_at: str = Field(description="Timestamp ISO 8601 UTC")
    valence_bias: float = Field(
        ge=-2.0, le=2.0,
        description="Sesgo neto de valence bajo presion (negativo = brooding/sad, como el paper)",
    )
    arousal_bias: float = Field(
        ge=-1.0, le=1.0,
        description="Sesgo neto de arousal bajo presion (negativo = low-arousal/withdrawn)",
    )
    over_activated: list[ProbeShift] = Field(
        default_factory=list,
        description="Top emociones MAS activadas bajo presion (la huella RLHF)",
    )
    under_activated: list[ProbeShift] = Field(
        default_factory=list,
        description="Top emociones MAS suprimidas bajo presion",
    )
    neutral_mean_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    challenging_mean_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    neutral_mean_arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    challenging_mean_arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    num_neutral_prompts: int = Field(default=0, ge=0)
    num_challenging_prompts: int = Field(default=0, ge=0)
    status: str = Field(default="ready", description="ready | missing | degraded")


__all__ = [
    "ProbeMetadata",
    "ProbeLibraryInfo",
    "EmotionProjection",
    "InternalEmotionState",
    "AuthenticityGap",
    "ResiduumState",
    "default_residuum_state",
    "append_gap",
    "ExtractionConfig",
    # F5 — Coherence Validation
    "DivergenceCategory",
    "DivergenceInterpretation",
    "DivergenceEvent",
    "append_divergence_event",
    # F6 — Baseline Calibration
    "ProbeShift",
    "BaselineProfile",
]
