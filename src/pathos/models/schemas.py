"""Request/Response schemas para la API."""

from pydantic import BaseModel, Field

from pathos.models.appraisal import AppraisalVector
from pathos.models.emotion import EmotionalState
from pathos.models.memory import EmotionalMemory
from pathos.models.values import ValueSystem


class ChatRequest(BaseModel):
    """Request para el endpoint /chat."""

    message: str = Field(..., max_length=10000)
    session_id: str = Field(default="default", pattern=r"^[\w\-]+$")


class PipelineStep(BaseModel):
    """Un paso del pipeline emocional con trace para visualizacion."""

    name: str
    label: str  # Human-readable label
    active: bool  # Whether this step ran
    skipped_reason: str = ""  # Why it was skipped (e.g. "Advanced mode off")
    duration_ms: float = 0.0
    # Simplified interpretation for the user
    summary: str = ""  # e.g. "Detected mild frustration from your tone"
    impact: str = "none"  # "none" | "low" | "medium" | "high" — visual indicator
    # Detailed values (shown when details mode is on)
    details: dict[str, object] = {}
    # Emotional state delta (what changed)
    delta: dict[str, float] = {}  # e.g. {"valence": -0.12, "arousal": +0.05}


class PipelineTrace(BaseModel):
    """Trace completo del pipeline emocional."""

    steps: list[PipelineStep] = []
    total_duration_ms: float = 0.0
    mode: str = "advanced"  # "advanced" | "lite" | "core"


class ChatResponse(BaseModel):
    """Response del endpoint /chat."""

    response: str
    emotional_state: EmotionalState
    session_id: str
    audio_available: bool = False
    turn_number: int = 0
    pipeline_trace: PipelineTrace | None = None


class StateResponse(BaseModel):
    """Response del endpoint /state."""

    emotional_state: EmotionalState
    session_id: str


# --- Research Mode schemas ---


class HomeostasisDetails(BaseModel):
    """Detalles del paso de homeostasis."""

    applied: bool
    state_before: EmotionalState
    state_after: EmotionalState


class AppraisalDetails(BaseModel):
    """Detalles completos del appraisal."""

    vector: AppraisalVector
    computed_valence: float
    computed_arousal: float
    computed_dominance: float
    computed_certainty: float


class MemoryAmplificationDetails(BaseModel):
    """Detalles de la amplificacion por memoria emocional."""

    amplification_factor: float
    memories_count: int
    memory_stored: bool


class MoodCongruenceDetails(BaseModel):
    """Detalles del sesgo de mood congruence."""

    valence_bias: float
    arousal_bias: float
    mood_label: str
    mood_trend: str


class EmotionGenerationDetails(BaseModel):
    """Detalles de la generacion emocional (pre y post blend)."""

    raw_valence: float
    raw_arousal: float
    raw_dominance: float
    raw_certainty: float
    blended_valence: float
    blended_arousal: float
    blended_dominance: float
    blended_certainty: float
    intensity_before_amplification: float
    intensity_after_amplification: float


class AuthenticityMetrics(BaseModel):
    """Metricas de autenticidad emocional (spec sec 9)."""

    coherence: float
    continuity: float
    proportionality: float
    recovery: float
    overall: float


# --- Advanced system details (Fase 4) ---


class NeedsDetails(BaseModel):
    """Detalles de las necesidades computacionales."""

    connection: float
    competence: float
    autonomy: float
    coherence: float
    stimulation: float
    safety: float
    amplification: float


class SocialDetails(BaseModel):
    """Detalles del modelo social del usuario."""

    perceived_intent: float
    perceived_engagement: float
    rapport: float
    trust_level: float
    communication_style: str
    valence_modulation: float
    intensity_modulation: float


class RegulationDetails(BaseModel):
    """Detalles de la regulacion emocional activa."""

    strategy_used: str | None
    intensity_reduced: float
    capacity_before: float
    capacity_after: float
    breakthrough: bool
    suppression_dissonance: float


class ReappraisalDetails(BaseModel):
    """Detalles de la reevaluacion cognitiva."""

    applied: bool
    strategy: str | None = None
    original_emotion: str | None = None
    reappraised_emotion: str | None = None
    intensity_change: float = 0.0
    valence_change: float = 0.0


class TemporalDetails(BaseModel):
    """Detalles de las dinamicas temporales."""

    rumination_active: bool
    rumination_emotion: str | None = None
    rumination_intensity: float = 0.0
    savoring_active: bool
    savoring_emotion: str | None = None
    anticipation_active: bool
    anticipation_emotion: str | None = None
    anticipation_intensity: float = 0.0


class MetaEmotionDetails(BaseModel):
    """Detalles de la meta-emocion."""

    active: bool
    target_emotion: str | None = None
    meta_response: str | None = None
    intensity: float = 0.0
    reason: str = ""


class SchemaDetails(BaseModel):
    """Detalles de los schemas emocionales."""

    schemas_count: int
    primed_emotion: str | None = None
    priming_amplification: float = 0.0
    pending_patterns: int = 0


class ContagionDetails(BaseModel):
    """Detalles del contagio emocional."""

    detected_valence: float = 0.0
    detected_arousal: float = 0.0
    signal_strength: float = 0.0
    shadow_valence: float = 0.0
    shadow_arousal: float = 0.0
    contagion_perturbation_v: float = 0.0
    contagion_perturbation_a: float = 0.0
    accumulated_contagion: float = 0.0
    susceptibility: float = 0.0


class SomaticDetails(BaseModel):
    """Detalles de los marcadores somaticos."""

    markers_count: int = 0
    somatic_bias: float = 0.0
    gut_feeling: str | None = None
    pending_category: str | None = None


class CreativityDetails(BaseModel):
    """Detalles del modulador de creatividad emocional."""

    thinking_mode: str = "standard"
    creativity_level: float = 0.0
    temperature_modifier: float = 0.0
    active_instructions: list[str] = []
    triggered_by: list[str] = []


class ImmuneDetails(BaseModel):
    """Detalles del sistema inmune emocional."""

    protection_mode: str = "none"
    protection_strength: float = 0.0
    reactivity_dampening: float = 0.0
    negative_streak: int = 0
    peak_negative_intensity: float = 0.0
    recovery_turns: int = 0
    total_activations: int = 0
    compartmentalized_topics: list[str] = []


class VoiceDetails(BaseModel):
    """Detalles del sistema de voz (opcional)."""

    mode: str = "text_only"
    voice_key: str = ""
    speed: float = 1.0
    pitch_semitones: float = 0.0
    volume: float = 1.0
    tremolo: float = 0.0
    stage_direction: str = ""
    backend: str = "kokoro"
    parler_description: str = ""
    audio_available: bool = False
    asr_available: bool = False
    last_transcription: str = ""
    detected_language: str = ""


class ForecastingDetails(BaseModel):
    """Detalles del emotional forecasting (opcional)."""

    enabled: bool = False
    user_valence: float = 0.0
    user_arousal: float = 0.0
    user_confidence: float = 0.0
    user_dominant_signal: str = "neutral"
    predicted_impact: float = 0.0
    predicted_user_valence: float = 0.0
    predicted_user_arousal: float = 0.0
    risk_flag: bool = False
    risk_reason: str = ""
    recommendation: str = ""
    accuracy_score: float = 0.5
    total_forecasts: int = 0
    total_evaluated: int = 0
    valence_bias: float = 0.0
    arousal_bias: float = 0.0


class NarrativeDetails(BaseModel):
    """Detalles del yo narrativo."""

    identity_statements_count: int = 0
    top_statements: list[str] = []
    coherence_score: float = 1.0
    crisis_active: bool = False
    crisis_source: str = ""
    growth_events_count: int = 0
    last_growth: str = ""
    narrative_age: int = 0
    total_contradictions: int = 0
    total_reinforcements: int = 0


class PersonalityDetails(BaseModel):
    """Detalles del perfil de personalidad."""

    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    variability: float
    regulation_capacity_base: float


class SelfAppraisalDetails(BaseModel):
    """Detalles de la self-appraisal (evaluación secundaria de la propia respuesta)."""

    applied: bool = False
    value_alignment: float = 1.0
    emotional_coherence: float = 1.0
    predicted_self_valence: float = 0.0
    should_regenerate: bool = False
    did_regenerate: bool = False
    reason: str = ""
    adjustments: list[str] = []


class SteeringDetails(BaseModel):
    """Detalles del steering vector engine (Representation Engineering)."""

    enabled: bool = False
    status: str = "not_loaded"  # "not_loaded" | "ready" | "disabled"
    model_id: str | None = None
    dimensions: list[str] = []
    layers: list[int] = []
    layer_roles: dict[str, list[int]] = {}  # "early"/"mid"/"late" -> layer indices
    multilayer: bool = True
    total_vectors: int = 0
    momentum_enabled: bool = False
    momentum_factor: float = 0.0
    momentum_turns_stored: int = 0


class AttentionDetails(BaseModel):
    """Detalles de la modulación de atención emocional."""

    enabled: bool = False
    status: str = "inactive"  # "active" | "inactive" | "disabled" | "provider_unsupported"
    categories_active: dict[str, float] = {}
    broadening_factor: float = 1.0
    positions_biased: int = 0
    layers_hooked: list[int] = []
    words_biased: int = 0


class EmotionalPrefixDetails(BaseModel):
    """Detalles del emotional prefix (tokens sintéticos en capa de embedding)."""

    enabled: bool = False
    status: str = "inactive"  # "active" | "inactive" | "disabled" | "provider_unsupported"
    num_tokens: int = 0
    embedding_norm: float = 0.0
    dominant_dimension: str = "neutral"
    scale: float = 0.5


class WorldModelDetails(BaseModel):
    """Detalles del world model emocional (simulación predictiva pre-envío)."""

    applied: bool = False
    predicted_self_valence_shift: float = 0.0
    predicted_self_effect: str = "neutral"
    predicted_user_valence_shift: float = 0.0
    predicted_user_effect: str = "neutral"
    meta_reaction_effect: str = "neutral"
    value_alignment: float = 1.0
    emotional_risk: float = 0.0
    should_modify: bool = False
    did_modify: bool = False
    reason: str = ""


class CouplingDetails(BaseModel):
    """Detalles del acoplamiento dimensional (cross-dimensional ODE interaction)."""

    active: bool = False
    matrix: list[list[float]] = []  # 4x4 [V,A,D,C] x [V,A,D,C]
    contribution_v: float = 0.0  # Coupling contribution to valence this turn
    contribution_a: float = 0.0  # Coupling contribution to arousal this turn
    contribution_d: float = 0.0  # Coupling contribution to dominance this turn
    contribution_c: float = 0.0  # Coupling contribution to certainty this turn


class ResearchChatResponse(BaseModel):
    """Response del endpoint /research/chat - expone todos los internos."""

    response: str
    session_id: str
    turn_number: int

    # Core pipeline stages
    homeostasis: HomeostasisDetails
    appraisal: AppraisalDetails
    memory_amplification: MemoryAmplificationDetails
    mood_congruence: MoodCongruenceDetails
    emotion_generation: EmotionGenerationDetails

    # Advanced systems (Fase 4)
    needs: NeedsDetails
    social: SocialDetails
    regulation: RegulationDetails
    reappraisal: ReappraisalDetails
    temporal: TemporalDetails
    meta_emotion: MetaEmotionDetails
    schemas: SchemaDetails
    personality: PersonalityDetails
    contagion: ContagionDetails
    somatic: SomaticDetails
    creativity: CreativityDetails
    immune: ImmuneDetails
    narrative: NarrativeDetails
    forecasting: ForecastingDetails
    coupling: CouplingDetails
    self_appraisal: SelfAppraisalDetails
    world_model: WorldModelDetails
    steering: SteeringDetails
    emotional_prefix: EmotionalPrefixDetails
    attention: AttentionDetails
    voice: VoiceDetails

    # Results
    emotional_state: EmotionalState
    emergent_emotions: list[str]
    behavior_prompt: str

    # Authenticity metrics
    authenticity_metrics: AuthenticityMetrics


class ResearchStateResponse(BaseModel):
    """Response del endpoint /research/state - estado completo de una sesion."""

    session_id: str
    turn_count: int
    emotional_state: EmotionalState
    value_system: ValueSystem
    memories: list[EmotionalMemory]
    conversation_length: int


# --- Scenario Sandbox schemas ---


class ScenarioRequest(BaseModel):
    """Request para /sandbox/simulate — ejecuta pipeline sin respuesta LLM."""

    scenario: str = Field(..., max_length=10000)
    session_id: str = Field(default="default", pattern=r"^[\w\-]+$")
    personality: dict[str, float] | None = None  # Big Five override parcial
    initial_state: str | None = None  # "neutral" | "current" (default: current)
    rapport: float | None = None  # Override rapport (0-1)
    trust: float | None = None  # Override trust (0-1)


class BatchScenarioRequest(BaseModel):
    """Request para /sandbox/batch — multiples escenarios."""

    scenarios: list[str] = Field(..., max_length=50)
    session_id: str = Field(default="default", pattern=r"^[\w\-]+$")
    personality: dict[str, float] | None = None
    initial_state: str | None = None
    rapport: float | None = None
    trust: float | None = None


class SandboxResult(BaseModel):
    """Resultado completo de un escenario simulado (sin respuesta LLM)."""

    scenario: str
    emotional_state: EmotionalState

    # Core pipeline
    homeostasis: HomeostasisDetails
    appraisal: AppraisalDetails
    memory_amplification: MemoryAmplificationDetails
    mood_congruence: MoodCongruenceDetails
    emotion_generation: EmotionGenerationDetails

    # Advanced systems
    needs: NeedsDetails
    social: SocialDetails
    regulation: RegulationDetails
    reappraisal: ReappraisalDetails
    temporal: TemporalDetails
    meta_emotion: MetaEmotionDetails
    schemas: SchemaDetails
    personality: PersonalityDetails
    contagion: ContagionDetails
    somatic: SomaticDetails
    creativity: CreativityDetails
    immune: ImmuneDetails
    narrative: NarrativeDetails
    forecasting: ForecastingDetails
    coupling: CouplingDetails = CouplingDetails()

    # Analysis
    emergent_emotions: list[str]
    behavior_prompt: str
    authenticity_metrics: AuthenticityMetrics


class SandboxResponse(BaseModel):
    """Response de /sandbox/simulate."""

    result: SandboxResult
    session_id: str
    personality_overridden: bool = False
    response: str = ""  # LLM-generated response reflecting the emotional state


class BatchSandboxResponse(BaseModel):
    """Response de /sandbox/batch."""

    results: list[SandboxResult]
    session_id: str
    count: int
    personality_overridden: bool = False


# --- Arena schemas ---


class ArenaContestant(BaseModel):
    """Una personalidad para la arena."""

    name: str
    personality: dict[str, float]  # Big Five values


class ArenaRequest(BaseModel):
    """Request para /arena/compare — N personalidades × 1 escenario."""

    scenario: str = Field(..., max_length=10000)
    contestants: list[ArenaContestant]
    session_id: str = Field(default="default", pattern=r"^[\w\-]+$")
    rapport: float | None = None
    trust: float | None = None


class ArenaEntry(BaseModel):
    """Resultado de un contestante en la arena."""

    name: str
    personality: dict[str, float]
    result: SandboxResult
    response: str = ""  # LLM-generated response reflecting this personality's emotional state


class ArenaDivergence(BaseModel):
    """Metricas de divergencia entre todos los contestantes."""

    valence_spread: float  # max - min valence
    arousal_spread: float
    intensity_spread: float
    emotion_diversity: int  # distinct primary emotions
    most_positive: str  # contestant name
    most_negative: str
    most_intense: str
    most_calm: str


class ArenaResponse(BaseModel):
    """Response de /arena/compare."""

    scenario: str
    entries: list[ArenaEntry]
    divergence: ArenaDivergence
    session_id: str
    count: int


# --- Mirror Test (Challenge) schemas ---


class ChallengeTarget(BaseModel):
    """Target emocional para un challenge."""

    emotion: str | None = None  # Target primary emotion (None = any)
    min_valence: float | None = None
    max_valence: float | None = None
    min_arousal: float | None = None
    max_arousal: float | None = None
    min_intensity: float | None = None
    stack_emotion: str | None = None  # Specific emotion in stack above threshold
    stack_threshold: float = 0.3


class ChallengeConfig(BaseModel):
    """Configuracion de un challenge."""

    id: str
    name: str
    description: str
    difficulty: str  # "easy" | "medium" | "hard" | "extreme"
    target: ChallengeTarget
    max_turns: int = 10
    hint: str = ""
    category: str = "emotion"  # "emotion" | "stack" | "dimensional" | "complex"


class ChallengeStartRequest(BaseModel):
    """Request para iniciar un challenge."""

    challenge_id: str
    session_id: str = "default"


class ChallengeState(BaseModel):
    """Estado actual de un challenge en progreso."""

    challenge: ChallengeConfig
    active: bool = True
    turn: int = 0
    max_turns: int = 10
    score: float = 0.0  # 0-100, current proximity to target
    best_score: float = 0.0  # Best score achieved during challenge
    completed: bool = False
    won: bool = False  # Score >= 75 at any point
    score_history: list[float] = []


class ChallengeChatRequest(BaseModel):
    """Request para /challenge/chat — wraps normal chat + scoring."""

    message: str = Field(..., max_length=10000)
    session_id: str = Field(default="default", pattern=r"^[\w\-]+$")


class ChallengeChatResponse(BaseModel):
    """Response de /challenge/chat — chat response + challenge progress."""

    response: str
    emotional_state: EmotionalState
    session_id: str
    turn_number: int
    audio_available: bool = False

    # Challenge-specific
    challenge: ChallengeState
    target: ChallengeTarget
    score_breakdown: dict[str, float] = {}  # Individual dimension scores
