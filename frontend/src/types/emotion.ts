/** TypeScript types matching backend Pydantic models exactly. */

// --- Enums ---

export type PrimaryEmotion =
  | "joy" | "excitement" | "gratitude" | "hope"
  | "contentment" | "relief"
  | "anger" | "frustration" | "fear" | "anxiety"
  | "sadness" | "helplessness" | "disappointment"
  | "surprise" | "alertness" | "contemplation"
  | "indifference" | "mixed" | "neutral";

export type MoodLabel = "buoyant" | "serene" | "agitated" | "melancholic" | "neutral";
export type MoodTrend = "improving" | "stable" | "declining";
export type ResponsibleAgent = "user" | "self" | "environment" | "other";

// --- Core Models ---

export interface BodyState {
  energy: number;
  tension: number;
  openness: number;
  warmth: number;
}

export interface Mood {
  baseline_valence: number;
  baseline_arousal: number;
  stability: number;
  trend: MoodTrend;
  label: MoodLabel;
  extreme_event_count: number;
  turns_since_extreme: number;
  original_baseline_valence: number;
  original_baseline_arousal: number;
}

export interface EmotionalState {
  valence: number;
  arousal: number;
  dominance: number;
  certainty: number;
  primary_emotion: PrimaryEmotion;
  secondary_emotion: PrimaryEmotion | null;
  intensity: number;
  emotional_stack: Record<string, number>;
  body_state: BodyState;
  mood: Mood;
  duration: number;
  triggered_by: string;
  timestamp: string;
}

// --- Appraisal ---

export interface RelevanceCheck {
  novelty: number;
  personal_significance: number;
  values_affected: string[];
}

export interface ValenceAssessment {
  goal_conduciveness: number;
  value_alignment: number;
  intrinsic_pleasantness: number;
}

export interface CopingPotential {
  control: number;
  power: number;
  adjustability: number;
}

export interface AgencyAttribution {
  responsible_agent: ResponsibleAgent;
  intentionality: number;
  fairness: number;
}

export interface NormCompatibility {
  internal_standards: number;
  external_standards: number;
  self_consistency: number;
}

export interface AppraisalVector {
  relevance: RelevanceCheck;
  valence: ValenceAssessment;
  coping: CopingPotential;
  agency: AgencyAttribution;
  norms: NormCompatibility;
}

// --- API Responses ---

// --- Pipeline Trace ---

export type PipelineImpact = "none" | "low" | "medium" | "high";

export interface PipelineStep {
  name: string;
  label: string;
  active: boolean;
  skipped_reason: string;
  duration_ms: number;
  summary: string;
  impact: PipelineImpact;
  details: Record<string, unknown>;
  delta: Record<string, number>;
}

export interface PipelineTrace {
  steps: PipelineStep[];
  total_duration_ms: number;
  mode: "advanced" | "lite" | "core";
}

export interface ChatResponse {
  response: string;
  emotional_state: EmotionalState;
  session_id: string;
  audio_available?: boolean;
  pipeline_trace?: PipelineTrace;
}

export interface StateResponse {
  emotional_state: EmotionalState;
  session_id: string;
}

// --- Research Mode ---

export interface HomeostasisDetails {
  applied: boolean;
  state_before: EmotionalState;
  state_after: EmotionalState;
}

export interface AppraisalDetails {
  vector: AppraisalVector;
  computed_valence: number;
  computed_arousal: number;
  computed_dominance: number;
  computed_certainty: number;
}

export interface MemoryAmplificationDetails {
  amplification_factor: number;
  memories_count: number;
  memory_stored: boolean;
}

export interface MoodCongruenceDetails {
  valence_bias: number;
  arousal_bias: number;
  mood_label: string;
  mood_trend: string;
}

export interface EmotionGenerationDetails {
  raw_valence: number;
  raw_arousal: number;
  raw_dominance: number;
  raw_certainty: number;
  blended_valence: number;
  blended_arousal: number;
  blended_dominance: number;
  blended_certainty: number;
  intensity_before_amplification: number;
  intensity_after_amplification: number;
}

export interface AuthenticityMetrics {
  coherence: number;
  continuity: number;
  proportionality: number;
  recovery: number;
  overall: number;
}

// --- Advanced System Details (Fase 4) ---

export interface NeedsDetails {
  connection: number;
  competence: number;
  autonomy: number;
  coherence: number;
  stimulation: number;
  safety: number;
  amplification: number;
}

export interface SocialDetails {
  perceived_intent: number;
  perceived_engagement: number;
  rapport: number;
  trust_level: number;
  communication_style: string;
  valence_modulation: number;
  intensity_modulation: number;
}

export interface RegulationDetails {
  strategy_used: string | null;
  intensity_reduced: number;
  capacity_before: number;
  capacity_after: number;
  breakthrough: boolean;
  suppression_dissonance: number;
}

export interface ReappraisalDetails {
  applied: boolean;
  strategy: string | null;
  original_emotion: string | null;
  reappraised_emotion: string | null;
  intensity_change: number;
  valence_change: number;
}

export interface TemporalDetails {
  rumination_active: boolean;
  rumination_emotion: string | null;
  rumination_intensity: number;
  savoring_active: boolean;
  savoring_emotion: string | null;
  anticipation_active: boolean;
  anticipation_emotion: string | null;
  anticipation_intensity: number;
}

export interface MetaEmotionDetails {
  active: boolean;
  target_emotion: string | null;
  meta_response: string | null;
  intensity: number;
  reason: string;
}

export interface SchemaDetails {
  schemas_count: number;
  primed_emotion: string | null;
  priming_amplification: number;
  pending_patterns: number;
}

export interface PersonalityDetails {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
  variability: number;
  regulation_capacity_base: number;
}

export interface ContagionDetails {
  detected_valence: number;
  detected_arousal: number;
  signal_strength: number;
  shadow_valence: number;
  shadow_arousal: number;
  contagion_perturbation_v: number;
  contagion_perturbation_a: number;
  accumulated_contagion: number;
  susceptibility: number;
}

export interface SomaticDetails {
  markers_count: number;
  somatic_bias: number;
  gut_feeling: string | null;
  pending_category: string | null;
}

export interface CreativityDetails {
  thinking_mode: string;
  creativity_level: number;
  temperature_modifier: number;
  active_instructions: string[];
  triggered_by: string[];
}

export interface ImmuneDetails {
  protection_mode: string;
  protection_strength: number;
  reactivity_dampening: number;
  negative_streak: number;
  peak_negative_intensity: number;
  recovery_turns: number;
  total_activations: number;
  compartmentalized_topics: string[];
}

export interface NarrativeDetails {
  identity_statements_count: number;
  top_statements: string[];
  coherence_score: number;
  crisis_active: boolean;
  crisis_source: string;
  growth_events_count: number;
  last_growth: string;
  narrative_age: number;
  total_contradictions: number;
  total_reinforcements: number;
}

export interface VoiceDetails {
  mode: string;
  voice_key: string;
  speed: number;
  pitch_semitones: number;
  volume: number;
  tremolo: number;
  stage_direction: string;
  backend: string;
  parler_description: string;
  audio_available: boolean;
  asr_available: boolean;
  last_transcription: string;
  detected_language: string;
}

export interface ForecastingDetails {
  enabled: boolean;
  user_valence: number;
  user_arousal: number;
  user_confidence: number;
  user_dominant_signal: string;
  predicted_impact: number;
  predicted_user_valence: number;
  predicted_user_arousal: number;
  risk_flag: boolean;
  risk_reason: string;
  recommendation: string;
  accuracy_score: number;
  total_forecasts: number;
  total_evaluated: number;
  valence_bias: number;
  arousal_bias: number;
}

export interface CouplingDetails {
  active: boolean;
  matrix: number[][];
  contribution_v: number;
  contribution_a: number;
  contribution_d: number;
  contribution_c: number;
}

// --- ARK Rework Systems ---

export interface SelfAppraisalDetails {
  applied: boolean;
  value_alignment: number;
  emotional_coherence: number;
  predicted_self_valence: number;
  should_regenerate: boolean;
  did_regenerate: boolean;
  reason: string;
  adjustments: string[];
}

export interface WorldModelDetails {
  applied: boolean;
  predicted_self_valence_shift: number;
  predicted_self_effect: string;
  predicted_user_valence_shift: number;
  predicted_user_effect: string;
  meta_reaction_effect: string;
  value_alignment: number;
  emotional_risk: number;
  should_modify: boolean;
  did_modify: boolean;
  reason: string;
}

export interface SteeringDetails {
  enabled: boolean;
  status: string;
  model_id: string | null;
  dimensions: string[];
  layers: number[];
  layer_roles: Record<string, number[]>;
  multilayer: boolean;
  total_vectors: number;
  momentum_enabled: boolean;
  momentum_factor: number;
  momentum_turns_stored: number;
}

export interface EmotionalPrefixDetails {
  enabled: boolean;
  status: string;
  num_tokens: number;
  embedding_norm: number;
  dominant_dimension: string;
  scale: number;
}

export interface AttentionDetails {
  enabled: boolean;
  status: string;
  categories_active: Record<string, number>;
  broadening_factor: number;
  positions_biased: number;
  layers_hooked: number[];
  words_biased: number;
}

// --- External Signals ---

export interface SignalSourceMeta {
  source: string;
  label: string;
  description: string;
  category: string;
  base_weight: number;
  enabled: boolean;
  valence_hint: number;
  arousal_hint: number;
  dominance_hint: number | null;
  confidence: number;
}

export interface SignalsConfig {
  enabled: boolean;
  active_count: number;
  sources: SignalSourceMeta[];
}

export interface SignalTestResult {
  status: string;
  source: string;
  processed: {
    valence_delta: number;
    arousal_delta: number;
    dominance_delta: number;
    weight: number;
  };
  fused_effect: {
    valence_modulation: number;
    arousal_modulation: number;
    dominance_modulation: number;
    total_confidence: number;
  };
}

// --- Research Chat Response ---

export interface ResearchChatResponse {
  response: string;
  session_id: string;
  turn_number: number;

  // Core pipeline
  homeostasis: HomeostasisDetails;
  appraisal: AppraisalDetails;
  memory_amplification: MemoryAmplificationDetails;
  mood_congruence: MoodCongruenceDetails;
  emotion_generation: EmotionGenerationDetails;

  // Advanced systems
  needs: NeedsDetails;
  social: SocialDetails;
  regulation: RegulationDetails;
  reappraisal: ReappraisalDetails;
  temporal: TemporalDetails;
  meta_emotion: MetaEmotionDetails;
  schemas: SchemaDetails;
  personality: PersonalityDetails;
  contagion: ContagionDetails;
  somatic: SomaticDetails;
  creativity: CreativityDetails;
  immune: ImmuneDetails;
  narrative: NarrativeDetails;
  forecasting: ForecastingDetails;
  coupling: CouplingDetails;
  voice: VoiceDetails;

  // ARK Rework systems
  self_appraisal: SelfAppraisalDetails;
  world_model: WorldModelDetails;
  steering: SteeringDetails;
  emotional_prefix: EmotionalPrefixDetails;
  attention: AttentionDetails;

  // Results
  emotional_state: EmotionalState;
  emergent_emotions: string[];
  behavior_prompt: string;
  authenticity_metrics: AuthenticityMetrics;
}

// --- Calibration ---

export interface CalibrationScenario {
  stimulus: string;
  expected_emotion: PrimaryEmotion;
  expected_valence: number;
  expected_arousal: number;
  expected_intensity: number;
}

export interface CalibrationResult {
  scenario: CalibrationScenario;
  system_emotion: PrimaryEmotion;
  system_valence: number;
  system_arousal: number;
  system_intensity: number;
  valence_delta: number;
  arousal_delta: number;
  intensity_delta: number;
  emotion_match: boolean;
}

export interface CalibrationProfile {
  valence_offset: number;
  arousal_scale: number;
  intensity_scale: number;
  scenarios_used: number;
  emotion_accuracy: number;
}

// --- Sandbox ---

export interface SandboxResult {
  scenario: string;
  emotional_state: EmotionalState;

  // Core pipeline
  homeostasis: HomeostasisDetails;
  appraisal: AppraisalDetails;
  memory_amplification: MemoryAmplificationDetails;
  mood_congruence: MoodCongruenceDetails;
  emotion_generation: EmotionGenerationDetails;

  // Advanced systems
  needs: NeedsDetails;
  social: SocialDetails;
  regulation: RegulationDetails;
  reappraisal: ReappraisalDetails;
  temporal: TemporalDetails;
  meta_emotion: MetaEmotionDetails;
  schemas: SchemaDetails;
  personality: PersonalityDetails;
  contagion: ContagionDetails;
  somatic: SomaticDetails;
  creativity: CreativityDetails;
  immune: ImmuneDetails;
  narrative: NarrativeDetails;
  forecasting: ForecastingDetails;
  coupling: CouplingDetails;

  // Analysis
  emergent_emotions: string[];
  behavior_prompt: string;
  authenticity_metrics: AuthenticityMetrics;
}

export interface SandboxResponse {
  result: SandboxResult;
  session_id: string;
  personality_overridden: boolean;
  response: string;
}

export interface BatchSandboxResponse {
  results: SandboxResult[];
  session_id: string;
  count: number;
  personality_overridden: boolean;
}

// --- Arena ---

export interface ArenaContestant {
  name: string;
  personality: Record<string, number>;
}

export interface ArenaEntry {
  name: string;
  personality: Record<string, number>;
  result: SandboxResult;
  response: string;
}

export interface ArenaDivergence {
  valence_spread: number;
  arousal_spread: number;
  intensity_spread: number;
  emotion_diversity: number;
  most_positive: string;
  most_negative: string;
  most_intense: string;
  most_calm: string;
}

export interface ArenaResponse {
  scenario: string;
  entries: ArenaEntry[];
  divergence: ArenaDivergence;
  session_id: string;
  count: number;
}

// --- Mirror Test (Challenge) ---

export interface ChallengeTarget {
  emotion: string | null;
  min_valence: number | null;
  max_valence: number | null;
  min_arousal: number | null;
  max_arousal: number | null;
  min_intensity: number | null;
  stack_emotion: string | null;
  stack_threshold: number;
}

export interface ChallengeConfig {
  id: string;
  name: string;
  description: string;
  difficulty: string;
  target: ChallengeTarget;
  max_turns: number;
  hint: string;
  category: string;
}

export interface ChallengeState {
  challenge: ChallengeConfig;
  active: boolean;
  turn: number;
  max_turns: number;
  score: number;
  best_score: number;
  completed: boolean;
  won: boolean;
  score_history: number[];
}

export interface ChallengeChatResponse {
  response: string;
  emotional_state: EmotionalState;
  session_id: string;
  turn_number: number;
  audio_available: boolean;
  challenge: ChallengeState;
  target: ChallengeTarget;
  score_breakdown: Record<string, number>;
}

// --- UI types ---

export type AppMode = "companion" | "research" | "sandbox" | "arena" | "mirror" | "calibration" | "raw" | "autonomous";

export type ResearchPipelineMode = "normal" | "lite" | "raw" | "extreme";

export interface ResearchEvent {
  type: string;
  data: Record<string, unknown>;
  emotional_state: EmotionalState | null;
  timestamp: string;
}

export interface AutonomousResearchState {
  session_id: string;
  pipeline_mode: ResearchPipelineMode;
  is_running: boolean;
  current_topic: string | null;
  total_findings: number;
  total_conclusions: number;
  topics_researched: Array<{
    query: string;
    findings: Array<{
      source_title: string;
      content_snippet: string;
      emotional_reflection: {
        how_it_feels: string;
        emotions_generated: string;
        emotional_insight: string;
        primary_emotion_after: string;
      };
    }>;
    conclusions: Array<{
      topic: string;
      conclusion_text: string;
      emotional_bias: string;
      primary_emotion: string;
      intensity: number;
    }>;
  }>;
  chat_history: Array<{ role: string; content: string }>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  emotional_state?: EmotionalState;
  research_data?: ResearchChatResponse;
  audioAvailable?: boolean;
  turnNumber?: number;
}

/** Emotion color mapping for UI */
export const EMOTION_COLORS: Record<PrimaryEmotion, string> = {
  joy: "#FFD700",
  excitement: "#FF6B35",
  gratitude: "#E8A87C",
  hope: "#85CDCA",
  contentment: "#98D8C8",
  relief: "#B8E6CF",
  anger: "#E23636",
  frustration: "#C84B31",
  fear: "#7B2D8E",
  anxiety: "#9B59B6",
  sadness: "#3498DB",
  helplessness: "#5B6C8A",
  disappointment: "#7F8C8D",
  surprise: "#F39C12",
  alertness: "#E67E22",
  contemplation: "#1ABC9C",
  indifference: "#95A5A6",
  mixed: "#BDC3C7",
  neutral: "#6C7A89",
};
