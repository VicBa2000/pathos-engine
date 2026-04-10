import { useState, useCallback } from "react";
import type { EmotionalState, SandboxResult } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { EmotionAvatar } from "./EmotionAvatar";
import { EmotionGenesis } from "./EmotionGenesis";
import * as api from "../api/client";
import "./SandboxPanel.css";

interface Props {
  sessionId: string;
  connected: boolean;
}

const BIG_FIVE = [
  { key: "openness", label: "Openness", desc: "Curiosity, creativity" },
  { key: "conscientiousness", label: "Conscientiousness", desc: "Discipline, order" },
  { key: "extraversion", label: "Extraversion", desc: "Social energy" },
  { key: "agreeableness", label: "Agreeableness", desc: "Empathy, cooperation" },
  { key: "neuroticism", label: "Neuroticism", desc: "Emotional reactivity" },
] as const;

const DEFAULT_PERSONALITY: Record<string, number> = {
  openness: 0.65,
  conscientiousness: 0.55,
  extraversion: 0.50,
  agreeableness: 0.60,
  neuroticism: 0.40,
};

const PRESET_PERSONALITIES: Record<string, Record<string, number>> = {
  "Default": { openness: 0.65, conscientiousness: 0.55, extraversion: 0.50, agreeableness: 0.60, neuroticism: 0.40 },
  "Anxious": { openness: 0.4, conscientiousness: 0.6, extraversion: 0.3, agreeableness: 0.7, neuroticism: 0.85 },
  "Bold": { openness: 0.8, conscientiousness: 0.4, extraversion: 0.85, agreeableness: 0.35, neuroticism: 0.2 },
  "Stoic": { openness: 0.5, conscientiousness: 0.8, extraversion: 0.3, agreeableness: 0.5, neuroticism: 0.15 },
  "Creative": { openness: 0.95, conscientiousness: 0.3, extraversion: 0.6, agreeableness: 0.5, neuroticism: 0.5 },
  "Empathic": { openness: 0.7, conscientiousness: 0.5, extraversion: 0.6, agreeableness: 0.9, neuroticism: 0.55 },
};

const EXAMPLE_SCENARIOS = [
  "You just found out a close friend has been lying to you for months.",
  "A stranger on the street gives you a genuine, unexpected compliment.",
  "You receive news that you didn't get the job you really wanted.",
  "Someone you care about tells you they're proud of who you've become.",
  "You witness an act of injustice but feel powerless to intervene.",
  "After months of effort, you finally solve a problem that seemed impossible.",
];

export function SandboxPanel({ sessionId, connected }: Props) {
  const [scenario, setScenario] = useState("");
  const [personality, setPersonality] = useState<Record<string, number>>({ ...DEFAULT_PERSONALITY });
  const [initialState, setInitialState] = useState<"current" | "neutral">("neutral");
  const [rapport, setRapport] = useState(0.5);
  const [trust, setTrust] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SandboxResult | null>(null);
  const [batchMode, setBatchMode] = useState(false);
  const [batchResults, setBatchResults] = useState<SandboxResult[]>([]);
  const [batchText, setBatchText] = useState("");
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(["emotion", "appraisal"]));
  // Local emotional state for embedded avatar/genesis — resets each simulation
  const [localEmotionalState, setLocalEmotionalState] = useState<EmotionalState | null>(null);
  // LLM response for the simulated scenario
  const [llmResponse, setLlmResponse] = useState<string>("");

  const toggleSection = (s: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  };

  const handleSimulate = useCallback(async () => {
    if (!scenario.trim() || !connected) return;
    setLoading(true);
    try {
      const res = await api.simulateSandbox(scenario.trim(), sessionId, {
        personality,
        initial_state: initialState,
        rapport,
        trust,
      });
      setResult(res.result);
      setBatchResults([]);
      setLocalEmotionalState(res.result.emotional_state);
      setLlmResponse(res.response || "");
    } catch (err) {
      console.error("Sandbox simulate failed:", err);
    } finally {
      setLoading(false);
    }
  }, [scenario, sessionId, personality, initialState, rapport, trust, connected]);

  const handleBatch = useCallback(async () => {
    const scenarios = batchText.split("\n").map(s => s.trim()).filter(Boolean);
    if (scenarios.length === 0 || !connected) return;
    setLoading(true);
    try {
      const res = await api.batchSandbox(scenarios, sessionId, {
        personality,
        initial_state: initialState,
        rapport,
        trust,
      });
      setBatchResults(res.results);
      setResult(null);
    } catch (err) {
      console.error("Sandbox batch failed:", err);
    } finally {
      setLoading(false);
    }
  }, [batchText, sessionId, personality, initialState, rapport, trust, connected]);

  return (
    <div className="sandbox">
      <div className="sandbox__input-area">
        <div className="sandbox__header">
          <h2 className="sandbox__title">Scenario Sandbox</h2>
          <div className="sandbox__mode-toggle">
            <button className={`sandbox__tab ${!batchMode ? "sandbox__tab--active" : ""}`} onClick={() => setBatchMode(false)}>Single</button>
            <button className={`sandbox__tab ${batchMode ? "sandbox__tab--active" : ""}`} onClick={() => setBatchMode(true)}>Batch</button>
          </div>
        </div>

        {!batchMode ? (
          <div className="sandbox__scenario-input">
            <textarea
              className="sandbox__textarea"
              value={scenario}
              onChange={e => setScenario(e.target.value)}
              placeholder="Describe a scenario to process through the emotional pipeline..."
              rows={3}
            />
            <div className="sandbox__examples">
              {EXAMPLE_SCENARIOS.map((ex, i) => (
                <button key={i} className="sandbox__example-btn" onClick={() => setScenario(ex)}>
                  {ex.slice(0, 60)}...
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="sandbox__scenario-input">
            <textarea
              className="sandbox__textarea sandbox__textarea--batch"
              value={batchText}
              onChange={e => setBatchText(e.target.value)}
              placeholder="One scenario per line (max 20)..."
              rows={6}
            />
          </div>
        )}

        {/* Personality sliders */}
        <div className="sandbox__personality">
          <div className="sandbox__personality-header">
            <span className="sandbox__section-label">Personality (Big Five)</span>
            <div className="sandbox__presets">
              {Object.entries(PRESET_PERSONALITIES).map(([name, p]) => (
                <button key={name} className="sandbox__preset-btn" onClick={() => setPersonality({ ...p })}>
                  {name}
                </button>
              ))}
            </div>
          </div>
          {BIG_FIVE.map(({ key, label, desc }) => (
            <div key={key} className="sandbox__slider-row">
              <label className="sandbox__slider-label">
                {label}
                <span className="sandbox__slider-desc">{desc}</span>
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={personality[key]}
                onChange={e => setPersonality(prev => ({ ...prev, [key]: parseFloat(e.target.value) }))}
                className="sandbox__slider"
              />
              <span className="sandbox__slider-value">{personality[key].toFixed(2)}</span>
            </div>
          ))}
        </div>

        {/* Context sliders */}
        <div className="sandbox__context">
          <span className="sandbox__section-label">Context</span>
          <div className="sandbox__context-row">
            <label className="sandbox__context-label">Initial State</label>
            <select
              value={initialState}
              onChange={e => setInitialState(e.target.value as "current" | "neutral")}
              className="sandbox__select"
            >
              <option value="neutral">Neutral (clean slate)</option>
              <option value="current">Current (session state)</option>
            </select>
          </div>
          <div className="sandbox__slider-row">
            <label className="sandbox__slider-label">Rapport</label>
            <input type="range" min={0} max={1} step={0.05} value={rapport} onChange={e => setRapport(parseFloat(e.target.value))} className="sandbox__slider" />
            <span className="sandbox__slider-value">{rapport.toFixed(2)}</span>
          </div>
          <div className="sandbox__slider-row">
            <label className="sandbox__slider-label">Trust</label>
            <input type="range" min={0} max={1} step={0.05} value={trust} onChange={e => setTrust(parseFloat(e.target.value))} className="sandbox__slider" />
            <span className="sandbox__slider-value">{trust.toFixed(2)}</span>
          </div>
        </div>

        <button
          className="sandbox__run-btn"
          onClick={batchMode ? handleBatch : handleSimulate}
          disabled={loading || !connected || (batchMode ? !batchText.trim() : !scenario.trim())}
        >
          {loading ? "Processing..." : batchMode ? "Run Batch" : "Simulate"}
        </button>
      </div>

      {/* Results + Embedded visuals */}
      <div className="sandbox__right">
        {/* Embedded Avatar + Genesis (sandbox-local, visually differentiated) */}
        {localEmotionalState && (
          <div className="sandbox__visuals">
            <div className="sandbox__visual-box">
              <EmotionAvatar emotionalState={localEmotionalState} />
            </div>
            <div className="sandbox__visual-box">
              <EmotionGenesis emotionalState={localEmotionalState} />
            </div>
          </div>
        )}

        {/* LLM Response */}
        {llmResponse && (
          <div className="sandbox__llm-response">
            <div className="sandbox__llm-label">Response</div>
            <div className="sandbox__llm-text">{llmResponse}</div>
          </div>
        )}

        <div className="sandbox__results">
          {result && <SandboxResultView result={result} expandedSections={expandedSections} toggleSection={toggleSection} />}
          {batchResults.length > 0 && (
            <div className="sandbox__batch-results">
              <h3 className="sandbox__batch-title">Batch Results ({batchResults.length})</h3>
              {batchResults.map((r, i) => (
                <div key={i} className="sandbox__batch-item">
                  <SandboxResultView result={r} expandedSections={expandedSections} toggleSection={toggleSection} compact />
                </div>
              ))}
            </div>
          )}
          {!result && batchResults.length === 0 && !loading && (
            <div className="sandbox__empty">
              Run a scenario to see the full emotional pipeline output
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Result display ---

function SandboxResultView({ result, expandedSections, toggleSection, compact }: {
  result: SandboxResult;
  expandedSections: Set<string>;
  toggleSection: (s: string) => void;
  compact?: boolean;
}) {
  const s = result.emotional_state;
  const color = EMOTION_COLORS[s.primary_emotion] || EMOTION_COLORS.neutral;

  return (
    <div className={`sandbox-result ${compact ? "sandbox-result--compact" : ""}`}>
      {compact && <div className="sandbox-result__scenario">{result.scenario}</div>}

      {/* Emotion summary */}
      <div className="sandbox-result__summary" style={{ borderLeftColor: color }}>
        <div className="sandbox-result__emotion" style={{ color }}>
          {s.primary_emotion}
          {s.secondary_emotion && s.secondary_emotion !== s.primary_emotion && (
            <span className="sandbox-result__secondary"> / {s.secondary_emotion}</span>
          )}
        </div>
        <div className="sandbox-result__dims">
          <Dim label="Valence" value={s.valence} range={[-1, 1]} />
          <Dim label="Arousal" value={s.arousal} />
          <Dim label="Dominance" value={s.dominance} />
          <Dim label="Intensity" value={s.intensity} />
        </div>
      </div>

      {/* Emotional stack */}
      <Section title="Emotional Stack" id="emotion" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__stack">
          {Object.entries(s.emotional_stack)
            .filter(([, v]) => v > 0.01)
            .sort((a, b) => b[1] - a[1])
            .map(([emo, val]) => (
              <div key={emo} className="sandbox-result__stack-item">
                <span className="sandbox-result__stack-label" style={{ color: EMOTION_COLORS[emo as keyof typeof EMOTION_COLORS] || "#999" }}>{emo}</span>
                <div className="sandbox-result__bar-bg">
                  <div className="sandbox-result__bar-fill" style={{ width: `${val * 100}%`, background: EMOTION_COLORS[emo as keyof typeof EMOTION_COLORS] || "#666" }} />
                </div>
                <span className="sandbox-result__stack-val">{val.toFixed(3)}</span>
              </div>
            ))}
        </div>
      </Section>

      {/* Appraisal */}
      <Section title="Appraisal" id="appraisal" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Novelty" value={result.appraisal.vector.relevance.novelty.toFixed(3)} />
          <KV label="Significance" value={result.appraisal.vector.relevance.personal_significance.toFixed(3)} />
          <KV label="Goal Conducive" value={result.appraisal.vector.valence.goal_conduciveness.toFixed(3)} />
          <KV label="Value Align" value={result.appraisal.vector.valence.value_alignment.toFixed(3)} />
          <KV label="Pleasantness" value={result.appraisal.vector.valence.intrinsic_pleasantness.toFixed(3)} />
          <KV label="Control" value={result.appraisal.vector.coping.control.toFixed(3)} />
          <KV label="Power" value={result.appraisal.vector.coping.power.toFixed(3)} />
          <KV label="Agency (intent)" value={result.appraisal.vector.agency.intentionality.toFixed(3)} />
          <KV label="Agency (fair)" value={result.appraisal.vector.agency.fairness.toFixed(3)} />
          <KV label="Norm (internal)" value={result.appraisal.vector.norms.internal_standards.toFixed(3)} />
          <KV label="Norm (external)" value={result.appraisal.vector.norms.external_standards.toFixed(3)} />
          <KV label="Computed V" value={result.appraisal.computed_valence.toFixed(3)} />
          <KV label="Computed A" value={result.appraisal.computed_arousal.toFixed(3)} />
        </div>
      </Section>

      {/* Body State */}
      <Section title="Body State" id="body" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Energy" value={s.body_state.energy.toFixed(3)} />
          <KV label="Tension" value={s.body_state.tension.toFixed(3)} />
          <KV label="Openness" value={s.body_state.openness.toFixed(3)} />
          <KV label="Warmth" value={s.body_state.warmth.toFixed(3)} />
        </div>
      </Section>

      {/* Mood */}
      <Section title="Mood" id="mood" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Label" value={result.mood_congruence.mood_label} />
          <KV label="Trend" value={result.mood_congruence.mood_trend} />
          <KV label="Valence Bias" value={result.mood_congruence.valence_bias.toFixed(4)} />
          <KV label="Arousal Bias" value={result.mood_congruence.arousal_bias.toFixed(4)} />
        </div>
      </Section>

      {/* Needs */}
      <Section title="Needs" id="needs" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Connection" value={result.needs.connection.toFixed(3)} />
          <KV label="Competence" value={result.needs.competence.toFixed(3)} />
          <KV label="Autonomy" value={result.needs.autonomy.toFixed(3)} />
          <KV label="Coherence" value={result.needs.coherence.toFixed(3)} />
          <KV label="Stimulation" value={result.needs.stimulation.toFixed(3)} />
          <KV label="Safety" value={result.needs.safety.toFixed(3)} />
          <KV label="Amplification" value={result.needs.amplification.toFixed(4)} />
        </div>
      </Section>

      {/* Social */}
      <Section title="Social Cognition" id="social" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Rapport" value={result.social.rapport.toFixed(3)} />
          <KV label="Trust" value={result.social.trust_level.toFixed(3)} />
          <KV label="Intent" value={result.social.perceived_intent.toFixed(3)} />
          <KV label="Engagement" value={result.social.perceived_engagement.toFixed(3)} />
          <KV label="Style" value={result.social.communication_style} />
          <KV label="V mod" value={result.social.valence_modulation.toFixed(4)} />
          <KV label="I mod" value={result.social.intensity_modulation.toFixed(4)} />
        </div>
      </Section>

      {/* Contagion */}
      <Section title="Emotion Contagion" id="contagion" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Detected V" value={result.contagion.detected_valence.toFixed(3)} />
          <KV label="Detected A" value={result.contagion.detected_arousal.toFixed(3)} />
          <KV label="Signal" value={result.contagion.signal_strength.toFixed(3)} />
          <KV label="Shadow V" value={result.contagion.shadow_valence.toFixed(3)} />
          <KV label="Shadow A" value={result.contagion.shadow_arousal.toFixed(3)} />
          <KV label="Susceptibility" value={result.contagion.susceptibility.toFixed(3)} />
        </div>
      </Section>

      {/* Somatic */}
      <Section title="Somatic Markers" id="somatic" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Markers" value={String(result.somatic.markers_count)} />
          <KV label="Bias" value={result.somatic.somatic_bias.toFixed(4)} />
          <KV label="Gut Feeling" value={result.somatic.gut_feeling || "none"} />
        </div>
      </Section>

      {/* Regulation */}
      <Section title="Regulation" id="regulation" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Strategy" value={result.regulation.strategy_used || "none"} />
          <KV label="Intensity Reduced" value={result.regulation.intensity_reduced.toFixed(4)} />
          <KV label="Capacity Before" value={result.regulation.capacity_before.toFixed(3)} />
          <KV label="Capacity After" value={result.regulation.capacity_after.toFixed(3)} />
          <KV label="Breakthrough" value={result.regulation.breakthrough ? "YES" : "no"} />
        </div>
      </Section>

      {/* Reappraisal */}
      <Section title="Reappraisal" id="reappraisal" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Applied" value={result.reappraisal.applied ? "yes" : "no"} />
          <KV label="Strategy" value={result.reappraisal.strategy || "none"} />
          <KV label="Original" value={result.reappraisal.original_emotion || "-"} />
          <KV label="Reappraised" value={result.reappraisal.reappraised_emotion || "-"} />
          <KV label="Intensity Change" value={result.reappraisal.intensity_change.toFixed(4)} />
        </div>
      </Section>

      {/* Creativity */}
      <Section title="Creativity" id="creativity" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Mode" value={result.creativity.thinking_mode} />
          <KV label="Level" value={result.creativity.creativity_level.toFixed(3)} />
          <KV label="Temp Modifier" value={result.creativity.temperature_modifier.toFixed(3)} />
          {result.creativity.triggered_by.length > 0 && (
            <KV label="Triggers" value={result.creativity.triggered_by.join(", ")} />
          )}
        </div>
      </Section>

      {/* Immune */}
      <Section title="Immune System" id="immune" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Mode" value={result.immune.protection_mode} />
          <KV label="Strength" value={result.immune.protection_strength.toFixed(3)} />
          <KV label="Dampening" value={result.immune.reactivity_dampening.toFixed(3)} />
          <KV label="Neg Streak" value={String(result.immune.negative_streak)} />
        </div>
      </Section>

      {/* Meta-Emotion */}
      <Section title="Meta-Emotion" id="meta" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Active" value={result.meta_emotion.active ? "yes" : "no"} />
          <KV label="Target" value={result.meta_emotion.target_emotion || "-"} />
          <KV label="Response" value={result.meta_emotion.meta_response || "-"} />
          <KV label="Intensity" value={result.meta_emotion.intensity.toFixed(3)} />
          {result.meta_emotion.reason && <KV label="Reason" value={result.meta_emotion.reason} />}
        </div>
      </Section>

      {/* Emergent */}
      {result.emergent_emotions.length > 0 && (
        <Section title="Emergent Emotions" id="emergent" expanded={expandedSections} toggle={toggleSection}>
          <div className="sandbox-result__tags">
            {result.emergent_emotions.map(e => <span key={e} className="sandbox-result__tag">{e}</span>)}
          </div>
        </Section>
      )}

      {/* Authenticity */}
      <Section title="Authenticity" id="auth" expanded={expandedSections} toggle={toggleSection}>
        <div className="sandbox-result__grid">
          <KV label="Coherence" value={result.authenticity_metrics.coherence.toFixed(3)} />
          <KV label="Continuity" value={result.authenticity_metrics.continuity.toFixed(3)} />
          <KV label="Proportionality" value={result.authenticity_metrics.proportionality.toFixed(3)} />
          <KV label="Recovery" value={result.authenticity_metrics.recovery.toFixed(3)} />
          <KV label="Overall" value={result.authenticity_metrics.overall.toFixed(3)} />
        </div>
      </Section>

      {/* Behavior Prompt */}
      <Section title="Behavior Prompt" id="prompt" expanded={expandedSections} toggle={toggleSection}>
        <pre className="sandbox-result__prompt">{result.behavior_prompt}</pre>
      </Section>
    </div>
  );
}

// --- Helper components ---

function Section({ title, id, expanded, toggle, children }: {
  title: string;
  id: string;
  expanded: Set<string>;
  toggle: (s: string) => void;
  children: React.ReactNode;
}) {
  const isOpen = expanded.has(id);
  return (
    <div className={`sandbox-section ${isOpen ? "sandbox-section--open" : ""}`}>
      <button className="sandbox-section__header" onClick={() => toggle(id)}>
        <span className="sandbox-section__chevron">{isOpen ? "\u25BC" : "\u25B6"}</span>
        {title}
      </button>
      {isOpen && <div className="sandbox-section__body">{children}</div>}
    </div>
  );
}

function Dim({ label, value, range }: { label: string; value: number; range?: [number, number] }) {
  const [min, max] = range || [0, 1];
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="sandbox-dim">
      <span className="sandbox-dim__label">{label}</span>
      <div className="sandbox-dim__bar">
        <div className="sandbox-dim__fill" style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
      </div>
      <span className="sandbox-dim__value">{value.toFixed(3)}</span>
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="sandbox-kv">
      <span className="sandbox-kv__label">{label}</span>
      <span className="sandbox-kv__value">{value}</span>
    </div>
  );
}
