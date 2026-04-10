import { useState, useCallback } from "react";
import type { ArenaContestant, ArenaEntry, ArenaDivergence } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { EmotionAvatar } from "./EmotionAvatar";
import * as api from "../api/client";
import "./ArenaPanel.css";

interface Props {
  sessionId: string;
  connected: boolean;
}

const BIG_FIVE_SHORT = { openness: "O", conscientiousness: "C", extraversion: "E", agreeableness: "A", neuroticism: "N" } as const;
const BIG_FIVE_KEYS = Object.keys(BIG_FIVE_SHORT) as (keyof typeof BIG_FIVE_SHORT)[];

// 10 polarized presets — each is an EXTREME version of what it represents
// Values are pushed to the edges (0.05-0.95) to maximize divergence in arena results
const PRESET_CONTESTANTS: (ArenaContestant & { desc: string })[] = [
  { name: "Neurotic",      desc: "Anxiety-driven, overthinks everything, emotionally volatile",
    personality: { openness: 0.30, conscientiousness: 0.65, extraversion: 0.15, agreeableness: 0.55, neuroticism: 0.95 } },
  { name: "Fearless",      desc: "Emotionally bulletproof, rushes in, zero hesitation",
    personality: { openness: 0.70, conscientiousness: 0.25, extraversion: 0.90, agreeableness: 0.20, neuroticism: 0.05 } },
  { name: "Stoic",         desc: "Iron discipline, suppresses emotion, values duty above all",
    personality: { openness: 0.20, conscientiousness: 0.95, extraversion: 0.15, agreeableness: 0.40, neuroticism: 0.05 } },
  { name: "Creative",      desc: "Wild imagination, chaotic, breaks every convention",
    personality: { openness: 0.95, conscientiousness: 0.10, extraversion: 0.65, agreeableness: 0.40, neuroticism: 0.55 } },
  { name: "Empath",        desc: "Absorbs everyone's pain, people-pleaser to the extreme",
    personality: { openness: 0.60, conscientiousness: 0.35, extraversion: 0.50, agreeableness: 0.95, neuroticism: 0.70 } },
  { name: "Narcissist",    desc: "Self-centered, dominant, dismissive of others' feelings",
    personality: { openness: 0.45, conscientiousness: 0.50, extraversion: 0.85, agreeableness: 0.05, neuroticism: 0.35 } },
  { name: "Hermit",        desc: "Deeply introverted, avoids all social contact, rich inner world",
    personality: { openness: 0.75, conscientiousness: 0.60, extraversion: 0.05, agreeableness: 0.50, neuroticism: 0.40 } },
  { name: "Volatile",      desc: "Explosive emotions, intense highs and lows, passionate",
    personality: { openness: 0.70, conscientiousness: 0.10, extraversion: 0.80, agreeableness: 0.30, neuroticism: 0.90 } },
  { name: "Perfectionist", desc: "Obsessive control, rigid standards, fears failure deeply",
    personality: { openness: 0.15, conscientiousness: 0.95, extraversion: 0.30, agreeableness: 0.35, neuroticism: 0.80 } },
  { name: "Free Spirit",   desc: "No rules, no worries, lives in the moment, carefree",
    personality: { openness: 0.90, conscientiousness: 0.05, extraversion: 0.75, agreeableness: 0.70, neuroticism: 0.10 } },
];

const EXAMPLE_SCENARIOS = [
  "You just found out a close friend has been lying to you for months.",
  "A stranger on the street gives you a genuine, unexpected compliment.",
  "You receive news that you didn't get the job you really wanted.",
  "Someone you care about tells you they're proud of who you've become.",
  "You witness an act of injustice but feel powerless to intervene.",
  "After months of effort, you finally solve a problem that seemed impossible.",
];

export function ArenaPanel({ sessionId, connected }: Props) {
  const [scenario, setScenario] = useState("");
  const [contestants, setContestants] = useState<ArenaContestant[]>([
    { name: PRESET_CONTESTANTS[0].name, personality: { ...PRESET_CONTESTANTS[0].personality } },
    { name: PRESET_CONTESTANTS[1].name, personality: { ...PRESET_CONTESTANTS[1].personality } },
    { name: PRESET_CONTESTANTS[2].name, personality: { ...PRESET_CONTESTANTS[2].personality } },
  ]);
  const [loading, setLoading] = useState(false);
  const [entries, setEntries] = useState<ArenaEntry[]>([]);
  const [divergence, setDivergence] = useState<ArenaDivergence | null>(null);

  const [expandedContestant, setExpandedContestant] = useState<string | null>(null);

  const toggleContestant = (preset: typeof PRESET_CONTESTANTS[number]) => {
    const exists = contestants.some(c => c.name === preset.name);
    if (exists) {
      if (contestants.length <= 2) return; // minimum 2
      setContestants(prev => prev.filter(c => c.name !== preset.name));
      if (expandedContestant === preset.name) setExpandedContestant(null);
    } else {
      if (contestants.length >= 10) return;
      setContestants(prev => [...prev, { name: preset.name, personality: { ...preset.personality } }]);
    }
  };

  const updateContestant = (name: string, key: string, val: number) => {
    setContestants(prev => prev.map(c =>
      c.name === name ? { ...c, personality: { ...c.personality, [key]: val } } : c
    ));
  };

  const resetContestant = (name: string) => {
    const preset = PRESET_CONTESTANTS.find(p => p.name === name);
    if (preset) {
      setContestants(prev => prev.map(c =>
        c.name === name ? { ...c, personality: { ...preset.personality } } : c
      ));
    }
  };

  const handleCompare = useCallback(async () => {
    if (!scenario.trim() || !connected || contestants.length < 2) return;
    setLoading(true);
    try {
      const res = await api.arenaCompare(scenario.trim(), contestants, sessionId);
      setEntries(res.entries);
      setDivergence(res.divergence);
    } catch (err) {
      console.error("Arena compare failed:", err);
    } finally {
      setLoading(false);
    }
  }, [scenario, contestants, sessionId, connected]);

  return (
    <div className="arena">
      {/* Setup panel */}
      <div className="arena__setup">
        <h2 className="arena__title">Multi-Personality Arena</h2>

        <textarea
          className="arena__textarea"
          value={scenario}
          onChange={e => setScenario(e.target.value)}
          placeholder="Describe a scenario to test across personalities..."
          rows={2}
        />
        <div className="arena__examples">
          {EXAMPLE_SCENARIOS.map((ex, i) => (
            <button key={i} className="arena__example-btn" onClick={() => setScenario(ex)}>
              {ex.slice(0, 50)}...
            </button>
          ))}
        </div>

        {/* Contestants — select from presets */}
        <div className="arena__contestants">
          <div className="arena__contestants-header">
            <span className="arena__section-label">Contestants ({contestants.length}/10)</span>
          </div>

          <div className="arena__preset-grid">
            {PRESET_CONTESTANTS.map(p => {
              const active = contestants.some(c => c.name === p.name);
              const contestant = contestants.find(c => c.name === p.name);
              const isExpanded = expandedContestant === p.name;
              return (
                <div key={p.name} className={`arena__preset-wrap ${active ? "arena__preset-wrap--active" : ""}`}>
                  <button
                    className={`arena__preset-card ${active ? "arena__preset-card--active" : ""}`}
                    onClick={() => toggleContestant(p)}
                  >
                    <span className="arena__preset-name">{p.name}</span>
                    <span className="arena__preset-desc">{p.desc}</span>
                  </button>
                  {active && (
                    <button
                      className="arena__tune-btn"
                      onClick={(e) => { e.stopPropagation(); setExpandedContestant(isExpanded ? null : p.name); }}
                      title="Tune personality"
                    >
                      {isExpanded ? "\u25B2" : "\u25BC"}
                    </button>
                  )}
                  {active && isExpanded && contestant && (
                    <div className="arena__sliders">
                      {(Object.keys(BIG_FIVE_SHORT) as (keyof typeof BIG_FIVE_SHORT)[]).map(k => (
                        <div key={k} className="arena__mini-slider">
                          <span className="arena__mini-label">{BIG_FIVE_SHORT[k]}</span>
                          <input type="range" min={0} max={1} step={0.05} value={contestant.personality[k]} onChange={e => updateContestant(p.name, k, parseFloat(e.target.value))} className="arena__slider" />
                          <span className="arena__mini-val">{contestant.personality[k].toFixed(2)}</span>
                        </div>
                      ))}
                      <button className="arena__reset-btn" onClick={() => resetContestant(p.name)}>Reset</button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <button
          className="arena__run-btn"
          onClick={handleCompare}
          disabled={loading || !connected || !scenario.trim() || contestants.length < 2}
        >
          {loading ? "Comparing..." : "Compare"}
        </button>
      </div>

      {/* Results */}
      <div className="arena__results">
        {divergence && <DivergenceCard divergence={divergence} />}

        {entries.length > 0 ? (
          <div className="arena__columns" style={{ gridTemplateColumns: `repeat(${entries.length}, 1fr)` }}>
            {entries.map((entry, i) => (
              <ArenaColumn key={i} entry={entry} />
            ))}
          </div>
        ) : !loading ? (
          <div className="arena__empty">Set up contestants and run a comparison</div>
        ) : null}
      </div>
    </div>
  );
}

// --- Divergence summary card ---

function DivergenceCard({ divergence }: { divergence: ArenaDivergence }) {
  return (
    <div className="arena-divergence">
      <div className="arena-divergence__title">Divergence Analysis</div>
      <div className="arena-divergence__grid">
        <div className="arena-divergence__metric">
          <span className="arena-divergence__label">Valence Spread</span>
          <span className="arena-divergence__value">{divergence.valence_spread.toFixed(3)}</span>
          <div className="arena-divergence__bar">
            <div className="arena-divergence__fill" style={{ width: `${(divergence.valence_spread / 2) * 100}%` }} />
          </div>
        </div>
        <div className="arena-divergence__metric">
          <span className="arena-divergence__label">Arousal Spread</span>
          <span className="arena-divergence__value">{divergence.arousal_spread.toFixed(3)}</span>
          <div className="arena-divergence__bar">
            <div className="arena-divergence__fill" style={{ width: `${divergence.arousal_spread * 100}%` }} />
          </div>
        </div>
        <div className="arena-divergence__metric">
          <span className="arena-divergence__label">Intensity Spread</span>
          <span className="arena-divergence__value">{divergence.intensity_spread.toFixed(3)}</span>
          <div className="arena-divergence__bar">
            <div className="arena-divergence__fill" style={{ width: `${divergence.intensity_spread * 100}%` }} />
          </div>
        </div>
        <div className="arena-divergence__metric">
          <span className="arena-divergence__label">Emotion Diversity</span>
          <span className="arena-divergence__value">{divergence.emotion_diversity} distinct</span>
        </div>
      </div>
      <div className="arena-divergence__extremes">
        <span>Most positive: <strong>{divergence.most_positive}</strong></span>
        <span>Most negative: <strong>{divergence.most_negative}</strong></span>
        <span>Most intense: <strong>{divergence.most_intense}</strong></span>
        <span>Most calm: <strong>{divergence.most_calm}</strong></span>
      </div>
    </div>
  );
}

// --- Single personality column ---

function ArenaColumn({ entry }: { entry: ArenaEntry }) {
  const s = entry.result.emotional_state;
  const color = EMOTION_COLORS[s.primary_emotion] || EMOTION_COLORS.neutral;

  // Top 5 active emotions
  const topEmotions = Object.entries(s.emotional_stack)
    .filter(([, v]) => v > 0.02)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="arena-col">
      <div className="arena-col__header" style={{ borderTopColor: color }}>
        <div className="arena-col__name">{entry.name}</div>
        <div className="arena-col__personality">
          {BIG_FIVE_KEYS.map(k => (
            <span key={k} className="arena-col__trait">
              {BIG_FIVE_SHORT[k]}:{entry.personality[k].toFixed(1)}
            </span>
          ))}
        </div>
      </div>

      {/* Per-contestant Avatar */}
      <div className="arena-col__avatar">
        <EmotionAvatar emotionalState={s} />
      </div>

      {/* Primary emotion */}
      <div className="arena-col__emotion" style={{ color }}>
        {s.primary_emotion}
        {s.secondary_emotion && s.secondary_emotion !== s.primary_emotion && (
          <span className="arena-col__secondary"> / {s.secondary_emotion}</span>
        )}
      </div>

      {/* Core dimensions */}
      <div className="arena-col__dims">
        <DimBar label="V" value={s.valence} min={-1} max={1} />
        <DimBar label="A" value={s.arousal} />
        <DimBar label="D" value={s.dominance} />
        <DimBar label="I" value={s.intensity} />
      </div>

      {/* Emotional stack */}
      <div className="arena-col__stack">
        {topEmotions.map(([emo, val]) => (
          <div key={emo} className="arena-col__stack-row">
            <span className="arena-col__stack-label" style={{ color: EMOTION_COLORS[emo as keyof typeof EMOTION_COLORS] || "#999" }}>
              {emo}
            </span>
            <div className="arena-col__stack-bar">
              <div className="arena-col__stack-fill" style={{
                width: `${val * 100}%`,
                background: EMOTION_COLORS[emo as keyof typeof EMOTION_COLORS] || "#666",
              }} />
            </div>
            <span className="arena-col__stack-val">{val.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {/* Body state */}
      <div className="arena-col__section-title">Body</div>
      <div className="arena-col__body">
        <MiniBar label="E" value={s.body_state.energy} />
        <MiniBar label="T" value={s.body_state.tension} />
        <MiniBar label="O" value={s.body_state.openness} />
        <MiniBar label="W" value={s.body_state.warmth} />
      </div>

      {/* Key systems */}
      <div className="arena-col__section-title">Systems</div>
      <div className="arena-col__systems">
        {entry.result.regulation.strategy_used && (
          <div className="arena-col__sys-row">
            <span className="arena-col__sys-label">Regulation</span>
            <span className="arena-col__sys-value">{entry.result.regulation.strategy_used}</span>
          </div>
        )}
        {entry.result.reappraisal.applied && (
          <div className="arena-col__sys-row">
            <span className="arena-col__sys-label">Reappraisal</span>
            <span className="arena-col__sys-value">{entry.result.reappraisal.strategy}</span>
          </div>
        )}
        {entry.result.meta_emotion.active && (
          <div className="arena-col__sys-row">
            <span className="arena-col__sys-label">Meta</span>
            <span className="arena-col__sys-value">{entry.result.meta_emotion.meta_response}</span>
          </div>
        )}
        {entry.result.creativity.thinking_mode !== "standard" && (
          <div className="arena-col__sys-row">
            <span className="arena-col__sys-label">Creativity</span>
            <span className="arena-col__sys-value">{entry.result.creativity.thinking_mode}</span>
          </div>
        )}
        {entry.result.immune.protection_mode !== "none" && (
          <div className="arena-col__sys-row">
            <span className="arena-col__sys-label">Immune</span>
            <span className="arena-col__sys-value">{entry.result.immune.protection_mode}</span>
          </div>
        )}
        {entry.result.emergent_emotions.length > 0 && (
          <div className="arena-col__sys-row">
            <span className="arena-col__sys-label">Emergent</span>
            <span className="arena-col__sys-value">{entry.result.emergent_emotions.join(", ")}</span>
          </div>
        )}
      </div>

      {/* Authenticity */}
      <div className="arena-col__section-title">Authenticity</div>
      <div className="arena-col__auth">
        <span>{entry.result.authenticity_metrics.overall.toFixed(2)}</span>
      </div>

      {/* LLM Response */}
      {entry.response && (
        <>
          <div className="arena-col__section-title">Response</div>
          <div className="arena-col__response">{entry.response}</div>
        </>
      )}
    </div>
  );
}

function DimBar({ label, value, min = 0, max = 1 }: { label: string; value: number; min?: number; max?: number }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="arena-dim">
      <span className="arena-dim__label">{label}</span>
      <div className="arena-dim__bar">
        <div className="arena-dim__fill" style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
      </div>
      <span className="arena-dim__value">{value.toFixed(2)}</span>
    </div>
  );
}

function MiniBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="arena-mini">
      <span className="arena-mini__label">{label}</span>
      <div className="arena-mini__bar">
        <div className="arena-mini__fill" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}
