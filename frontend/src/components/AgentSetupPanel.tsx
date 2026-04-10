import { useState, useEffect, useRef } from "react";
import * as api from "../api/client";
import "./AgentSetupPanel.css";

interface Props {
  sessionId: string;
  visible: boolean;
  onClose: () => void;
}

const PERSONALITY_PRESETS: Record<string, Record<string, number>> = {
  balanced: { openness: 0.6, conscientiousness: 0.6, extraversion: 0.5, agreeableness: 0.6, neuroticism: 0.4, emotional_granularity: 0.6, emotional_reactivity: 0.5, emotional_recovery: 0.5 },
  sensitive: { openness: 0.7, conscientiousness: 0.4, extraversion: 0.4, agreeableness: 0.7, neuroticism: 0.8, emotional_granularity: 0.8, emotional_reactivity: 0.8, emotional_recovery: 0.3 },
  resilient: { openness: 0.5, conscientiousness: 0.8, extraversion: 0.5, agreeableness: 0.5, neuroticism: 0.15, emotional_granularity: 0.5, emotional_reactivity: 0.3, emotional_recovery: 0.9 },
  creative: { openness: 0.95, conscientiousness: 0.4, extraversion: 0.6, agreeableness: 0.5, neuroticism: 0.5, emotional_granularity: 0.9, emotional_reactivity: 0.6, emotional_recovery: 0.5 },
  empathic: { openness: 0.6, conscientiousness: 0.5, extraversion: 0.6, agreeableness: 0.95, neuroticism: 0.5, emotional_granularity: 0.7, emotional_reactivity: 0.6, emotional_recovery: 0.5 },
};

const BIG_FIVE_LABELS: Record<string, { label: string; low: string; high: string }> = {
  openness: { label: "Openness", low: "Practical", high: "Curious" },
  conscientiousness: { label: "Conscientiousness", low: "Flexible", high: "Disciplined" },
  extraversion: { label: "Extraversion", low: "Reserved", high: "Energetic" },
  agreeableness: { label: "Agreeableness", low: "Challenging", high: "Empathic" },
  neuroticism: { label: "Neuroticism", low: "Stable", high: "Reactive" },
};

const VALUE_LABELS: Record<string, string> = {
  truth: "Truth",
  compassion: "Compassion",
  fairness: "Fairness",
  growth: "Growth",
  creativity: "Creativity",
};

export function AgentSetupPanel({ sessionId, visible, onClose }: Props) {
  const [tab, setTab] = useState<"personality" | "values" | "identity">("personality");
  const [personality, setPersonality] = useState<Record<string, number>>({});
  const [activePreset, setActivePreset] = useState("balanced");
  const [values, setValues] = useState<Array<{ name: string; weight: number; description: string }>>([]);
  const [agentName, setAgentName] = useState("");
  const [background, setBackground] = useState("");
  const [saving, setSaving] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    }
    if (visible) {
      setTimeout(() => document.addEventListener("mousedown", handleClick), 0);
    }
    return () => document.removeEventListener("mousedown", handleClick);
  }, [visible, onClose]);

  // Load data when panel opens
  useEffect(() => {
    if (!visible) return;
    api.getPersonality(sessionId).then(d => {
      setPersonality(d.personality);
      // Detect active preset
      for (const [name, preset] of Object.entries(PERSONALITY_PRESETS)) {
        const match = Object.keys(BIG_FIVE_LABELS).every(k =>
          Math.abs((preset[k] ?? 0) - (d.personality[k] ?? 0)) < 0.05
        );
        if (match) { setActivePreset(name); break; }
      }
    }).catch(() => {});
    api.getValues(sessionId).then(d => setValues(d.core)).catch(() => {});
    api.getIdentity(sessionId).then(d => {
      const nameStmt = d.statements.find((s: Record<string, unknown>) => s.trigger_category === "identity");
      if (nameStmt) {
        const m = String(nameStmt.statement).match(/My name is (.+)/);
        if (m) setAgentName(m[1]);
      }
      const bgStmt = d.statements.find((s: Record<string, unknown>) => s.trigger_category === "background");
      if (bgStmt) setBackground(String(bgStmt.statement));
    }).catch(() => {});
  }, [visible, sessionId]);

  async function applyPreset(name: string) {
    setActivePreset(name);
    const preset = PERSONALITY_PRESETS[name];
    setPersonality(prev => ({ ...prev, ...preset }));
    setSaving(true);
    await api.setPersonality(sessionId, preset).catch(console.error);
    setSaving(false);
  }

  async function handleSlider(key: string, val: number) {
    setPersonality(prev => ({ ...prev, [key]: val }));
    setActivePreset("");
    await api.setPersonality(sessionId, { [key]: val }).catch(console.error);
  }

  async function handleValue(name: string, weight: number) {
    setValues(prev => prev.map(v => v.name === name ? { ...v, weight } : v));
    await api.setValues(sessionId, { [name]: weight }).catch(console.error);
  }

  async function saveIdentity() {
    setSaving(true);
    await api.setIdentity(sessionId, { name: agentName, background }).catch(console.error);
    setSaving(false);
  }

  if (!visible) return null;

  return (
    <div className="agent-setup" ref={panelRef}>
      <div className="agent-setup__header">
        <span className="agent-setup__title">Agent Setup</span>
        <button className="agent-setup__close" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Tabs */}
      <div className="agent-setup__tabs">
        {(["personality", "values", "identity"] as const).map(t => (
          <button
            key={t}
            className={`agent-setup__tab ${tab === t ? "agent-setup__tab--active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "personality" ? "Personality" : t === "values" ? "Values" : "Identity"}
          </button>
        ))}
      </div>

      {/* Personality Tab */}
      {tab === "personality" && (
        <div className="agent-setup__content">
          <label className="agent-setup__label">Presets</label>
          <div className="agent-setup__presets">
            {Object.keys(PERSONALITY_PRESETS).map(name => (
              <button
                key={name}
                className={`agent-setup__preset ${activePreset === name ? "agent-setup__preset--active" : ""}`}
                onClick={() => applyPreset(name)}
              >
                {name}
              </button>
            ))}
          </div>

          <label className="agent-setup__label">Big Five</label>
          {Object.entries(BIG_FIVE_LABELS).map(([key, info]) => (
            <div key={key} className="agent-setup__slider-row">
              <div className="agent-setup__slider-header">
                <span className="agent-setup__slider-label">{info.label}</span>
                <span className="agent-setup__slider-value">{(personality[key] ?? 0.5).toFixed(2)}</span>
              </div>
              <div className="agent-setup__slider-track">
                <span className="agent-setup__slider-end">{info.low}</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={personality[key] ?? 0.5}
                  onChange={e => handleSlider(key, parseFloat(e.target.value))}
                />
                <span className="agent-setup__slider-end">{info.high}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Values Tab */}
      {tab === "values" && (
        <div className="agent-setup__content">
          <label className="agent-setup__label">Core Values</label>
          {values.map(v => (
            <div key={v.name} className="agent-setup__slider-row">
              <div className="agent-setup__slider-header">
                <span className="agent-setup__slider-label">{VALUE_LABELS[v.name] ?? v.name}</span>
                <span className="agent-setup__slider-value">{v.weight.toFixed(2)}</span>
              </div>
              <div className="agent-setup__slider-desc">{v.description}</div>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={v.weight}
                onChange={e => handleValue(v.name, parseFloat(e.target.value))}
              />
            </div>
          ))}
        </div>
      )}

      {/* Identity Tab */}
      {tab === "identity" && (
        <div className="agent-setup__content">
          <label className="agent-setup__label">Agent Name</label>
          <input
            className="agent-setup__text-input"
            type="text"
            placeholder="Pathos"
            value={agentName}
            onChange={e => setAgentName(e.target.value)}
          />

          <label className="agent-setup__label">Background</label>
          <textarea
            className="agent-setup__textarea"
            placeholder="Describe the agent's personality, background, or role..."
            value={background}
            onChange={e => setBackground(e.target.value)}
            rows={4}
          />

          <button
            className="agent-setup__save-btn"
            onClick={saveIdentity}
            disabled={saving}
          >
            {saving ? "Saving..." : "Save Identity"}
          </button>
        </div>
      )}

      {saving && <div className="agent-setup__saving">Applying...</div>}
    </div>
  );
}
