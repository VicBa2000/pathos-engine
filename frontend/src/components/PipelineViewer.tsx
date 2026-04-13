import { useState, useEffect, useRef } from "react";
import type { PipelineTrace, PipelineStep, PipelineImpact } from "../types/emotion";
import "./PipelineViewer.css";

interface Props {
  trace: PipelineTrace | null;
  detailed: boolean;
  onToggleDetailed: () => void;
}

const IMPACT_COLORS: Record<PipelineImpact, string> = {
  none: "#2a2a4a",
  low: "#3a5a6a",
  medium: "#6a5acd",
  high: "#ff6b35",
};

const IMPACT_GLOW: Record<PipelineImpact, string> = {
  none: "transparent",
  low: "rgba(58, 90, 106, 0.3)",
  medium: "rgba(106, 90, 205, 0.4)",
  high: "rgba(255, 107, 53, 0.5)",
};

const STEP_ICONS: Record<string, string> = {
  homeostasis: "\u2696",      // balance scale
  temporal_pre: "\u23f3",     // hourglass
  appraisal: "\u{1f9e0}",    // brain
  needs: "\u2764",            // heart
  schemas: "\u{1f4a1}",      // lightbulb (pattern)
  social: "\u{1f465}",       // people
  contagion: "\u{1f30a}",    // wave
  somatic: "\u26a1",         // lightning
  forecast_eval: "\u{1f4ca}",// chart
  emotion_gen: "\u2728",     // sparkles
  calibration: "\u{1f527}",  // wrench
  reappraisal: "\u{1f504}",  // cycle
  regulation: "\u{1f6e1}",   // shield
  temporal: "\u{1f570}",     // clock
  immune: "\u{1f9ec}",       // dna
  narrative: "\u{1f4d6}",    // book
  meta_emotion: "\u{1f6aa}",    // mirror/door
  emergent: "\u{1f33f}",     // sprout
  creativity: "\u{1f3a8}",   // palette
  forecasting: "\u{1f52e}",  // crystal ball
  interoception: "\u{1f9e0}",  // brain (body->emotion)
  post_processing: "\u{1f4be}",// floppy
  behavior: "\u{1f3ad}",     // theater masks
  steering: "\u{1f9f2}",     // magnet (hidden state modification)
  emotional_prefix: "\u{1f3f7}",// label (embedding injection)
  attention: "\u{1f441}",    // eye (attention modulation)
  llm_response: "\u{1f4ac}",  // speech bubble
  self_appraisal: "\u{1fa9e}",// mirror (self-evaluation)
  world_model: "\u{1f30d}",  // globe (predictive world model)
  voice: "\u{1f50a}",        // speaker
};

export function PipelineViewer({ trace, detailed, onToggleDetailed }: Props) {
  const [expandedStep, setExpandedStep] = useState<string | null>(null);
  const [animatedSteps, setAnimatedSteps] = useState<Set<string>>(new Set());
  const prevTraceRef = useRef<PipelineTrace | null>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Animate steps sequentially when new trace arrives
  useEffect(() => {
    if (!trace || trace === prevTraceRef.current) return;
    prevTraceRef.current = trace;

    // Clear previous timers
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];

    setAnimatedSteps(new Set());
    const activeSteps = trace.steps.filter(s => s.active);
    activeSteps.forEach((step, i) => {
      const id = setTimeout(() => {
        setAnimatedSteps(prev => new Set([...prev, step.name]));
      }, i * 80);
      timersRef.current.push(id);
    });

    return () => {
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
    };
  }, [trace]);

  if (!trace) {
    return (
      <div className="pipeline">
        <div className="pipeline__header">
          <span className="pipeline__title">Pipeline Viewer</span>
        </div>
        <div className="pipeline__empty">
          <div className="pipeline__empty-icon">{"\u{1f50c}"}</div>
          <p>Send a message to see the emotional pipeline in action</p>
        </div>
      </div>
    );
  }

  const activeCount = trace.steps.filter(s => s.active).length;
  const highImpactCount = trace.steps.filter(s => s.impact === "high").length;

  return (
    <div className="pipeline">
      <div className="pipeline__header">
        <span className="pipeline__title">Pipeline Viewer</span>
        <span className="pipeline__meta">
          {activeCount}/{trace.steps.length} active
          {" \u00b7 "}
          {trace.total_duration_ms.toFixed(0)}ms
        </span>
      </div>

      <div className="pipeline__controls">
        <button
          className={`pipeline__detail-toggle ${detailed ? "pipeline__detail-toggle--on" : ""}`}
          onClick={onToggleDetailed}
        >
          {detailed ? "Detailed" : "Simple"}
        </button>
        <span className={`pipeline__mode-badge pipeline__mode-badge--${trace.mode}`}>
          {trace.mode}
        </span>
      </div>

      {/* Impact summary bar */}
      <div className="pipeline__impact-bar">
        {trace.steps.filter(s => s.active).map(step => (
          <div
            key={step.name}
            className="pipeline__impact-segment"
            style={{
              backgroundColor: IMPACT_COLORS[step.impact],
              flex: Math.max(step.duration_ms, 1),
            }}
            title={`${step.label}: ${step.impact} impact (${step.duration_ms.toFixed(0)}ms)`}
          />
        ))}
      </div>

      <div className="pipeline__flow">
        {trace.steps.map((step, idx) => (
          <PipelineNode
            key={step.name}
            step={step}
            index={idx}
            isLast={idx === trace.steps.length - 1}
            detailed={detailed}
            expanded={expandedStep === step.name}
            animated={animatedSteps.has(step.name)}
            highImpactCount={highImpactCount}
            onToggleExpand={() => setExpandedStep(prev => prev === step.name ? null : step.name)}
          />
        ))}
      </div>
    </div>
  );
}


function PipelineNode({
  step, index, isLast, detailed, expanded, animated, highImpactCount, onToggleExpand,
}: {
  step: PipelineStep;
  index: number;
  isLast: boolean;
  detailed: boolean;
  expanded: boolean;
  animated: boolean;
  highImpactCount: number;
  onToggleExpand: () => void;
}) {
  const icon = STEP_ICONS[step.name] || "\u2699";
  const impactColor = IMPACT_COLORS[step.impact];
  const glowColor = IMPACT_GLOW[step.impact];

  return (
    <div className={`pipeline-node ${!step.active ? "pipeline-node--inactive" : ""} ${animated ? "pipeline-node--animated" : ""}`}>
      {/* Connection line to next step */}
      {!isLast && (
        <div className={`pipeline-node__connector ${step.active ? "pipeline-node__connector--active" : ""}`}>
          <div
            className="pipeline-node__connector-line"
            style={step.active ? { borderColor: impactColor } : undefined}
          />
        </div>
      )}

      {/* Main node */}
      <div
        className={`pipeline-node__card ${expanded ? "pipeline-node__card--expanded" : ""}`}
        onClick={onToggleExpand}
        style={step.active ? {
          borderColor: impactColor,
          boxShadow: `0 0 ${step.impact === "high" ? 12 : step.impact === "medium" ? 6 : 0}px ${glowColor}`,
        } : undefined}
      >
        {/* Left: icon + indicator */}
        <div className="pipeline-node__icon-wrap">
          <span
            className={`pipeline-node__icon ${step.impact === "high" ? "pipeline-node__icon--pulse" : ""}`}
            style={step.active ? { backgroundColor: impactColor } : undefined}
          >
            {icon}
          </span>
          {step.active && step.impact !== "none" && (
            <div
              className="pipeline-node__impact-dot"
              style={{ backgroundColor: impactColor }}
            />
          )}
        </div>

        {/* Center: info */}
        <div className="pipeline-node__info">
          <div className="pipeline-node__label-row">
            <span className="pipeline-node__label">{step.label}</span>
            {step.active && (
              <span className="pipeline-node__duration">{step.duration_ms.toFixed(0)}ms</span>
            )}
          </div>

          {step.active ? (
            <span className="pipeline-node__summary">{step.summary}</span>
          ) : (
            <span className="pipeline-node__skipped">{step.skipped_reason || "Skipped"}</span>
          )}

          {/* Delta badges */}
          {step.active && Object.keys(step.delta).length > 0 && (
            <div className="pipeline-node__deltas">
              {Object.entries(step.delta).map(([key, val]) => (
                <span
                  key={key}
                  className={`pipeline-node__delta ${val > 0 ? "pipeline-node__delta--pos" : "pipeline-node__delta--neg"}`}
                >
                  {key}: {val > 0 ? "+" : ""}{val.toFixed(3)}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Right: expand arrow */}
        {step.active && Object.keys(step.details).length > 0 && (
          <span className={`pipeline-node__expand ${expanded ? "pipeline-node__expand--open" : ""}`}>
            {"\u25b6"}
          </span>
        )}
      </div>

      {/* Expanded details */}
      {expanded && step.active && detailed && Object.keys(step.details).length > 0 && (
        <div className="pipeline-node__details">
          {Object.entries(step.details).map(([key, val]) => (
            <div key={key} className="pipeline-node__detail-row">
              <span className="pipeline-node__detail-key">{key}</span>
              <span className="pipeline-node__detail-val">
                {formatDetailValue(val)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Expanded summary (simple mode) */}
      {expanded && step.active && !detailed && Object.keys(step.details).length > 0 && (
        <div className="pipeline-node__details pipeline-node__details--simple">
          <SimplifiedDetails step={step} />
        </div>
      )}
    </div>
  );
}


function SimplifiedDetails({ step }: { step: PipelineStep }) {
  const details = step.details;

  switch (step.name) {
    case "appraisal": {
      const appraisal = details.appraisal as Record<string, number> | undefined;
      if (!appraisal) return null;
      return (
        <div className="pipeline-simple">
          <div className="pipeline-simple__bar-group">
            {Object.entries(appraisal).map(([key, val]) => (
              <div key={key} className="pipeline-simple__bar-row">
                <span className="pipeline-simple__bar-label">{key}</span>
                <div className="pipeline-simple__bar-track">
                  <div
                    className="pipeline-simple__bar-fill"
                    style={{
                      width: `${Math.abs(val as number) * 100}%`,
                      backgroundColor: (val as number) >= 0 ? "#6a5acd" : "#e23636",
                    }}
                  />
                </div>
                <span className="pipeline-simple__bar-val">{(val as number).toFixed(2)}</span>
              </div>
            ))}
          </div>
          {details.amplification !== undefined && (details.amplification as number) > 0 && (
            <div className="pipeline-simple__note">
              Memory amplified by {((details.amplification as number) * 100).toFixed(0)}%
            </div>
          )}
        </div>
      );
    }

    case "emotion_gen": {
      return (
        <div className="pipeline-simple">
          <div className="pipeline-simple__emotion-display">
            <span className="pipeline-simple__primary">
              {(details.primary as string || "").replace(/^./, (c: string) => c.toUpperCase())}
            </span>
            {details.secondary && (
              <span className="pipeline-simple__secondary">
                + {(details.secondary as string).replace(/^./, (c: string) => c.toUpperCase())}
              </span>
            )}
          </div>
          <div className="pipeline-simple__bar-group">
            {["valence", "arousal", "dominance", "intensity"].map(key => {
              const val = details[key] as number;
              if (val === undefined) return null;
              const isValence = key === "valence";
              const normalizedWidth = isValence ? Math.abs(val) * 50 : val * 100;
              return (
                <div key={key} className="pipeline-simple__bar-row">
                  <span className="pipeline-simple__bar-label">{key}</span>
                  <div className="pipeline-simple__bar-track">
                    <div
                      className="pipeline-simple__bar-fill"
                      style={{
                        width: `${normalizedWidth}%`,
                        marginLeft: isValence && val < 0 ? `${50 - normalizedWidth}%` : isValence ? "50%" : "0",
                        backgroundColor: isValence ? (val >= 0 ? "#85cdca" : "#e23636") : "#6a5acd",
                      }}
                    />
                    {isValence && <div className="pipeline-simple__bar-center" />}
                  </div>
                  <span className="pipeline-simple__bar-val">{val.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    case "contagion": {
      const strength = details.signal_strength as number || 0;
      return (
        <div className="pipeline-simple">
          <div className="pipeline-simple__bar-row">
            <span className="pipeline-simple__bar-label">signal</span>
            <div className="pipeline-simple__bar-track">
              <div
                className="pipeline-simple__bar-fill pipeline-simple__bar-fill--wave"
                style={{ width: `${strength * 100}%` }}
              />
            </div>
            <span className="pipeline-simple__bar-val">{(strength * 100).toFixed(0)}%</span>
          </div>
          {strength > 0.1 && (
            <div className="pipeline-simple__note">
              User emotion detected (v:{(details.detected_valence as number).toFixed(2)}, a:{(details.detected_arousal as number).toFixed(2)})
            </div>
          )}
        </div>
      );
    }

    case "regulation": {
      return (
        <div className="pipeline-simple">
          {details.strategy && (
            <div className="pipeline-simple__tag">{details.strategy as string}</div>
          )}
          <div className="pipeline-simple__bar-row">
            <span className="pipeline-simple__bar-label">capacity</span>
            <div className="pipeline-simple__bar-track">
              <div
                className="pipeline-simple__bar-fill"
                style={{
                  width: `${(details.capacity as number || 0) * 100}%`,
                  backgroundColor: (details.capacity as number || 0) > 0.5 ? "#85cdca" : "#e67e22",
                }}
              />
            </div>
            <span className="pipeline-simple__bar-val">{((details.capacity as number || 0) * 100).toFixed(0)}%</span>
          </div>
          {details.breakthrough && (
            <div className="pipeline-simple__alert">Breakthrough!</div>
          )}
        </div>
      );
    }

    case "creativity": {
      return (
        <div className="pipeline-simple">
          <div className="pipeline-simple__tag">{details.mode as string || "standard"}</div>
          <div className="pipeline-simple__bar-row">
            <span className="pipeline-simple__bar-label">level</span>
            <div className="pipeline-simple__bar-track">
              <div
                className="pipeline-simple__bar-fill"
                style={{
                  width: `${(details.level as number || 0) * 100}%`,
                  backgroundColor: "#f39c12",
                }}
              />
            </div>
            <span className="pipeline-simple__bar-val">{((details.level as number || 0) * 100).toFixed(0)}%</span>
          </div>
        </div>
      );
    }

    default: {
      // Generic: show numeric values as bars, strings as tags
      const numericEntries = Object.entries(details).filter(([, v]) => typeof v === "number");
      const stringEntries = Object.entries(details).filter(([, v]) => typeof v === "string" && v);
      const boolEntries = Object.entries(details).filter(([, v]) => typeof v === "boolean");
      return (
        <div className="pipeline-simple">
          {stringEntries.map(([key, val]) => (
            <div key={key} className="pipeline-simple__tag-row">
              <span className="pipeline-simple__bar-label">{key}</span>
              <span className="pipeline-simple__tag">{val as string}</span>
            </div>
          ))}
          {numericEntries.map(([key, val]) => (
            <div key={key} className="pipeline-simple__bar-row">
              <span className="pipeline-simple__bar-label">{key}</span>
              <div className="pipeline-simple__bar-track">
                <div
                  className="pipeline-simple__bar-fill"
                  style={{ width: `${Math.min(Math.abs(val as number), 1) * 100}%` }}
                />
              </div>
              <span className="pipeline-simple__bar-val">{(val as number).toFixed(3)}</span>
            </div>
          ))}
          {boolEntries.map(([key, val]) => (
            <div key={key} className="pipeline-simple__tag-row">
              <span className="pipeline-simple__bar-label">{key}</span>
              <span className={`pipeline-simple__bool ${val ? "pipeline-simple__bool--on" : ""}`}>
                {val ? "ON" : "OFF"}
              </span>
            </div>
          ))}
        </div>
      );
    }
  }
}


function formatDetailValue(val: unknown): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") return val.toFixed(4);
  if (typeof val === "boolean") return val ? "true" : "false";
  if (typeof val === "string") return val;
  if (Array.isArray(val)) return val.length ? val.join(", ") : "[]";
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}
