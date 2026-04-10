import type { EmotionalState } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { CircumplexChart } from "./CircumplexChart";
import { BodyStateDisplay } from "./BodyStateDisplay";
import "./EmotionalStatePanel.css";

interface Props {
  state: EmotionalState;
  history: Array<{ valence: number; arousal: number; emotion: string }>;
}

export function EmotionalStatePanel({ state, history }: Props) {
  const color = EMOTION_COLORS[state.primary_emotion];

  return (
    <div className="emotion-panel">
      <div className="emotion-panel__header">
        <h2>Emotional State</h2>
        <span className="emotion-panel__live">LIVE</span>
      </div>

      <CircumplexChart state={state} history={history} />

      <div className="emotion-panel__info">
        <div className="emotion-panel__primary">
          <span
            className="emotion-pill"
            style={{ backgroundColor: color + "30", borderColor: color }}
          >
            {state.primary_emotion}
          </span>
          <span className="emotion-intensity">
            {(state.intensity * 100).toFixed(0)}%
          </span>
        </div>

        {state.secondary_emotion && (
          <div className="emotion-panel__secondary">
            Secondary:{" "}
            <span
              className="emotion-pill emotion-pill--small"
              style={{
                backgroundColor: EMOTION_COLORS[state.secondary_emotion] + "30",
                borderColor: EMOTION_COLORS[state.secondary_emotion],
              }}
            >
              {state.secondary_emotion}
            </span>
          </div>
        )}

        <div className="emotion-panel__dimensions">
          <DimensionRow label="Valence" value={state.valence} min={-1} max={1} />
          <DimensionRow label="Arousal" value={state.arousal} min={0} max={1} />
          <DimensionRow label="Dominance" value={state.dominance} min={0} max={1} />
          <DimensionRow label="Certainty" value={state.certainty} min={0} max={1} />
        </div>
      </div>

      <BodyStateDisplay bodyState={state.body_state} />

      <div className="emotion-panel__mood">
        <h3>Mood</h3>
        <div className="mood-info">
          <span className="mood-label">{state.mood.label}</span>
          <span className={`mood-trend mood-trend--${state.mood.trend}`}>
            {state.mood.trend === "improving" ? "\u2191" : state.mood.trend === "declining" ? "\u2193" : "\u2192"}
            {" "}{state.mood.trend}
          </span>
        </div>
        <div className="mood-baseline">
          Baseline: V={state.mood.baseline_valence.toFixed(2)} A={state.mood.baseline_arousal.toFixed(2)}
          {" "}| Stability: {(state.mood.stability * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
}

function DimensionRow({ label, value, min, max }: { label: string; value: number; min: number; max: number }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="dimension-row">
      <span className="dimension-label">{label}</span>
      <div className="dimension-track">
        <div className="dimension-fill" style={{ width: `${pct}%` }} />
        <div className="dimension-marker" style={{ left: `${pct}%` }} />
      </div>
      <span className="dimension-value">{value.toFixed(2)}</span>
    </div>
  );
}
