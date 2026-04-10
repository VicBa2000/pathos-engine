import { useState, useEffect } from "react";
import type { CalibrationProfile, CalibrationResult, PrimaryEmotion } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import type { ExportResult, ModelInfo } from "../api/client";
import * as api from "../api/client";
import "./CalibrationPanel.css";

const EMOTIONS: PrimaryEmotion[] = [
  "joy", "excitement", "gratitude", "hope", "contentment", "relief",
  "anger", "frustration", "fear", "anxiety",
  "sadness", "helplessness", "disappointment",
  "surprise", "alertness", "contemplation", "indifference", "mixed", "neutral",
];

interface Props {
  sessionId: string;
}

export function CalibrationPanel({ sessionId }: Props) {
  const [stimulus, setStimulus] = useState("");
  const [expectedEmotion, setExpectedEmotion] = useState<PrimaryEmotion>("joy");
  const [expectedValence, setExpectedValence] = useState(0.5);
  const [expectedArousal, setExpectedArousal] = useState(0.5);
  const [expectedIntensity, setExpectedIntensity] = useState(0.5);
  const [results, setResults] = useState<CalibrationResult[]>([]);
  const [profile, setProfile] = useState<CalibrationProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Export state
  const [ollamaModels, setOllamaModels] = useState<ModelInfo[]>([]);
  const [exportBaseModel, setExportBaseModel] = useState("");
  const [exportModelName, setExportModelName] = useState("pathos");
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);

  useEffect(() => {
    api.listModels()
      .then(m => {
        const ollama = (m ?? []).filter(x => x.provider === "ollama");
        setOllamaModels(ollama);
        if (ollama.length > 0 && !exportBaseModel) {
          setExportBaseModel(ollama[0].name);
        }
      })
      .catch(() => {});
  }, []);

  async function handleSubmitScenario(e: React.FormEvent) {
    e.preventDefault();
    if (!stimulus.trim() || loading) return;

    setLoading(true);
    setError(null);
    try {
      const result = await api.submitCalibrationScenario(
        {
          stimulus: stimulus.trim(),
          expected_emotion: expectedEmotion,
          expected_valence: expectedValence,
          expected_arousal: expectedArousal,
          expected_intensity: expectedIntensity,
        },
        sessionId,
      );
      setResults(prev => [...prev, result]);
      setStimulus("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function handleApply() {
    setLoading(true);
    setError(null);
    try {
      const p = await api.applyCalibration(sessionId);
      setProfile(p);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    setLoading(true);
    setError(null);
    try {
      await api.resetCalibration(sessionId);
      setResults([]);
      setProfile(null);
      setExportResult(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function handleExport() {
    if (!exportBaseModel || !exportModelName.trim()) return;
    setExporting(true);
    setExportResult(null);
    setError(null);
    try {
      const result = await api.exportModel(sessionId, exportBaseModel, exportModelName.trim());
      setExportResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="calibration-panel">
      <h2>Calibration Mode</h2>
      <p className="calibration-desc">
        Describe how <em>you</em> would feel in a scenario. The system compares its response to yours and builds a calibration profile.
      </p>

      <form className="calibration-form" onSubmit={handleSubmitScenario}>
        <label>
          <span>Stimulus</span>
          <textarea
            value={stimulus}
            onChange={e => setStimulus(e.target.value)}
            placeholder="Describe a scenario... e.g. 'Someone cancels plans at the last minute'"
            rows={3}
          />
        </label>

        <label>
          <span>Expected Emotion</span>
          <select value={expectedEmotion} onChange={e => setExpectedEmotion(e.target.value as PrimaryEmotion)}>
            {EMOTIONS.map(em => (
              <option key={em} value={em}>{em}</option>
            ))}
          </select>
        </label>

        <div className="calibration-sliders">
          <SliderField label="Valence" value={expectedValence} onChange={setExpectedValence} min={-1} max={1} step={0.05} />
          <SliderField label="Arousal" value={expectedArousal} onChange={setExpectedArousal} min={0} max={1} step={0.05} />
          <SliderField label="Intensity" value={expectedIntensity} onChange={setExpectedIntensity} min={0} max={1} step={0.05} />
        </div>

        <button type="submit" disabled={loading || !stimulus.trim()}>
          {loading ? "Processing..." : "Submit Scenario"}
        </button>
      </form>

      {error && <div className="calibration-error">{error}</div>}

      {results.length > 0 && (
        <div className="calibration-results">
          <h3>Scenarios ({results.length})</h3>
          {results.map((r, i) => (
            <div key={i} className="calibration-result">
              <div className="calibration-result__stimulus">"{r.scenario.stimulus}"</div>
              <div className="calibration-result__comparison">
                <div className="calibration-result__col">
                  <span className="calibration-result__heading">Expected</span>
                  <span
                    className="emotion-pill emotion-pill--small"
                    style={{
                      backgroundColor: EMOTION_COLORS[r.scenario.expected_emotion] + "30",
                      borderColor: EMOTION_COLORS[r.scenario.expected_emotion],
                    }}
                  >
                    {r.scenario.expected_emotion}
                  </span>
                  <span className="calibration-result__nums">
                    V={r.scenario.expected_valence.toFixed(2)} A={r.scenario.expected_arousal.toFixed(2)} I={r.scenario.expected_intensity.toFixed(2)}
                  </span>
                </div>
                <div className="calibration-result__arrow">vs</div>
                <div className="calibration-result__col">
                  <span className="calibration-result__heading">System</span>
                  <span
                    className="emotion-pill emotion-pill--small"
                    style={{
                      backgroundColor: EMOTION_COLORS[r.system_emotion] + "30",
                      borderColor: EMOTION_COLORS[r.system_emotion],
                    }}
                  >
                    {r.system_emotion}
                  </span>
                  <span className="calibration-result__nums">
                    V={r.system_valence.toFixed(2)} A={r.system_arousal.toFixed(2)} I={r.system_intensity.toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="calibration-result__deltas">
                <Delta label="V" value={r.valence_delta} />
                <Delta label="A" value={r.arousal_delta} />
                <Delta label="I" value={r.intensity_delta} />
                <span className={`calibration-result__match ${r.emotion_match ? "calibration-result__match--yes" : ""}`}>
                  {r.emotion_match ? "Match" : "Mismatch"}
                </span>
              </div>
            </div>
          ))}

          <div className="calibration-actions">
            <button onClick={handleApply} disabled={loading}>
              Apply Calibration ({results.length} scenarios)
            </button>
            <button className="calibration-actions__reset" onClick={handleReset} disabled={loading}>
              Reset
            </button>
          </div>
        </div>
      )}

      {profile && (
        <div className="calibration-profile">
          <h3>Active Profile</h3>
          <div className="calibration-profile__grid">
            <ProfileRow label="Valence Offset" value={profile.valence_offset} format="signed" />
            <ProfileRow label="Arousal Scale" value={profile.arousal_scale} format="multiplier" />
            <ProfileRow label="Intensity Scale" value={profile.intensity_scale} format="multiplier" />
            <ProfileRow label="Emotion Accuracy" value={profile.emotion_accuracy} format="percent" />
            <ProfileRow label="Scenarios Used" value={profile.scenarios_used} format="integer" />
          </div>
        </div>
      )}

      {profile && (
        <div className="calibration-export">
          <h3>Export as Ollama Model</h3>
          <p className="calibration-export__desc">
            Bake the emotional architecture + calibration into a standalone Ollama model.
            Use it from any Ollama-compatible app.
          </p>

          <div className="calibration-export__fields">
            <label>
              <span>Base Model</span>
              {ollamaModels.length > 0 ? (
                <select value={exportBaseModel} onChange={e => setExportBaseModel(e.target.value)}>
                  {ollamaModels.map(m => (
                    <option key={m.name} value={m.name}>{m.name} ({m.size})</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={exportBaseModel}
                  onChange={e => setExportBaseModel(e.target.value)}
                  placeholder="e.g. qwen3:4b"
                />
              )}
            </label>
            <label>
              <span>Model Name</span>
              <input
                type="text"
                value={exportModelName}
                onChange={e => setExportModelName(e.target.value)}
                placeholder="e.g. pathos"
              />
            </label>
          </div>

          <button
            className="calibration-export__btn"
            onClick={handleExport}
            disabled={exporting || !exportBaseModel || !exportModelName.trim()}
          >
            {exporting ? "Exporting..." : `Export as ${exportModelName || "..."}:latest`}
          </button>

          {exportResult && (
            <div className={`calibration-export__result calibration-export__result--${exportResult.status === "model_created" ? "success" : "warning"}`}>
              {exportResult.status === "model_created" ? (
                <>
                  <div className="calibration-export__result-title">Model created!</div>
                  <div className="calibration-export__result-body">
                    Run: <code>ollama run {exportResult.model}</code>
                  </div>
                  <div className="calibration-export__result-body">
                    Or select <strong>{exportResult.model}</strong> from the model selector above.
                  </div>
                </>
              ) : (
                <>
                  <div className="calibration-export__result-title">Modelfile saved</div>
                  <div className="calibration-export__result-body">
                    {exportResult.ollama_error && <div>Ollama: {exportResult.ollama_error}</div>}
                    {exportResult.hint && <div>{exportResult.hint}</div>}
                    {exportResult.modelfile_path && <div>Path: <code>{exportResult.modelfile_path}</code></div>}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SliderField({
  label, value, onChange, min, max, step,
}: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number;
}) {
  return (
    <div className="slider-field">
      <span className="slider-field__label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
      />
      <span className="slider-field__value">{value.toFixed(2)}</span>
    </div>
  );
}

function Delta({ label, value }: { label: string; value: number }) {
  const color = Math.abs(value) < 0.1 ? "#2ecc71" : Math.abs(value) < 0.3 ? "#f1c40f" : "#e74c3c";
  return (
    <span className="calibration-delta" style={{ color }}>
      {label}={value >= 0 ? "+" : ""}{value.toFixed(2)}
    </span>
  );
}

function ProfileRow({ label, value, format }: { label: string; value: number; format: "signed" | "multiplier" | "percent" | "integer" }) {
  let display: string;
  switch (format) {
    case "signed": display = `${value >= 0 ? "+" : ""}${value.toFixed(3)}`; break;
    case "multiplier": display = `${value.toFixed(2)}x`; break;
    case "percent": display = `${(value * 100).toFixed(0)}%`; break;
    case "integer": display = String(value); break;
  }
  return (
    <div className="profile-row">
      <span className="profile-row__label">{label}</span>
      <span className="profile-row__value">{display}</span>
    </div>
  );
}
