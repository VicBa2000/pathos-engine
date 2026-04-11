import { useState, useEffect, useRef, useCallback } from "react";
import type { SignalsConfig, SignalSourceMeta, SignalTestResult } from "../types/emotion";
import {
  fetchTimeOfDaySignal,
  fetchWeatherSignal,
  fetchKeyboardSignal,
  fetchFacialSignal,
  updateSignalFromProvider,
  type ProviderReading,
} from "../signals/providers";
import { FacialDetector, type DetectionResult } from "../signals/facial-detector";
import "./SignalsConfigPanel.css";

interface Props {
  visible: boolean;
  onClose: () => void;
  sessionId: string;
}

const API = "http://localhost:8000";

const CATEGORY_ICONS: Record<string, string> = {
  physiological: "♥",
  behavioral: "⌨",
  environmental: "☁",
  custom: "★",
};

/** Sources that have real providers (auto-detect, not manual sliders) */
const REAL_PROVIDERS = new Set(["time_of_day", "weather", "keyboard_dynamics", "facial_au"]);

/** Sources that need external hardware — not yet implemented */
const HARDWARE_SOURCES = new Set<string>();

/**
 * External Signals Configuration Panel.
 *
 * - Master toggle to enable/disable all signals
 * - Real providers: auto-detect for time, weather, keyboard, facial
 * - Manual sliders: reserved for future hardware sources
 * - Test button per source to preview effect
 * - Grouped by category
 */
export function SignalsConfigPanel({ visible, onClose, sessionId }: Props) {
  const [config, setConfig] = useState<SignalsConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [testResult, setTestResult] = useState<SignalTestResult | null>(null);
  const [testingSource, setTestingSource] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [providerReadings, setProviderReadings] = useState<Record<string, ProviderReading>>({});
  const [providerErrors, setProviderErrors] = useState<Record<string, string>>({});
  const [providerLoading, setProviderLoading] = useState<Record<string, boolean>>({});
  const panelRef = useRef<HTMLDivElement>(null);

  // Facial detector state
  const facialDetectorRef = useRef<FacialDetector | null>(null);
  const facialVideoRef = useRef<HTMLVideoElement | null>(null);
  const facialPreviewRef = useRef<HTMLDivElement>(null);
  const facialUpdateTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const latestFacialResult = useRef<DetectionResult | null>(null);
  const [facialActive, setFacialActive] = useState(false);
  const [facialStatus, setFacialStatus] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [facialError, setFacialError] = useState("");
  const [facialLastExpr, setFacialLastExpr] = useState<DetectionResult | null>(null);

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

  // Load config when panel opens
  useEffect(() => {
    if (visible) loadConfig();
  }, [visible, sessionId]);

  const loadConfig = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/signals/config/${sessionId}`);
      if (!res.ok) throw new Error("Failed to load signals config");
      const data: SignalsConfig = await res.json();
      setConfig(data);
    } catch {
      setError("Could not load signals configuration");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // ── Facial detector lifecycle ──

  const stopFacialDetector = useCallback(() => {
    if (facialDetectorRef.current) {
      facialDetectorRef.current.stop();
      facialDetectorRef.current = null;
    }
    if (facialUpdateTimer.current) {
      clearInterval(facialUpdateTimer.current);
      facialUpdateTimer.current = null;
    }
    if (facialVideoRef.current && facialPreviewRef.current) {
      if (facialPreviewRef.current.contains(facialVideoRef.current)) {
        facialPreviewRef.current.removeChild(facialVideoRef.current);
      }
      facialVideoRef.current = null;
    }
    latestFacialResult.current = null;
    setFacialActive(false);
    setFacialStatus("idle");
    setFacialLastExpr(null);
  }, []);

  const startFacialDetector = useCallback(async () => {
    // Stop any existing detector
    stopFacialDetector();

    setFacialStatus("loading");
    setFacialError("");

    try {
      const detector = new FacialDetector();
      facialDetectorRef.current = detector;

      const video = await detector.start((result: DetectionResult) => {
        latestFacialResult.current = result;
        setFacialLastExpr(result);
      });

      // Attach video to preview container
      facialVideoRef.current = video;
      video.className = "signals-config__webcam-video";
      if (facialPreviewRef.current) {
        facialPreviewRef.current.appendChild(video);
      }

      setFacialActive(true);
      setFacialStatus("running");

      // Send consolidated updates to backend every 2 seconds.
      // Uses the 5-second recency-weighted buffer instead of the last frame,
      // filtering micro-expressions and noise for stable readings.
      facialUpdateTimer.current = setInterval(async () => {
        const consolidated = detector.getConsolidated();
        if (!consolidated) return;

        try {
          const reading = await fetchFacialSignal(consolidated.expressions);
          setProviderReadings(prev => ({ ...prev, facial_au: reading }));
          await updateSignalFromProvider(sessionId, "facial_au", reading);
        } catch {
          // Silent failure — will retry next interval
        }
      }, 2000);
    } catch (err) {
      const msg = err instanceof DOMException && err.name === "NotAllowedError"
        ? "Camera access denied. Enable in browser settings."
        : err instanceof DOMException && err.name === "NotFoundError"
          ? "No camera found on this device."
          : err instanceof Error
            ? err.message
            : "Failed to start facial detection";
      setFacialError(msg);
      setFacialStatus("error");
    }
  }, [sessionId, stopFacialDetector]);

  // Cleanup on unmount or panel close
  useEffect(() => {
    return () => stopFacialDetector();
  }, [stopFacialDetector]);

  // Stop facial when panel closes
  useEffect(() => {
    if (!visible && facialActive) {
      stopFacialDetector();
    }
  }, [visible, facialActive, stopFacialDetector]);

  async function updateConfig(patch: Record<string, unknown>) {
    try {
      const res = await fetch(`${API}/signals/config/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!res.ok) throw new Error("Failed to update");
      await loadConfig();
    } catch {
      setError("Failed to update configuration");
    }
  }

  function handleMasterToggle() {
    if (!config) return;
    updateConfig({ enabled: !config.enabled });
  }

  async function handleSourceToggle(source: string, currentEnabled: boolean) {
    const enabling = !currentEnabled;
    await updateConfig({ sources: { [source]: { enabled: enabling } } });

    // Stop facial detector when disabling
    if (!enabling && source === "facial_au" && facialActive) {
      stopFacialDetector();
    }

    // Auto-detect when enabling a real provider
    if (enabling && REAL_PROVIDERS.has(source)) {
      autoDetectSource(source);
    }
  }

  function handleSourceValue(source: string, field: string, value: number) {
    updateConfig({ sources: { [source]: { [field]: value } } });
  }

  /** Auto-detect a real signal source and update backend with real values */
  async function autoDetectSource(source: string) {
    setProviderLoading(prev => ({ ...prev, [source]: true }));
    setProviderErrors(prev => ({ ...prev, [source]: "" }));

    try {
      let reading: ProviderReading;

      if (source === "time_of_day") {
        reading = await fetchTimeOfDaySignal();
      } else if (source === "weather") {
        reading = await fetchWeatherSignal();
      } else if (source === "keyboard_dynamics") {
        // For keyboard, we send empty metrics initially — real data comes from chat input
        reading = await fetchKeyboardSignal({
          chars_per_second: 0,
          avg_pause_ms: 300,
          delete_ratio: 0,
          total_chars: 0,
        });
      } else if (source === "facial_au") {
        // Facial uses the real-time webcam detector
        if (!facialActive) {
          startFacialDetector();
        } else {
          stopFacialDetector();
        }
        return;
      } else {
        return;
      }

      setProviderReadings(prev => ({ ...prev, [source]: reading }));
      // Auto-update the backend config with real values
      await updateSignalFromProvider(sessionId, source, reading);
      await loadConfig();
    } catch (err) {
      const msg = err instanceof GeolocationPositionError
        ? "Location access denied. Enable in browser settings."
        : err instanceof Error
          ? err.message
          : "Detection failed";
      setProviderErrors(prev => ({ ...prev, [source]: msg }));
    } finally {
      setProviderLoading(prev => ({ ...prev, [source]: false }));
    }
  }

  async function handleTest(source: SignalSourceMeta) {
    setTestingSource(source.source);
    setTestResult(null);
    try {
      const res = await fetch(`${API}/signals/test/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: source.source,
          valence_hint: source.valence_hint !== 0 ? source.valence_hint : null,
          arousal_hint: source.arousal_hint !== 0.5 ? source.arousal_hint : null,
          dominance_hint: source.dominance_hint,
          confidence: source.confidence,
        }),
      });
      if (!res.ok) throw new Error("Test failed");
      const data: SignalTestResult = await res.json();
      setTestResult(data);
    } catch {
      setError("Signal test failed");
    } finally {
      setTimeout(() => setTestingSource(null), 200);
    }
  }

  if (!visible) return null;

  const categories = config
    ? groupByCategory(config.sources)
    : {};

  return (
    <div className="signals-config" ref={panelRef}>
      <div className="signals-config__header">
        <span className="signals-config__title">External Signals</span>
        <button className="signals-config__close" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {loading && !config ? (
        <div className="signals-config__loading">Loading...</div>
      ) : error && !config ? (
        <div className="signals-config__error">{error}</div>
      ) : config ? (
        <>
          {/* Master toggle */}
          <div className="signals-config__master">
            <label className="signals-config__toggle-row">
              <span className="signals-config__toggle-label">Enable External Signals</span>
              <button
                className={`signals-config__toggle ${config.enabled ? "signals-config__toggle--on" : ""}`}
                onClick={handleMasterToggle}
              >
                <span className="signals-config__toggle-knob" />
              </button>
            </label>
            {config.enabled && (
              <div className="signals-config__active-count">
                {config.active_count} source{config.active_count !== 1 ? "s" : ""} active
              </div>
            )}
          </div>

          {/* Per-category sources */}
          {config.enabled && Object.entries(categories).map(([category, sources]) => (
            <div key={category} className="signals-config__category">
              <div className="signals-config__category-header">
                <span className="signals-config__category-icon">
                  {CATEGORY_ICONS[category] || "?"}
                </span>
                <span className="signals-config__category-name">{category}</span>
              </div>

              {sources.map((src) => {
                const isReal = REAL_PROVIDERS.has(src.source);
                const isHardware = HARDWARE_SOURCES.has(src.source);
                const reading = providerReadings[src.source];
                const pError = providerErrors[src.source];
                const pLoading = providerLoading[src.source];

                return (
                  <div
                    key={src.source}
                    className={`signals-config__source ${src.enabled ? "signals-config__source--active" : ""}`}
                  >
                    <div className="signals-config__source-header">
                      <button
                        className={`signals-config__toggle signals-config__toggle--small ${src.enabled ? "signals-config__toggle--on" : ""}`}
                        onClick={() => handleSourceToggle(src.source, src.enabled)}
                      >
                        <span className="signals-config__toggle-knob" />
                      </button>
                      <div className="signals-config__source-info">
                        <span className="signals-config__source-label">
                          {src.label}
                          {isReal && <span className="signals-config__badge signals-config__badge--auto">AUTO</span>}
                          {isHardware && <span className="signals-config__badge signals-config__badge--manual">MANUAL</span>}
                        </span>
                        <span className="signals-config__source-desc">{src.description}</span>
                      </div>
                      <button
                        className={`signals-config__test-btn ${testingSource === src.source ? "signals-config__test-btn--testing" : ""} ${src.source === "facial_au" && facialActive ? "signals-config__test-btn--active" : ""}`}
                        onClick={() => isReal && src.enabled ? autoDetectSource(src.source) : handleTest(src)}
                        disabled={!src.enabled || (src.source === "facial_au" && facialStatus === "loading")}
                        title={src.source === "facial_au"
                          ? (facialActive ? "Stop webcam detection" : "Start webcam detection")
                          : isReal ? "Detect real signal" : "Test this signal"}
                      >
                        {src.source === "facial_au"
                          ? (facialStatus === "loading" ? "Loading..." : facialActive ? "Stop" : "Start")
                          : pLoading ? "..." : isReal ? "Detect" : "Test"}
                      </button>
                    </div>

                    {/* Facial AU — Webcam preview + expressions */}
                    {src.source === "facial_au" && src.enabled && (facialActive || facialStatus === "loading" || facialStatus === "error") && (
                      <div className="signals-config__facial">
                        <div className="signals-config__webcam-container" ref={facialPreviewRef}>
                          {facialStatus === "loading" && (
                            <div className="signals-config__webcam-loading">Loading models & camera...</div>
                          )}
                          {!facialLastExpr?.faceDetected && facialStatus === "running" && (
                            <div className="signals-config__webcam-overlay">No face detected</div>
                          )}
                        </div>

                        {facialLastExpr?.faceDetected && (
                          <div className="signals-config__facial-expressions">
                            {Object.entries(facialLastExpr.expressions)
                              .sort(([, a], [, b]) => b - a)
                              .map(([name, value]) => (
                                <div key={name} className="signals-config__expr-row">
                                  <span className="signals-config__expr-name">{name}</span>
                                  <div className="signals-config__expr-bar-bg">
                                    <div
                                      className="signals-config__expr-bar"
                                      style={{ width: `${value * 100}%` }}
                                    />
                                  </div>
                                  <span className="signals-config__expr-val">{(value * 100).toFixed(0)}%</span>
                                </div>
                              ))}
                          </div>
                        )}

                        {facialError && (
                          <div className="signals-config__provider-error">{facialError}</div>
                        )}
                      </div>
                    )}

                    {/* Real provider reading (non-facial sources) */}
                    {src.enabled && isReal && reading && src.source !== "facial_au" && (
                      <div className="signals-config__reading">
                        <div className="signals-config__reading-title">Live Reading</div>
                        <div className="signals-config__reading-row">
                          <span>V:</span>
                          <span className={reading.valence_hint >= 0 ? "signals-config__val--pos" : "signals-config__val--neg"}>
                            {reading.valence_hint >= 0 ? "+" : ""}{reading.valence_hint.toFixed(3)}
                          </span>
                          <span>A:</span>
                          <span>{reading.arousal_hint.toFixed(3)}</span>
                          <span>C:</span>
                          <span>{(reading.confidence * 100).toFixed(0)}%</span>
                        </div>
                        {reading.detail && (
                          <div className="signals-config__reading-detail">
                            {Object.entries(reading.detail).map(([k, v]) => (
                              <span key={k}>{k}: {String(v)}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Facial AU — pipeline reading (V/A/C from backend) */}
                    {src.source === "facial_au" && src.enabled && facialActive && reading && (
                      <div className="signals-config__reading">
                        <div className="signals-config__reading-title">Pipeline Signal</div>
                        <div className="signals-config__reading-row">
                          <span>V:</span>
                          <span className={reading.valence_hint >= 0 ? "signals-config__val--pos" : "signals-config__val--neg"}>
                            {reading.valence_hint >= 0 ? "+" : ""}{reading.valence_hint.toFixed(3)}
                          </span>
                          <span>A:</span>
                          <span>{reading.arousal_hint.toFixed(3)}</span>
                          <span>C:</span>
                          <span>{(reading.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    )}

                    {/* Provider error */}
                    {src.enabled && pError && (
                      <div className="signals-config__provider-error">{pError}</div>
                    )}

                    {/* Manual sliders (for hardware sources or override) */}
                    {src.enabled && isHardware && (
                      <div className="signals-config__sliders">
                        <SliderRow
                          label="Valence"
                          value={src.valence_hint}
                          min={-1} max={1} step={0.05}
                          onChange={(v) => handleSourceValue(src.source, "valence_hint", v)}
                          format={(v) => v.toFixed(2)}
                        />
                        <SliderRow
                          label="Arousal"
                          value={src.arousal_hint}
                          min={0} max={1} step={0.05}
                          onChange={(v) => handleSourceValue(src.source, "arousal_hint", v)}
                          format={(v) => v.toFixed(2)}
                        />
                        <SliderRow
                          label="Confidence"
                          value={src.confidence}
                          min={0} max={1} step={0.05}
                          onChange={(v) => handleSourceValue(src.source, "confidence", v)}
                          format={(v) => `${Math.round(v * 100)}%`}
                        />
                      </div>
                    )}

                    {/* Test result */}
                    {testResult && testResult.source === src.source && (
                      <div className="signals-config__test-result">
                        <div className="signals-config__test-result-title">Pipeline Effect</div>
                        <div className="signals-config__test-row">
                          <span>V mod:</span>
                          <span className={testResult.fused_effect.valence_modulation >= 0 ? "signals-config__val--pos" : "signals-config__val--neg"}>
                            {testResult.fused_effect.valence_modulation >= 0 ? "+" : ""}{testResult.fused_effect.valence_modulation.toFixed(4)}
                          </span>
                        </div>
                        <div className="signals-config__test-row">
                          <span>A mod:</span>
                          <span className={testResult.fused_effect.arousal_modulation >= 0 ? "signals-config__val--pos" : "signals-config__val--neg"}>
                            {testResult.fused_effect.arousal_modulation >= 0 ? "+" : ""}{testResult.fused_effect.arousal_modulation.toFixed(4)}
                          </span>
                        </div>
                        <div className="signals-config__test-row">
                          <span>Weight:</span>
                          <span>{testResult.processed.weight.toFixed(3)}</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}

          {!config.enabled && (
            <div className="signals-config__disabled-info">
              Enable external signals to configure individual sources.
              Sources marked AUTO detect real-world data automatically.
              Sources marked MANUAL require external hardware.
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}


// ── Helpers ──

function groupByCategory(sources: SignalSourceMeta[]): Record<string, SignalSourceMeta[]> {
  const groups: Record<string, SignalSourceMeta[]> = {};
  for (const src of sources) {
    const cat = src.category || "other";
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(src);
  }
  return groups;
}

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format: (value: number) => string;
}

function SliderRow({ label, value, min, max, step, onChange, format }: SliderRowProps) {
  return (
    <div className="signals-config__slider-row">
      <span className="signals-config__slider-label">{label}</span>
      <input
        type="range"
        className="signals-config__slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <span className="signals-config__slider-value">{format(value)}</span>
    </div>
  );
}
