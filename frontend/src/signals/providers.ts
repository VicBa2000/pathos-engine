/**
 * Real-world signal providers — Convert actual data into signal values.
 *
 * Each provider auto-detects its signal and sends values to the backend.
 * Providers are independent — enable only the ones you want.
 */

const API = "http://localhost:8000";

export interface ProviderReading {
  valence_hint: number;
  arousal_hint: number;
  dominance_hint?: number | null;
  confidence: number;
  detail: Record<string, unknown>;
}

// ── Time of Day ──

export async function fetchTimeOfDaySignal(): Promise<ProviderReading> {
  const now = new Date();
  const resp = await fetch(`${API}/signals/providers/time`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hour: now.getHours(), minute: now.getMinutes() }),
  });
  if (!resp.ok) throw new Error("Time provider failed");
  return resp.json();
}

// ── Weather ──

export async function fetchWeatherSignal(): Promise<ProviderReading> {
  // Get user's geolocation
  const pos = await new Promise<GeolocationPosition>((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      timeout: 10000,
      maximumAge: 30 * 60 * 1000, // Cache for 30 min
    });
  });

  const resp = await fetch(`${API}/signals/providers/weather`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
  });
  if (!resp.ok) throw new Error("Weather provider failed");
  return resp.json();
}

// ── Keyboard Dynamics ──

export interface KeystrokeTracker {
  keyTimes: number[];
  deleteCount: number;
  totalKeys: number;
  startTime: number;
}

export function createKeystrokeTracker(): KeystrokeTracker {
  return { keyTimes: [], deleteCount: 0, totalKeys: 0, startTime: Date.now() };
}

export function recordKeystroke(tracker: KeystrokeTracker, isDelete: boolean): void {
  tracker.keyTimes.push(Date.now());
  tracker.totalKeys++;
  if (isDelete) tracker.deleteCount++;
}

export function computeKeyboardMetrics(tracker: KeystrokeTracker): {
  chars_per_second: number;
  avg_pause_ms: number;
  delete_ratio: number;
  total_chars: number;
} {
  const { keyTimes, deleteCount, totalKeys } = tracker;
  if (totalKeys < 2) {
    return { chars_per_second: 0, avg_pause_ms: 300, delete_ratio: 0, total_chars: 0 };
  }

  const elapsed = (keyTimes[keyTimes.length - 1] - keyTimes[0]) / 1000;
  const chars_per_second = elapsed > 0 ? totalKeys / elapsed : 0;

  // Average pause between keystrokes
  let totalPause = 0;
  for (let i = 1; i < keyTimes.length; i++) {
    totalPause += keyTimes[i] - keyTimes[i - 1];
  }
  const avg_pause_ms = totalPause / (keyTimes.length - 1);

  return {
    chars_per_second,
    avg_pause_ms,
    delete_ratio: deleteCount / totalKeys,
    total_chars: totalKeys - deleteCount,
  };
}

export async function fetchKeyboardSignal(
  metrics: ReturnType<typeof computeKeyboardMetrics>,
): Promise<ProviderReading> {
  const resp = await fetch(`${API}/signals/providers/keyboard`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metrics),
  });
  if (!resp.ok) throw new Error("Keyboard provider failed");
  return resp.json();
}

// ── Facial AU ──

/**
 * Send facial expression detection results to the backend for mapping.
 * The actual face detection runs in the browser via face-api.js.
 * This function just sends the expression probabilities for computation.
 */
export async function fetchFacialSignal(
  expressions: Record<string, number>,
): Promise<ProviderReading> {
  const resp = await fetch(`${API}/signals/providers/facial`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expressions }),
  });
  if (!resp.ok) throw new Error("Facial provider failed");
  return resp.json();
}

// ── Auto-update helper ──

/**
 * Update a signal source on the backend with real provider values.
 * This is the bridge: provider detects → backend config updated → pipeline uses it.
 */
export async function updateSignalFromProvider(
  sessionId: string,
  source: string,
  reading: ProviderReading,
): Promise<void> {
  await fetch(`${API}/signals/config/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sources: {
        [source]: {
          enabled: true,
          valence_hint: reading.valence_hint,
          arousal_hint: reading.arousal_hint,
          dominance_hint: reading.dominance_hint ?? undefined,
          confidence: reading.confidence,
        },
      },
    }),
  });
}
