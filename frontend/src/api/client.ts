/** API client for Pathos Engine backend. */

import type {
  ArenaContestant,
  ArenaResponse,
  CalibrationProfile,
  CalibrationResult,
  CalibrationScenario,
  ChallengeConfig,
  ChallengeState,
  ChallengeChatResponse,
  ChatResponse,
  ResearchChatResponse,
  SandboxResponse,
  BatchSandboxResponse,
  StateResponse,
} from "../types/emotion";

// In dev, requests go through Vite proxy (/api -> localhost:8000)
// In prod, set VITE_API_URL to the backend URL
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}

// --- Companion Mode ---

export function sendChat(message: string, sessionId: string): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}

export function getState(sessionId: string): Promise<StateResponse> {
  return request(`/state/${sessionId}`);
}

export function resetSession(sessionId: string): Promise<{ status: string }> {
  return request(`/reset/${sessionId}`, { method: "POST" });
}

// --- Research Mode ---

export function sendResearchChat(message: string, sessionId: string): Promise<ResearchChatResponse> {
  return request("/research/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}

// --- Calibration Mode ---

export function submitCalibrationScenario(
  scenario: CalibrationScenario,
  sessionId: string,
): Promise<CalibrationResult> {
  return request(`/calibration/scenario?session_id=${sessionId}`, {
    method: "POST",
    body: JSON.stringify(scenario),
  });
}

export function applyCalibration(sessionId: string): Promise<CalibrationProfile> {
  return request(`/calibration/apply?session_id=${sessionId}`, { method: "POST" });
}

export function getCalibrationProfile(sessionId: string): Promise<CalibrationProfile> {
  return request(`/calibration/profile/${sessionId}`);
}

export function resetCalibration(sessionId: string): Promise<{ status: string }> {
  return request(`/calibration/reset/${sessionId}`, { method: "DELETE" });
}

// --- Health ---

export function healthCheck(): Promise<{ status: string; provider: string; model: string; active_session: string | null; turn_count: number }> {
  return request("/health");
}

// --- Models ---

export interface ModelInfo {
  name: string;
  size: string;
  provider: string;
  steering_compatible: boolean;
  vectors_cached: boolean;
}

export function listModels(): Promise<ModelInfo[]> {
  return request("/models");
}

export function extractSteeringVectors(model: string): Promise<{ status: string; model: string; total_time_s?: number }> {
  return request("/models/steering/extract", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export interface ArkSwitchInfo {
  direct_available: boolean;
  vectors_ready: boolean;
  adapter_loaded: boolean;
  message: string;
}

export interface SwitchModelResult {
  status: string;
  provider: string;
  model: string;
  ark: ArkSwitchInfo;
}

export function switchModel(provider: string, model: string, sessionId?: string): Promise<SwitchModelResult> {
  return request("/models/switch", {
    method: "POST",
    body: JSON.stringify({ provider, model, session_id: sessionId || "default" }),
  });
}

// --- ARK Status ---

export interface ArkSystemStatus {
  available: boolean;
  enabled: boolean;
  active: boolean;
  reason?: string;
  momentum_factor?: number;
}

export interface ArkStatus {
  provider: string;
  model: string;
  direct_available: boolean;
  direct_active: boolean;
  direct_mode_toggle: boolean;
  fallback_reason: string;
  systems: Record<string, ArkSystemStatus>;
}

export function getArkStatus(sessionId: string): Promise<ArkStatus> {
  return request(`/config/ark-status/${sessionId}`);
}

export function setArkMode(sessionId: string, body: { direct_mode?: boolean; system?: string; enabled?: boolean }): Promise<ArkStatus> {
  return request(`/config/ark-mode/${sessionId}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// --- Model Manager ---

export interface FeaturedModel {
  name: string;
  description: string;
  size: string;
  vram_estimate: string;
  category: string;
}

export interface SearchResult {
  name: string;
  description: string;
  pulls: string;
  tags: string[];
}

export interface DownloadStatus {
  name: string;
  status: string;
  completed: number;
  total: number;
  percent: number;
  error: string;
}

export interface HuggingFaceCheck {
  valid: boolean;
  error?: string;
  quantizations: string[];
  files: Array<{ name: string; size: number }>;
  ollama_name?: string;
}

export function getFeaturedModels(): Promise<FeaturedModel[]> {
  return request("/models/featured");
}

export function searchModels(query: string): Promise<SearchResult[]> {
  return request(`/models/search?q=${encodeURIComponent(query)}`);
}

export function getDownloads(): Promise<{ downloads: DownloadStatus[] }> {
  return request("/models/downloads");
}

export function cancelDownload(name: string): Promise<{ status: string }> {
  return request(`/models/pull/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export function deleteModel(name: string): Promise<{ status: string }> {
  return request(`/models/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export function setClaudeKey(apiKey: string): Promise<{ status: string }> {
  return request("/config/claude-key", {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
  });
}

export function getClaudeKeyStatus(): Promise<{ configured: boolean; masked_key: string }> {
  return request("/config/claude-key/status");
}

// --- Cloud Providers ---

export interface CloudPreset {
  label: string;
  base_url: string;
  key_prefix: string;
  description: string;
  default_model: string;
}

export interface CloudProviderInfo {
  id: string;
  preset: string;
  label: string;
  base_url: string;
  model: string;
  masked_key: string;
}

export function getCloudPresets(): Promise<{ presets: Record<string, CloudPreset> }> {
  return request("/config/cloud-presets");
}

export function addCloudProvider(
  sessionId: string,
  preset: string,
  apiKey: string,
  baseUrl?: string,
  model?: string,
): Promise<{ status: string; provider_id: string }> {
  return request("/config/cloud-provider", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      preset,
      api_key: apiKey,
      base_url: baseUrl || "",
      model: model || "",
    }),
  });
}

export function listCloudProviders(sessionId: string): Promise<{ providers: CloudProviderInfo[] }> {
  return request(`/config/cloud-providers/${sessionId}`);
}

export function removeCloudProvider(sessionId: string, providerId: string): Promise<{ status: string }> {
  return request(`/config/cloud-provider/${sessionId}/${providerId}`, { method: "DELETE" });
}

export interface CloudTestResult {
  ok: boolean;
  error: string;
  models: Array<{ name: string; size: string }>;
}

export function testCloudProvider(preset: string, apiKey: string, baseUrl?: string): Promise<CloudTestResult> {
  return request("/config/cloud-provider/test", {
    method: "POST",
    body: JSON.stringify({ preset, api_key: apiKey, base_url: baseUrl || "" }),
  });
}

export function checkHuggingFace(repo: string): Promise<HuggingFaceCheck> {
  return request("/models/huggingface/check", {
    method: "POST",
    body: JSON.stringify({ repo }),
  });
}

/** Pull a model via SSE. Calls onProgress for each event, returns when done. */
export async function pullModel(
  name: string,
  onProgress: (status: DownloadStatus) => void,
): Promise<void> {
  const resp = await fetch(`${API_BASE}/models/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!resp.ok) throw new Error(`Pull failed: ${resp.status}`);
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          onProgress({ name, ...data });
        } catch { /* skip malformed */ }
      }
    }
  }
}

// --- Agent Setup ---

export function getPersonality(sessionId: string): Promise<{ personality: Record<string, number> }> {
  return request(`/personality/${sessionId}`);
}

export function setPersonality(sessionId: string, profile: Record<string, number>): Promise<Record<string, unknown>> {
  return request(`/personality/${sessionId}`, { method: "POST", body: JSON.stringify(profile) });
}

export function getValues(sessionId: string): Promise<{
  core: Array<{ name: string; weight: number; description: string }>;
  relational: Record<string, number>;
  self_model: Record<string, number>;
}> {
  return request(`/values/${sessionId}`);
}

export function setValues(sessionId: string, weights: Record<string, number>): Promise<Record<string, unknown>> {
  return request(`/values/${sessionId}`, { method: "POST", body: JSON.stringify(weights) });
}

export function getIdentity(sessionId: string): Promise<{
  statements: Array<Record<string, unknown>>;
  coherence: number;
  growth_events: number;
  age: number;
}> {
  return request(`/identity/${sessionId}`);
}

export function setIdentity(sessionId: string, data: { name?: string; background?: string }): Promise<Record<string, unknown>> {
  return request(`/identity/${sessionId}`, { method: "POST", body: JSON.stringify(data) });
}

// --- Save/Load ---

export function saveSession(sessionId: string): Promise<{ status: string; filename: string }> {
  return request(`/session/save/${sessionId}`, { method: "POST" });
}

export function listSaves(): Promise<{ saves: Array<{ filename: string; session_id: string; model: string; saved_at: string; turn_count: number }> }> {
  return request("/session/saves");
}

export function restoreSessionInfo(sessionId: string): Promise<{
  session_id: string;
  conversation: Array<{ role: string; content: string }>;
  emotional_state: StateResponse["state"];
  turn_count: number;
  lite_mode: boolean;
  advanced_mode: boolean;
}> {
  return request(`/session/restore/${sessionId}`);
}

export function loadSave(filename: string): Promise<{ status: string; session_id: string; model: string; turn_count: number }> {
  return request(`/session/load/${filename}`, { method: "POST" });
}

// --- Export ---

export function exportPortable(sessionId: string): Promise<Blob> {
  return fetch(`${API_BASE}/export/portable?session_id=${sessionId}`, {
    method: "POST",
  }).then(res => {
    if (!res.ok) throw new Error(`Export failed: ${res.status}`);
    return res.blob();
  });
}

// --- Voice ---

export function configureVoice(
  sessionId: string,
  config: { mode?: string; voice?: string; language?: string; auto_speak?: boolean; tts_backend?: string },
): Promise<{ status: string; voice_config: Record<string, unknown> }> {
  return request(`/voice/config/${sessionId}`, {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function sendAudio(
  sessionId: string,
  audioBlob: Blob,
): Promise<{ text: string; language: string; segments: Array<{ start: number; end: number; text: string }> }> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  return fetch(`${API_BASE}/voice/listen/${sessionId}`, {
    method: "POST",
    body: formData,
  }).then(res => {
    if (!res.ok) return res.text().then(t => { throw new Error(`ASR ${res.status}: ${t}`); });
    return res.json();
  });
}

export function getAudio(sessionId: string, turn?: number): Promise<Blob> {
  const url = turn != null
    ? `${API_BASE}/voice/audio/${sessionId}?turn=${turn}`
    : `${API_BASE}/voice/audio/${sessionId}`;
  return fetch(url)
    .then(res => {
      if (!res.ok) throw new Error(`Audio fetch failed: ${res.status}`);
      return res.blob();
    });
}

export function listVoices(): Promise<{
  voices: Array<{ key: string; language: string; name: string; gender: string }>;
  default_by_language: Record<string, string>;
}> {
  return request("/voice/voices");
}

// --- Forecasting ---

export function toggleForecasting(
  sessionId: string,
  enabled: boolean,
): Promise<{ status: string; forecasting_enabled: boolean }> {
  return request(`/forecasting/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

// --- Lite Mode ---

export function toggleLiteMode(
  sessionId: string,
  enabled: boolean,
): Promise<{ status: string; lite_mode: boolean }> {
  return request(`/lite-mode/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

// --- Advanced Mode ---

export function toggleAdvancedMode(
  sessionId: string,
  enabled: boolean,
): Promise<{ status: string; advanced_mode: boolean }> {
  return request(`/advanced-mode/${sessionId}`, {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

// --- Export ---

export interface ExportResult {
  status: string;
  model?: string;
  modelfile_path?: string;
  message?: string;
  ollama_error?: string;
  hint?: string;
}

export function exportModel(
  sessionId: string,
  baseModel: string,
  modelName: string,
  temperature: number = 0.7,
  numCtx: number = 8192,
): Promise<ExportResult> {
  return request(`/models/export?session_id=${sessionId}`, {
    method: "POST",
    body: JSON.stringify({
      base_model: baseModel,
      model_name: modelName,
      temperature,
      num_ctx: numCtx,
    }),
  });
}

// --- Sandbox ---

export function simulateSandbox(
  scenario: string,
  sessionId: string,
  options?: {
    personality?: Record<string, number>;
    initial_state?: string;
    rapport?: number;
    trust?: number;
  },
): Promise<SandboxResponse> {
  return request("/sandbox/simulate", {
    method: "POST",
    body: JSON.stringify({
      scenario,
      session_id: sessionId,
      ...options,
    }),
  });
}

export function batchSandbox(
  scenarios: string[],
  sessionId: string,
  options?: {
    personality?: Record<string, number>;
    initial_state?: string;
    rapport?: number;
    trust?: number;
  },
): Promise<BatchSandboxResponse> {
  return request("/sandbox/batch", {
    method: "POST",
    body: JSON.stringify({
      scenarios,
      session_id: sessionId,
      ...options,
    }),
  });
}

// --- Arena ---

export function arenaCompare(
  scenario: string,
  contestants: ArenaContestant[],
  sessionId: string,
  options?: { rapport?: number; trust?: number },
): Promise<ArenaResponse> {
  return request("/arena/compare", {
    method: "POST",
    body: JSON.stringify({
      scenario,
      contestants,
      session_id: sessionId,
      ...options,
    }),
  });
}

// --- Raw Mode ---

export function rawChat(
  message: string,
  sessionId: string,
): Promise<ChatResponse> {
  return request("/raw/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}

export function rawToggleExtreme(sessionId: string, enabled: boolean): Promise<{ status: string; extreme_mode: boolean }> {
  return request(`/raw/extreme?session_id=${sessionId}&enabled=${enabled}`, { method: "POST" });
}

export function rawReset(sessionId: string): Promise<{ status: string }> {
  return request(`/raw/reset?session_id=${sessionId}`, { method: "POST" });
}

// --- Mirror Test (Challenge) ---

export function getChallengeLibrary(): Promise<ChallengeConfig[]> {
  return request("/challenge/library");
}

export function startChallenge(challengeId: string, sessionId: string): Promise<ChallengeState> {
  return request("/challenge/start", {
    method: "POST",
    body: JSON.stringify({ challenge_id: challengeId, session_id: sessionId }),
  });
}

export function challengeChat(message: string, sessionId: string): Promise<ChallengeChatResponse> {
  return request("/challenge/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}

export function abandonChallenge(sessionId: string): Promise<{ status: string; best_score?: number }> {
  return request(`/challenge/abandon/${sessionId}`, { method: "POST" });
}

// --- Autonomous Research ---

export function startAutonomousResearch(
  sessionId: string,
  pipelineMode: string,
  seedTopics?: string[],
): Promise<{ status: string; session_id: string }> {
  return request("/autonomous/start", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      pipeline_mode: pipelineMode,
      seed_topics: seedTopics || [],
    }),
  });
}

export function stopAutonomousResearch(sessionId: string): Promise<{ status: string }> {
  return request(`/autonomous/stop?session_id=${sessionId}`, { method: "POST" });
}

export function getAutonomousStatus(sessionId: string): Promise<Record<string, unknown>> {
  return request(`/autonomous/status/${sessionId}`);
}

export function autonomousChat(
  message: string,
  sessionId: string,
): Promise<{ response: string; emotional_state: Record<string, unknown> }> {
  return request("/autonomous/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}

export function saveAutonomousResearch(sessionId: string): Promise<{ status: string; filename: string }> {
  return request(`/autonomous/save/${sessionId}`, { method: "POST" });
}

export function listAutonomousSaves(): Promise<Array<Record<string, unknown>>> {
  return request("/autonomous/saves");
}

export function loadAutonomousResearch(filename: string): Promise<{ status: string; session_id: string }> {
  return request(`/autonomous/load/${filename}`, { method: "POST" });
}

export function subscribeResearchEvents(
  sessionId: string,
  onEvent: (event: Record<string, unknown>) => void,
): () => void {
  const source = new EventSource(`${API_BASE}/autonomous/events/${sessionId}`);
  source.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data);
      if (event.type !== "heartbeat") onEvent(event);
    } catch { /* skip parse errors */ }
  };
  return () => source.close();
}
