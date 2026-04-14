import { useState, useCallback, useEffect } from "react";
import type {
  AppMode,
  ChatMessage,
  EmotionalState,
  PipelineTrace,
  PrimaryEmotion,
  ResearchChatResponse,
} from "./types/emotion";
import { ChatPanel } from "./components/ChatPanel";
import { EmotionalStatePanel } from "./components/EmotionalStatePanel";
import { JourneyTimeline } from "./components/JourneyTimeline";
import { ModeSelector } from "./components/ModeSelector";
import { ResearchPanel } from "./components/ResearchPanel";
import { CalibrationPanel } from "./components/CalibrationPanel";
import { SandboxPanel } from "./components/SandboxPanel";
import { ArenaPanel } from "./components/ArenaPanel";
import { MirrorTest } from "./components/MirrorTest";
import { RawChatPanel } from "./components/RawChatPanel";
import { AutonomousResearchPanel } from "./components/AutonomousResearchPanel";
import { EmotionNetwork } from "./components/EmotionNetwork";
import { EmotionGenesis } from "./components/EmotionGenesis";
import { QualiaOrb } from "./components/QualiaOrb";
import { EmotionAvatar } from "./components/EmotionAvatar";
import { PipelineViewer } from "./components/PipelineViewer";
import { ErrorBoundary } from "./components/ErrorBoundary";
import * as api from "./api/client";
import "./App.css";

function generateSessionId(): string {
  return `session-${crypto.randomUUID()}`;
}

const DEFAULT_STATE: EmotionalState = {
  valence: 0,
  arousal: 0.3,
  dominance: 0.5,
  certainty: 0.5,
  primary_emotion: "neutral",
  secondary_emotion: null,
  intensity: 0,
  emotional_stack: { neutral: 1.0 },
  body_state: { energy: 0.5, tension: 0.3, openness: 0.5, warmth: 0.5 },
  mood: {
    baseline_valence: 0.1,
    baseline_arousal: 0.3,
    stability: 0.7,
    trend: "stable",
    label: "neutral",
    extreme_event_count: 0,
    turns_since_extreme: 0,
    original_baseline_valence: 0.1,
    original_baseline_arousal: 0.3,
  },
  duration: 0,
  triggered_by: "initialization",
  timestamp: new Date().toISOString(),
};

export default function App() {
  const [mode, _setMode] = useState<AppMode>(() => {
    const saved = sessionStorage.getItem("pathos_mode") as AppMode | null;
    // Raw is ephemeral — never restore it after reload
    if (saved && saved !== "raw") return saved;
    return "companion";
  });
  const setMode = useCallback((m: AppMode) => {
    _setMode(m);
    sessionStorage.setItem("pathos_mode", m);
  }, []);
  const [sessionId, _setSessionId] = useState(() => {
    return sessionStorage.getItem("pathos_session_id") || generateSessionId();
  });
  const setSessionId = useCallback((id: string) => {
    _setSessionId(id);
    sessionStorage.setItem("pathos_session_id", id);
  }, []);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [emotionalState, setEmotionalState] = useState<EmotionalState>(DEFAULT_STATE);
  const [autoResearchEmotionalState, setAutoResearchEmotionalState] = useState<EmotionalState>(DEFAULT_STATE);
  const [researchData, setResearchData] = useState<ResearchChatResponse | null>(null);
  const [currentModel, setCurrentModel] = useState("");
  const [currentProvider, setCurrentProvider] = useState("ollama");

  // History for circumplex trail
  const [circumplexHistory, setCircumplexHistory] = useState<
    Array<{ valence: number; arousal: number; emotion: string }>
  >([]);

  // Journey timeline data
  const [journey, setJourney] = useState<
    Array<{ turn: number; emotion: PrimaryEmotion; valence: number; arousal: number; intensity: number }>
  >([]);

  // Emotion transition history (for EmotionNetwork)
  const [emotionHistory, setEmotionHistory] = useState<
    Array<{ emotion: PrimaryEmotion; intensity: number }>
  >([{ emotion: "neutral", intensity: 0 }]);

  // UI toggles
  const [showNetwork, setShowNetwork] = useState(false);
  const [showEmotionSidebar, setShowEmotionSidebar] = useState(true);
  const [forecastingEnabled, setForecastingEnabled] = useState(false);
  const [advancedMode, setAdvancedMode] = useState(true);
  const [liteMode, setLiteMode] = useState(false);
  const [animaEnabled, setAnimaEnabled] = useState(false);
  const [devStage, setDevStage] = useState("sensorimotor");
  const [devSpeed, setDevSpeed] = useState("natural");
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [micEnabled, setMicEnabled] = useState(false);
  const [micReady, setMicReady] = useState(false);
  const [micStream, setMicStream] = useState<MediaStream | null>(null);
  const [voiceLoading, setVoiceLoading] = useState(false);
  const [showPipeline, setShowPipeline] = useState(false);
  const [showGenesis, setShowGenesis] = useState(false);
  const [showOrb, setShowOrb] = useState(false);
  const [showAvatar, setShowAvatar] = useState(false);

  // Panel limit — prevent too many panels from overflowing screen
  const MAX_DISPLAY_PANELS = 3;
  const activePanelCount = [showEmotionSidebar, showNetwork, showPipeline, showGenesis, showOrb, showAvatar].filter(Boolean).length;
  const panelLimitReached = activePanelCount >= MAX_DISPLAY_PANELS;
  const [audioAnalyser, setAudioAnalyser] = useState<AnalyserNode | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [pipelineTrace, setPipelineTrace] = useState<PipelineTrace | null>(null);
  const [pipelineDetailed, setPipelineDetailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportingPortable, setExportingPortable] = useState(false);
  const [modelLocked, setModelLocked] = useState(false);
  const [autonomousSessionId, setAutonomousSessionId] = useState<string | null>(null);

  // Health check on mount — restore session if backend has one
  useEffect(() => {
    const savedSessionId = sessionStorage.getItem("pathos_session_id");

    api.healthCheck()
      .then((h) => {
        setConnected(true);
        setCurrentModel(h.model);
        setCurrentProvider(h.provider || "ollama");

        // Determine which session to restore:
        // 1. If we have a saved sessionId from before reload, try that first
        // 2. Otherwise use the backend's active_session (from auto-loaded save)
        const restoreId = savedSessionId || (h.active_session && h.turn_count > 0 ? h.active_session : null);
        if (!restoreId) return;

        setSessionId(restoreId);
        api.restoreSessionInfo(restoreId)
          .then((info) => {
            if (info.turn_count === 0) return; // Empty session, nothing to restore
            setEmotionalState(info.emotional_state);
            setLiteMode(info.lite_mode);
            setAdvancedMode(info.advanced_mode);
            if (info.anima_enabled !== undefined) {
              setAnimaEnabled(info.anima_enabled);
              if (info.anima_enabled) {
                api.getDevelopmentStatus(restoreId).then((s) => {
                  setDevStage(s.current_stage);
                  setDevSpeed(s.speed);
                }).catch(console.error);
              }
            }
            // Rebuild chat messages from conversation history
            const restored: ChatMessage[] = info.conversation.map((msg) => ({
              role: msg.role as "user" | "assistant",
              content: msg.content,
            }));
            setMessages(restored);
          })
          .catch(() => {});
      })
      .catch(() => setConnected(false));
  }, []);

  const handleSend = useCallback(async (message: string) => {
    setLoading(true);

    // Add user message immediately
    const userMsg: ChatMessage = { role: "user", content: message };
    setMessages(prev => [...prev, userMsg]);

    try {
      let responseText: string;
      let state: EmotionalState;
      let audioAvailable = false;
      let turnNumber = 0;

      if (mode === "research") {
        const res = await api.sendResearchChat(message, sessionId);
        responseText = res.response;
        state = res.emotional_state;
        setResearchData(res);
        audioAvailable = res.voice?.audio_available ?? false;
        turnNumber = res.turn_number ?? 0;
      } else {
        const res = await api.sendChat(message, sessionId);
        responseText = res.response;
        state = res.emotional_state;
        audioAvailable = res.audio_available ?? false;
        turnNumber = (res as Record<string, unknown>).turn_number as number ?? 0;
        setResearchData(null);
        if (res.pipeline_trace) setPipelineTrace(res.pipeline_trace);
      }

      setEmotionalState(state);

      // Update circumplex history (cap at 200 entries)
      setCircumplexHistory(prev => [
        ...prev,
        { valence: state.valence, arousal: state.arousal, emotion: state.primary_emotion },
      ].slice(-200));

      // Update journey (cap at 200 entries)
      setJourney(prev => [
        ...prev,
        {
          turn: prev.length + 1,
          emotion: state.primary_emotion,
          valence: state.valence,
          arousal: state.arousal,
          intensity: state.intensity,
        },
      ].slice(-200));

      // Update emotion transition history (cap at 200 entries)
      setEmotionHistory(prev => [
        ...prev,
        { emotion: state.primary_emotion, intensity: state.intensity },
      ].slice(-200));

      // Add assistant message
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: responseText,
        emotional_state: state,
        audioAvailable,
        turnNumber,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  }, [mode, sessionId]);

  const handleSendAudio = useCallback(async (audioBlob: Blob): Promise<string | null> => {
    try {
      const result = await api.sendAudio(sessionId, audioBlob);
      return result.text;
    } catch (err) {
      console.error("Audio transcription failed:", err);
      return null;
    }
  }, [sessionId]);

  async function handleExitRaw() {
    // Switch to companion — RawChatPanel unmounts and its cleanup calls rawReset
    setMode("companion");
    // Re-run health check to restore auto-loaded session if available
    try {
      const h = await api.healthCheck();
      setConnected(true);
      setCurrentModel(h.model);
      setCurrentProvider(h.provider || "ollama");
      if (h.active_session && h.turn_count > 0) {
        setSessionId(h.active_session);
        const info = await api.restoreSessionInfo(h.active_session);
        setEmotionalState(info.emotional_state);
        setLiteMode(info.lite_mode);
        setAdvancedMode(info.advanced_mode);
        const restored: ChatMessage[] = info.conversation.map((msg) => ({
          role: msg.role as "user" | "assistant",
          content: msg.content,
        }));
        setMessages(restored);
      }
    } catch {
      // If no save, just go to fresh companion
    }
  }

  function handleNewSession() {
    const newId = generateSessionId();
    setSessionId(newId);
    setMessages([]);
    setEmotionalState(DEFAULT_STATE);
    setCircumplexHistory([]);
    setJourney([]);
    setEmotionHistory([{ emotion: "neutral", intensity: 0 }]);
    setResearchData(null);
    setForecastingEnabled(false);
    setLiteMode(false);
    setVoiceEnabled(false);
    setMicEnabled(false);
  }

  async function handleSave() {
    setSaving(true);
    try {
      const result = await api.saveSession(sessionId);
      console.log("Session saved:", result.filename);
    } catch (err) {
      console.error("Save failed:", err);
    } finally {
      setSaving(false);
    }
  }

  async function handleExport() {
    setExporting(true);
    try {
      const baseModel = currentModel || "qwen3:4b";
      await api.exportModel(sessionId, baseModel, "pathos");
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExporting(false);
    }
  }

  async function handleExportPortable() {
    setExportingPortable(true);
    try {
      const blob = await api.exportPortable(sessionId);
      // Trigger download
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "pathos-portable.zip";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Portable export failed:", err);
    } finally {
      setExportingPortable(false);
    }
  }

  // Determine sidebar content for research/calibration modes
  const isResearchMode = mode === "research";
  const isCalibrationMode = mode === "calibration";
  const isSandboxMode = mode === "sandbox";
  const isArenaMode = mode === "arena";
  const isMirrorMode = mode === "mirror";
  const isRawMode = mode === "raw";
  const isAutonomousMode = mode === "autonomous";
  const activeEmotionalState = isAutonomousMode ? autoResearchEmotionalState : emotionalState;

  return (
    <div className="app">
      <ModeSelector
        mode={mode}
        onModeChange={setMode}
        connected={connected}
        currentModel={currentModel}
        onModelChanged={(provider, model) => { setCurrentProvider(provider); setCurrentModel(model); }}
        showNetwork={showNetwork}
        onToggleNetwork={() => { if (showNetwork || !panelLimitReached) setShowNetwork(p => !p); }}
        showForecasting={forecastingEnabled}
        onToggleForecasting={() => {
          const newVal = !forecastingEnabled;
          setForecastingEnabled(newVal);
          api.toggleForecasting(sessionId, newVal).catch(console.error);
        }}
        advancedMode={advancedMode}
        onToggleAdvancedMode={() => {
          const newVal = !advancedMode;
          setAdvancedMode(newVal);
          api.toggleAdvancedMode(sessionId, newVal).catch(console.error);
          // Mutually exclusive: activating Advanced disables Lite
          if (newVal && liteMode) {
            setLiteMode(false);
            api.toggleLiteMode(sessionId, false).catch(console.error);
          }
        }}
        liteMode={liteMode}
        onToggleLiteMode={() => {
          const newVal = !liteMode;
          setLiteMode(newVal);
          api.toggleLiteMode(sessionId, newVal).catch(console.error);
          // Mutually exclusive: activating Lite disables Advanced
          if (newVal && advancedMode) {
            setAdvancedMode(false);
            api.toggleAdvancedMode(sessionId, false).catch(console.error);
          }
        }}
        animaEnabled={animaEnabled}
        onToggleAnima={() => {
          const newVal = !animaEnabled;
          setAnimaEnabled(newVal);
          api.toggleAnima(sessionId, newVal).then(() => {
            if (newVal) {
              api.getDevelopmentStatus(sessionId).then((s) => {
                setDevStage(s.current_stage);
                setDevSpeed(s.speed);
              }).catch(console.error);
            }
          }).catch(console.error);
        }}
        devStage={devStage}
        devSpeed={devSpeed}
        onDevStageChange={(stage) => {
          setDevStage(stage);
          api.setDevelopmentConfig(sessionId, { initial_stage: stage }).then((res) => {
            setDevStage(res.current_stage);
          }).catch(console.error);
        }}
        onDevSpeedChange={(speed) => {
          setDevSpeed(speed);
          api.setDevelopmentConfig(sessionId, { speed }).then((res) => {
            setDevSpeed(res.speed);
          }).catch(console.error);
        }}
        voiceEnabled={voiceEnabled}
        voiceLoading={voiceLoading}
        onToggleVoice={async () => {
          if (voiceLoading) return;
          const newVal = !voiceEnabled;
          const newMode = !newVal ? "text_only" : micEnabled ? "full_voice" : "voice_out";
          setVoiceLoading(true);
          try {
            await api.configureVoice(sessionId, { mode: newMode });
            setVoiceEnabled(newVal);
            if (!newVal) setMicEnabled(false);
          } catch (err) {
            console.error("Voice toggle failed:", err);
            setVoiceEnabled(false);
            setMicEnabled(false);
          } finally {
            setVoiceLoading(false);
          }
        }}
        micEnabled={micEnabled}
        micReady={micReady}
        onMicReady={setMicReady}
        onStreamReady={setMicStream}
        onToggleMic={async () => {
          if (voiceLoading) return;
          const newVal = !micEnabled;
          const newMode = newVal ? "full_voice" : "voice_out";
          setVoiceLoading(true);
          try {
            await api.configureVoice(sessionId, { mode: newMode });
            setMicEnabled(newVal);
          } catch (err) {
            console.error("Mic toggle failed:", err);
            setMicEnabled(false);
          } finally {
            setVoiceLoading(false);
          }
        }}
        showEmotionSidebar={showEmotionSidebar}
        onToggleEmotionSidebar={() => { if (showEmotionSidebar || !panelLimitReached) setShowEmotionSidebar(p => !p); }}
        showPipeline={showPipeline}
        onTogglePipeline={() => { if (showPipeline || !panelLimitReached) setShowPipeline(p => !p); }}
        showGenesis={showGenesis}
        onToggleGenesis={() => { if (showGenesis || !panelLimitReached) setShowGenesis(p => !p); }}
        showOrb={showOrb}
        onToggleOrb={() => { if (showOrb || !panelLimitReached) setShowOrb(p => !p); }}
        showAvatar={showAvatar}
        onToggleAvatar={() => { if (showAvatar || !panelLimitReached) setShowAvatar(p => !p); }}
        panelLimitReached={panelLimitReached}
        sessionId={sessionId}
        onNewSession={handleNewSession}
        onSave={handleSave}
        saving={saving}
        onExport={handleExport}
        onExportPortable={handleExportPortable}
        exporting={exporting}
        exportingPortable={exportingPortable}
        currentProvider={currentProvider}
        onExitRaw={handleExitRaw}
        modelLocked={modelLocked}
        onExitAutonomous={() => setMode("companion")}
      />

      <div className="app__main">
        {/* Left sidebar: Emotion state (collapsible) */}
        {showEmotionSidebar && !isCalibrationMode && !isSandboxMode && !isArenaMode && !isMirrorMode && !isRawMode && !isAutonomousMode && (
          <div className="app__sidebar-left">
            <EmotionalStatePanel state={activeEmotionalState} history={circumplexHistory} />
          </div>
        )}

        {/* Center: Chat + Research/Calibration/Sandbox panels */}
        <div className="app__center">
          {isCalibrationMode ? (
            <ErrorBoundary fallbackLabel="Calibration"><CalibrationPanel sessionId={sessionId} /></ErrorBoundary>
          ) : isSandboxMode ? (
            <ErrorBoundary fallbackLabel="Sandbox"><SandboxPanel sessionId={sessionId} connected={connected} /></ErrorBoundary>
          ) : isArenaMode ? (
            <ErrorBoundary fallbackLabel="Arena"><ArenaPanel sessionId={sessionId} connected={connected} /></ErrorBoundary>
          ) : isMirrorMode ? (
            <ErrorBoundary fallbackLabel="Mirror Test"><MirrorTest sessionId={sessionId} connected={connected} /></ErrorBoundary>
          ) : isRawMode ? (
            <ErrorBoundary fallbackLabel="Raw Mode"><RawChatPanel connected={connected} currentProvider={currentProvider} voiceEnabled={voiceEnabled} voiceInputEnabled={micEnabled && micReady} micStream={micStream} /></ErrorBoundary>
          ) : mode === "autonomous" ? (
            <ErrorBoundary fallbackLabel="Autonomous Research"><AutonomousResearchPanel
              connected={connected}
              currentProvider={currentProvider}
              onEmotionalStateUpdate={(s) => setAutoResearchEmotionalState(s)}
              onRunningChange={(r) => setModelLocked(r)}
              persistedSessionId={autonomousSessionId}
              onSessionId={setAutonomousSessionId}
            /></ErrorBoundary>
          ) : (
            <>
              <ErrorBoundary fallbackLabel="Chat"><ChatPanel
                messages={messages}
                onSend={handleSend}
                loading={loading}
                disabled={!connected || !currentModel}
                disabledReason={!connected ? "Backend not connected" : "Select a model first"}
                voiceInputEnabled={micEnabled && micReady}
                micStream={micStream}
                onSendAudio={handleSendAudio}
                sessionId={sessionId}
                voiceEnabled={voiceEnabled}
                onClearChat={() => setMessages([])}
                onAudioState={(a, p) => { setAudioAnalyser(a); setIsSpeaking(p); }}
              /></ErrorBoundary>
              {isResearchMode && researchData && (
                <div className="app__research-drawer">
                  <ErrorBoundary fallbackLabel="Research"><ResearchPanel data={researchData} /></ErrorBoundary>
                </div>
              )}
            </>
          )}
        </div>

        {/* Right panel: Pipeline Viewer, Network graph, Genesis */}
        {(showPipeline || showNetwork || showGenesis || showOrb || showAvatar) && !isCalibrationMode && !isSandboxMode && !isArenaMode && !isMirrorMode && !isRawMode && !isAutonomousMode && (
          <div className="app__sidebar-right">
            {showAvatar && (
              <ErrorBoundary fallbackLabel="Avatar"><EmotionAvatar emotionalState={activeEmotionalState} analyser={audioAnalyser} speaking={isSpeaking} /></ErrorBoundary>
            )}
            {showOrb && (
              <ErrorBoundary fallbackLabel="Qualia Orb"><QualiaOrb emotionalState={activeEmotionalState} metaphor={researchData?.phenomenology?.current_profile?.metaphor ?? null} /></ErrorBoundary>
            )}
            {showGenesis && (
              <ErrorBoundary fallbackLabel="Genesis"><EmotionGenesis emotionalState={activeEmotionalState} /></ErrorBoundary>
            )}
            {showPipeline && (
              <ErrorBoundary fallbackLabel="Pipeline"><PipelineViewer
                trace={pipelineTrace}
                detailed={pipelineDetailed}
                onToggleDetailed={() => setPipelineDetailed(p => !p)}
              /></ErrorBoundary>
            )}
            {showNetwork && (
              <ErrorBoundary fallbackLabel="Network"><EmotionNetwork state={emotionalState} history={emotionHistory} /></ErrorBoundary>
            )}
          </div>
        )}
      </div>

      <JourneyTimeline journey={journey} />
    </div>
  );
}
