import { useState, useRef, useEffect } from "react";
import type { AppMode } from "../types/emotion";
import { ModelSelector } from "./ModelSelector";
import { VoiceConfigPanel } from "./VoiceConfigPanel";
import { MicConfigPanel } from "./MicConfigPanel";
import { AgentSetupPanel } from "./AgentSetupPanel";
import { SignalsConfigPanel } from "./SignalsConfigPanel";
import { ConfirmModal } from "./ConfirmModal";
import "./ModeSelector.css";

interface Props {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
  connected: boolean;
  currentModel: string;
  onModelChanged: (provider: string, model: string) => void;
  // Settings toggles
  showNetwork: boolean;
  onToggleNetwork: () => void;
  showForecasting: boolean;
  onToggleForecasting: () => void;
  advancedMode: boolean;
  onToggleAdvancedMode: () => void;
  liteMode: boolean;
  onToggleLiteMode: () => void;
  animaEnabled: boolean;
  onToggleAnima: () => void;
  voiceEnabled: boolean;
  voiceLoading: boolean;
  onToggleVoice: () => void;
  micEnabled: boolean;
  onToggleMic: () => void;
  micReady: boolean;
  onMicReady: (ready: boolean) => void;
  onStreamReady: (stream: MediaStream | null) => void;
  // Sidebar toggles
  showEmotionSidebar: boolean;
  onToggleEmotionSidebar: () => void;
  showPipeline: boolean;
  onTogglePipeline: () => void;
  showGenesis: boolean;
  onToggleGenesis: () => void;
  showOrb: boolean;
  onToggleOrb: () => void;
  showAvatar: boolean;
  onToggleAvatar: () => void;
  panelLimitReached: boolean;
  sessionId: string;
  onNewSession: () => void;
  onSave: () => void;
  saving: boolean;
  onExport: () => void;
  onExportPortable: () => void;
  exporting: boolean;
  exportingPortable: boolean;
  currentProvider: string;
  onExitRaw?: () => void;
  modelLocked?: boolean;
  onExitAutonomous?: () => void;
}

export function ModeSelector({
  mode, onModeChange, connected, currentModel, onModelChanged,
  showNetwork, onToggleNetwork,
  showForecasting, onToggleForecasting,
  advancedMode, onToggleAdvancedMode,
  liteMode, onToggleLiteMode,
  animaEnabled, onToggleAnima,
  voiceEnabled, voiceLoading, onToggleVoice,
  micEnabled, onToggleMic, micReady, onMicReady, onStreamReady,
  showEmotionSidebar, onToggleEmotionSidebar, showPipeline, onTogglePipeline, showGenesis, onToggleGenesis, showOrb, onToggleOrb, showAvatar, onToggleAvatar,
  panelLimitReached,
  sessionId, onNewSession, onSave, saving, onExport, onExportPortable, exporting, exportingPortable,
  currentProvider, onExitRaw, modelLocked, onExitAutonomous,
}: Props) {
  const isRawMode = mode === "raw";
  const isAutonomousMode = mode === "autonomous";
  const [confirmAction, setConfirmAction] = useState<{ title: string; message: string; action: () => void; danger?: boolean } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [voiceConfigOpen, setVoiceConfigOpen] = useState(false);
  const [micConfigOpen, setMicConfigOpen] = useState(false);
  const [agentSetupOpen, setAgentSetupOpen] = useState(false);
  const [signalsConfigOpen, setSignalsConfigOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  // Close settings on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setSettingsOpen(false);
      }
    }
    if (settingsOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [settingsOpen]);

  return (
    <><header className="header">
      <div className="header__left">
        <span className="header__logo">P</span>
        <span className="header__title">PATHOS</span>
        <span className={`header__status ${connected ? "header__status--on" : ""}`}>
          {connected ? "Connected" : "Offline"}
        </span>
      </div>

      <div className="header__center">
        {isRawMode ? (
          <>
            <span className="header__raw-badge">RAW MODE</span>
            <button className="header__exit-raw" onClick={onExitRaw}>
              EXIT
            </button>
          </>
        ) : isAutonomousMode ? (
          <>
            <span className="header__raw-badge header__raw-badge--auto">AUTO-RESEARCH</span>
            <button className="header__exit-raw" onClick={onExitAutonomous}>
              EXIT
            </button>
          </>
        ) : (
          (["companion", "research", "sandbox", "arena", "mirror", "autonomous", "raw", "calibration"] as AppMode[]).map(m => {
            const labels: Record<string, string> = { companion: "Chat", research: "Research", sandbox: "Sandbox", arena: "Arena", mirror: "Mirror", autonomous: "Auto-Research", raw: "Raw", calibration: "Calibrate" };
            return (
              <button
                key={m}
                className={`header__mode ${mode === m ? "header__mode--active" : ""} ${m === "raw" ? "header__mode--raw" : ""}`}
                onClick={() => onModeChange(m)}
              >
                {labels[m]}
              </button>
            );
          })
        )}
      </div>

      <div className="header__right">
        <ModelSelector currentModel={currentModel} sessionId={sessionId} onModelChanged={onModelChanged} localOnly={isRawMode} locked={modelLocked} />

        {!isRawMode && !isAutonomousMode && (
          <div className="header__agent-wrap">
            <button
              className={`header__agent-btn ${agentSetupOpen ? "header__agent-btn--open" : ""}`}
              onClick={() => setAgentSetupOpen(p => !p)}
              title="Agent Setup"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </button>
            <AgentSetupPanel
              sessionId={sessionId}
              visible={agentSetupOpen}
              onClose={() => setAgentSetupOpen(false)}
            />
          </div>
        )}

        {voiceEnabled && (
          <>
            <div className="header__voice-wrap">
              <button
                className={`header__voice-btn ${voiceConfigOpen ? "header__voice-btn--open" : ""}`}
                onClick={() => setVoiceConfigOpen(p => !p)}
                title="Voice settings"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14" /><path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                </svg>
              </button>
              <VoiceConfigPanel
                sessionId={sessionId}
                visible={voiceConfigOpen}
                onClose={() => setVoiceConfigOpen(false)}
              />
            </div>
            {micEnabled && (
              <div className="header__mic-wrap">
                <button
                  className={`header__mic-btn ${micConfigOpen ? "header__mic-btn--open" : ""} ${micReady ? "header__mic-btn--ready" : ""}`}
                  onClick={() => setMicConfigOpen(p => !p)}
                  title="Microphone settings"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                  </svg>
                </button>
                <MicConfigPanel
                  visible={micConfigOpen}
                  onClose={() => setMicConfigOpen(false)}
                  onMicReady={onMicReady}
                  onStreamReady={onStreamReady}
                />
              </div>
            )}
          </>
        )}

        {/* External Signals config button — always visible, independent of voice */}
        <div className="header__signals-wrap">
          <button
            className={`header__signals-btn ${signalsConfigOpen ? "header__signals-btn--open" : ""}`}
            onClick={() => setSignalsConfigOpen(p => !p)}
            title="External signals configuration"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </button>
          <SignalsConfigPanel
            visible={signalsConfigOpen}
            onClose={() => setSignalsConfigOpen(false)}
            sessionId={sessionId}
          />
        </div>

        <div className="header__settings-wrap" ref={settingsRef}>
          <button
            className={`header__settings-btn ${settingsOpen ? "header__settings-btn--open" : ""}`}
            onClick={() => setSettingsOpen(p => !p)}
            title="Settings"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>

          {settingsOpen && (
            <div className="settings-panel">
              {!isRawMode && !isAutonomousMode && (
                <div className="settings-panel__section">
                  <div className="settings-panel__title">Display</div>
                  <SettingsSwitch label="Emotion Panel" on={showEmotionSidebar} onToggle={onToggleEmotionSidebar} disabled={!showEmotionSidebar && panelLimitReached} hint={!showEmotionSidebar && panelLimitReached ? "Max panels reached" : undefined} />
                  <SettingsSwitch label="Network Graph" on={showNetwork} onToggle={onToggleNetwork} disabled={!showNetwork && panelLimitReached} hint={!showNetwork && panelLimitReached ? "Max panels reached" : undefined} />
                  <SettingsSwitch label="Pipeline Viewer" on={showPipeline} onToggle={onTogglePipeline} disabled={!showPipeline && panelLimitReached} hint={!showPipeline && panelLimitReached ? "Max panels reached" : "Visual step-by-step flow"} />
                  <SettingsSwitch label="Emotion Genesis" on={showGenesis} onToggle={onToggleGenesis} disabled={!showGenesis && panelLimitReached} hint={!showGenesis && panelLimitReached ? "Max panels reached" : "Living organism view"} />
                  <SettingsSwitch label="Qualia Orb" on={showOrb} onToggle={onToggleOrb} disabled={!showOrb && panelLimitReached} hint={!showOrb && panelLimitReached ? "Max panels reached" : "3D phenomenological orb"} />
                  <SettingsSwitch label="Emotion Avatar" on={showAvatar} onToggle={onToggleAvatar} disabled={!showAvatar && panelLimitReached} hint={!showAvatar && panelLimitReached ? "Max panels reached" : "Animated face view"} />
                  {panelLimitReached && (
                    <div className="settings-panel__limit-notice">Max 3 panels — disable one to enable another</div>
                  )}
                </div>
              )}

              <div className="settings-panel__section">
                <div className="settings-panel__title">Engine</div>
                <SettingsSwitch label="Advanced" on={advancedMode} onToggle={onToggleAdvancedMode} hint="All emotional systems" />
                <SettingsSwitch label="Lite Mode" on={liteMode} onToggle={onToggleLiteMode} hint="1 LLM call, keyword appraisal" />
                {!isRawMode && !isAutonomousMode && (
                  <SettingsSwitch label="Forecasting" on={showForecasting} onToggle={onToggleForecasting} hint="Predict emotional impact" />
                )}
                <SettingsSwitch label="ANIMA v5" on={animaEnabled} onToggle={onToggleAnima} hint="Emergent emotion pillars" />
              </div>

              <div className="settings-panel__section">
                <div className="settings-panel__title">Voice</div>
                <SettingsSwitch label="Voice Output" on={voiceEnabled} onToggle={onToggleVoice} loading={voiceLoading} hint={voiceLoading ? "Loading model..." : "TTS with emotion"} />
                <SettingsSwitch label="Mic Input" on={micEnabled} onToggle={onToggleMic} disabled={!voiceEnabled || voiceLoading} hint="Whisper ASR" />
              </div>

              {!isRawMode && !isAutonomousMode && (
                <>
                  <div className="settings-panel__divider" />

                  <button className="settings-panel__action settings-panel__action--primary" onClick={() => setConfirmAction({
                    title: "Save Session",
                    message: "This will save the current emotional state, conversation history, and all system configurations to disk. You can restore it later.",
                    action: onSave,
                  })} disabled={saving}>
                    {saving ? "Saving..." : "Save Session"}
                  </button>
                  {currentProvider === "ollama" ? (
                    <button className="settings-panel__action" onClick={() => setConfirmAction({
                      title: "Export to Ollama",
                      message: "This will create an Ollama model with the current emotional architecture baked into its system prompt. The model will be available locally via 'ollama run pathos'.",
                      action: onExport,
                    })} disabled={exporting}>
                      {exporting ? "Exporting..." : "Export to Ollama"}
                    </button>
                  ) : (
                    <button className="settings-panel__action" disabled title="Only available with local Ollama models">
                      Export to Ollama (local only)
                    </button>
                  )}
                  <button className="settings-panel__action" onClick={() => setConfirmAction({
                    title: "Export Portable",
                    message: "This will package the current session into a standalone ZIP file that can run independently. API keys will NOT be included — you will need to configure them manually in the exported package.",
                    action: onExportPortable,
                  })} disabled={exportingPortable}>
                    {exportingPortable ? "Packaging..." : "Export Portable"}
                  </button>
                  <button className="settings-panel__action" onClick={() => setConfirmAction({
                    title: "New Session",
                    message: "This will reset the emotional state, clear all conversation history, and start fresh. Unsaved progress will be lost.",
                    action: onNewSession,
                    danger: true,
                  })}>
                    New Session
                  </button>

                  <div className="settings-panel__footer">
                    {sessionId.slice(0, 24)}...
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
    <ConfirmModal
      visible={confirmAction !== null}
      title={confirmAction?.title ?? ""}
      message={confirmAction?.message ?? ""}
      danger={confirmAction?.danger}
      onConfirm={() => { confirmAction?.action(); setConfirmAction(null); }}
      onCancel={() => setConfirmAction(null)}
    />
    </>
  );
}

function SettingsSwitch({ label, on, onToggle, disabled, hint, loading }: {
  label: string;
  on: boolean;
  onToggle: () => void;
  disabled?: boolean;
  hint?: string;
  loading?: boolean;
}) {
  const isDisabled = disabled || loading;
  return (
    <div className={`settings-switch ${isDisabled ? "settings-switch--disabled" : ""}`} onClick={isDisabled ? undefined : onToggle}>
      <div className="settings-switch__info">
        <span className="settings-switch__label">{label}</span>
        {hint && <span className={`settings-switch__hint ${loading ? "settings-switch__hint--loading" : ""}`}>{hint}</span>}
      </div>
      {loading ? (
        <div className="settings-switch__spinner" />
      ) : (
        <div className={`settings-switch__track ${on ? "settings-switch__track--on" : ""}`}>
          <div className="settings-switch__thumb" />
        </div>
      )}
    </div>
  );
}
