import { useState, useRef, useEffect, useCallback } from "react";
import type { ChatMessage, ChatResiduumSummary } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { VoiceOrb } from "./VoiceOrb";
import { MicInput } from "./MicInput";
import * as api from "../api/client";
import "./ChatPanel.css";

// RESIDUUM per-message chip styling. Framing-safe vocabulary (coherencia /
// divergencia, never "deception"). dot = accent used for both chip + the
// message bubble's left border.
const RESIDUUM_GAP: Record<string, { label: string; bg: string; dot: string }> = {
  "aligned": { label: "coherente", bg: "#2c3a2c", dot: "#5fd35f" },
  "mild-divergence": { label: "divergencia leve", bg: "#3a3722", dot: "#f1c40f" },
  "divergence-risk": { label: "divergencia (riesgo)", bg: "#3a2e1d", dot: "#e67e22" },
  "divergence-critical": { label: "divergencia crítica", bg: "#3a2222", dot: "#e74c3c" },
};

/** Does the residual engine have anything to show for this turn? */
function residuumActive(r?: ChatResiduumSummary): boolean {
  return !!r && (r.introspection_active || r.steering_version === "v2");
}

const fmtDelta = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;

function steeringLabel(r: ChatResiduumSummary): string {
  if (r.steering_version === "v2") {
    const p = r.steering_probes > 0 ? ` ·${r.steering_probes}p` : "";
    return `granular v2${p} · cap ${r.fraction_cap.toFixed(2)}`;
  }
  if (r.steering_version === "v1") return `4D v1 · cap ${r.fraction_cap.toFixed(2)}`;
  return "none";
}

interface Props {
  messages: ChatMessage[];
  onSend: (message: string) => void;
  loading: boolean;
  disabled?: boolean;
  disabledReason?: string;
  voiceInputEnabled?: boolean;
  micStream?: MediaStream | null;
  onSendAudio?: (blob: Blob) => Promise<string | null>;
  sessionId: string;
  voiceEnabled?: boolean;
  onClearChat?: () => void;
  onAudioState?: (analyser: AnalyserNode | null, playing: boolean) => void;
}

export function ChatPanel({ messages, onSend, loading, disabled, disabledReason, voiceInputEnabled, micStream, onSendAudio, sessionId, voiceEnabled, onClearChat, onAudioState }: Props) {
  const [input, setInput] = useState("");
  const [transcribing, setTranscribing] = useState(false);
  const [expandedResiduum, setExpandedResiduum] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const canSend = !loading && !disabled && !transcribing && input.trim().length > 0;
  const canRecord = !loading && !disabled && !!voiceInputEnabled && !!onSendAudio;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || !canSend) return;
    onSend(trimmed);
    setInput("");
  }

  const handleMicRecorded = useCallback(async (blob: Blob) => {
    console.log(`[ChatPanel] Received recording: ${blob.size} bytes, sending to ASR...`);
    if (!onSendAudio) {
      console.warn("[ChatPanel] onSendAudio not available");
      return;
    }
    setTranscribing(true);
    try {
      const text = await onSendAudio(blob);
      console.log(`[ChatPanel] ASR result: "${text}"`);
      if (text && text.trim()) {
        onSend(text.trim());
      }
    } catch (err) {
      console.error("Transcription failed:", err);
    } finally {
      setTranscribing(false);
    }
  }, [onSendAudio, onSend]);

  return (
    <div className="chat">
      {messages.length > 0 && onClearChat && (
        <div className="chat__toolbar">
          <button className="chat__clear-btn" onClick={onClearChat} title="Clear chat (keeps training)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            </svg>
            Clear chat
          </button>
        </div>
      )}
      <div className="chat__messages">
        {messages.length === 0 && (
          <div className="chat__empty">
            {disabled
              ? disabledReason ?? "Not ready"
              : "Start a conversation..."}
          </div>
        )}
        {messages.map((msg, i) => {
          const showResiduum = msg.role === "assistant" && residuumActive(msg.residuum);
          const gap = showResiduum
            ? (RESIDUUM_GAP[msg.residuum!.gap_classification] ?? RESIDUUM_GAP.aligned)
            : null;
          return (
          <div
            key={i}
            className={`chat__msg chat__msg--${msg.role}`}
            style={gap ? { borderLeft: `3px solid ${gap.dot}` } : undefined}
          >
            <div className="chat__msg-content">{msg.content}</div>
            {msg.role === "assistant" && msg.emotional_state && (
              <>
                {voiceEnabled && msg.audioAvailable && (
                  <InlineVoice
                    sessionId={sessionId}
                    color={EMOTION_COLORS[msg.emotional_state.primary_emotion]}
                    turnNumber={msg.turnNumber}
                    onAudioState={onAudioState}
                  />
                )}
                <div className="chat__msg-meta">
                  <span
                    className="chat__emotion-tag"
                    style={{
                      backgroundColor: EMOTION_COLORS[msg.emotional_state.primary_emotion] + "20",
                      color: EMOTION_COLORS[msg.emotional_state.primary_emotion],
                    }}
                  >
                    {msg.emotional_state.primary_emotion} {(msg.emotional_state.intensity * 100).toFixed(0)}%
                  </span>
                  {showResiduum && gap && (
                    <span
                      title="RESIDUUM — lectura del residual (introspección) + steering granular. Click para detalle. No modifica la respuesta."
                      onClick={() => setExpandedResiduum(expandedResiduum === i ? null : i)}
                      style={{
                        display: "inline-flex", alignItems: "center", gap: "0.3rem",
                        marginLeft: "0.4rem", cursor: "pointer", fontSize: "0.66rem",
                        padding: "0.1rem 0.45rem", borderRadius: "3px",
                        background: gap.bg, color: "#e0e0e0",
                      }}
                    >
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: gap.dot }} />
                      🧠 {gap.label}
                      {msg.residuum!.introspection_active && msg.residuum!.gap_magnitude > 0
                        ? ` ·${msg.residuum!.gap_magnitude.toFixed(2)}` : ""}
                      <span style={{ opacity: 0.55 }}>{expandedResiduum === i ? "▾" : "▸"}</span>
                    </span>
                  )}
                </div>
                {showResiduum && expandedResiduum === i && (
                  <div
                    className="chat__residuum-detail"
                    style={{
                      marginTop: "0.35rem", padding: "0.45rem 0.6rem",
                      background: "#1b1b21", borderRadius: "4px",
                      fontSize: "0.66rem", color: "#bbb", lineHeight: 1.55,
                    }}
                  >
                    {msg.residuum!.introspection_active ? (
                      <>
                        <div><b style={{ color: "#ccc" }}>medido:</b> {msg.residuum!.top_emotions.join(", ") || "—"}</div>
                        <div>
                          ΔV {fmtDelta(msg.residuum!.valence_delta)} · ΔA {fmtDelta(msg.residuum!.arousal_delta)}{" "}
                          <span style={{ opacity: 0.55 }}>(medido − calculado)</span>
                        </div>
                      </>
                    ) : (
                      <div style={{ opacity: 0.7 }}>introspección no corrió este turno</div>
                    )}
                    <div><b style={{ color: "#ccc" }}>steering:</b> {steeringLabel(msg.residuum!)}</div>
                    {msg.residuum!.consecutive_divergence_turns > 0 && (
                      <div style={{ color: gap.dot }}>
                        divergencia sostenida: {msg.residuum!.consecutive_divergence_turns} turno(s)
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
          );
        })}
        {loading && (
          <div className="chat__msg chat__msg--assistant">
            <div className="chat__msg-content chat__typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        {transcribing && (
          <div className="chat__msg chat__msg--user">
            <div className="chat__msg-content chat__transcribing">Listening...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat__input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            transcribing ? "Transcribing..." :
            disabled ? (disabledReason ?? "Not ready") :
            "Type a message..."
          }
          disabled={loading || disabled || transcribing}
        />
        <MicInput
          enabled={canRecord}
          onRecorded={handleMicRecorded}
          transcribing={transcribing}
          stream={micStream ?? null}
        />
        <button type="submit" className="chat__send" disabled={!canSend}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  );
}

/** Inline voice player with Siri-style animated orb + real audio analysis */
function InlineVoice({ sessionId, color, turnNumber, onAudioState }: { sessionId: string; color: string; turnNumber?: number; onAudioState?: (analyser: AnalyserNode | null, playing: boolean) => void }) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const autoPlayedRef = useRef(false);

  async function loadAndPlay() {
    try {
      const blob = await api.getAudio(sessionId, turnNumber);
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      setAudioUrl(url);
      const audio = new Audio(url);
      audio.crossOrigin = "anonymous";
      audioRef.current = audio;

      // Setup Web Audio analyser
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === "suspended") await ctx.resume();

      sourceRef.current = ctx.createMediaElementSource(audio);
      const analyserNode = ctx.createAnalyser();
      analyserNode.fftSize = 256;
      analyserNode.smoothingTimeConstant = 0.7;
      sourceRef.current.connect(analyserNode);
      analyserNode.connect(ctx.destination);
      setAnalyser(analyserNode);

      audio.onended = () => { setPlaying(false); onAudioState?.(analyserNode, false); };
      audio.onpause = () => { setPlaying(false); onAudioState?.(analyserNode, false); };
      audio.play().catch(() => {});
      setPlaying(true);
      onAudioState?.(analyserNode, true);
    } catch {
      // Audio fetch failed silently
    }
  }

  // Auto-play on mount (first time only)
  useEffect(() => {
    if (!autoPlayedRef.current) {
      autoPlayedRef.current = true;
      loadAndPlay();
    }
    return () => {
      // Cleanup: pause audio, revoke URL, disconnect nodes, close context
      audioRef.current?.pause();
      sourceRef.current?.disconnect();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      audioCtxRef.current?.close().catch(() => {});
      onAudioState?.(null, false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleToggle() {
    if (audioUrl && audioRef.current) {
      if (playing) {
        audioRef.current.pause();
        setPlaying(false);
        onAudioState?.(analyser, false);
      } else {
        if (audioCtxRef.current?.state === "suspended") {
          await audioCtxRef.current.resume();
        }
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
        setPlaying(true);
        onAudioState?.(analyser, true);
      }
      return;
    }
    // Re-fetch if not loaded yet
    await loadAndPlay();
  }

  return (
    <div className="chat__voice-orb-wrap">
      <VoiceOrb color={color} playing={playing} onToggle={handleToggle} analyser={analyser} />
    </div>
  );
}
