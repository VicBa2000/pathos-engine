import { useState, useCallback, useEffect, useRef } from "react";
import type { EmotionalState, ChatMessage } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { EmotionAvatar } from "./EmotionAvatar";
import { EmotionGenesis } from "./EmotionGenesis";
import { VoiceOrb } from "./VoiceOrb";
import { MicInput } from "./MicInput";
import * as api from "../api/client";
import "./RawChatPanel.css";

interface Props {
  connected: boolean;
  currentProvider: string;
  voiceEnabled?: boolean;
  voiceInputEnabled?: boolean;
  micStream?: MediaStream | null;
}

export function RawChatPanel({ connected, currentProvider, voiceEnabled, voiceInputEnabled, micStream }: Props) {
  const [accepted, setAccepted] = useState(false);
  const [sessionId] = useState(() => `raw-${crypto.randomUUID()}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [emotionalState, setEmotionalState] = useState<EmotionalState | null>(null);
  const [extremeMode, setExtremeMode] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const isOllama = currentProvider === "ollama";

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Reset session on unmount
  useEffect(() => {
    return () => {
      api.rawReset(sessionId).catch(() => {});
    };
  }, [sessionId]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading || !isOllama) return;
    const msg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setLoading(true);

    try {
      const res = await api.rawChat(msg, sessionId);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: res.response,
        emotional_state: res.emotional_state,
        audioAvailable: res.audio_available,
        turnNumber: res.turn_number,
      }]);
      setEmotionalState(res.emotional_state);
    } catch (err) {
      const errorText = err instanceof Error ? err.message : "Unknown error";
      setMessages(prev => [...prev, { role: "assistant", content: `Error: ${errorText}` }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId, isOllama]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canRecord = !loading && !transcribing && !!voiceInputEnabled;

  const handleMicRecorded = useCallback(async (blob: Blob) => {
    setTranscribing(true);
    try {
      const result = await api.sendAudio(sessionId, blob);
      if (result.text && result.text.trim()) {
        const msg = result.text.trim();
        setMessages(prev => [...prev, { role: "user", content: msg }]);
        setLoading(true);
        try {
          const res = await api.rawChat(msg, sessionId);
          setMessages(prev => [...prev, {
            role: "assistant",
            content: res.response,
            emotional_state: res.emotional_state,
            audioAvailable: res.audio_available,
            turnNumber: res.turn_number,
          }]);
          setEmotionalState(res.emotional_state);
        } catch (err) {
          const errorText = err instanceof Error ? err.message : "Unknown error";
          setMessages(prev => [...prev, { role: "assistant", content: `Error: ${errorText}` }]);
        } finally {
          setLoading(false);
        }
      }
    } catch (err) {
      console.error("Transcription failed:", err);
    } finally {
      setTranscribing(false);
    }
  }, [sessionId]);

  const handleReset = useCallback(async () => {
    await api.rawReset(sessionId).catch(() => {});
    setMessages([]);
    setEmotionalState(null);
    setExtremeMode(false);
  }, [sessionId]);

  const handleToggleExtreme = useCallback(async () => {
    const newVal = !extremeMode;
    await api.rawToggleExtreme(sessionId, newVal).catch(() => {});
    setExtremeMode(newVal);
  }, [extremeMode, sessionId]);

  // Warning gate
  if (!accepted) {
    return (
      <div className="raw-warning">
        <div className="raw-warning__card">
          <div className="raw-warning__icon">&#x26A0;</div>
          <h2 className="raw-warning__title">Raw Mode</h2>
          <p className="raw-warning__desc">
            This mode removes all emotional filters and social constraints from the agent's responses.
            The agent will express emotions exactly as a human would — including anger, insults,
            hostility, profanity, and emotional outbursts.
          </p>
          <div className="raw-warning__rules">
            <div className="raw-warning__rule">Requires local Ollama model (cloud providers block unfiltered content)</div>
            <div className="raw-warning__rule">Nothing is saved — no conversation, no memory, no training data</div>
            <div className="raw-warning__rule">Cannot export or package this session</div>
            <div className="raw-warning__rule">Session is destroyed when you leave this tab</div>
          </div>
          {!isOllama && (
            <div className="raw-warning__blocked">
              Current provider is <strong>{currentProvider}</strong> — switch to a local Ollama model first.
            </div>
          )}
          <button
            className="raw-warning__accept"
            onClick={() => setAccepted(true)}
            disabled={!isOllama || !connected}
          >
            I understand — Enter Raw Mode
          </button>
        </div>
      </div>
    );
  }

  const primaryColor = emotionalState
    ? EMOTION_COLORS[emotionalState.primary_emotion] || "#6c63ff"
    : "#6c63ff";

  return (
    <div className="raw-chat">
      {/* Chat area */}
      <div className="raw-chat__main">
        <div className="raw-chat__header">
          <span className="raw-chat__badge">RAW MODE</span>
          {emotionalState && (
            <span className="raw-chat__emotion" style={{ color: primaryColor }}>
              {emotionalState.primary_emotion} ({emotionalState.intensity.toFixed(2)})
            </span>
          )}
          <button
            className={`raw-chat__extreme-toggle ${extremeMode ? "raw-chat__extreme-toggle--active" : ""}`}
            onClick={handleToggleExtreme}
          >
            {extremeMode ? "EXTREME ON" : "Extreme"}
          </button>
          <button className="raw-chat__reset" onClick={handleReset}>Reset</button>
        </div>

        <div className="raw-chat__messages">
          {messages.length === 0 && (
            <div className="raw-chat__hint">
              Say something. The agent will respond with unfiltered emotions.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`raw-chat__msg raw-chat__msg--${m.role}`}>
              <div className="raw-chat__bubble">{m.content}</div>
              {m.role === "assistant" && voiceEnabled && m.audioAvailable && (
                <RawInlineVoice
                  sessionId={sessionId}
                  color={m.emotional_state ? EMOTION_COLORS[m.emotional_state.primary_emotion] || "#ef4444" : "#ef4444"}
                  turnNumber={m.turnNumber}
                />
              )}
            </div>
          ))}
          {transcribing && (
            <div className="raw-chat__msg raw-chat__msg--user">
              <div className="raw-chat__bubble raw-chat__bubble--transcribing">Listening...</div>
            </div>
          )}
          {loading && (
            <div className="raw-chat__msg raw-chat__msg--assistant">
              <div className="raw-chat__bubble raw-chat__bubble--loading">
                <span className="raw-chat__dots"><span>.</span><span>.</span><span>.</span></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="raw-chat__input-area">
          <textarea
            className="raw-chat__input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={transcribing ? "Transcribing..." : "Say something..."}
            rows={2}
            disabled={loading || transcribing}
          />
          <MicInput
            enabled={canRecord}
            onRecorded={handleMicRecorded}
            transcribing={transcribing}
            stream={micStream ?? null}
          />
          <button className="raw-chat__send" onClick={handleSend} disabled={loading || transcribing || !input.trim()}>
            Send
          </button>
        </div>
      </div>

      {/* Right panel: Avatar + Genesis */}
      <div className="raw-chat__visuals">
        <div className="raw-chat__visual-box">
          <EmotionAvatar emotionalState={emotionalState} />
        </div>
        <div className="raw-chat__visual-box">
          <EmotionGenesis emotionalState={emotionalState} />
        </div>
      </div>
    </div>
  );
}

/** Inline voice player for raw mode */
function RawInlineVoice({ sessionId, color, turnNumber }: { sessionId: string; color: string; turnNumber?: number }) {
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
      const audio = new Audio(url);
      audio.crossOrigin = "anonymous";
      audioRef.current = audio;

      if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
      const ctx = audioCtxRef.current;
      if (ctx.state === "suspended") await ctx.resume();

      sourceRef.current = ctx.createMediaElementSource(audio);
      const node = ctx.createAnalyser();
      node.fftSize = 256;
      node.smoothingTimeConstant = 0.7;
      sourceRef.current.connect(node);
      node.connect(ctx.destination);
      setAnalyser(node);

      audio.onended = () => setPlaying(false);
      audio.onpause = () => setPlaying(false);
      audio.play().catch(() => {});
      setPlaying(true);
    } catch {
      // Audio fetch failed silently
    }
  }

  useEffect(() => {
    if (!autoPlayedRef.current) {
      autoPlayedRef.current = true;
      loadAndPlay();
    }
    return () => {
      audioRef.current?.pause();
      sourceRef.current?.disconnect();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      audioCtxRef.current?.close().catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleToggle() {
    if (audioRef.current) {
      if (playing) {
        audioRef.current.pause();
        setPlaying(false);
      } else {
        if (audioCtxRef.current?.state === "suspended") await audioCtxRef.current.resume();
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
        setPlaying(true);
      }
      return;
    }
    await loadAndPlay();
  }

  return (
    <div className="raw-chat__voice-wrap">
      <VoiceOrb color={color} playing={playing} onToggle={handleToggle} analyser={analyser} />
    </div>
  );
}
