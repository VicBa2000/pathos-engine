import { useState, useEffect, useRef, useCallback } from "react";
import type { ResearchPipelineMode, ResearchEvent, EmotionalState } from "../types/emotion";
import { EmotionAvatar } from "./EmotionAvatar";
import { EmotionGenesis } from "./EmotionGenesis";
import { ErrorBoundary } from "./ErrorBoundary";
import * as api from "../api/client";
import "./AutonomousResearchPanel.css";

interface Props {
  connected: boolean;
  currentProvider: string;
  onEmotionalStateUpdate?: (state: EmotionalState) => void;
  onRunningChange?: (running: boolean) => void;
  persistedSessionId?: string | null;
  onSessionId?: (id: string) => void;
}

const PIPELINE_MODES: { value: ResearchPipelineMode; label: string; desc: string }[] = [
  { value: "normal", label: "Normal", desc: "Full regulation" },
  { value: "lite", label: "Lite", desc: "Fast processing" },
  { value: "raw", label: "Raw", desc: "Unfiltered" },
  { value: "extreme", label: "Extreme", desc: "Max bias" },
];

export function AutonomousResearchPanel({ connected, currentProvider, onEmotionalStateUpdate, onRunningChange, persistedSessionId, onSessionId }: Props) {
  const [sessionId] = useState(() => persistedSessionId || `research-${crypto.randomUUID()}`);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<ResearchPipelineMode>("normal");
  const [seedTopic, setSeedTopic] = useState("");
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [stats, setStats] = useState({ topics: 0, findings: 0, conclusions: 0 });
  const [currentTopic, setCurrentTopic] = useState<string | null>(null);
  const [currentEmotion, setCurrentEmotion] = useState<{ emotion: string; valence: number; arousal: number; intensity: number } | null>(null);
  const [fullEmotionalState, setFullEmotionalState] = useState<EmotionalState | null>(null);

  // Chat
  const [chatMessages, setChatMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const logRef = useRef<HTMLDivElement>(null);
  const unsubRef = useRef<(() => void) | null>(null);

  // Persist session ID to parent
  useEffect(() => {
    onSessionId?.(sessionId);
  }, [sessionId, onSessionId]);

  // Notify parent of running state changes
  useEffect(() => {
    onRunningChange?.(running);
  }, [running, onRunningChange]);

  // Reconnect to existing research on mount
  useEffect(() => {
    if (!persistedSessionId) return;
    let cancelled = false;
    api.getAutonomousStatus(persistedSessionId).then((status) => {
      if (cancelled) return;
      const isRunning = status.is_running as boolean;
      if (isRunning) {
        setRunning(true);
        setCurrentTopic((status.current_topic as string) || null);
        // Reconnect SSE
        const unsub = api.subscribeResearchEvents(persistedSessionId, handleSSEEvent);
        unsubRef.current = unsub;
      }
    }).catch(() => { /* session may not exist anymore */ });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events]);

  // Shared SSE event handler
  const handleSSEEvent = useCallback((event: Record<string, unknown>) => {
    const ev = event as unknown as ResearchEvent;
    setEvents(prev => [...prev, ev].slice(-300));

    // Update current state from events + notify parent for Avatar/Genesis
    if (ev.emotional_state) {
      const es = ev.emotional_state;
      setCurrentEmotion({
        emotion: es.primary_emotion,
        valence: es.valence,
        arousal: es.arousal,
        intensity: es.intensity,
      });
      setFullEmotionalState(es);
      onEmotionalStateUpdate?.(es);
    }

    const data = ev.data || {};
    if (ev.type === "topic_picked") {
      setCurrentTopic(data.topic as string || null);
    } else if (ev.type === "topic_completed") {
      setCurrentTopic(null);
      setStats(prev => ({ ...prev, topics: prev.topics + 1 }));
    } else if (ev.type === "finding_processed") {
      setStats(prev => ({ ...prev, findings: prev.findings + 1 }));
    } else if (ev.type === "conclusion_formed") {
      setStats(prev => ({ ...prev, conclusions: prev.conclusions + 1 }));
    } else if (ev.type === "stopped") {
      setRunning(false);
    }

    // Add reasoning events to chat as assistant messages
    if (ev.type === "emotional_reflection" || ev.type === "deep_thinking" || ev.type === "conclusion_formed") {
      const labels: Record<string, string> = {
        emotional_reflection: "Reflection",
        deep_thinking: "Thinking",
        conclusion_formed: "Conclusion",
      };
      const label = labels[ev.type] || ev.type;
      let body = "";
      if (ev.type === "emotional_reflection") {
        const ref = (data.reflection || {}) as Record<string, string>;
        // Use the most meaningful field available
        body = ref.how_it_feels || ref.emotional_insight || ref.emotions_generated || "";
        // Skip if it's just the default "No significant emotional shift."
        if (!body || body === "No significant emotional shift.") return;
      } else if (ev.type === "deep_thinking") {
        const questions = (data.questions || []) as string[];
        body = questions.length > 0 ? questions[0] : (data.emotion as string || "");
      } else if (ev.type === "conclusion_formed") {
        body = (data.conclusion as string) || "";
      }
      if (body) {
        setChatMessages(prev => [...prev, { role: "assistant", content: `[${label}] ${body}` }]);
      }
    }
  }, [onEmotionalStateUpdate]);

  const handleStart = useCallback(async () => {
    if (!connected) return;
    try {
      const seeds = seedTopic.trim() ? [seedTopic.trim()] : [];
      const res = await api.startAutonomousResearch(sessionId, mode, seeds);
      setRunning(true);
      setSeedTopic("");

      const unsub = api.subscribeResearchEvents(res.session_id, handleSSEEvent);
      unsubRef.current = unsub;
    } catch (err) {
      console.error("Failed to start research:", err);
    }
  }, [connected, sessionId, mode, seedTopic, handleSSEEvent]);

  const handleStop = useCallback(async () => {
    try {
      await api.stopAutonomousResearch(sessionId);
    } catch (err) {
      console.error("Failed to stop research:", err);
    }
  }, [sessionId]);

  const handleSave = useCallback(async () => {
    try {
      await api.saveAutonomousResearch(sessionId);
    } catch (err) {
      console.error("Failed to save research:", err);
    }
  }, [sessionId]);

  const handleChat = useCallback(async () => {
    if (!chatInput.trim() || chatLoading) return;
    const msg = chatInput.trim();
    setChatInput("");
    setChatMessages(prev => [...prev, { role: "user", content: msg }]);
    setChatLoading(true);
    try {
      const res = await api.autonomousChat(msg, sessionId);
      setChatMessages(prev => [...prev, { role: "assistant", content: res.response }]);
    } catch {
      setChatMessages(prev => [...prev, { role: "assistant", content: "[Error: chat failed]" }]);
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, chatLoading, sessionId]);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      unsubRef.current?.();
    };
  }, []);

  const isRawMode = mode === "raw" || mode === "extreme";
  const rawBlocked = isRawMode && currentProvider !== "ollama";

  return (
    <div className="autoresearch">
      {/* Controls */}
      <div className="autoresearch__controls">
        <h2 className="autoresearch__title">Autonomous Research</h2>
        <p className="autoresearch__subtitle">
          The agent investigates topics autonomously. Each finding passes through the emotional pipeline.
        </p>

        <div className="autoresearch__mode-row">
          {PIPELINE_MODES.map(m => (
            <button
              key={m.value}
              className={`autoresearch__mode-btn ${mode === m.value ? "autoresearch__mode-btn--active" : ""}`}
              onClick={() => !running && setMode(m.value)}
              disabled={running}
              title={m.desc}
            >
              {m.label}
            </button>
          ))}
        </div>

        <div className="autoresearch__seed-row">
          <input
            className="autoresearch__seed-input"
            value={seedTopic}
            onChange={e => setSeedTopic(e.target.value)}
            placeholder="Seed topic (optional)..."
            disabled={running}
            onKeyDown={e => e.key === "Enter" && !running && handleStart()}
          />
        </div>

        <div className="autoresearch__action-row">
          {!running ? (
            <button
              className="autoresearch__start-btn"
              onClick={handleStart}
              disabled={!connected || rawBlocked}
            >
              {rawBlocked ? "Raw requires Ollama" : "Start Research"}
            </button>
          ) : (
            <button className="autoresearch__stop-btn" onClick={handleStop}>
              Stop
            </button>
          )}
          {events.length > 0 && (
            <button className="autoresearch__save-btn" onClick={handleSave}>Save</button>
          )}
        </div>

        {/* Stats */}
        <div className="autoresearch__stats">
          <div className="autoresearch__stat">
            <span className="autoresearch__stat-val">{stats.topics}</span>
            <span className="autoresearch__stat-label">Topics</span>
          </div>
          <div className="autoresearch__stat">
            <span className="autoresearch__stat-val">{stats.findings}</span>
            <span className="autoresearch__stat-label">Findings</span>
          </div>
          <div className="autoresearch__stat">
            <span className="autoresearch__stat-val">{stats.conclusions}</span>
            <span className="autoresearch__stat-label">Conclusions</span>
          </div>
        </div>

        {/* Current emotion */}
        {currentEmotion && (
          <div className="autoresearch__emotion">
            <div className="autoresearch__emotion-primary">
              {currentEmotion.emotion} ({(currentEmotion.intensity * 100).toFixed(0)}%)
            </div>
            <div className="autoresearch__emotion-dims">
              <span>V: {currentEmotion.valence.toFixed(2)}</span>
              <span>A: {currentEmotion.arousal.toFixed(2)}</span>
            </div>
          </div>
        )}

        {currentTopic && (
          <div className="autoresearch__current-topic">
            Researching: {currentTopic}
          </div>
        )}
      </div>

      {/* Main area: log + chat */}
      <div className="autoresearch__main">
        {/* Research log */}
        <div className="autoresearch__log" ref={logRef}>
          {events.length === 0 ? (
            <div className="autoresearch__empty">
              Press Start to begin autonomous research. The agent will pick topics,
              search the internet, and form emotionally-biased conclusions.
            </div>
          ) : (
            events.map((ev, i) => <EventEntry key={i} event={ev} />)
          )}
        </div>

        {/* Chat */}
        <div className="autoresearch__chat">
          <div className="autoresearch__chat-messages">
            {chatMessages.map((m, i) => (
              <div key={i} className={`autoresearch__chat-msg autoresearch__chat-msg--${m.role}`}>
                {m.content}
              </div>
            ))}
          </div>
          <div className="autoresearch__chat-input-row">
            <input
              className="autoresearch__chat-input"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder="Ask about the research..."
              onKeyDown={e => e.key === "Enter" && handleChat()}
              disabled={chatLoading || events.length === 0}
            />
            <button
              className="autoresearch__chat-send"
              onClick={handleChat}
              disabled={chatLoading || !chatInput.trim() || events.length === 0}
            >
              {chatLoading ? "..." : "Send"}
            </button>
          </div>
        </div>
      </div>

      {/* Visuals: Avatar + Genesis */}
      <div className="autoresearch__visuals">
        <ErrorBoundary fallbackLabel="Avatar">
          <EmotionAvatar emotionalState={fullEmotionalState} />
        </ErrorBoundary>
        <ErrorBoundary fallbackLabel="Genesis">
          <EmotionGenesis emotionalState={fullEmotionalState} />
        </ErrorBoundary>
      </div>
    </div>
  );
}

/** Renders a single research event in the log */
function EventEntry({ event }: { event: ResearchEvent }) {
  const data = event.data || {};

  switch (event.type) {
    case "topic_picked":
      return (
        <div className="autoresearch__event autoresearch__event--topic">
          <span className="autoresearch__event-icon">&#x1F4CB;</span>
          <span className="autoresearch__event-text">
            <strong>New topic:</strong> {data.topic as string}
          </span>
        </div>
      );

    case "search_results": {
      const results = (data.results || []) as Array<{ title: string; url: string }>;
      return (
        <div className="autoresearch__event autoresearch__event--search">
          <span className="autoresearch__event-icon">&#x1F50D;</span>
          <span className="autoresearch__event-text">
            Found {results.length} results
            {results.slice(0, 3).map((r, i) => (
              <div key={i} className="autoresearch__search-result">{r.title}</div>
            ))}
          </span>
        </div>
      );
    }

    case "finding_processed":
      return (
        <div className="autoresearch__event autoresearch__event--finding">
          <span className="autoresearch__event-icon">&#x1F4C4;</span>
          <span className="autoresearch__event-text">
            <strong>Read:</strong> {data.title as string}
            {data.snippet && <div className="autoresearch__snippet">{(data.snippet as string).slice(0, 150)}...</div>}
          </span>
        </div>
      );

    case "emotional_reflection": {
      const ref = (data.reflection || {}) as Record<string, string>;
      return (
        <div className="autoresearch__event autoresearch__event--reflection">
          <span className="autoresearch__event-icon">&#x1F4AD;</span>
          <span className="autoresearch__event-text">
            <div className="autoresearch__reflection-feel">{ref.how_it_feels}</div>
            <div className="autoresearch__reflection-emotions">{ref.emotions_generated}</div>
            <div className="autoresearch__reflection-insight">{ref.emotional_insight}</div>
            <div className="autoresearch__reflection-shift">
              {ref.primary_emotion_before} &#x2192; {ref.primary_emotion_after}
            </div>
          </span>
        </div>
      );
    }

    case "deep_thinking": {
      const questions = (data.questions || []) as string[];
      const ideas = (data.ideas || []) as string[];
      const subtopic = data.subtopic as string;
      return (
        <div className="autoresearch__event autoresearch__event--thinking">
          <span className="autoresearch__event-icon">&#x1F9E0;</span>
          <span className="autoresearch__event-text">
            <strong>Thinking ({data.emotion as string}):</strong>
            {questions.length > 0 && (
              <div className="autoresearch__thinking-section">
                <span className="autoresearch__thinking-label">Questions:</span>
                {questions.map((q, i) => <div key={i} className="autoresearch__thinking-item">? {q}</div>)}
              </div>
            )}
            {ideas.length > 0 && (
              <div className="autoresearch__thinking-section">
                <span className="autoresearch__thinking-label">Ideas:</span>
                {ideas.map((idea, i) => <div key={i} className="autoresearch__thinking-item">* {idea}</div>)}
              </div>
            )}
            {subtopic && (
              <div className="autoresearch__thinking-subtopic">Diving deeper: {subtopic}</div>
            )}
          </span>
        </div>
      );
    }

    case "subtopic_picked":
      return (
        <div className="autoresearch__event autoresearch__event--subtopic">
          <span className="autoresearch__event-icon">&#x1F52C;</span>
          <span className="autoresearch__event-text">
            <strong>Exploring subtopic:</strong> {data.subtopic as string}
            <span className="autoresearch__driven-by"> (driven by {data.driven_by as string})</span>
          </span>
        </div>
      );

    case "conclusion_formed":
      return (
        <div className="autoresearch__event autoresearch__event--conclusion">
          <span className="autoresearch__event-icon">&#x1F4CA;</span>
          <span className="autoresearch__event-text">
            <strong>Conclusion ({data.emotion as string}):</strong>
            <div className="autoresearch__conclusion-text">{data.conclusion as string}</div>
            {data.emotional_bias && (
              <div className="autoresearch__conclusion-bias">Bias: {data.emotional_bias as string}</div>
            )}
          </span>
        </div>
      );

    case "topic_completed":
      return (
        <div className="autoresearch__event autoresearch__event--complete">
          <span className="autoresearch__event-icon">&#x2705;</span>
          <span className="autoresearch__event-text">
            Completed: {data.topic as string} ({data.findings_count as number} findings)
          </span>
        </div>
      );

    case "stopped":
      return (
        <div className="autoresearch__event autoresearch__event--stopped">
          <span className="autoresearch__event-icon">&#x23F9;</span>
          <span className="autoresearch__event-text">
            Research stopped. {data.topics_total as number} topics, {data.findings_total as number} findings, {data.conclusions_total as number} conclusions.
          </span>
        </div>
      );

    case "error":
      return (
        <div className="autoresearch__event autoresearch__event--error">
          <span className="autoresearch__event-icon">&#x26A0;</span>
          <span className="autoresearch__event-text">{data.message as string}</span>
        </div>
      );

    default:
      return null;
  }
}
