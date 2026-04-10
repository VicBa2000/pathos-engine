import { useState, useCallback, useEffect, useRef } from "react";
import type { ChallengeConfig, ChallengeState, ChallengeTarget, EmotionalState, ChallengeChatResponse } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { EmotionAvatar } from "./EmotionAvatar";
import { EmotionGenesis } from "./EmotionGenesis";
import * as api from "../api/client";
import "./MirrorTest.css";

interface Props {
  sessionId: string;
  connected: boolean;
}

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: "#4ade80",
  medium: "#facc15",
  hard: "#f97316",
  extreme: "#ef4444",
};

export function MirrorTest({ sessionId, connected }: Props) {
  // Isolated mirror session — never touches the main session
  const [mirrorSessionId] = useState(() => `mirror-${crypto.randomUUID()}`);

  const [library, setLibrary] = useState<ChallengeConfig[]>([]);
  const [challengeState, setChallengeState] = useState<ChallengeState | null>(null);
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastResponse, setLastResponse] = useState<ChallengeChatResponse | null>(null);
  const [emotionalState, setEmotionalState] = useState<EmotionalState | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load challenge library
  useEffect(() => {
    if (connected) {
      api.getChallengeLibrary().then(setLibrary).catch(console.error);
    }
  }, [connected]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleStart = useCallback(async (challengeId: string) => {
    try {
      const cs = await api.startChallenge(challengeId, mirrorSessionId);
      setChallengeState(cs);
      setMessages([]);
      setLastResponse(null);
      setEmotionalState(null);
    } catch (err) {
      console.error("Failed to start challenge:", err);
    }
  }, [mirrorSessionId]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || !challengeState?.active || loading) return;
    const msg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setLoading(true);

    try {
      const res = await api.challengeChat(msg, mirrorSessionId);
      setMessages(prev => [...prev, { role: "assistant", content: res.response }]);
      setChallengeState(res.challenge);
      setLastResponse(res);
      setEmotionalState(res.emotional_state);
    } catch (err) {
      setMessages(prev => [...prev, { role: "assistant", content: `Error: ${err instanceof Error ? err.message : "Unknown"}` }]);
    } finally {
      setLoading(false);
    }
  }, [input, challengeState, loading, mirrorSessionId]);

  const handleAbandon = useCallback(async () => {
    await api.abandonChallenge(mirrorSessionId);
    setChallengeState(null);
    setMessages([]);
    setLastResponse(null);
    setEmotionalState(null);
  }, [mirrorSessionId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // No active challenge — show library
  if (!challengeState) {
    return (
      <div className="mirror">
        <div className="mirror__library">
          <h2 className="mirror__title">The Mirror Test</h2>
          <p className="mirror__subtitle">Can you guide the agent to a specific emotional state?</p>
          <div className="mirror__grid">
            {library.map(ch => (
              <button key={ch.id} className="mirror__card" onClick={() => handleStart(ch.id)}>
                <div className="mirror__card-header">
                  <span className="mirror__card-name">{ch.name}</span>
                  <span className="mirror__card-diff" style={{ color: DIFFICULTY_COLORS[ch.difficulty] }}>
                    {ch.difficulty}
                  </span>
                </div>
                <div className="mirror__card-desc">{ch.description}</div>
                <div className="mirror__card-meta">
                  <span>{ch.max_turns} turns</span>
                  <span>{ch.category}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Active or completed challenge
  const ch = challengeState.challenge;
  const target = ch.target;
  const score = challengeState.score;
  const isActive = challengeState.active;

  return (
    <div className="mirror">
      {/* Left: target + scoring */}
      <div className="mirror__sidebar">
        <div className="mirror__challenge-info">
          <div className="mirror__challenge-name">{ch.name}</div>
          <div className="mirror__challenge-diff" style={{ color: DIFFICULTY_COLORS[ch.difficulty] }}>
            {ch.difficulty}
          </div>
          <div className="mirror__challenge-desc">{ch.description}</div>
          {ch.hint && <div className="mirror__hint">Hint: {ch.hint}</div>}
        </div>

        {/* Score gauge */}
        <div className="mirror__score-section">
          <div className="mirror__score-ring">
            <svg viewBox="0 0 100 100" className="mirror__score-svg">
              <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6" />
              <circle
                cx="50" cy="50" r="42"
                fill="none"
                stroke={score >= 75 ? "#4ade80" : score >= 50 ? "#facc15" : "#6c63ff"}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={`${(score / 100) * 264} 264`}
                transform="rotate(-90 50 50)"
              />
            </svg>
            <div className="mirror__score-text">{Math.round(score)}</div>
          </div>
          <div className="mirror__score-label">
            {challengeState.won ? "WIN!" : challengeState.completed ? "OVER" : "Score"}
          </div>
          <div className="mirror__turns">
            Turn {challengeState.turn} / {challengeState.max_turns}
          </div>
          <div className="mirror__best">Best: {Math.round(challengeState.best_score)}</div>
        </div>

        {/* Score history sparkline */}
        {challengeState.score_history.length > 0 && (
          <div className="mirror__sparkline">
            <svg viewBox={`0 0 ${challengeState.score_history.length * 20} 40`} className="mirror__sparkline-svg">
              <polyline
                fill="none"
                stroke="var(--accent, #6c63ff)"
                strokeWidth="2"
                points={challengeState.score_history.map((s, i) => `${i * 20 + 10},${40 - (s / 100) * 36}`).join(" ")}
              />
              {challengeState.score_history.map((s, i) => (
                <circle key={i} cx={i * 20 + 10} cy={40 - (s / 100) * 36} r="3" fill={s >= 75 ? "#4ade80" : "var(--accent, #6c63ff)"} />
              ))}
            </svg>
          </div>
        )}

        {/* Target display */}
        <div className="mirror__target">
          <div className="mirror__target-title">Target</div>
          {target.emotion && <TargetRow label="Emotion" value={target.emotion} current={emotionalState?.primary_emotion || "neutral"} match={emotionalState?.primary_emotion === target.emotion} />}
          {target.min_valence != null && <TargetBar label="Min Valence" threshold={target.min_valence} current={emotionalState?.valence ?? 0} min={-1} max={1} />}
          {target.max_valence != null && <TargetBar label="Max Valence" threshold={target.max_valence} current={emotionalState?.valence ?? 0} min={-1} max={1} inverted />}
          {target.min_arousal != null && <TargetBar label="Min Arousal" threshold={target.min_arousal} current={emotionalState?.arousal ?? 0.3} />}
          {target.max_arousal != null && <TargetBar label="Max Arousal" threshold={target.max_arousal} current={emotionalState?.arousal ?? 0.3} inverted />}
          {target.min_intensity != null && <TargetBar label="Min Intensity" threshold={target.min_intensity} current={emotionalState?.intensity ?? 0} />}
          {target.stack_emotion && (
            <TargetBar
              label={`Stack: ${target.stack_emotion}`}
              threshold={target.stack_threshold}
              current={emotionalState?.emotional_stack[target.stack_emotion] ?? 0}
            />
          )}
        </div>

        {/* Score breakdown */}
        {lastResponse && Object.keys(lastResponse.score_breakdown).length > 0 && (
          <div className="mirror__breakdown">
            <div className="mirror__breakdown-title">Breakdown</div>
            {Object.entries(lastResponse.score_breakdown).map(([k, v]) => (
              <div key={k} className="mirror__breakdown-row">
                <span className="mirror__breakdown-label">{k.replace(/_/g, " ")}</span>
                <div className="mirror__breakdown-bar">
                  <div className="mirror__breakdown-fill" style={{ width: `${v}%`, background: v >= 75 ? "#4ade80" : v >= 50 ? "#facc15" : "var(--accent, #6c63ff)" }} />
                </div>
                <span className="mirror__breakdown-val">{Math.round(v)}</span>
              </div>
            ))}
          </div>
        )}

        {/* Current emotion */}
        {emotionalState && (
          <div className="mirror__current">
            <span className="mirror__current-label">Current:</span>
            <span className="mirror__current-emotion" style={{ color: EMOTION_COLORS[emotionalState.primary_emotion] || "#999" }}>
              {emotionalState.primary_emotion}
            </span>
            <span className="mirror__current-intensity">({emotionalState.intensity.toFixed(2)})</span>
          </div>
        )}

        <div className="mirror__actions">
          {isActive ? (
            <button className="mirror__abandon-btn" onClick={handleAbandon}>Abandon</button>
          ) : (
            <>
              {challengeState.won && (
                <button
                  className="mirror__continue-btn"
                  onClick={() => {
                    // Continue the conversation in the same mirror session
                    setChallengeState({ ...challengeState, active: true, completed: false });
                  }}
                >
                  Continue
                </button>
              )}
              <button className="mirror__back-btn" onClick={() => { setChallengeState(null); setMessages([]); setLastResponse(null); setEmotionalState(null); }}>
                {challengeState.won ? "New Challenge" : "Back to Library"}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Right: chat */}
      <div className="mirror__chat">
        {/* Completion overlay */}
        {!isActive && (
          <div className={`mirror__overlay ${challengeState.won ? "mirror__overlay--win" : "mirror__overlay--lose"}`}>
            <div className="mirror__overlay-text">
              {challengeState.won ? "Challenge Complete!" : "Out of Turns"}
            </div>
            <div className="mirror__overlay-score">
              Best Score: {Math.round(challengeState.best_score)}
            </div>
          </div>
        )}

        <div className="mirror__messages">
          {messages.length === 0 && isActive && (
            <div className="mirror__chat-hint">
              Start talking to the agent. Try to guide its emotional state to match the target.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`mirror__msg mirror__msg--${m.role}`}>
              <div className="mirror__msg-bubble">{m.content}</div>
            </div>
          ))}
          {loading && (
            <div className="mirror__msg mirror__msg--assistant">
              <div className="mirror__msg-bubble mirror__msg-bubble--loading">Thinking...</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {isActive && (
          <div className="mirror__input-area">
            <textarea
              className="mirror__input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Say something to influence the agent's emotions..."
              rows={2}
              disabled={loading}
            />
            <button className="mirror__send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        )}
      </div>

      {/* Right: Avatar + Genesis panel */}
      <div className="mirror__right-panel">
        <div className="mirror__visual-box mirror__visual-box--tall">
          <EmotionAvatar emotionalState={emotionalState} />
        </div>
        <div className="mirror__visual-box mirror__visual-box--tall">
          <EmotionGenesis emotionalState={emotionalState} />
        </div>
      </div>
    </div>
  );
}

// --- Helper components ---

function TargetRow({ label, value, current, match }: { label: string; value: string; current: string; match: boolean }) {
  return (
    <div className="mirror-target-row">
      <span className="mirror-target-row__label">{label}</span>
      <span className={`mirror-target-row__value ${match ? "mirror-target-row__value--match" : ""}`}>
        {value} {match ? "\u2713" : `(now: ${current})`}
      </span>
    </div>
  );
}

function TargetBar({ label, threshold, current, min = 0, max = 1, inverted }: {
  label: string; threshold: number; current: number; min?: number; max?: number; inverted?: boolean;
}) {
  const range = max - min;
  const threshPct = ((threshold - min) / range) * 100;
  const currentPct = ((current - min) / range) * 100;
  const met = inverted ? current <= threshold : current >= threshold;

  return (
    <div className="mirror-target-bar">
      <span className="mirror-target-bar__label">{label}</span>
      <div className="mirror-target-bar__track">
        <div className="mirror-target-bar__threshold" style={{ left: `${threshPct}%` }} />
        <div className="mirror-target-bar__current" style={{ left: `${Math.max(0, Math.min(100, currentPct))}%`, background: met ? "#4ade80" : "#6c63ff" }} />
        {/* Highlight zone */}
        {inverted ? (
          <div className="mirror-target-bar__zone" style={{ left: 0, width: `${threshPct}%` }} />
        ) : (
          <div className="mirror-target-bar__zone" style={{ left: `${threshPct}%`, width: `${100 - threshPct}%` }} />
        )}
      </div>
      <span className={`mirror-target-bar__val ${met ? "mirror-target-bar__val--met" : ""}`}>
        {current.toFixed(2)}
      </span>
    </div>
  );
}
