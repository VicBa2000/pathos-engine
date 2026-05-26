import { useEffect, useState } from "react";
import { getResiduumStatus } from "../api/client";
import type { ResiduumStatus } from "../types/emotion";

/**
 * Global, always-visible indicator of RESIDUUM (Pillar 8) introspection state.
 * Polls /residuum/status so the user can tell at a glance whether the model is
 * actually being read at the residual-stream level — without opening Research
 * mode. ON only happens on the Transformers path with a probe library loaded;
 * on Ollama/cloud it shows OFF (architecture limitation).
 */
export function ResiduumStatusBadge({ sessionId }: { sessionId: string }) {
  const [st, setSt] = useState<ResiduumStatus | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = () => {
      getResiduumStatus(sessionId)
        .then((s) => { if (alive) setSt(s); })
        .catch(() => { /* backend not ready / no-op */ });
    };
    poll();
    const id = setInterval(poll, 8000);
    return () => { alive = false; clearInterval(id); };
  }, [sessionId]);

  if (!st) return null;

  const cfg = st.enabled
    ? {
        bg: "#234a23", dot: "#5fd35f", label: "Introspection ON",
        title: `RESIDUUM is reading the residual stream (layer ${st.library_layer}, ${st.library_num_probes} probes). The measured state is a functional readout, not evidence of subjective experience.`,
      }
    : st.ready_to_enable
    ? {
        bg: "#4a3f23", dot: "#e0b050", label: "Introspection ready",
        title: "Transformers path + probe library present. Auto-enables in Advanced mode; toggle in Research → Residuum.",
      }
    : {
        bg: "#2f2f2f", dot: "#777", label: "Introspection off",
        title: "Unavailable — " + (st.blockers && st.blockers.length
          ? st.blockers.join("; ")
          : "needs the local Transformers path + a probe library for this model (Ollama/cloud run with F2 off)."),
      };

  return (
    <div
      title={cfg.title}
      style={{
        position: "fixed", bottom: 10, left: 10, zIndex: 900,
        display: "inline-flex", alignItems: "center", gap: "0.35rem",
        fontSize: "0.66rem", padding: "0.2rem 0.55rem", borderRadius: "4px",
        background: cfg.bg, color: "#e8e8e8", border: "1px solid #0006", cursor: "help",
      }}
    >
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: cfg.dot }} />
      🧠 {cfg.label}
    </div>
  );
}
