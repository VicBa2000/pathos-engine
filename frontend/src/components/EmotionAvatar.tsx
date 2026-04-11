import { useState, Suspense } from "react";
import type { EmotionalState } from "../types/emotion";
import { PainterlyFace } from "./PainterlyFace";
import { RealisticFace } from "./RealisticFace";
import "./EmotionAvatar.css";

interface Props {
  emotionalState: EmotionalState | null;
  analyser?: AnalyserNode | null;
  speaking?: boolean;
}

export function EmotionAvatar({ emotionalState, analyser, speaking }: Props) {
  const [mode, setMode] = useState<"painterly" | "realistic">("painterly");

  return (
    <div className="emotion-avatar">
      <div className="emotion-avatar__toggle">
        <button
          onClick={() => setMode("painterly")}
          className={mode === "painterly" ? "active" : ""}
        >
          Painterly
        </button>
        <button
          onClick={() => setMode("realistic")}
          className={mode === "realistic" ? "active" : ""}
        >
          Realistic
        </button>
      </div>

      {mode === "painterly" ? (
        <PainterlyFace
          emotionalState={emotionalState}
          analyser={analyser}
          speaking={speaking}
        />
      ) : (
        <Suspense fallback={<div className="emotion-avatar__placeholder">Loading 3D...</div>}>
          <RealisticFace
            emotionalState={emotionalState}
            analyser={analyser}
            speaking={speaking}
          />
        </Suspense>
      )}

      {emotionalState && (
        <div className="emotion-avatar__label">
          {emotionalState.primary_emotion}
          {emotionalState.secondary_emotion &&
          emotionalState.secondary_emotion !== emotionalState.primary_emotion
            ? ` / ${emotionalState.secondary_emotion}`
            : ""}
        </div>
      )}
    </div>
  );
}
