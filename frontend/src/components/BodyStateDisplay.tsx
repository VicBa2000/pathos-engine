import type { BodyState } from "../types/emotion";
import "./BodyStateDisplay.css";

interface Props {
  bodyState: BodyState;
}

const BARS: Array<{ key: keyof BodyState; label: string; color: string }> = [
  { key: "energy", label: "Energy", color: "#f1c40f" },
  { key: "tension", label: "Tension", color: "#e74c3c" },
  { key: "openness", label: "Openness", color: "#2ecc71" },
  { key: "warmth", label: "Warmth", color: "#e67e22" },
];

export function BodyStateDisplay({ bodyState }: Props) {
  return (
    <div className="body-state">
      <h3>Body State</h3>
      <div className="body-state__bars">
        {BARS.map(({ key, label, color }) => {
          const value = bodyState[key];
          return (
            <div key={key} className="body-state__bar">
              <span className="body-state__label">{label}</span>
              <div className="body-state__track">
                <div
                  className="body-state__fill"
                  style={{ width: `${value * 100}%`, backgroundColor: color }}
                />
              </div>
              <span className="body-state__value">{value.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
