import { useState } from "react";
import { ModelManagerPanel } from "./ModelManagerPanel";
import "./ModelSelector.css";

interface Props {
  currentModel: string;
  sessionId: string;
  onModelChanged: (provider: string, model: string) => void;
  localOnly?: boolean;
  locked?: boolean;
}

export function ModelSelector({ currentModel, sessionId, onModelChanged, localOnly, locked }: Props) {
  const [open, setOpen] = useState(false);

  const modelName = currentModel ?? "";
  const shortName = modelName.length > 20 ? modelName.slice(0, 20) + "..." : modelName;

  return (
    <div className="model-selector">
      <button
        className={`model-selector__trigger ${locked ? "model-selector__trigger--locked" : ""}`}
        onClick={() => !locked && setOpen(!open)}
        title={locked ? "Model locked by Auto-Research" : undefined}
      >
        <span className="model-selector__icon">&#x2699;</span>
        <span className="model-selector__current">{shortName || "No model"}</span>
        <span className="model-selector__caret">{open ? "\u25B4" : "\u25BE"}</span>
      </button>

      <ModelManagerPanel
        visible={open}
        onClose={() => setOpen(false)}
        currentModel={currentModel}
        sessionId={sessionId}
        onModelChanged={(provider, model) => {
          onModelChanged(provider, model);
        }}
        localOnly={localOnly}
      />
    </div>
  );
}
