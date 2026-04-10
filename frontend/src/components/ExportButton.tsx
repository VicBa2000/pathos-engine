import { useState } from "react";
import * as api from "../api/client";
import "./ExportButton.css";

interface Props {
  sessionId: string;
  currentModel: string;
}

export function ExportButton({ sessionId, currentModel }: Props) {
  const [exporting, setExporting] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  async function handleExport() {
    setExporting(true);
    setResult(null);
    try {
      const baseModel = currentModel || "qwen3:4b";
      const res = await api.exportModel(sessionId, baseModel, "pathos");
      if (res.status === "model_created") {
        setResult(`Model 'pathos' created from ${baseModel}`);
      } else if (res.status === "modelfile_saved") {
        setResult(`Modelfile saved. Run: ollama create pathos -f Modelfile`);
      } else {
        setResult("Export failed");
      }
    } catch (err) {
      setResult(`Error: ${err instanceof Error ? err.message : "unknown"}`);
    } finally {
      setExporting(false);
      setTimeout(() => setResult(null), 6000);
    }
  }

  return (
    <div className="export-btn-wrap">
      <button
        className="export-btn"
        onClick={handleExport}
        disabled={exporting}
        title="Export emotional architecture as Ollama model"
      >
        {exporting ? "Exporting..." : "Export to Ollama"}
      </button>
      {result && <span className="export-btn__result">{result}</span>}
    </div>
  );
}
