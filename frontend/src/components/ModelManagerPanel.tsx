import { useState, useEffect, useRef, useCallback } from "react";
import * as api from "../api/client";
import type { ModelInfo, FeaturedModel, SearchResult, DownloadStatus, HuggingFaceCheck, CloudPreset, CloudProviderInfo, CloudTestResult, ArkSwitchInfo } from "../api/client";
import "./ModelManagerPanel.css";

type Tab = "local" | "featured" | "search" | "cloud" | "hf";

interface Props {
  visible: boolean;
  onClose: () => void;
  currentModel: string;
  sessionId: string;
  onModelChanged: (provider: string, model: string) => void;
  localOnly?: boolean;
}

export function ModelManagerPanel({ visible, onClose, currentModel, sessionId, onModelChanged, localOnly }: Props) {
  const [tab, setTab] = useState<Tab>("local");
  const panelRef = useRef<HTMLDivElement>(null);
  const [arkNotice, setArkNotice] = useState<ArkSwitchInfo | null>(null);
  const [activeProvider, setActiveProvider] = useState<string>("ollama");

  // --- Shared state ---
  const [downloads, setDownloads] = useState<DownloadStatus[]>([]);
  const [localModels, setLocalModels] = useState<ModelInfo[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);

  // Close on outside click
  useEffect(() => {
    if (!visible) return;
    function handleClick(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [visible, onClose]);

  // Load local models
  useEffect(() => {
    if (!visible) return;
    api.listModels().then(setLocalModels).catch(() => {});
  }, [visible, refreshKey]);

  // Poll downloads while panel is open and there are active downloads
  const hasActiveDownloads = downloads.some(d => d.status === "downloading");
  useEffect(() => {
    if (!visible) return;
    // Initial fetch
    api.getDownloads().then(d => setDownloads(d.downloads)).catch(() => {});
    if (!hasActiveDownloads) return;
    const poll = setInterval(() => {
      api.getDownloads().then(d => setDownloads(d.downloads)).catch(() => {});
    }, 2000);
    return () => clearInterval(poll);
  }, [visible, hasActiveDownloads]);

  const refreshLocal = useCallback(() => setRefreshKey(k => k + 1), []);

  const handlePull = useCallback((name: string) => {
    // Optimistic: add to downloads immediately
    setDownloads(prev => [...prev.filter(d => d.name !== name), { name, status: "downloading", percent: 0, completed: 0, total: 0, error: "" }]);
    api.pullModel(name, (status) => {
      setDownloads(prev => prev.map(d => d.name === name ? { ...d, ...status } : d));
    }).then(() => {
      refreshLocal();
      // Remove completed after 3s
      setTimeout(() => setDownloads(prev => prev.filter(d => d.name !== name)), 3000);
      // Auto-extract steering vectors in background (no-op if incompatible)
      api.extractSteeringVectors(name).then(() => refreshLocal()).catch(() => {});
    }).catch(() => {});
  }, [refreshLocal]);

  const handleSwitch = useCallback(async (provider: string, model: string) => {
    try {
      const result = await api.switchModel(provider, model, sessionId);
      onModelChanged(provider, model);
      setActiveProvider(provider);
      if (result.ark) {
        setArkNotice(result.ark);
        setTimeout(() => setArkNotice(null), 8000);
      }
    } catch (err) {
      console.error("Switch failed:", err);
      const msg = err instanceof Error ? err.message : "Switch failed";
      setArkNotice({ direct_available: false, vectors_ready: false, adapter_loaded: false, message: msg });
      setTimeout(() => setArkNotice(null), 8000);
    }
  }, [onModelChanged, sessionId]);

  if (!visible) return null;

  const ollamaModels = localModels.filter(m => m.provider === "ollama");
  const localNames = new Set(ollamaModels.map(m => m.name));

  return (
    <div className="mm-panel" ref={panelRef}>
      <div className="mm-panel__tabs">
        {(localOnly
          ? [["local", "Local"]] as [Tab, string][]
          : [["local", "Local"], ["featured", "Featured"], ["search", "Search"], ["cloud", "Cloud"], ["hf", "HuggingFace"]] as [Tab, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            className={`mm-panel__tab ${tab === key ? "mm-panel__tab--active" : ""}`}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {arkNotice && (
        <div className={`mm-ark-notice ${arkNotice.direct_available ? "mm-ark-notice--direct" : !arkNotice.vectors_ready && !arkNotice.adapter_loaded && !arkNotice.direct_available ? "mm-ark-notice--error" : "mm-ark-notice--injection"}`}>
          <span className="mm-ark-notice__icon">{arkNotice.direct_available ? "\u{1f9f2}" : !arkNotice.vectors_ready && !arkNotice.adapter_loaded && !arkNotice.direct_available ? "\u26a0\ufe0f" : "\u{1f4dd}"}</span>
          <span className="mm-ark-notice__text">{arkNotice.message}</span>
          <button className="mm-ark-notice__close" onClick={() => setArkNotice(null)}>&times;</button>
        </div>
      )}

      <div className="mm-panel__content">
        {tab === "local" && (
          <LocalTab
            models={ollamaModels}
            currentModel={currentModel}
            activeProvider={activeProvider}
            onSwitch={handleSwitch}
            onRefresh={refreshLocal}
            onDelete={(name) => {
              if (confirm(`Delete model "${name}"?`)) {
                api.deleteModel(name).then(refreshLocal).catch(console.error);
              }
            }}
          />
        )}
        {tab === "featured" && (
          <FeaturedTab localNames={localNames} onPull={handlePull} onSwitch={handleSwitch} currentModel={currentModel} />
        )}
        {tab === "search" && (
          <SearchTab localNames={localNames} onPull={handlePull} />
        )}
        {tab === "cloud" && (
          <CloudTab sessionId={sessionId} currentModel={currentModel} onSwitch={handleSwitch} />
        )}
        {tab === "hf" && (
          <HuggingFaceTab onPull={handlePull} />
        )}
      </div>

      {downloads.length > 0 && (
        <div className="mm-panel__downloads">
          <div className="mm-panel__downloads-title">Downloads</div>
          {downloads.map(d => (
            <div key={d.name} className="mm-dl">
              <span className="mm-dl__name">{d.name}</span>
              <div className="mm-dl__bar">
                <div className="mm-dl__fill" style={{ width: `${d.percent}%` }} />
              </div>
              <span className="mm-dl__pct">
                {d.status === "success" ? "Done" : d.status === "error" ? "Error" : `${d.percent.toFixed(0)}%`}
              </span>
              {d.status === "downloading" && (
                <button className="mm-dl__cancel" onClick={() => api.cancelDownload(d.name).catch(() => {})}>✕</button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================
// Tab: Local Models
// ============================================================

function LocalTab({ models, currentModel, onSwitch, onRefresh, onDelete, activeProvider }: {
  models: ModelInfo[];
  currentModel: string;
  onSwitch: (p: string, m: string) => Promise<void> | void;
  onRefresh: () => void;
  onDelete: (name: string) => void;
  activeProvider: string;
}) {
  const [extracting, setExtracting] = useState<string | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);

  const handleToggleSteering = async (model: string, hasCached: boolean, currentlyDirect: boolean) => {
    if (currentlyDirect) {
      setSwitching(model);
      try { await onSwitch("ollama", model); } finally { setSwitching(null); }
      return;
    }
    if (hasCached) {
      setSwitching(model);
      try { await onSwitch("transformers", model); } finally { setSwitching(null); }
      return;
    }
    // Extract vectors first, then switch
    setExtracting(model);
    try {
      const result = await api.extractSteeringVectors(model);
      if (result.status === "done" || result.status === "already_cached") {
        onRefresh();
        setSwitching(model);
        setExtracting(null);
        try { await onSwitch("transformers", model); } finally { setSwitching(null); }
      }
    } catch (err) {
      console.error("Extraction failed:", err);
      setExtracting(null);
    }
  };

  return (
    <div className="mm-local">
      <div className="mm-local__header">
        <span className="mm-section-label">Installed Models</span>
        <button className="mm-btn-sm" onClick={onRefresh}>↻ Refresh</button>
      </div>
      {models.length === 0 && <div className="mm-empty">No local models. Is Ollama running?</div>}
      {models.map(m => {
        const isActive = m.name === currentModel;
        const isDirectActive = isActive && activeProvider === "transformers";
        const isExtracting = extracting === m.name;
        const isSwitching = switching === m.name;
        const isBusy = isExtracting || isSwitching;
        return (
          <div
            key={m.name}
            className={`mm-model-row ${isActive ? "mm-model-row--active" : ""}`}
            onClick={() => { if (!isBusy) onSwitch("ollama", m.name); }}
          >
            <span className="mm-model-row__check">{isActive ? "✓" : ""}</span>
            <span className="mm-model-row__name">{m.name}</span>
            <span className="mm-model-row__size">{m.size}</span>
            <div className="mm-model-row__actions">
              <div
                className={`mm-toggle ${isDirectActive ? "mm-toggle--on" : ""} ${!m.steering_compatible ? "mm-toggle--disabled" : ""} ${isBusy ? "mm-toggle--loading" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (m.steering_compatible && !isBusy && extracting === null && switching === null) {
                    handleToggleSteering(m.name, m.vectors_cached, isDirectActive);
                  }
                }}
                title={!m.steering_compatible
                  ? "Steering not available for this architecture"
                  : isExtracting
                  ? "Extracting steering vectors..."
                  : isSwitching
                  ? "Loading model..."
                  : isDirectActive
                  ? "Click to switch to Ollama (prompt injection)"
                  : "Click to enable Steering (direct LLM modification)"}
              >
                <span className="mm-toggle__label mm-toggle__label--left">Ollama</span>
                <div className="mm-toggle__track">
                  <div className={`mm-toggle__thumb ${isBusy ? "mm-toggle__thumb--loading" : ""}`} />
                </div>
                <span className={`mm-toggle__label mm-toggle__label--right ${isBusy ? "mm-toggle__label--loading" : ""}`}>
                  {isExtracting ? "Extracting..." : isSwitching ? "Loading..." : "Steering"}
                </span>
              </div>
              <button
                className="mm-model-row__delete"
                onClick={(e) => { e.stopPropagation(); onDelete(m.name); }}
                title="Delete model"
              >🗑</button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Tab: Featured Models
// ============================================================

function FeaturedTab({ localNames, onPull, onSwitch, currentModel }: {
  localNames: Set<string>;
  onPull: (name: string) => void;
  onSwitch: (p: string, m: string) => void;
  currentModel: string;
}) {
  const [featured, setFeatured] = useState<FeaturedModel[]>([]);
  useEffect(() => { api.getFeaturedModels().then(setFeatured).catch(() => {}); }, []);

  const categories = [...new Set(featured.map(m => m.category))];

  return (
    <div className="mm-featured">
      {categories.map(cat => (
        <div key={cat} className="mm-featured__group">
          <div className="mm-section-label">{cat}</div>
          {featured.filter(m => m.category === cat).map(m => {
            const installed = localNames.has(m.name);
            const vramHigh = m.vram_estimate.includes("6GB") || m.vram_estimate.includes("8GB");
            return (
              <div key={m.name} className="mm-feat-row">
                <div className="mm-feat-row__info">
                  <span className="mm-feat-row__name">{m.name}</span>
                  <span className="mm-feat-row__desc">{m.description}</span>
                </div>
                <div className="mm-feat-row__meta">
                  <span className="mm-feat-row__size">{m.size}</span>
                  <span className={`mm-feat-row__vram ${vramHigh ? "mm-feat-row__vram--warn" : ""}`}>
                    {vramHigh ? "⚠ " : ""}{m.vram_estimate}
                  </span>
                </div>
                <div className="mm-feat-row__action">
                  {installed ? (
                    <button
                      className={`mm-btn-sm mm-btn-sm--installed ${m.name === currentModel ? "mm-btn-sm--active" : ""}`}
                      onClick={() => onSwitch("ollama", m.name)}
                    >
                      {m.name === currentModel ? "Active" : "Use"}
                    </button>
                  ) : (
                    <button className="mm-btn-sm mm-btn-sm--download" onClick={() => onPull(m.name)}>
                      ↓ Pull
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// Tab: Search
// ============================================================

function SearchTab({ localNames, onPull }: {
  localNames: Set<string>;
  onPull: (name: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    if (!query.trim()) { setResults([]); return; }
    const timer = setTimeout(() => {
      setSearching(true);
      api.searchModels(query).then(setResults).catch(() => setResults([])).finally(() => setSearching(false));
    }, 500);
    return () => clearTimeout(timer);
  }, [query]);

  return (
    <div className="mm-search">
      <input
        className="mm-search__input"
        placeholder="Search Ollama library..."
        value={query}
        onChange={e => setQuery(e.target.value)}
        autoFocus
      />
      {searching && <div className="mm-empty">Searching...</div>}
      {!searching && query && results.length === 0 && <div className="mm-empty">No results</div>}
      {results.map(r => (
        <div key={r.name} className="mm-search-row">
          <div className="mm-search-row__info">
            <span className="mm-search-row__name">{r.name}</span>
            <span className="mm-search-row__desc">{r.description}</span>
          </div>
          {r.pulls && <span className="mm-search-row__pulls">{r.pulls} pulls</span>}
          {localNames.has(r.name) ? (
            <span className="mm-badge mm-badge--installed">Installed</span>
          ) : (
            <button className="mm-btn-sm mm-btn-sm--download" onClick={() => onPull(r.name)}>↓ Pull</button>
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// Tab: Cloud Providers
// ============================================================

function CloudTab({ sessionId, currentModel, onSwitch }: {
  sessionId: string;
  currentModel: string;
  onSwitch: (p: string, m: string) => void;
}) {
  const [presets, setPresets] = useState<Record<string, CloudPreset>>({});
  const [providers, setProviders] = useState<CloudProviderInfo[]>([]);
  const [adding, setAdding] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState("");
  const [keyInput, setKeyInput] = useState("");
  const [urlInput, setUrlInput] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [saving, setSaving] = useState(false);

  // Test connection state
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<CloudTestResult | null>(null);

  const refresh = useCallback(() => {
    api.listCloudProviders(sessionId).then(d => {
      setProviders(d.providers);
    }).catch((err) => {
      console.warn("listCloudProviders failed:", err);
    });
  }, [sessionId]);

  useEffect(() => {
    api.getCloudPresets().then(d => setPresets(d.presets)).catch(() => {});
  }, []);

  // Refresh providers on mount and whenever sessionId changes
  useEffect(() => {
    refresh();
  }, [refresh]);

  const resetForm = () => {
    setAdding(false);
    setKeyInput("");
    setUrlInput("");
    setSelectedModel("");
    setSelectedPreset("");
    setTestResult(null);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await api.testCloudProvider(selectedPreset, keyInput, urlInput);
      setTestResult(result);
      if (result.ok && result.models.length > 0) {
        // Auto-select default model if in list, otherwise first
        const preset = presets[selectedPreset];
        const defaultName = preset?.default_model || "";
        const found = result.models.find(m => m.name === defaultName);
        setSelectedModel(found ? found.name : result.models[0].name);
      }
    } catch (err) {
      setTestResult({ ok: false, error: err instanceof Error ? err.message : "Test failed", models: [] });
    } finally {
      setTesting(false);
    }
  };

  const handleAdd = async () => {
    if (!selectedPreset || !keyInput.trim() || !selectedModel) return;
    setSaving(true);
    try {
      await api.addCloudProvider(sessionId, selectedPreset, keyInput, urlInput, selectedModel);
      resetForm();
      refresh();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : "Failed"}`);
    } finally {
      setSaving(false);
    }
  };

  const handleRemove = async (id: string) => {
    if (!confirm(`Remove ${id} provider?`)) return;
    await api.removeCloudProvider(sessionId, id).catch(console.error);
    refresh();
  };

  const presetForAdding = selectedPreset ? presets[selectedPreset] : null;

  return (
    <div className="mm-cloud">
      {/* Configured providers */}
      <div className="mm-section-label">Configured Providers</div>
      {providers.length === 0 && !adding && (
        <div className="mm-empty">No cloud providers. Click + Add to configure one.</div>
      )}
      {providers.map(p => (
        <div key={p.id} className="mm-cloud__provider">
          <div className="mm-cloud__provider-header">
            <span className="mm-cloud__provider-label">{p.label}</span>
            <span className="mm-cloud__provider-key">{p.masked_key}</span>
            <button className="mm-cloud__remove" onClick={() => handleRemove(p.id)} title="Remove">✕</button>
          </div>
          <div
            className={`mm-model-row ${p.model === currentModel ? "mm-model-row--active" : ""}`}
            onClick={() => onSwitch(p.id, p.model)}
          >
            <span className="mm-model-row__check">{p.model === currentModel ? "✓" : ""}</span>
            <span className="mm-model-row__name">{p.model}</span>
            <span className="mm-model-row__size">Cloud</span>
          </div>
        </div>
      ))}

      {/* Add new provider */}
      {!adding ? (
        <button className="mm-btn-sm mm-cloud__add-btn" onClick={() => setAdding(true)}>+ Add Provider</button>
      ) : (
        <div className="mm-cloud__form">
          <div className="mm-section-label">Add Cloud Provider</div>

          {/* Preset selector */}
          <div className="mm-cloud__preset-grid">
            {Object.entries(presets).filter(([k]) => k !== "custom").map(([key, p]) => (
              <button
                key={key}
                className={`mm-cloud__preset ${key === selectedPreset ? "mm-cloud__preset--selected" : ""}`}
                onClick={() => {
                  setSelectedPreset(key);
                  setUrlInput(p.base_url);
                  setSelectedModel("");
                  setTestResult(null);
                }}
              >
                <span className="mm-cloud__preset-label">{p.label}</span>
                <span className="mm-cloud__preset-desc">{p.description}</span>
              </button>
            ))}
            <button
              className={`mm-cloud__preset ${"custom" === selectedPreset ? "mm-cloud__preset--selected" : ""}`}
              onClick={() => { setSelectedPreset("custom"); setUrlInput(""); setSelectedModel(""); setTestResult(null); }}
            >
              <span className="mm-cloud__preset-label">Custom</span>
              <span className="mm-cloud__preset-desc">Any OpenAI-compatible endpoint</span>
            </button>
          </div>

          {selectedPreset && (
            <>
              <input
                className="mm-cloud__input"
                type="password"
                placeholder={`API Key${presetForAdding?.key_prefix ? ` (${presetForAdding.key_prefix}...)` : ""}`}
                value={keyInput}
                onChange={e => { setKeyInput(e.target.value); setTestResult(null); }}
              />
              {selectedPreset === "custom" && (
                <input
                  className="mm-cloud__input"
                  placeholder="Base URL (https://api.example.com/v1)"
                  value={urlInput}
                  onChange={e => { setUrlInput(e.target.value); setTestResult(null); }}
                />
              )}

              {/* Test connection button */}
              {!testResult && (
                <button
                  className="mm-btn-sm mm-btn-sm--test"
                  onClick={handleTest}
                  disabled={testing || !keyInput.trim() || (selectedPreset === "custom" && !urlInput.trim())}
                >
                  {testing ? "Testing..." : "Test Connection"}
                </button>
              )}

              {/* Test error */}
              {testResult && !testResult.ok && (
                <div className="mm-cloud__error">
                  {testResult.error || "Connection failed"}
                  <button className="mm-btn-sm" onClick={() => setTestResult(null)} style={{ marginLeft: 8 }}>Retry</button>
                </div>
              )}

              {/* Model list from test */}
              {testResult?.ok && testResult.models.length > 0 && (
                <div className="mm-cloud__models">
                  <div className="mm-section-label">Select Model</div>
                  <div className="mm-cloud__model-list">
                    {testResult.models.map(m => (
                      <div
                        key={m.name}
                        className={`mm-model-row ${m.name === selectedModel ? "mm-model-row--active" : ""}`}
                        onClick={() => setSelectedModel(m.name)}
                      >
                        <span className="mm-model-row__check">{m.name === selectedModel ? "✓" : ""}</span>
                        <span className="mm-model-row__name">{m.name}</span>
                        <span className="mm-model-row__size">{m.size}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Save / Cancel */}
              <div className="mm-cloud__form-actions">
                {testResult?.ok && (
                  <button
                    className="mm-btn-sm mm-btn-sm--download"
                    onClick={handleAdd}
                    disabled={saving || !selectedModel}
                  >
                    {saving ? "Saving..." : "Save"}
                  </button>
                )}
                <button className="mm-btn-sm" onClick={resetForm}>Cancel</button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================
// Tab: HuggingFace
// ============================================================

function HuggingFaceTab({ onPull }: { onPull: (name: string) => void }) {
  const [repo, setRepo] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<HuggingFaceCheck | null>(null);
  const [selectedQuant, setSelectedQuant] = useState("");

  const handleCheck = async () => {
    setChecking(true);
    setResult(null);
    try {
      const r = await api.checkHuggingFace(repo);
      setResult(r);
      if (r.quantizations.length > 0) setSelectedQuant(r.quantizations[0]);
    } catch {
      setResult({ valid: false, error: "Network error", quantizations: [], files: [] });
    } finally {
      setChecking(false);
    }
  };

  const handlePull = () => {
    if (!result?.ollama_name) return;
    const name = selectedQuant ? `${result.ollama_name}:${selectedQuant}` : result.ollama_name;
    onPull(name);
  };

  return (
    <div className="mm-hf">
      <div className="mm-section-label">Import from HuggingFace</div>
      <div className="mm-hf__form">
        <input
          className="mm-hf__input"
          placeholder="user/model-GGUF"
          value={repo}
          onChange={e => setRepo(e.target.value)}
        />
        <button className="mm-btn-sm mm-btn-sm--download" onClick={handleCheck} disabled={checking || !repo.includes("/")}>
          {checking ? "..." : "Check"}
        </button>
      </div>

      {result && !result.valid && (
        <div className="mm-hf__error">{result.error}</div>
      )}

      {result?.valid && (
        <div className="mm-hf__result">
          <div className="mm-hf__info">
            ✓ {result.files.length} GGUF file{result.files.length !== 1 ? "s" : ""} found
          </div>
          {result.quantizations.length > 0 && (
            <div className="mm-hf__quants">
              <span className="mm-section-label">Quantization</span>
              <div className="mm-hf__quant-list">
                {result.quantizations.map(q => (
                  <button
                    key={q}
                    className={`mm-hf__quant ${q === selectedQuant ? "mm-hf__quant--selected" : ""}`}
                    onClick={() => setSelectedQuant(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          <button className="mm-btn-sm mm-btn-sm--download mm-hf__pull-btn" onClick={handlePull}>
            ↓ Pull {selectedQuant ? `(${selectedQuant})` : ""}
          </button>
        </div>
      )}
    </div>
  );
}
