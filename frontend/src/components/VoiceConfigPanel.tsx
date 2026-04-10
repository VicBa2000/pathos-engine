import { useState, useEffect, useRef } from "react";
import * as api from "../api/client";
import "./VoiceConfigPanel.css";

interface VoiceInfo {
  key: string;
  language: string;
  name: string;
  gender: string;
}

interface Props {
  sessionId: string;
  visible: boolean;
  onClose: () => void;
}

const LANG_LABELS: Record<string, string> = {
  en: "English",
  es: "Spanish",
  de: "German",
  fr: "French",
  it: "Italian",
  ja: "Japanese",
  ko: "Korean",
  nl: "Dutch",
  pl: "Polish",
  pt: "Portuguese",
  hi: "Hindi",
};

export function VoiceConfigPanel({ sessionId, visible, onClose }: Props) {
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [defaults, setDefaults] = useState<Record<string, string>>({});
  const [selectedVoice, setSelectedVoice] = useState("en-Carter_man");
  const [selectedLang, setSelectedLang] = useState("en");
  const [ttsBackend, setTtsBackend] = useState<"kokoro" | "parler">("kokoro");
  const [parlerAvailable, setParlerAvailable] = useState(true);
  const [parlerInitialized, setParlerInitialized] = useState(false);
  const [parlerLoading, setParlerLoading] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (visible && voices.length === 0) {
      api.listVoices().then((data: Record<string, unknown>) => {
        setVoices(data.voices as VoiceInfo[]);
        setDefaults(data.default_by_language as Record<string, string>);
        if (typeof data.parler_available === "boolean") setParlerAvailable(data.parler_available);
        if (typeof data.parler_initialized === "boolean") setParlerInitialized(data.parler_initialized);
      }).catch(() => {});
    }
  }, [visible, voices.length]);

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    }
    if (visible) {
      setTimeout(() => document.addEventListener("mousedown", handleClick), 0);
    }
    return () => document.removeEventListener("mousedown", handleClick);
  }, [visible, onClose]);

  if (!visible) return null;

  const languages = [...new Set(voices.map(v => v.language))];
  const filteredVoices = voices.filter(v => v.language === selectedLang);

  function handleVoiceChange(key: string) {
    setSelectedVoice(key);
    api.configureVoice(sessionId, { voice: key }).catch(console.error);
  }

  function handleLangChange(lang: string) {
    if (ttsBackend === "parler" && lang !== "en") return; // Parler: English only
    setSelectedLang(lang);
    const defaultVoice = defaults[lang];
    if (defaultVoice) {
      setSelectedVoice(defaultVoice);
      api.configureVoice(sessionId, { voice: defaultVoice, language: lang }).catch(console.error);
    }
  }

  function handleBackendChange(backend: "kokoro" | "parler") {
    setTtsBackend(backend);
    if (backend === "parler") {
      setParlerLoading(true);
    }
    api.configureVoice(sessionId, { tts_backend: backend })
      .then((res: Record<string, unknown>) => {
        if (typeof res.parler_available === "boolean") setParlerAvailable(res.parler_available);
        if (typeof res.parler_initialized === "boolean") setParlerInitialized(res.parler_initialized);
        setParlerLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setParlerLoading(false);
      });
    // Parler: lock to English
    if (backend === "parler" && selectedLang !== "en") {
      setSelectedLang("en");
      const defaultVoice = defaults["en"];
      if (defaultVoice) {
        setSelectedVoice(defaultVoice);
      }
    }
  }

  const isParler = ttsBackend === "parler";

  return (
    <div className="voice-config" ref={panelRef}>
      <div className="voice-config__header">
        <span className="voice-config__title">Voice</span>
        <button className="voice-config__close" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
        </button>
      </div>

      {/* TTS Backend selector */}
      <div className="voice-config__section">
        <label className="voice-config__label">TTS Engine</label>
        <div className="voice-config__backend-row">
          <button
            className={`voice-config__backend ${!isParler ? "voice-config__backend--active" : ""}`}
            onClick={() => handleBackendChange("kokoro")}
          >
            <span className="voice-config__backend-name">Kokoro</span>
            <span className="voice-config__backend-desc">Fast, multi-language</span>
          </button>
          <button
            className={`voice-config__backend ${isParler ? "voice-config__backend--active voice-config__backend--parler" : ""}`}
            onClick={() => handleBackendChange("parler")}
          >
            <span className="voice-config__backend-name">Parler</span>
            <span className="voice-config__backend-desc">Expressive, EN only</span>
          </button>
        </div>
        {isParler && (
          <div className="voice-config__parler-note">
            {parlerLoading ? (
              "Loading Parler-TTS model (~2.5GB)..."
            ) : !parlerAvailable ? (
              <>
                <strong>Parler-TTS not installed.</strong> Install with: <code>pip install parler-tts</code>
                <br />Falling back to Kokoro with enriched post-processing.
              </>
            ) : (
              <>
                {parlerInitialized ? (
                  <span className="voice-config__parler-ready">Parler-TTS active</span>
                ) : (
                  "Parler-TTS will load on first speech generation (~2.5GB VRAM)"
                )}
                <br />Full emotional prosody: emotion, intensity, body state, breathing, mood.
              </>
            )}
          </div>
        )}
      </div>

      <div className="voice-config__section">
        <label className="voice-config__label">Language</label>
        <div className="voice-config__lang-grid">
          {languages.map(lang => (
            <button
              key={lang}
              className={`voice-config__lang ${selectedLang === lang ? "voice-config__lang--active" : ""} ${isParler && lang !== "en" ? "voice-config__lang--disabled" : ""}`}
              onClick={() => handleLangChange(lang)}
              disabled={isParler && lang !== "en"}
            >
              {LANG_LABELS[lang] ?? lang}
            </button>
          ))}
        </div>
      </div>

      <div className="voice-config__section">
        <label className="voice-config__label">
          {isParler ? "Base Voice (timbre)" : "Voice"}
        </label>
        {isParler && (
          <div className="voice-config__voice-hint">
            Parler controls emotion/prosody. This selects the base voice timbre.
          </div>
        )}
        <div className="voice-config__voice-list">
          {filteredVoices.map(v => (
            <button
              key={v.key}
              className={`voice-config__voice ${selectedVoice === v.key ? "voice-config__voice--active" : ""}`}
              onClick={() => handleVoiceChange(v.key)}
            >
              <span className="voice-config__voice-name">{v.name}</span>
              <span className="voice-config__voice-gender">{v.gender === "man" ? "M" : "F"}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="voice-config__section">
        <label className="voice-config__label">
          Emotion-driven parameters
          <span className="voice-config__auto-badge">AUTO</span>
        </label>
        <div className="voice-config__info">
          {isParler
            ? "Parler generates prosody from rich emotional descriptions: emotion, intensity, body state, breathing, and mood are all encoded in the voice."
            : "Speed, voice blend, pitch, volume, and tremolo are automatically controlled by the agent's emotional state."
          }
        </div>
      </div>
    </div>
  );
}
