import { useState, useEffect, useRef, useCallback } from "react";
import "./MicConfigPanel.css";

interface Props {
  visible: boolean;
  onClose: () => void;
  /** Called when mic is tested and working — unlocks mic button in chat */
  onMicReady: (ready: boolean) => void;
  /** Called with a live MediaStream when mic is confirmed — chat uses this to record instantly */
  onStreamReady: (stream: MediaStream | null) => void;
}

/**
 * Mic configuration panel — separate dropdown panel next to voice.
 * - Lists available audio input devices
 * - Real-time volume level test
 * - Warns if no mic or no sound detected
 * - Unlocks mic input in chat only when mic is confirmed working
 */
export function MicConfigPanel({ visible, onClose, onMicReady, onStreamReady }: Props) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [testing, setTesting] = useState(false);
  const [volume, setVolume] = useState(0);
  const [peakVolume, setPeakVolume] = useState(0);
  const [error, setError] = useState("");
  const [micConfirmed, setMicConfirmed] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const peakVolumeRef = useRef(0);

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

  // Enumerate devices when panel opens
  useEffect(() => {
    if (!visible) return;
    loadDevices();
  }, [visible]);

  // Stop test when panel closes
  useEffect(() => {
    if (!visible) stopTest();
  }, [visible]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(animRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  async function loadDevices() {
    try {
      // Need temporary permission to get labels
      const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      tempStream.getTracks().forEach(t => t.stop());

      const all = await navigator.mediaDevices.enumerateDevices();
      const inputs = all.filter(d => d.kind === "audioinput");
      setDevices(inputs);
      setError("");

      if (inputs.length === 0) {
        setError("No microphone detected");
        onMicReady(false);
        return;
      }

      if (!selectedDevice || !inputs.find(d => d.deviceId === selectedDevice)) {
        setSelectedDevice(inputs[0].deviceId);
      }
    } catch {
      setError("Microphone access denied. Check browser permissions.");
      onMicReady(false);
    }
  }

  // Volume meter loop
  const updateVolume = useCallback(() => {
    if (!analyserRef.current) return;
    const data = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.min(1, Math.sqrt(sum / data.length) * 3.5);
    setVolume(rms);
    peakVolumeRef.current = Math.max(peakVolumeRef.current, rms);
    setPeakVolume(peakVolumeRef.current);
    animRef.current = requestAnimationFrame(updateVolume);
  }, []);

  async function startTest() {
    setError("");
    setPeakVolume(0);
    peakVolumeRef.current = 0;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: selectedDevice ? { deviceId: { exact: selectedDevice } } : true,
      });
      streamRef.current = stream;

      if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
      const ctx = audioCtxRef.current;
      if (ctx.state === "suspended") await ctx.resume();

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      setTesting(true);
      animRef.current = requestAnimationFrame(updateVolume);

      // After 3 seconds, check if we detected any sound
      timeoutRef.current = setTimeout(() => {
        // Read peak from ref to avoid calling onMicReady inside setState
        const peak = peakVolumeRef.current;
        if (peak < 0.02) {
          setError("No sound detected. Check your microphone connection.");
          onMicReady(false);
          onStreamReady(null);
        } else {
          setMicConfirmed(true);
          onMicReady(true);
          // Keep stream alive and pass to parent for instant recording
          if (streamRef.current) {
            onStreamReady(streamRef.current);
          }
        }
      }, 3000);
    } catch {
      setError("Could not access microphone.");
      onMicReady(false);
    }
  }

  function stopTest() {
    cancelAnimationFrame(animRef.current);
    if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
    analyserRef.current = null;
    setTesting(false);
    setVolume(0);
    // Only kill stream if mic is NOT confirmed (keep alive for chat recording)
    if (!micConfirmed) {
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  }

  function handleDeviceChange(deviceId: string) {
    setSelectedDevice(deviceId);
    setMicConfirmed(false);
    onMicReady(false);
    onStreamReady(null);
    // Kill old stream
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (testing) {
      stopTest();
    }
  }

  if (!visible) return null;

  return (
    <div className="mic-config" ref={panelRef}>
      <div className="mic-config__header">
        <span className="mic-config__title">Microphone</span>
        <button className="mic-config__close" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Device selector */}
      <div className="mic-config__section">
        <label className="mic-config__label">Input Device</label>
        {devices.length === 0 && !error ? (
          <div className="mic-config__info">Detecting devices...</div>
        ) : (
          <div className="mic-config__device-list">
            {devices.map(d => (
              <button
                key={d.deviceId}
                className={`mic-config__device ${selectedDevice === d.deviceId ? "mic-config__device--active" : ""}`}
                onClick={() => handleDeviceChange(d.deviceId)}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                </svg>
                <span className="mic-config__device-name">
                  {d.label || `Microphone ${devices.indexOf(d) + 1}`}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Volume level test */}
      <div className="mic-config__section">
        <label className="mic-config__label">Level Test</label>
        <div className="mic-config__meter-wrap">
          <div className="mic-config__meter">
            <div
              className={`mic-config__meter-fill ${volume > 0.6 ? "mic-config__meter-fill--hot" : ""}`}
              style={{ width: `${volume * 100}%` }}
            />
          </div>
          <span className="mic-config__meter-value">{Math.round(volume * 100)}%</span>
        </div>
        {!testing ? (
          <button
            className="mic-config__test-btn"
            onClick={startTest}
            disabled={devices.length === 0}
          >
            Test Microphone
          </button>
        ) : (
          <button className="mic-config__test-btn mic-config__test-btn--stop" onClick={stopTest}>
            Stop Test
          </button>
        )}
      </div>

      {/* Status */}
      <div className="mic-config__section">
        {error ? (
          <div className="mic-config__error">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </div>
        ) : micConfirmed ? (
          <div className="mic-config__ok">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Microphone ready — use the mic button in chat to record
          </div>
        ) : testing ? (
          <div className="mic-config__info">Speak into your microphone...</div>
        ) : (
          <div className="mic-config__info">Test your microphone to enable voice input</div>
        )}
      </div>
    </div>
  );
}
