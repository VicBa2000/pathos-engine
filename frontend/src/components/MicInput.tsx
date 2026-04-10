import { useCallback, useRef, useState } from "react";
import "./MicInput.css";

interface Props {
  enabled: boolean;
  onRecorded: (blob: Blob) => void;
  transcribing: boolean;
  /** Pre-acquired stream from MicConfigPanel — no getUserMedia delay */
  stream: MediaStream | null;
}

/**
 * Simple mic record button for chat.
 * Uses a pre-acquired MediaStream (from MicConfigPanel) for instant recording.
 * Click = start recording. Click again = stop → send to ASR.
 */
export function MicInput({ enabled, onRecorded, transcribing, stream }: Props) {
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const mimeRef = useRef("");

  const startRecording = useCallback(() => {
    if (recording || transcribing || !stream) return;

    // Detect supported mime once
    if (!mimeRef.current) {
      for (const t of ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4", ""]) {
        if (!t || MediaRecorder.isTypeSupported(t)) { mimeRef.current = t; break; }
      }
    }

    // Check stream is still active
    if (stream.getTracks().every(t => t.readyState === "ended")) {
      console.warn("[MicInput] Stream tracks ended — need to re-test mic");
      return;
    }

    const options: MediaRecorderOptions = {};
    if (mimeRef.current) options.mimeType = mimeRef.current;

    const recorder = new MediaRecorder(stream, options);
    chunksRef.current = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      const totalSize = chunksRef.current.reduce((s, c) => s + c.size, 0);
      if (totalSize > 200) { // >200 bytes = has actual audio, not just header
        const blob = new Blob(chunksRef.current, { type: mimeRef.current || "audio/webm" });
        console.log(`[MicInput] Sending ${blob.size} bytes for transcription`);
        onRecorded(blob);
      } else {
        console.warn(`[MicInput] Recording too short (${totalSize} bytes), discarding`);
      }
    };

    recorderRef.current = recorder;
    recorder.start();
    setRecording(true);
    console.log(`[MicInput] Recording started (mime=${recorder.mimeType})`);
  }, [recording, transcribing, stream, onRecorded]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
      recorderRef.current = null;
      setRecording(false);
      console.log("[MicInput] Stopped");
    }
  }, []);

  if (!enabled) return null;

  return (
    <button
      type="button"
      className={`mic-btn ${recording ? "mic-btn--recording" : ""} ${transcribing ? "mic-btn--transcribing" : ""}`}
      onClick={recording ? stopRecording : startRecording}
      disabled={transcribing || !stream}
      title={
        !stream ? "Test mic first (click mic icon in header)" :
        transcribing ? "Transcribing..." :
        recording ? "Stop recording" : "Record"
      }
    >
      {transcribing ? (
        <div className="mic-btn__spinner" />
      ) : recording ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      )}
    </button>
  );
}
