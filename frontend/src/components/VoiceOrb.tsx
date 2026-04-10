import { useEffect, useRef, useCallback } from "react";
import { hexToRgba, lighten, darken } from "../lib/colorUtils";
import "./VoiceOrb.css";

interface Props {
  /** Emotion color (hex) */
  color: string;
  /** Is audio currently playing? */
  playing: boolean;
  /** Toggle play/pause */
  onToggle: () => void;
  /** Web Audio AnalyserNode — connect to get real audio data */
  analyser?: AnalyserNode | null;
}

/**
 * Siri-style animated voice orb that reacts to real audio.
 *
 * - Idle: gentle breathing sphere with soft glow
 * - Playing: sphere pulses with volume, frequency bars dance around it
 * - Color matches agent's current emotion
 *
 * Uses Web Audio API AnalyserNode for real-time audio visualization.
 */
export function VoiceOrb({ color, playing, onToggle, analyser }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);
  const smoothVolumeRef = useRef(0);
  const freqDataRef = useRef<Uint8Array | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H / 2;
    const t = timeRef.current;

    // --- Read audio data ---
    let volume = 0;
    let freqBins: Uint8Array | null = null;
    if (analyser && playing) {
      if (!freqDataRef.current || freqDataRef.current.length !== analyser.frequencyBinCount) {
        freqDataRef.current = new Uint8Array(analyser.frequencyBinCount);
      }
      freqBins = freqDataRef.current;
      analyser.getByteFrequencyData(freqBins);
      // Compute RMS volume from frequency data
      let sum = 0;
      for (let i = 0; i < freqBins.length; i++) sum += freqBins[i];
      volume = sum / (freqBins.length * 255);
    }

    // Smooth volume for organic feel
    const targetVol = playing ? (analyser ? volume : 0.3 + Math.sin(t * 3) * 0.15) : 0;
    smoothVolumeRef.current += (targetVol - smoothVolumeRef.current) * 0.15;
    const sv = smoothVolumeRef.current;

    const baseRadius = Math.min(W, H) * 0.18;

    if (playing) {
      // === PLAYING: Reactive sphere + frequency bars ===

      // Outer glow (pulsing with volume)
      const glowRadius = baseRadius * (1.8 + sv * 2.5);
      const glow = ctx.createRadialGradient(cx, cy, baseRadius * 0.5, cx, cy, glowRadius);
      glow.addColorStop(0, hexToRgba(color, 0.15 + sv * 0.15));
      glow.addColorStop(0.5, hexToRgba(color, 0.05 + sv * 0.05));
      glow.addColorStop(1, hexToRgba(color, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2);
      ctx.fill();

      // Frequency bars in a circle around the orb
      const barCount = 48;
      const barMinR = baseRadius * (1.15 + sv * 0.3);
      const barMaxH = baseRadius * (0.8 + sv * 1.2);

      for (let i = 0; i < barCount; i++) {
        const angle = (i / barCount) * Math.PI * 2 - Math.PI / 2;

        // Map frequency bin to bar
        let barH: number;
        if (freqBins) {
          const binIdx = Math.floor((i / barCount) * freqBins.length * 0.6); // use lower 60% of spectrum
          barH = (freqBins[binIdx] / 255) * barMaxH;
        } else {
          // Fallback: synthetic waveform
          const w1 = Math.sin(t * 3 + i * 0.3) * 0.4;
          const w2 = Math.sin(t * 5 + i * 0.7) * 0.3;
          const w3 = Math.sin(t * 1.8 + i * 0.15) * 0.3;
          barH = Math.abs(w1 + w2 + w3) * barMaxH;
        }

        barH = Math.max(barH, 1.5);

        const x1 = cx + Math.cos(angle) * barMinR;
        const y1 = cy + Math.sin(angle) * barMinR;
        const x2 = cx + Math.cos(angle) * (barMinR + barH);
        const y2 = cy + Math.sin(angle) * (barMinR + barH);

        const alpha = 0.3 + (barH / barMaxH) * 0.7;
        ctx.strokeStyle = hexToRgba(color, alpha);
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      // Main sphere (morphing blob)
      const wobbleAmount = sv * 4;
      ctx.beginPath();
      const points = 64;
      for (let i = 0; i <= points; i++) {
        const angle = (i / points) * Math.PI * 2;
        const wobble =
          Math.sin(angle * 3 + t * 4) * wobbleAmount +
          Math.sin(angle * 5 + t * 2.5) * wobbleAmount * 0.5 +
          Math.sin(angle * 7 + t * 6) * wobbleAmount * 0.25;
        const r = baseRadius * (1 + sv * 0.25) + wobble;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();

      // Gradient fill
      const sphereGrad = ctx.createRadialGradient(
        cx - baseRadius * 0.2, cy - baseRadius * 0.2, 0,
        cx, cy, baseRadius * (1.3 + sv * 0.3),
      );
      sphereGrad.addColorStop(0, hexToRgba(lighten(color, 40), 0.9));
      sphereGrad.addColorStop(0.4, hexToRgba(color, 0.8));
      sphereGrad.addColorStop(1, hexToRgba(darken(color, 30), 0.6));
      ctx.fillStyle = sphereGrad;
      ctx.fill();

      // Inner highlight
      const hlGrad = ctx.createRadialGradient(
        cx - baseRadius * 0.25, cy - baseRadius * 0.3, 0,
        cx, cy, baseRadius * 0.7,
      );
      hlGrad.addColorStop(0, "rgba(255,255,255,0.3)");
      hlGrad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = hlGrad;
      ctx.fill();

      timeRef.current += 0.03;
    } else {
      // === IDLE: Gentle breathing sphere ===
      const breath = Math.sin(t * 1.5) * 0.08;
      const r = baseRadius * (0.7 + breath);

      // Soft glow
      const idleGlow = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 3);
      idleGlow.addColorStop(0, hexToRgba(color, 0.08));
      idleGlow.addColorStop(1, hexToRgba(color, 0));
      ctx.fillStyle = idleGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 3, 0, Math.PI * 2);
      ctx.fill();

      // Outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = hexToRgba(color, 0.15 + breath * 0.5);
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Main sphere
      const idleGrad = ctx.createRadialGradient(
        cx - r * 0.2, cy - r * 0.2, 0,
        cx, cy, r,
      );
      idleGrad.addColorStop(0, hexToRgba(lighten(color, 30), 0.6));
      idleGrad.addColorStop(0.6, hexToRgba(color, 0.4));
      idleGrad.addColorStop(1, hexToRgba(darken(color, 20), 0.3));
      ctx.fillStyle = idleGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      // Play icon hint
      ctx.fillStyle = hexToRgba("#ffffff", 0.5);
      ctx.beginPath();
      const triR = r * 0.35;
      ctx.moveTo(cx + triR, cy);
      ctx.lineTo(cx - triR * 0.5, cy - triR * 0.8);
      ctx.lineTo(cx - triR * 0.5, cy + triR * 0.8);
      ctx.closePath();
      ctx.fill();

      timeRef.current += 0.02;
    }

    animRef.current = requestAnimationFrame(draw);
  }, [color, playing, analyser]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  return (
    <button
      className={`voice-orb ${playing ? "voice-orb--playing" : ""}`}
      onClick={onToggle}
      title={playing ? "Pause" : "Play"}
    >
      <canvas ref={canvasRef} className="voice-orb__canvas" />
    </button>
  );
}

