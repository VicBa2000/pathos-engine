import { useRef, useEffect, useCallback } from "react";
import type { EmotionalState, PrimaryEmotion } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { noise2D, fbm } from "../lib/perlin";
import { hexToRgb, lerpColor } from "../lib/colorUtils";
import "./EmotionGenesis.css";

interface Props {
  emotionalState: EmotionalState | null;
}

// --- Particle system ---

interface Particle {
  x: number;
  y: number;
  baseX: number;
  baseY: number;
  size: number;
  speed: number;
  phase: number;
  layer: number; // 0=deep, 1=mid, 2=surface
}

// Emotional color palette: primary + secondary blended
function getEmotionPalette(state: EmotionalState | null): {
  primary: [number, number, number];
  secondary: [number, number, number];
  ambient: [number, number, number];
} {
  if (!state) {
    return {
      primary: [108, 122, 137],   // neutral grey-blue
      secondary: [26, 188, 156],  // contemplation teal
      ambient: [20, 20, 30],
    };
  }

  const pHex = EMOTION_COLORS[state.primary_emotion] || EMOTION_COLORS.neutral;
  const sHex = state.secondary_emotion
    ? EMOTION_COLORS[state.secondary_emotion] || pHex
    : pHex;

  const primary = hexToRgb(pHex);
  const secondary = hexToRgb(sHex);

  // Ambient: dark desaturated version of primary, influenced by valence
  const valShift = (state.valence + 1) / 2; // 0..1
  const ambient: [number, number, number] = [
    Math.round(primary[0] * 0.08 + valShift * 8),
    Math.round(primary[1] * 0.08 + valShift * 5),
    Math.round(primary[2] * 0.12 + 15),
  ];

  return { primary, secondary, ambient };
}

// Get top N active emotions from the stack for multi-color plasma
function getActiveEmotions(state: EmotionalState | null, n: number): Array<{
  emotion: PrimaryEmotion;
  activation: number;
  color: [number, number, number];
}> {
  if (!state?.emotional_stack) return [];
  const entries = Object.entries(state.emotional_stack)
    .filter(([, v]) => v > 0.05)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);

  return entries.map(([emotion, activation]) => ({
    emotion: emotion as PrimaryEmotion,
    activation,
    color: hexToRgb(EMOTION_COLORS[emotion as PrimaryEmotion] || EMOTION_COLORS.neutral),
  }));
}

const PARTICLE_COUNT = 180;
const PLASMA_RESOLUTION = 3; // pixel step for plasma field

export function EmotionGenesis({ emotionalState }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);
  const timeRef = useRef(0);
  // Smooth transition targets
  const currentPalette = useRef({ primary: [108, 122, 137] as [number, number, number], secondary: [26, 188, 156] as [number, number, number], ambient: [20, 20, 30] as [number, number, number] });
  const targetPalette = useRef({ ...currentPalette.current });
  const currentIntensity = useRef(0.3);
  const currentArousal = useRef(0.3);
  const currentValence = useRef(0);

  // Initialize particles
  const initParticles = useCallback((w: number, h: number) => {
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) * 0.38;
    const particles: Particle[] = [];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const layer = i < 40 ? 0 : i < 120 ? 1 : 2;
      const layerR = maxR * (0.3 + layer * 0.3);
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * layerR;
      const x = cx + Math.cos(angle) * dist;
      const y = cy + Math.sin(angle) * dist;
      particles.push({
        x, y, baseX: x, baseY: y,
        size: 1 + Math.random() * 2.5 * (1 + layer * 0.3),
        speed: 0.3 + Math.random() * 0.7 + layer * 0.2,
        phase: Math.random() * Math.PI * 2,
        layer,
      });
    }
    particlesRef.current = particles;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    // Resize handler
    function resize() {
      const rect = canvas!.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas!.width = rect.width * dpr;
      canvas!.height = rect.height * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      initParticles(rect.width, rect.height);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function frame() {
      const w = canvas!.getBoundingClientRect().width;
      const h = canvas!.getBoundingClientRect().height;
      const t = timeRef.current;
      timeRef.current += 0.008;

      // Smooth lerp towards target values
      const lerpF = 0.04;
      const cp = currentPalette.current;
      const tp = targetPalette.current;
      for (let i = 0; i < 3; i++) {
        cp.primary[i] += (tp.primary[i] - cp.primary[i]) * lerpF;
        cp.secondary[i] += (tp.secondary[i] - cp.secondary[i]) * lerpF;
        cp.ambient[i] += (tp.ambient[i] - cp.ambient[i]) * lerpF;
      }
      const targetI = emotionalState ? emotionalState.intensity : 0.3;
      const targetA = emotionalState ? emotionalState.arousal : 0.3;
      const targetV = emotionalState ? emotionalState.valence : 0;
      currentIntensity.current += (targetI - currentIntensity.current) * lerpF;
      currentArousal.current += (targetA - currentArousal.current) * lerpF;
      currentValence.current += (targetV - currentValence.current) * lerpF;

      const intensity = currentIntensity.current;
      const arousal = currentArousal.current;
      const valence = currentValence.current;

      const { primary, secondary, ambient } = cp;
      const cx = w / 2;
      const cy = h / 2;
      const maxR = Math.min(w, h) * 0.38;

      // === Layer 0: Background fill ===
      ctx!.fillStyle = `rgb(${ambient[0] | 0},${ambient[1] | 0},${ambient[2] | 0})`;
      ctx!.fillRect(0, 0, w, h);

      // === Layer 1: Plasma field ===
      const activeEmotions = getActiveEmotions(emotionalState, 5);
      const plasmaAlpha = 0.12 + intensity * 0.18;

      for (let px = 0; px < w; px += PLASMA_RESOLUTION) {
        for (let py = 0; py < h; py += PLASMA_RESOLUTION) {
          const dx = (px - cx) / maxR;
          const dy = (py - cy) / maxR;
          const distFromCenter = Math.sqrt(dx * dx + dy * dy);

          // Skip pixels far from the organism
          if (distFromCenter > 1.8) continue;

          const n = fbm(dx * 1.5, dy * 1.5, t * (0.3 + arousal * 0.5), 3);
          const n2 = noise2D(dx * 2 + t * 0.2, dy * 2 + t * 0.15);

          // Blend active emotions based on noise
          let r = 0, g = 0, b = 0;
          if (activeEmotions.length > 0) {
            let totalWeight = 0;
            for (let i = 0; i < activeEmotions.length; i++) {
              const e = activeEmotions[i];
              const phase = noise2D(dx * 3 + i * 7.3, dy * 3 + t * 0.4 + i * 3.1);
              const weight = e.activation * (0.5 + 0.5 * phase);
              r += e.color[0] * weight;
              g += e.color[1] * weight;
              b += e.color[2] * weight;
              totalWeight += weight;
            }
            if (totalWeight > 0) { r /= totalWeight; g /= totalWeight; b /= totalWeight; }
          } else {
            r = primary[0]; g = primary[1]; b = primary[2];
          }

          // Modulate by noise
          const nMod = 0.5 + 0.5 * n;
          r *= nMod; g *= nMod; b *= nMod;

          // Radial falloff
          const falloff = Math.max(0, 1 - distFromCenter * 0.7);
          const alpha = plasmaAlpha * falloff * (0.6 + 0.4 * Math.abs(n2));

          if (alpha > 0.01) {
            ctx!.fillStyle = `rgba(${r | 0},${g | 0},${b | 0},${alpha})`;
            ctx!.fillRect(px, py, PLASMA_RESOLUTION, PLASMA_RESOLUTION);
          }
        }
      }

      // === Layer 2: Core glow ===
      const glowR = maxR * (0.5 + intensity * 0.3);
      const pulseR = glowR * (1 + 0.08 * Math.sin(t * (2 + arousal * 4)));
      const coreGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, pulseR);
      coreGrad.addColorStop(0, `rgba(${primary[0] | 0},${primary[1] | 0},${primary[2] | 0},${0.25 + intensity * 0.2})`);
      coreGrad.addColorStop(0.4, `rgba(${secondary[0] | 0},${secondary[1] | 0},${secondary[2] | 0},${0.1 + intensity * 0.08})`);
      coreGrad.addColorStop(1, `rgba(${ambient[0] | 0},${ambient[1] | 0},${ambient[2] | 0},0)`);
      ctx!.fillStyle = coreGrad;
      ctx!.beginPath();
      ctx!.arc(cx, cy, pulseR, 0, Math.PI * 2);
      ctx!.fill();

      // === Layer 3: Particles ===
      const particles = particlesRef.current;
      const speedMul = 0.5 + arousal * 1.5;
      const wanderMul = 15 + intensity * 30;

      for (const p of particles) {
        const age = t * p.speed * speedMul + p.phase;
        // Perlin-driven wandering
        const nx = noise2D(p.baseX * 0.005 + age * 0.3, p.baseY * 0.005);
        const ny = noise2D(p.baseY * 0.005 + age * 0.3, p.baseX * 0.005 + 100);
        p.x = p.baseX + nx * wanderMul;
        p.y = p.baseY + ny * wanderMul;

        // Distance from center for alpha
        const dx = p.x - cx;
        const dy = p.y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const distRatio = dist / maxR;
        if (distRatio > 1.5) continue;

        // Color: blend primary/secondary by layer
        const layerBlend = p.layer / 2;
        const pr = primary[0] + (secondary[0] - primary[0]) * layerBlend;
        const pg = primary[1] + (secondary[1] - primary[1]) * layerBlend;
        const pb = primary[2] + (secondary[2] - primary[2]) * layerBlend;

        // Twinkle
        const twinkle = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(age * 3 + p.phase * 5));
        const alpha = twinkle * (1 - distRatio * 0.6) * (0.3 + intensity * 0.5);
        const size = p.size * (0.8 + 0.4 * Math.sin(age * 2));

        ctx!.beginPath();
        ctx!.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${pr | 0},${pg | 0},${pb | 0},${alpha.toFixed(2)})`;
        ctx!.fill();
      }

      // === Layer 4: Membrane / boundary ===
      const membranePoints = 64;
      const membraneR = maxR * (0.85 + 0.05 * Math.sin(t * 1.5));
      ctx!.beginPath();
      for (let i = 0; i <= membranePoints; i++) {
        const angle = (i / membranePoints) * Math.PI * 2;
        const noiseVal = noise2D(
          Math.cos(angle) * 2 + t * 0.3,
          Math.sin(angle) * 2 + t * 0.25,
        );
        const r = membraneR + noiseVal * (10 + intensity * 20);
        const mx = cx + Math.cos(angle) * r;
        const my = cy + Math.sin(angle) * r;
        if (i === 0) ctx!.moveTo(mx, my);
        else ctx!.lineTo(mx, my);
      }
      ctx!.closePath();
      ctx!.strokeStyle = `rgba(${primary[0] | 0},${primary[1] | 0},${primary[2] | 0},${0.08 + intensity * 0.12})`;
      ctx!.lineWidth = 1 + intensity * 1.5;
      ctx!.stroke();

      // === Layer 5: Energy tendrils (high arousal) ===
      if (arousal > 0.4) {
        const tendrilCount = Math.floor(3 + arousal * 5);
        const tendrilAlpha = (arousal - 0.4) * 0.5;
        ctx!.lineWidth = 0.8;

        for (let i = 0; i < tendrilCount; i++) {
          const baseAngle = (i / tendrilCount) * Math.PI * 2 + t * 0.5;
          ctx!.beginPath();
          ctx!.strokeStyle = `rgba(${secondary[0] | 0},${secondary[1] | 0},${secondary[2] | 0},${tendrilAlpha.toFixed(2)})`;

          for (let s = 0; s < 30; s++) {
            const progress = s / 30;
            const r = maxR * (0.3 + progress * 0.7);
            const angleOff = noise2D(i * 5 + s * 0.1, t * 0.8) * 0.8;
            const mx = cx + Math.cos(baseAngle + angleOff) * r;
            const my = cy + Math.sin(baseAngle + angleOff) * r;
            if (s === 0) ctx!.moveTo(mx, my);
            else ctx!.lineTo(mx, my);
          }
          ctx!.stroke();
        }
      }

      // === Layer 6: Valence aura ===
      if (Math.abs(valence) > 0.3) {
        const auraR = maxR * 1.1;
        const auraGrad = ctx!.createRadialGradient(cx, cy, maxR * 0.6, cx, cy, auraR);
        const auraColor = valence > 0
          ? lerpColor("#FFD700", "#98D8C8", (valence - 0.3) / 0.7)
          : lerpColor("#3498DB", "#7B2D8E", (-valence - 0.3) / 0.7);
        const auraAlpha = Math.abs(valence) * 0.15;
        auraGrad.addColorStop(0, `rgba(${auraColor[0]},${auraColor[1]},${auraColor[2]},${auraAlpha.toFixed(2)})`);
        auraGrad.addColorStop(1, `rgba(${auraColor[0]},${auraColor[1]},${auraColor[2]},0)`);
        ctx!.fillStyle = auraGrad;
        ctx!.beginPath();
        ctx!.arc(cx, cy, auraR, 0, Math.PI * 2);
        ctx!.fill();
      }

      animRef.current = requestAnimationFrame(frame);
    }

    animRef.current = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(animRef.current);
      ro.disconnect();
    };
  }, [initParticles, emotionalState]);

  // Update target palette when emotion changes
  useEffect(() => {
    targetPalette.current = getEmotionPalette(emotionalState);
  }, [emotionalState]);

  return (
    <div className="emotion-genesis">
      <canvas ref={canvasRef} className="emotion-genesis__canvas" />
      {emotionalState && (
        <div className="emotion-genesis__label">
          {emotionalState.primary_emotion}
          {emotionalState.secondary_emotion && emotionalState.secondary_emotion !== emotionalState.primary_emotion
            ? ` / ${emotionalState.secondary_emotion}`
            : ""}
        </div>
      )}
    </div>
  );
}
