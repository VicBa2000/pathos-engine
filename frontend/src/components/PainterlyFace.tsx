/**
 * PainterlyFace — Canvas 2D android/humanoid face renderer.
 *
 * Aesthetic: smooth bald android head, black background,
 * oversized amber/golden eyes with large pupils, minimal nose/mouth,
 * matte white skin, dramatic side lighting, deep eye sockets.
 *
 * Expressiveness comes from the emotional system:
 *  - Eyes: blink, gaze, pupil dilation, sclera reddening, veins
 *  - Brows: raise, furrow (rendered as subtle skin creases)
 *  - Mouth: curve, open, tension (thin and minimal but expressive)
 *  - Skin: flush/pallor, blush
 *  - Tears, sweat, tremor for extreme states
 */

import { useRef, useEffect } from "react";
import type { EmotionalState } from "../types/emotion";
import type { FaceParams, FaceColors } from "../lib/faceParams";
import {
  DEFAULT_FACE_PARAMS, DEFAULT_FACE_COLORS,
  deriveFaceParams, deriveFaceColors,
  lerpFaceParams, lerpFaceColors,
} from "../lib/faceParams";
import { noise2D } from "../lib/perlin";

interface Props {
  emotionalState: EmotionalState | null;
  analyser?: AnalyserNode | null;
  speaking?: boolean;
}

// Matte white android skin
const BASE_SKIN: [number, number, number] = [215, 218, 222];
// Muted lip — barely tinted
const BASE_LIP: [number, number, number] = [180, 170, 172];

const G_NONE = 0, G_GLANCE_L = 1, G_GLANCE_R = 2, G_GLANCE_UP = 3,
  G_TILT = 4, G_SQUINT = 5, G_BROW = 6;
const M_NONE = 0, M_TWITCH = 1, M_LIPPRESS = 2, M_BROWFLICK = 3;

function rgb(r: number, g: number, b: number): string {
  return `rgb(${Math.max(0, Math.min(255, r)) | 0},${Math.max(0, Math.min(255, g)) | 0},${Math.max(0, Math.min(255, b)) | 0})`;
}
function rgba(r: number, g: number, b: number, a: number): string {
  return `rgba(${Math.max(0, Math.min(255, r)) | 0},${Math.max(0, Math.min(255, g)) | 0},${Math.max(0, Math.min(255, b)) | 0},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
}

/** Android head — smooth, round, slightly wider at cranium */
function headPath(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.beginPath();
  // chin — rounded, not pointed
  ctx.moveTo(0, h * 0.85);
  // right jaw — smooth, rounded
  ctx.bezierCurveTo(w * 0.25, h * 0.92, w * 0.55, h * 0.80, w * 0.75, h * 0.50);
  // right cheek to temple
  ctx.bezierCurveTo(w * 0.88, h * 0.22, w * 0.95, -h * 0.08, w * 0.92, -h * 0.35);
  // cranium — wider, rounded top
  ctx.bezierCurveTo(w * 0.88, -h * 0.65, w * 0.65, -h * 0.92, w * 0.30, -h * 0.98);
  ctx.bezierCurveTo(w * 0.10, -h * 1.02, -w * 0.10, -h * 1.02, -w * 0.30, -h * 0.98);
  ctx.bezierCurveTo(-w * 0.65, -h * 0.92, -w * 0.88, -h * 0.65, -w * 0.92, -h * 0.35);
  // left temple to jaw
  ctx.bezierCurveTo(-w * 0.95, -h * 0.08, -w * 0.88, h * 0.22, -w * 0.75, h * 0.50);
  ctx.bezierCurveTo(-w * 0.55, h * 0.80, -w * 0.25, h * 0.92, 0, h * 0.85);
  ctx.closePath();
}

export function PainterlyFace({ emotionalState, analyser, speaking }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const timeRef = useRef(0);
  const curP = useRef<FaceParams>({ ...DEFAULT_FACE_PARAMS });
  const tgtP = useRef<FaceParams>({ ...DEFAULT_FACE_PARAMS });
  const curC = useRef<FaceColors>({ ...DEFAULT_FACE_COLORS });
  const tgtC = useRef<FaceColors>({ ...DEFAULT_FACE_COLORS });
  const blinkPhase = useRef(0);
  const blinkVal = useRef(0);
  const blinkTimer = useRef(0);
  const gestType = useRef(G_NONE);
  const gestProg = useRef(0);
  const gestTimer = useRef(0);
  const microType = useRef(M_NONE);
  const microProg = useRef(0);
  const microTimer = useRef(0);
  const smoothVol = useRef(0);
  const freqBuf = useRef<Uint8Array | null>(null);
  const energy = useRef(0.5);
  const arousalRef = useRef(0.3);
  const stackRef = useRef<Record<string, number>>({});

  useEffect(() => {
    if (emotionalState) {
      tgtP.current = deriveFaceParams(emotionalState);
      tgtC.current = deriveFaceColors(emotionalState);
      energy.current = emotionalState.body_state.energy;
      arousalRef.current = emotionalState.arousal;
      stackRef.current = emotionalState.emotional_stack || {};
    } else {
      tgtP.current = { ...DEFAULT_FACE_PARAMS };
      tgtC.current = { ...DEFAULT_FACE_COLORS };
      stackRef.current = {};
    }
  }, [emotionalState]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    function resize() {
      const r = canvas!.getBoundingClientRect();
      const d = window.devicePixelRatio || 1;
      canvas!.width = r.width * d;
      canvas!.height = r.height * d;
      ctx!.setTransform(d, 0, 0, d, 0, 0);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function frame() {
      const w = canvas!.getBoundingClientRect().width;
      const h = canvas!.getBoundingClientRect().height;
      const t = timeRef.current;
      timeRef.current += 0.008;
      const dt = 0.008;
      const g = ctx!;

      /* ======================== ANIMATE ======================== */
      curP.current = lerpFaceParams(curP.current, tgtP.current, 0.06);
      curC.current = lerpFaceColors(curC.current, tgtC.current, 0.06);
      const p: FaceParams = { ...curP.current };
      const c = curC.current;

      let vol = 0;
      if (analyser && speaking) {
        if (!freqBuf.current || freqBuf.current.length !== analyser.frequencyBinCount)
          freqBuf.current = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(freqBuf.current);
        let sum = 0;
        for (let i = 0; i < freqBuf.current.length; i++) sum += freqBuf.current[i];
        vol = sum / (freqBuf.current.length * 255);
      }
      const tgtVol = speaking ? (analyser ? vol : 0.25 + Math.sin(t * 9) * 0.2) : 0;
      smoothVol.current += (tgtVol - smoothVol.current) * 0.3;
      if (smoothVol.current > 0.01) {
        const lv = smoothVol.current;
        p.mouthOpenness = Math.max(p.mouthOpenness, lv * 3.5);
        p.jawDrop = Math.max(p.jawDrop, lv * 2.5);
        p.mouthWidth = Math.max(p.mouthWidth, 0.9 + lv * 0.5);
      }

      blinkTimer.current += dt;
      const blinkInt = 2.8 + (1 - energy.current) * 3.5 + noise2D(t * 0.08, 50) * 1.5;
      if (blinkPhase.current === 0 && blinkTimer.current > blinkInt) { blinkPhase.current = 1; blinkTimer.current = 0; }
      if (blinkPhase.current === 1) { blinkVal.current += 0.22; if (blinkVal.current >= 1) { blinkVal.current = 1; blinkPhase.current = 2; } }
      else if (blinkPhase.current === 2) { blinkVal.current -= 0.09; if (blinkVal.current <= 0) { blinkVal.current = 0; blinkPhase.current = 0; } }

      gestTimer.current += dt;
      if (gestType.current === G_NONE && gestTimer.current > 5.0 + noise2D(t * 0.04, 88) * 4.0) {
        const r = Math.random();
        gestType.current = r < 0.18 ? G_GLANCE_L : r < 0.36 ? G_GLANCE_R : r < 0.50 ? G_GLANCE_UP
          : r < 0.65 ? G_TILT : r < 0.82 ? G_SQUINT : G_BROW;
        gestProg.current = 0; gestTimer.current = 0;
      }
      if (gestType.current !== G_NONE) {
        gestProg.current += dt * 0.6;
        const gp = gestProg.current;
        const gi = gp < 0.35 ? gp / 0.35 : gp < 0.65 ? 1.0 : Math.max(0, 1 - (gp - 0.65) / 0.35);
        if (gestType.current === G_GLANCE_L) { p.pupilOffsetX -= 0.7 * gi; p.headTilt -= 0.12 * gi; }
        else if (gestType.current === G_GLANCE_R) { p.pupilOffsetX += 0.7 * gi; p.headTilt += 0.12 * gi; }
        else if (gestType.current === G_GLANCE_UP) { p.pupilOffsetY -= 0.6 * gi; p.browHeight += 0.4 * gi; }
        else if (gestType.current === G_TILT) { p.headTilt += 0.5 * gi * (noise2D(t * 0.01, 77) > 0 ? 1 : -1); }
        else if (gestType.current === G_SQUINT) { p.eyeSquint += 0.6 * gi; p.browHeight -= 0.25 * gi; }
        else if (gestType.current === G_BROW) { p.browHeight += 0.5 * gi; p.eyeOpenness += 0.15 * gi; }
        if (gestProg.current > 1.0) gestType.current = G_NONE;
      }

      microTimer.current += dt;
      if (microType.current === M_NONE && microTimer.current > 6.0 + noise2D(t * 0.03, 33) * 5.0) {
        const r = Math.random();
        microType.current = r < 0.35 ? M_TWITCH : r < 0.65 ? M_LIPPRESS : M_BROWFLICK;
        microProg.current = 0; microTimer.current = 0;
      }
      if (microType.current !== M_NONE) {
        microProg.current += dt * 2.5;
        const mi = microProg.current < 0.25 ? microProg.current / 0.25
          : Math.max(0, 1 - (microProg.current - 0.25) / 0.75);
        if (microType.current === M_TWITCH) p.mouthCurve += 0.15 * mi * (noise2D(t, 44) > 0 ? 1 : -1);
        else if (microType.current === M_LIPPRESS) { p.mouthTension += 0.35 * mi; p.mouthCurve -= 0.06 * mi; }
        else if (microType.current === M_BROWFLICK) p.browHeight += 0.25 * mi;
        if (microProg.current > 1.0) microType.current = M_NONE;
      }

      const swayX = noise2D(t * 0.15, 0) * 0.008;
      const swayY = noise2D(0, t * 0.12) * 0.006;
      const gazeNX = noise2D(t * 0.25, 10) * 0.10;
      const gazeNY = noise2D(10, t * 0.22) * 0.08;
      const breathY = Math.sin(t * (0.9 + arousalRef.current * 1.5)) * 0.003;

      /* ======================== GEOMETRY ======================== */
      const F = Math.min(w, h) * 0.90;
      const hW = F * 0.30;
      const hH = F * 0.38;
      const hx = w / 2 + swayX * F;
      const hy = h * 0.46 + swayY * F + breathY * F;
      const tilt = p.headTilt * 0.07;

      const stack = stackRef.current;
      const anger = stack.anger || 0;
      const sadness = stack.sadness || 0;
      const fear = stack.fear || 0;
      const anxiety = stack.anxiety || 0;
      const joy = stack.joy || 0;
      const tension = Math.max(anger, anxiety, p.mouthTension);

      // skin color — matte white/grey, subtle emotion shifts
      const ss = c.skinTintStrength * 0.4; // very subtle tinting
      let sr = BASE_SKIN[0] * (1 - ss) + c.skinTint[0] * ss;
      let sg = BASE_SKIN[1] * (1 - ss) + c.skinTint[1] * ss;
      let sb = BASE_SKIN[2] * (1 - ss) + c.skinTint[2] * ss;
      sr += anger * 15; sg -= anger * 10; sb -= anger * 12;
      const pallor = Math.max(fear, anxiety) * 0.2;
      sr += pallor * 8; sg += pallor * 6; sb += pallor * 5;
      sr -= sadness * 5; sg -= sadness * 3; sb += sadness * 3;

      const gc = c.glowColor;

      /* ======================== DRAW ======================== */
      g.fillStyle = "#000000";
      g.fillRect(0, 0, w, h);

      // subtle emotional glow on background
      const gR = F * 0.5;
      const glowGrad = g.createRadialGradient(hx, hy, 0, hx, hy, gR);
      glowGrad.addColorStop(0, rgba(gc[0], gc[1], gc[2], c.glowIntensity * 0.2));
      glowGrad.addColorStop(1, rgba(gc[0], gc[1], gc[2], 0));
      g.fillStyle = glowGrad;
      g.beginPath(); g.arc(hx, hy, gR, 0, Math.PI * 2); g.fill();

      g.save();
      g.translate(hx, hy);
      g.rotate(tilt);

      // === NECK — simple cylinder ===
      const nkW = hW * 0.32;
      g.beginPath();
      g.moveTo(-nkW, hH * 0.75);
      g.lineTo(-nkW * 1.05, hH * 1.15);
      g.lineTo(nkW * 1.05, hH * 1.15);
      g.lineTo(nkW, hH * 0.75);
      g.fillStyle = rgb(sr * 0.80, sg * 0.78, sb * 0.78);
      g.fill();
      // neck seam line (android detail)
      g.strokeStyle = rgba(sr * 0.55, sg * 0.52, sb * 0.52, 0.15);
      g.lineWidth = 0.8;
      g.beginPath(); g.moveTo(-nkW * 0.9, hH * 0.88); g.lineTo(nkW * 0.9, hH * 0.88); g.stroke();

      // === EARS — small, simplified ===
      for (const side of [-1, 1]) {
        const earX = hW * 0.88 * side;
        const earY = -hH * 0.02;
        g.beginPath();
        g.ellipse(earX, earY, hW * 0.07, hH * 0.10, side * 0.1, 0, Math.PI * 2);
        g.fillStyle = rgb(sr * 0.85, sg * 0.83, sb * 0.83);
        g.fill();
        g.beginPath();
        g.ellipse(earX + side * hW * 0.01, earY, hW * 0.04, hH * 0.06, 0, 0, Math.PI * 2);
        g.fillStyle = rgb(sr * 0.70, sg * 0.68, sb * 0.68);
        g.fill();
      }

      // === HEAD — matte white with dramatic side lighting ===
      headPath(g, hW, hH);
      // main skin — gradient simulating left-side key light
      const skinGrad = g.createLinearGradient(-hW * 1.0, -hH * 0.3, hW * 1.0, hH * 0.3);
      skinGrad.addColorStop(0, rgb(sr + 18, sg + 16, sb + 14));   // lit side
      skinGrad.addColorStop(0.45, rgb(sr + 5, sg + 4, sb + 3));
      skinGrad.addColorStop(0.65, rgb(sr, sg, sb));
      skinGrad.addColorStop(1, rgb(sr - 50, sg - 52, sb - 48));   // deep shadow side
      g.fillStyle = skinGrad;
      g.fill();

      // overlay: subtle radial for 3D roundness
      headPath(g, hW, hH);
      const roundGrad = g.createRadialGradient(-hW * 0.15, -hH * 0.2, hW * 0.1, 0, 0, hW * 1.1);
      roundGrad.addColorStop(0, rgba(255, 255, 255, 0.06));
      roundGrad.addColorStop(0.5, rgba(255, 255, 255, 0));
      roundGrad.addColorStop(1, rgba(0, 0, 0, 0.08));
      g.fillStyle = roundGrad;
      g.fill();

      // === DEEP EYE SOCKETS — dark recessed areas ===
      for (const side of [-1, 1]) {
        const sox = 0.11 * F * side;
        const soy = -0.02 * F;
        // large dark socket
        const sockG = g.createRadialGradient(sox, soy, F * 0.02, sox, soy, F * 0.11);
        sockG.addColorStop(0, rgba(25, 20, 22, 0.45 + tension * 0.1));
        sockG.addColorStop(0.5, rgba(40, 32, 35, 0.30 + sadness * 0.08));
        sockG.addColorStop(1, rgba(sr * 0.6, sg * 0.58, sb * 0.58, 0));
        g.fillStyle = sockG;
        g.beginPath(); g.arc(sox, soy, F * 0.11, 0, Math.PI * 2); g.fill();
        // deeper inner shadow
        const innerG = g.createRadialGradient(sox, soy - F * 0.01, F * 0.015, sox, soy, F * 0.06);
        innerG.addColorStop(0, rgba(15, 10, 12, 0.35));
        innerG.addColorStop(1, rgba(20, 15, 18, 0));
        g.fillStyle = innerG;
        g.beginPath(); g.arc(sox, soy, F * 0.06, 0, Math.PI * 2); g.fill();
      }

      // === NOSE — very minimal, just shadows ===
      const noseY = 0.06 * F;
      // left shadow line (from lighting)
      g.beginPath();
      g.moveTo(F * 0.005, -0.04 * F);
      g.quadraticCurveTo(F * 0.012, noseY * 0.5, F * 0.004, noseY);
      g.strokeStyle = rgba(sr * 0.55, sg * 0.52, sb * 0.52, 0.2);
      g.lineWidth = 1.2; g.stroke();
      // right shadow (deeper — shadow side)
      g.beginPath();
      g.moveTo(-F * 0.003, -0.04 * F);
      g.quadraticCurveTo(-F * 0.008, noseY * 0.5, -F * 0.003, noseY);
      g.strokeStyle = rgba(sr * 0.50, sg * 0.48, sb * 0.48, 0.12);
      g.lineWidth = 0.8; g.stroke();
      // nostrils — tiny dark dots
      const nFlare = 1 + Math.max(anger, fear) * 0.6;
      for (const side of [-1, 1]) {
        g.beginPath();
        g.ellipse(F * 0.016 * side * nFlare, noseY + F * 0.005, F * 0.007 * nFlare, F * 0.005 * nFlare, side * 0.2, 0, Math.PI * 2);
        g.fillStyle = rgba(sr * 0.30, sg * 0.28, sb * 0.28, 0.3 + anger * 0.15);
        g.fill();
      }
      // nose tip highlight
      const ntg = g.createRadialGradient(-F * 0.003, noseY - F * 0.005, 0, 0, noseY, F * 0.015);
      ntg.addColorStop(0, rgba(255, 255, 255, 0.1));
      ntg.addColorStop(1, rgba(255, 255, 255, 0));
      g.fillStyle = ntg;
      g.beginPath(); g.arc(0, noseY, F * 0.015, 0, Math.PI * 2); g.fill();
      // nose wrinkle (anger/disgust)
      if (p.noseWrinkle > 0.1) {
        const na = (p.noseWrinkle - 0.1) * 0.6;
        g.strokeStyle = rgba(sr * 0.45, sg * 0.42, sb * 0.42, na);
        g.lineWidth = 0.9;
        for (const side of [-1, 1]) {
          g.beginPath();
          g.moveTo(F * 0.006 * side, noseY - F * 0.015);
          g.quadraticCurveTo(F * 0.028 * side, noseY - F * 0.008, F * 0.04 * side, noseY - F * 0.018);
          g.stroke();
        }
      }

      // === EYES ===
      const eOpen = p.eyeOpenness * (1 - blinkVal.current);

      for (const side of [-1, 1]) {
        const ex = 0.11 * F * side;
        const ey = -0.02 * F;
        const eW = F * 0.082;
        const eH = F * 0.062 * Math.max(0.03, eOpen);
        const sqH = F * 0.02 * p.eyeSquint;

        if (eOpen > 0.03) {
          // === SCLERA — warm white, nearly round ===
          g.beginPath();
          g.moveTo(ex - eW, ey);
          g.quadraticCurveTo(ex, ey - eH * 1.15, ex + eW, ey);
          g.quadraticCurveTo(ex, ey + eH * 0.85 - sqH, ex - eW, ey);
          const scR = 248 + anger * 7;
          const scG = 244 - anger * 35 - tension * 10;
          const scB = 240 - anger * 45 - tension * 12;
          g.fillStyle = rgb(scR, scG, scB);
          g.fill();

          // sclera veins (anger/tension)
          if (tension > 0.35) {
            g.save();
            g.beginPath();
            g.moveTo(ex - eW, ey); g.quadraticCurveTo(ex, ey - eH * 1.15, ex + eW, ey);
            g.quadraticCurveTo(ex, ey + eH * 0.85 - sqH, ex - eW, ey); g.clip();
            const vA = (tension - 0.35) * 0.45;
            g.strokeStyle = rgba(185, 50, 40, vA);
            g.lineWidth = 0.35;
            for (let v = 0; v < 5; v++) {
              const vx = ex + (v - 2) * eW * 0.32;
              const vy = ey + noise2D(22, v * 5) * eH * 0.2;
              g.beginPath(); g.moveTo(vx, vy);
              g.quadraticCurveTo(vx + noise2D(v * 3, 11) * eW * 0.2, vy + noise2D(11, v * 3) * eH * 0.15,
                vx + noise2D(v * 11, 33) * eW * 0.3, vy + noise2D(33, v * 11) * eH * 0.2);
              g.stroke();
            }
            g.restore();
          }

          // === IRIS — amber/golden, glass-like ===
          const iR = Math.min(eH * 0.75, F * 0.038);
          if (iR > 2) {
            const gx = ex + (p.pupilOffsetX + gazeNX) * eW * 0.25;
            const gy = ey + (p.pupilOffsetY + gazeNY) * eH * 0.15;
            const ec = c.eyeColor;

            // outer dark ring
            g.beginPath(); g.arc(gx, gy, iR * 1.06, 0, Math.PI * 2);
            g.fillStyle = rgba(ec[0] * 0.15, ec[1] * 0.12, ec[2] * 0.08, 0.8);
            g.fill();

            // iris body — rich amber gradient
            const ig = g.createRadialGradient(gx - iR * 0.1, gy - iR * 0.12, iR * 0.04, gx, gy, iR);
            ig.addColorStop(0, rgb(Math.min(255, ec[0] + 100), Math.min(255, ec[1] + 80), Math.min(255, ec[2] + 40)));
            ig.addColorStop(0.2, rgb(Math.min(255, ec[0] + 50), Math.min(255, ec[1] + 35), Math.min(255, ec[2] + 10)));
            ig.addColorStop(0.5, rgb(ec[0], ec[1], ec[2]));
            ig.addColorStop(0.8, rgb(ec[0] * 0.5, ec[1] * 0.4, ec[2] * 0.2));
            ig.addColorStop(1, rgb(ec[0] * 0.2, ec[1] * 0.15, ec[2] * 0.08));
            g.beginPath(); g.arc(gx, gy, iR, 0, Math.PI * 2);
            g.fillStyle = ig; g.fill();

            // iris fibers
            g.save();
            g.beginPath(); g.arc(gx, gy, iR, 0, Math.PI * 2); g.clip();
            g.lineWidth = 0.3;
            for (let a = 0; a < Math.PI * 2; a += Math.PI / 16) {
              g.strokeStyle = rgba(ec[0] * 0.3, ec[1] * 0.25, ec[2] * 0.1, 0.08 + noise2D(a * 5, 88) * 0.04);
              g.beginPath();
              g.moveTo(gx + Math.cos(a) * iR * 0.2, gy + Math.sin(a) * iR * 0.2);
              g.lineTo(gx + Math.cos(a) * iR * 0.95, gy + Math.sin(a) * iR * 0.95);
              g.stroke();
            }
            g.restore();

            // inner glow ring (amber warmth)
            const iGlow = g.createRadialGradient(gx, gy, iR * 0.35, gx, gy, iR * 0.7);
            iGlow.addColorStop(0, rgba(255, 200, 80, 0.12));
            iGlow.addColorStop(1, rgba(255, 180, 60, 0));
            g.fillStyle = iGlow;
            g.beginPath(); g.arc(gx, gy, iR * 0.7, 0, Math.PI * 2); g.fill();

            // === PUPIL — large, deep black, hyper-reactive ===
            const pupilMult = 1 + fear * 0.5 + arousalRef.current * 0.3 - anger * 0.18;
            const pR = iR * p.pupilSize * Math.max(0.45, Math.min(1.5, pupilMult));
            // pupil gradient — deep black with very slight edge fade
            const pGrad = g.createRadialGradient(gx, gy, 0, gx, gy, pR);
            pGrad.addColorStop(0, "rgb(2,2,8)");
            pGrad.addColorStop(0.85, "rgb(5,4,10)");
            pGrad.addColorStop(1, rgba(ec[0] * 0.15, ec[1] * 0.12, ec[2] * 0.06, 0.5));
            g.beginPath(); g.arc(gx, gy, pR, 0, Math.PI * 2);
            g.fillStyle = pGrad; g.fill();

            // === SPECULARS — bright, glass-like ===
            // main specular
            g.beginPath();
            g.arc(gx - iR * 0.3, gy - iR * 0.3, Math.max(3, pR * 0.5), 0, Math.PI * 2);
            g.fillStyle = "rgba(255,255,255,0.95)"; g.fill();
            // secondary
            g.beginPath();
            g.arc(gx + iR * 0.2, gy + iR * 0.22, Math.max(1.5, pR * 0.25), 0, Math.PI * 2);
            g.fillStyle = "rgba(255,255,255,0.4)"; g.fill();
            // ambient glow reflection at bottom of eye
            g.beginPath();
            g.arc(gx, gy + iR * 0.55, iR * 0.25, 0, Math.PI * 2);
            g.fillStyle = rgba(gc[0], gc[1], gc[2], c.glowIntensity * 0.08);
            g.fill();
          }

          // upper eyelid edge — thick dark line
          g.beginPath();
          g.moveTo(ex - eW, ey);
          g.quadraticCurveTo(ex, ey - eH * 1.15, ex + eW, ey);
          g.strokeStyle = rgba(20, 15, 15, 0.75);
          g.lineWidth = 2.5; g.stroke();

          // eyelid shadow above (adds depth)
          g.beginPath();
          g.moveTo(ex - eW * 0.95, ey - eH * 0.4);
          g.quadraticCurveTo(ex, ey - eH * 1.5, ex + eW * 0.95, ey - eH * 0.4);
          g.strokeStyle = rgba(sr * 0.50, sg * 0.48, sb * 0.48, 0.15 + tension * 0.08);
          g.lineWidth = 0.8; g.stroke();

          // lower lid — subtle
          g.beginPath();
          g.moveTo(ex - eW * 0.7, ey + eH * 0.35 - sqH);
          g.quadraticCurveTo(ex, ey + eH * 0.85 - sqH, ex + eW * 0.7, ey + eH * 0.35 - sqH);
          g.strokeStyle = rgba(30, 22, 22, 0.12);
          g.lineWidth = 1.0; g.stroke();

          // crow's feet (squint/joy)
          if (p.eyeSquint > 0.25 || joy > 0.3) {
            const cra = Math.max((p.eyeSquint - 0.25) * 0.5, joy * 0.3);
            g.strokeStyle = rgba(sr * 0.50, sg * 0.48, sb * 0.48, cra);
            g.lineWidth = 0.7;
            for (let i = 0; i < 3; i++) {
              g.beginPath();
              g.moveTo(ex + eW * side * 0.92, ey - eH * 0.25 + i * eH * 0.22);
              g.lineTo(ex + eW * side * 0.92 + side * F * 0.01, ey - eH * 0.25 + i * eH * 0.22 + (i - 1) * F * 0.003);
              g.stroke();
            }
          }
        } else {
          // closed — simple line
          g.beginPath();
          g.moveTo(ex - eW, ey);
          g.quadraticCurveTo(ex, ey + F * 0.005, ex + eW, ey);
          g.strokeStyle = rgba(20, 15, 15, 0.7);
          g.lineWidth = 2.2; g.stroke();
        }
      }

      // === EYEBROWS — thin, elegant, android ===
      g.lineCap = "round";
      for (const side of [-1, 1]) {
        const bxBase = 0.11 * F * side;
        const byBase = -0.11 * F;
        const bh = p.browHeight * 0.15 * F;
        const ba = p.browAngle * 0.10 * F * side;
        const bc = p.browCurvature * 0.035 * F;

        const inner: [number, number] = [bxBase - side * 0.068 * F, byBase - bh + ba];
        const mid: [number, number] = [bxBase, byBase - bh - bc];
        const outer: [number, number] = [bxBase + side * 0.072 * F, byBase - bh * 0.5 - ba * 0.3];

        // main brow — thin elegant stroke
        g.beginPath();
        g.moveTo(inner[0], inner[1]);
        g.quadraticCurveTo(mid[0], mid[1], outer[0], outer[1]);
        g.strokeStyle = rgba(sr * 0.32, sg * 0.28, sb * 0.28, 0.7);
        g.lineWidth = F * 0.011;
        g.stroke();

        // subtle second stroke for thickness
        g.beginPath();
        g.moveTo(inner[0], inner[1] + F * 0.002);
        g.quadraticCurveTo(mid[0], mid[1] + F * 0.003, outer[0], outer[1] + F * 0.002);
        g.strokeStyle = rgba(sr * 0.35, sg * 0.30, sb * 0.30, 0.3);
        g.lineWidth = F * 0.006;
        g.stroke();

        // brow furrow V-lines (anger)
        if (p.browAngle < -0.2) {
          const fa = Math.abs(p.browAngle + 0.2) * 0.8;
          g.strokeStyle = rgba(sr * 0.40, sg * 0.36, sb * 0.36, fa);
          g.lineWidth = 1.3;
          g.beginPath();
          g.moveTo(side * 0.038 * F, byBase - bh + F * 0.008);
          g.lineTo(side * 0.018 * F, byBase - bh + F * 0.052);
          g.stroke();
          if (fa > 0.25) {
            g.beginPath();
            g.moveTo(side * 0.030 * F, byBase - bh + F * 0.003);
            g.lineTo(side * 0.014 * F, byBase - bh + F * 0.042);
            g.stroke();
          }
        }
      }

      // forehead lines
      if (p.foreheadTension > 0.12) {
        const fa = (p.foreheadTension - 0.12) * 0.7;
        g.strokeStyle = rgba(sr * 0.48, sg * 0.45, sb * 0.45, fa);
        g.lineWidth = 1.0;
        for (let i = 0; i < (p.foreheadTension > 0.4 ? 3 : 2); i++) {
          const ly = -0.20 * F + i * 0.016 * F;
          g.beginPath(); g.moveTo(-0.08 * F, ly);
          g.quadraticCurveTo(0, ly - 0.005 * F, 0.08 * F, ly); g.stroke();
        }
      }

      // === MOUTH — filled lip shapes, expressive ===
      const mY = 0.15 * F;
      const mc = p.mouthCurve;
      const mo = Math.max(p.mouthOpenness, p.jawDrop * 0.8);
      const mt = p.mouthTension;
      const mw = p.mouthWidth * 0.12 * F;
      const cDip = mc * -0.13 * F;

      let tremor = 0;
      if (anxiety > 0.4 || fear > 0.4) tremor = Math.max(anxiety, fear) * noise2D(t * 10, 99) * F * 0.003;

      const uCPy = mY - mc * 0.06 * F - mt * 0.01 * F;
      const openPx = mo * F * 0.16 * (1 - mt * 0.4);

      const lipR = BASE_LIP[0] + anger * 12 - sadness * 5;
      const lipG = BASE_LIP[1] - anger * 8;
      const lipB = BASE_LIP[2] - anger * 6;

      // mouth interior + teeth
      if (mo > 0.04 && openPx > 2) {
        const lmy = mY + openPx + Math.abs(mc) * 0.018 * F + tremor;
        g.beginPath();
        g.moveTo(-mw, mY + cDip + tremor);
        g.quadraticCurveTo(-mw * 0.4, uCPy - F * 0.002, 0, uCPy);
        g.quadraticCurveTo(mw * 0.4, uCPy - F * 0.002, mw, mY + cDip + tremor);
        g.quadraticCurveTo(0, lmy, -mw, mY + cDip + tremor);
        g.fillStyle = rgba(10, 5, 5, 0.92);
        g.fill();

        if (openPx > F * 0.012) {
          const tW = mw * 0.55, tH = Math.min(openPx * 0.25, F * 0.014), tY = uCPy + F * 0.002;
          g.save();
          g.beginPath();
          g.moveTo(-mw, mY + cDip + tremor);
          g.quadraticCurveTo(-mw * 0.4, uCPy - F * 0.002, 0, uCPy);
          g.quadraticCurveTo(mw * 0.4, uCPy - F * 0.002, mw, mY + cDip + tremor);
          g.quadraticCurveTo(0, lmy, -mw, mY + cDip + tremor);
          g.clip();
          g.fillStyle = rgba(242, 238, 232, 0.88);
          g.fillRect(-tW, tY, tW * 2, tH);
          g.strokeStyle = rgba(185, 180, 172, 0.22);
          g.lineWidth = 0.4;
          for (let i = -3; i <= 3; i++) {
            g.beginPath(); g.moveTo(i * tW * 0.27, tY); g.lineTo(i * tW * 0.27, tY + tH); g.stroke();
          }
          if (openPx > F * 0.04) {
            const tg = g.createRadialGradient(0, lmy - openPx * 0.3, 0, 0, lmy - openPx * 0.3, mw * 0.4);
            tg.addColorStop(0, rgba(165, 70, 70, 0.45));
            tg.addColorStop(1, rgba(120, 40, 40, 0));
            g.fillStyle = tg;
            g.beginPath(); g.ellipse(0, lmy - openPx * 0.3, mw * 0.35, openPx * 0.2, 0, 0, Math.PI * 2); g.fill();
          }
          g.restore();
        }
      }

      // teeth clench
      if (mt > 0.5 && mo < 0.15) {
        const cG = F * 0.009 * (mt - 0.5) * 2.5;
        g.fillStyle = rgba(242, 238, 232, 0.65);
        g.fillRect(-mw * 0.55, mY + cDip - cG * 0.3 + tremor, mw * 1.1, cG);
      }

      // upper lip — filled shape
      {
        const lipThick = F * 0.008 + mo * F * 0.003;
        g.beginPath();
        g.moveTo(-mw, mY + cDip + tremor);
        g.quadraticCurveTo(-mw * 0.4, uCPy - F * 0.003, 0, uCPy);
        g.quadraticCurveTo(mw * 0.4, uCPy - F * 0.003, mw, mY + cDip + tremor);
        g.quadraticCurveTo(mw * 0.35, uCPy + lipThick, 0, uCPy + lipThick * 1.1);
        g.quadraticCurveTo(-mw * 0.35, uCPy + lipThick, -mw, mY + cDip + tremor);
        g.closePath();
        const ulGrad = g.createLinearGradient(0, uCPy - F * 0.003, 0, uCPy + lipThick);
        ulGrad.addColorStop(0, rgb(lipR - 10, lipG - 5, lipB - 5));
        ulGrad.addColorStop(0.5, rgb(lipR, lipG, lipB));
        ulGrad.addColorStop(1, rgb(lipR - 15, lipG - 10, lipB - 8));
        g.fillStyle = ulGrad;
        g.fill();
      }

      // lower lip — filled shape
      if (mo > 0.04 && openPx > 2) {
        const lmy = mY + openPx + Math.abs(mc) * 0.018 * F + tremor;
        const lThick = F * 0.01 + mo * F * 0.004;
        g.beginPath();
        g.moveTo(-mw * 0.9, mY + cDip + tremor);
        g.quadraticCurveTo(0, lmy - lThick * 0.2, mw * 0.9, mY + cDip + tremor);
        g.quadraticCurveTo(0, lmy + lThick * 0.5, -mw * 0.9, mY + cDip + tremor);
        g.closePath();
        const llGrad = g.createLinearGradient(0, lmy - lThick, 0, lmy + lThick * 0.5);
        llGrad.addColorStop(0, rgb(lipR + 5, lipG + 3, lipB + 3));
        llGrad.addColorStop(0.5, rgb(lipR + 8, lipG + 4, lipB + 4));
        llGrad.addColorStop(1, rgb(lipR - 8, lipG - 6, lipB - 5));
        g.fillStyle = llGrad;
        g.fill();
        // shine
        const lsg = g.createRadialGradient(0, lmy - lThick * 0.1, 0, 0, lmy - lThick * 0.1, mw * 0.25);
        lsg.addColorStop(0, rgba(255, 255, 255, 0.1));
        lsg.addColorStop(1, rgba(255, 255, 255, 0));
        g.fillStyle = lsg;
        g.beginPath(); g.arc(0, lmy - lThick * 0.1, mw * 0.25, 0, Math.PI * 2); g.fill();
      } else {
        // closed lower lip
        const clY = mY + cDip + F * 0.004 + tremor;
        const clThick = F * 0.01;
        g.beginPath();
        g.moveTo(-mw * 0.82, clY);
        g.quadraticCurveTo(0, clY + clThick * 1.6, mw * 0.82, clY);
        g.quadraticCurveTo(0, clY + clThick * 0.4, -mw * 0.82, clY);
        g.closePath();
        g.fillStyle = rgb(lipR + 5, lipG + 2, lipB + 2);
        g.fill();
      }

      // mouth tension dimples
      if (mt > 0.25 || Math.abs(mc) > 0.3) {
        const dA = Math.max((mt - 0.25) * 0.45, Math.abs(mc) * 0.3);
        for (const side of [-1, 1]) {
          g.beginPath();
          g.arc(side * (mw + F * 0.005), mY + cDip + tremor, F * 0.004, 0, Math.PI * 2);
          g.fillStyle = rgba(sr * 0.42, sg * 0.38, sb * 0.38, dA);
          g.fill();
        }
      }

      // nasolabial folds (smile/cheekRaise)
      if (p.cheekRaise > 0.12 || joy > 0.2) {
        const cra = Math.max((p.cheekRaise - 0.12) * 0.6, joy * 0.3);
        g.strokeStyle = rgba(sr * 0.48, sg * 0.44, sb * 0.44, cra);
        g.lineWidth = 1.0;
        for (const side of [-1, 1]) {
          g.beginPath(); g.moveTo(side * 0.04 * F, 0.02 * F);
          g.quadraticCurveTo(side * 0.08 * F, 0.08 * F, side * 0.07 * F, 0.14 * F); g.stroke();
        }
      }

      // === SPECULAR HIGHLIGHTS ===
      // forehead
      {
        const fg = g.createRadialGradient(-hW * 0.2, -hH * 0.6, 0, -hW * 0.15, -hH * 0.55, hW * 0.3);
        fg.addColorStop(0, rgba(255, 255, 255, 0.12));
        fg.addColorStop(0.4, rgba(255, 255, 255, 0.04));
        fg.addColorStop(1, rgba(255, 255, 255, 0));
        g.fillStyle = fg;
        g.beginPath(); g.arc(-hW * 0.15, -hH * 0.55, hW * 0.3, 0, Math.PI * 2); g.fill();
      }
      // cranium top
      {
        const cg = g.createRadialGradient(-hW * 0.05, -hH * 0.85, 0, 0, -hH * 0.8, hW * 0.35);
        cg.addColorStop(0, rgba(255, 255, 255, 0.08));
        cg.addColorStop(1, rgba(255, 255, 255, 0));
        g.fillStyle = cg;
        g.beginPath(); g.arc(0, -hH * 0.8, hW * 0.35, 0, Math.PI * 2); g.fill();
      }
      // left cheekbone
      {
        const chg = g.createRadialGradient(-hW * 0.55, hH * 0.05, 0, -hW * 0.5, hH * 0.05, hW * 0.12);
        chg.addColorStop(0, rgba(255, 255, 255, 0.06));
        chg.addColorStop(1, rgba(255, 255, 255, 0));
        g.fillStyle = chg;
        g.beginPath(); g.arc(-hW * 0.5, hH * 0.05, hW * 0.12, 0, Math.PI * 2); g.fill();
      }

      // === RIM LIGHT — emotional color ===
      if (c.glowIntensity > 0.05) {
        g.save();
        headPath(g, hW, hH);
        g.clip();
        // right side rim (emotional color)
        const rg = g.createRadialGradient(hW * 1.05, -hH * 0.05, hW * 0.1, hW * 1.0, 0, hW * 0.8);
        rg.addColorStop(0, rgba(gc[0], gc[1], gc[2], c.glowIntensity * 0.35));
        rg.addColorStop(1, rgba(gc[0], gc[1], gc[2], 0));
        g.fillStyle = rg;
        g.fillRect(-hW * 1.5, -hH * 1.5, hW * 3, hH * 3);
        g.restore();
      }

      // === EXTREME EXPRESSIONS ===

      // tears
      if (sadness > 0.5) {
        const tearA = Math.min((sadness - 0.5) * 2.5, 0.75);
        for (const side of [-1, 1]) {
          const tx = 0.11 * F * side + side * F * 0.012;
          const ty = -0.02 * F + F * 0.065;
          g.beginPath(); g.moveTo(tx, ty);
          g.quadraticCurveTo(tx + side * F * 0.015, ty + F * 0.08, tx + side * F * 0.02, ty + F * 0.16);
          g.strokeStyle = rgba(170, 200, 230, tearA * 0.4);
          g.lineWidth = F * 0.005; g.stroke();
          const tdrop = (t * 0.35 + side * 0.5) % 1;
          const tdy = ty + tdrop * F * 0.15;
          g.beginPath(); g.arc(tx + side * F * 0.015 * tdrop, tdy, F * 0.006, 0, Math.PI * 2);
          g.fillStyle = rgba(170, 210, 240, tearA * (1 - tdrop) * 0.6); g.fill();
          g.beginPath(); g.arc(tx + side * F * 0.015 * tdrop - F * 0.002, tdy - F * 0.002, F * 0.003, 0, Math.PI * 2);
          g.fillStyle = rgba(255, 255, 255, tearA * (1 - tdrop) * 0.3); g.fill();
        }
      }

      // sweat
      if (anxiety > 0.6) {
        const swA = Math.min((anxiety - 0.6) * 1.8, 0.55);
        for (let i = 0; i < 3; i++) {
          const sx = (-0.04 + i * 0.04) * F + noise2D(t * 0.08 + i, 55) * F * 0.006;
          const sy = -0.24 * F + noise2D(i * 3.7, t * 0.04) * F * 0.012;
          g.beginPath(); g.arc(sx, sy, F * 0.004, 0, Math.PI * 2);
          g.fillStyle = rgba(190, 210, 230, swA * 0.4); g.fill();
        }
      }

      // head outline — very subtle
      headPath(g, hW, hH);
      g.strokeStyle = rgba(sr * 0.35, sg * 0.32, sb * 0.32, 0.15);
      g.lineWidth = 1.0; g.stroke();

      g.restore();
      animRef.current = requestAnimationFrame(frame);
    }

    animRef.current = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(animRef.current); ro.disconnect(); };
  }, [emotionalState, analyser, speaking]);

  return <canvas ref={canvasRef} className="emotion-avatar__canvas" />;
}
