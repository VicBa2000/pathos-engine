import { useRef, useEffect } from "react";
import type { EmotionalState } from "../types/emotion";
import type { FaceParams, FaceColors } from "../lib/faceParams";
import {
  DEFAULT_FACE_PARAMS, DEFAULT_FACE_COLORS,
  deriveFaceParams, deriveFaceColors,
  lerpFaceParams, lerpFaceColors,
} from "../lib/faceParams";
import { noise2D } from "../lib/perlin";
import "./EmotionAvatar.css";

interface Props {
  emotionalState: EmotionalState | null;
  analyser?: AnalyserNode | null;
  speaking?: boolean;
}

const SKIN: [number, number, number] = [235, 220, 205];
const LIP: [number, number, number] = [190, 125, 118];

// Gesture types
const G_NONE = 0, G_GLANCE_L = 1, G_GLANCE_R = 2, G_GLANCE_UP = 3,
  G_TILT = 4, G_SQUINT = 5, G_BROW = 6;
// Micro types
const M_NONE = 0, M_TWITCH = 1, M_LIPPRESS = 2, M_BROWFLICK = 3;

export function EmotionAvatar({ emotionalState, analyser, speaking }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const timeRef = useRef(0);

  const curP = useRef<FaceParams>({ ...DEFAULT_FACE_PARAMS });
  const tgtP = useRef<FaceParams>({ ...DEFAULT_FACE_PARAMS });
  const curC = useRef<FaceColors>({ ...DEFAULT_FACE_COLORS });
  const tgtC = useRef<FaceColors>({ ...DEFAULT_FACE_COLORS });

  // Blink
  const blinkPhase = useRef(0); // 0=idle,1=closing,2=opening
  const blinkVal = useRef(0);
  const blinkTimer = useRef(0);
  // Gesture
  const gestType = useRef(G_NONE);
  const gestProg = useRef(0);
  const gestTimer = useRef(0);
  // Micro
  const microType = useRef(M_NONE);
  const microProg = useRef(0);
  const microTimer = useRef(0);
  // Lip sync
  const smoothVol = useRef(0);
  const freqBuf = useRef<Uint8Array | null>(null);
  // Cache
  const energy = useRef(0.5);
  const arousalRef = useRef(0.3);

  useEffect(() => {
    if (emotionalState) {
      tgtP.current = deriveFaceParams(emotionalState);
      tgtC.current = deriveFaceColors(emotionalState);
      energy.current = emotionalState.body_state.energy;
      arousalRef.current = emotionalState.arousal;
    } else {
      tgtP.current = { ...DEFAULT_FACE_PARAMS };
      tgtC.current = { ...DEFAULT_FACE_COLORS };
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

      // --- Lerp ---
      curP.current = lerpFaceParams(curP.current, tgtP.current, 0.06);
      curC.current = lerpFaceColors(curC.current, tgtC.current, 0.06);

      const p: FaceParams = { ...curP.current }; // mutable for overlays
      const c = curC.current;

      // --- Lip sync ---
      let vol = 0;
      if (analyser && speaking) {
        if (!freqBuf.current || freqBuf.current.length !== analyser.frequencyBinCount) {
          freqBuf.current = new Uint8Array(analyser.frequencyBinCount);
        }
        analyser.getByteFrequencyData(freqBuf.current);
        let sum = 0;
        for (let i = 0; i < freqBuf.current.length; i++) sum += freqBuf.current[i];
        vol = sum / (freqBuf.current.length * 255);
      }
      const tgtVol = speaking ? (analyser ? vol : 0.25 + Math.sin(t * 9) * 0.2) : 0;
      smoothVol.current += (tgtVol - smoothVol.current) * 0.3;

      // Lip sync: STRONG mouth opening
      if (smoothVol.current > 0.01) {
        const lv = smoothVol.current;
        p.mouthOpenness = Math.max(p.mouthOpenness, lv * 3.5);
        p.jawDrop = Math.max(p.jawDrop, lv * 2.5);
        p.mouthWidth = Math.max(p.mouthWidth, 0.9 + lv * 0.5);
      }

      // --- Blink ---
      blinkTimer.current += dt;
      const blinkInt = 2.0 + (1 - energy.current) * 3.0 + noise2D(t * 0.08, 50) * 1.5;
      if (blinkPhase.current === 0 && blinkTimer.current > blinkInt) {
        blinkPhase.current = 1; blinkTimer.current = 0;
      }
      if (blinkPhase.current === 1) {
        blinkVal.current += 0.3;
        if (blinkVal.current >= 1) { blinkVal.current = 1; blinkPhase.current = 2; }
      } else if (blinkPhase.current === 2) {
        blinkVal.current -= 0.12;
        if (blinkVal.current <= 0) { blinkVal.current = 0; blinkPhase.current = 0; }
      }

      // --- Idle gestures ---
      gestTimer.current += dt;
      if (gestType.current === G_NONE && gestTimer.current > 4.0 + noise2D(t * 0.04, 88) * 3.5) {
        const r = Math.random();
        gestType.current = r < 0.18 ? G_GLANCE_L : r < 0.36 ? G_GLANCE_R : r < 0.50 ? G_GLANCE_UP
          : r < 0.65 ? G_TILT : r < 0.82 ? G_SQUINT : G_BROW;
        gestProg.current = 0; gestTimer.current = 0;
      }
      if (gestType.current !== G_NONE) {
        gestProg.current += dt * 0.7;
        const gp = gestProg.current;
        const gi = gp < 0.35 ? gp / 0.35 : gp < 0.65 ? 1.0 : Math.max(0, 1 - (gp - 0.65) / 0.35);
        if (gestType.current === G_GLANCE_L) { p.pupilOffsetX -= 0.7 * gi; p.headTilt -= 0.2 * gi; }
        else if (gestType.current === G_GLANCE_R) { p.pupilOffsetX += 0.7 * gi; p.headTilt += 0.2 * gi; }
        else if (gestType.current === G_GLANCE_UP) { p.pupilOffsetY -= 0.6 * gi; p.browHeight += 0.4 * gi; }
        else if (gestType.current === G_TILT) { p.headTilt += 0.5 * gi * (noise2D(t * 0.01, 77) > 0 ? 1 : -1); }
        else if (gestType.current === G_SQUINT) { p.eyeSquint += 0.6 * gi; p.browHeight -= 0.25 * gi; }
        else if (gestType.current === G_BROW) { p.browHeight += 0.6 * gi; p.eyeOpenness += 0.2 * gi; }
        if (gestProg.current > 1.0) gestType.current = G_NONE;
      }

      // --- Microexpressions ---
      microTimer.current += dt;
      if (microType.current === M_NONE && microTimer.current > 5.0 + noise2D(t * 0.03, 33) * 4.0) {
        const r = Math.random();
        microType.current = r < 0.35 ? M_TWITCH : r < 0.65 ? M_LIPPRESS : M_BROWFLICK;
        microProg.current = 0; microTimer.current = 0;
      }
      if (microType.current !== M_NONE) {
        microProg.current += dt * 2.8;
        const mi = microProg.current < 0.25 ? microProg.current / 0.25
          : Math.max(0, 1 - (microProg.current - 0.25) / 0.75);
        if (microType.current === M_TWITCH) p.mouthCurve += 0.2 * mi * (noise2D(t, 44) > 0 ? 1 : -1);
        else if (microType.current === M_LIPPRESS) { p.mouthTension += 0.4 * mi; p.mouthCurve -= 0.08 * mi; }
        else if (microType.current === M_BROWFLICK) p.browHeight += 0.3 * mi;
        if (microProg.current > 1.0) microType.current = M_NONE;
      }

      // --- Perlin micro-movement ---
      const swayX = noise2D(t * 0.2, 0) * 0.015;
      const swayY = noise2D(0, t * 0.18) * 0.010;
      const gazeNX = noise2D(t * 0.35, 10) * 0.2;
      const gazeNY = noise2D(10, t * 0.3) * 0.15;
      const breathY = Math.sin(t * (1.2 + arousalRef.current * 2.0)) * 0.006;

      // --- Geometry ---
      const F = Math.min(w, h) * 0.82; // face size in pixels
      const ox = (w - F) / 2;
      const oy = (h - F) / 2;
      // Convert normalized coord to screen pixel
      const X = (n: number) => ox + (n + swayX) * F;
      const Y = (n: number) => oy + (n + swayY + breathY) * F;

      const cx = 0.5, cy = 0.48;
      const tilt = p.headTilt * 0.1; // radians — 2x bigger than before

      // Skin color
      const ss = c.skinTintStrength;
      const sr = SKIN[0] * (1 - ss) + c.skinTint[0] * ss;
      const sg = SKIN[1] * (1 - ss) + c.skinTint[1] * ss;
      const sb = SKIN[2] * (1 - ss) + c.skinTint[2] * ss;

      // === CLEAR ===
      ctx!.fillStyle = "#0a0a14";
      ctx!.fillRect(0, 0, w, h);

      // === GLOW ===
      const gx = X(cx), gy = Y(cy), gR = F * 0.52;
      const gg = ctx!.createRadialGradient(gx, gy, 0, gx, gy, gR);
      const gc = c.glowColor;
      gg.addColorStop(0, `rgba(${gc[0] | 0},${gc[1] | 0},${gc[2] | 0},${(c.glowIntensity * 0.35).toFixed(2)})`);
      gg.addColorStop(1, `rgba(${gc[0] | 0},${gc[1] | 0},${gc[2] | 0},0)`);
      ctx!.fillStyle = gg;
      ctx!.beginPath(); ctx!.arc(gx, gy, gR, 0, Math.PI * 2); ctx!.fill();

      // === HEAD ===
      const hW = F * 0.30, hH = F * 0.37;
      const hx = X(cx), hy = Y(cy);
      ctx!.save(); ctx!.translate(hx, hy); ctx!.rotate(tilt);
      ctx!.beginPath(); ctx!.ellipse(0, 0, hW, hH, 0, 0, Math.PI * 2);
      ctx!.fillStyle = `rgb(${sr | 0},${sg | 0},${sb | 0})`;
      ctx!.fill(); ctx!.restore();

      // === CHEEK BLUSH ===
      if (p.blushIntensity > 0.02) {
        const ba = p.blushIntensity * 0.45;
        const bc = c.blushColor;
        for (const side of [-1, 1]) {
          const bx = X(cx + side * 0.135), by = Y(0.54);
          const br = F * 0.07;
          const bg = ctx!.createRadialGradient(bx, by, 0, bx, by, br);
          bg.addColorStop(0, `rgba(${bc[0] | 0},${bc[1] | 0},${bc[2] | 0},${ba.toFixed(2)})`);
          bg.addColorStop(1, `rgba(${bc[0] | 0},${bc[1] | 0},${bc[2] | 0},0)`);
          ctx!.fillStyle = bg; ctx!.beginPath(); ctx!.arc(bx, by, br, 0, Math.PI * 2); ctx!.fill();
        }
      }

      // === CHEEK RAISE LINES (nasolabial folds — smile muscles) ===
      if (p.cheekRaise > 0.15) {
        const cra = (p.cheekRaise - 0.15) * 0.6;
        ctx!.strokeStyle = `rgba(170,150,135,${cra.toFixed(2)})`;
        ctx!.lineWidth = 1.2;
        for (const side of [-1, 1]) {
          ctx!.beginPath();
          ctx!.moveTo(X(cx + side * 0.06), Y(0.48));
          ctx!.quadraticCurveTo(X(cx + side * 0.10), Y(0.55), X(cx + side * 0.09), Y(0.61));
          ctx!.stroke();
        }
      }

      // === FOREHEAD LINES ===
      if (p.foreheadTension > 0.12) {
        const fa = (p.foreheadTension - 0.12) * 0.7;
        ctx!.strokeStyle = `rgba(170,150,135,${fa.toFixed(2)})`;
        ctx!.lineWidth = 1.2;
        const nL = p.foreheadTension > 0.5 ? 3 : 2;
        for (let i = 0; i < nL; i++) {
          const ly = 0.27 + i * 0.022;
          ctx!.beginPath();
          ctx!.moveTo(X(cx - 0.09), Y(ly));
          ctx!.quadraticCurveTo(X(cx), Y(ly - 0.007), X(cx + 0.09), Y(ly));
          ctx!.stroke();
        }
      }

      // === NOSE WRINKLE (anger/disgust) ===
      if (p.noseWrinkle > 0.1) {
        const na = (p.noseWrinkle - 0.1) * 0.5;
        ctx!.strokeStyle = `rgba(160,140,125,${na.toFixed(2)})`;
        ctx!.lineWidth = 1.0;
        for (const side of [-1, 1]) {
          ctx!.beginPath();
          ctx!.moveTo(X(cx + side * 0.015), Y(0.49));
          ctx!.quadraticCurveTo(X(cx + side * 0.04), Y(0.50), X(cx + side * 0.05), Y(0.48));
          ctx!.stroke();
        }
      }

      // === EYES ===
      const eOpen = p.eyeOpenness * (1 - blinkVal.current);

      for (const side of [-1, 1]) {
        const ex = cx + side * 0.115, ey = 0.42;
        const ecx = X(ex), ecy = Y(ey);
        // LARGE eyes
        const eW = F * 0.085;
        const eH = F * 0.06 * Math.max(0.04, eOpen);
        const sqH = F * 0.02 * p.eyeSquint;

        ctx!.save(); ctx!.translate(ecx, ecy); ctx!.rotate(tilt);

        if (eOpen > 0.04) {
          // Eye white
          ctx!.beginPath();
          ctx!.moveTo(-eW, 0);
          ctx!.quadraticCurveTo(0, -eH, eW, 0);
          ctx!.quadraticCurveTo(0, eH * 0.65 - sqH, -eW, 0);
          ctx!.fillStyle = "#f4efea";
          ctx!.fill();

          // Iris
          const iR = Math.min(eH * 0.6, F * 0.032);
          if (iR > 2) {
            const gx2 = (p.pupilOffsetX + gazeNX) * eW * 0.35;
            const gy2 = (p.pupilOffsetY + gazeNY) * eH * 0.25;
            const ec = c.eyeColor;

            // Iris with gradient
            const ig = ctx!.createRadialGradient(gx2, gy2, iR * 0.15, gx2, gy2, iR);
            ig.addColorStop(0, `rgb(${Math.min(255, ec[0] + 50) | 0},${Math.min(255, ec[1] + 50) | 0},${Math.min(255, ec[2] + 50) | 0})`);
            ig.addColorStop(0.5, `rgb(${ec[0] | 0},${ec[1] | 0},${ec[2] | 0})`);
            ig.addColorStop(1, `rgb(${(ec[0] * 0.5) | 0},${(ec[1] * 0.5) | 0},${(ec[2] * 0.5) | 0})`);
            ctx!.beginPath(); ctx!.arc(gx2, gy2, iR, 0, Math.PI * 2);
            ctx!.fillStyle = ig; ctx!.fill();

            // Pupil
            const pR = iR * p.pupilSize;
            ctx!.beginPath(); ctx!.arc(gx2, gy2, pR, 0, Math.PI * 2);
            ctx!.fillStyle = "rgb(12,12,22)"; ctx!.fill();

            // Specular
            ctx!.beginPath(); ctx!.arc(gx2 - iR * 0.3, gy2 - iR * 0.3, Math.max(2, pR * 0.4), 0, Math.PI * 2);
            ctx!.fillStyle = "rgba(255,255,255,0.85)"; ctx!.fill();
            ctx!.beginPath(); ctx!.arc(gx2 + iR * 0.15, gy2 + iR * 0.2, Math.max(1, pR * 0.2), 0, Math.PI * 2);
            ctx!.fillStyle = "rgba(255,255,255,0.3)"; ctx!.fill();
          }

          // Upper lid
          ctx!.beginPath(); ctx!.moveTo(-eW, 0);
          ctx!.quadraticCurveTo(0, -eH, eW, 0);
          ctx!.strokeStyle = "rgba(60,48,40,0.65)"; ctx!.lineWidth = 2.0; ctx!.stroke();

          // Lower lid
          ctx!.beginPath(); ctx!.moveTo(-eW * 0.6, eH * 0.25 - sqH);
          ctx!.quadraticCurveTo(0, eH * 0.65 - sqH, eW * 0.6, eH * 0.25 - sqH);
          ctx!.strokeStyle = "rgba(60,48,40,0.12)"; ctx!.lineWidth = 0.8; ctx!.stroke();

          // Eye squint crinkle lines (crow's feet)
          if (p.eyeSquint > 0.3) {
            const cra = (p.eyeSquint - 0.3) * 0.5;
            ctx!.strokeStyle = `rgba(150,130,115,${cra.toFixed(2)})`;
            ctx!.lineWidth = 0.8;
            for (let i = 0; i < 2; i++) {
              const ly = -eH * 0.3 + i * eH * 0.4;
              ctx!.beginPath();
              ctx!.moveTo(eW * 0.9, ly);
              ctx!.lineTo(eW * 1.15, ly + (i === 0 ? -2 : 2));
              ctx!.stroke();
            }
          }
        } else {
          // Closed
          ctx!.beginPath(); ctx!.moveTo(-eW, 0);
          ctx!.quadraticCurveTo(0, F * 0.006, eW, 0);
          ctx!.strokeStyle = "rgba(60,48,40,0.7)"; ctx!.lineWidth = 2.0; ctx!.stroke();
        }
        ctx!.restore();
      }

      // === EYEBROWS — DRAMATIC ===
      ctx!.lineCap = "round";
      for (const side of [-1, 1]) {
        const bx = cx + side * 0.115, by = 0.33;
        // LARGE multipliers
        const bh = p.browHeight * 0.09;
        const ba = p.browAngle * 0.06 * side;
        const bc = p.browCurvature * 0.035;

        const inner = [X(bx - side * 0.07), Y(by - bh + ba)] as const;
        const mid = [X(bx), Y(by - bh - bc)] as const;
        const outer = [X(bx + side * 0.075), Y(by - bh * 0.5 - ba * 0.3)] as const;

        ctx!.beginPath();
        ctx!.moveTo(inner[0], inner[1]);
        ctx!.quadraticCurveTo(mid[0], mid[1], outer[0], outer[1]);
        ctx!.strokeStyle = SKIN_DARK(sr, sg, sb);
        ctx!.lineWidth = F * 0.012;
        ctx!.stroke();

        // Brow furrow line (angry V between brows)
        if (p.browAngle < -0.3) {
          const fa = Math.abs(p.browAngle + 0.3) * 0.6;
          ctx!.strokeStyle = `rgba(150,130,115,${fa.toFixed(2)})`;
          ctx!.lineWidth = 1.0;
          ctx!.beginPath();
          ctx!.moveTo(X(cx + side * 0.04), Y(0.32 - bh));
          ctx!.lineTo(X(cx + side * 0.025), Y(0.35 - bh));
          ctx!.stroke();
        }
      }

      // === NOSE ===
      const nx = X(cx), ny = Y(0.52);
      ctx!.beginPath();
      ctx!.moveTo(nx, ny - F * 0.025);
      ctx!.quadraticCurveTo(nx + F * 0.016, ny, nx, ny + F * 0.008);
      ctx!.strokeStyle = `rgba(${(sr * 0.65) | 0},${(sg * 0.65) | 0},${(sb * 0.65) | 0},0.3)`;
      ctx!.lineWidth = 1.3; ctx!.stroke();

      // === MOUTH — DRAMATIC ===
      const mY = 0.62;
      const mc = p.mouthCurve;
      const mo = Math.max(p.mouthOpenness, p.jawDrop * 0.8);
      const mt = p.mouthTension;
      const mw = p.mouthWidth * 0.12; // WIDE

      // Corner displacement — BIG
      const cDip = mc * -0.08; // ±8% of face size = ±20px on 250px face
      const mlx = X(cx - mw), mly = Y(mY + cDip);
      const mrx = X(cx + mw), mry = Y(mY + cDip);

      // Lip color
      const lr = LIP[0] * (1 - ss * 0.5) + c.skinTint[0] * ss * 0.5;
      const lg = LIP[1] * (1 - ss * 0.5) + c.skinTint[1] * ss * 0.5;
      const lb = LIP[2] * (1 - ss * 0.5) + c.skinTint[2] * ss * 0.5;

      // Upper lip control point
      const uCPy = mY - mc * 0.05 - mt * 0.008;
      const umx = X(cx), umy = Y(uCPy);

      // Mouth interior (dark)
      const openPx = mo * F * 0.14 * (1 - mt * 0.4); // VERY open
      if (mo > 0.03 && openPx > 2) {
        const lmy = Y(mY + openPx / F + Math.abs(mc) * 0.015);
        ctx!.beginPath();
        ctx!.moveTo(mlx, mly);
        ctx!.quadraticCurveTo(umx, umy, mrx, mry);
        ctx!.quadraticCurveTo(X(cx), lmy, mlx, mly);
        ctx!.fillStyle = "rgba(30,15,15,0.8)";
        ctx!.fill();
      }

      // Upper lip stroke
      ctx!.beginPath();
      ctx!.moveTo(mlx, mly);
      ctx!.quadraticCurveTo(umx, umy, mrx, mry);
      ctx!.strokeStyle = `rgb(${lr | 0},${lg | 0},${lb | 0})`;
      ctx!.lineWidth = 2.5; ctx!.lineCap = "round"; ctx!.stroke();

      // Lower lip
      if (mo > 0.03 && openPx > 2) {
        const lmy2 = Y(mY + openPx / F + Math.abs(mc) * 0.015);
        ctx!.beginPath();
        ctx!.moveTo(mlx, mly);
        ctx!.quadraticCurveTo(X(cx), lmy2, mrx, mry);
        ctx!.strokeStyle = `rgba(${lr | 0},${lg | 0},${lb | 0},0.7)`;
        ctx!.lineWidth = 2.0; ctx!.stroke();
      }

      // Mouth tension lines (pressed lips)
      if (mt > 0.3) {
        const ta = (mt - 0.3) * 0.5;
        ctx!.strokeStyle = `rgba(160,135,125,${ta.toFixed(2)})`;
        ctx!.lineWidth = 0.8;
        for (const side of [-1, 1]) {
          ctx!.beginPath();
          ctx!.moveTo(X(cx + side * mw * 0.95), Y(mY + cDip));
          ctx!.lineTo(X(cx + side * (mw + 0.02)), Y(mY + cDip + 0.005));
          ctx!.stroke();
        }
      }

      // === HEAD OUTLINE ===
      ctx!.save(); ctx!.translate(hx, hy); ctx!.rotate(tilt);
      ctx!.beginPath(); ctx!.ellipse(0, 0, hW, hH, 0, 0, Math.PI * 2);
      ctx!.strokeStyle = "rgba(50,42,38,0.3)";
      ctx!.lineWidth = 1.5; ctx!.stroke(); ctx!.restore();

      animRef.current = requestAnimationFrame(frame);
    }

    animRef.current = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(animRef.current); ro.disconnect(); };
  }, [emotionalState, analyser, speaking]);

  return (
    <div className="emotion-avatar">
      <canvas ref={canvasRef} className="emotion-avatar__canvas" />
      {emotionalState && (
        <div className="emotion-avatar__label">
          {emotionalState.primary_emotion}
          {emotionalState.secondary_emotion && emotionalState.secondary_emotion !== emotionalState.primary_emotion
            ? ` / ${emotionalState.secondary_emotion}` : ""}
        </div>
      )}
    </div>
  );
}

function SKIN_DARK(r: number, g: number, b: number): string {
  return `rgb(${(r * 0.3) | 0},${(g * 0.28) | 0},${(b * 0.25) | 0})`;
}
