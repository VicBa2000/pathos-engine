/**
 * RealisticFace — Three.js WebGL face renderer using FaceCap.glb model.
 *
 * Model: 52 ARKit blend shapes, 4 meshes (head, left eye, right eye, teeth).
 * Maps FaceParams → ARKit morph target weights.
 * Same animation logic as PainterlyFace (blink, gestures, micro, lip sync, Perlin).
 */

import { useRef, useEffect, useState, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "meshoptimizer";
import { KTX2Loader } from "three/examples/jsm/loaders/KTX2Loader.js";
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

const BASE_SKIN: [number, number, number] = [215, 218, 222];

const G_NONE = 0, G_GLANCE_L = 1, G_GLANCE_R = 2, G_GLANCE_UP = 3,
  G_TILT = 4, G_SQUINT = 5, G_BROW = 6;
const M_NONE = 0, M_TWITCH = 1, M_LIPPRESS = 2, M_BROWFLICK = 3;

/* ------------------------------------------------------------------ */
/*  Morph target helper                                                */
/* ------------------------------------------------------------------ */

function setMorph(
  inf: number[],
  dict: Record<string, number> | undefined,
  name: string,
  value: number,
): void {
  if (!dict || dict[name] === undefined) return;
  inf[dict[name]] = Math.max(0, Math.min(1, value));
}

/* ------------------------------------------------------------------ */
/*  FaceScene                                                          */
/* ------------------------------------------------------------------ */

function FaceScene({ emotionalState, analyser, speaking }: Props) {
  const { gl } = useThree();
  const [scene, setScene] = useState<THREE.Group | null>(null);

  // Load model with KTX2 + meshopt decoders, auto-center and scale
  useEffect(() => {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const ktx2 = new KTX2Loader();
    ktx2.setTranscoderPath("/basis/");
    ktx2.detectSupport(gl);
    loader.setKTX2Loader(ktx2);
    loader.load(
      "/models/FaceCap.glb",
      (gltf) => {
        const s = gltf.scene;
        // Force world matrix update so bounding box is correct
        s.updateMatrixWorld(true);
        // Compute bounding box of the entire model
        const box = new THREE.Box3().setFromObject(s);
        const center = new THREE.Vector3();
        const size = new THREE.Vector3();
        box.getCenter(center);
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        // Normalize to ~0.5 units tall and center at origin
        const scale = 0.5 / maxDim;
        s.scale.multiplyScalar(scale);
        // Center the model at origin
        s.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
        console.log("[RealisticFace] Model loaded. Size:", size, "Center:", center, "Scale:", scale.toFixed(4));
        // Debug: log materials
        s.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const m = child as THREE.Mesh;
            const mat = m.material as THREE.Material & { map?: THREE.Texture | null; color?: THREE.Color };
            console.log("[RealisticFace] Mesh:", m.parent?.name || "(root)",
              "material:", mat.type, "hasMap:", !!mat.map,
              "color:", mat.color?.getHexString?.() || "none",
              "morphTargets:", m.geometry.morphAttributes?.position?.length || 0);
          }
        });
        setScene(s);
      },
      undefined,
      (err) => console.error("[RealisticFace] Load error:", err),
    );
    return () => { ktx2.dispose(); };
  }, [gl]);

  // Refs for scene objects
  const rootRef = useRef<THREE.Group>(null);
  const headMesh = useRef<THREE.Mesh | null>(null);
  const morphDict = useRef<Record<string, number> | undefined>(undefined);
  const leftEyeGrp = useRef<THREE.Object3D | null>(null);
  const rightEyeGrp = useRef<THREE.Object3D | null>(null);
  const rimLight = useRef<THREE.PointLight>(null);
  const skinMatRef = useRef<THREE.MeshStandardMaterial | null>(null);

  // Animation state
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
  const energyRef = useRef(0.5);
  const arousalRef = useRef(0.3);
  const timeRef = useRef(0);
  const analyserRef = useRef(analyser);
  analyserRef.current = analyser;
  const speakingRef = useRef(speaking);
  speakingRef.current = speaking;

  // Setup: find meshes, apply materials
  useEffect(() => {
    if (!scene) return;

    scene.traverse((child) => {
      if (!(child as THREE.Mesh).isMesh) return;
      const mesh = child as THREE.Mesh;

      // Head mesh — the one with morph targets
      if (mesh.geometry.morphAttributes?.position) {
        headMesh.current = mesh;
        morphDict.current = mesh.morphTargetDictionary ?? undefined;
        // Keep the original material — it already has the KTX2 texture
        const mat = mesh.material as THREE.MeshStandardMaterial;
        mat.roughness = 0.55;
        mat.metalness = 0.02;
        skinMatRef.current = mat;
      }

      // Eyes — just tweak, don't replace (they have textures too)
      const parentName = mesh.parent?.name || "";
      if (parentName === "eyeLeft" || parentName === "eyeRight") {
        const mat = mesh.material as THREE.MeshStandardMaterial;
        mat.roughness = 0.08;
        mat.metalness = 0.05;
      }

      // Teeth
      if (parentName === "teeth") {
        const mat = mesh.material as THREE.MeshStandardMaterial;
        mat.roughness = 0.35;
        mat.metalness = 0.0;
      }
    });

    leftEyeGrp.current = scene.getObjectByName("grp_eyeLeft") ?? null;
    rightEyeGrp.current = scene.getObjectByName("grp_eyeRight") ?? null;
  }, [scene]);

  // Update targets
  useEffect(() => {
    if (emotionalState) {
      tgtP.current = deriveFaceParams(emotionalState);
      tgtC.current = deriveFaceColors(emotionalState);
      energyRef.current = emotionalState.body_state.energy;
      arousalRef.current = emotionalState.arousal;
    } else {
      tgtP.current = { ...DEFAULT_FACE_PARAMS };
      tgtC.current = { ...DEFAULT_FACE_COLORS };
    }
  }, [emotionalState]);

  // Per-frame
  useFrame(() => {
    const t = timeRef.current;
    timeRef.current += 0.008;
    const dt = 0.008;
    const an = analyserRef.current;
    const sp = speakingRef.current;

    curP.current = lerpFaceParams(curP.current, tgtP.current, 0.06);
    curC.current = lerpFaceColors(curC.current, tgtC.current, 0.06);
    const p: FaceParams = { ...curP.current };
    const c = curC.current;

    // Lip sync
    let vol = 0;
    if (an && sp) {
      if (!freqBuf.current || freqBuf.current.length !== an.frequencyBinCount)
        freqBuf.current = new Uint8Array(an.frequencyBinCount);
      an.getByteFrequencyData(freqBuf.current);
      let sum = 0;
      for (let i = 0; i < freqBuf.current.length; i++) sum += freqBuf.current[i];
      vol = sum / (freqBuf.current.length * 255);
    }
    const tgtVol = sp ? (an ? vol : 0.25 + Math.sin(t * 9) * 0.2) : 0;
    smoothVol.current += (tgtVol - smoothVol.current) * 0.3;
    if (smoothVol.current > 0.01) {
      const lv = smoothVol.current;
      p.mouthOpenness = Math.max(p.mouthOpenness, lv * 3.5);
      p.jawDrop = Math.max(p.jawDrop, lv * 2.5);
    }

    // Blink
    blinkTimer.current += dt;
    const blinkInt = 2.8 + (1 - energyRef.current) * 3.5 + noise2D(t * 0.08, 50) * 1.5;
    if (blinkPhase.current === 0 && blinkTimer.current > blinkInt) { blinkPhase.current = 1; blinkTimer.current = 0; }
    if (blinkPhase.current === 1) { blinkVal.current += 0.22; if (blinkVal.current >= 1) { blinkVal.current = 1; blinkPhase.current = 2; } }
    else if (blinkPhase.current === 2) { blinkVal.current -= 0.09; if (blinkVal.current <= 0) { blinkVal.current = 0; blinkPhase.current = 0; } }

    // Gestures
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

    // Micro-expressions
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

    // Perlin
    const swayX = noise2D(t * 0.15, 0) * 0.008;
    const swayY = noise2D(0, t * 0.12) * 0.006;
    const gazeNX = noise2D(t * 0.25, 10) * 0.10;
    const gazeNY = noise2D(10, t * 0.22) * 0.08;
    const breathY = Math.sin(t * (0.9 + arousalRef.current * 1.5)) * 0.003;

    /* ==================== Apply to scene ==================== */

    // Root group: sway + tilt
    if (rootRef.current) {
      rootRef.current.position.set(swayX, swayY + breathY, 0);
      rootRef.current.rotation.set(0, 0, p.headTilt * 0.07);
    }

    // Skin color
    if (skinMatRef.current) {
      const ss = c.skinTintStrength * 0.3;
      const sr = (BASE_SKIN[0] * (1 - ss) + c.skinTint[0] * ss) / 255;
      const sg = (BASE_SKIN[1] * (1 - ss) + c.skinTint[1] * ss) / 255;
      const sb = (BASE_SKIN[2] * (1 - ss) + c.skinTint[2] * ss) / 255;
      skinMatRef.current.color.setRGB(sr, sg, sb);
    }

    // Rim light
    if (rimLight.current) {
      const gc = c.glowColor;
      rimLight.current.color.setRGB(gc[0] / 255, gc[1] / 255, gc[2] / 255);
      rimLight.current.intensity = 0.4 + c.glowIntensity * 2.0;
    }

    // --- Morph targets ---
    const eOpen = p.eyeOpenness * (1 - blinkVal.current);
    const dict = morphDict.current;

    if (headMesh.current?.morphTargetInfluences && dict) {
      const inf = headMesh.current.morphTargetInfluences;
      // Reset
      inf.fill(0);

      // === MOUTH ===
      const smile = Math.max(0, p.mouthCurve);
      const frown = Math.max(0, -p.mouthCurve);
      setMorph(inf, dict, "mouthSmile_L", smile);
      setMorph(inf, dict, "mouthSmile_R", smile);
      setMorph(inf, dict, "mouthFrown_L", frown);
      setMorph(inf, dict, "mouthFrown_R", frown);
      setMorph(inf, dict, "jawOpen", Math.max(p.mouthOpenness, p.jawDrop * 0.8));
      setMorph(inf, dict, "mouthPress_L", p.mouthTension);
      setMorph(inf, dict, "mouthPress_R", p.mouthTension);
      if (p.mouthWidth > 1) {
        const stretch = (p.mouthWidth - 1) * 2;
        setMorph(inf, dict, "mouthStretch_L", stretch);
        setMorph(inf, dict, "mouthStretch_R", stretch);
      }
      // Dimples with smile
      setMorph(inf, dict, "mouthDimple_L", smile * 0.4);
      setMorph(inf, dict, "mouthDimple_R", smile * 0.4);

      // === EYES ===
      const blink = Math.max(0, 1 - eOpen);
      setMorph(inf, dict, "eyeBlink_L", blink);
      setMorph(inf, dict, "eyeBlink_R", blink);
      if (eOpen > 0.7) {
        const wide = (eOpen - 0.7) * 3.3;
        setMorph(inf, dict, "eyeWide_L", wide);
        setMorph(inf, dict, "eyeWide_R", wide);
      }
      setMorph(inf, dict, "eyeSquint_L", p.eyeSquint);
      setMorph(inf, dict, "eyeSquint_R", p.eyeSquint);

      // === BROWS ===
      if (p.browHeight > 0) {
        setMorph(inf, dict, "browOuterUp_L", p.browHeight);
        setMorph(inf, dict, "browOuterUp_R", p.browHeight);
      } else {
        setMorph(inf, dict, "browDown_L", -p.browHeight);
        setMorph(inf, dict, "browDown_R", -p.browHeight);
      }
      if (p.browAngle > 0) {
        setMorph(inf, dict, "browInnerUp", p.browAngle);
      } else {
        // Angry brow: inner down = browDown + browInnerUp inverted
        setMorph(inf, dict, "browDown_L", Math.max(inf[dict["browDown_L"]] || 0, -p.browAngle * 0.6));
        setMorph(inf, dict, "browDown_R", Math.max(inf[dict["browDown_R"]] || 0, -p.browAngle * 0.6));
      }

      // === NOSE ===
      setMorph(inf, dict, "noseSneer_L", p.noseWrinkle);
      setMorph(inf, dict, "noseSneer_R", p.noseWrinkle);

      // === CHEEKS ===
      setMorph(inf, dict, "cheekSquint_L", Math.max(0, p.cheekRaise));
      setMorph(inf, dict, "cheekSquint_R", Math.max(0, p.cheekRaise));

      // === EYE GAZE (eyelid follow via morph targets) ===
      const gx = (p.pupilOffsetX + gazeNX) * 0.3;
      const gy = (p.pupilOffsetY + gazeNY) * 0.3;
      if (gy < 0) { setMorph(inf, dict, "eyeLookUp_L", -gy); setMorph(inf, dict, "eyeLookUp_R", -gy); }
      if (gy > 0) { setMorph(inf, dict, "eyeLookDown_L", gy); setMorph(inf, dict, "eyeLookDown_R", gy); }
      if (gx > 0) { setMorph(inf, dict, "eyeLookIn_L", gx); setMorph(inf, dict, "eyeLookOut_R", gx); }
      if (gx < 0) { setMorph(inf, dict, "eyeLookOut_L", -gx); setMorph(inf, dict, "eyeLookIn_R", -gx); }
    }

    // Eye rotation (actual eyeball movement)
    const gazeRotX = -(p.pupilOffsetY + gazeNY) * 0.12;
    const gazeRotY = (p.pupilOffsetX + gazeNX) * 0.15;
    if (leftEyeGrp.current) leftEyeGrp.current.rotation.set(gazeRotX, gazeRotY, 0);
    if (rightEyeGrp.current) rightEyeGrp.current.rotation.set(gazeRotX, gazeRotY, 0);
  });

  return (
    <>
      <color attach="background" args={["#0a0a14"]} />

      {/* Lighting */}
      <ambientLight intensity={0.45} color="#c0c8d4" />
      <directionalLight position={[1.5, 2.5, 4]} intensity={1.6} color="#fff0e0" />
      <directionalLight position={[-2, 1, 2]} intensity={0.3} color="#90a0c0" />
      <pointLight ref={rimLight} position={[2, 0, -2]} intensity={0.8} />

      {/* Model — auto-centered and scaled in load callback */}
      {scene && (
        <group ref={rootRef}>
          <primitive object={scene} />
        </group>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  Exported wrapper                                                   */
/* ------------------------------------------------------------------ */

export function RealisticFace({ emotionalState, analyser, speaking }: Props) {
  return (
    <div style={{ width: "100%", height: "100%", background: "#0a0a14" }}>
      <Canvas
        camera={{ position: [0, 0.02, 2.5], fov: 18 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
        dpr={[1, 2]}
        onCreated={(state) => {
          state.gl.setClearColor("#0a0a14");
          state.scene.background = new THREE.Color("#0a0a14");
        }}
      >
        <FaceScene
          emotionalState={emotionalState}
          analyser={analyser}
          speaking={speaking}
        />
      </Canvas>
    </div>
  );
}

