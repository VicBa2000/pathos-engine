/**
 * Face parameter system for EmotionAvatar.
 * Maps EmotionalState → FaceParams via weighted emotional_stack blending.
 */

import type { EmotionalState } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import { hexToRgb } from "./colorUtils";

// --- Face parameter interface ---

export interface FaceParams {
  // Head
  headTilt: number;          // -1 to 1
  headNod: number;           // -1 to 1

  // Eyes
  eyeOpenness: number;       // 0 (closed) to 1 (wide open)
  pupilSize: number;         // 0.15 to 0.9
  pupilOffsetX: number;      // -1 to 1
  pupilOffsetY: number;      // -1 to 1
  eyeSquint: number;         // 0 to 1

  // Eyebrows
  browHeight: number;        // -1 (lowered) to 1 (raised)
  browAngle: number;         // -1 (inner down/angry) to 1 (inner up/worried)
  browCurvature: number;     // 0 (flat) to 1 (arched)

  // Mouth
  mouthCurve: number;        // -1 (deep frown) to 1 (big smile)
  mouthOpenness: number;     // 0 (closed) to 1 (wide open)
  mouthWidth: number;        // 0.5 (tight) to 1.5 (wide)
  mouthTension: number;      // 0 to 1 (lips pressed, grimace)

  // Face muscles
  noseWrinkle: number;       // 0 to 1 (disgust/anger nose scrunch)
  cheekRaise: number;        // 0 to 1 (smile cheek push up)
  jawDrop: number;           // 0 to 1 (jaw open, amplifies mouth open)

  // Cheeks
  blushIntensity: number;    // 0 to 1

  // Forehead
  foreheadTension: number;   // 0 to 1
}

export interface FaceColors {
  skinTint: [number, number, number];
  skinTintStrength: number;
  eyeColor: [number, number, number];
  glowColor: [number, number, number];
  glowIntensity: number;
  blushColor: [number, number, number];
}

export const DEFAULT_FACE_PARAMS: FaceParams = {
  headTilt: 0, headNod: 0,
  eyeOpenness: 0.5, pupilSize: 0.4, pupilOffsetX: 0, pupilOffsetY: 0, eyeSquint: 0,
  browHeight: 0, browAngle: 0, browCurvature: 0.3,
  mouthCurve: 0.03, mouthOpenness: 0, mouthWidth: 1.0, mouthTension: 0,
  noseWrinkle: 0, cheekRaise: 0, jawDrop: 0,
  blushIntensity: 0,
  foreheadTension: 0,
};

// --- Emotion → Face contributions ---
// Values are STRONG (0-1 scale maps to full expression range).
// Multiple emotions blend via weighted sum from emotional_stack.

type PartialFace = Partial<Record<keyof FaceParams, number>>;

const EMOTION_FACE: Record<string, PartialFace> = {
  joy: {
    mouthCurve: 1.0, cheekRaise: 0.8, eyeSquint: 0.4,
    eyeOpenness: 0.65, browHeight: 0.3, browCurvature: 0.6,
    blushIntensity: 0.2, mouthWidth: 1.15,
  },
  excitement: {
    mouthCurve: 0.85, eyeOpenness: 1.0, browHeight: 0.7,
    pupilSize: 0.8, mouthOpenness: 0.4, mouthWidth: 1.3,
    cheekRaise: 0.5, jawDrop: 0.3,
  },
  gratitude: {
    mouthCurve: 0.7, eyeSquint: 0.5, cheekRaise: 0.6,
    browHeight: 0.2, blushIntensity: 0.5, eyeOpenness: 0.55,
  },
  hope: {
    mouthCurve: 0.5, browHeight: 0.5, eyeOpenness: 0.7,
    pupilOffsetY: -0.4, browCurvature: 0.5,
  },
  contentment: {
    mouthCurve: 0.6, eyeSquint: 0.6, eyeOpenness: 0.45,
    cheekRaise: 0.4, browCurvature: 0.4,
  },
  relief: {
    mouthCurve: 0.4, eyeOpenness: 0.4, browHeight: -0.2,
    mouthOpenness: 0.2, jawDrop: 0.15,
  },
  anger: {
    mouthCurve: -0.7, browAngle: -1.0, browHeight: -0.6,
    eyeOpenness: 0.75, mouthTension: 0.9, noseWrinkle: 0.7,
    foreheadTension: 0.6, pupilSize: -0.2, eyeSquint: 0.3,
  },
  frustration: {
    mouthCurve: -0.5, browAngle: -0.7, browHeight: -0.4,
    mouthTension: 0.7, foreheadTension: 0.6, noseWrinkle: 0.3,
    eyeSquint: 0.2,
  },
  fear: {
    eyeOpenness: 1.0, browHeight: 0.9, browAngle: 0.8,
    pupilSize: 0.8, mouthOpenness: 0.5, mouthCurve: -0.3,
    foreheadTension: 0.8, jawDrop: 0.4,
  },
  anxiety: {
    eyeOpenness: 0.75, browAngle: 0.6, browHeight: 0.3,
    foreheadTension: 0.8, mouthTension: 0.6, pupilSize: -0.15,
    noseWrinkle: 0.15,
  },
  sadness: {
    mouthCurve: -0.9, browAngle: 0.7, browHeight: 0.4,
    eyeOpenness: 0.35, eyeSquint: 0.3, pupilOffsetY: 0.4,
    cheekRaise: -0.2, jawDrop: 0.1,
  },
  helplessness: {
    mouthCurve: -0.6, browAngle: 0.8, browHeight: 0.5,
    eyeOpenness: 0.3, headTilt: -0.4, jawDrop: 0.15,
  },
  disappointment: {
    mouthCurve: -0.7, browHeight: -0.15, eyeOpenness: 0.45,
    pupilOffsetY: 0.3, mouthTension: 0.3,
  },
  surprise: {
    eyeOpenness: 1.0, browHeight: 1.0, mouthOpenness: 0.9,
    pupilSize: 0.7, jawDrop: 0.8, browCurvature: 0.7,
  },
  alertness: {
    eyeOpenness: 0.9, browHeight: 0.5, pupilSize: 0.5,
    foreheadTension: 0.3, mouthTension: 0.2,
  },
  contemplation: {
    eyeOpenness: 0.5, browHeight: 0.15, browCurvature: 0.4,
    pupilOffsetX: 0.4, mouthCurve: 0.1, headTilt: 0.3,
  },
  indifference: {
    eyeOpenness: 0.4, browHeight: -0.2, mouthCurve: 0.0,
    mouthWidth: 0.8,
  },
  mixed: {
    browAngle: 0.3, foreheadTension: 0.4, mouthTension: 0.2,
  },
  neutral: {},
};

const BLENDABLE_KEYS: (keyof FaceParams)[] = [
  "headTilt", "headNod",
  "eyeOpenness", "pupilSize", "pupilOffsetX", "pupilOffsetY", "eyeSquint",
  "browHeight", "browAngle", "browCurvature",
  "mouthCurve", "mouthOpenness", "mouthWidth", "mouthTension",
  "noseWrinkle", "cheekRaise", "jawDrop",
  "blushIntensity", "foreheadTension",
];

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

/** Derive face parameters from emotional state via weighted stack blending. */
export function deriveFaceParams(state: EmotionalState): FaceParams {
  const p: FaceParams = { ...DEFAULT_FACE_PARAMS };

  // Weighted sum from ALL active emotions in the stack
  let totalWeight = 0;
  for (const [emotion, activation] of Object.entries(state.emotional_stack)) {
    if (activation < 0.02) continue;
    const contrib = EMOTION_FACE[emotion];
    if (!contrib) continue;

    const w = activation;
    totalWeight += w;

    for (const [key, value] of Object.entries(contrib)) {
      (p as Record<string, number>)[key] += (value as number) * w;
    }
  }

  // Normalize so multi-emotion stacks don't overshoot
  if (totalWeight > 1) {
    for (const key of BLENDABLE_KEYS) {
      p[key] /= totalWeight;
    }
  }

  // Dimensional overlays — additive on top of stack blend
  p.headTilt += (state.dominance - 0.5) * 0.5;
  p.headNod += (state.dominance - 0.5) * 0.3;
  p.pupilSize += state.arousal * 0.25;
  p.eyeOpenness += state.arousal * 0.25;

  // Body state — strong influence
  p.foreheadTension += state.body_state.tension * 0.5;
  p.mouthTension += state.body_state.tension * 0.4;
  p.noseWrinkle += state.body_state.tension * 0.2;
  p.blushIntensity += state.body_state.warmth * 0.5;
  p.eyeOpenness += (state.body_state.energy - 0.5) * 0.2;
  p.cheekRaise += state.body_state.warmth * 0.2;

  // Intensity scales delta from neutral — high floor so even subtle emotions show
  const intensityScale = 0.7 + state.intensity * 0.3;
  const defaults = DEFAULT_FACE_PARAMS;
  for (const key of BLENDABLE_KEYS) {
    const delta = p[key] - defaults[key];
    p[key] = defaults[key] + delta * intensityScale;
  }

  // Clamp all
  p.headTilt = clamp(p.headTilt, -1, 1);
  p.headNod = clamp(p.headNod, -1, 1);
  p.eyeOpenness = clamp(p.eyeOpenness, 0, 1);
  p.pupilSize = clamp(p.pupilSize, 0.15, 0.9);
  p.pupilOffsetX = clamp(p.pupilOffsetX, -1, 1);
  p.pupilOffsetY = clamp(p.pupilOffsetY, -1, 1);
  p.eyeSquint = clamp(p.eyeSquint, 0, 1);
  p.browHeight = clamp(p.browHeight, -1, 1);
  p.browAngle = clamp(p.browAngle, -1, 1);
  p.browCurvature = clamp(p.browCurvature, 0, 1);
  p.mouthCurve = clamp(p.mouthCurve, -1, 1);
  p.mouthOpenness = clamp(p.mouthOpenness, 0, 1);
  p.mouthWidth = clamp(p.mouthWidth, 0.5, 1.5);
  p.mouthTension = clamp(p.mouthTension, 0, 1);
  p.noseWrinkle = clamp(p.noseWrinkle, 0, 1);
  p.cheekRaise = clamp(p.cheekRaise, -0.5, 1);
  p.jawDrop = clamp(p.jawDrop, 0, 1);
  p.blushIntensity = clamp(p.blushIntensity, 0, 1);
  p.foreheadTension = clamp(p.foreheadTension, 0, 1);

  return p;
}

/** Derive face colors from emotional state. */
export function deriveFaceColors(state: EmotionalState): FaceColors {
  const primaryHex = EMOTION_COLORS[state.primary_emotion] || EMOTION_COLORS.neutral;
  const skinTint = hexToRgb(primaryHex);
  const skinTintStrength = 0.05 + state.intensity * 0.2;

  const eyeColor = hexToRgb(
    EMOTION_COLORS[state.primary_emotion] || EMOTION_COLORS.neutral,
  );

  const glowColor = skinTint;
  const glowIntensity = 0.15 + state.intensity * 0.45;

  const w = state.body_state.warmth;
  const blushColor: [number, number, number] = [
    Math.round(220 + w * 35),
    Math.round(120 - w * 30),
    Math.round(120 - w * 20),
  ];

  return { skinTint, skinTintStrength, eyeColor, glowColor, glowIntensity, blushColor };
}

/** Lerp a single FaceParams toward target. */
export function lerpFaceParams(current: FaceParams, target: FaceParams, factor: number): FaceParams {
  const out = { ...current };
  for (const key of BLENDABLE_KEYS) {
    out[key] += (target[key] - out[key]) * factor;
  }
  return out;
}

/** Lerp face colors. */
export function lerpFaceColors(current: FaceColors, target: FaceColors, factor: number): FaceColors {
  const lerpN = (a: number, b: number) => a + (b - a) * factor;
  const lerpRgb = (a: [number, number, number], b: [number, number, number]): [number, number, number] =>
    [lerpN(a[0], b[0]), lerpN(a[1], b[1]), lerpN(a[2], b[2])];

  return {
    skinTint: lerpRgb(current.skinTint, target.skinTint),
    skinTintStrength: lerpN(current.skinTintStrength, target.skinTintStrength),
    eyeColor: lerpRgb(current.eyeColor, target.eyeColor),
    glowColor: lerpRgb(current.glowColor, target.glowColor),
    glowIntensity: lerpN(current.glowIntensity, target.glowIntensity),
    blushColor: lerpRgb(current.blushColor, target.blushColor),
  };
}

export const DEFAULT_FACE_COLORS: FaceColors = {
  skinTint: [108, 122, 137],
  skinTintStrength: 0,
  eyeColor: [108, 140, 170],
  glowColor: [108, 122, 137],
  glowIntensity: 0.1,
  blushColor: [220, 120, 120],
};
