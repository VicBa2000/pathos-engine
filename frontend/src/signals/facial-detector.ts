/**
 * Real-time facial expression detector using face-api.js.
 *
 * Manages webcam stream, face detection loop, and expression extraction.
 * Designed to feed into the Pathos external signals pipeline.
 *
 * Uses a 5-second temporal buffer with recency-weighted averaging to produce
 * stable expression readings. This filters out micro-expressions (40-200ms)
 * and frame-to-frame noise, capturing sustained macro-expressions (0.5-4s)
 * that reflect genuine emotional state (Ekman 1992, Dimberg 1990).
 */

import * as faceapi from "@vladmandic/face-api";

export interface FacialExpressions {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
}

export interface DetectionResult {
  expressions: FacialExpressions;
  faceDetected: boolean;
  confidence: number;
}

export type OnDetection = (result: DetectionResult) => void;

const MODEL_URL = "/models";
const DETECTION_INTERVAL_MS = 500; // ~2 detections per second
const MIN_FACE_SCORE = 0.4;

/**
 * Temporal buffer window in milliseconds.
 * 5 seconds balances reactivity vs stability:
 *   - Filters micro-expressions (40-200ms) and noisy frames
 *   - Captures macro-expressions (0.5-4s, Ekman/Friesen)
 *   - Matches facial EMG stabilization time (~2-3s, Dimberg 1990)
 *   - Compatible with DynAffect emotional inertia model (Kuppens 2010)
 */
const BUFFER_WINDOW_MS = 5000;

/** Recency decay — how much more recent detections weigh vs older ones.
 * weight = e^(-age_ms * RECENCY_DECAY)
 * At 5s old: weight ≈ 0.37 (still contributes but less than recent frames)
 */
const RECENCY_DECAY = 0.0002;

const EXPRESSION_KEYS: (keyof FacialExpressions)[] = [
  "neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised",
];

interface BufferEntry {
  expressions: FacialExpressions;
  confidence: number;
  timestamp: number;
}

let modelsLoaded = false;
let modelsLoading = false;

/** Load face-api.js models (tiny face detector + expression net). Idempotent. */
export async function loadModels(): Promise<void> {
  if (modelsLoaded) return;
  if (modelsLoading) {
    // Wait for in-flight load
    while (modelsLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    return;
  }

  modelsLoading = true;
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
    modelsLoaded = true;
  } finally {
    modelsLoading = false;
  }
}

/**
 * FacialDetector — manages webcam capture and continuous face expression detection.
 *
 * Maintains a 5-second temporal buffer of detections. The consolidated reading
 * (via `getConsolidated()`) returns recency-weighted average expressions,
 * filtering noise and micro-expressions to reflect sustained emotional state.
 *
 * Usage:
 *   const detector = new FacialDetector();
 *   const video = await detector.start(onDetection);
 *   const stable = detector.getConsolidated(); // 5s weighted average
 *   detector.stop();
 */
export class FacialDetector {
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private running = false;
  private onDetection: OnDetection | null = null;

  /** Circular buffer of recent detections (5s window). */
  private buffer: BufferEntry[] = [];

  /** Start webcam capture and detection loop. Returns the video element for preview. */
  async start(onDetection: OnDetection): Promise<HTMLVideoElement> {
    if (this.running) {
      throw new Error("Detector already running");
    }

    this.onDetection = onDetection;
    this.buffer = [];

    // Load models first
    await loadModels();

    // Request webcam — retry once if the device is busy (common on Windows
    // when the camera was recently released by another process).
    let lastErr: unknown;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 320 },
            height: { ideal: 240 },
            facingMode: "user",
            frameRate: { ideal: 15 },
          },
          audio: false,
        });
        lastErr = null;
        break;
      } catch (err) {
        lastErr = err;
        // Only retry on device-busy / abort errors — not on permission denied
        if (
          err instanceof DOMException &&
          (err.name === "NotAllowedError" || err.name === "NotFoundError")
        ) {
          throw err;
        }
        if (attempt === 0) {
          await new Promise((r) => setTimeout(r, 1500));
        }
      }
    }
    if (lastErr) throw lastErr;

    // Create hidden video element for processing
    this.video = document.createElement("video");
    this.video.srcObject = this.stream!;
    this.video.setAttribute("playsinline", "true");
    this.video.muted = true;

    // Wait for video to actually start with a timeout — on Windows the camera
    // can hang indefinitely after getUserMedia succeeds but the device is stuck.
    try {
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error(
            "Camera took too long to start. Try closing other apps using the camera, or restart the browser."
          ));
        }, 10000);
        this.video!.onloadeddata = () => { clearTimeout(timeout); resolve(); };
        this.video!.play().catch((err) => { clearTimeout(timeout); reject(err); });
      });
    } catch (err) {
      // Release the camera if video playback failed
      this.stop();
      throw err;
    }

    // Start detection loop
    this.running = true;
    this.intervalId = setInterval(() => this.detect(), DETECTION_INTERVAL_MS);

    return this.video;
  }

  /** Stop detection and release webcam. */
  stop(): void {
    this.running = false;

    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }

    if (this.video) {
      this.video.srcObject = null;
      this.video = null;
    }

    this.onDetection = null;
    this.buffer = [];
  }

  get isRunning(): boolean {
    return this.running;
  }

  /**
   * Get consolidated expressions from the 5-second temporal buffer.
   *
   * Uses recency-weighted averaging: recent detections contribute more
   * than older ones, providing a smooth, stable reading that reflects
   * the user's sustained emotional expression rather than frame noise.
   *
   * Returns null if the buffer has no valid detections.
   */
  getConsolidated(): DetectionResult | null {
    const now = Date.now();
    const cutoff = now - BUFFER_WINDOW_MS;

    // Filter to window and only face-detected entries
    const valid = this.buffer.filter(
      (e) => e.timestamp >= cutoff && e.confidence > 0,
    );

    if (valid.length === 0) return null;

    // Recency-weighted average
    const weighted: FacialExpressions = {
      neutral: 0, happy: 0, sad: 0, angry: 0,
      fearful: 0, disgusted: 0, surprised: 0,
    };
    let totalWeight = 0;
    let totalConfidence = 0;

    for (const entry of valid) {
      const age = now - entry.timestamp;
      const recencyWeight = Math.exp(-age * RECENCY_DECAY);
      const w = recencyWeight * entry.confidence;
      totalWeight += w;
      totalConfidence += entry.confidence;

      for (const key of EXPRESSION_KEYS) {
        weighted[key] += entry.expressions[key] * w;
      }
    }

    if (totalWeight < 1e-6) return null;

    // Normalize
    for (const key of EXPRESSION_KEYS) {
      weighted[key] /= totalWeight;
    }

    return {
      expressions: weighted,
      faceDetected: true,
      confidence: totalConfidence / valid.length,
    };
  }

  /** Number of valid detections currently in the buffer window. */
  get bufferSize(): number {
    const cutoff = Date.now() - BUFFER_WINDOW_MS;
    return this.buffer.filter((e) => e.timestamp >= cutoff && e.confidence > 0).length;
  }

  /** Single detection pass. */
  private async detect(): Promise<void> {
    if (!this.running || !this.video || !this.onDetection) return;

    try {
      const detection = await faceapi
        .detectSingleFace(this.video, new faceapi.TinyFaceDetectorOptions({
          inputSize: 224,
          scoreThreshold: MIN_FACE_SCORE,
        }))
        .withFaceExpressions();

      const now = Date.now();

      if (detection) {
        const expr = detection.expressions;
        const result: DetectionResult = {
          expressions: {
            neutral: expr.neutral,
            happy: expr.happy,
            sad: expr.sad,
            angry: expr.angry,
            fearful: expr.fearful,
            disgusted: expr.disgusted,
            surprised: expr.surprised,
          },
          faceDetected: true,
          confidence: detection.detection.score,
        };

        // Add to buffer
        this.buffer.push({
          expressions: result.expressions,
          confidence: result.confidence,
          timestamp: now,
        });

        // Prune old entries beyond the window
        const cutoff = now - BUFFER_WINDOW_MS;
        this.buffer = this.buffer.filter((e) => e.timestamp >= cutoff);

        this.onDetection(result);
      } else {
        this.onDetection({
          expressions: { neutral: 1, happy: 0, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0 },
          faceDetected: false,
          confidence: 0,
        });
      }
    } catch {
      // Detection failed silently — will retry next interval
    }
  }
}
