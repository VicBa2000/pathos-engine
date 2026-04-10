/**
 * Shared color utility functions.
 * Extracted from VoiceOrb, used by EmotionGenesis and other canvas components.
 */

function expandHex(hex: string): string {
  hex = hex.replace("#", "");
  if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  return hex;
}

export function hexToRgba(hex: string, alpha: number): string {
  const h = expandHex(hex);
  const r = parseInt(h.slice(0, 2), 16) || 0;
  const g = parseInt(h.slice(2, 4), 16) || 0;
  const b = parseInt(h.slice(4, 6), 16) || 0;
  return `rgba(${r},${g},${b},${alpha})`;
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = expandHex(hex);
  return [
    parseInt(h.slice(0, 2), 16) || 0,
    parseInt(h.slice(2, 4), 16) || 0,
    parseInt(h.slice(4, 6), 16) || 0,
  ];
}

export function lighten(hex: string, pct: number): string {
  const h = expandHex(hex);
  let r = parseInt(h.slice(0, 2), 16) || 0;
  let g = parseInt(h.slice(2, 4), 16) || 0;
  let b = parseInt(h.slice(4, 6), 16) || 0;
  r = Math.min(255, r + Math.round((255 - r) * pct / 100));
  g = Math.min(255, g + Math.round((255 - g) * pct / 100));
  b = Math.min(255, b + Math.round((255 - b) * pct / 100));
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

export function darken(hex: string, pct: number): string {
  const h = expandHex(hex);
  let r = parseInt(h.slice(0, 2), 16) || 0;
  let g = parseInt(h.slice(2, 4), 16) || 0;
  let b = parseInt(h.slice(4, 6), 16) || 0;
  r = Math.max(0, Math.round(r * (1 - pct / 100)));
  g = Math.max(0, Math.round(g * (1 - pct / 100)));
  b = Math.max(0, Math.round(b * (1 - pct / 100)));
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** Lerp between two hex colors, t in [0,1] */
export function lerpColor(a: string, b: string, t: number): [number, number, number] {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  return [
    Math.round(r1 + (r2 - r1) * t),
    Math.round(g1 + (g2 - g1) * t),
    Math.round(b1 + (b2 - b1) * t),
  ];
}
