/**
 * Classic 2D/3D Perlin noise implementation.
 * Used by EmotionGenesis for organic particle movement and plasma fields.
 */

// Permutation table (doubled to avoid wrapping)
const perm: number[] = [];
const grad3 = [
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
];

// Initialize permutation table
const p = [
  151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
  140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
  247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
  57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
  74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
  60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
  65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
  200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
  52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
  207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
  119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
  129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
  218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
  81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
  184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
  222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];

for (let i = 0; i < 256; i++) {
  perm[i] = p[i];
  perm[256 + i] = p[i];
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

function dot3(g: number[], x: number, y: number, z: number): number {
  return g[0] * x + g[1] * y + g[2] * z;
}

/** 2D Perlin noise, returns value in [-1, 1] */
export function noise2D(x: number, y: number): number {
  const X = Math.floor(x) & 255;
  const Y = Math.floor(y) & 255;
  const xf = x - Math.floor(x);
  const yf = y - Math.floor(y);
  const u = fade(xf);
  const v = fade(yf);

  const aa = perm[perm[X] + Y];
  const ab = perm[perm[X] + Y + 1];
  const ba = perm[perm[X + 1] + Y];
  const bb = perm[perm[X + 1] + Y + 1];

  const g00 = grad3[aa % 12];
  const g10 = grad3[ba % 12];
  const g01 = grad3[ab % 12];
  const g11 = grad3[bb % 12];

  const n00 = g00[0] * xf + g00[1] * yf;
  const n10 = g10[0] * (xf - 1) + g10[1] * yf;
  const n01 = g01[0] * xf + g01[1] * (yf - 1);
  const n11 = g11[0] * (xf - 1) + g11[1] * (yf - 1);

  return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v);
}

/** 3D Perlin noise, returns value in [-1, 1] */
export function noise3D(x: number, y: number, z: number): number {
  const X = Math.floor(x) & 255;
  const Y = Math.floor(y) & 255;
  const Z = Math.floor(z) & 255;
  const xf = x - Math.floor(x);
  const yf = y - Math.floor(y);
  const zf = z - Math.floor(z);
  const u = fade(xf);
  const v = fade(yf);
  const w = fade(zf);

  const aaa = perm[perm[perm[X] + Y] + Z];
  const aba = perm[perm[perm[X] + Y + 1] + Z];
  const aab = perm[perm[perm[X] + Y] + Z + 1];
  const abb = perm[perm[perm[X] + Y + 1] + Z + 1];
  const baa = perm[perm[perm[X + 1] + Y] + Z];
  const bba = perm[perm[perm[X + 1] + Y + 1] + Z];
  const bab = perm[perm[perm[X + 1] + Y] + Z + 1];
  const bbb = perm[perm[perm[X + 1] + Y + 1] + Z + 1];

  return lerp(
    lerp(
      lerp(dot3(grad3[aaa % 12], xf, yf, zf), dot3(grad3[baa % 12], xf - 1, yf, zf), u),
      lerp(dot3(grad3[aba % 12], xf, yf - 1, zf), dot3(grad3[bba % 12], xf - 1, yf - 1, zf), u),
      v,
    ),
    lerp(
      lerp(dot3(grad3[aab % 12], xf, yf, zf - 1), dot3(grad3[bab % 12], xf - 1, yf, zf - 1), u),
      lerp(dot3(grad3[abb % 12], xf, yf - 1, zf - 1), dot3(grad3[bbb % 12], xf - 1, yf - 1, zf - 1), u),
      v,
    ),
    w,
  );
}

/** Fractal Brownian Motion — layered noise for richer organic patterns */
export function fbm(x: number, y: number, t: number, octaves = 4): number {
  let value = 0;
  let amplitude = 0.5;
  let frequency = 1;
  for (let i = 0; i < octaves; i++) {
    value += amplitude * noise3D(x * frequency, y * frequency, t);
    amplitude *= 0.5;
    frequency *= 2;
  }
  return value;
}
