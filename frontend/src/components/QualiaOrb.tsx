import { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { EmotionalState } from "../types/emotion";
import "./QualiaOrb.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  emotionalState: EmotionalState | null;
  metaphor?: string | null;
}

interface OrbColors {
  core: THREE.Color;
  glow: THREE.Color;
  particle: THREE.Color;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map valence (-1..1) to hue (0..360). Negative = blue, positive = gold. */
function hueFromValence(v: number): number {
  // -1 → 220 (blue), 0 → 160 (teal), 1 → 40 (gold)
  return 220 - (v + 1) * 90;
}

function emotionToColors(state: EmotionalState): OrbColors {
  const hue = hueFromValence(state.valence);
  const saturation = 0.3 + state.arousal * 0.6; // 0.3..0.9
  const lightness = 0.25 + (state.valence + 1) * 0.15 + state.intensity * 0.15; // 0.25..0.55

  const core = new THREE.Color().setHSL(hue / 360, saturation, lightness);
  const glow = new THREE.Color().setHSL(hue / 360, saturation * 0.7, lightness * 1.4);
  const particle = new THREE.Color().setHSL(
    ((hue + 30) % 360) / 360,
    saturation * 0.8,
    lightness * 1.2,
  );
  return { core, glow, particle };
}

// ---------------------------------------------------------------------------
// Custom vertex shader: distortion driven by dominance/certainty
// ---------------------------------------------------------------------------

const orbVertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uDistortion;    // 0 = perfect sphere, 1 = very distorted
  uniform float uAmplitude;     // movement amplitude (intensity)
  uniform float uSpeed;         // movement speed (arousal)

  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vDisplacement;

  // Simplex-like noise
  vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
  }

  void main() {
    float t = uTime * uSpeed;

    // Multi-octave noise for organic distortion
    float n1 = snoise(position * 1.5 + t * 0.3) * 0.5;
    float n2 = snoise(position * 3.0 + t * 0.5) * 0.25;
    float n3 = snoise(position * 6.0 + t * 0.7) * 0.125;
    float noise = (n1 + n2 + n3) * uDistortion;

    // Breathing motion
    float breath = sin(t * 0.4) * 0.02 * uAmplitude;

    float displacement = noise * uAmplitude * 0.3 + breath;
    vec3 newPos = position + normal * displacement;

    vNormal = normalize(normalMatrix * normal);
    vPosition = (modelViewMatrix * vec4(newPos, 1.0)).xyz;
    vDisplacement = displacement;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
  }
`;

const orbFragmentShader = /* glsl */ `
  uniform vec3 uCoreColor;
  uniform vec3 uGlowColor;
  uniform float uTime;
  uniform float uArousal;

  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vDisplacement;

  void main() {
    // Fresnel for rim glow
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.5);

    // Base color mix: core → glow at edges
    vec3 color = mix(uCoreColor, uGlowColor, fresnel * 0.7);

    // Displacement-based iridescence
    float dispColor = vDisplacement * 3.0;
    color += vec3(dispColor * 0.1, dispColor * 0.05, -dispColor * 0.05);

    // Pulsing intensity
    float pulse = 0.85 + sin(uTime * uArousal * 2.0) * 0.15;
    color *= pulse;

    // Rim emission
    float rim = fresnel * (0.3 + uArousal * 0.5);
    color += uGlowColor * rim;

    // Alpha: solid core, semi-transparent edges
    float alpha = 0.85 + fresnel * 0.15;

    gl_FragColor = vec4(color, alpha);
  }
`;

// ---------------------------------------------------------------------------
// Orb mesh component
// ---------------------------------------------------------------------------

function OrbMesh({ state }: { state: EmotionalState }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetColors = useRef<OrbColors>(emotionToColors(state));
  const currentColors = useRef<OrbColors>(emotionToColors(state));

  // Smooth transition targets
  const targetDistortion = useRef(0.5);
  const targetAmplitude = useRef(0.3);
  const targetSpeed = useRef(0.5);
  const currentDistortion = useRef(0.5);
  const currentAmplitude = useRef(0.3);
  const currentSpeed = useRef(0.5);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uDistortion: { value: 0.5 },
      uAmplitude: { value: 0.3 },
      uSpeed: { value: 0.5 },
      uCoreColor: { value: new THREE.Color(0.2, 0.2, 0.4) },
      uGlowColor: { value: new THREE.Color(0.4, 0.4, 0.8) },
      uArousal: { value: 0.3 },
    }),
    [],
  );

  // Update targets when state changes
  useEffect(() => {
    targetColors.current = emotionToColors(state);
    // Distortion: low dominance + low certainty → more distorted
    targetDistortion.current = 1.0 - (state.dominance * 0.5 + state.certainty * 0.5);
    targetAmplitude.current = 0.1 + state.intensity * 0.6;
    targetSpeed.current = 0.2 + state.arousal * 1.2;
  }, [state]);

  useFrame((_, delta) => {
    const t = Math.min(delta * 2.0, 0.1); // smooth lerp factor

    // Lerp colors
    currentColors.current.core.lerp(targetColors.current.core, t);
    currentColors.current.glow.lerp(targetColors.current.glow, t);
    currentColors.current.particle.lerp(targetColors.current.particle, t);

    // Lerp parameters
    currentDistortion.current += (targetDistortion.current - currentDistortion.current) * t;
    currentAmplitude.current += (targetAmplitude.current - currentAmplitude.current) * t;
    currentSpeed.current += (targetSpeed.current - currentSpeed.current) * t;

    // Update uniforms
    uniforms.uTime.value += delta;
    uniforms.uDistortion.value = currentDistortion.current;
    uniforms.uAmplitude.value = currentAmplitude.current;
    uniforms.uSpeed.value = currentSpeed.current;
    uniforms.uCoreColor.value.copy(currentColors.current.core);
    uniforms.uGlowColor.value.copy(currentColors.current.glow);
    uniforms.uArousal.value = state.arousal;

    // Slow rotation
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1 * currentSpeed.current;
      meshRef.current.rotation.x += delta * 0.04 * currentSpeed.current;
    }
  });

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[1.2, 64]} />
      <shaderMaterial
        vertexShader={orbVertexShader}
        fragmentShader={orbFragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Particle system
// ---------------------------------------------------------------------------

const PARTICLE_COUNT = 120;

function ParticleField({ state }: { state: EmotionalState }) {
  const pointsRef = useRef<THREE.Points>(null);
  const positionsRef = useRef<Float32Array | null>(null);
  const velocitiesRef = useRef<Float32Array | null>(null);

  const targetColor = useRef(new THREE.Color(0.5, 0.5, 0.8));

  const { positions, velocities } = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3);
    const vel = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Distribute around orb
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.5 + Math.random() * 1.5;
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
      // Orbital velocities
      vel[i * 3] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
    }
    positionsRef.current = pos;
    velocitiesRef.current = vel;
    return { positions: pos, velocities: vel };
  }, []);

  useEffect(() => {
    const colors = emotionToColors(state);
    targetColor.current.copy(colors.particle);
  }, [state]);

  useFrame((_, delta) => {
    if (!pointsRef.current || !positionsRef.current || !velocitiesRef.current) return;

    const pos = positionsRef.current;
    const vel = velocitiesRef.current;
    const speed = 0.3 + state.arousal * 1.5;
    const energy = state.body_state.energy;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const ix = i * 3;
      const iy = ix + 1;
      const iz = ix + 2;

      // Move
      pos[ix] += vel[ix] * speed * delta * 60;
      pos[iy] += vel[iy] * speed * delta * 60;
      pos[iz] += vel[iz] * speed * delta * 60;

      // Distance from center
      const dist = Math.sqrt(pos[ix] ** 2 + pos[iy] ** 2 + pos[iz] ** 2);

      // Keep particles in shell (1.4 .. 3.0)
      if (dist > 3.0 || dist < 1.4) {
        const norm = 1.0 / (dist || 1);
        const targetR = 1.5 + Math.random() * 1.0;
        pos[ix] = pos[ix] * norm * targetR;
        pos[iy] = pos[iy] * norm * targetR;
        pos[iz] = pos[iz] * norm * targetR;
        // Reverse radial velocity
        vel[ix] *= -0.5;
        vel[iy] *= -0.5;
        vel[iz] *= -0.5;
      }

      // Orbital drift — tangential acceleration
      const tangentX = -pos[iy];
      const tangentY = pos[ix];
      vel[ix] += tangentX * 0.0001 * energy;
      vel[iy] += tangentY * 0.0001 * energy;
    }

    const geom = pointsRef.current.geometry;
    geom.attributes.position.needsUpdate = true;

    // Lerp material color
    const mat = pointsRef.current.material as THREE.PointsMaterial;
    mat.color.lerp(targetColor.current, Math.min(delta * 2, 0.1));

    // Particle size responds to intensity
    mat.size = 0.03 + state.intensity * 0.04;
    mat.opacity = 0.3 + energy * 0.5;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={PARTICLE_COUNT}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        color={0x8888ff}
        transparent
        opacity={0.5}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// Ambient glow (inner point light + ambient)
// ---------------------------------------------------------------------------

function AmbientGlow({ state }: { state: EmotionalState }) {
  const lightRef = useRef<THREE.PointLight>(null);
  const targetColor = useRef(new THREE.Color());

  useEffect(() => {
    const colors = emotionToColors(state);
    targetColor.current.copy(colors.glow);
  }, [state]);

  useFrame((_, delta) => {
    if (!lightRef.current) return;
    lightRef.current.color.lerp(targetColor.current, Math.min(delta * 2, 0.1));
    lightRef.current.intensity = 0.8 + state.intensity * 1.5;
  });

  return <pointLight ref={lightRef} position={[0, 0, 0]} distance={6} decay={2} />;
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

function OrbScene({ state }: { state: EmotionalState }) {
  return (
    <>
      <ambientLight intensity={0.08} />
      <AmbientGlow state={state} />
      <OrbMesh state={state} />
      <ParticleField state={state} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function QualiaOrb({ emotionalState, metaphor }: Props) {
  if (!emotionalState) return null;

  return (
    <div className="qualia-orb">
      <Canvas
        className="qualia-orb__canvas"
        camera={{ position: [0, 0, 3.5], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 1.5]}
      >
        <OrbScene state={emotionalState} />
      </Canvas>
      {metaphor && (
        <div className="qualia-orb__metaphor">{metaphor}</div>
      )}
      <div className="qualia-orb__label">
        {emotionalState.primary_emotion}
      </div>
    </div>
  );
}
