"""Interoception — Body-state feedback ascendente.

Cognición encarnada (Damasio, Craig): el cuerpo INFORMA a la emoción.
Tensión muscular prolongada → ansiedad.
Energía baja prolongada → apatía/tristeza.
Warmth alta sostenida → refuerza valence positiva.

Los deltas se inyectan ANTES de emotion generation como perturbaciones
al estado emocional (similar a contagion o external signals).
"""

from __future__ import annotations

from dataclasses import dataclass

from pathos.models.emotion import BodyState


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Duración mínima (en turnos) para que el body state genere feedback
TENSION_THRESHOLD_TURNS = 3
LOW_ENERGY_THRESHOLD_TURNS = 3
WARMTH_THRESHOLD_TURNS = 2

# Umbrales de activación del body state
TENSION_HIGH = 0.7
ENERGY_LOW = 0.25
WARMTH_HIGH = 0.7

# Límite máximo de perturbación interoceptiva (±0.15)
MAX_PERTURBATION = 0.15


@dataclass
class InteroceptiveState:
    """Tracking de duraciones del body state para feedback interoceptivo."""

    high_tension_turns: int = 0
    low_energy_turns: int = 0
    high_warmth_turns: int = 0


@dataclass
class InteroceptiveResult:
    """Resultado del feedback interoceptivo."""

    active: bool = False
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    source: str = ""  # Qué lo activó: "tension", "low_energy", "warmth", ""


def update_interoceptive_state(
    intero: InteroceptiveState,
    body: BodyState,
) -> InteroceptiveState:
    """Actualiza los contadores de duración del body state.

    Llamar al inicio de cada turno DESPUÉS de homeostasis.
    """
    new = InteroceptiveState(
        high_tension_turns=intero.high_tension_turns + 1 if body.tension >= TENSION_HIGH else 0,
        low_energy_turns=intero.low_energy_turns + 1 if body.energy <= ENERGY_LOW else 0,
        high_warmth_turns=intero.high_warmth_turns + 1 if body.warmth >= WARMTH_HIGH else 0,
    )
    return new


def compute_interoceptive_feedback(
    intero: InteroceptiveState,
    body: BodyState,
) -> InteroceptiveResult:
    """Computa el feedback interoceptivo: body → emotion.

    Returns perturbaciones (valence_delta, arousal_delta) para inyectar
    en el pipeline ANTES de emotion generation.

    Reglas:
    - Tensión alta prolongada (>3 turnos): genera ansiedad
      (valence baja, arousal sube)
    - Energía muy baja prolongada (>3 turnos): genera apatía/tristeza
      (valence baja, arousal baja)
    - Warmth alta sostenida (>2 turnos): refuerza valence positiva
      (valence sube ligeramente)

    Solo se aplica el efecto más fuerte para evitar perturbaciones
    acumulativas excesivas.
    """
    effects: list[tuple[float, float, float, str]] = []  # (priority, dv, da, source)

    # Alta tensión prolongada → ansiedad
    if intero.high_tension_turns >= TENSION_THRESHOLD_TURNS and body.tension >= TENSION_HIGH:
        excess = body.tension - TENSION_HIGH
        duration_factor = min((intero.high_tension_turns - TENSION_THRESHOLD_TURNS + 1) * 0.3, 1.0)
        dv = -excess * 0.4 * duration_factor
        da = excess * 0.3 * duration_factor
        priority = abs(dv) + abs(da)
        effects.append((priority, dv, da, "tension"))

    # Energía baja prolongada → apatía
    if intero.low_energy_turns >= LOW_ENERGY_THRESHOLD_TURNS and body.energy <= ENERGY_LOW:
        deficit = ENERGY_LOW - body.energy
        duration_factor = min((intero.low_energy_turns - LOW_ENERGY_THRESHOLD_TURNS + 1) * 0.3, 1.0)
        dv = -deficit * 0.5 * duration_factor
        da = -deficit * 0.3 * duration_factor
        priority = abs(dv) + abs(da)
        effects.append((priority, dv, da, "low_energy"))

    # Warmth alta sostenida → refuerzo positivo
    if intero.high_warmth_turns >= WARMTH_THRESHOLD_TURNS and body.warmth >= WARMTH_HIGH:
        excess = body.warmth - WARMTH_HIGH
        dv = excess * 0.2
        da = 0.0
        priority = abs(dv)
        effects.append((priority, dv, da, "warmth"))

    if not effects:
        return InteroceptiveResult(active=False)

    # Aplicar solo el efecto más fuerte
    effects.sort(key=lambda x: -x[0])
    _, dv, da, source = effects[0]

    return InteroceptiveResult(
        active=True,
        valence_delta=_clamp(dv, -MAX_PERTURBATION, MAX_PERTURBATION),
        arousal_delta=_clamp(da, -MAX_PERTURBATION, MAX_PERTURBATION),
        source=source,
    )
