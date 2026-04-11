"""Emotion Dynamics - Modelos diferenciales para transiciones emocionales.

Reemplaza las interpolaciones lineales (lerp) con ecuaciones diferenciales
inspiradas en DynAffect (Kuppens et al., 2010):

    dx/dt = -k*(x - attractor) + noise + perturbation

Produce:
- Oscilaciones naturales (micro-fluctuaciones incluso sin estímulo)
- Inercia variable por emoción (anger persiste, surprise se disipa)
- Transiciones no-lineales más realistas
"""

from __future__ import annotations

import random

from pathos.models.coupling import CouplingMatrix
from pathos.models.emotion import PrimaryEmotion


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Inercia por emoción: cuánto resiste el cambio (0=cambia fácil, 1=persiste)
EMOTION_INERTIA: dict[PrimaryEmotion, float] = {
    PrimaryEmotion.JOY:            0.5,
    PrimaryEmotion.EXCITEMENT:     0.3,   # Se disipa rápido
    PrimaryEmotion.GRATITUDE:      0.6,
    PrimaryEmotion.HOPE:           0.5,
    PrimaryEmotion.CONTENTMENT:    0.7,   # Persiste (mood-like)
    PrimaryEmotion.RELIEF:         0.3,   # Transitoria
    PrimaryEmotion.ANGER:          0.8,   # Alta inercia (persiste mucho)
    PrimaryEmotion.FRUSTRATION:    0.7,
    PrimaryEmotion.FEAR:           0.4,   # Responde rápido a cambios
    PrimaryEmotion.ANXIETY:        0.75,  # Sticky
    PrimaryEmotion.SADNESS:        0.8,   # Persiste mucho
    PrimaryEmotion.HELPLESSNESS:   0.85,  # Muy difícil de salir
    PrimaryEmotion.DISAPPOINTMENT: 0.6,
    PrimaryEmotion.SURPRISE:       0.1,   # Se disipa muy rápido
    PrimaryEmotion.ALERTNESS:      0.3,
    PrimaryEmotion.CONTEMPLATION:  0.5,
    PrimaryEmotion.INDIFFERENCE:   0.6,
    PrimaryEmotion.MIXED:          0.4,
    PrimaryEmotion.NEUTRAL:        0.5,
}


class EmotionDynamics:
    """Sistema de dinámicas emocionales con ecuaciones diferenciales."""

    def __init__(
        self,
        attractor_strength: float = 0.15,
        variability: float = 0.3,
        base_inertia: float = 0.5,
    ):
        """
        Args:
            attractor_strength: Fuerza del punto atractor (homeostasis).
            variability: Tendencia a fluctuar (personalidad, neuroticism).
            base_inertia: Inercia base (personality).
        """
        self.attractor_strength = attractor_strength
        self.variability = variability
        self.base_inertia = base_inertia

    def step(
        self,
        current: float,
        target: float,
        attractor: float,
        emotion: PrimaryEmotion,
        dimension: str = "valence",
        dt: float = 1.0,
        contagion: float = 0.0,
    ) -> float:
        """Calcula un paso de la dinámica emocional.

        Args:
            current: Valor actual de la dimensión.
            target: Valor objetivo (del nuevo appraisal).
            attractor: Punto atractor (baseline del mood).
            emotion: Emoción actual (determina inercia).
            dimension: Nombre de la dimensión (para rangos).
            dt: Paso temporal (1.0 = 1 turno).
            contagion: Perturbación de contagio emocional (pre-cognitivo).

        Returns:
            Nuevo valor de la dimensión.
        """
        # Inercia específica de la emoción
        emotion_inertia = EMOTION_INERTIA.get(emotion, self.base_inertia)
        effective_inertia = (emotion_inertia + self.base_inertia) / 2

        # 1. Perturbación del estímulo (appraisal)
        perturbation = (target - current) * (1 - effective_inertia)

        # 2. Pull del atractor (homeostasis)
        pull = -self.attractor_strength * (current - attractor)

        # 3. Ruido estocástico (micro-fluctuaciones)
        noise = random.gauss(0, self.variability * 0.03)

        # 4. Integrar (contagion se suma como perturbación pre-cognitiva)
        new_value = current + (perturbation + pull + noise + contagion) * dt

        # Clamp según dimensión
        if dimension == "valence":
            return _clamp(round(new_value, 4), -1, 1)
        else:
            return _clamp(round(new_value, 4), 0, 1)

    def step_4d(
        self,
        current_v: float, current_a: float, current_d: float, current_c: float,
        target_v: float, target_a: float, target_d: float, target_c: float,
        baseline_v: float, baseline_a: float,
        emotion: PrimaryEmotion,
        contagion_v: float = 0.0,
        contagion_a: float = 0.0,
        coupling: CouplingMatrix | None = None,
    ) -> tuple[float, float, float, float]:
        """Aplica dinámicas a las 4 dimensiones emocionales.

        Args:
            contagion_v: Perturbación de contagio en valence.
            contagion_a: Perturbación de contagio en arousal.
            coupling: Optional coupling matrix for cross-dimensional interaction.
                If None or all-zero, dimensions are computed independently (legacy).

        Returns:
            (valence, arousal, dominance, certainty)
        """
        # Attractors: mood baselines for V/A, 0.5 neutral for D/C
        attr_v = baseline_v
        attr_a = baseline_a
        attr_d = 0.5
        attr_c = 0.5

        # Compute coupling contributions (cross-dimensional interaction)
        coup_v = coup_a = coup_d = coup_c = 0.0
        if coupling is not None and not coupling.is_zero:
            dev_v = current_v - attr_v
            dev_a = current_a - attr_a
            dev_d = current_d - attr_d
            dev_c = current_c - attr_c
            coup_v, coup_a, coup_d, coup_c = coupling.get_coupling_contribution(
                dev_v, dev_a, dev_d, dev_c,
            )

        # Step each dimension with its coupling contribution added to contagion
        v = self.step(current_v, target_v, attr_v, emotion, "valence",
                      contagion=contagion_v + coup_v)
        a = self.step(current_a, target_a, attr_a, emotion, "arousal",
                      contagion=contagion_a + coup_a)
        d = self.step(current_d, target_d, attr_d, emotion, "dominance",
                      contagion=coup_d)
        c = self.step(current_c, target_c, attr_c, emotion, "certainty",
                      contagion=coup_c)
        return v, a, d, c

    def idle_fluctuation(
        self,
        current: float,
        attractor: float,
        dimension: str = "valence",
    ) -> float:
        """Fluctuación cuando no hay estímulo (entre turnos).

        Produce micro-oscilaciones alrededor del atractor.
        """
        pull = -self.attractor_strength * 0.5 * (current - attractor)
        noise = random.gauss(0, self.variability * 0.02)
        new_value = current + pull + noise

        if dimension == "valence":
            return _clamp(round(new_value, 4), -1, 1)
        return _clamp(round(new_value, 4), 0, 1)
