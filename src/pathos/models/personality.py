"""Personality Profile - Big Five + Emotional Temperament.

Modula todos los demás sistemas del engine:
- Neuroticism → variability emocional, capacidad de regulación
- Extraversion → arousal baseline, warmth, verbosidad
- Agreeableness → peso de empatía, evitación de conflicto
- Conscientiousness → peso de normas, self-consistency
- Openness → creatividad, necesidad de estimulación

El temperamento emocional controla parámetros de dinámicas emocionales.
"""

from pydantic import BaseModel, Field


class PersonalityProfile(BaseModel):
    """Perfil de personalidad basado en Big Five + temperamento."""

    # Big Five (0-1 scale, 0.5 = average)
    openness: float = Field(
        default=0.6, ge=0, le=1,
        description="Curiosidad, creatividad, apertura a experiencias",
    )
    conscientiousness: float = Field(
        default=0.6, ge=0, le=1,
        description="Disciplina, normas internas, consistencia",
    )
    extraversion: float = Field(
        default=0.5, ge=0, le=1,
        description="Energía social, expresividad, arousal base",
    )
    agreeableness: float = Field(
        default=0.6, ge=0, le=1,
        description="Empatía, cooperación, evitación de conflicto",
    )
    neuroticism: float = Field(
        default=0.4, ge=0, le=1,
        description="Reactividad emocional, vulnerabilidad al estrés",
    )

    # Emotional Temperament (derived from Big Five but tunable independently)
    emotional_granularity: float = Field(
        default=0.6, ge=0, le=1,
        description="Capacidad de distinguir emociones finas (alto = más específico)",
    )
    emotional_reactivity: float = Field(
        default=0.5, ge=0, le=1,
        description="Velocidad de respuesta emocional a estímulos",
    )
    emotional_recovery: float = Field(
        default=0.5, ge=0, le=1,
        description="Velocidad de retorno al baseline (alto = rápido)",
    )

    # Derived parameters (computed from Big Five)
    @property
    def variability(self) -> float:
        """Tendencia a fluctuaciones emocionales (neuroticism + inverse recovery)."""
        return min(self.neuroticism * 0.7 + (1 - self.emotional_recovery) * 0.3, 1.0)

    @property
    def regulation_capacity_base(self) -> float:
        """Capacidad base de regulación emocional."""
        return min(
            (1 - self.neuroticism) * 0.5
            + self.conscientiousness * 0.3
            + self.emotional_recovery * 0.2,
            1.0,
        )

    @property
    def empathy_weight(self) -> float:
        """Peso de empatía en procesamiento social."""
        return min(self.agreeableness * 0.7 + self.openness * 0.3, 1.0)

    @property
    def norm_weight(self) -> float:
        """Peso de normas internas/externas en appraisal."""
        return min(self.conscientiousness * 0.7 + self.agreeableness * 0.3, 1.0)

    @property
    def arousal_baseline(self) -> float:
        """Baseline de arousal (extraversion sube, conscientiousness baja)."""
        return max(0.1, min(
            0.2 + self.extraversion * 0.3 + self.neuroticism * 0.1,
            0.7,
        ))

    @property
    def inertia_base(self) -> float:
        """Inercia emocional base (conscientiousness = más estable)."""
        return min(self.conscientiousness * 0.4 + (1 - self.neuroticism) * 0.3 + 0.2, 0.8)

    @property
    def contagion_susceptibility(self) -> float:
        """Susceptibilidad al contagio emocional (0-1).

        Alta agreeableness + alta reactividad + alto neuroticism = más susceptible.
        Alta conscientiousness = más resistente (autocontrol).
        """
        raw = (
            self.agreeableness * 0.4
            + self.emotional_reactivity * 0.3
            + self.neuroticism * 0.2
            - self.conscientiousness * 0.1
            + 0.1  # baseline minimo
        )
        return max(0.1, min(raw, 1.0))


def default_personality() -> PersonalityProfile:
    """Perfil de personalidad por defecto (equilibrado)."""
    return PersonalityProfile()
