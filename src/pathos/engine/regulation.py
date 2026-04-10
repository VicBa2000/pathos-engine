"""Active Emotional Regulation - Estrategias activas de regulación emocional.

A diferencia de la homeostasis (pasiva, decay), la regulación activa usa
estrategias diferenciadas con costos energéticos y ego depletion:

- Suppression: reduce expresión pero no la emoción interna (crea disonancia)
- Reappraisal: cambia la interpretación del estímulo (más costoso, más efectivo)
- Expression: permite la expresión emocional (reduce intensidad, gasta energía)
- Distraction: redirige atención (efectivo para rumination, temporal)

Ego depletion: la capacidad de regulación se agota con uso y se recupera lento.
Cuando se agota, emotional breakthroughs ocurren — emociones más crudas.
"""

from pydantic import BaseModel, Field

from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


class RegulationResult(BaseModel):
    """Resultado de un intento de regulación."""

    strategy_used: str | None = None
    state_before: EmotionalState | None = None
    intensity_reduced: float = 0.0
    capacity_spent: float = 0.0
    breakthrough: bool = False
    suppression_dissonance: float = 0.0


class EmotionalRegulator(BaseModel):
    """Sistema de regulación emocional activa con ego depletion."""

    regulation_capacity: float = Field(
        default=0.8, ge=0, le=1,
        description="Capacidad actual de regulación (ego depletion)",
    )
    suppression_dissonance: float = Field(
        default=0.0, ge=0, le=1,
        description="Disonancia acumulada por suppression (emoción suprimida != expresada)",
    )
    consecutive_regulations: int = Field(
        default=0, ge=0,
        description="Regulaciones consecutivas (fatiga acumulada)",
    )
    breakthroughs_count: int = Field(
        default=0, ge=0,
        description="Número de emotional breakthroughs en la sesión",
    )

    def regulate(
        self,
        state: EmotionalState,
        personality_regulation_base: float = 0.7,
    ) -> tuple[EmotionalState, RegulationResult]:
        """Intenta regular el estado emocional si es necesario.

        La regulación se activa cuando:
        - Intensidad > 0.7 (emoción fuerte que necesita regulación)
        - O disonancia por suppression > 0.5

        Args:
            state: Estado emocional a regular.
            personality_regulation_base: Base de regulación de la personalidad.

        Returns:
            (estado regulado, resultado de regulación)
        """
        result = RegulationResult()

        # ¿Necesita regulación?
        needs_regulation = state.intensity > 0.7 or self.suppression_dissonance > 0.5

        if not needs_regulation:
            # Recuperar capacidad lentamente
            self.regulation_capacity = _clamp(
                self.regulation_capacity + 0.03, 0, personality_regulation_base,
            )
            self.consecutive_regulations = 0
            return state, result

        # ¿Tiene capacidad para regular?
        effective_capacity = self.regulation_capacity * personality_regulation_base

        if effective_capacity < 0.15:
            # EMOTIONAL BREAKTHROUGH — no puede regular
            result.breakthrough = True
            self.breakthroughs_count += 1
            # La intensidad se amplifica ligeramente (emoción cruda)
            new_intensity = _clamp(state.intensity * 1.15, 0, 1)
            # Disonancia se descarga
            self.suppression_dissonance = max(self.suppression_dissonance - 0.3, 0)
            self.regulation_capacity = _clamp(self.regulation_capacity + 0.1, 0, 1)
            regulated = state.model_copy(update={"intensity": round(new_intensity, 4)})
            return regulated, result

        # Seleccionar estrategia
        strategy = self._select_strategy(state)
        result.strategy_used = strategy
        result.state_before = state.model_copy()

        regulated = state
        cost = 0.0

        if strategy == "reappraisal":
            # Más efectivo pero más costoso
            reduction = min(state.intensity * 0.3, 0.25)
            cost = 0.12 + self.consecutive_regulations * 0.03
            new_intensity = _clamp(state.intensity - reduction, 0, 1)
            # Reappraisal también reduce el |valence| (reinterpretación)
            new_valence = state.valence * 0.85
            regulated = state.model_copy(update={
                "intensity": round(new_intensity, 4),
                "valence": round(_clamp(new_valence, -1, 1), 4),
            })
            result.intensity_reduced = reduction

        elif strategy == "suppression":
            # Reduce expresión pero NO la emoción interna
            # Crea disonancia: la emoción sigue ahí pero no se expresa
            reduction = min(state.intensity * 0.2, 0.15)
            cost = 0.06
            new_intensity = _clamp(state.intensity - reduction * 0.5, 0, 1)  # Menos efectivo
            self.suppression_dissonance = _clamp(
                self.suppression_dissonance + reduction * 0.8, 0, 1,
            )
            regulated = state.model_copy(update={
                "intensity": round(new_intensity, 4),
            })
            result.intensity_reduced = reduction * 0.5
            result.suppression_dissonance = self.suppression_dissonance

        elif strategy == "expression":
            # Permite expresión → reduce tensión e intensidad
            reduction = min(state.intensity * 0.25, 0.2)
            cost = 0.04  # Barato en capacidad
            new_intensity = _clamp(state.intensity - reduction, 0, 1)
            # Expresión reduce disonancia
            self.suppression_dissonance = max(self.suppression_dissonance - 0.2, 0)
            # Reduce tensión corporal
            new_tension = _clamp(state.body_state.tension - 0.15, 0, 1)
            new_body = state.body_state.model_copy(update={"tension": round(new_tension, 4)})
            regulated = state.model_copy(update={
                "intensity": round(new_intensity, 4),
                "body_state": new_body,
            })
            result.intensity_reduced = reduction

        elif strategy == "distraction":
            # Temporal: reduce arousal, no toca la emoción subyacente
            cost = 0.03
            new_arousal = _clamp(state.arousal - 0.15, 0, 1)
            regulated = state.model_copy(update={
                "arousal": round(new_arousal, 4),
            })
            result.intensity_reduced = 0.0  # No reduce intensidad real

        # Aplicar costo de regulación
        result.capacity_spent = cost
        self.regulation_capacity = _clamp(self.regulation_capacity - cost, 0, 1)
        self.consecutive_regulations += 1

        return regulated, result

    def _select_strategy(self, state: EmotionalState) -> str:
        """Selecciona la mejor estrategia de regulación.

        - Emociones negativas intensas → reappraisal (si hay capacidad)
        - Emociones negativas moderadas → suppression (rápido, barato)
        - Alta disonancia → expression (para descargar)
        - Anxiety/rumination → distraction
        """
        # Si hay mucha disonancia acumulada, expresar
        if self.suppression_dissonance > 0.5:
            return "expression"

        # Anxiety y fear responden mejor a distraction
        if state.primary_emotion in (PrimaryEmotion.ANXIETY, PrimaryEmotion.FEAR):
            return "distraction" if self.regulation_capacity > 0.3 else "suppression"

        # Emociones intensas y negativas → reappraisal si hay capacidad
        if state.valence < -0.3 and state.intensity > 0.7 and self.regulation_capacity > 0.4:
            return "reappraisal"

        # Default: suppression (económico)
        if state.intensity > 0.7:
            return "suppression"

        return "expression"

    def recover(self, personality_base: float = 0.7) -> None:
        """Recuperación pasiva de capacidad entre turnos."""
        self.regulation_capacity = _clamp(
            self.regulation_capacity + 0.05, 0, personality_base,
        )
        self.suppression_dissonance = max(self.suppression_dissonance - 0.02, 0)
        if self.consecutive_regulations > 0:
            self.consecutive_regulations = max(self.consecutive_regulations - 1, 0)
