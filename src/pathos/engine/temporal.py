"""Temporal Dynamics - Anticipación, Ruminación y Savoring.

El sistema ya no solo reacciona al presente:

- Anticipation: detecta patrones recurrentes y genera emociones anticipatorias
- Rumination: emociones negativas no procesadas persisten como "intrusive thoughts"
- Savoring: emociones positivas validadas se extienden

Basado en Response Styles Theory (Nolen-Hoeksema) y Anticipatory Affect (Wilson & Gilbert).
"""

from pydantic import BaseModel, Field

from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


class RuminationEntry(BaseModel):
    """Una emoción no procesada que persiste como pensamiento intrusivo."""

    emotion: PrimaryEmotion
    intensity: float = Field(ge=0, le=1)
    trigger: str
    turns_active: int = 0
    max_turns: int = 5  # Después de esto, se resuelve solo


class SavoringEntry(BaseModel):
    """Una emoción positiva que se extiende por validación."""

    emotion: PrimaryEmotion
    intensity: float = Field(ge=0, le=1)
    trigger: str
    turns_remaining: int = 3


class TemporalResult(BaseModel):
    """Resultado del procesamiento temporal."""

    rumination_active: bool = False
    rumination_emotion: PrimaryEmotion | None = None
    rumination_intensity: float = 0.0
    savoring_active: bool = False
    savoring_emotion: PrimaryEmotion | None = None
    anticipation_active: bool = False
    anticipation_emotion: PrimaryEmotion | None = None
    anticipation_intensity: float = 0.0


class TemporalProcessor:
    """Procesador de dinámicas temporales."""

    def __init__(self) -> None:
        self._ruminations: list[RuminationEntry] = []
        self._savorings: list[SavoringEntry] = []
        # Pattern tracking para anticipation
        self._topic_history: list[str] = []  # Últimos N temas
        self._topic_emotions: dict[str, list[PrimaryEmotion]] = {}  # tema -> emociones

    def process_pre_turn(
        self,
        stimulus: str,
    ) -> TemporalResult:
        """Procesamiento ANTES del appraisal — anticipación y rumiación activa.

        Args:
            stimulus: Texto del usuario.

        Returns:
            TemporalResult con emociones anticipatorias/rumination activas.
        """
        result = TemporalResult()

        # 1. Rumination: ¿hay emociones no procesadas que resurgen?
        active_ruminations = [r for r in self._ruminations if r.turns_active < r.max_turns]
        if active_ruminations:
            # La rumination más intensa afecta al procesamiento
            strongest = max(active_ruminations, key=lambda r: r.intensity)
            result.rumination_active = True
            result.rumination_emotion = strongest.emotion
            result.rumination_intensity = strongest.intensity * 0.5  # Atenuado
            # Incrementar turns_active
            for r in active_ruminations:
                r.turns_active += 1

        # 2. Anticipation: ¿el estímulo recuerda a un patrón conocido?
        topic = self._extract_topic(stimulus)
        if topic and topic in self._topic_emotions:
            past_emotions = self._topic_emotions[topic]
            if len(past_emotions) >= 2:
                # Hay un patrón: anticipar la emoción más frecuente
                emotion_counts: dict[PrimaryEmotion, int] = {}
                for e in past_emotions:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                anticipated = max(emotion_counts, key=emotion_counts.get)  # type: ignore[arg-type]
                frequency = emotion_counts[anticipated] / len(past_emotions)

                if frequency > 0.5:  # Solo si el patrón es fuerte
                    result.anticipation_active = True
                    result.anticipation_emotion = anticipated
                    result.anticipation_intensity = frequency * 0.3  # Suave

        # 3. Savoring: ¿hay emociones positivas extendiéndose?
        active_savorings = [s for s in self._savorings if s.turns_remaining > 0]
        if active_savorings:
            strongest = max(active_savorings, key=lambda s: s.intensity)
            result.savoring_active = True
            result.savoring_emotion = strongest.emotion
            # Decrementar turns
            for s in active_savorings:
                s.turns_remaining -= 1

        # Limpiar expirados
        self._ruminations = [r for r in self._ruminations if r.turns_active < r.max_turns]
        self._savorings = [s for s in self._savorings if s.turns_remaining > 0]

        return result

    def process_post_turn(
        self,
        stimulus: str,
        state: EmotionalState,
        previous_state: EmotionalState,
    ) -> None:
        """Procesamiento DESPUÉS de la generación emocional.

        Detecta:
        - Emociones negativas abruptas → rumination
        - Emociones positivas validadas → savoring
        - Registra patrones para anticipation

        Args:
            stimulus: Texto del usuario.
            state: Estado emocional actual.
            previous_state: Estado emocional previo.
        """
        # 1. Detectar rumination trigger
        # Si el usuario cambió de tema abruptamente mientras había emoción negativa intensa
        was_negative_intense = (
            previous_state.valence < -0.3
            and previous_state.intensity > 0.6
        )
        topic_changed = self._topic_changed(stimulus)

        if was_negative_intense and topic_changed:
            # Emoción negativa no procesada → rumination
            self._ruminations.append(RuminationEntry(
                emotion=previous_state.primary_emotion,
                intensity=previous_state.intensity,
                trigger=previous_state.triggered_by[:100],
            ))
            # Limitar ruminations activas
            if len(self._ruminations) > 3:
                self._ruminations = sorted(
                    self._ruminations, key=lambda r: r.intensity, reverse=True,
                )[:3]

        # 2. Detectar savoring trigger
        # Si la emoción positiva es intensa y el usuario parece validarla
        if (state.valence > 0.4
            and state.intensity > 0.5
            and state.primary_emotion in (
                PrimaryEmotion.JOY, PrimaryEmotion.GRATITUDE,
                PrimaryEmotion.CONTENTMENT, PrimaryEmotion.EXCITEMENT,
            )):
            # Check if user seems to be engaging positively
            positive_words = ["yes", "si", "great", "love", "genial", "exactly", "wow"]
            if any(w in stimulus.lower() for w in positive_words):
                self._savorings.append(SavoringEntry(
                    emotion=state.primary_emotion,
                    intensity=state.intensity * 0.6,
                    trigger=stimulus[:100],
                ))

        # 3. Registrar para anticipation
        topic = self._extract_topic(stimulus)
        if topic:
            self._topic_history.append(topic)
            if len(self._topic_history) > 20:
                self._topic_history = self._topic_history[-20:]

            if topic not in self._topic_emotions:
                self._topic_emotions[topic] = []
            self._topic_emotions[topic].append(state.primary_emotion)
            if len(self._topic_emotions[topic]) > 10:
                self._topic_emotions[topic] = self._topic_emotions[topic][-10:]
            # Cap total topic keys to prevent unbounded growth
            if len(self._topic_emotions) > 50:
                oldest_topic = next(iter(self._topic_emotions))
                del self._topic_emotions[oldest_topic]

    def apply_temporal_effects(
        self,
        state: EmotionalState,
        temporal_result: TemporalResult,
    ) -> EmotionalState:
        """Aplica los efectos temporales al estado emocional.

        Args:
            state: Estado emocional post-generación.
            temporal_result: Resultado del procesamiento pre-turn.

        Returns:
            Estado con efectos temporales aplicados.
        """
        valence = state.valence
        arousal = state.arousal
        intensity = state.intensity

        # Rumination: tira el estado hacia lo negativo
        if temporal_result.rumination_active and temporal_result.rumination_intensity > 0:
            rumination_pull = temporal_result.rumination_intensity
            valence = _clamp(valence - rumination_pull * 0.2, -1, 1)
            arousal = _clamp(arousal + rumination_pull * 0.1, 0, 1)  # Más activated
            intensity = _clamp(intensity + rumination_pull * 0.1, 0, 1)

        # Savoring: extiende lo positivo
        if temporal_result.savoring_active:
            valence = _clamp(valence + 0.05, -1, 1)
            intensity = _clamp(intensity + 0.05, 0, 1)

        # Anticipation: pre-carga la emoción anticipada
        if temporal_result.anticipation_active and temporal_result.anticipation_intensity > 0:
            # Suave bias hacia la emoción anticipada
            intensity = _clamp(intensity + temporal_result.anticipation_intensity * 0.5, 0, 1)

        if valence == state.valence and arousal == state.arousal and intensity == state.intensity:
            return state

        return state.model_copy(update={
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "intensity": round(intensity, 4),
        })

    def _extract_topic(self, stimulus: str) -> str | None:
        """Extrae un tema simplificado del estímulo (para pattern matching)."""
        # Simple: primeras 2-3 palabras significativas
        words = [w.lower() for w in stimulus.split() if len(w) > 3]
        if len(words) >= 2:
            return " ".join(words[:3])
        return words[0] if words else None

    def _topic_changed(self, stimulus: str) -> bool:
        """Detecta si el tema cambió respecto al anterior."""
        current = self._extract_topic(stimulus)
        if not self._topic_history or not current:
            return False
        previous = self._topic_history[-1] if self._topic_history else None
        if previous is None:
            return False
        # Simple check: ¿comparten palabras?
        current_words = set(current.split())
        previous_words = set(previous.split())
        overlap = current_words & previous_words
        return len(overlap) == 0  # Sin overlap = cambio de tema
