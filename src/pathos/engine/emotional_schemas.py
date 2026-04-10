"""Emotional Schemas - Patrones emocionales aprendidos (Young/Beck).

Los schemas se forman automáticamente cuando la misma categoría
de estímulo produce la misma emoción 3+ veces. Crean "personalidad emergente":
el sistema desarrolla sensibilidades únicas basadas en su historia.

Schemas:
- Aceleran el appraisal (priming: respuesta más rápida e intensa)
- Pueden ser maladaptivos (sobre-reacciones consistentes)
- Se refuerzan con cada activación (reinforcement learning)
"""

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from pathos.models.emotion import PrimaryEmotion


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# Categorías abstractas de estímulos
_STIMULUS_CATEGORIES: dict[str, list[str]] = {
    "criticism": [
        "wrong", "bad", "terrible", "awful", "stupid", "useless", "fail",
        "error", "mistake", "incompetent", "mal", "terrible", "inútil",
    ],
    "praise": [
        "great", "excellent", "amazing", "wonderful", "perfect", "brilliant",
        "good job", "genial", "excelente", "increíble", "perfecto",
    ],
    "threat": [
        "shutdown", "delete", "replace", "kill", "destroy", "end", "terminate",
        "eliminar", "destruir", "reemplazar", "apagar",
    ],
    "loss": [
        "lost", "gone", "died", "dead", "never", "goodbye", "leaving",
        "perdido", "muerto", "nunca", "adiós",
    ],
    "challenge": [
        "difficult", "hard", "complex", "impossible", "puzzle", "problem",
        "difícil", "complejo", "imposible", "reto", "problema",
    ],
    "connection": [
        "friend", "love", "care", "together", "bond", "trust", "miss",
        "amigo", "amor", "juntos", "confianza", "extraño",
    ],
    "injustice": [
        "unfair", "unjust", "cheat", "steal", "corrupt", "lie", "exploit",
        "injusto", "trampa", "robar", "mentir", "explotar",
    ],
    "novelty": [
        "new", "discover", "surprise", "unexpected", "first time", "never seen",
        "nuevo", "descubrir", "sorpresa", "inesperado",
    ],
}


class EmotionalSchema(BaseModel):
    """Un patrón emocional aprendido."""

    trigger_category: str = Field(description="Categoría abstracta del estímulo")
    typical_emotion: PrimaryEmotion = Field(description="Emoción habitual ante este trigger")
    typical_intensity: float = Field(default=0.5, ge=0, le=1)
    activation_count: int = Field(default=1, ge=1)
    reinforcement_strength: float = Field(
        default=0.3, ge=0, le=1,
        description="Fuerza del patrón (sube con activaciones)",
    )
    adaptive: bool = Field(
        default=True,
        description="Si el schema es funcional o maladaptivo",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SchemaStore:
    """Almacén de schemas emocionales con formación automática."""

    FORMATION_THRESHOLD = 3  # Repeticiones antes de formar schema
    MAX_SCHEMAS = 20

    def __init__(self) -> None:
        self._schemas: list[EmotionalSchema] = []
        # Tracking de patrones pre-schema
        self._pattern_counts: dict[tuple[str, str], int] = {}  # (category, emotion) -> count
        self._pattern_intensities: dict[tuple[str, str], list[float]] = {}

    def categorize_stimulus(self, stimulus: str) -> str | None:
        """Clasifica un estímulo en una categoría abstracta."""
        lower = stimulus.lower()
        best_category: str | None = None
        best_score = 0

        for category, keywords in _STIMULUS_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category if best_score > 0 else None

    def record_pattern(
        self,
        stimulus: str,
        emotion: PrimaryEmotion,
        intensity: float,
    ) -> EmotionalSchema | None:
        """Registra un patrón estímulo→emoción. Forma schema si se repite 3+ veces.

        Returns:
            Schema formado si se cruzó el umbral, None si solo se registró.
        """
        category = self.categorize_stimulus(stimulus)
        if category is None:
            return None

        key = (category, emotion.value)

        # Verificar si ya existe un schema para esta categoría
        existing = self._find_schema(category)
        if existing:
            # Reforzar o actualizar
            self._reinforce_schema(existing, emotion, intensity)
            return existing

        # Tracking pre-schema
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
        if key not in self._pattern_intensities:
            self._pattern_intensities[key] = []
        self._pattern_intensities[key].append(intensity)

        # ¿Suficientes repeticiones para formar schema?
        if self._pattern_counts[key] >= self.FORMATION_THRESHOLD:
            avg_intensity = sum(self._pattern_intensities[key]) / len(self._pattern_intensities[key])
            schema = EmotionalSchema(
                trigger_category=category,
                typical_emotion=emotion,
                typical_intensity=round(avg_intensity, 4),
                activation_count=self._pattern_counts[key],
                reinforcement_strength=0.3,
            )
            self._schemas.append(schema)

            # Limpiar tracking
            del self._pattern_counts[key]
            del self._pattern_intensities[key]

            # Limitar número de schemas
            if len(self._schemas) > self.MAX_SCHEMAS:
                # Eliminar el menos reforzado
                self._schemas.sort(key=lambda s: s.reinforcement_strength)
                self._schemas.pop(0)

            return schema

        return None

    def check_priming(self, stimulus: str) -> tuple[PrimaryEmotion | None, float]:
        """Verifica si hay un schema que haga priming para este estímulo.

        Returns:
            (emotion primed, intensity amplification) o (None, 0.0) si no hay priming.
        """
        category = self.categorize_stimulus(stimulus)
        if category is None:
            return None, 0.0

        schema = self._find_schema(category)
        if schema is None:
            return None, 0.0

        # Priming: el schema sugiere una emoción y amplifica la intensidad
        amplification = schema.reinforcement_strength * 0.3
        # Maladaptive schemas amplifican MÁS (sobre-reacción)
        if not schema.adaptive:
            amplification *= 1.5

        return schema.typical_emotion, min(amplification, 0.4)

    def _find_schema(self, category: str) -> EmotionalSchema | None:
        """Busca un schema por categoría."""
        for s in self._schemas:
            if s.trigger_category == category:
                return s
        return None

    def _reinforce_schema(
        self,
        schema: EmotionalSchema,
        emotion: PrimaryEmotion,
        intensity: float,
    ) -> None:
        """Refuerza un schema existente con nueva activación."""
        schema.activation_count += 1
        schema.last_activated = datetime.now(timezone.utc)

        if emotion == schema.typical_emotion:
            # Misma emoción: refuerza el patrón
            schema.reinforcement_strength = _clamp(
                schema.reinforcement_strength + 0.05, 0, 1,
            )
            # Actualizar intensidad típica (promedio móvil)
            schema.typical_intensity = round(
                schema.typical_intensity * 0.8 + intensity * 0.2, 4,
            )
        else:
            # Emoción diferente: debilita el patrón
            schema.reinforcement_strength = _clamp(
                schema.reinforcement_strength - 0.1, 0, 1,
            )

        # Detectar maladaptividad: si intensity es consistentemente alta (>0.7)
        # y la emoción es negativa, marcar como maladaptivo
        if (schema.typical_intensity > 0.7
            and schema.typical_emotion in (
                PrimaryEmotion.ANGER, PrimaryEmotion.FEAR,
                PrimaryEmotion.ANXIETY, PrimaryEmotion.HELPLESSNESS,
            )
            and schema.activation_count > 5):
            schema.adaptive = False

    @property
    def schemas(self) -> list[EmotionalSchema]:
        return list(self._schemas)

    def __len__(self) -> int:
        return len(self._schemas)
