"""Emotional Memory Store - Almacena y recupera memorias emocionales.

Fase 3: busqueda semantica con embeddings + cosine similarity.
Fallback a keyword matching si no hay embeddings disponibles.
"""

import logging
import math
import re
import uuid

from pathos.llm.base import LLMProvider

logger = logging.getLogger(__name__)
from pathos.models.emotion import EmotionalState
from pathos.models.memory import EmotionalMemory

# Intensidad minima para que una experiencia se guarde como memoria
MEMORY_THRESHOLD = 0.3

# Amplificacion maxima por memorias
MAX_AMPLIFICATION = 0.5

# Umbral de similitud semantica (cosine similarity)
SEMANTIC_SIMILARITY_THRESHOLD = 0.7

# Umbral de similitud por keywords (mas bajo porque es menos preciso)
KEYWORD_SIMILARITY_THRESHOLD = 0.3

# Limite maximo de memorias almacenadas (eviction por menor intensidad)
MAX_MEMORIES = 200

# Palabras comunes a ignorar
_STOP_WORDS = {
    "el", "la", "los", "las", "un", "una", "de", "del", "en", "con", "por",
    "para", "que", "es", "no", "si", "se", "al", "lo", "le", "me", "te",
    "su", "mi", "tu", "ya", "muy", "mas", "the", "a", "an", "is", "are",
    "to", "of", "in", "and", "or", "it", "i", "you", "he", "she", "we",
    "do", "not", "but", "was", "be", "have", "has", "this", "that", "my",
}


def _extract_keywords(text: str) -> list[str]:
    """Extrae keywords significativas de un texto."""
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


def _keyword_similarity(keywords_a: list[str], keywords_b: list[str]) -> float:
    """Calcula similitud simple entre dos listas de keywords (Jaccard)."""
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = set(keywords_a)
    set_b = set(keywords_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcula cosine similarity entre dos vectores."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmotionalMemoryStore:
    """Almacen de memorias emocionales con busqueda semantica."""

    def __init__(self) -> None:
        self._memories: list[EmotionalMemory] = []

    async def store(
        self,
        stimulus: str,
        state: EmotionalState,
        llm: LLMProvider | None = None,
    ) -> EmotionalMemory | None:
        """Guarda una memoria si la intensidad supera el umbral.

        Args:
            stimulus: Texto del estimulo.
            state: Estado emocional resultante.
            llm: Provider para generar embeddings (opcional).
        """
        if state.intensity < MEMORY_THRESHOLD:
            return None

        # Generar embedding si hay provider disponible
        embedding: list[float] = []
        if llm is not None:
            try:
                embedding = await llm.embed(stimulus)
            except Exception:
                logger.warning("Embedding failed for memory store, using keyword fallback", exc_info=True)

        memory = EmotionalMemory(
            id=str(uuid.uuid4()),
            stimulus=stimulus[:200],
            intensity_at_time=state.intensity,
            valence_at_time=state.valence,
            primary_emotion=state.primary_emotion.value,
            keywords=_extract_keywords(stimulus),
            embedding=embedding,
        )
        self._memories.append(memory)
        if len(self._memories) > MAX_MEMORIES:
            # Evict lowest intensity memory
            weakest = min(range(len(self._memories)), key=lambda i: self._memories[i].intensity_at_time)
            self._memories.pop(weakest)
        return memory

    async def check_amplification(
        self,
        stimulus: str,
        llm: LLMProvider | None = None,
    ) -> float:
        """Busca memorias similares y calcula amplificacion.

        Usa cosine similarity si hay embeddings, fallback a keywords.

        Args:
            stimulus: Texto del estimulo actual.
            llm: Provider para generar embedding del estimulo.

        Returns:
            Factor de amplificacion (0.0 a MAX_AMPLIFICATION).
        """
        if not self._memories:
            return 0.0

        # Intentar busqueda semantica
        stimulus_embedding: list[float] = []
        if llm is not None:
            try:
                stimulus_embedding = await llm.embed(stimulus)
            except Exception:
                logger.warning("Embedding failed for amplification check, using keyword fallback", exc_info=True)

        amplification = 0.0
        for memory in self._memories:
            similarity = self._compute_similarity(
                stimulus, stimulus_embedding, memory,
            )
            threshold = (
                SEMANTIC_SIMILARITY_THRESHOLD
                if stimulus_embedding and memory.embedding
                else KEYWORD_SIMILARITY_THRESHOLD
            )
            if similarity > threshold:
                amplification += memory.intensity_at_time * similarity * 0.3

        return min(amplification, MAX_AMPLIFICATION)

    def _compute_similarity(
        self,
        stimulus: str,
        stimulus_embedding: list[float],
        memory: EmotionalMemory,
    ) -> float:
        """Calcula similitud entre estimulo y memoria.

        Prioriza cosine similarity si ambos tienen embeddings.
        Fallback a keyword similarity.
        """
        if stimulus_embedding and memory.embedding:
            return cosine_similarity(stimulus_embedding, memory.embedding)
        # Fallback: keyword matching
        stimulus_keywords = _extract_keywords(stimulus)
        return _keyword_similarity(stimulus_keywords, memory.keywords)

    @property
    def memories(self) -> list[EmotionalMemory]:
        return list(self._memories)

    def __len__(self) -> int:
        return len(self._memories)
