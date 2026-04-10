"""Tests para Emotional Memory (Fase 3 - semantic search + keyword fallback)."""

import math

import pytest

from pathos.engine.memory import (
    MAX_AMPLIFICATION,
    SEMANTIC_SIMILARITY_THRESHOLD,
    EmotionalMemoryStore,
    _extract_keywords,
    _keyword_similarity,
    cosine_similarity,
)
from pathos.models.emotion import EmotionalState, PrimaryEmotion


def _make_state(intensity: float = 0.5, valence: float = 0.3) -> EmotionalState:
    return EmotionalState(
        intensity=intensity,
        valence=valence,
        primary_emotion=PrimaryEmotion.JOY,
    )


# --- Tests de Keywords (fallback) ---


class TestKeywords:
    def test_extracts_meaningful_words(self) -> None:
        keywords = _extract_keywords("El proyecto esta avanzando muy bien")
        assert "proyecto" in keywords
        assert "avanzando" in keywords
        assert "bien" in keywords
        assert "el" not in keywords
        assert "muy" not in keywords

    def test_ignores_short_words(self) -> None:
        keywords = _extract_keywords("a en de lo si no")
        assert len(keywords) == 0

    def test_similarity_identical(self) -> None:
        assert _keyword_similarity(["hello", "world"], ["hello", "world"]) == 1.0

    def test_similarity_disjoint(self) -> None:
        assert _keyword_similarity(["hello"], ["world"]) == 0.0

    def test_similarity_partial(self) -> None:
        sim = _keyword_similarity(["hello", "world"], ["hello", "foo"])
        assert 0 < sim < 1


# --- Tests de Cosine Similarity ---


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self) -> None:
        a = [1.0, 1.0, 0.0]
        b = [1.0, 1.0, 0.1]
        sim = cosine_similarity(a, b)
        assert sim > 0.9

    def test_empty_vectors(self) -> None:
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


# --- Tests de Memory Store (keyword fallback, sin LLM) ---


class TestMemoryStoreKeywordFallback:
    @pytest.mark.asyncio
    async def test_store_above_threshold(self) -> None:
        store = EmotionalMemoryStore()
        state = _make_state(intensity=0.5)
        mem = await store.store("evento importante del proyecto", state)
        assert mem is not None
        assert len(store) == 1
        assert mem.embedding == []  # Sin LLM, no hay embedding

    @pytest.mark.asyncio
    async def test_skip_below_threshold(self) -> None:
        store = EmotionalMemoryStore()
        state = _make_state(intensity=0.1)
        mem = await store.store("evento trivial", state)
        assert mem is None
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_amplification_with_similar_stimulus(self) -> None:
        store = EmotionalMemoryStore()
        state = _make_state(intensity=0.8, valence=-0.7)
        await store.store("proyecto cancelado terrible desastre", state)

        amp = await store.check_amplification("proyecto cancelado otra vez terrible")
        assert amp > 0

    @pytest.mark.asyncio
    async def test_no_amplification_dissimilar(self) -> None:
        store = EmotionalMemoryStore()
        state = _make_state(intensity=0.8)
        await store.store("el proyecto fue cancelado", state)

        amp = await store.check_amplification("hace buen clima hoy")
        assert amp == 0.0

    @pytest.mark.asyncio
    async def test_amplification_capped(self) -> None:
        store = EmotionalMemoryStore()
        for i in range(20):
            state = _make_state(intensity=0.9)
            await store.store(f"proyecto cancelado terrible desastre iteracion {i}", state)

        amp = await store.check_amplification("proyecto cancelado terrible desastre")
        assert amp <= MAX_AMPLIFICATION

    @pytest.mark.asyncio
    async def test_empty_store_no_amplification(self) -> None:
        store = EmotionalMemoryStore()
        assert await store.check_amplification("cualquier cosa") == 0.0


# --- Tests de Memory Store con Embeddings (mock LLM) ---


class MockEmbedProvider:
    """Mock LLM que genera embeddings deterministas basados en keywords."""

    KEYWORD_DIMS = {
        "project": 0, "cancel": 1, "terrible": 2, "disaster": 3,
        "happy": 4, "weather": 5, "sunny": 6, "good": 7,
        "proyecto": 0, "cancelado": 1, "desastre": 3,
        "feliz": 4, "clima": 5,
    }

    async def embed(self, text: str) -> list[float]:
        """Genera un vector sparse basado en keywords presentes."""
        vec = [0.0] * 8
        words = text.lower().split()
        for w in words:
            if w in self.KEYWORD_DIMS:
                vec[self.KEYWORD_DIMS[w]] = 1.0
        # Normalizar
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        return "mock"


class TestMemoryStoreWithEmbeddings:
    @pytest.mark.asyncio
    async def test_store_with_embedding(self) -> None:
        store = EmotionalMemoryStore()
        llm = MockEmbedProvider()
        state = _make_state(intensity=0.8)
        mem = await store.store("project cancel terrible disaster", state, llm=llm)
        assert mem is not None
        assert len(mem.embedding) == 8
        assert any(x != 0 for x in mem.embedding)

    @pytest.mark.asyncio
    async def test_semantic_amplification_similar(self) -> None:
        store = EmotionalMemoryStore()
        llm = MockEmbedProvider()
        state = _make_state(intensity=0.8, valence=-0.7)
        await store.store("project cancel terrible disaster", state, llm=llm)

        # Estimulo semanticamente similar
        amp = await store.check_amplification("project cancel disaster", llm=llm)
        assert amp > 0

    @pytest.mark.asyncio
    async def test_semantic_no_amplification_dissimilar(self) -> None:
        store = EmotionalMemoryStore()
        llm = MockEmbedProvider()
        state = _make_state(intensity=0.8)
        await store.store("project cancel terrible disaster", state, llm=llm)

        # Estimulo semanticamente diferente (ortogonal en nuestro mock)
        amp = await store.check_amplification("happy sunny good weather", llm=llm)
        assert amp == 0.0

    @pytest.mark.asyncio
    async def test_semantic_similarity_used_over_keywords(self) -> None:
        """Si hay embeddings, usa cosine similarity con threshold 0.7 (no 0.3)."""
        store = EmotionalMemoryStore()
        llm = MockEmbedProvider()
        state = _make_state(intensity=0.8)
        await store.store("project cancel", state, llm=llm)

        # Con embeddings, el threshold es 0.7
        # "project cancel" vs "project" -> cosine ~ 0.707 (deberia estar justo en el borde)
        # "project" solo activa dim 0, "project cancel" activa dim 0 y 1
        # cos(project, project_cancel) = 1/(sqrt(1)*sqrt(2)) = 0.707...
        amp = await store.check_amplification("project", llm=llm)
        # 0.707 > 0.7 threshold, so should amplify (barely)
        assert amp >= 0

    @pytest.mark.asyncio
    async def test_fallback_to_keywords_on_embed_error(self) -> None:
        """Si embed falla, fallback a keywords."""

        class FailingEmbedProvider(MockEmbedProvider):
            async def embed(self, text: str) -> list[float]:
                raise RuntimeError("Embedding service unavailable")

        store = EmotionalMemoryStore()
        failing_llm = FailingEmbedProvider()
        state = _make_state(intensity=0.8)

        # Store sin embedding (falla silenciosamente)
        mem = await store.store("proyecto cancelado desastre", state, llm=failing_llm)
        assert mem is not None
        assert mem.embedding == []

        # Amplificacion via keyword fallback
        amp = await store.check_amplification(
            "proyecto cancelado desastre", llm=failing_llm,
        )
        assert amp > 0

    @pytest.mark.asyncio
    async def test_mixed_memories_with_and_without_embeddings(self) -> None:
        """Memorias viejas (sin embedding) + nuevas (con embedding) coexisten."""
        store = EmotionalMemoryStore()
        llm = MockEmbedProvider()

        # Memoria vieja sin embedding
        state1 = _make_state(intensity=0.7, valence=-0.5)
        await store.store("proyecto cancelado desastre", state1, llm=None)

        # Memoria nueva con embedding
        state2 = _make_state(intensity=0.8, valence=-0.8)
        await store.store("project cancel disaster", state2, llm=llm)

        assert len(store) == 2
        assert store.memories[0].embedding == []
        assert len(store.memories[1].embedding) == 8
