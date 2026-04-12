"""LLM Provider - Interfaz abstracta."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interfaz base para proveedores de LLM."""

    @property
    def supports_steering(self) -> bool:
        """Whether this provider supports steering vector injection.

        Steering requires direct access to model hidden states during forward pass.
        Only local models loaded via transformers/llama-cpp-python support this.
        HTTP-based providers (Ollama, Claude, OpenAI) return False.
        """
        return False

    @property
    def steerable_model(self) -> "object | None":
        """Return the underlying model object for steering hook registration.

        Only available when supports_steering is True. Returns None otherwise.
        Must be a transformers-compatible model with .model.layers attribute.
        """
        return None

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        think: bool = True,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
    ) -> str:
        """Genera una respuesta del LLM.

        Args:
            system_prompt: Instrucciones del sistema.
            messages: Lista de mensajes [{"role": "user"|"assistant", "content": "..."}].
            temperature: Temperatura de sampling (0.0-1.5). None = default del provider.
            think: Whether to allow model thinking/reasoning. False = faster.
            top_p: Nucleus sampling threshold [0.0-1.0]. None = default.
            top_k: Top-k sampling. None = default.
            repetition_penalty: Repetition penalty [1.0+]. None = default.
            presence_penalty: Presence penalty [-2.0 to 2.0]. None = default.
            frequency_penalty: Frequency penalty [-2.0 to 2.0]. None = default.
            logit_bias: Token ID → bias value mapping. Shifts token probabilities
                in logit space. Ollama supports this natively. For TransformersProvider
                it's applied via a LogitsProcessor. Cloud APIs ignore this.

        Returns:
            Texto de respuesta del LLM.

        Note:
            Not all providers support all sampling parameters. Unsupported params
            are silently ignored. Ollama supports temperature, top_p, top_k,
            repetition_penalty, presence_penalty, frequency_penalty natively.
            TransformersProvider supports temperature, top_p, top_k, repetition_penalty.
            Cloud APIs (Claude, OpenAI) typically only support temperature and top_p.
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Genera un embedding vector para el texto dado.

        Args:
            text: Texto a embeddear.

        Returns:
            Vector de embedding.
        """
        ...
