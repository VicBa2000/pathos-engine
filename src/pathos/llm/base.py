"""LLM Provider - Interfaz abstracta."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interfaz base para proveedores de LLM."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        think: bool = True,
    ) -> str:
        """Genera una respuesta del LLM.

        Args:
            system_prompt: Instrucciones del sistema.
            messages: Lista de mensajes [{"role": "user"|"assistant", "content": "..."}].
            temperature: Temperatura de sampling (0.0-1.0). None = default del provider.
            think: Whether to allow model thinking/reasoning. False = faster.

        Returns:
            Texto de respuesta del LLM.
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
