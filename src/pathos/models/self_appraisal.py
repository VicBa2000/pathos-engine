"""Self-Appraisal Result — evaluación secundaria de la propia respuesta.

Lazarus secondary appraisal: después de actuar, el sistema evalúa su propia
respuesta contra sus valores. Si hay conflicto, señala que debe re-generar.
"""

from dataclasses import dataclass, field


@dataclass
class SelfAppraisalResult:
    """Resultado de evaluar la propia respuesta del agente."""

    applied: bool = False
    value_alignment: float = 1.0  # 0=viola valores, 1=coherente
    emotional_coherence: float = 1.0  # 0=respuesta contradice estado, 1=coherente
    predicted_self_valence: float = 0.0  # cómo se siente el agente tras decir esto
    should_regenerate: bool = False
    reason: str = ""
    original_response: str = ""  # solo se llena si se re-generó
    adjustments: list[str] = field(default_factory=list)  # qué se ajustó
