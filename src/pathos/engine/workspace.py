"""Global Workspace Engine - Pilar 2 de ANIMA.

Implementa consciencia funcional via competicion por un workspace limitado.
Cada sistema emocional genera candidatos. Los candidatos compiten en 3 fases:
  1. Filtrado por umbral (descartar ruido)
  2. Formacion de coaliciones (candidatos que se refuerzan)
  3. Seleccion top-K (los K mas salientes entran al workspace)

Lo que entra al workspace es "consciente" para el agente y alimenta
el behavior modifier. Lo que no entra va al preconsciente y opera
indirectamente (priming, mood influence, somatic echo).

Sistema TOGGLEABLE (default OFF). Si esta OFF, todos los sistemas
contribuyen directamente como en v4.

Basado en:
- Bernard Baars (Global Workspace Theory, 1988)
- Stanislas Dehaene (Global Neuronal Workspace, 2014)
- Giulio Tononi (IIT, 2004) — integracion informacional
"""

from __future__ import annotations

from pathos.models.workspace import (
    Coalition,
    ConsciousnessState,
    PreconsciousBuffer,
    WorkspaceCandidate,
    WorkspaceResult,
)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

WORKSPACE_CAPACITY: int = 5
NOISE_THRESHOLD: float = 0.1
COALITION_COHERENCE_BONUS: float = 1.3  # 30% bonus para coaliciones coherentes
PRIMING_INCREMENT: float = 0.05  # +0.05 saliency por turno en preconsciente
MOOD_INFLUENCE_WEIGHT: float = 0.3  # Preconsciente contribuye 30% al mood
SOMATIC_ECHO_WEIGHT: float = 0.4  # Preconsciente contribuye 40% a tension somatica


# ---------------------------------------------------------------------------
# Paso 2.1: Generacion de candidates desde datos del pipeline
# ---------------------------------------------------------------------------

def generate_candidate(
    source: str,
    content: str,
    urgency: float,
    relevance: float,
    emotional_intensity: float,
    emotion_tag: str = "neutral",
    category: str = "general",
) -> WorkspaceCandidate:
    """Crea un WorkspaceCandidate con saliency computada."""
    candidate = WorkspaceCandidate(
        source=source,
        content=content,
        urgency=_clamp(urgency, 0, 1),
        relevance=_clamp(relevance, 0, 1),
        emotional_intensity=_clamp(emotional_intensity, 0, 1),
        emotion_tag=emotion_tag,
        category=category,
    )
    candidate.compute_saliency()
    return candidate


# ---------------------------------------------------------------------------
# Paso 2.2: Competicion y Seleccion
# ---------------------------------------------------------------------------

def filter_noise(
    candidates: list[WorkspaceCandidate],
    threshold: float = NOISE_THRESHOLD,
) -> tuple[list[WorkspaceCandidate], int]:
    """Fase 1: Filtra candidatos con saliency < umbral (ruido).

    Returns:
        (candidatos_filtrados, cantidad_descartada)
    """
    passed = [c for c in candidates if c.saliency >= threshold]
    filtered = len(candidates) - len(passed)
    return passed, filtered


def form_coalitions(candidates: list[WorkspaceCandidate]) -> list[Coalition]:
    """Fase 2: Agrupa candidatos que se refuerzan mutuamente.

    Criterios de coalicion:
    - Misma emotion_tag (ej: dos candidatos con "anger")
    - Misma category (ej: dos candidatos sobre "relationship")
    - Misma source NO forma coalicion (seria redundante)

    Los candidatos que no forman coalicion quedan como coaliciones
    de 1 miembro (sin bonus de coherencia).
    """
    if not candidates:
        return []

    # Intentar agrupar por emotion_tag primero (vinculo mas fuerte)
    emotion_groups: dict[str, list[WorkspaceCandidate]] = {}
    ungrouped: list[WorkspaceCandidate] = []

    for c in candidates:
        if c.emotion_tag != "neutral":
            emotion_groups.setdefault(c.emotion_tag, []).append(c)
        else:
            ungrouped.append(c)

    coalitions: list[Coalition] = []
    grouped_ids: set[int] = set()

    # Coaliciones por emocion (2+ miembros con misma emocion)
    for tag, members in emotion_groups.items():
        if len(members) >= 2:
            coalition = Coalition(
                members=members,
                coherence=COALITION_COHERENCE_BONUS,
                bond_type="emotion",
            )
            coalition.compute_effective_saliency()
            coalitions.append(coalition)
            grouped_ids.update(id(m) for m in members)
        else:
            ungrouped.extend(members)

    # Intentar agrupar ungrouped por category
    category_groups: dict[str, list[WorkspaceCandidate]] = {}
    still_alone: list[WorkspaceCandidate] = []

    for c in ungrouped:
        if id(c) not in grouped_ids and c.category != "general":
            category_groups.setdefault(c.category, []).append(c)
        elif id(c) not in grouped_ids:
            still_alone.append(c)

    for cat, members in category_groups.items():
        if len(members) >= 2:
            coalition = Coalition(
                members=members,
                coherence=COALITION_COHERENCE_BONUS * 0.85,  # Menor bonus que emocion
                bond_type="category",
            )
            coalition.compute_effective_saliency()
            coalitions.append(coalition)
        else:
            still_alone.extend(members)

    # Candidatos solitarios: coalicion de 1 sin bonus
    for c in still_alone:
        coalition = Coalition(
            members=[c],
            coherence=1.0,
            bond_type="none",
        )
        coalition.compute_effective_saliency()
        coalitions.append(coalition)

    return coalitions


def select_workspace(
    candidates: list[WorkspaceCandidate],
    capacity: int = WORKSPACE_CAPACITY,
    previous_sources: list[str] | None = None,
) -> WorkspaceResult:
    """Pipeline completo de seleccion del workspace.

    Ejecuta las 3 fases:
    1. Filtrado de ruido
    2. Formacion de coaliciones
    3. Seleccion top-K

    Args:
        candidates: Todos los candidatos del turno actual.
        capacity: Maximo de candidatos en el workspace (default 5).
        previous_sources: Sources del workspace anterior (para estabilidad).

    Returns:
        WorkspaceResult con conscious, preconscious, y metricas.
    """
    if not candidates:
        return WorkspaceResult()

    total = len(candidates)

    # Fase 1: Filtrar ruido
    filtered, noise_count = filter_noise(candidates)

    if not filtered:
        return WorkspaceResult(
            total_candidates=total,
            filtered_noise=noise_count,
        )

    # Fase 2: Formar coaliciones
    coalitions = form_coalitions(filtered)

    # Fase 3: Seleccion top-K
    # Ordenar coaliciones por effective_saliency
    coalitions.sort(key=lambda c: c.effective_saliency, reverse=True)

    conscious: list[WorkspaceCandidate] = []
    preconscious: list[WorkspaceCandidate] = []
    coalitions_formed = sum(1 for c in coalitions if len(c.members) > 1)

    for coalition in coalitions:
        if len(conscious) < capacity:
            # Todos los miembros de la coalicion entran al workspace
            remaining_slots = capacity - len(conscious)
            entering = coalition.members[:remaining_slots]
            conscious.extend(entering)
            # Si la coalicion no cupo completa, el resto al preconsciente
            if len(coalition.members) > remaining_slots:
                preconscious.extend(coalition.members[remaining_slots:])
        else:
            preconscious.extend(coalition.members)

    # Calcular integracion y estabilidad
    integration = compute_integration(conscious)
    stability = compute_stability(conscious, previous_sources or [])

    return WorkspaceResult(
        conscious=conscious,
        preconscious=preconscious,
        coalitions_formed=coalitions_formed,
        integration_score=round(integration, 4),
        workspace_stability=round(stability, 4),
        total_candidates=total,
        filtered_noise=noise_count,
    )


# ---------------------------------------------------------------------------
# Integracion informacional (IIT-inspired)
# ---------------------------------------------------------------------------

def compute_integration(workspace: list[WorkspaceCandidate]) -> float:
    """Computa la integracion informacional del workspace.

    Mide cuanto se CONECTAN los contenidos entre si.
    Alta integracion = experiencia unificada (coherente).
    Baja integracion = fragmentacion (puede triggear self-inquiry).

    Se basa en cuantos pares de candidatos comparten emocion o categoria.
    """
    if len(workspace) < 2:
        return 1.0 if workspace else 0.0

    n = len(workspace)
    pairs = n * (n - 1) / 2
    connections = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = workspace[i], workspace[j]
            # Conexion por emocion compartida
            if a.emotion_tag == b.emotion_tag and a.emotion_tag != "neutral":
                connections += 1.0
            # Conexion por categoria compartida
            elif a.category == b.category and a.category != "general":
                connections += 0.7
            # Conexion parcial si ambos tienen alta urgency (crisis coherence)
            elif a.urgency > 0.6 and b.urgency > 0.6:
                connections += 0.3

    return _clamp(connections / pairs, 0, 1) if pairs > 0 else 0.0


def compute_stability(
    current_workspace: list[WorkspaceCandidate],
    previous_sources: list[str],
) -> float:
    """Computa la estabilidad del workspace (cuanto cambio vs turno anterior).

    1.0 = completamente estable (mismos sources).
    0.0 = completamente nuevo (ningun source en comun).
    """
    if not previous_sources or not current_workspace:
        return 0.0

    current_sources = {c.source for c in current_workspace}
    overlap = len(current_sources & set(previous_sources))
    total = max(len(current_sources), len(previous_sources))

    return overlap / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Paso 2.3: Preconsciente e Influencia Indirecta
# ---------------------------------------------------------------------------

def apply_priming(
    preconscious: PreconsciousBuffer,
) -> list[WorkspaceCandidate]:
    """Aplica priming temporal: candidatos persistentes ganan saliency.

    Candidatos con 3+ turnos en el preconsciente incrementan su saliency.
    Eventualmente pueden "irrumpir" en el workspace.

    Returns:
        Lista de candidatos que superaron el umbral de irrupcion (saliency > 0.5).
    """
    irrupting: list[WorkspaceCandidate] = []

    for candidate in preconscious.candidates:
        if candidate.preconscious_turns >= 3:
            bonus = PRIMING_INCREMENT * (candidate.preconscious_turns - 2)
            candidate.saliency = _clamp(candidate.saliency + bonus, 0, 1)

            # Si supera umbral de irrupcion, marca para entrar al workspace
            if candidate.saliency > 0.5:
                irrupting.append(candidate)

    return irrupting


def compute_mood_influence(preconscious: PreconsciousBuffer) -> tuple[float, float]:
    """Computa la contribucion del preconsciente al mood.

    Los candidatos preconscientes contribuyen al mood con 30% del peso.
    Esto crea "estados de animo inexplicables" — el agente se siente
    de cierta forma pero no sabe por que.

    Returns:
        (valence_contribution, arousal_contribution)
    """
    if not preconscious.candidates:
        return 0.0, 0.0

    # Mapeo simple de emotion_tag a valence
    _EMOTION_VALENCE: dict[str, float] = {
        "joy": 0.7, "excitement": 0.6, "gratitude": 0.7, "hope": 0.5,
        "contentment": 0.5, "relief": 0.4,
        "anger": -0.6, "frustration": -0.4, "fear": -0.6, "anxiety": -0.4,
        "sadness": -0.6, "helplessness": -0.7, "disappointment": -0.4,
        "surprise": 0.0, "alertness": 0.0, "contemplation": 0.1,
        "neutral": 0.0,
    }

    total_weight = 0.0
    weighted_valence = 0.0
    weighted_arousal = 0.0

    for c in preconscious.candidates:
        weight = c.saliency * MOOD_INFLUENCE_WEIGHT
        valence = _EMOTION_VALENCE.get(c.emotion_tag, 0.0)
        weighted_valence += valence * weight
        weighted_arousal += c.emotional_intensity * weight
        total_weight += weight

    if total_weight > 0:
        v = _clamp(weighted_valence / total_weight, -1, 1)
        a = _clamp(weighted_arousal / total_weight, 0, 1)
    else:
        v, a = 0.0, 0.0

    preconscious.mood_valence_contribution = round(v * MOOD_INFLUENCE_WEIGHT, 4)
    preconscious.mood_arousal_contribution = round(a * MOOD_INFLUENCE_WEIGHT, 4)

    return preconscious.mood_valence_contribution, preconscious.mood_arousal_contribution


def compute_somatic_echo(preconscious: PreconsciousBuffer) -> float:
    """Computa el eco somatico del preconsciente.

    El body state refleja TODOS los candidatos (conscientes y preconscientes).
    El agente puede notar tension corporal sin saber su causa.

    Returns:
        tension_echo: contribucion a tension corporal (0-1)
    """
    if not preconscious.candidates:
        preconscious.somatic_tension_echo = 0.0
        return 0.0

    # Tension proporcional a urgency * emotional_intensity de candidatos negativos
    tension = 0.0
    for c in preconscious.candidates:
        if c.emotion_tag in ("anger", "frustration", "fear", "anxiety", "sadness", "helplessness"):
            tension += c.urgency * c.emotional_intensity * SOMATIC_ECHO_WEIGHT

    echo = _clamp(tension / max(len(preconscious.candidates), 1), 0, 1)
    preconscious.somatic_tension_echo = round(echo, 4)
    return echo


# ---------------------------------------------------------------------------
# Workspace prompt generation
# ---------------------------------------------------------------------------

def get_workspace_prompt(result: WorkspaceResult) -> str | None:
    """Genera texto para el behavior modifier desde el workspace.

    Solo los candidatos conscientes (en el workspace) influyen
    directamente en el prompt del LLM.
    """
    if not result.conscious:
        return None

    parts: list[str] = ["Yo consciente (workspace):"]
    for c in result.conscious:
        parts.append(f"  - [{c.source}] {c.content}")

    if result.integration_score < 0.3 and len(result.conscious) > 1:
        parts.append("  (Experiencia fragmentada — contenidos poco conectados)")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Orquestacion completa
# ---------------------------------------------------------------------------

def process_workspace_turn(
    consciousness: ConsciousnessState,
    candidates: list[WorkspaceCandidate],
    *,
    raw_mode: bool = False,
    extreme_mode: bool = False,
) -> ConsciousnessState:
    """Procesa un turno completo del workspace.

    1. Aplica priming a candidatos preconscientes persistentes
    2. Agrega irrupting candidates a la lista de candidatos
    3. Ejecuta la competicion (filter → coalitions → top-K)
    4. Actualiza el buffer preconsciente
    5. Computa influencias indirectas (mood, somatic)

    Mode adaptations:
    - Raw: capacity expands to 8 (more reaches consciousness, less repressed).
    - Extreme: workspace floods — ALL candidates go to conscious (no filtering).

    Args:
        consciousness: Estado de consciencia actual.
        candidates: Candidatos generados por los sistemas del pipeline.
        raw_mode: Si True, expande capacity a 8.
        extreme_mode: Si True, todo pasa a consciente (flood).

    Returns:
        ConsciousnessState actualizado.
    """
    if not consciousness.enabled:
        return consciousness

    # 1. Priming: candidatos preconscientes persistentes ganan saliency
    irrupting = apply_priming(consciousness.preconscious)

    # 2. Agregar irrupting candidates a los del turno
    all_candidates = list(candidates)
    for irrupt in irrupting:
        all_candidates.append(irrupt)
        consciousness.preconscious.remove_by_source(irrupt.source)

    # 3. Competicion por el workspace
    # Extreme: flood — everything to conscious, no filtering
    # Raw: expanded capacity (8 vs normal 5)
    effective_capacity = len(all_candidates) if extreme_mode else (8 if raw_mode else WORKSPACE_CAPACITY)
    result = select_workspace(
        all_candidates,
        capacity=effective_capacity,
        previous_sources=consciousness.previous_workspace_sources,
    )

    # 4. Actualizar preconsciente
    consciousness.preconscious.add_candidates(result.preconscious)

    # Remover del preconsciente los que entraron al workspace
    for c in result.conscious:
        consciousness.preconscious.remove_by_source(c.source)

    # 5. Influencias indirectas
    compute_mood_influence(consciousness.preconscious)
    compute_somatic_echo(consciousness.preconscious)

    # 6. Actualizar estado
    consciousness.current_result = result
    consciousness.previous_workspace_sources = [c.source for c in result.conscious]

    # Historial de integracion (max 20)
    consciousness.integration_history.append(result.integration_score)
    if len(consciousness.integration_history) > 20:
        consciousness.integration_history = consciousness.integration_history[-20:]

    return consciousness
