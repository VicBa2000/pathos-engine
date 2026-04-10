"""Somatic Markers Engine - Marcadores emocionales en decisiones.

Implementa la teoria de Damasio: las decisiones se "marcan" emocionalmente.
Cuando el agente enfrenta una decision similar a una anterior, el marcador
somatico genera un bias pre-racional ("gut feeling").

Flujo:
1. Agente responde al usuario (turno N)
2. Usuario reacciona (turno N+1) — reaccion positiva o negativa
3. Se forma un marcador: contexto + valencia de la reaccion
4. En futuras situaciones similares, el marcador sesga la respuesta

Basado en:
- Damasio (1994) "Descartes' Error"
- Bechara et al. (2000) "Emotion, Decision Making and the OFC"
"""

from pathos.engine.emotional_schemas import _STIMULUS_CATEGORIES
from pathos.models.somatic import SomaticMarker, SomaticMarkerStore


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def categorize_stimulus(stimulus: str) -> tuple[str | None, list[str]]:
    """Categoriza un estimulo y extrae keywords relevantes.

    Reutiliza las categorias del schema system.

    Returns:
        (category, keywords) — category puede ser None si no matchea.
    """
    lower = stimulus.lower()
    best_category: str | None = None
    best_score = 0
    matched_keywords: list[str] = []

    for category, keywords in _STIMULUS_CATEGORIES.items():
        matches = [kw for kw in keywords if kw in lower]
        score = len(matches)
        if score > best_score:
            best_score = score
            best_category = category
            matched_keywords = matches

    return best_category, matched_keywords


def register_pending_decision(
    store: SomaticMarkerStore,
    stimulus: str,
) -> SomaticMarkerStore:
    """Registra que el agente tomo una decision en este turno.

    La decision queda "pendiente" hasta que veamos la reaccion del usuario.

    Args:
        store: Store de marcadores actual.
        stimulus: Estimulo al que el agente respondio.

    Returns:
        Store actualizado con la decision pendiente.
    """
    category, keywords = categorize_stimulus(stimulus)
    if category is None:
        # Sin categoria clara, no registrar
        return SomaticMarkerStore(
            markers=store.markers,
            pending_category=None,
            pending_keywords=[],
        )

    return SomaticMarkerStore(
        markers=store.markers,
        pending_category=category,
        pending_keywords=keywords,
    )


def evaluate_user_reaction(
    store: SomaticMarkerStore,
    user_valence: float,
    turn_count: int,
) -> SomaticMarkerStore:
    """Evalua la reaccion del usuario al turno anterior y forma/refuerza marcadores.

    Si habia una decision pendiente, la reaccion del usuario determina
    si se forma un marcador positivo o negativo.

    Args:
        store: Store con posible decision pendiente.
        user_valence: Valencia detectada del usuario en este turno (-1 a 1).
        turn_count: Turno actual.

    Returns:
        Store con marcador formado/reforzado.
    """
    if store.pending_category is None:
        return store

    category = store.pending_category
    keywords = store.pending_keywords

    # Buscar si ya existe un marcador para esta categoria
    existing_idx = _find_marker(store.markers, category)
    markers = list(store.markers)

    if existing_idx is not None:
        # Reforzar o debilitar marcador existente
        old = markers[existing_idx]
        markers[existing_idx] = _update_marker(old, user_valence, turn_count)
    else:
        # Solo formar marcador si la senal es suficientemente clara
        if abs(user_valence) > 0.1:
            new_marker = SomaticMarker(
                stimulus_category=category,
                stimulus_keywords=keywords,
                valence_tag=_clamp(user_valence, -1, 1),
                strength=min(0.3 + abs(user_valence) * 0.3, 0.8),
                formation_turn=turn_count,
                reinforcement_count=1,
            )
            markers.append(new_marker)
            # Limitar cantidad de marcadores
            if len(markers) > 15:
                # Eliminar el mas debil
                weakest = min(range(len(markers)), key=lambda i: markers[i].strength)
                markers.pop(weakest)

    return SomaticMarkerStore(
        markers=markers,
        pending_category=None,
        pending_keywords=[],
    )


def _find_marker(markers: list[SomaticMarker], category: str) -> int | None:
    """Busca un marcador existente para la categoria dada."""
    for i, m in enumerate(markers):
        if m.stimulus_category == category:
            return i
    return None


def _update_marker(
    marker: SomaticMarker,
    new_valence: float,
    turn: int,
) -> SomaticMarker:
    """Actualiza un marcador existente con nueva evidencia.

    Si la nueva valencia es congruente, refuerza. Si es opuesta, debilita.
    """
    same_direction = (marker.valence_tag >= 0 and new_valence >= 0) or \
                     (marker.valence_tag < 0 and new_valence < 0)

    if same_direction:
        # Reforzar: mover valencia hacia la nueva, subir strength
        blend = 0.3  # 30% nueva evidencia
        new_tag = marker.valence_tag * (1 - blend) + new_valence * blend
        new_strength = _clamp(marker.strength + 0.1, 0, 1)
        new_count = marker.reinforcement_count + 1
    else:
        # Contradiccion: debilitar el marcador
        new_tag = marker.valence_tag * 0.7 + new_valence * 0.3
        new_strength = _clamp(marker.strength - 0.15, 0, 1)
        new_count = marker.reinforcement_count

    return SomaticMarker(
        stimulus_category=marker.stimulus_category,
        stimulus_keywords=marker.stimulus_keywords,
        valence_tag=round(_clamp(new_tag, -1, 1), 4),
        strength=round(new_strength, 4),
        formation_turn=marker.formation_turn,
        reinforcement_count=new_count,
    )


def compute_somatic_bias(
    store: SomaticMarkerStore,
    stimulus: str,
) -> tuple[float, str | None]:
    """Busca marcadores somaticos relevantes para el estimulo actual.

    Si hay un marcador que matchea, retorna un bias de valencia
    que representa el "gut feeling" del agente.

    Args:
        store: Store de marcadores.
        stimulus: Estimulo actual del usuario.

    Returns:
        (valence_bias, gut_feeling_reason)
        valence_bias: -0.2 a 0.2 (bias sutil pre-racional)
        gut_feeling_reason: Descripcion del marcador activado, o None.
    """
    category, _ = categorize_stimulus(stimulus)
    if category is None:
        return 0.0, None

    # Buscar marcador para esta categoria
    idx = _find_marker(store.markers, category)
    if idx is None:
        return 0.0, None

    marker = store.markers[idx]

    # Solo activar si el marcador es suficientemente fuerte
    if marker.strength < 0.2:
        return 0.0, None

    # Bias proporcional a valencia × fuerza
    # Factor 0.2 para mantener el bias sutil (pre-racional, no dominante)
    bias = marker.valence_tag * marker.strength * 0.2

    # Generar razon para el gut feeling
    if marker.valence_tag < -0.3 and marker.strength > 0.4:
        reason = f"marcador negativo en '{category}' (fuerza={marker.strength:.2f}, refuerzos={marker.reinforcement_count})"
    elif marker.valence_tag > 0.3 and marker.strength > 0.4:
        reason = f"marcador positivo en '{category}' (fuerza={marker.strength:.2f}, refuerzos={marker.reinforcement_count})"
    else:
        reason = None

    return round(_clamp(bias, -0.2, 0.2), 4), reason
