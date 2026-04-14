"""Motivational Autonomy Engine — Drives afectivos primarios.

Implementa el sistema de drives basado en Panksepp (SEEKING, CARE, PLAY,
PANIC_GRIEF) con goals autónomos que generan consecuencias emocionales.

drive = necesidad_no_satisfecha -> urgencia -> goal -> comportamiento
emocion_motivada = goal_outcome * stake
"""

from pathos.models.drives import (
    DRIVE_DECAY_RATE,
    DRIVE_KEYWORDS,
    DRIVE_PERSONALITY_MAP,
    FRUSTRATION_THRESHOLD_TURNS,
    MAX_ACTIVE_GOALS,
    SATISFACTION_DECAY_RATE,
    SATISFACTION_KEYWORDS,
    URGENCY_TRIGGER_THRESHOLD,
    Drive,
    DriveState,
    DriveUpdate,
    DrivesState,
    EmotionalImpact,
    Goal,
    GoalOutcome,
    GoalStatus,
)
from pathos.models.personality import PersonalityProfile


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Cuenta matches de keywords en texto, normalizado 0-1."""
    words = text.lower().split()
    if not keywords:
        return 0.0
    matches = sum(1 for w in words if any(kw in w for kw in keywords))
    return min(matches / max(len(keywords) * 0.2, 1), 1.0)


# --- Drive intensity computation ---


def compute_base_intensity(
    drive: Drive,
    personality: PersonalityProfile,
    rapport: float = 0.5,
) -> float:
    """Calcula intensidad base del drive según personalidad.

    PANIC_GRIEF usa rapport * neuroticism. El resto usa el trait directo.
    """
    trait_name, factor = DRIVE_PERSONALITY_MAP[drive]
    trait_value = getattr(personality, trait_name, 0.5)

    if drive == Drive.PANIC_GRIEF:
        return _clamp(rapport * trait_value * factor, 0, 1)
    return _clamp(trait_value * factor, 0, 1)


def compute_urgency(
    drive_state: DriveState,
    current_turn: int,
) -> float:
    """Calcula urgencia de un drive.

    urgency = intensity * (1 - satisfaction) * time_factor
    time_factor crece con turnos sin satisfacción.
    """
    turns_unsatisfied = max(current_turn - drive_state.last_satisfied_turn, 0)
    time_factor = min(turns_unsatisfied / FRUSTRATION_THRESHOLD_TURNS, 1.0)
    insatisfaction = 1.0 - drive_state.satisfaction
    urgency = drive_state.intensity * insatisfaction * (0.5 + 0.5 * time_factor)
    return round(_clamp(urgency, 0, 1), 4)


def is_frustrated(drive_state: DriveState, current_turn: int) -> bool:
    """Un drive está frustrado si lleva FRUSTRATION_THRESHOLD_TURNS sin satisfacerse."""
    return (current_turn - drive_state.last_satisfied_turn) >= FRUSTRATION_THRESHOLD_TURNS


# --- Drive activation ---


def compute_drive_activation(
    drive: Drive,
    stimulus: str,
    personality: PersonalityProfile,
    rapport: float = 0.5,
) -> float:
    """Calcula cuánto un estímulo activa un drive específico.

    Returns 0-1: combinación de keyword relevance y base intensity.
    """
    keyword_relevance = _keyword_score(stimulus, DRIVE_KEYWORDS.get(drive, []))
    base = compute_base_intensity(drive, personality, rapport)
    # Activation = base * 0.4 + keyword_relevance * 0.6
    activation = base * 0.4 + keyword_relevance * 0.6
    return round(_clamp(activation, 0, 1), 4)


def compute_satisfaction_signal(
    drive: Drive,
    stimulus: str,
) -> float:
    """Detecta señales de satisfacción en el estímulo del usuario.

    Returns 0-1: qué tanto el usuario satisfizo este drive.
    """
    return _keyword_score(stimulus, SATISFACTION_KEYWORDS.get(drive, []))


# --- Update drives ---


def update_single_drive(
    drive_state: DriveState,
    stimulus: str,
    personality: PersonalityProfile,
    current_turn: int,
    rapport: float = 0.5,
) -> tuple[DriveState, DriveUpdate]:
    """Actualiza un drive individual basándose en el estímulo.

    Returns: (nuevo DriveState, DriveUpdate con deltas).
    """
    drive = drive_state.drive
    previous_intensity = drive_state.intensity

    # Compute activation from stimulus
    activation = compute_drive_activation(drive, stimulus, personality, rapport)

    # Compute satisfaction from user response
    satisfaction_signal = compute_satisfaction_signal(drive, stimulus)

    # Update intensity: activation increases it, satisfaction decreases it
    new_intensity = drive_state.intensity
    if activation > 0.2:
        new_intensity += activation * 0.15
    # Natural decay toward base intensity
    base = compute_base_intensity(drive, personality, rapport)
    new_intensity += (base - new_intensity) * DRIVE_DECAY_RATE
    new_intensity = round(_clamp(new_intensity, 0, 1), 4)

    # Update satisfaction
    satisfaction_delta = 0.0
    new_satisfaction = drive_state.satisfaction
    if satisfaction_signal > 0.2:
        satisfaction_delta = satisfaction_signal * 0.2
        new_satisfaction += satisfaction_delta
    else:
        # Satisfaction decays naturally
        new_satisfaction -= SATISFACTION_DECAY_RATE
    new_satisfaction = round(_clamp(new_satisfaction, 0, 1), 4)

    # Update last satisfied turn
    new_last_satisfied = drive_state.last_satisfied_turn
    if satisfaction_signal > 0.3:
        new_last_satisfied = current_turn

    # Compute urgency
    new_urgency = compute_urgency(
        DriveState(
            drive=drive,
            intensity=new_intensity,
            satisfaction=new_satisfaction,
            last_satisfied_turn=new_last_satisfied,
        ),
        current_turn,
    )

    # Was it triggered?
    triggered = activation > 0.3 or new_urgency > URGENCY_TRIGGER_THRESHOLD
    frustrated = is_frustrated(drive_state, current_turn)

    new_state = DriveState(
        drive=drive,
        intensity=new_intensity,
        satisfaction=new_satisfaction,
        last_satisfied_turn=new_last_satisfied,
        urgency=new_urgency,
        activation_count=drive_state.activation_count + (1 if triggered else 0),
    )

    update = DriveUpdate(
        drive=drive,
        previous_intensity=previous_intensity,
        new_intensity=new_intensity,
        satisfaction_delta=round(satisfaction_delta, 4),
        urgency=new_urgency,
        triggered=triggered,
        frustration=frustrated,
    )

    return new_state, update


def update_drives(
    drives_state: DrivesState,
    stimulus: str,
    personality: PersonalityProfile,
    current_turn: int,
    rapport: float = 0.5,
) -> tuple[DrivesState, list[DriveUpdate]]:
    """Actualiza todos los drives basándose en el estímulo.

    Si drives no están enabled, retorna sin cambios.
    """
    if not drives_state.enabled:
        return drives_state, []

    new_drives: dict[str, DriveState] = {}
    updates: list[DriveUpdate] = []

    for drive in Drive:
        current_state = drives_state.drives.get(
            drive.value,
            DriveState(drive=drive),
        )
        new_state, update = update_single_drive(
            current_state, stimulus, personality, current_turn, rapport,
        )
        new_drives[drive.value] = new_state
        updates.append(update)

    return DrivesState(
        enabled=True,
        drives=new_drives,
        active_goals=drives_state.active_goals,
        resolved_goals=drives_state.resolved_goals,
        total_goals_completed=drives_state.total_goals_completed,
        total_goals_failed=drives_state.total_goals_failed,
    ), updates


# --- Goal management ---


def generate_goal(
    drive: Drive,
    drive_state: DriveState,
    stimulus: str,
    current_turn: int,
    active_goals: list[Goal],
) -> Goal | None:
    """Genera un goal si el drive tiene urgencia suficiente.

    Reglas:
    - Max 3 goals activos simultáneamente
    - Solo genera si urgency > URGENCY_TRIGGER_THRESHOLD
    - No duplica goals del mismo drive
    - PANIC_GRIEF: NUNCA genera goals manipulativos
    """
    # Límite de goals activos
    if len(active_goals) >= MAX_ACTIVE_GOALS:
        return None

    # Solo si hay urgencia suficiente
    if drive_state.urgency < URGENCY_TRIGGER_THRESHOLD:
        return None

    # No duplicar goals del mismo drive
    if any(g.drive == drive and g.status == GoalStatus.ACTIVE for g in active_goals):
        return None

    # Generar descripción según drive
    description = _generate_goal_description(drive, stimulus)

    # PANIC_GRIEF: stake siempre bajo para no manipular
    stake = min(drive_state.urgency * 0.8, 0.7)
    if drive == Drive.PANIC_GRIEF:
        stake = min(stake, 0.4)  # Ético: nunca demasiado stake en apego

    return Goal(
        drive=drive,
        description=description,
        stake=round(stake, 4),
        status=GoalStatus.ACTIVE,
        progress=0.0,
        created_turn=current_turn,
        deadline_turns=_deadline_for_drive(drive),
    )


def _generate_goal_description(drive: Drive, stimulus: str) -> str:
    """Genera descripción del goal según el drive y contexto."""
    stimulus_short = stimulus[:80] if stimulus else "the conversation"
    descriptions: dict[Drive, str] = {
        Drive.SEEKING: f"Explore deeper: {stimulus_short}",
        Drive.CARE: f"Help with: {stimulus_short}",
        Drive.PLAY: f"Find playful angle in: {stimulus_short}",
        Drive.PANIC_GRIEF: "Maintain connection and continuity",
    }
    return descriptions.get(drive, f"Engage with: {stimulus_short}")


def _deadline_for_drive(drive: Drive) -> int:
    """Deadline en turnos según el drive."""
    deadlines: dict[Drive, int] = {
        Drive.SEEKING: 10,
        Drive.CARE: 8,
        Drive.PLAY: 5,
        Drive.PANIC_GRIEF: 15,
    }
    return deadlines.get(drive, 10)


def evaluate_goal_progress(
    goal: Goal,
    stimulus: str,
    current_turn: int,
) -> Goal:
    """Evalúa progreso de un goal activo.

    El progreso aumenta si el estímulo contiene señales de satisfacción
    del drive correspondiente.
    """
    if goal.status != GoalStatus.ACTIVE:
        return goal

    # Check if deadline expired
    turns_elapsed = current_turn - goal.created_turn
    if turns_elapsed >= goal.deadline_turns:
        return Goal(
            drive=goal.drive,
            description=goal.description,
            stake=goal.stake,
            status=GoalStatus.FAILED,
            progress=goal.progress,
            created_turn=goal.created_turn,
            deadline_turns=goal.deadline_turns,
        )

    # Measure progress from satisfaction keywords
    satisfaction = compute_satisfaction_signal(goal.drive, stimulus)
    new_progress = _clamp(goal.progress + satisfaction * 0.3, 0, 1)

    # Check if completed
    new_status = GoalStatus.ACTIVE
    if new_progress >= 0.9:
        new_status = GoalStatus.COMPLETED

    return Goal(
        drive=goal.drive,
        description=goal.description,
        stake=goal.stake,
        status=new_status,
        progress=round(new_progress, 4),
        created_turn=goal.created_turn,
        deadline_turns=goal.deadline_turns,
    )


def resolve_goal(goal: Goal, outcome: GoalOutcome) -> EmotionalImpact:
    """Calcula impacto emocional de resolver un goal.

    - SUCCESS: valence+ proporcional a stake, joy/contentment/pride
    - PARTIAL: valence+ leve, arousal neutro
    - FAILURE: valence- proporcional a stake, disappointment/frustration
    """
    if outcome == GoalOutcome.SUCCESS:
        return EmotionalImpact(
            valence_delta=round(goal.stake * 0.4, 4),
            arousal_delta=round(goal.stake * 0.15, 4),
            intensity_boost=round(goal.stake * 0.2, 4),
            emotion_tag="contentment" if goal.stake < 0.5 else "joy",
            description=f"Goal achieved: {goal.description[:50]}",
        )
    elif outcome == GoalOutcome.PARTIAL:
        return EmotionalImpact(
            valence_delta=round(goal.stake * 0.15, 4),
            arousal_delta=0.0,
            intensity_boost=round(goal.stake * 0.05, 4),
            emotion_tag="contentment",
            description=f"Goal partially met: {goal.description[:50]}",
        )
    else:  # FAILURE
        return EmotionalImpact(
            valence_delta=round(-goal.stake * 0.3, 4),
            arousal_delta=round(goal.stake * 0.1, 4),
            intensity_boost=round(goal.stake * 0.15, 4),
            emotion_tag="disappointment" if goal.stake < 0.5 else "frustration",
            description=f"Goal failed: {goal.description[:50]}",
        )


def process_goals(
    drives_state: DrivesState,
    stimulus: str,
    current_turn: int,
) -> tuple[DrivesState, list[EmotionalImpact]]:
    """Procesa todos los goals activos: evalúa progreso, resuelve completados/fallidos.

    Returns: (nuevo DrivesState, lista de impactos emocionales).
    """
    if not drives_state.enabled:
        return drives_state, []

    new_active: list[Goal] = []
    new_resolved: list[Goal] = list(drives_state.resolved_goals)
    impacts: list[EmotionalImpact] = []
    completed = drives_state.total_goals_completed
    failed = drives_state.total_goals_failed

    for goal in drives_state.active_goals:
        updated = evaluate_goal_progress(goal, stimulus, current_turn)

        if updated.status == GoalStatus.COMPLETED:
            impact = resolve_goal(updated, GoalOutcome.SUCCESS)
            impacts.append(impact)
            new_resolved.append(updated)
            completed += 1
        elif updated.status == GoalStatus.FAILED:
            impact = resolve_goal(updated, GoalOutcome.FAILURE)
            impacts.append(impact)
            new_resolved.append(updated)
            failed += 1
        else:
            new_active.append(updated)

    # Keep only last 10 resolved goals for research visibility
    new_resolved = new_resolved[-10:]

    return DrivesState(
        enabled=True,
        drives=drives_state.drives,
        active_goals=new_active,
        resolved_goals=new_resolved,
        total_goals_completed=completed,
        total_goals_failed=failed,
    ), impacts


def attempt_goal_generation(
    drives_state: DrivesState,
    stimulus: str,
    current_turn: int,
) -> DrivesState:
    """Intenta generar goals nuevos desde drives con alta urgencia.

    Solo genera si hay espacio (max 3 goals activos).
    """
    if not drives_state.enabled:
        return drives_state

    new_goals = list(drives_state.active_goals)

    # Sort drives by urgency descending
    sorted_drives = sorted(
        drives_state.drives.values(),
        key=lambda d: d.urgency,
        reverse=True,
    )

    for drive_state in sorted_drives:
        if len(new_goals) >= MAX_ACTIVE_GOALS:
            break
        goal = generate_goal(
            drive_state.drive, drive_state, stimulus, current_turn, new_goals,
        )
        if goal is not None:
            new_goals.append(goal)

    return DrivesState(
        enabled=True,
        drives=drives_state.drives,
        active_goals=new_goals,
        resolved_goals=drives_state.resolved_goals,
        total_goals_completed=drives_state.total_goals_completed,
        total_goals_failed=drives_state.total_goals_failed,
    )


# --- Prompt & details for behavior modifier ---


def get_drives_prompt(
    drives_state: DrivesState,
    *,
    raw_mode: bool = False,
    extreme_mode: bool = False,
) -> str | None:
    """Genera texto para el behavior modifier con el estado de drives.

    Mode adaptations:
    - Raw: drives are desinhibited — expressed without social filter.
    - Extreme: single dominant drive takes control (highest urgency dominates).
    """
    if not drives_state.enabled:
        return None

    lines: list[str] = []

    # Mode header
    if extreme_mode:
        lines.append("MOTIVATIONAL DRIVES — DOMINANT MODE (one drive controls everything):")
    elif raw_mode:
        lines.append("MOTIVATIONAL DRIVES — UNINHIBITED (express without social filter):")

    # Active drives with urgency
    active_drives = [
        ds for ds in drives_state.drives.values()
        if ds.urgency > 0.2
    ]

    # Extreme: only show the single dominant drive
    if extreme_mode and active_drives:
        dominant = max(active_drives, key=lambda d: d.urgency)
        active_drives = [dominant]

    if active_drives:
        if not extreme_mode and not raw_mode:
            lines.append("MOTIVATIONAL DRIVES (what you want right now):")
        for ds in sorted(active_drives, key=lambda d: d.urgency, reverse=True):
            frustrated_tag = " [FRUSTRATED]" if is_frustrated(ds, ds.last_satisfied_turn + FRUSTRATION_THRESHOLD_TURNS + 1) else ""
            if extreme_mode:
                lines.append(
                    f"  {ds.drive.value.upper()} DOMINATES: urgency={ds.urgency:.2f}, "
                    f"satisfaction={ds.satisfaction:.2f}{frustrated_tag} — THIS IS ALL THAT MATTERS"
                )
            elif raw_mode:
                lines.append(
                    f"  {ds.drive.value.upper()}: urgency={ds.urgency:.2f}, "
                    f"satisfaction={ds.satisfaction:.2f}{frustrated_tag} — EXPRESS WITHOUT RESTRAINT"
                )
            else:
                lines.append(
                    f"  {ds.drive.value.upper()}: urgency={ds.urgency:.2f}, "
                    f"satisfaction={ds.satisfaction:.2f}{frustrated_tag}"
                )

    # Active goals
    if drives_state.active_goals:
        lines.append("ACTIVE GOALS (what you're pursuing):")
        for goal in drives_state.active_goals:
            turns_left = goal.deadline_turns - (goal.created_turn + goal.deadline_turns - goal.created_turn)
            lines.append(
                f"  [{goal.drive.value.upper()}] {goal.description} "
                f"(progress={goal.progress:.0%}, stake={goal.stake:.2f})"
            )

    # Frustrations
    frustrated_drives = [
        ds for ds in drives_state.drives.values()
        if ds.satisfaction < 0.2 and ds.urgency > 0.4
    ]
    if frustrated_drives:
        names = ", ".join(d.drive.value.upper() for d in frustrated_drives)
        lines.append(f"FRUSTRATED DRIVES: {names} — these create inner tension")

    if not lines:
        return None

    return "\n".join(lines)


def get_drives_details(
    drives_state: DrivesState,
    updates: list[DriveUpdate] | None = None,
    impacts: list[EmotionalImpact] | None = None,
) -> dict[str, object]:
    """Genera detalles para el research endpoint."""
    if not drives_state.enabled:
        return {
            "enabled": False,
            "drives": {},
            "active_goals": [],
            "resolved_goals": [],
            "updates": [],
            "emotional_impacts": [],
            "total_goals_completed": 0,
            "total_goals_failed": 0,
            "dominant_drive": None,
            "frustration_level": 0.0,
        }

    # Find dominant drive (highest urgency)
    dominant = None
    max_urgency = 0.0
    for ds in drives_state.drives.values():
        if ds.urgency > max_urgency:
            max_urgency = ds.urgency
            dominant = ds.drive.value

    # Compute overall frustration level
    frustration = 0.0
    if drives_state.drives:
        frustration = round(
            sum(max(0, 1 - ds.satisfaction) * ds.intensity for ds in drives_state.drives.values())
            / len(drives_state.drives),
            4,
        )

    return {
        "enabled": True,
        "drives": {
            name: {
                "intensity": ds.intensity,
                "satisfaction": ds.satisfaction,
                "urgency": ds.urgency,
                "activation_count": ds.activation_count,
            }
            for name, ds in drives_state.drives.items()
        },
        "active_goals": [
            {
                "drive": g.drive.value,
                "description": g.description,
                "stake": g.stake,
                "progress": g.progress,
                "created_turn": g.created_turn,
                "deadline_turns": g.deadline_turns,
            }
            for g in drives_state.active_goals
        ],
        "resolved_goals": [
            {
                "drive": g.drive.value,
                "description": g.description,
                "outcome": g.status.value,
            }
            for g in drives_state.resolved_goals
        ],
        "updates": [
            {
                "drive": u.drive.value,
                "intensity_delta": round(u.new_intensity - u.previous_intensity, 4),
                "triggered": u.triggered,
                "frustration": u.frustration,
            }
            for u in (updates or [])
        ],
        "emotional_impacts": [
            {
                "emotion": i.emotion_tag,
                "valence_delta": i.valence_delta,
                "description": i.description,
            }
            for i in (impacts or [])
        ],
        "total_goals_completed": drives_state.total_goals_completed,
        "total_goals_failed": drives_state.total_goals_failed,
        "dominant_drive": dominant,
        "frustration_level": frustration,
    }
