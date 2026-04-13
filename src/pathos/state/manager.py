"""State Manager - Gestiona el estado emocional por sesion (in-memory + save/load).

Fase 4: incluye todos los sistemas avanzados por sesión.
Soporta serialización completa a JSON para persistencia en disco.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pathos.engine.dynamics import EmotionDynamics
from pathos.engine.interoception import InteroceptiveState
from pathos.engine.steering import SteeringMomentum
from pathos.models.coupling import CouplingMatrix, coupling_from_personality
from pathos.engine.narrative import NarrativeTracker
from pathos.engine.emotional_schemas import SchemaStore, EmotionalSchema
from pathos.engine.memory import EmotionalMemoryStore
from pathos.engine.regulation import EmotionalRegulator
from pathos.engine.temporal import TemporalProcessor, RuminationEntry, SavoringEntry
from pathos.models.calibration import CalibrationProfile, CalibrationResult
from pathos.models.contagion import ShadowState, default_shadow_state
from pathos.models.emotion import EmotionalState, PrimaryEmotion, neutral_state
from pathos.models.forecasting import ForecastState, default_forecast_state
from pathos.models.external_signals import ExternalSignalsConfig, default_signals_config
from pathos.models.voice import VoiceConfig, default_voice_config
from pathos.models.immune import ImmuneState, default_immune_state
from pathos.models.memory import EmotionalMemory
from pathos.models.narrative import NarrativeSelf, default_narrative_self
from pathos.models.somatic import SomaticMarkerStore, default_somatic_store
from pathos.models.needs import ComputationalNeeds, default_needs
from pathos.models.personality import PersonalityProfile, default_personality
from pathos.models.social import UserModel, default_user_model
from pathos.models.values import ValueSystem, default_value_system
from pathos.state.crypto import encrypt_cloud_providers, decrypt_cloud_providers

logger = logging.getLogger(__name__)

SAVES_DIR = Path(__file__).parent.parent.parent.parent / "saves"


class SessionState:
    """Estado de una sesion individual con todos los sistemas avanzados."""

    def __init__(self) -> None:
        # Core
        self.emotional_state: EmotionalState = neutral_state()
        self.value_system: ValueSystem = default_value_system()
        self.conversation: list[dict[str, str]] = []
        self.memory: EmotionalMemoryStore = EmotionalMemoryStore()
        self.turn_count: int = 0
        self.state_history: list[EmotionalState] = []

        # Calibration
        self.calibration_results: list[CalibrationResult] = []
        self.calibration_profile: CalibrationProfile = CalibrationProfile()

        # Advanced systems (Fase 4)
        self.personality: PersonalityProfile = default_personality()
        self.needs: ComputationalNeeds = default_needs()
        self.user_model: UserModel = default_user_model()
        self.regulator: EmotionalRegulator = EmotionalRegulator()
        self.schemas: SchemaStore = SchemaStore()
        self.temporal: TemporalProcessor = TemporalProcessor()
        self.shadow_state: ShadowState = default_shadow_state()
        self.somatic_markers: SomaticMarkerStore = default_somatic_store()
        self.immune: ImmuneState = default_immune_state()
        self.narrative: NarrativeSelf = default_narrative_self()
        self.forecast: ForecastState = default_forecast_state()
        self.voice_config: VoiceConfig = default_voice_config()
        self.signals_config: ExternalSignalsConfig = default_signals_config()
        self.last_audio: bytes | None = None  # Last TTS audio (WAV bytes)
        self.audio_history: dict[int, bytes] = {}  # turn_number -> WAV bytes
        self.lite_mode: bool = False  # Lite mode: keyword appraisal, no embeddings, 1 LLM call
        self.advanced_mode: bool = True  # Advanced mode: activa/desactiva sistemas emocionales avanzados
        self.raw_mode: bool = False  # Raw mode: unfiltered emotional expression (censorless)
        self.extreme_mode: bool = False  # Extreme: bypass regulation/reappraisal/immune + amplify
        self.self_appraisal_enabled: bool = True  # Self-appraisal: evaluate own response against values
        self.world_model_enabled: bool = True  # World model: predict emotional impact before sending
        self.interoceptive_state: InteroceptiveState = InteroceptiveState()  # Body-state duration tracking
        self.direct_mode: bool = True  # Direct LLM modification (steering/prefix/attention) vs prompt injection only
        self.steering_enabled: bool = True  # Steering vectors: modify LLM hidden states (local models only)
        self.steering_momentum_enabled: bool = True  # Steering momentum: temporal inertia across turns
        self.steering_momentum: SteeringMomentum = SteeringMomentum()  # Momentum state per session
        self.emotional_prefix_enabled: bool = True  # Emotional prefix: inject synthetic emotional tokens at embedding layer (local models only)
        self.conditioning_tokens_enabled: bool = False  # Trained conditioning tokens: requires QLoRA adapter (5.3b, heavy GPU)
        self.emotional_adapter_enabled: bool = False  # QLoRA emotional adapter: requires trained adapter (5.2b, heavy GPU)
        self.emotional_sampler_enabled: bool = True  # Emotional sampler: modify sampling params from state
        self.emotional_attention_enabled: bool = True  # Attention modulation: bias attention weights by emotion (local models only)
        # Cloud providers: {id: {preset, label, api_key, base_url, model, models[]}}
        self.cloud_providers: dict[str, dict[str, Any]] = {}
        self.narrative_tracker: "NarrativeTracker" = self._create_narrative_tracker()

        # Dynamics (configured from personality)
        self.coupling: CouplingMatrix = self._create_coupling()
        self.dynamics: EmotionDynamics = self._create_dynamics()

    @staticmethod
    def _create_narrative_tracker() -> NarrativeTracker:
        return NarrativeTracker()

    def _create_coupling(self) -> CouplingMatrix:
        """Crea CouplingMatrix desde la personalidad actual."""
        p = self.personality
        return coupling_from_personality(
            openness=p.openness,
            conscientiousness=p.conscientiousness,
            extraversion=p.extraversion,
            agreeableness=p.agreeableness,
            neuroticism=p.neuroticism,
            emotional_reactivity=p.emotional_reactivity,
        )

    def _create_dynamics(self) -> EmotionDynamics:
        """Crea EmotionDynamics configurada desde la personalidad."""
        p = self.personality
        return EmotionDynamics(
            attractor_strength=0.15 * (0.5 + p.emotional_recovery),
            variability=p.variability,
            base_inertia=p.inertia_base,
        )

    def update_personality(self, personality: PersonalityProfile) -> None:
        """Actualiza la personalidad y reconfigura los sistemas dependientes."""
        self.personality = personality
        self.coupling = self._create_coupling()
        self.dynamics = self._create_dynamics()
        self.regulator.regulation_capacity = personality.regulation_capacity_base

    def to_dict(self, model_name: str = "") -> dict[str, Any]:
        """Serializa todo el estado a un dict para JSON save."""
        return {
            "_version": 1,
            "_saved_at": datetime.now(timezone.utc).isoformat(),
            "_model": model_name,
            # Core
            "emotional_state": self.emotional_state.model_dump(),
            "value_system": self.value_system.model_dump(),
            "conversation": self.conversation,
            "turn_count": self.turn_count,
            "state_history": [s.model_dump() for s in self.state_history[-20:]],
            # Calibration
            "calibration_profile": self.calibration_profile.model_dump(),
            "calibration_results": [r.model_dump() for r in self.calibration_results],
            # Advanced
            "personality": self.personality.model_dump(),
            "needs": self.needs.model_dump(),
            "user_model": self.user_model.model_dump(),
            "regulator": self.regulator.model_dump(),
            "shadow_state": self.shadow_state.model_dump(),
            "immune": self.immune.model_dump(),
            "narrative": self.narrative.model_dump(),
            "forecast": self.forecast.model_dump(),
            "somatic_markers": self.somatic_markers.model_dump(),
            # Non-Pydantic systems
            "memory": [m.model_dump() for m in self.memory.memories],
            "schemas": {
                "schemas": [s.model_dump() for s in self.schemas.schemas],
                "pattern_counts": {f"{k[0]}|{k[1]}": v for k, v in self.schemas._pattern_counts.items()},
                "pattern_intensities": {f"{k[0]}|{k[1]}": v for k, v in self.schemas._pattern_intensities.items()},
            },
            "temporal": {
                "ruminations": [r.model_dump() for r in self.temporal._ruminations],
                "savorings": [s.model_dump() for s in self.temporal._savorings],
                "topic_history": self.temporal._topic_history,
                "topic_emotions": {k: [e.value for e in v] for k, v in self.temporal._topic_emotions.items()},
            },
            "narrative_tracker": {
                "pattern_counts": {f"{k[0]}|{k[1]}": v for k, v in self.narrative_tracker._pattern_counts.items()},
                "pattern_intensities": {f"{k[0]}|{k[1]}": v for k, v in self.narrative_tracker._pattern_intensities.items()},
            },
            # Coupling
            "coupling": self.coupling.model_dump(),
            # Settings
            "lite_mode": self.lite_mode,
            "advanced_mode": self.advanced_mode,
            "cloud_providers": encrypt_cloud_providers(self.cloud_providers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Reconstruye un SessionState completo desde un dict."""
        session = cls()

        # Core
        session.emotional_state = EmotionalState(**data["emotional_state"])
        session.value_system = ValueSystem(**data["value_system"])
        session.conversation = data.get("conversation", [])
        session.turn_count = data.get("turn_count", 0)
        session.state_history = [EmotionalState(**s) for s in data.get("state_history", [])]

        # Calibration
        session.calibration_profile = CalibrationProfile(**data.get("calibration_profile", {}))
        session.calibration_results = [CalibrationResult(**r) for r in data.get("calibration_results", [])]

        # Advanced Pydantic models
        if "personality" in data:
            session.personality = PersonalityProfile(**data["personality"])
            session.coupling = session._create_coupling()
            session.dynamics = session._create_dynamics()
        if "coupling" in data:
            session.coupling = CouplingMatrix(**data["coupling"])
        if "needs" in data:
            session.needs = ComputationalNeeds(**data["needs"])
        if "user_model" in data:
            session.user_model = UserModel(**data["user_model"])
        if "regulator" in data:
            session.regulator = EmotionalRegulator(**data["regulator"])
        if "shadow_state" in data:
            session.shadow_state = ShadowState(**data["shadow_state"])
        if "immune" in data:
            session.immune = ImmuneState(**data["immune"])
        if "narrative" in data:
            session.narrative = NarrativeSelf(**data["narrative"])
        if "forecast" in data:
            session.forecast = ForecastState(**data["forecast"])
        if "somatic_markers" in data:
            session.somatic_markers = SomaticMarkerStore(**data["somatic_markers"])

        # Non-Pydantic: memory
        if "memory" in data:
            session.memory = EmotionalMemoryStore()
            session.memory._memories = [EmotionalMemory(**m) for m in data["memory"]]

        # Non-Pydantic: schemas
        if "schemas" in data:
            sd = data["schemas"]
            session.schemas = SchemaStore()
            session.schemas._schemas = [EmotionalSchema(**s) for s in sd.get("schemas", [])]
            session.schemas._pattern_counts = {
                tuple(k.split("|", 1)): v for k, v in sd.get("pattern_counts", {}).items()
            }
            session.schemas._pattern_intensities = {
                tuple(k.split("|", 1)): v for k, v in sd.get("pattern_intensities", {}).items()
            }

        # Non-Pydantic: temporal
        if "temporal" in data:
            td = data["temporal"]
            session.temporal = TemporalProcessor()
            session.temporal._ruminations = [RuminationEntry(**r) for r in td.get("ruminations", [])]
            session.temporal._savorings = [SavoringEntry(**s) for s in td.get("savorings", [])]
            session.temporal._topic_history = td.get("topic_history", [])
            session.temporal._topic_emotions = {
                k: [PrimaryEmotion(e) for e in v] for k, v in td.get("topic_emotions", {}).items()
            }

        # Non-Pydantic: narrative_tracker
        if "narrative_tracker" in data:
            nt = data["narrative_tracker"]
            session.narrative_tracker = NarrativeTracker()
            session.narrative_tracker._pattern_counts = {
                tuple(k.split("|", 1)): v for k, v in nt.get("pattern_counts", {}).items()
            }
            session.narrative_tracker._pattern_intensities = {
                tuple(k.split("|", 1)): v for k, v in nt.get("pattern_intensities", {}).items()
            }

        # Settings
        session.lite_mode = data.get("lite_mode", False)
        session.advanced_mode = data.get("advanced_mode", True)
        session.cloud_providers = decrypt_cloud_providers(data.get("cloud_providers", {}))

        return session


class StateManager:
    """Gestor de estado in-memory con save/load a disco."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    _MAX_SESSIONS = 50

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._MAX_SESSIONS:
                    # Evict oldest session (first inserted key)
                    oldest = next(iter(self._sessions))
                    del self._sessions[oldest]
                    logger.info("Session evicted (limit %d): %s", self._MAX_SESSIONS, oldest)
                self._sessions[session_id] = SessionState()
            return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = SessionState()

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def save_session(self, session_id: str, model_name: str = "") -> str:
        """Guarda una sesion a disco en saves/.

        Returns:
            Filename del save.
        """
        SAVES_DIR.mkdir(parents=True, exist_ok=True)
        session = self.get_session(session_id)
        data = session.to_dict(model_name=model_name)
        data["_session_id"] = session_id

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{session_id}_{ts}.json"
        filepath = SAVES_DIR / filename

        filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Session saved: %s (%d turns)", filepath, session.turn_count)
        return filename

    def load_session(self, filename: str) -> tuple[str, dict[str, Any]]:
        """Carga una sesion desde un archivo en saves/.

        Returns:
            (session_id, metadata dict with _model, _saved_at)

        Raises:
            FileNotFoundError, json.JSONDecodeError
        """
        filepath = (SAVES_DIR / filename).resolve()
        if not filepath.is_relative_to(SAVES_DIR.resolve()):
            raise ValueError(f"Invalid filename: path traversal detected")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        session_id = data.get("_session_id", "restored")
        session = SessionState.from_dict(data)
        self._sessions[session_id] = session
        logger.info("Session loaded: %s (%d turns)", filename, session.turn_count)
        return session_id, {
            "model": data.get("_model", ""),
            "saved_at": data.get("_saved_at", ""),
            "turn_count": session.turn_count,
        }

    def list_saves(self) -> list[dict[str, Any]]:
        """Lista saves disponibles en saves/."""
        if not SAVES_DIR.exists():
            return []
        saves = []
        for f in sorted(SAVES_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                saves.append({
                    "filename": f.name,
                    "session_id": data.get("_session_id", ""),
                    "model": data.get("_model", ""),
                    "saved_at": data.get("_saved_at", ""),
                    "turn_count": data.get("turn_count", 0),
                })
            except Exception:
                continue
        return saves

    def auto_load_latest(self, available_models: list[str] | None = None) -> tuple[str | None, str]:
        """Intenta cargar el save mas reciente al arrancar.

        Carga el save mas reciente sin importar el modelo — el usuario
        puede cambiar el modelo en runtime via el Model Selector.
        Solo salta un save si esta corrupto o no se puede parsear.

        Args:
            available_models: lista de modelos disponibles (solo para logging).

        Returns:
            (session_id o None, mensaje de status)
        """
        saves = self.list_saves()
        if not saves:
            return None, "No saves found, starting fresh"

        # Intentar cargar saves en orden (mas reciente primero)
        for save in saves:
            model = save.get("model", "")
            try:
                sid, meta = self.load_session(save["filename"])
                model_note = ""
                if model and available_models is not None and model not in available_models:
                    model_note = f" (model '{model}' not in Ollama — switch model in UI)"
                return sid, f"Restored session '{sid}' ({meta['turn_count']} turns, model={model}){model_note}"
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping save '%s': %s", save["filename"], e,
                )
                continue

        return None, "All saves failed to load, starting fresh"
