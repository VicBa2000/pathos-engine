# PATHOS ENGINE v2 - Development Rules

## Project Description

Pathos Engine is a functional emotional architecture system for LLMs. Based on emotional functionalism: emotions are defined by their FUNCTION, not their substrate. The system detects stimuli, evaluates them against values, generates persistent internal states that modify behavior, and regulates them over time.

### Theoretical Pillars
1. **Appraisal Theory** (Lazarus/Scherer) - Emotions = stimulus x evaluation
2. **Dimensional Model** (Russell) - Emotional state as continuous vector (valence, arousal, dominance, certainty)
3. **Emotional Homeostasis** - Decay toward baseline, with baseline shift from extreme experiences
4. **Adapted Embodied Cognition** - Computational body (energy, tension, openness, warmth)
5. **DynAffect** (Kuppens) - Emotional dynamics with differential equations
6. **Big Five** (Costa/McCrae) - Configurable personality that modulates the entire system
7. **Young/Beck Schemas** - Learned emotional patterns that create emergent personality
8. **Interoception** (Craig) - Self-initiated emotional reflection from internal state thresholds

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + D3.js |
| Backend | Python 3.13 + FastAPI |
| LLM Local | Ollama (qwen3:4b default) |
| LLM Cloud | Anthropic Claude API |
| Embeddings | nomic-embed-text via Ollama |
| State | In-memory per session |
| Tests | pytest + pytest-asyncio (686 tests) |

### Implemented Systems (23 emotional systems + voice + UI)

**Core:**
- Value System, Appraisal Module, Emotion Generator, Emotional State,
  Homeostasis, Emotional Memory, Behavior Modifier, Mood System,
  Calibration, Authenticity Metrics

**Advanced:**
- Emotional Stack (19 simultaneous emotions), Computational Needs,
  Active Regulation (ego depletion + breakthroughs), Social Cognition,
  Emotion Dynamics (ODE), Reappraisal, Emotional Schemas,
  Temporal Dynamics, Meta-Emotions, Self-Initiated Inquiry,
  Personality Profile (Big Five + temperament)

**Extended:**
- Emotion Contagion, Somatic Markers, Emotional Creativity,
  Emotional Immune System, Narrative Self, Emotional Forecasting

**Voice:**
- TTS (Kokoro + Parler-TTS), ASR (Whisper)

**Modes (8):**
- Companion, Research, Calibration, Sandbox, Arena, Mirror,
  Auto-Research (autonomous), Raw/Extreme

**Frontend (27 components):**
- Chat, Research Panel, Calibration, Sandbox, Arena, Mirror Test,
  Raw Mode, Auto-Research, Emotion Network (D3), Circumplex Chart,
  Body State, Journey Timeline, Pipeline Viewer, Emotion Avatar,
  Emotion Genesis, Voice Orb, Model Manager, Agent Setup,
  Mode Selector, Mic Config, Voice Config, Confirm Modal,
  Error Boundary, Export Button

---

## Development Rules

### 1. Step-by-Step Development
- Go step by step. Before starting any task, understand the current state.
- After a `clear`, read `manual.txt` and recent git history to understand current project state.

### 2. Sources of Truth
- `manual.txt` — complete documentation of the current system
- `instructions.txt` — user-facing usage instructions
- `README.md` — public project description
- If there is ambiguity, consult these files before making decisions.

### 3. Code Quality
- Strict TypeScript in frontend (no `any`)
- Type hints on all Python code
- Interfaces and types defined BEFORE implementing logic
- Emotional calculations must be deterministic and testable
- Numeric values always clamped to defined ranges (e.g., valence -1 to 1, arousal 0 to 1)
- New modules: create in `src/pathos/engine/` (logic) or `src/pathos/models/` (data)
- Pydantic for data models, dataclass only for lightweight internal results

### 4. Architecture
- Clear separation between modules — each system in its own file
- Each component must be testable in isolation
- Emotional state is immutable: each operation generates a new state
- Pipeline of 22+ steps in main.py — new systems are inserted as additional steps
- Advanced systems are OPTIONAL and modular:
  - Voice is optional (text-only works perfectly without it)
  - SER is optional (user decides whether to activate vocal detection)
  - Schemas form automatically, no configuration required
- New systems must expose their internals in the research endpoint
- `SessionState` in state/manager.py centralizes all per-session state

### 5. Ethical Considerations (non-negotiable)
- The system NEVER claims to "feel" in a conscious sense
- Full transparency: user can see emotional state internals
- No emotional manipulation of the user
- User distress has PRIORITY over the agent's emotional authenticity
- SER (vocal emotion detection) is always OPT-IN, never forced
- User can disable any advanced system

### 6. Testing
- Unit tests for every emotional calculation function
- Integration tests for the complete pipeline
- Emotional coherence tests (emotions consistent with stimuli)
- Range tests (no value out of bounds)
- Each new module MUST include its test file
- Keep 100% of tests passing before considering a feature complete
- Naming: `tests/test_{module}.py` with classes `Test{Feature}`

### 7. Frontend
- New visualizations: D3.js on SVG (established pattern)
- Components in `frontend/src/components/` with dedicated CSS
- TypeScript types in `frontend/src/types/emotion.ts` must match backend schemas
- New research endpoint data -> new section in ResearchPanel
- Visual features with toggle (like EmotionNetwork) to avoid screen clutter
- Verify build: `npx tsc --noEmit && npx vite build` with zero errors

### 8. Voice
- Voice modules in `src/pathos/voice/` (separate directory)
- 3 modes: text-only | voice-out | full-voice — user chooses
- TTS and ASR are independent services from the emotional pipeline
- Voice Parameter Generator translates emotional_state to vocal parameters
- Hardware target: GTX 1660 Super 6GB — everything must fit
- Minimum config: qwen3:4b (2.5GB) + Kokoro (~1GB) + Whisper-small (0.5GB)

### 9. Documentation
- Update `manual.txt` when significant features are added
- Update `instructions.txt` when usage flow changes
- Update `README.md` when major features are added
- Do NOT create unnecessary additional markdown files
