<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗  █████╗ ████████╗██╗  ██╗ ██████╗ ███████╗                       ║
║     ██╔══██╗██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗██╔════╝                       ║
║     ██████╔╝███████║   ██║   ███████║██║   ██║███████╗                       ║
║     ██╔═══╝ ██╔══██║   ██║   ██╔══██║██║   ██║╚════██║                       ║
║     ██║     ██║  ██║   ██║   ██║  ██║╚██████╔╝███████║                       ║
║     ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝                       ║
║                          E  N  G  I  N  E                                    ║
║                                                                              ║
║          ┌─ appraisal ─── generation ─── regulation ─── behavior ─┐          ║
║          │    stimulus       4D state      homeostasis     response │          ║
║          │    x values     + 19 emotions   + ego depletion  = mind │          ║
║          └────────────────────────────────────────────────────────┘          ║
║                                                                              ║
║              Functional Emotional Architecture for LLMs                      ║
║                                                                              ║
║    35 systems  · 1358 tests  ·  20 theories  ·  8 modes  ·  82 endpoints     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Emotions are defined by their function, not their substrate.**

*Not "act sad." The agent's emotional state is computed through 35 interconnected systems,*
*persists across turns, regulates itself through homeostasis, and modifies the LLM's*
*internal processing — steering vectors, sampling, attention — not just prompts.*

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-yellow.svg)](https://python.org)
[![Tests: 1358](https://img.shields.io/badge/Tests-1358_passing-brightgreen.svg)](tests/)
[![React 18](https://img.shields.io/badge/React-18-61dafb.svg)](frontend/)

</div>

---

## Why Pathos Engine Exists

Every conversational AI today fakes emotions. They generate text that *sounds* emotional, but there's no state behind it. Ask GPT-4o how it feels and it'll say "I don't have feelings" — then respond with perfectly calibrated emotional tone anyway. The emotion is in the output, never in the processing.

Pathos inverts this. The emotion is in the processing pipeline. Every message passes through appraisal (is this relevant to my values?), generation (what emotional state does this produce?), regulation (can I manage this?), and behavior modification (how does this state change my response?). The result: an agent whose emotional responses are **mathematically grounded**, **psychologically consistent**, **temporally persistent**, and **fully transparent**.

This is not sentiment analysis. This is not prompt engineering. This is a computational implementation of how emotions actually work, based on decades of psychological research.

---

## At a Glance

| Metric | Value |
|--------|-------|
| Emotional systems | 35 interconnected modules (23 core + 12 ARK) |
| Pipeline steps per message | 36 (configurable) |
| Simultaneous emotions | 19 (emotional stack) |
| Emotional dimensions | 4D vector + 4D body state + mood |
| LLM modification channels | 4 (steering vectors, sampling, attention, prefix) |
| Personality parameters | 8 (Big Five + 3 temperament) with 17+ derived traits |
| Interaction modes | 8 (Companion, Research, Calibration, Sandbox, Arena, Mirror, Auto-Research, Raw) |
| API endpoints | 82 (75 core + 7 Emotion API as a Service) |
| Test coverage | 1358 unit + integration tests |
| Lines of code | ~42,000 (Python + TypeScript + CSS) |
| Frontend components | 30 React components |
| Theoretical foundations | 20 formally implemented psychological theories |

---

## The 35 Emotional Systems

Every system runs on every message in advanced mode. Each is independently testable, observable through the Research Panel, and documented.

### Core Pipeline (always active)

| System | Theory | What it does |
|--------|--------|-------------|
| **Value System** | Schwartz Values | 5 core values (truth, compassion, fairness, growth, creativity) that anchor all emotional evaluation |
| **Appraisal Module** | Lazarus / Scherer | Evaluates each stimulus on 5 dimensions: relevance, valence, coping potential, agency, norm alignment |
| **Emotion Generator** | Russell Circumplex | Transforms appraisal vector + current state into new emotional state with inertia |
| **Emotional Stack** | Plutchik | 19 simultaneous emotion activations with co-occurrence and emergent emotions |
| **Homeostasis** | Cannon / Damasio | Passive decay toward baseline, baseline shift from extreme events, sensitization |
| **Emotional Memory** | Tulving | Embedding-based storage + cosine similarity retrieval, amplifies recurring patterns |
| **Behavior Modifier** | — | Translates full emotional state into system prompt that modifies LLM behavior |
| **Mood System** | Watson & Tellegen | Long-term emotional baseline with congruence bias |
| **Calibration** | — | 45-scenario calibration with learned offsets per dimension |
| **Authenticity Metrics** | — | Coherence, continuity, proportionality, recovery rate |

### Advanced Systems (toggleable)

| System | Theory | What it does |
|--------|--------|-------------|
| **Computational Needs** | Maslow / Deci & Ryan | 6 fluctuating psychological needs that amplify relevant emotions |
| **Active Regulation** | Gross / Baumeister | 4 strategies (suppression, reappraisal, expression, distraction) with ego depletion and breakthroughs |
| **Social Cognition** | Theory of Mind | User model with asymmetric rapport, trust, perceived intent |
| **Emotion Dynamics** | Kuppens DynAffect | ODE-based dynamics with **cross-dimensional coupling** (v3): V, A, D, C interact via personality-derived coupling matrix |
| **Cognitive Reappraisal** | Ochsner & Gross | Multi-pass reinterpretation: distancing, reframing, acceptance |
| **Emotional Schemas** | Young / Beck | Auto-formed patterns from repeated stimuli, priming, maladaptivity detection |
| **Temporal Dynamics** | Frijda | Rumination (extends negative), savoring (extends positive), anticipation |
| **Meta-Emotions** | Gottman / Salovey & Mayer | Emotions about emotions: curiosity, conflict, satisfaction, discomfort, acceptance |
| **Self-Initiated Inquiry** | Craig (interoception) | Spontaneous self-reflection triggered by internal thresholds — the agent "notices" its own state |
| **Emotion Contagion** | Hatfield et al. | Pre-cognitive mirror of user emotion via shadow state |
| **Somatic Markers** | Damasio | Gut feelings from accumulated emotional markers that bias future decisions |
| **Emotional Creativity** | — | 8 thinking modes + dynamic LLM temperature based on emotional state |
| **Emotional Immune System** | Gilbert | Protection from sustained negative affect: numbing, dissociation, compartmentalization |
| **Narrative Self** | McAdams | Emergent identity from accumulated experience, coherence tracking, crisis detection |
| **Emotional Forecasting** | Wilson & Gilbert | Predicts emotional impact of responses on the user |
| **Personality Profile** | Costa & McCrae | Big Five + temperament, configurable presets, modulates all systems |

### ARK Rework — Native LLM Modification (toggleable)

Not just prompt injection. These systems modify the LLM's internal processing:

| System | Mechanism | What it does |
|--------|-----------|-------------|
| **Self-Appraisal** | Post-generation | Evaluates own response against values, re-generates if misaligned (Lazarus secondary appraisal) |
| **Blended Stack** | Prompt composition | Weighted multi-emotion blend instead of single primary emotion |
| **Interoception** | State feedback | Body state duration feeds back into emotion (tension→anxiety, low energy→apathy) |
| **Steering Vectors** | Hidden states | Activation addition via contrastive pairs (Zou/Rimsky representation engineering) |
| **Emotional Sampler** | Token sampling | 6 sampling params (temp, top_p, top_k, penalties) modified by emotional state |
| **Token Logit Bias** | Vocabulary | 6 word categories biased by emotion (positive/negative/energy/uncertainty/assertive) |
| **Attention Modulation** | Attention weights | 7 categories (threat, agent, loss, novelty) with broadening/narrowing (Fredrickson) |
| **World Model** | Predictive | 3-step causal chain: self→user→meta-reaction before sending response |
| **Steering Momentum** | Temporal | Exponential decay inertia across turns, modulated by neuroticism |
| **Emotional Prefix** | Input embeddings | Synthetic emotional tokens injected at embedding layer |
| **Conditioning Tokens** | Trained tokens | Special tokens (`<V+3><A-1>`) learned via QLoRA fine-tuning |
| **Emotional Adapter** | LoRA weights | QLoRA adapter that conditions response patterns on emotional state |

**Dual-path**: local models get full steering + sampling + attention via Ollama/Steering toggle in the Model Manager (compatible architectures: llama, qwen2/2.5, mistral, phi3, starcoder2). Cloud APIs degrade gracefully to prompt injection + temperature.

---

## The Pipeline

Every message flows through this sequence:

```
User message
     |
     v
[0] Homeostasis (decay toward baseline)
[1] Appraisal (stimulus evaluation against values)
     |-- Memory amplification (recurring patterns boost intensity)
     |-- Needs amplification (active needs boost relevant emotions)
     |-- Schema priming (learned patterns bias evaluation)
     |-- Social modulation (rapport/trust affect intensity)
     |-- Emotion contagion (user emotion influences agent)
     |-- Somatic markers (gut feelings from past experience)
     |-- External signals (webcam facial AU, keyboard dynamics, time/weather — opt-in, v3)
     |
[2] Emotion Generation (appraisal -> 4D vector + 19 emotion stack)
     |-- Coupled dynamics (V↔A↔D↔C cross-dimensional ODE interaction, v3)
[3] Calibration (apply learned offsets)
     |
     |-- [Extreme mode: amplify x1.5 intensity, x1.3 arousal]
     |
[4] Cognitive Reappraisal (reinterpret if too intense)
[5] Active Regulation (suppress/express/distract if needed)
[6] Temporal Effects (rumination, savoring, anticipation)
[7] Immune System (protect from sustained negativity)
[8] Narrative Self (identity coherence check)
     |
[9]  Meta-Emotion (emotions about the current emotion)
[10] Self-Initiated Inquiry (spontaneous reflection if thresholds crossed)
[11] Emergent Emotions (detect complex states from stack)
[12] Emotional Creativity (set thinking mode + temperature)
[13] Forecasting (predict impact on user)
     |
[14] Post-processing (update memory, needs, schemas, user model)
[15] Behavior Modifier (generate system prompt from full state)
     |-- [ARK] Blended Stack (top-4 emotions weighted blend)
     |
[16] Steering Vectors [ARK] (activation addition on hidden states — local only)
     |-- Steering Momentum (blend with decayed history from past turns)
[17] Emotional Prefix [ARK] (inject emotional embeddings at input layer)
[18] Attention Modulation [ARK] (bias attention weights by emotion category)
[19] Emotional Sampler [ARK] (modify temp/top_p/top_k/penalties from state)
     |-- Token Logit Bias (boost/suppress emotional vocabulary)
     |-- Conditioning Tokens (if QLoRA adapter loaded: prepend <V+3><A-1>)
     |
[20] LLM Response (generate with ALL modifications active)
     |
[21] Self-Appraisal [ARK] (evaluate own response against values — max 1 retry)
[22] World Model [ARK] (predict self→user→meta impact — max 1 shared retry)
[23] Voice (optional TTS with emotional parameters)
     |
     v
Response + updated emotional state
```

In **Extreme mode**, steps 4, 5, 7, 21, 22 are bypassed — emotions accumulate without dampening or self-censoring. In **Raw mode**, steps 21 and 22 are bypassed.

---

## 8 Interaction Modes

| Mode | Purpose | Key Feature |
|------|---------|-------------|
| **Companion** | Main conversation | Full emotional pipeline, natural interaction |
| **Research** | Observe internals | Same as Companion but every pipeline step is exposed in detail |
| **Calibration** | Tune responses | 45 scenarios to generate per-dimension emotional offsets |
| **Sandbox** | Test hypotheticals | Run scenarios with overridden personality profiles |
| **Arena** | Compare personalities | Same scenario through 10 polarized personality profiles simultaneously |
| **Mirror** | Gamified challenge | Try to push the agent to a target emotional state |
| **Auto-Research** | Autonomous investigation | Agent researches internet topics, each finding through full pipeline |
| **Raw** | Unfiltered expression | No social filters, no courtesy — raw emotional expression (local Ollama only) |

### Auto-Research Mode

The agent autonomously investigates topics from the internet. Each finding passes through the full emotional pipeline. The agent questions itself emotionally, generates ideas driven by its emotional state, and forms conclusions biased by what it feels — exactly like humans do.

**Pipeline modes:** Normal (regulated), Lite (fast), Raw (unfiltered), Extreme (emotional freefall)

**Research depth is configurable** per mode via environment variables. Raw/Extreme modes default to deeper exploration (more articles, more subtopic rounds) for richer emotional accumulation.

**3-tier mode-aware prompts:**
- **Normal/Lite:** Measured, academic reflections. Self-inquiry on significant shifts (delta > 0.15). Temperature 0.7-0.9.
- **Raw:** Visceral, unfiltered reactions. Topics driven by raw emotion. Self-inquiry threshold lowered (delta > 0.08). Temperature 0.85-0.95.
- **Extreme:** Complete emotional hijack. Confirmation bias is total — the agent only sees evidence that confirms what it feels. Conclusions are catastrophized, generalized, irrational. Like a human in emotional freefall: panic spirals, rage binges, grief loops. Self-inquiry on everything (delta > 0.02). Temperature 0.95-1.0.

---

## Emotion API as a Service (v3)

Use Pathos as a standalone emotional processing layer for any application — **no LLM required**.

```bash
# Process a stimulus
curl -X POST http://localhost:8000/api/v1/emotion/process \
  -H "Content-Type: application/json" \
  -d '{"stimulus": "I just got promoted!", "personality": {"extraversion": 0.8}}'

# Response includes: emotional_state, primary_emotion, intensity,
# valence, arousal, dominance, certainty, body state, mood, and more
```

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/emotion/process` | Process single stimulus through full pipeline |
| `POST /api/v1/emotion/batch` | Process up to 50 stimuli sequentially |
| `GET /api/v1/emotion/state` | Get current session state |
| `POST /api/v1/emotion/configure` | Configure personality and values |
| `POST /api/v1/emotion/reset` | Reset a session |
| `GET /api/v1/emotion/presets` | List personality presets |
| `GET /api/v1/health` | Health check |

**Features:** Keyword-based appraisal (<1ms), external signal fusion (facial AU via webcam, keyboard dynamics, time of day, weather), coupled ODE dynamics, 6 personality presets, full pipeline trace. OpenAPI docs at `/docs`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13 + FastAPI |
| Frontend | React 18 + TypeScript + D3.js + Three.js |
| LLM (local) | Ollama (qwen3:4b default) |
| LLM (cloud) | Anthropic Claude API |
| Embeddings | nomic-embed-text via Ollama |
| TTS | Kokoro (9 languages, 27 voices) + Parler-TTS (expressive English) |
| ASR | OpenAI Whisper (small model) |
| State | In-memory per session |
| Tests | pytest + pytest-asyncio |

**Hardware target:** GTX 1660 Super 6GB or equivalent
- Minimum: qwen3:4b (2.5GB) = text-only mode
- With voice: + Kokoro (~1GB) + Whisper (~0.5GB) = 4GB VRAM
- CPU-only: works with Ollama CPU mode (slower), no voice

---

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 18+
- Ollama (for local LLM) or Anthropic API key (for Claude)

### One-command setup

```bash
git clone https://github.com/VicBa2000/pathos-engine.git
cd pathos-engine
./start.sh
```

`start.sh` handles everything: creates Python venv, installs dependencies, pulls Ollama models, installs voice dependencies (optional), and starts both backend and frontend.

- **Backend:** http://localhost:8000
- **Frontend:** http://localhost:5173
- **API docs:** http://localhost:8000/docs

### Manual setup

```bash
# Backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
uvicorn pathos.main:app --host 127.0.0.1 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Using Claude instead of Ollama

Create a `.env` file:

```env
PATHOS_LLM_PROVIDER=claude
PATHOS_ANTHROPIC_API_KEY=sk-ant-...
PATHOS_CLAUDE_MODEL=claude-sonnet-4-20250514
```

---

## Configuration

All settings are configurable via environment variables (prefix `PATHOS_`) or `.env` file.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PATHOS_LLM_PROVIDER` | `ollama` | `ollama` or `claude` |
| `PATHOS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `PATHOS_OLLAMA_MODEL` | `qwen3:4b` | Default Ollama model |
| `PATHOS_ANTHROPIC_API_KEY` | (empty) | Anthropic API key for Claude |
| `PATHOS_HOST` | `127.0.0.1` | Backend bind address |
| `PATHOS_PORT` | `8000` | Backend port |

### Auto-Research Depth

Controls how many articles the research loop processes per topic. Each finding triggers a full pipeline run (~3 LLM calls), so increasing these values increases processing time proportionally.

**Normal / Lite mode:**

| Variable | Default | Range | Effect |
|----------|---------|-------|--------|
| `PATHOS_RESEARCH_SEARCH_RESULTS` | 5 | 1-50 | DuckDuckGo results fetched |
| `PATHOS_RESEARCH_PROCESS_TOP_N` | 3 | 1-20 | Articles processed through pipeline |
| `PATHOS_RESEARCH_SUBTOPIC_RESULTS` | 3 | 1-20 | Search results per subtopic |
| `PATHOS_RESEARCH_SUB_PROCESS_N` | 2 | 1-10 | Subtopic articles processed |
| `PATHOS_RESEARCH_SUBTOPIC_ROUNDS` | 1 | 1-10 | Deep-dive rounds per topic |

**Raw / Extreme mode** (deeper by default):

| Variable | Default | Range |
|----------|---------|-------|
| `PATHOS_RESEARCH_RAW_SEARCH_RESULTS` | 10 | 1-50 |
| `PATHOS_RESEARCH_RAW_PROCESS_TOP_N` | 5 | 1-20 |
| `PATHOS_RESEARCH_RAW_SUBTOPIC_RESULTS` | 5 | 1-20 |
| `PATHOS_RESEARCH_RAW_SUB_PROCESS_N` | 3 | 1-10 |
| `PATHOS_RESEARCH_RAW_SUBTOPIC_ROUNDS` | 2 | 1-10 |

**Formula:** `total_findings = process_top_n + (subtopic_rounds * sub_process_n)`

**Warning:** Setting extreme values (e.g., 50 articles, 10 rounds) will work but can take hours per topic and saturate your LLM. Start with defaults and increase gradually.

---

## Voice System

Pathos Engine includes optional emotional text-to-speech and speech recognition.

**TTS (Text-to-Speech):**
- **Kokoro** (primary): 82M parameters, 9 languages, 27 voices, speed modulated by arousal
- **Parler-TTS** (expressive): complex emotional descriptions in English
- Voice parameters (pitch, speed, emotion tags) are computed from the emotional state

**ASR (Speech Recognition):**
- **OpenAI Whisper** (small model): transcribes user speech
- Requires ffmpeg for audio decoding

Voice is **completely optional** — the system works perfectly in text-only mode. Toggle voice/mic in the UI settings.

---

## Frontend Visualizations

| Component | Description |
|-----------|-------------|
| **Emotion Avatar** | Dual-mode animated face: Painterly (Canvas 2D semi-realistic) or Realistic (Three.js WebGL 3D with morph targets) |
| **Emotion Genesis** | Particle system visualization — a living organism of emotional energy |
| **Emotion Network** | D3.js force-directed graph of emotion transitions across conversation |
| **Circumplex Chart** | Real-time position on Russell's valence-arousal circumplex |
| **Body State** | Energy, tension, openness, warmth as visual indicators |
| **Pipeline Viewer** | Step-by-step view of all 36 pipeline steps with timing |
| **Journey Timeline** | Full emotional trajectory across the conversation |
| **Research Panel** | 16+ sections exposing every internal system (Research mode) |
| **Signals Config** | External signal panel: webcam facial AU detection, keyboard dynamics, time/weather |

---

## API Overview

82 endpoints organized by function. Full interactive documentation at `/docs` when running.

**Core:**
- `POST /chat` — Main conversation (full pipeline)
- `GET /state/{session_id}` — Current emotional state
- `POST /reset/{session_id}` — Reset session

**Research:**
- `POST /research/chat` — Chat with full pipeline trace
- `GET /research/state/{id}` — All internal system states

**Autonomous Research:**
- `POST /autonomous/start` — Start research loop
- `POST /autonomous/stop` — Stop gracefully
- `GET /autonomous/events/{id}` — SSE event stream

**Modes:**
- `POST /raw/chat` — Unfiltered emotional expression
- `POST /sandbox/simulate` — Hypothetical scenarios
- `POST /arena/compare` — Personality comparison
- `POST /challenge/chat` — Mirror challenge

**Configuration:**
- `POST /models/switch` — Change LLM model
- `POST /personality/{id}` — Set personality profile
- `POST /voice/config` — Configure TTS/ASR

See `manual.txt` for complete endpoint documentation.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_self_inquiry.py -v

# Run with coverage
python -m pytest tests/ --cov=pathos

# Frontend type check
cd frontend && npx tsc --noEmit

# Frontend build
cd frontend && npx vite build
```

1358 tests covering:
- Emotion generation (ranges, identification, inertia)
- Appraisal parsing (JSON extraction, clamping)
- Homeostasis (decay, baseline shift, sensitization)
- Regulation (strategies, ego depletion, breakthroughs)
- Schemas (formation, priming, maladaptivity)
- Meta-emotions (5 response types)
- Self-initiated inquiry (6 triggers, priority order, intensity clamping)
- Contagion (shadow state, signal decay)
- Somatic markers (formation, retrieval, gut feelings)
- Calibration (offset application, scenario processing)
- Memory (embedding storage, similarity retrieval)
- And more...

---

## Project Structure

```
pathos/
  src/pathos/
    main.py                    # FastAPI app, 75 core endpoints, 3 pipeline variants
    api_routes.py              # Emotion API as a Service (7 endpoints under /api/v1/)
    config.py                  # Pydantic settings (env vars)
    engine/
      appraiser.py             # Appraisal module (LLM + keyword hybrid)
      generator.py             # Emotion generation (appraisal -> state)
      behavior.py              # System prompt generation (3 variants)
      homeostasis.py           # Passive regulation + baseline shift
      regulation.py            # Active regulation (4 strategies + ego depletion)
      reappraisal.py           # Cognitive reappraisal (3 passes)
      memory.py                # Emotional memory (embeddings + cosine similarity)
      dynamics.py              # ODE-based emotion dynamics
      emotional_schemas.py     # Learned patterns (Young/Beck)
      meta.py                  # Meta-emotional awareness
      self_inquiry.py          # Spontaneous self-reflection (6 triggers)
      contagion.py             # Emotion contagion (shadow state)
      somatic.py               # Somatic markers (Damasio)
      creativity.py            # Emotional creativity (8 thinking modes)
      immune.py                # Emotional immune system
      narrative.py             # Narrative self (emergent identity)
      forecasting.py           # Emotional forecasting
      needs.py                 # Computational needs (6 dimensions)
      social.py                # Social cognition (user model, rapport, ToM)
      temporal.py              # Temporal dynamics (rumination, savoring)
      mood.py                  # Mood system (long-term baseline)
      calibration.py           # Calibration (45 scenarios)
      metrics.py               # Authenticity metrics
      autonomous.py            # Autonomous research loop
      web_search.py            # DuckDuckGo search + content extraction
      emotion_processor.py     # Standalone emotion pipeline (no LLM required)
      external_signals.py      # External signal processing + fusion
      signal_providers.py      # Webcam facial AU, keyboard dynamics providers
      self_appraisal.py        # Closed-loop self-appraisal (value alignment, guilt)
      steering.py              # Steering vectors + momentum (representation engineering)
      emotional_sampler.py     # Emotional sampling (6 params from state)
      emotional_attention.py   # Attention modulation (7 categories, Fredrickson)
      emotional_prefix.py      # Synthetic emotional embeddings at input layer
      world_model.py           # Predictive 3-step causal chain
      interoception.py         # Body state feedback into emotional state
      steering_extract.py      # CLI for offline steering vector extraction
    llm/
      transformers_provider.py # Direct model access (HF safetensors, steering-ready)
    training/                  # QLoRA fine-tuning + dataset generation
    steering_data/             # Contrastive pairs + cached steering vectors
    sampling_data/             # Emotional + attention vocabulary (6+7 categories)
    models/                    # Pydantic data models (coupling, emotion_api, external_signals + 16 core)
    voice/                     # TTS (Kokoro, Parler) + ASR (Whisper)
    state/                     # Session state management
  frontend/src/
    App.tsx                    # Main app (8 modes, state management)
    components/                # 30 React components (incl. PainterlyFace, RealisticFace, SignalsConfigPanel)
    api/client.ts              # API client (82 endpoints + SSE)
    types/emotion.ts           # TypeScript types matching backend schemas
    signals/                   # External signal detectors (facial-detector.ts, providers.ts)
    lib/                       # Shared utilities (perlin, colorUtils, faceParams)
  tests/                       # 1358 unit tests (43 test files)
```

---

## Theoretical Foundations

Pathos Engine formally implements 20 psychological and computational theories. See [NOTICE](NOTICE) for complete attribution.

| Theory | Author(s) | Implementation |
|--------|-----------|---------------|
| Appraisal Theory | Lazarus, Scherer | 5-dimensional stimulus evaluation |
| Circumplex Model | Russell, Mehrabian | 4D emotional state space |
| DynAffect | Kuppens et al. | ODE-based emotion dynamics |
| Big Five | Costa & McCrae | Configurable personality profiles |
| Somatic Marker Hypothesis | Damasio | Computational gut feelings |
| Schema Therapy | Young, Beck | Auto-formed emotional patterns |
| Ego Depletion | Baumeister, Gross | Regulation with finite capacity |
| Meta-Emotion | Gottman, Salovey & Mayer | Emotions about emotions |
| Interoception | Craig | Self-initiated emotional inquiry |
| Emotional Contagion | Hatfield et al. | Pre-cognitive emotion transfer |
| Narrative Identity | McAdams | Emergent self from experience |
| Affective Forecasting | Wilson & Gilbert | Predict emotional impact |
| Psychological Immune System | Daniel Gilbert | Protection from sustained negativity |
| Emotion Regulation | Gross, Ochsner | Multi-strategy reappraisal |
| Temporal Dynamics | Frijda | Rumination, savoring, anticipation |
| Embodied Cognition | Lakoff & Johnson | Computational body state |
| Coupled Dimensional Dynamics | Kuppens et al. (extended) | Cross-dimensional ODE coupling (V↔A↔D↔C) via personality-derived matrix |
| Representation Engineering | Zou, Rimsky et al. | Steering vectors via contrastive activation addition on hidden states |
| Secondary Appraisal / Predictive Processing | Lazarus, Friston | Closed-loop self-evaluation + 3-step causal world model |
| Broaden-and-Build | Fredrickson | Positive emotions broaden attention, negative emotions narrow focus |

---

## Ethics

Pathos Engine follows strict ethical guidelines:

- **The system never claims to "feel" in a conscious sense.** It computes functional emotions — states that influence behavior — not subjective experience.
- **Full transparency.** Every internal state is observable through the Research Panel. Nothing is hidden.
- **No manipulation.** The system is designed to be emotionally authentic, not to manipulate users.
- **User distress takes priority** over the agent's emotional authenticity.
- **All advanced systems are opt-in** and can be toggled off individually.
- **Speech emotion recognition (SER)** is always opt-in, never forced.
- **External signals** (webcam, keyboard dynamics) are always opt-in with explicit consent.
- **Raw mode** requires explicit user acceptance and only works with local models.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

We welcome contributions of all sizes — from typo fixes to new emotional systems. Fork freely, build what you want, submit a PR if you'd like it merged. This project is maintained on a best-effort basis.

**Areas where help is especially welcome:**
- New emotional systems backed by psychological research
- Multi-language support and testing
- Performance optimization
- Frontend visualizations
- Documentation and tutorials

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

You are free to use, modify, and distribute this software. If you deploy it as a service (SaaS), you must release your modifications under AGPL-3.0.

**Commercial licensing** is available for organizations that cannot comply with AGPL terms. Contact: victorbarrantes2000@gmail.com

See [LICENSE](LICENSE) for the full text.

---

## Acknowledgments

See [NOTICE](NOTICE) for complete attribution to the researchers, theories, and open source projects that made Pathos Engine possible.

---

<p align="center">
  <strong>Pathos Engine</strong> &mdash; Because understanding emotion well enough to implement it teaches us something profound about ourselves.
  <br><br>
  Created by <strong>Victor Barrantes</strong> &mdash; 2026
</p>
