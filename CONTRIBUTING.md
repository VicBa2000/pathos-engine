# Contributing to Pathos Engine

Thank you for your interest in contributing to Pathos Engine. This document explains how to get started, what we expect from contributions, and how the project is organized.

---

## Before You Start

1. **Read the README.md** to understand what Pathos Engine is and how it works.
2. **Read the NOTICE** to understand the theoretical foundations.
3. **Fork freely.** You are welcome to fork, modify, extend, and experiment with the project in any direction you want. That's the point of open source.
4. **PRs are welcome** but this project is maintained on a best-effort basis. There is no guaranteed review timeline. If your PR follows the guidelines below and passes all tests, it has a good chance of being merged — but no promises.
5. **You don't need permission to start working.** Build what you want. If it's good, submit it.

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/pathos-engine.git
cd pathos-engine

# Option A: Use start.sh (sets up everything)
./start.sh

# Option B: Manual setup
python -m venv .venv
source .venv/bin/activate    # .venv\Scripts\activate on Windows
pip install -e ".[dev]"

cd frontend
npm install
cd ..

# Run tests to verify setup
python -m pytest tests/ -v
cd frontend && npx tsc --noEmit
```

You need **Ollama** running with `qwen3:4b` for full integration testing, but unit tests run without it.

---

## Project Architecture

Understanding the architecture helps you contribute effectively.

### Backend (`src/pathos/`)

```
main.py              The FastAPI app. Pipeline variants (chat, research, sandbox, raw).
                     This is where pipeline steps are orchestrated.
api_routes.py        Emotion API as a Service — standalone REST endpoints (/api/v1/).
config.py            Pydantic settings — all configurable via env vars.
engine/              Emotional processing modules. Each system = 1 file.
  emotion_processor.py   Standalone pipeline (runs without LLM).
  external_signals.py    Real-world signal fusion (heart rate, weather, etc.).
models/              Pydantic data models (immutable state objects).
  coupling.py            Cross-dimensional ODE coupling matrix.
  emotion_api.py         Emotion API request/response schemas.
  external_signals.py    Signal config + source definitions.
llm/                 LLM provider abstraction (Ollama, Claude, OpenAI-compat).
voice/               TTS + ASR (optional, not needed for core development).
state/manager.py     Session state management (in-memory).
```

**Key principle:** Each engine module is independently testable. If you add a new system, it should have its own file in `engine/` and its own test file.

### Frontend (`frontend/src/`)

```
App.tsx              Main component. Manages all state and mode switching.
components/          One component per file. CSS in dedicated files.
api/client.ts        API client — mirrors every backend endpoint.
types/emotion.ts     TypeScript types matching backend Pydantic schemas.
```

**Key principle:** Types in `emotion.ts` must match the backend schemas exactly. If you add a new data structure to the backend, add the corresponding TypeScript type.

---

## How to Add a New Emotional System

This is the most common significant contribution. Follow this pattern:

### 1. Create the engine module

```
src/pathos/engine/your_system.py
```

- Use Pydantic `BaseModel` for any data structures
- Pure functions where possible (input state -> output state)
- Type hints on everything
- Clamp all numeric values to their defined ranges

### 2. Create tests

```
tests/test_your_system.py
```

- Test edge cases and range boundaries
- Test that outputs are deterministic for the same inputs
- Use the naming pattern `TestYourFeature` with descriptive method names
- Aim for at least 10 tests per module

### 3. Integrate into the pipeline

In `main.py`, add your system as a new step in the appropriate location within the pipeline. Follow the existing pattern:

```python
# N. Your System
t0 = time.perf_counter()
your_result = your_function(new_state, ...)
trace_steps.append(PipelineStep(
    name="your_system", label="Your System",
    active=True, duration_ms=(time.perf_counter() - t0) * 1000,
    summary="...",
    impact="medium",
    details={...},
))
```

### 4. Expose in behavior modifier

If your system should influence the agent's response, add it to `behavior.py` in `generate_behavior_modifier()` as an optional parameter with default `None`.

### 5. Add to Research Panel (optional)

If your system produces interesting observable data, expose it through the research endpoint and add a section in `frontend/src/components/ResearchPanel.tsx`.

### 6. Update documentation

- Add to `manual.txt` (feature description)
- Add to git commit message (clear description of what and why)
- Add to `README.md` (system table, pipeline diagram if step order changed)

---

## Code Style

### Python

- **Type hints everywhere.** No untyped function signatures.
- **Pydantic** for data models, `dataclass` only for lightweight internal results.
- **Clamp numeric values.** Valence: -1 to 1. Arousal/intensity: 0 to 1. Never trust inputs.
- **No `any` equivalent.** Be specific about types.
- **Logging** via `logging.getLogger(__name__)`. Never `print()` in production code.
- **Async** where I/O is involved. Pure computation stays synchronous.

### TypeScript

- **Strict mode.** No `any` types.
- **Interfaces and types** defined in `types/emotion.ts` before implementation.
- **D3.js over SVG** for new visualizations (established pattern).
- **CSS in dedicated files** per component (no inline styles).

### General

- No unnecessary abstractions. Three similar lines > premature abstraction.
- No feature flags or backwards-compatibility shims. Just change the code.
- No extra error handling for scenarios that can't happen.
- Comments only where logic isn't self-evident.

---

## Testing Requirements

Every PR must:

1. **Pass all existing tests:** `python -m pytest tests/ -x`
2. **Pass TypeScript check:** `cd frontend && npx tsc --noEmit`
3. **Pass Vite build:** `cd frontend && npx vite build`
4. **Include new tests** for any new functionality

Run the full check before submitting:

```bash
python -m pytest tests/ -x -q && cd frontend && npx tsc --noEmit && npx vite build
```

---

## Commit Messages

Use concise, descriptive commit messages:

```
add: somatic markers engine module + 12 tests
fix: PDF content being processed as text in web_search
update: research depth now configurable via env vars
refactor: extract SSE event handler in AutonomousResearchPanel
```

Prefixes: `add`, `fix`, `update`, `refactor`, `docs`, `test`

---

## Pull Request Process

1. **Fork** the repository and create a branch from `main`
2. **Make your changes** following the code style and architecture guidelines
3. **Add tests** for new functionality
4. **Run the full test suite** (Python tests + TypeScript + Vite build)
5. **Update documentation** (`manual.txt`, `README.md` as needed)
6. **Open a PR** with a clear description of what and why

### PR Description Template

```markdown
## What
Brief description of the change.

## Why
What problem does this solve or what capability does it add?

## How
Key implementation decisions, trade-offs, or things reviewers should know.

## Testing
How was this tested? New tests added?

## Documentation
What docs were updated?
```

### What Makes a Good PR

PRs that follow these guidelines are more likely to be merged:

- Follows the existing architecture patterns
- Has tests that cover edge cases
- Numeric values are clamped to valid ranges
- Doesn't introduce unnecessary complexity
- New systems are independently testable
- TypeScript types match backend schemas
- All existing tests still pass

---

## Contributor License Agreement (CLA)

By submitting a pull request, you agree to the project's CLA. This grants the project maintainer the right to distribute your contribution under the project's license (AGPL-3.0) and, if applicable, under a commercial license.

The CLA is required to maintain the dual-licensing model that keeps the project sustainable. You retain copyright of your contributions.

---

## Areas Where Help Is Welcome

**New emotional systems** (must be grounded in psychological research):
- Emotional granularity (expanding the 19-emotion vocabulary)
- Attachment styles (Bowlby) influencing social cognition
- Cognitive load effects on emotional processing
- Group emotion dynamics (multi-agent scenarios)

**Infrastructure:**
- Redis backend for persistent state across sessions
- WebSocket support for streaming responses
- Docker containerization
- CI/CD pipeline (GitHub Actions)

**Frontend:**
- New D3.js visualizations
- Mobile-responsive layout
- Accessibility improvements
- Theme customization

**Testing:**
- Integration tests for full pipeline
- Endpoint tests for all 76 API routes
- Performance benchmarks
- Cross-model testing (different Ollama models)

**Documentation:**
- Tutorials for getting started
- Architecture deep-dives
- Video walkthroughs
- Translation to other languages

---

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Short version: be respectful, be constructive, focus on the work. We're building something interesting together.

---

## Questions?

- **Issues:** Open a GitHub issue for bugs or feature discussions
- **Discussions:** Use GitHub Discussions for general questions

Note: This project is maintained on a best-effort basis. Response times may vary. The best way to get something done is to do it yourself and submit a PR.

Thank you for contributing to Pathos Engine.
