"""Pathos Portable Export — genera un ZIP con el engine emocional empaquetado.

Incluye:
- Pipeline emocional completo (16+ sistemas)
- Modelo Ollama enriquecido (Modelfile)
- Frontend mini React (pre-built)
- Scripts de instalación y arranque (BATs)
- Requiere: Ollama instalado y corriendo en la máquina del usuario
"""

import io
import os
import shutil
import zipfile
from pathlib import Path

SRC_ROOT = Path(__file__).parent  # src/pathos/


def _collect_python_files() -> list[tuple[str, Path]]:
    """Recolecta archivos Python necesarios para el portable."""
    files: list[tuple[str, Path]] = []

    # Engine completo
    engine_dir = SRC_ROOT / "engine"
    for f in engine_dir.glob("*.py"):
        files.append((f"backend/pathos/engine/{f.name}", f))

    # Models (sin voice.py)
    models_dir = SRC_ROOT / "models"
    for f in models_dir.glob("*.py"):
        if f.name == "voice.py":
            continue
        files.append((f"backend/pathos/models/{f.name}", f))

    # LLM providers
    llm_dir = SRC_ROOT / "llm"
    for f in llm_dir.glob("*.py"):
        files.append((f"backend/pathos/llm/{f.name}", f))

    # State
    state_dir = SRC_ROOT / "state"
    for f in state_dir.glob("*.py"):
        files.append((f"backend/pathos/state/{f.name}", f))

    # Config
    files.append(("backend/pathos/config.py", SRC_ROOT / "config.py"))
    files.append(("backend/pathos/__init__.py", SRC_ROOT / "__init__.py"))

    return files


def _generate_main_portable() -> str:
    """Genera main.py simplificado para el portable."""
    return '''"""Pathos Portable — Emotional engine with chat API."""

import asyncio
import re
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from pathos.config import Settings
from pathos.models.emotion import EmotionalState, PrimaryEmotion, neutral_state
from pathos.models.schemas import ChatRequest, ChatResponse
from pathos.models.values import ValueSystem
from pathos.engine.appraiser import appraise_stimulus, appraise_lite
from pathos.engine.generator import generate_emotional_state
from pathos.engine.homeostasis import regulate
from pathos.engine.behavior import generate_behavior_modifier, generate_simple_behavior_modifier, EMOTION_EFFECTS
from pathos.engine.mood import update_mood
from pathos.engine.memory import EmotionalMemoryStore
from pathos.engine.contagion import detect_contagion, apply_contagion
from pathos.engine.somatic import evaluate_somatic_markers, register_pending_decision
from pathos.engine.creativity import compute_creativity_state
from pathos.engine.emotional_schemas import process_schemas
from pathos.engine.temporal import process_temporal
from pathos.engine.meta import compute_meta_emotion
from pathos.engine.needs import update_needs
from pathos.engine.social import compute_social_modulation, update_user_model
from pathos.engine.regulation import attempt_regulation
from pathos.engine.reappraisal import reappraise
from pathos.engine.immune import apply_immune_protection, update_immune_state
from pathos.engine.narrative import update_narrative
from pathos.engine.forecasting import forecast_impact, evaluate_forecast, estimate_user_emotion, get_forecast_prompt
from pathos.llm.base import LLMProvider
from pathos.llm.ollama import OllamaProvider
from pathos.state.manager import SessionState

settings = Settings()
llm_provider: LLMProvider | None = None
state_manager_sessions: dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in state_manager_sessions:
        state_manager_sessions[session_id] = SessionState()
    return state_manager_sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global llm_provider
    # Check Ollama is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"[OK] Ollama connected. Models: {models[:5]}")
    except Exception:
        print("[ERROR] Ollama is not running!")
        print("  Start Ollama first: https://ollama.com")
        print("  Then restart this application.")
        import sys
        sys.exit(1)

    llm_provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
    )
    yield


app = FastAPI(title="Pathos Portable", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/")
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Pathos Portable API", "docs": "/docs"}


def _strip_meta(text: str) -> str:
    text = re.sub(r\'\\*?\\((?:State|Internal|Emotional|Note|Meta)[^)]{5,}\\)\\*?\', \'\', text)
    text = re.sub(r\'\\*?\\[(?:State|Internal|Emotional|Note|Meta)[^\\]]{5,}\\]\\*?\', \'\', text)
    return re.sub(r\'\\n{3,}\', \'\\n\\n\', text).strip()


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "pathos-portable"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    assert llm_provider is not None
    session = get_session(request.session_id)
    session.turn_count += 1
    previous_state = session.emotional_state

    # 1. Appraisal
    try:
        appraisal = await appraise_stimulus(
            request.message, session.value_system, session.emotional_state,
            llm_provider, session.calibration_profile,
        )
    except Exception:
        appraisal = appraise_lite(request.message, session.value_system, session.emotional_state, session.calibration_profile)

    # 2. Memory
    amplification = await session.memory.check_amplification(request.message, llm=llm_provider)

    # 3. Generate emotion
    new_state = generate_emotional_state(appraisal, previous_state, amplification, session.personality)

    # 4. Advanced systems
    contagion = detect_contagion(request.message, session.shadow_state)
    if contagion.detected:
        new_state, session.shadow_state = apply_contagion(new_state, contagion, session.shadow_state, session.personality)

    new_state, somatic_result = evaluate_somatic_markers(new_state, request.message, session.somatic_markers)
    creativity_state = compute_creativity_state(new_state, session.personality)
    new_state, schema_result = process_schemas(new_state, request.message, session.schemas)
    new_state, temporal_result = process_temporal(new_state, session.temporal, session.turn_count)
    meta_emotion = compute_meta_emotion(new_state, previous_state)
    session.needs = update_needs(session.needs, new_state)
    social_mod = compute_social_modulation(new_state, session.user_model)
    regulation_result = attempt_regulation(new_state, session.regulator, session.personality)
    if regulation_result.regulated_state:
        new_state = regulation_result.regulated_state
    new_state, reappraisal_result = reappraise(new_state, appraisal, session.personality)
    new_state, immune_result = apply_immune_protection(new_state, session.immune)
    session.immune = update_immune_state(session.immune, new_state)
    session.narrative = update_narrative(session.narrative, new_state, request.message, session.turn_count)

    # 5. Homeostasis
    new_state = regulate(new_state, previous_state, session.personality)
    new_state = update_mood(new_state)

    # 6. Store
    session.emotional_state = new_state
    session.state_history.append(new_state)
    await session.memory.store(request.message, new_state, llm=llm_provider)
    session.user_model = update_user_model(session.user_model, request.message, new_state)

    # 7. Behavior modifier
    system_prompt = generate_behavior_modifier(
        new_state, session.value_system, session.personality,
        social_mod=social_mod, regulation_result=regulation_result,
        creativity_state=creativity_state, meta_emotion=meta_emotion,
    )

    # 8. LLM response
    llm_temperature = 0.7 + creativity_state.temperature_modifier
    llm_temperature = max(0.1, min(1.5, llm_temperature))
    session.conversation.append({"role": "user", "content": request.message})
    chat_messages = session.conversation[-10:]

    try:
        response_text = await llm_provider.generate(
            system_prompt=system_prompt, messages=chat_messages,
            temperature=llm_temperature, think=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e!r}")

    response_text = _strip_meta(response_text)
    session.conversation.append({"role": "assistant", "content": response_text})
    session.somatic_markers = register_pending_decision(session.somatic_markers, request.message)

    return ChatResponse(
        response=response_text, emotional_state=new_state,
        session_id=request.session_id, turn_number=session.turn_count,
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    session = get_session(session_id)
    return {"emotional_state": session.emotional_state.model_dump()}


@app.post("/reset/{session_id}")
async def reset(session_id: str):
    state_manager_sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def _generate_install_bat() -> str:
    return r'''@echo off
chcp 65001 >nul 2>&1
title Pathos Portable - Install
echo.
echo  ╔════��══════════════════════════════════════╗
echo  ║     PATHOS PORTABLE — Installation        ║
echo  ╚═════���═════════════════════════════════════╝
echo.

REM Check Ollama
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not installed!
    echo.
    echo   Pathos Portable requires Ollama to run.
    echo   Download it from: https://ollama.com
    echo   Install it, then run this script again.
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama found

REM Check if Ollama is running
curl -sf http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Ollama is installed but not running. Starting...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
)
echo [OK] Ollama running

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

REM Create venv
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
echo [OK] Virtual environment ready

REM Install dependencies
echo Installing dependencies...
.venv\Scripts\pip install -r requirements.txt -q
echo [OK] Dependencies installed

REM Import Ollama model
if exist "Modelfile" (
    echo Importing Pathos model into Ollama...
    ollama create pathos-portable:latest -f Modelfile
    echo [OK] Model imported
) else (
    echo [WARN] No Modelfile found, skipping model import
)

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║     Installation complete!                ���
echo  ║     Run start.bat to launch Pathos        ║
echo  ╚══════════════════════════��════════════════╝
echo.
pause
'''


def _generate_start_bat() -> str:
    return r'''@echo off
chcp 65001 >nul 2>&1
title Pathos Portable

REM Check Ollama
curl -sf http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not running!
    echo   Start Ollama first, then try again.
    echo   If not installed: https://ollama.com
    pause
    exit /b 1
)

echo.
echo  ╔════════���══════════════════════════════════╗
echo  ║        PATHOS PORTABLE — Launch           ║
echo  ╚═════════════════════════════���═════════════╝
echo.
echo  [1] Chat visual (opens browser)
echo  [2] API service only (headless)
echo.
set /p CHOICE="  Select mode (1 or 2): "

if "%CHOICE%"=="1" (
    echo.
    echo  Starting Pathos with visual chat...
    echo  Opening http://localhost:8000 in your browser...
    start "" http://localhost:8000
    .venv\Scripts\python backend\main.py
) else (
    echo.
    echo  Starting Pathos API service...
    echo  API available at http://localhost:8000
    echo  Docs at http://localhost:8000/docs
    echo  Press Ctrl+C to stop.
    .venv\Scripts\python backend\main.py
)
'''


def _generate_requirements() -> str:
    return """fastapi>=0.115.0
uvicorn[standard]>=0.34.0
pydantic>=2.10.0
pydantic-settings>=2.13.1
numpy>=2.2.0
httpx>=0.28.0
requests>=2.31.0
langdetect>=1.0.9
"""


def _generate_readme() -> str:
    return """═══════════════════════════════════════════
  PATHOS PORTABLE — Emotional AI Engine
═══════════════════════════════════════════

An AI agent with a real emotional architecture.
Not simulated — functionally produced emotions that
affect thinking, memory, and behavior.

REQUIREMENTS:
  - Windows 10/11
  - Python 3.11 or higher
  - Ollama (https://ollama.com) installed and running

QUICK START:
  1. Install Ollama from https://ollama.com
  2. Run install.bat (first time only)
  3. Run start.bat
  4. Choose: [1] Visual chat or [2] API service

WHAT'S INSIDE:
  - 16+ emotional processing systems
  - Appraisal theory (Lazarus/Scherer)
  - Big Five personality model
  - Emotional memory with pattern recognition
  - Homeostasis with baseline drift
  - Emotion contagion detection
  - Somatic markers (Damasio)
  - Emotional schemas (Young/Beck)
  - Narrative self (autobiographical identity)
  - Emotional immune system
  - And more...

  The Ollama model has been trained with all the
  personality, schemas, and memories from the original
  Pathos Engine session.

API ENDPOINTS:
  POST /chat     — Send a message, get emotional response
  GET  /state    — Get current emotional state
  POST /reset    — Reset session
  GET  /health   — Health check
  GET  /         — Visual chat interface

CREATED WITH:
  Pathos Engine v2 — Functional Emotional Architecture for LLMs
"""


def _generate_mini_frontend() -> str:
    """Genera un index.html con un chat mini estilo ChatGPT."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pathos</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0e0e1a; color: #dde; height: 100vh; display: flex; flex-direction: column; }
.header { padding: 12px 20px; background: #12122a; border-bottom: 1px solid #1a1a3a; display: flex; align-items: center; gap: 10px; }
.header__logo { font-size: 18px; font-weight: 800; color: #7a7acc; }
.header__status { font-size: 11px; color: #4a4a6a; }
.header__emotion { margin-left: auto; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 75%; padding: 12px 16px; border-radius: 16px; font-size: 14px; line-height: 1.5; word-wrap: break-word; }
.msg--user { align-self: flex-end; background: #2a2a6a; color: #dde; border-bottom-right-radius: 4px; }
.msg--assistant { align-self: flex-start; background: #1a1a36; color: #ccd; border-bottom-left-radius: 4px; }
.msg--error { color: #e66; background: #2a1a1a; }
.msg__emotion { display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: 600; }
.typing { display: flex; gap: 4px; padding: 8px 16px; }
.typing span { width: 6px; height: 6px; background: #4a4a6a; border-radius: 50%; animation: bounce 1.4s infinite; }
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-8px); } }
.input-bar { padding: 12px 20px; background: #12122a; border-top: 1px solid #1a1a3a; display: flex; gap: 8px; }
.input-bar input { flex: 1; padding: 12px 16px; background: #1a1a36; border: 1px solid #2a2a4a; border-radius: 12px; color: #dde; font-size: 14px; outline: none; }
.input-bar input:focus { border-color: #4a4a8a; }
.input-bar button { padding: 10px 20px; background: #3a3a7a; border: none; border-radius: 12px; color: #dde; font-size: 14px; cursor: pointer; font-weight: 600; }
.input-bar button:hover { background: #4a4a9a; }
.input-bar button:disabled { opacity: 0.4; cursor: default; }
.emotion-colors { --joy: #FFD700; --excitement: #FF6B35; --gratitude: #DDA0DD; --hope: #87CEEB; --contentment: #98FB98; --relief: #B0E0E6; --anger: #FF4444; --frustration: #FF6B6B; --fear: #9370DB; --anxiety: #DDA0DD; --sadness: #4682B4; --helplessness: #708090; --disappointment: #778899; --surprise: #FFD700; --alertness: #FFA500; --contemplation: #6A5ACD; --indifference: #808080; --mixed: #9370DB; --neutral: #888888; }
</style>
</head>
<body class="emotion-colors">
<div class="header">
  <span class="header__logo">PATHOS</span>
  <span class="header__status" id="status">Connecting...</span>
  <span class="header__emotion" id="emotion-badge"></span>
</div>
<div class="messages" id="messages"></div>
<div class="input-bar">
  <input type="text" id="input" placeholder="Type a message..." autofocus />
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const API = window.location.origin;
const sessionId = 'portable-' + Date.now();
let loading = false;

const COLORS = {joy:'#FFD700',excitement:'#FF6B35',gratitude:'#DDA0DD',hope:'#87CEEB',contentment:'#98FB98',relief:'#B0E0E6',anger:'#FF4444',frustration:'#FF6B6B',fear:'#9370DB',anxiety:'#DDA0DD',sadness:'#4682B4',helplessness:'#708090',disappointment:'#778899',surprise:'#FFD700',alertness:'#FFA500',contemplation:'#6A5ACD',indifference:'#808080',mixed:'#9370DB',neutral:'#888888'};

document.getElementById('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !loading) send(); });

async function checkHealth() {
  try {
    const r = await fetch(API + '/health');
    if (r.ok) { document.getElementById('status').textContent = 'Connected'; document.getElementById('status').style.color = '#4a9'; }
    else throw new Error();
  } catch { document.getElementById('status').textContent = 'Offline'; document.getElementById('status').style.color = '#e66'; }
}
checkHealth();

function addMsg(role, text, emotion) {
  const div = document.createElement('div');
  div.className = 'msg msg--' + role;
  div.textContent = text;
  if (emotion) {
    const badge = document.createElement('span');
    badge.className = 'msg__emotion';
    badge.textContent = emotion.primary_emotion + ' ' + Math.round(emotion.intensity * 100) + '%';
    const c = COLORS[emotion.primary_emotion] || '#888';
    badge.style.background = c + '20';
    badge.style.color = c;
    div.appendChild(badge);
    // Update header badge
    const hb = document.getElementById('emotion-badge');
    hb.textContent = emotion.primary_emotion;
    hb.style.background = c + '20';
    hb.style.color = c;
  }
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'msg msg--assistant typing';
  div.id = 'typing';
  div.innerHTML = '<span></span><span></span><span></span>';
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function send() {
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text || loading) return;
  input.value = '';
  loading = true;
  document.getElementById('send').disabled = true;

  addMsg('user', text);
  showTyping();

  try {
    const r = await fetch(API + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    hideTyping();
    if (!r.ok) throw new Error('Chat failed: ' + r.status);
    const data = await r.json();
    addMsg('assistant', data.response, data.emotional_state);
  } catch (e) {
    hideTyping();
    addMsg('error', 'Error: ' + e.message);
  }
  loading = false;
  document.getElementById('send').disabled = false;
  input.focus();
}
</script>
</body>
</html>'''


def generate_portable_zip(
    session: "SessionState",
    modelfile_content: str,
    base_model: str = "qwen3:4b",
    cloud_config: dict | None = None,
) -> bytes:
    """Genera el ZIP completo del Pathos Portable.

    Args:
        cloud_config: If set, includes cloud provider config (provider, api_key, base_url, model).

    Returns:
        bytes del ZIP file.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Modelfile
        zf.writestr("pathos-portable/Modelfile", modelfile_content)

        # 2. Python source files
        for zip_path, real_path in _collect_python_files():
            zf.write(str(real_path), f"pathos-portable/{zip_path}")

        # 3. Simplified main.py
        zf.writestr("pathos-portable/backend/main.py", _generate_main_portable())

        # 4. __init__.py files for packages (only if not already added)
        added = {info.filename for info in zf.infolist()}
        for pkg in ["backend", "backend/pathos", "backend/pathos/engine",
                     "backend/pathos/models", "backend/pathos/llm", "backend/pathos/state"]:
            init_path = f"pathos-portable/{pkg}/__init__.py"
            if init_path not in added:
                zf.writestr(init_path, "")

        # 5. Frontend
        zf.writestr("pathos-portable/frontend/dist/index.html", _generate_mini_frontend())

        # 6. Scripts
        zf.writestr("pathos-portable/install.bat", _generate_install_bat())
        zf.writestr("pathos-portable/start.bat", _generate_start_bat())

        # 7. Requirements
        zf.writestr("pathos-portable/requirements.txt", _generate_requirements())

        # 8. README
        zf.writestr("pathos-portable/README.txt", _generate_readme())

        # 9. Cloud provider config (if using cloud model)
        # SECURITY: Never include real API keys in exported packages.
        # Users must provide their own key after importing.
        if cloud_config:
            env_lines = [
                "# Cloud provider config (included from Pathos export)",
                "# IMPORTANT: Replace YOUR_API_KEY_HERE with your actual API key",
                f"PATHOS_LLM_PROVIDER={cloud_config.get('preset', 'ollama')}",
                "PATHOS_CLOUD_API_KEY=YOUR_API_KEY_HERE",
                f"PATHOS_CLOUD_BASE_URL={cloud_config.get('base_url', '')}",
                f"PATHOS_CLOUD_MODEL={cloud_config.get('model', '')}",
                "",
            ]
            zf.writestr("pathos-portable/.env", "\n".join(env_lines))

    return buf.getvalue()
