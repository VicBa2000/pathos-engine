#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; exit 1; }
dim()   { echo -e "${DIM}$1${NC}"; }

echo ""
echo -e "${BOLD}${CYAN}  ╔═══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}  ║      PATHOS ENGINE v3 + ARK — Setup        ║${NC}"
echo -e "${BOLD}${CYAN}  ║   Emotional Architecture for LLMs         ║${NC}"
echo -e "${BOLD}${CYAN}  ╚═══════════════════════════════════════════╝${NC}"
echo ""

# --- Windows Python Scripts paths (MINGW64 needs Unix-style paths) ---
add_to_path() {
  local unix_path
  unix_path="$(cygpath -u "$1" 2>/dev/null || echo "$1")"
  if [ -d "$unix_path" ]; then
    export PATH="$unix_path:$PATH"
  fi
}

add_to_path "$APPDATA/Python/Python313/Scripts"
add_to_path "$APPDATA/Python/Python312/Scripts"
add_to_path "$APPDATA/Python/Python311/Scripts"
add_to_path "$LOCALAPPDATA/Programs/Python/Python313/Scripts"
add_to_path "$LOCALAPPDATA/Programs/Python/Python312/Scripts"
add_to_path "$HOME/.local/bin"
add_to_path "$HOME/.cargo/bin"

# ==========================================================================
# 1. Check required tools
# ==========================================================================
echo -e "${CYAN}--- Checking required tools ---${NC}"

command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || error "Python not found. Install Python 3.13+"
PYTHON=$(command -v python 2>/dev/null || command -v python3)
PY_VERSION=$($PYTHON --version 2>&1)
info "Python: $PY_VERSION"

command -v node >/dev/null 2>&1 || error "Node.js not found. Install Node 18+"
info "Node: $(node --version)"

command -v npm >/dev/null 2>&1 || error "npm not found. Install Node 18+"

# uv (Python package manager) — optional, pip fallback available
HAS_UV=false
if command -v uv >/dev/null 2>&1; then
  HAS_UV=true
  info "uv found"
else
  warn "uv not found. Trying to install via pip..."
  $PYTHON -m pip install uv 2>/dev/null || pip3 install uv 2>/dev/null || true
  add_to_path "$APPDATA/Python/Python313/Scripts"
  add_to_path "$APPDATA/Python/Python312/Scripts"
  add_to_path "$LOCALAPPDATA/Programs/Python/Python313/Scripts"
  if command -v uv >/dev/null 2>&1; then
    HAS_UV=true
    info "uv installed"
  else
    warn "uv not available. Using pip directly."
  fi
fi

# ==========================================================================
# 2. Backend dependencies
# ==========================================================================
echo ""
echo -e "${CYAN}--- Backend dependencies ---${NC}"

VENV_OK=false
if [ -d ".venv" ]; then
  # Venv exists — verify its Python is functional (breaks on disk/path changes)
  VENV_PY=""
  if [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PY=".venv/Scripts/python"
  elif [ -f ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
  fi
  if [ -n "$VENV_PY" ] && $VENV_PY --version >/dev/null 2>&1; then
    VENV_OK=true
  else
    warn "Python venv found but broken (disk/path change?). Recreating..."
    rm -rf .venv
  fi
fi

if ! $VENV_OK; then
  warn "Python venv not found. Creating..."
  if $HAS_UV; then
    uv venv
  else
    $PYTHON -m venv .venv
  fi
fi
info "Python venv ready"

# IMPORTANT: Use the venv Python for all subsequent checks and installs
# so that dependencies are installed in the venv, not the global Python
if [ -f ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python"
elif [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

# Ensure pip is available in the venv
if ! $PYTHON -m pip --version >/dev/null 2>&1; then
  warn "pip not found in venv. Bootstrapping..."
  $PYTHON -m ensurepip 2>&1 | tail -3
fi

warn "Syncing backend dependencies..."
if $HAS_UV; then
  uv sync 2>&1 | tail -3
else
  $PYTHON -m pip install -e ".[dev]" -q 2>&1 | tail -3
fi
info "Backend core dependencies installed"

# python-multipart (required for file uploads / ASR endpoint)
if ! $PYTHON -c "import multipart" 2>/dev/null; then
  warn "Installing python-multipart (required for voice upload)..."
  if $HAS_UV; then
    uv pip install python-multipart --python .venv 2>/dev/null
  else
    $PYTHON -m pip install python-multipart -q 2>/dev/null
  fi
  info "python-multipart installed"
fi

# ==========================================================================
# 3. Ollama models (optional, non-blocking)
# ==========================================================================
echo ""
echo -e "${CYAN}--- Ollama models ---${NC}"

OLLAMA_PULL_PIDS=()
if command -v ollama >/dev/null 2>&1; then
  info "Ollama found"
  # Check if Ollama is actually running
  if ollama list >/dev/null 2>&1; then
    if ! ollama list 2>/dev/null | grep -q "qwen3:4b"; then
      warn "Pulling qwen3:4b model (this may take a while)..."
      ollama pull qwen3:4b &
      OLLAMA_PULL_PIDS+=($!)
    else
      info "qwen3:4b ready"
    fi
    if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
      warn "Pulling nomic-embed-text model..."
      ollama pull nomic-embed-text &
      OLLAMA_PULL_PIDS+=($!)
    else
      info "nomic-embed-text ready"
    fi
  else
    warn "Ollama installed but not running. Start it with: ollama serve"
    warn "Models will be pulled when Ollama is available."
  fi
else
  warn "Ollama not found. Install it for local LLM support: https://ollama.com"
  warn "Or set PATHOS_LLM_PROVIDER=claude and PATHOS_ANTHROPIC_API_KEY in .env"
fi

# ==========================================================================
# 4. Voice dependencies (optional — TTS + ASR)
# ==========================================================================
echo ""
echo -e "${CYAN}--- Voice dependencies (optional) ---${NC}"

# --- PyTorch ---
HAS_TORCH=false
if $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
  HAS_TORCH=true
  TORCH_INFO=$($PYTHON -c "import torch; cuda='CUDA '+torch.version.cuda if torch.cuda.is_available() else 'CPU only'; print(f'PyTorch {torch.__version__} ({cuda})')" 2>/dev/null)
  info "$TORCH_INFO"
else
  warn "PyTorch not found. Installing with CUDA 12.4 (this may take a while)..."
  if $HAS_UV; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --python .venv 2>&1 | tail -5
  else
    $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -5
  fi
  if $PYTHON -c "import torch" 2>/dev/null; then
    HAS_TORCH=true
    info "PyTorch installed"
  else
    warn "PyTorch installation failed. Voice features will be unavailable."
    echo -e "  ${DIM}Manual install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124${NC}"
  fi
fi

# --- TTS (Kokoro + Parler-TTS) ---
HAS_TTS=false
HAS_PARLER=false
if $HAS_TORCH; then
  # --- Kokoro TTS (primary backend) ---
  if $PYTHON -c "from kokoro import KPipeline" 2>/dev/null; then
    HAS_TTS=true
    info "Kokoro TTS ready"
  else
    warn "Installing Kokoro TTS + dependencies..."
    # spacy needs pre-built binaries (no Visual Studio required)
    if ! $PYTHON -c "import spacy" 2>/dev/null; then
      if $HAS_UV; then
        uv pip install spacy --prefer-binary --python .venv 2>&1 | tail -3
      else
        $PYTHON -m pip install spacy --prefer-binary -q 2>&1 | tail -3
      fi
    fi
    # phonemizer (required for multi-language support via espeak)
    if ! $PYTHON -c "import phonemizer" 2>/dev/null; then
      if $HAS_UV; then
        uv pip install phonemizer --python .venv 2>&1 | tail -3
      else
        $PYTHON -m pip install phonemizer -q 2>&1 | tail -3
      fi
    fi
    # kokoro itself (install without deps to avoid numpy==1.26.4 conflict on Python 3.13)
    if $HAS_UV; then
      uv pip install kokoro --no-deps --python .venv 2>&1 | tail -3
    else
      $PYTHON -m pip install kokoro --no-deps -q 2>&1 | tail -3
    fi
    # misaki (text processing for kokoro)
    if ! $PYTHON -c "import misaki" 2>/dev/null; then
      if $HAS_UV; then
        uv pip install misaki loguru num2words soundfile --python .venv 2>&1 | tail -3
      else
        $PYTHON -m pip install misaki loguru num2words soundfile -q 2>&1 | tail -3
      fi
    fi
    # scipy (required for audio post-processing: pitch shift, resample)
    if ! $PYTHON -c "import scipy" 2>/dev/null; then
      if $HAS_UV; then
        uv pip install scipy --python .venv 2>&1 | tail -3
      else
        $PYTHON -m pip install scipy -q 2>&1 | tail -3
      fi
    fi

    if $PYTHON -c "from kokoro import KPipeline" 2>/dev/null; then
      HAS_TTS=true
      info "Kokoro TTS installed"
    else
      warn "Kokoro TTS installation failed. TTS will be unavailable."
      echo -e "  ${DIM}Manual install: pip install spacy phonemizer kokoro soundfile${NC}"
    fi
  fi

  # --- espeak-ng (required for non-English languages in Kokoro) ---
  if $HAS_TTS; then
    if command -v espeak-ng >/dev/null 2>&1 || command -v espeak >/dev/null 2>&1 || [ -f "/c/Program Files/eSpeak NG/espeak-ng.exe" ] || [ -f "C:/Program Files/eSpeak NG/espeak-ng.exe" ]; then
      info "espeak-ng found (multi-language TTS available)"
    else
      warn "espeak-ng not found — TTS works for English, other languages need espeak-ng"
      echo -e "  ${DIM}Install: winget install espeak-ng.espeak-ng  (then restart terminal)${NC}"
    fi
  fi

  # --- Parler-TTS (expressive emotional TTS, English only) ---
  if $PYTHON -c "from parler_tts import ParlerTTSForConditionalGeneration" 2>/dev/null; then
    HAS_PARLER=true
    info "Parler-TTS ready (expressive emotions, English)"
  else
    warn "Installing Parler-TTS (expressive emotional voice, ~2.5GB model)..."
    if $HAS_UV; then
      uv pip install parler-tts --python .venv 2>&1 | tail -5
    else
      $PYTHON -m pip install parler-tts -q 2>&1 | tail -5
    fi
    if $PYTHON -c "from parler_tts import ParlerTTSForConditionalGeneration" 2>/dev/null; then
      HAS_PARLER=true
      info "Parler-TTS installed"
    else
      warn "Parler-TTS installation failed. Kokoro will be used for all voices."
      echo -e "  ${DIM}Manual install: pip install parler-tts${NC}"
    fi
  fi
fi

# --- ffmpeg (required for Whisper ASR to decode WebM/MP3/etc.) ---
HAS_FFMPEG=false
if command -v ffmpeg >/dev/null 2>&1; then
  HAS_FFMPEG=true
  info "ffmpeg found: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"
else
  warn "ffmpeg not found. Installing..."
  # Try winget (Windows), apt (Linux), brew (macOS)
  if command -v winget >/dev/null 2>&1; then
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements 2>&1 | tail -3
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y ffmpeg 2>&1 | tail -3
  elif command -v brew >/dev/null 2>&1; then
    brew install ffmpeg 2>&1 | tail -3
  fi
  # Re-check (winget may need terminal restart for PATH)
  if command -v ffmpeg >/dev/null 2>&1; then
    HAS_FFMPEG=true
    info "ffmpeg installed"
  else
    warn "ffmpeg auto-install failed. ASR needs ffmpeg for WebM audio."
    echo -e "  ${DIM}Manual install:${NC}"
    echo -e "  ${DIM}  Windows: winget install ffmpeg  (then restart terminal)${NC}"
    echo -e "  ${DIM}  Linux:   sudo apt install ffmpeg${NC}"
    echo -e "  ${DIM}  macOS:   brew install ffmpeg${NC}"
  fi
fi

# --- Whisper (ASR) ---
HAS_ASR=false
if $HAS_TORCH; then
  if $PYTHON -c "import whisper" 2>/dev/null; then
    if $HAS_FFMPEG; then
      HAS_ASR=true
      info "Whisper (ASR) ready"
    else
      warn "Whisper installed but ffmpeg missing — ASR won't work without ffmpeg"
    fi
  else
    warn "Installing openai-whisper..."
    if $HAS_UV; then
      uv pip install openai-whisper --python .venv 2>&1 | tail -5
    else
      $PYTHON -m pip install openai-whisper -q 2>&1 | tail -5
    fi
    if $PYTHON -c "import whisper" 2>/dev/null; then
      if $HAS_FFMPEG; then
        HAS_ASR=true
        info "Whisper installed"
      else
        warn "Whisper installed but ffmpeg missing — install ffmpeg to enable ASR"
      fi
    else
      warn "Whisper installation failed. ASR will be unavailable."
      echo -e "  ${DIM}Manual install: pip install openai-whisper${NC}"
    fi
  fi
fi

# Pre-download Whisper model (~461MB, only first time)
if $HAS_ASR; then
  if ! $PYTHON -c "import whisper; import os; p = os.path.join(os.path.expanduser('~'), '.cache', 'whisper', 'small.pt'); assert os.path.exists(p)" 2>/dev/null; then
    warn "Pre-downloading Whisper 'small' model (~461MB, first time only)..."
    $PYTHON -c "
import whisper
whisper.load_model('small', device='cpu')
print('  \033[0;32m[OK]\033[0m Whisper model cached')
" 2>&1
  else
    info "Whisper model already cached"
  fi
fi

# ==========================================================================
# 5. Steering / TransformersProvider dependencies (optional — direct LLM modification)
# ==========================================================================
echo ""
echo -e "${CYAN}--- Steering dependencies (optional) ---${NC}"

if $HAS_TORCH; then
  HAS_STEERING_DEPS=true

  # gguf (required for loading GGUF models via TransformersProvider)
  if ! $PYTHON -c "import gguf" 2>/dev/null; then
    warn "Installing gguf (required for Steering mode with GGUF models)..."
    if $HAS_UV; then
      uv pip install "gguf>=0.10.0" --python .venv 2>&1 | tail -3
    else
      $PYTHON -m pip install "gguf>=0.10.0" -q 2>&1 | tail -3
    fi
    if $PYTHON -c "import gguf" 2>/dev/null; then
      info "gguf installed"
    else
      HAS_STEERING_DEPS=false
      warn "gguf installation failed."
    fi
  else
    info "gguf ready"
  fi

  # accelerate (required for device_map="auto" in transformers)
  if ! $PYTHON -c "import accelerate" 2>/dev/null; then
    warn "Installing accelerate (required for TransformersProvider)..."
    if $HAS_UV; then
      uv pip install "accelerate>=0.25.0" --python .venv 2>&1 | tail -3
    else
      $PYTHON -m pip install "accelerate>=0.25.0" -q 2>&1 | tail -3
    fi
    if $PYTHON -c "import accelerate" 2>/dev/null; then
      info "accelerate installed"
    else
      HAS_STEERING_DEPS=false
      warn "accelerate installation failed."
    fi
  else
    info "accelerate ready"
  fi

  if $HAS_STEERING_DEPS; then
    info "Steering mode available (select 'Steering' in Model Manager)"
  fi
else
  warn "PyTorch not available — Steering mode disabled (needs torch + gguf + accelerate)"
fi

# ==========================================================================
# 6. Frontend dependencies
# ==========================================================================
echo ""
echo -e "${CYAN}--- Frontend dependencies ---${NC}"

cd "$ROOT/frontend"

if [ ! -d "node_modules" ]; then
  warn "Frontend node_modules not found. Installing..."
  npm install 2>&1 | tail -3
else
  # Always sync to catch new dependencies (e.g. @vladmandic/face-api)
  dim "  Syncing frontend packages..."
  npm install 2>&1 | tail -3
fi
info "Frontend dependencies ready"

# Copy face-api.js models to public/ (needed for facial AU detection)
FACE_API_MODELS="$ROOT/frontend/node_modules/@vladmandic/face-api/model"
FACE_API_PUBLIC="$ROOT/frontend/public/models"
if [ -d "$FACE_API_MODELS" ]; then
  mkdir -p "$FACE_API_PUBLIC"
  for f in tiny_face_detector_model.bin tiny_face_detector_model-weights_manifest.json \
           face_expression_model.bin face_expression_model-weights_manifest.json; do
    if [ -f "$FACE_API_MODELS/$f" ] && [ ! -f "$FACE_API_PUBLIC/$f" ]; then
      cp "$FACE_API_MODELS/$f" "$FACE_API_PUBLIC/"
    fi
  done
  info "Face-api.js models in public/models/"
fi

# Copy KTX2 basis transcoder to public/ (needed for 3D avatar textures)
BASIS_SRC="$ROOT/frontend/node_modules/three/examples/jsm/libs/basis"
BASIS_PUBLIC="$ROOT/frontend/public/basis"
if [ -d "$BASIS_SRC" ]; then
  mkdir -p "$BASIS_PUBLIC"
  for f in basis_transcoder.js basis_transcoder.wasm; do
    if [ -f "$BASIS_SRC/$f" ] && [ ! -f "$BASIS_PUBLIC/$f" ]; then
      cp "$BASIS_SRC/$f" "$BASIS_PUBLIC/"
    fi
  done
  info "KTX2 basis transcoder in public/basis/"
fi

cd "$ROOT"

# --- Wait for ollama pulls if any ---
for pid in "${OLLAMA_PULL_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
if [ ${#OLLAMA_PULL_PIDS[@]} -gt 0 ]; then
  info "Ollama models ready"
fi

# ==========================================================================
# 7. Start services
# ==========================================================================
echo ""
echo -e "${GREEN}${BOLD}  ╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}  ║         PATHOS ENGINE — Ready!            ║${NC}"
echo -e "${GREEN}${BOLD}  ╚═══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Backend:   ${CYAN}http://localhost:8000${NC}"
echo -e "  Frontend:  ${CYAN}http://localhost:5173${NC}"
echo -e "  API docs:  ${CYAN}http://localhost:8000/docs${NC}"
echo ""
echo -e "  ${BOLD}Voice status:${NC}"
if $HAS_TTS; then
  echo -e "    TTS (Kokoro):     ${GREEN}Available${NC} — toggle 'Voice' in the UI"
else
  echo -e "    TTS (Kokoro):     ${DIM}Not installed${NC}"
fi
if $HAS_PARLER; then
  echo -e "    TTS (Parler):     ${GREEN}Available${NC} — complex emotions in English"
else
  echo -e "    TTS (Parler):     ${DIM}Not installed (optional)${NC}"
fi
if $HAS_ASR; then
  echo -e "    ASR (Whisper):    ${GREEN}Available${NC} — toggle 'Mic' in the UI"
else
  if $HAS_FFMPEG; then
    echo -e "    ASR (Whisper):    ${DIM}Not installed${NC}"
  else
    echo -e "    ASR (Whisper):    ${DIM}Not installed (needs ffmpeg + openai-whisper)${NC}"
  fi
fi
echo ""
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop both services"
echo ""

# Trap to kill both on exit
cleanup() {
  echo ""
  info "Shutting down..."
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null
  info "Done"
}
trap cleanup EXIT INT TERM

# Start backend
# --reload disabled: on Windows, watchfiles + multiprocessing spawn crashes (WinError 87).
# Always use venv python directly — 'uv run' has trampoline bugs on Windows/Git Bash.
$PYTHON -m uvicorn pathos.main:app --host "${PATHOS_HOST:-127.0.0.1}" --port 8000 &
BACKEND_PID=$!

# Start frontend
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

cd "$ROOT"

# Wait for either to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
