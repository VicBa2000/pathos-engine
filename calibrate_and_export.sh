#!/usr/bin/env bash
# =============================================================================
# PATHOS ENGINE v2 - Calibrate & Export to Ollama
# =============================================================================
# Este script:
#   1. Configura la personalidad del agente (Big Five)
#   2. Envia todos los escenarios de calibracion al backend
#   3. Aplica el perfil de calibracion
#   4. Exporta un modelo Ollama con la arquitectura emocional bakeada
#
# Requisitos:
#   - Backend corriendo en localhost:8000
#   - Ollama instalado y corriendo
#
# Uso:
#   ./calibrate_and_export.sh [base_model] [model_name] [personality]
#
# Personality presets: balanced, sensitive, resilient, creative, empathic
#
# Ejemplos:
#   ./calibrate_and_export.sh                              # qwen3:4b -> pathos, balanced
#   ./calibrate_and_export.sh llama3:8b pathos-llama       # llama export
#   ./calibrate_and_export.sh qwen3:4b pathos sensitive    # sensitive personality
# =============================================================================

set -e

BASE_MODEL="${1:-qwen3:4b}"
MODEL_NAME="${2:-pathos}"
PERSONALITY="${3:-balanced}"
SESSION_ID="calibration-export"
API="http://localhost:8000"

# Validate inputs — these are interpolated into Python heredocs
if [[ ! "$BASE_MODEL" =~ ^[a-zA-Z0-9_.:/-]+$ ]]; then
  echo "ERROR: BASE_MODEL contains invalid characters"; exit 1
fi
if [[ ! "$MODEL_NAME" =~ ^[a-zA-Z0-9_.:/-]+$ ]]; then
  echo "ERROR: MODEL_NAME contains invalid characters"; exit 1
fi
if [[ ! "$PERSONALITY" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "ERROR: PERSONALITY must be alphanumeric/hyphens/underscores only"; exit 1
fi

# Python command: 'python3' on Linux/macOS, 'python' on Windows
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: Python not found"; exit 1
fi

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
step()  { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# Banner
echo ""
echo -e "${BOLD}${CYAN}  ╔═══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}  ║  PATHOS ENGINE v2 — Calibrate & Export   ║${NC}"
echo -e "${BOLD}${CYAN}  ║  10 Systems • 45 Scenarios • Ollama      ║${NC}"
echo -e "${BOLD}${CYAN}  ╚═══════════════════════════════════════════╝${NC}"
echo ""

# --- Pre-checks ---
step "Pre-checks"

if ! curl -sf "$API/health" > /dev/null 2>&1; then
  error "Backend not running at $API. Start with: ./start.sh"
fi
info "Backend connected"

if ! command -v jq >/dev/null 2>&1; then
  warn "jq not found. Output will not be formatted."
  JQ="cat"
else
  JQ="jq"
  info "jq found"
fi

if command -v ollama >/dev/null 2>&1; then
  info "Ollama found"
else
  warn "Ollama not in PATH. Modelfile will be saved but model won't be auto-created."
fi

echo ""
echo "  Base model:   $BASE_MODEL"
echo "  Output name:  $MODEL_NAME:latest"
echo "  Personality:  $PERSONALITY"
echo "  Session:      $SESSION_ID"
echo ""

# --- Step 1: Configure Personality ---
step "Step 1: Configure Personality ($PERSONALITY)"

$PY -c "
import json, urllib.request

presets = {
    'balanced':  {'openness': 0.6, 'conscientiousness': 0.6, 'extraversion': 0.5, 'agreeableness': 0.6, 'neuroticism': 0.4, 'emotional_granularity': 0.6, 'emotional_reactivity': 0.5, 'emotional_recovery': 0.5},
    'sensitive': {'openness': 0.7, 'conscientiousness': 0.4, 'extraversion': 0.4, 'agreeableness': 0.7, 'neuroticism': 0.8, 'emotional_granularity': 0.8, 'emotional_reactivity': 0.8, 'emotional_recovery': 0.3},
    'resilient': {'openness': 0.5, 'conscientiousness': 0.8, 'extraversion': 0.5, 'agreeableness': 0.5, 'neuroticism': 0.15, 'emotional_granularity': 0.5, 'emotional_reactivity': 0.3, 'emotional_recovery': 0.9},
    'creative':  {'openness': 0.95, 'conscientiousness': 0.4, 'extraversion': 0.6, 'agreeableness': 0.5, 'neuroticism': 0.5, 'emotional_granularity': 0.9, 'emotional_reactivity': 0.6, 'emotional_recovery': 0.5},
    'empathic':  {'openness': 0.6, 'conscientiousness': 0.5, 'extraversion': 0.6, 'agreeableness': 0.95, 'neuroticism': 0.5, 'emotional_granularity': 0.7, 'emotional_reactivity': 0.6, 'emotional_recovery': 0.5},
}

preset_name = '$PERSONALITY'
if preset_name not in presets:
    print(f'  \033[0;31mUnknown preset: {preset_name}. Available: {list(presets.keys())}\033[0m')
    import sys; sys.exit(1)

data = json.dumps(presets[preset_name]).encode()
req = urllib.request.Request('$API/personality/$SESSION_ID', data=data, headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=10) as resp:
    result = json.loads(resp.read())
    p = result['personality']
    d = result['derived']
    print(f'  O={p[\"openness\"]:.1f} C={p[\"conscientiousness\"]:.1f} E={p[\"extraversion\"]:.1f} A={p[\"agreeableness\"]:.1f} N={p[\"neuroticism\"]:.1f}')
    print(f'  variability={d[\"variability\"]:.3f}  regulation={d[\"regulation_capacity_base\"]:.3f}  empathy={d[\"empathy_weight\"]:.3f}  inertia={d[\"inertia_base\"]:.3f}')
" 2>&1

info "Personality configured"

# --- Step 2: Reset + Warmup ---
step "Step 2: Reset calibration + Warmup LLM"
curl -sf -X DELETE "$API/calibration/reset/$SESSION_ID" | $JQ
info "Calibration reset"

# Enable lite mode for calibration (keyword appraisal, no embeddings = faster)
curl -sf -X POST "$API/lite-mode/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}' > /dev/null 2>&1 || true
curl -sf -X POST "$API/lite-mode/__warmup__" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}' > /dev/null 2>&1 || true
info "Lite mode enabled for calibration"

echo -e "  ${DIM}Warming up LLM (first load can take 30-60s)...${NC}"
$PY -c "
import json, urllib.request
data = json.dumps({'message': 'hello', 'session_id': '__warmup__'}).encode()
req = urllib.request.Request('$API/chat', data=data, headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()
    print('  \033[0;32m[OK]\033[0m Model loaded and ready')
except Exception as e:
    print(f'  \033[1;33m[!!]\033[0m Warmup slow, continuing...')
" 2>&1
curl -sf -X POST "$API/reset/__warmup__" > /dev/null 2>&1 || true

# --- Step 3: Send scenarios ---
step "Step 3: Sending calibration scenarios"

SCENARIOS_FILE="$(dirname "$0")/calibration_scenarios.json"

if [ ! -f "$SCENARIOS_FILE" ]; then
  error "Scenarios file not found: $SCENARIOS_FILE"
fi

TOTAL=$($PY -c "import json; print(len(json.load(open('$SCENARIOS_FILE'))['scenarios']))" 2>/dev/null || echo "?")
echo "  Total scenarios: $TOTAL"
echo ""

$PY -c "
import json, urllib.request, time

with open('$SCENARIOS_FILE') as f:
    scenarios = json.load(f)['scenarios']

total = len(scenarios)
matches = 0
errors = 0

for i, s in enumerate(scenarios, 1):
    data = json.dumps(s).encode()
    req = urllib.request.Request(
        '$API/calibration/scenario?session_id=$SESSION_ID',
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
                match = result.get('emotion_match', False)
                if match:
                    matches += 1
                sys_emo = result.get('system_emotion', '?')
                exp_emo = s['expected_emotion']
                vd = result.get('valence_delta', 0)
                mark = '\033[0;32m+\033[0m' if match else '\033[0;31m-\033[0m'
                print(f'  [{i:2d}/{total}] {mark} {exp_emo:16s} -> {sys_emo:16s}  dv={vd:+.2f}')
                break
        except Exception as e:
            if attempt < 2:
                print(f'  [{i:2d}/{total}] \033[1;33mRETRY\033[0m {s[\"expected_emotion\"]}: {e}')
                time.sleep(3 * (attempt + 1))
            else:
                errors += 1
                print(f'  [{i:2d}/{total}] \033[0;31mERR\033[0m {s[\"expected_emotion\"]}: {e}')
    time.sleep(0.5)

print(f'\n  Matches: {matches}/{total} ({matches/total*100:.0f}%)')
if errors:
    print(f'  Errors: {errors}/{total}')
" 2>&1

info "All scenarios submitted"

# --- Step 4: Apply calibration ---
step "Step 4: Apply calibration profile"
PROFILE=$(curl -sf -X POST "$API/calibration/apply?session_id=$SESSION_ID")
echo "$PROFILE" | $JQ
info "Calibration applied"

# --- Step 5: Export to Ollama ---
step "Step 5: Export model"

EXPORT_BODY=$($PY -c "
import json
print(json.dumps({
    'base_model': '$BASE_MODEL',
    'model_name': '$MODEL_NAME',
    'temperature': 0.7,
    'num_ctx': 8192,
}))
")

RESULT=$(curl -sf -X POST "$API/models/export?session_id=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d "$EXPORT_BODY")

echo "$RESULT" | $JQ

STATUS=$(echo "$RESULT" | $PY -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

echo ""
if [ "$STATUS" = "model_created" ]; then
  info "Model '$MODEL_NAME:latest' created successfully!"
  echo ""
  echo -e "  ${GREEN}To use your model:${NC}"
  echo "    ollama run $MODEL_NAME"
  echo ""
  echo -e "  ${GREEN}To use in Pathos Engine:${NC}"
  echo "    curl -X POST $API/models/switch -H 'Content-Type: application/json' -d '{\"provider\":\"ollama\",\"model\":\"$MODEL_NAME\"}'"
elif [ "$STATUS" = "modelfile_saved" ]; then
  warn "Modelfile saved but model wasn't auto-created."
  echo -e "  ${YELLOW}Create manually:${NC} ollama create $MODEL_NAME -f Modelfile"
else
  error "Export failed. Check backend logs."
fi

echo ""
step "Done!"
echo ""
echo -e "  Your model '${BOLD}$MODEL_NAME${NC}' is based on '${BASE_MODEL}'"
echo -e "  with personality '${CYAN}${PERSONALITY}${NC}' and Pathos Engine v2:"
echo ""
echo "    - 10 advanced emotional systems"
echo "    - 19 emotion definitions with simultaneous activations"
echo "    - Calibration profile from $TOTAL human scenarios"
echo "    - Personality-driven dynamics & regulation"
echo ""
echo "  Use it anywhere Ollama models are supported."
echo ""
