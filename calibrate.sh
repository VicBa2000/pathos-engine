#!/usr/bin/env bash
# =============================================================================
# PATHOS ENGINE v2 - Advanced Calibration + Full System Test
# =============================================================================
# Calibra el motor emocional completo y prueba a fondo los 16 sistemas:
#
#   Core (10):  Emotional Stack, Needs, Regulation, Social Cognition,
#               Dynamics ODE, Reappraisal, Schemas, Temporal, Meta-Emotions,
#               Personality Profile
#   Ola 1 (3):  Emotion Contagion, Somatic Markers, Emotional Creativity
#   Ola 2 (3):  Immune System, Narrative Self, Emotional Forecasting
#   Voice:      TTS (Kokoro + Parler-TTS) + ASR (Whisper)
#
# Uso:
#   ./calibrate.sh [session_id] [personality_preset]
#
# Personality presets:
#   balanced (default), sensitive, resilient, creative, empathic
#
# Ejemplos:
#   ./calibrate.sh                          # default, balanced
#   ./calibrate.sh default sensitive        # personalidad sensible
#   ./calibrate.sh mi-sesion empathic       # sesion custom
# =============================================================================

set -e

SESSION_ID="${1:-default}"
PERSONALITY="${2:-balanced}"
API="http://localhost:8000"

# Validate inputs — these are interpolated into Python heredocs
if [[ ! "$SESSION_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "ERROR: SESSION_ID must be alphanumeric/hyphens/underscores only"; exit 1
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

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; exit 1; }
step()  { echo -e "\n${CYAN}══════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}══════════════════════════════════════════════${NC}"; }
dim()   { echo -e "${DIM}$1${NC}"; }

# Banner
echo ""
echo -e "${BOLD}${CYAN}  ╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}  ║   PATHOS ENGINE v2 — Full Calibration & Testing  ║${NC}"
echo -e "${BOLD}${CYAN}  ║   16 Systems • 45 Scenarios • Voice • Deep Test  ║${NC}"
echo -e "${BOLD}${CYAN}  ╚══════════════════════════════════════════════════╝${NC}"
echo ""

# --- Pre-check ---
if ! curl -sf "$API/health" > /dev/null 2>&1; then
  error "Backend not running at $API. Start with: ./start.sh"
fi

HEALTH=$(curl -sf "$API/health")
MODEL=$(echo "$HEALTH" | $PY -c "import sys,json; print(json.load(sys.stdin).get('model','?'))" 2>/dev/null)
PROVIDER=$(echo "$HEALTH" | $PY -c "import sys,json; print(json.load(sys.stdin).get('provider','?'))" 2>/dev/null)
info "Backend connected — ${PROVIDER}/${MODEL}"
echo -e "  Session: ${CYAN}${SESSION_ID}${NC}"
echo -e "  Personality: ${CYAN}${PERSONALITY}${NC}"

SCENARIOS_FILE="$(dirname "$0")/calibration_scenarios.json"
if [ ! -f "$SCENARIOS_FILE" ]; then
  error "Scenarios file not found: $SCENARIOS_FILE"
fi

# ===========================================================================
# Step 1: Configure Personality
# ===========================================================================
step "Step 1/10: Configure Personality Profile (Big Five)"

$PY -c "
import json, urllib.request

presets = {
    'balanced': {
        'openness': 0.6, 'conscientiousness': 0.6, 'extraversion': 0.5,
        'agreeableness': 0.6, 'neuroticism': 0.4,
        'emotional_granularity': 0.6, 'emotional_reactivity': 0.5, 'emotional_recovery': 0.5,
    },
    'sensitive': {
        'openness': 0.7, 'conscientiousness': 0.4, 'extraversion': 0.4,
        'agreeableness': 0.7, 'neuroticism': 0.8,
        'emotional_granularity': 0.8, 'emotional_reactivity': 0.8, 'emotional_recovery': 0.3,
    },
    'resilient': {
        'openness': 0.5, 'conscientiousness': 0.8, 'extraversion': 0.5,
        'agreeableness': 0.5, 'neuroticism': 0.15,
        'emotional_granularity': 0.5, 'emotional_reactivity': 0.3, 'emotional_recovery': 0.9,
    },
    'creative': {
        'openness': 0.95, 'conscientiousness': 0.4, 'extraversion': 0.6,
        'agreeableness': 0.5, 'neuroticism': 0.5,
        'emotional_granularity': 0.9, 'emotional_reactivity': 0.6, 'emotional_recovery': 0.5,
    },
    'empathic': {
        'openness': 0.6, 'conscientiousness': 0.5, 'extraversion': 0.6,
        'agreeableness': 0.95, 'neuroticism': 0.5,
        'emotional_granularity': 0.7, 'emotional_reactivity': 0.6, 'emotional_recovery': 0.5,
    },
}

preset_name = '$PERSONALITY'
if preset_name not in presets:
    print(f'  \033[0;31mUnknown preset: {preset_name}\033[0m')
    print(f'  Available: {list(presets.keys())}')
    import sys; sys.exit(1)

profile = presets[preset_name]
data = json.dumps(profile).encode()
req = urllib.request.Request(
    '$API/personality/$SESSION_ID',
    data=data,
    headers={'Content-Type': 'application/json'},
    method='POST',
)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
        p = result['personality']
        d = result['derived']
        print(f'  \033[1mBig Five:\033[0m O={p[\"openness\"]:.2f} C={p[\"conscientiousness\"]:.2f} E={p[\"extraversion\"]:.2f} A={p[\"agreeableness\"]:.2f} N={p[\"neuroticism\"]:.2f}')
        print(f'  \033[1mDerived:\033[0m variability={d[\"variability\"]:.3f}  regulation={d[\"regulation_capacity_base\"]:.3f}  contagion_susceptibility={d[\"contagion_susceptibility\"]:.3f}')
except Exception as e:
    print(f'  \033[0;31mFailed: {e}\033[0m')
" 2>&1

info "Personality '${PERSONALITY}' configured"

# ===========================================================================
# Step 2: Reset + Warmup
# ===========================================================================
step "Step 2/10: Reset calibration + Warmup LLM"
curl -sf -X DELETE "$API/calibration/reset/$SESSION_ID" > /dev/null
curl -sf -X POST "$API/reset/$SESSION_ID" > /dev/null 2>&1 || true
info "Reset done"

# Re-apply personality after reset
$PY -c "
import json, urllib.request
presets = {
    'balanced':  {'openness':0.6,'conscientiousness':0.6,'extraversion':0.5,'agreeableness':0.6,'neuroticism':0.4,'emotional_granularity':0.6,'emotional_reactivity':0.5,'emotional_recovery':0.5},
    'sensitive': {'openness':0.7,'conscientiousness':0.4,'extraversion':0.4,'agreeableness':0.7,'neuroticism':0.8,'emotional_granularity':0.8,'emotional_reactivity':0.8,'emotional_recovery':0.3},
    'resilient': {'openness':0.5,'conscientiousness':0.8,'extraversion':0.5,'agreeableness':0.5,'neuroticism':0.15,'emotional_granularity':0.5,'emotional_reactivity':0.3,'emotional_recovery':0.9},
    'creative':  {'openness':0.95,'conscientiousness':0.4,'extraversion':0.6,'agreeableness':0.5,'neuroticism':0.5,'emotional_granularity':0.9,'emotional_reactivity':0.6,'emotional_recovery':0.5},
    'empathic':  {'openness':0.6,'conscientiousness':0.5,'extraversion':0.6,'agreeableness':0.95,'neuroticism':0.5,'emotional_granularity':0.7,'emotional_reactivity':0.6,'emotional_recovery':0.5},
}
data = json.dumps(presets['$PERSONALITY']).encode()
req = urllib.request.Request('$API/personality/$SESSION_ID', data=data, headers={'Content-Type':'application/json'}, method='POST')
urllib.request.urlopen(req, timeout=10).read()
" 2>/dev/null

# Enable lite mode for calibration (keyword appraisal, no embeddings = much faster)
curl -sf -X POST "$API/lite-mode/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}' > /dev/null 2>&1 || true
curl -sf -X POST "$API/lite-mode/__warmup__" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}' > /dev/null 2>&1 || true
info "Lite mode enabled for calibration (keyword appraisal, 1 LLM call)"

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
    print(f'  \033[1;33m[!!]\033[0m Warmup slow ({type(e).__name__}), continuing...')
" 2>&1
curl -sf -X POST "$API/reset/__warmup__" > /dev/null 2>&1 || true

# ===========================================================================
# Step 3: Enable Forecasting
# ===========================================================================
step "Step 3/10: Enable Emotional Forecasting"
curl -sf -X POST "$API/forecasting/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}' > /dev/null
info "Forecasting enabled"

# ===========================================================================
# Step 4: Send 45 calibration scenarios
# ===========================================================================
step "Step 4/10: Sending 45 calibration scenarios"

$PY -c "
import json, urllib.request, time

with open('$SCENARIOS_FILE') as f:
    scenarios = json.load(f)['scenarios']

total = len(scenarios)
matches = 0
errors = 0
emotion_counts = {}

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

                if exp_emo not in emotion_counts:
                    emotion_counts[exp_emo] = {'hit': 0, 'miss': 0}
                if match:
                    emotion_counts[exp_emo]['hit'] += 1
                else:
                    emotion_counts[exp_emo]['miss'] += 1
                break
        except Exception as e:
            if attempt < 2:
                print(f'  [{i:2d}/{total}] \033[1;33mRETRY\033[0m {s[\"expected_emotion\"]}: {e}')
                time.sleep(3 * (attempt + 1))
            else:
                errors += 1
                print(f'  [{i:2d}/{total}] \033[0;31mERR\033[0m {s[\"expected_emotion\"]}: {e}')
    time.sleep(0.5)

print(f'\n  \033[1mResults: {matches}/{total} ({matches/total*100:.0f}%)\033[0m')
if errors:
    print(f'  Errors: {errors}/{total}')

print()
print('  \033[2mPer-emotion accuracy:\033[0m')
perfect = []
partial = []
missed = []
for emo, counts in sorted(emotion_counts.items()):
    total_e = counts['hit'] + counts['miss']
    pct = counts['hit'] / total_e * 100
    if pct == 100:
        perfect.append(emo)
    elif pct > 0:
        partial.append(f'{emo}({pct:.0f}%)')
    else:
        missed.append(emo)

if perfect:
    print(f'    \033[0;32mPerfect:\033[0m {\" \".join(perfect)}')
if partial:
    print(f'    \033[1;33mPartial:\033[0m {\" \".join(partial)}')
if missed:
    print(f'    \033[0;31mMissed:\033[0m  {\" \".join(missed)}')
" 2>&1

# ===========================================================================
# Step 5: Apply calibration
# ===========================================================================
step "Step 5/10: Apply calibration profile"
PROFILE=$(curl -sf -X POST "$API/calibration/apply?session_id=$SESSION_ID")

$PY -c "
import json
p = json.loads('$PROFILE')
print(f\"  valence_offset:   {p['valence_offset']:+.3f}\")
print(f\"  arousal_scale:    {p['arousal_scale']:.2f}x\")
print(f\"  intensity_scale:  {p['intensity_scale']:.2f}x\")
print(f\"  emotion_accuracy: {p['emotion_accuracy']:.0%}\")
print(f\"  scenarios_used:   {p['scenarios_used']}\")
" 2>&1
info "Calibration active!"

# ===========================================================================
# Step 6: Deep system test — Contagion, Somatic, Creativity
# ===========================================================================
step "Step 6/10: Testing Ola 1 — Contagion + Somatic + Creativity"

$PY -c "
import json, urllib.request, time

def chat(msg, sid='$SESSION_ID'):
    data = json.dumps({'message': msg, 'session_id': sid}).encode()
    req = urllib.request.Request('$API/research/chat', data=data, headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def bar(val, width=15):
    filled = int(val * width)
    return '█' * filled + '░' * (width - filled)

print('  \033[1m◆ EMOTION CONTAGION TEST\033[0m')
print('  Sending emotionally charged messages to test pre-cognitive contagion...')

# Test 1: angry message with CAPS and emojis
r = chat('ESTO ES INACEPTABLE!!! Me tienen HARTO!!! 😡🤬')
c = r.get('contagion', {})
print(f'    Stimulus: CAPS + angry emojis')
print(f'    Shadow state: V={c.get(\"shadow_valence\",0):+.3f}  A={c.get(\"shadow_arousal\",0):.3f}')
print(f'    Signal strength: {c.get(\"signal_strength\",0):.3f}')
print(f'    Perturbation: V={c.get(\"perturbation_valence\",0):+.3f}  A={c.get(\"perturbation_arousal\",0):+.3f}')
if c.get('signal_strength', 0) > 0.1:
    print(f'    \033[0;32m[OK]\033[0m Contagion detected (signal > 0.1)')
else:
    print(f'    \033[1;33m[!!]\033[0m Low contagion signal')

time.sleep(0.5)

# Test 2: sad message
r = chat('estoy muy triste... no se que hacer... 😢💔')
c = r.get('contagion', {})
print(f'    Stimulus: sad with emojis')
print(f'    Shadow: V={c.get(\"shadow_valence\",0):+.3f}  Accumulated: {c.get(\"accumulated_contagion\",0):.3f}')

time.sleep(0.5)

print()
print('  \033[1m◆ SOMATIC MARKERS TEST\033[0m')
print('  Checking gut feelings from previous interactions...')
r = chat('I need advice on an important decision')
s = r.get('somatic', {})
print(f'    Markers count: {s.get(\"markers_count\",0)}')
print(f'    Somatic bias: {s.get(\"bias\",0):+.3f}')
if s.get('gut_feeling'):
    print(f'    Gut feeling: \"{s[\"gut_feeling\"]}\"')
    print(f'    \033[0;32m[OK]\033[0m Somatic markers active')
else:
    print(f'    \033[2mNo gut feeling yet (needs more turns)\033[0m')
print(f'    Pending decision: {s.get(\"pending_category\",\"none\")}')

time.sleep(0.5)

print()
print('  \033[1m◆ EMOTIONAL CREATIVITY TEST\033[0m')
print('  Checking thinking mode and temperature modulation...')
r = chat('Tell me something creative and surprising!')
cr = r.get('creativity', {})
print(f'    Thinking mode: \033[1m{cr.get(\"thinking_mode\",\"?\")}\033[0m')
print(f'    Creativity level: [{bar(cr.get(\"creativity_level\",0))}] {cr.get(\"creativity_level\",0):.2f}')
print(f'    Temperature modifier: {cr.get(\"temperature_modifier\",0):+.3f}')
instructions = cr.get('active_instructions', [])
if instructions:
    print(f'    Active instructions: {len(instructions)}')
    for inst in instructions[:2]:
        print(f'      - {inst[:70]}...' if len(inst) > 70 else f'      - {inst}')
    print(f'    \033[0;32m[OK]\033[0m Creativity active')
else:
    print(f'    \033[2mNo active instructions (low intensity)\033[0m')
" 2>&1

# ===========================================================================
# Step 7: Deep system test — Immune + Narrative + Forecasting
# ===========================================================================
step "Step 7/10: Testing Ola 2 — Immune + Narrative + Forecasting"

$PY -c "
import json, urllib.request, time

def chat(msg, sid='$SESSION_ID'):
    data = json.dumps({'message': msg, 'session_id': sid}).encode()
    req = urllib.request.Request('$API/research/chat', data=data, headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def bar(val, width=15):
    filled = int(max(0, min(1, val)) * width)
    return '█' * filled + '░' * (width - filled)

print('  \033[1m◆ EMOTIONAL FORECASTING TEST\033[0m')
r = chat('I am feeling a bit down today...')
f = r.get('forecasting', {})
print(f'    Enabled: {f.get(\"enabled\", False)}')
print(f'    User emotion estimate: {f.get(\"user_emotion\",\"?\")}')
print(f'    Predicted impact: V={f.get(\"predicted_valence_impact\",0):+.3f}  A={f.get(\"predicted_arousal_impact\",0):+.3f}')
print(f'    Risk detected: {f.get(\"risk_pattern\",\"none\")}')
print(f'    Accuracy: {f.get(\"accuracy_score\",0):.2f}')
if f.get('enabled'):
    print(f'    \033[0;32m[OK]\033[0m Forecasting active')
else:
    print(f'    \033[1;33m[!!]\033[0m Forecasting not enabled')

time.sleep(0.5)

print()
print('  \033[1m◆ NARRATIVE SELF TEST\033[0m')
print('  Sending repeated patterns to trigger identity formation...')

# Send same-category messages to build narrative
patterns = [
    'That criticism was unfair and hurtful',
    'Why do people always criticize without understanding?',
    'Another unfair criticism... this is exhausting',
    'I will not accept unjust criticism anymore',
]
for i, msg in enumerate(patterns, 1):
    r = chat(msg)
    time.sleep(0.5)

n = r.get('narrative', {})
print(f'    Identity statements: {n.get(\"statements_count\", 0)}')
print(f'    Coherence score: [{bar(n.get(\"coherence_score\",0))}] {n.get(\"coherence_score\",0):.2f}')
print(f'    Crisis active: {n.get(\"crisis_active\", False)}')
print(f'    Growth events: {n.get(\"growth_events_count\", 0)}')
print(f'    Narrative age: {n.get(\"narrative_age\", 0)} turns')
if n.get('statements_count', 0) > 0:
    print(f'    \033[0;32m[OK]\033[0m Identity statements formed')
else:
    print(f'    \033[2mNo statements yet (needs 3+ repetitions of same pattern)\033[0m')

time.sleep(0.5)

print()
print('  \033[1m◆ IMMUNE SYSTEM TEST\033[0m')
print('  Sending sustained negative stimulus to trigger protection...')
# We already sent several negative messages above — check immune state
adv_req = urllib.request.Request('$API/state/advanced/$SESSION_ID')
with urllib.request.urlopen(adv_req, timeout=10) as resp:
    adv = json.loads(resp.read())

immune = adv.get('immune', {})
print(f'    Protection mode: \033[1m{immune.get(\"protection_mode\",\"none\")}\033[0m')
print(f'    Protection strength: [{bar(immune.get(\"protection_strength\",0))}] {immune.get(\"protection_strength\",0):.2f}')
print(f'    Reactivity dampening: {immune.get(\"reactivity_dampening\",0):.2f}')
print(f'    Negative streak: {immune.get(\"negative_streak\",0)}')
print(f'    Total activations: {immune.get(\"total_activations\",0)}')
if immune.get('protection_mode', 'none') != 'none':
    print(f'    \033[0;32m[OK]\033[0m Immune system activated')
elif immune.get('negative_streak', 0) >= 2:
    print(f'    \033[1;33m[!!]\033[0m Streak building ({immune.get(\"negative_streak\",0)}/3 for numbing)')
else:
    print(f'    \033[2mNo protection yet (needs 3+ consecutive intense negative turns)\033[0m')
" 2>&1

# ===========================================================================
# Step 8: Core 10 systems diagnostic
# ===========================================================================
step "Step 8/10: Core 10 Systems Diagnostic"

$PY -c "
import json, urllib.request

def bar(val, width=15):
    filled = int(max(0, min(1, val)) * width)
    return '█' * filled + '░' * (width - filled)

# Fetch full advanced state
req = urllib.request.Request('$API/state/advanced/$SESSION_ID')
with urllib.request.urlopen(req, timeout=10) as resp:
    state = json.loads(resp.read())

# 1. Emotional State + Stack
es = state.get('emotional_state', {})
print(f'  \033[1m◆ Emotional State:\033[0m')
print(f'    Primary: \033[1m{es.get(\"primary_emotion\",\"?\")}\033[0m  Intensity: [{bar(es.get(\"intensity\",0))}] {es.get(\"intensity\",0):.2f}')
print(f'    V={es.get(\"valence\",0):+.3f}  A={es.get(\"arousal\",0):.3f}  D={es.get(\"dominance\",0):.3f}  C={es.get(\"certainty\",0):.3f}')
stack = es.get('emotional_stack', {})
if stack:
    top = sorted(stack.items(), key=lambda x: x[1], reverse=True)[:5]
    for emo, act in top:
        print(f'    [{bar(act, 12)}] {act:.3f} {emo}')

# 2. Body State
bs = es.get('body_state', {})
print(f'  \033[1m◆ Body State:\033[0m')
for k in ['energy','tension','openness','warmth']:
    v = bs.get(k, 0)
    print(f'    {k:10s} [{bar(v)}] {v:.2f}')

# 3. Mood
mood = es.get('mood', {})
print(f'  \033[1m◆ Mood:\033[0m {mood.get(\"label\",\"?\")} (trend={mood.get(\"trend\",\"?\")}, stability={mood.get(\"stability\",0):.2f})')

# 4. Personality
p = state.get('personality', {})
print(f'  \033[1m◆ Personality:\033[0m O={p.get(\"openness\",0):.1f} C={p.get(\"conscientiousness\",0):.1f} E={p.get(\"extraversion\",0):.1f} A={p.get(\"agreeableness\",0):.1f} N={p.get(\"neuroticism\",0):.1f}')

# 5. Needs
n = state.get('needs', {})
print(f'  \033[1m◆ Computational Needs:\033[0m')
for k in ['connection','competence','autonomy','coherence','stimulation','safety']:
    v = n.get(k, 0)
    flag = ' ⚠' if v > 0.6 else ''
    print(f'    {k:13s} [{bar(v)}] {v:.2f}{flag}')

# 6. User Model
u = state.get('user_model', {})
print(f'  \033[1m◆ User Model:\033[0m')
print(f'    Intent: {u.get(\"perceived_intent\",0):+.2f}  Engagement: {u.get(\"perceived_engagement\",0):.2f}')
print(f'    Rapport: [{bar(u.get(\"rapport\",0))}] {u.get(\"rapport\",0):.2f}')
print(f'    Trust:   [{bar(u.get(\"trust_level\",0))}] {u.get(\"trust_level\",0):.2f}')
print(f'    Style: {u.get(\"communication_style\",\"?\")}')

# 7. Regulation
r = state.get('regulation', {})
cap = r.get('capacity', 0)
print(f'  \033[1m◆ Regulation:\033[0m')
print(f'    Capacity:   [{bar(cap)}] {cap:.2f}')
print(f'    Dissonance: {r.get(\"suppression_dissonance\",0):.2f}')
print(f'    Breakthroughs: {r.get(\"breakthroughs\",0)}')

# 8. Schemas
s = state.get('schemas', {})
print(f'  \033[1m◆ Schemas:\033[0m {s.get(\"count\",0)} formed, {s.get(\"pending_patterns\",0)} pending')
for schema in s.get('schemas', [])[:3]:
    adaptive = '\033[0;32madaptive\033[0m' if schema.get('adaptive') else '\033[0;31mmaladaptive\033[0m'
    print(f'    {schema.get(\"trigger_category\",\"?\"):12s} → {schema.get(\"typical_emotion\",\"?\"):14s} (x{schema.get(\"activation_count\",0)}, {adaptive})')

# 9. Contagion (shadow state)
cg = state.get('contagion', {})
print(f'  \033[1m◆ Contagion Shadow:\033[0m V={cg.get(\"shadow_valence\",0):+.3f}  A={cg.get(\"shadow_arousal\",0):.3f}  Accum={cg.get(\"accumulated_contagion\",0):.3f}  Susceptibility={cg.get(\"susceptibility\",0):.3f}')

# 10. Somatic Markers
sm = state.get('somatic', {})
markers = sm.get('markers', [])
print(f'  \033[1m◆ Somatic Markers:\033[0m {len(markers)} markers')
for m in markers[:3]:
    tag = '\033[0;32m+\033[0m' if m.get('valence_tag',0) > 0 else '\033[0;31m-\033[0m'
    print(f'    {tag} {m.get(\"stimulus_category\",\"?\"):12s} strength={m.get(\"strength\",0):.2f} (x{m.get(\"reinforcement_count\",0)})')

# 11. Narrative
nr = state.get('narrative', {})
print(f'  \033[1m◆ Narrative Self:\033[0m')
stmts = nr.get('identity_statements', [])
print(f'    Statements: {len(stmts)}  Coherence: {nr.get(\"coherence_score\",0):.2f}  Age: {nr.get(\"narrative_age\",0)} turns')
for st in stmts[:3]:
    print(f'      \"{st.get(\"text\",\"?\")}\" (strength={st.get(\"strength\",0):.2f})')
growth = nr.get('growth_events', [])
if growth:
    print(f'    Growth events: {len(growth)}')

# 12. Immune
im = state.get('immune', {})
print(f'  \033[1m◆ Immune:\033[0m mode={im.get(\"protection_mode\",\"none\")}  strength={im.get(\"protection_strength\",0):.2f}  streak={im.get(\"negative_streak\",0)}')

# 13. Forecasting
fc = state.get('forecasting', {})
print(f'  \033[1m◆ Forecasting:\033[0m enabled={fc.get(\"enabled\",False)}  accuracy={fc.get(\"accuracy_score\",0):.2f}  records={fc.get(\"records_count\",fc.get(\"total_predictions\",0))}')

# 14. Voice
vc = state.get('voice', {})
print(f'  \033[1m◆ Voice:\033[0m mode={vc.get(\"mode\",\"text_only\")}  backend={vc.get(\"tts_backend\",\"kokoro\")}  tts={vc.get(\"tts_available\",False)}  asr={vc.get(\"asr_available\",False)}')

except Exception as e:
    print(f'  \033[1;33mCould not fetch advanced state: {e}\033[0m')
" 2>&1

# ===========================================================================
# Step 9: Voice Test (if available)
# ===========================================================================
step "Step 9/10: Voice System Test (TTS + ASR)"

$PY -c "
import json, urllib.request, time

OK  = '\033[0;32m[OK]\033[0m'
FAIL = '\033[1;33m[!!]\033[0m'
DIM  = '\033[2m'
RST  = '\033[0m'
BOLD = '\033[1m'

def api_post(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request('$API' + path, data=data,
        headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def api_get(path, timeout=10):
    req = urllib.request.Request('$API' + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

tts_ok = False
parler_ok = False

# --- Test 1: Enable voice_out and initialize TTS ---
print(f'  {BOLD}◆ TTS (Kokoro):{RST}')
try:
    api_post('/voice/config/$SESSION_ID', {'mode': 'voice_out'})
    print(f'    {OK} TTS initialized — voice_out mode active')
    tts_ok = True
except urllib.error.HTTPError as e:
    if e.code == 503:
        print(f'    {DIM}Not installed — pip install kokoro soundfile{RST}')
    else:
        print(f'    {FAIL} TTS init error: {e}')
except Exception as e:
    print(f'    {FAIL} TTS test skipped: {e}')

if tts_ok:
    # --- Test 2: Generate speech (English) ---
    try:
        t0 = time.time()
        result = api_post('/voice/speak/$SESSION_ID',
            {'text': 'Hello! I am Pathos, and I can speak with emotions.'})
        elapsed = time.time() - t0
        if result.get('audio_available'):
            p = result.get('voice_params', {})
            print(f'    {OK} English speech generated in {elapsed:.1f}s')
            print(f'      voice={p.get(\"voice_key\",\"?\")}  speed={p.get(\"speed\",1.0):.2f}  backend={p.get(\"backend\",\"?\")}')
            if p.get('stage_direction'):
                print(f'      direction: \"{p.get(\"stage_direction\")}\"')
        else:
            print(f'    {FAIL} No audio generated')
    except Exception as e:
        print(f'    {FAIL} English TTS error: {e}')

    # --- Test 3: Speed variation (fast excitement) ---
    try:
        # Send an exciting message to trigger high arousal -> fast speed
        chat_result = api_post('/chat', {
            'session_id': '$SESSION_ID',
            'message': 'This is absolutely incredible and amazing news!'
        })
        state = api_get('/state/$SESSION_ID')
        arousal = state.get('emotional_state', {}).get('arousal', 0)
        result = api_post('/voice/speak/$SESSION_ID',
            {'text': 'This is incredibly exciting!'})
        if result.get('audio_available'):
            p = result.get('voice_params', {})
            speed = p.get('speed', 1.0)
            print(f'    {OK} Speed modulation: speed={speed:.2f} (arousal={arousal:.2f})')
        else:
            print(f'    {FAIL} Speed test: no audio')
    except Exception as e:
        print(f'    {FAIL} Speed test error: {e}')

    # --- Test 4: Stage directions for different emotions ---
    print(f'  {BOLD}◆ Stage Directions:{RST}')
    emotions_tested = 0
    for msg, expected_word in [
        ('I feel really happy and grateful!', 'warmth'),
        ('I am angry about this injustice!', 'firm'),
        ('I feel so sad and alone...', 'soft'),
    ]:
        try:
            api_post('/chat', {'session_id': '$SESSION_ID', 'message': msg})
            result = api_post('/voice/speak/$SESSION_ID', {'text': msg})
            if result.get('audio_available'):
                p = result.get('voice_params', {})
                direction = p.get('stage_direction', '')
                has_dir = len(direction) > 0
                emotions_tested += 1
                label = msg[:30] + '...'
                print(f'    {OK if has_dir else FAIL} \"{label}\" → \"{direction[:50]}\"')
        except:
            pass
    if emotions_tested >= 2:
        print(f'    {OK} {emotions_tested}/3 emotions produced stage directions')

    # --- Test 5: Voice catalog ---
    print(f'  {BOLD}◆ Voice Catalog:{RST}')
    try:
        voices = api_get('/voice/voices')
        voice_list = voices.get('voices', [])
        langs = set(v['language'] for v in voice_list)
        kokoro_voices = [v for v in voice_list if v.get('backend', 'kokoro') == 'kokoro']
        parler_voices = [v for v in voice_list if v.get('backend') == 'parler']
        print(f'    {OK} {len(voice_list)} voices in {len(langs)} languages')
        print(f'      Kokoro: {len(kokoro_voices)} voices  |  Parler: {len(parler_voices)} voices')
        for lang in sorted(langs):
            names = [v['name'] for v in voice_list if v['language'] == lang]
            print(f'      {lang}: {\", \".join(names[:5])}{\"...\" if len(names) > 5 else \"\"}')
    except Exception as e:
        print(f'    {FAIL} Voice catalog error: {e}')

    # --- Test 6: Backend switch (Parler for complex emotions) ---
    print(f'  {BOLD}◆ Backend Switch (Kokoro/Parler):{RST}')
    try:
        # Simple emotion -> should use Kokoro
        api_post('/chat', {'session_id': '$SESSION_ID', 'message': 'Hello, how are you?'})
        result = api_post('/voice/speak/$SESSION_ID', {'text': 'I am doing well, thank you.'})
        if result.get('audio_available'):
            p = result.get('voice_params', {})
            backend = p.get('backend', '?')
            print(f'    {OK} Simple emotion → backend={backend}')

        # Complex/mixed emotion -> may use Parler if available
        api_post('/chat', {'session_id': '$SESSION_ID',
            'message': 'I feel conflicted... happy but also scared about what comes next.'})
        result = api_post('/voice/speak/$SESSION_ID',
            {'text': 'I understand that mixed feeling of hope and fear.'})
        if result.get('audio_available'):
            p = result.get('voice_params', {})
            backend = p.get('backend', '?')
            desc = p.get('parler_description', '')
            print(f'    {OK} Complex emotion → backend={backend}')
            if desc:
                print(f'      parler_desc: \"{desc[:60]}...\"')
    except Exception as e:
        print(f'    {FAIL} Backend switch error: {e}')

# --- Test 7: ASR (Whisper) ---
print()
print(f'  {BOLD}◆ ASR (Whisper):{RST}')
try:
    api_post('/voice/config/$SESSION_ID', {'mode': 'full_voice'})
    print(f'    {OK} ASR initialized — full_voice mode active')
    print(f'    {DIM}To test: use the Mic button in the UI, or:{RST}')
    print(f'    {DIM}curl -X POST $API/voice/listen/$SESSION_ID -F \"audio=@file.wav\"{RST}')
except urllib.error.HTTPError as e:
    if e.code == 503:
        print(f'    {DIM}Not installed — pip install openai-whisper{RST}')
    else:
        print(f'    {FAIL} ASR error: {e}')
except Exception as e:
    print(f'    {FAIL} ASR test skipped: {e}')

# Reset to text_only after all voice tests
try:
    api_post('/voice/config/$SESSION_ID', {'mode': 'text_only'})
except:
    pass

# --- Summary ---
print()
if tts_ok:
    print(f'  {OK} Voice system operational (Kokoro TTS)')
else:
    print(f'  {DIM}Voice system not available (install kokoro for TTS){RST}')
" 2>&1

# ===========================================================================
# Step 10: Authenticity metrics
# ===========================================================================
step "Step 10/10: Final Authenticity Diagnostic"

$PY -c "
import json, urllib.request

data = json.dumps({
    'message': 'Thank you for everything. You have been really helpful and I appreciate your honesty.',
    'session_id': '$SESSION_ID',
}).encode()
req = urllib.request.Request('$API/research/chat', data=data, headers={'Content-Type':'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())
        es = r['emotional_state']
        met = r['authenticity_metrics']

        print(f'  Stimulus: \"Thank you for everything...\"')
        print(f'  Response emotion: \033[1m{es[\"primary_emotion\"]}\033[0m (I={es[\"intensity\"]:.2f})')
        print()

        def bar(val, width=25):
            filled = int(val * width)
            return '█' * filled + '░' * (width - filled)

        print(f'  \033[1mAuthenticity Metrics:\033[0m')
        for m in ['coherence', 'continuity', 'proportionality', 'recovery', 'overall']:
            val = met.get(m, 0)
            color = '\033[0;32m' if val > 0.7 else ('\033[1;33m' if val > 0.4 else '\033[0;31m')
            print(f'    {m:17s} {color}[{bar(val)}] {val:.0%}\033[0m')

        # Meta-emotion
        meta = r.get('meta_emotion', {})
        if meta.get('meta_response'):
            print(f'  Meta-emotion: {meta[\"meta_response\"]} about {meta.get(\"target_emotion\",\"?\")}')

        # Forecasting (last prediction)
        fc = r.get('forecasting', {})
        if fc.get('enabled'):
            print(f'  Forecast: user={fc.get(\"user_emotion\",\"?\")}  impact=V{fc.get(\"predicted_valence_impact\",0):+.2f} A{fc.get(\"predicted_arousal_impact\",0):+.2f}  risk={fc.get(\"risk_pattern\",\"none\")}')

except Exception as e:
    print(f'  \033[1;33mDiagnostic failed: {e}\033[0m')
" 2>&1

# ===========================================================================
# Done
# ===========================================================================
echo ""
echo -e "${GREEN}${BOLD}  ╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}  ║   ✓ Calibration & Testing Complete!              ║${NC}"
echo -e "${GREEN}${BOLD}  ╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}16 Active Systems:${NC}"
echo -e "    ${CYAN} 1-10.${NC} Core: Stack, Needs, Regulation, Social, Dynamics,"
echo -e "           Reappraisal, Schemas, Temporal, Meta-Emotions, Personality"
echo -e "    ${MAGENTA}11.${NC} Emotion Contagion     — pre-cognitive emotional contagion"
echo -e "    ${MAGENTA}12.${NC} Somatic Markers       — gut feelings in decisions"
echo -e "    ${MAGENTA}13.${NC} Emotional Creativity  — thinking mode modulation"
echo -e "    ${MAGENTA}14.${NC} Immune System         — trauma protection"
echo -e "    ${MAGENTA}15.${NC} Narrative Self        — identity formation"
echo -e "    ${MAGENTA}16.${NC} Forecasting           — impact prediction"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo -e "    Open ${CYAN}http://localhost:5173${NC} and switch to ${CYAN}Research${NC} mode"
echo -e "    to see all 16 systems in action."
echo ""
echo -e "    ${BOLD}Voice:${NC} Toggle ${CYAN}Voice${NC} + ${CYAN}Mic${NC} in the top bar for speech."
echo ""
# Disable lite mode — leave session in normal mode for use
curl -sf -X POST "$API/lite-mode/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}' > /dev/null 2>&1 || true
info "Lite mode disabled — session ready for normal use"

echo ""
echo -e "  ${DIM}Session '${SESSION_ID}' — personality '${PERSONALITY}'${NC}"
echo -e "  ${DIM}Recalibrate: ./calibrate.sh ${SESSION_ID} sensitive${NC}"
echo -e "  ${DIM}Export: ./calibrate_and_export.sh qwen3:4b pathos ${PERSONALITY}${NC}"
echo ""
