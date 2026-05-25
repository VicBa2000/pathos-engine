"""Steering Vector Extraction Utility — Offline batch extraction.

Extracts emotional direction vectors from a model and caches them for
runtime steering. This is an OFFLINE operation — run once per model,
then the cached vectors are loaded automatically at startup.

Modes:
  - 4D (legacy): valence/arousal/dominance/certainty from contrastive pairs.
  - 171 (RESIDUUM F1.2): 171 emotion probes from stories with neutral PC
    projection (Lindsey/Sofroniew et al. 2026, Anthropic).

Usage as script:
    python -m pathos.engine.steering_extract --model qwen3:4b
    python -m pathos.engine.steering_extract --model Qwen/Qwen3-4B --device cpu
    python -m pathos.engine.steering_extract --model /path/to/model.gguf
    python -m pathos.engine.steering_extract --generate-stories --stories-per-emotion 15
    python -m pathos.engine.steering_extract --extract-171 --model qwen3:4b

Usage from code:
    from pathos.engine.steering_extract import extract_and_cache, extract_171_probes
    result = extract_and_cache("qwen3:4b", device="auto")
    result = extract_171_probes("qwen3:4b", layer=None)

The 4D script:
1. Locates the model (Ollama GGUF, HuggingFace, or local path)
2. Loads it via transformers (dequantized to fp16/fp32)
3. Runs contrastive pairs through the model
4. Extracts hidden states at early/mid/late layers
5. Computes direction vectors (mean difference)
6. Saves to src/pathos/steering_data/cached_vectors/<model>.npz
7. Unloads the model to free memory

Typical runtime: 2-5 min on GPU, 10-30 min on CPU (one-time cost).

The 171-probe script (RESIDUUM F1.2):
1. Reads 171 emotions and ~15 stories per emotion from steering_data/stories/
2. Reads ~100 neutral transcripts from neutral_transcripts.json
3. Extracts residual activations at a single ~2/3 depth layer
4. Computes raw probe = mean(emotion_stories) - mean(all_stories)
5. PCA over neutral activations, projects out PCs covering 50% of variance
6. Normalizes to unit vectors
7. Saves to cached_vectors/<model>_171.npz

Typical runtime: 15-30 min on GPU, 1-3 h on CPU (one-time cost).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# --- RESIDUUM F1.2 paths ---
_STEERING_DATA_DIR = Path(__file__).parent.parent / "steering_data"
_EMOTIONS_171_PATH = _STEERING_DATA_DIR / "emotions_171.json"
_STORY_TOPICS_PATH = _STEERING_DATA_DIR / "story_topics.json"
_NEUTRAL_TRANSCRIPTS_PATH = _STEERING_DATA_DIR / "neutral_transcripts.json"
_STORIES_DIR = _STEERING_DATA_DIR / "stories"
_DIALOGUES_171_PATH = _STEERING_DATA_DIR / "dialogues_171.json"
_CACHED_VECTORS_DIR_171 = _STEERING_DATA_DIR / "cached_vectors"


def extract_and_cache(
    model_id: str,
    device: str = "auto",
    layers: list[int] | None = None,
    batch_size: int = 4,
    max_seq_len: int = 64,
) -> dict[str, Any]:
    """Extract steering vectors from a model and save to cache.

    Args:
        model_id: Model identifier — Ollama name (e.g. "qwen3:4b"),
            HuggingFace ID (e.g. "Qwen/Qwen3-4B"), or local path.
        device: Device map — "auto", "cpu", or "cuda".
        layers: Specific layers to extract from. None = auto-select.
        batch_size: Batch size for extraction (reduce if OOM).
        max_seq_len: Max tokens per prompt.

    Returns:
        Dict with extraction results and metadata.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Steering extraction requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e

    from pathos.engine.steering import EmotionalSteeringEngine
    from pathos.llm.transformers_provider import find_ollama_gguf, _ollama_to_hf_id

    t_start = time.perf_counter()

    # --- Resolve model path ---
    # Use Ollama's local GGUF when available (no extra download).
    # GGUF + device_map="auto" causes a bug in transformers 4.46.x, so
    # GGUF models are always loaded on CPU for extraction (one-time cost).
    load_path = model_id
    gguf_file: str | None = None
    source = "huggingface"
    force_cpu = False

    if ":" in model_id and "/" not in model_id:
        # Ollama model name — try local GGUF first (avoids re-downloading)
        gguf_path = find_ollama_gguf(model_id)
        if gguf_path is not None:
            load_path = str(gguf_path.parent)
            gguf_file = gguf_path.name
            source = "ollama_gguf"
            force_cpu = True  # GGUF + device_map="auto" crashes in transformers 4.46.x
            logger.info("Found Ollama GGUF: %s (loading on CPU to avoid transformers bug)", gguf_path)
        else:
            hf_id = _ollama_to_hf_id(model_id)
            if hf_id:
                load_path = hf_id
                source = "huggingface_mapped"
                logger.info("Mapped '%s' → '%s'", model_id, hf_id)
            else:
                logger.warning("Cannot find model '%s'. Trying as HF ID.", model_id)

    # Check if local path
    if Path(load_path).is_dir() or Path(load_path).is_file():
        source = "local_path"

    # --- Determine dtype and device ---
    if force_cpu or device == "cpu":
        device_map = "cpu"
        dtype = torch.float32
    elif device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32
    else:
        device_map = device
        dtype = torch.float16

    logger.info(
        "Loading model '%s' (source=%s, device=%s, dtype=%s)...",
        model_id, source, device_map, dtype,
    )

    # --- Load tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            load_path, trust_remote_code=True,
        )
    except Exception:
        hf_id = _ollama_to_hf_id(model_id)
        if hf_id:
            logger.info("Tokenizer fallback to HF ID: %s", hf_id)
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=True,
            )
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if gguf_file:
        load_kwargs["gguf_file"] = gguf_file

    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
    model.eval()

    t_load = time.perf_counter()
    load_time = t_load - t_start
    logger.info("Model loaded in %.1fs", load_time)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info("GPU memory used: %.2f GB", gpu_mem)

    # --- Extract vectors ---
    engine = EmotionalSteeringEngine()
    cached = engine.extract_vectors(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        layers=layers,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    t_extract = time.perf_counter()
    extract_time = t_extract - t_load
    logger.info("Extraction completed in %.1fs", extract_time)

    # --- Save cache ---
    save_path = engine.save_vectors()
    logger.info("Vectors saved to: %s", save_path)

    # --- Cleanup ---
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start

    result = {
        "model_id": model_id,
        "source": source,
        "device": str(device_map),
        "dtype": str(dtype),
        "num_layers": cached.num_layers,
        "hidden_size": cached.hidden_size,
        "dimensions": cached.available_dimensions,
        "layers_extracted": sorted(cached.available_layers),
        "total_vectors": sum(len(ld) for ld in cached.vectors.values()),
        "cache_path": str(save_path),
        "load_time_s": round(load_time, 1),
        "extract_time_s": round(extract_time, 1),
        "total_time_s": round(total_time, 1),
    }

    logger.info(
        "Done! %d vectors for %d dimensions × %d layers in %.1fs total",
        result["total_vectors"],
        len(result["dimensions"]),
        len(result["layers_extracted"]),
        total_time,
    )

    return result


def list_available_models() -> list[dict[str, str]]:
    """List Ollama models that have cached steering vectors or GGUFs available.

    Returns list of dicts with model_id, status ("cached", "gguf_found", "not_found").
    """
    from pathos.engine.steering import _CACHED_VECTORS_DIR, load_cached_vectors
    from pathos.llm.transformers_provider import find_ollama_gguf, _OLLAMA_TO_HF

    results = []
    # Check common model names
    model_names = list(_OLLAMA_TO_HF.keys())

    for name in model_names:
        cached = load_cached_vectors(name, _CACHED_VECTORS_DIR)
        if cached is not None:
            results.append({
                "model_id": name,
                "status": "cached",
                "dimensions": len(cached.available_dimensions),
                "layers": len(cached.available_layers),
            })
        elif find_ollama_gguf(name) is not None:
            results.append({
                "model_id": name,
                "status": "gguf_found",
            })
        # Skip models that aren't present at all

    return results


# ============================================================================
# RESIDUUM F1.2 — 171-probe extraction with neutral PC projection
# ============================================================================
#
# Pipeline (from RESIDUUMREWORK.txt):
#   1. Generate N stories per emotion (171 x 15 = 2565 default).
#   2. Load Transformers model at layer ~2/3 of depth.
#   3. For each story: mean residual activation over tokens [50:].
#   4. For each emotion: probe_raw = mean(its stories) - mean(all stories).
#   5. Over neutral transcripts: PCA -> top PCs covering >=50% variance.
#   6. Project those PCs out of each probe_raw.
#   7. Normalize to unit vector. Save NPZ.


def _slugify_emotion(name: str) -> str:
    """Safe directory name for a given emotion string."""
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def _load_emotions_171() -> list[dict[str, Any]]:
    with _EMOTIONS_171_PATH.open(encoding="utf-8") as f:
        return json.load(f)["emotions"]


def _load_story_topics() -> list[str]:
    with _STORY_TOPICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)["topics"]


def _load_neutral_transcripts() -> list[str]:
    with _NEUTRAL_TRANSCRIPTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)["transcripts"]


def _build_story_prompt(emotion_name: str, topic: str) -> str:
    """Build a generation prompt for one emotion story.

    The anti-leak rule from the paper: the emotion word MUST NOT appear in
    the story. Emotion is conveyed through actions, sensations, dialogue.
    """
    return (
        "Write a short 1-2 paragraph narrative passage set around the "
        f"following everyday situation: '{topic}'. The passage must evoke "
        f"the emotional experience of a character who is feeling {emotion_name}, "
        "BUT the word '" + emotion_name + "' and any of its direct forms "
        "(conjugations, -ness, -ing, -ed, -ly) must NEVER appear in the text. "
        "Convey the emotion through actions, body sensations, thoughts, "
        "dialogue, and context. Mix first-person and third-person narration "
        "across different passages. Write only the narrative, no preamble, "
        "no commentary, no quotation marks around the whole thing, no "
        "meta-references."
    )


def _default_ollama_llm_call(
    base_url: str = "http://localhost:11434",
    model: str = "qwen3:4b",
    timeout_s: float = 180.0,
    temperature: float = 0.9,
    num_predict: int = 1500,
    system_message: str | None = None,
    assistant_prefix: str | None = None,
) -> Callable[[str], str]:
    """Build a sync LLM call to Ollama's /api/chat endpoint.

    The `think` field in the JSON payload is ignored by current qwen3 builds —
    they always emit a chain-of-thought preamble. Two reliable defenses:
      1. A strict system_message (e.g. _DIALOGUE_SYSTEM_PROMPT) that tells
         the model to skip reasoning and emit the artefact directly.
      2. assistant_prefix: an assistant turn at the end of the message array
         that Ollama treats as a continuation seed. With assistant_prefix
         set, the model picks up FROM that prefix, so the response is the
         continuation; the helper concatenates the prefix back onto the
         returned text so callers always see a complete response.
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx required for Ollama story generation") from e

    def _call(prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # ignored by current qwen3 builds; kept for older models
            "keep_alive": "30m",  # keep model resident in VRAM across the batch
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        msg = data.get("message", {})
        content = str(msg.get("content", "")).strip()
        # Ollama returns only the continuation after the assistant prefix.
        # Prepend it back so downstream cleaners see a complete dialogue.
        if assistant_prefix and not content.lower().lstrip().startswith(
            assistant_prefix.rstrip(":").lower()
        ):
            content = assistant_prefix + " " + content
        return content

    return _call


def _clean_generated_story(raw: str, emotion_name: str) -> str:
    """Strip common generation artifacts: think tags, stray reasoning, code fences."""
    import re

    text = raw.strip()
    # Full <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Reasoning leaked without opening tag but closed — drop everything before </think>
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>"):].strip()
    # Strip triple-backtick fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines and lines[-1].startswith("```") else lines[1:])
    return text.strip()


def _story_contains_emotion_word(text: str, emotion_name: str) -> bool:
    """True if the emotion stem appears in the text (morphological match, case insensitive).

    Catches the base form plus common English morphology:
      - happy -> happy, happily, happiness, happier, happiest (via -y -> -i stem)
      - excite -> excite, excited, excitement, exciting (via dropped -e stem)
      - angry -> angry, angrily, angrier (via -y -> -i)
      - sad -> sad, sadness, sadder (suffix chars allowed)
    """
    import re

    e = emotion_name.lower().strip()
    if not e:
        return False
    stems = {e}
    if e.endswith("y") and len(e) > 1:
        stems.add(e[:-1] + "i")  # happy -> happi (for happily, happiness, happier)
    if e.endswith("e") and len(e) > 1:
        stems.add(e[:-1])  # excite -> excit (for excited, exciting, excitement)
    if e.endswith("ed") and len(e) > 3:
        # 62 of 171 emotions end in -ed (alarmed, amused, annoyed...).
        # Cover the verb root and -ing form: amused -> amuse, amus, amusing.
        stems.add(e[:-2])            # amuse (catches amused, amuses, amusement)
        stems.add(e[:-1])            # amused (catches amusedly, already covered by base)
        stems.add(e[:-2] + "ing")    # amusing
    if e.endswith("ing") and len(e) > 4:
        # brooding, loving -> brood, love (catches brooded, lovable, etc.).
        stems.add(e[:-3])

    t = text.lower()
    for stem in stems:
        pattern = r"\b" + re.escape(stem) + r"\w{0,5}\b"
        if re.search(pattern, t):
            return True
    return False


def generate_stories_for_emotions(
    n_per_emotion: int = 15,
    llm_call: Callable[[str], str] | None = None,
    output_dir: Path | None = None,
    skip_existing: bool = True,
    max_retries_per_story: int = 3,
    seed: int | None = None,
    min_bytes: int = 200,
) -> dict[str, Any]:
    """Generate N stories per emotion offline using the active LLM.

    Stories are saved to output_dir/<emotion_slug>/story_<n>.txt. If a file
    already exists and skip_existing is True, generation is skipped.

    Args:
        n_per_emotion: Target number of stories per emotion (default 15).
        llm_call: Sync callable (prompt) -> text. Defaults to Ollama at
            localhost:11434 with qwen3:4b.
        output_dir: Override output directory.
        skip_existing: If True, do not overwrite existing story files that
            meet the min_bytes threshold. Existing files below the threshold
            are treated as missing and regenerated.
        max_retries_per_story: Retry attempts when the generated text leaks
            the emotion word or is shorter than min_bytes.
        seed: Random seed for topic sampling (reproducibility).
        min_bytes: Minimum UTF-8 byte length for a story to count as valid.
            Shorter generations are rejected (they usually represent early
            EOS / truncated output that would degrade the probe mean).

    Returns:
        Dict with counts: {"generated", "skipped", "failed", "total_target"}.
    """
    import random

    out = output_dir or _STORIES_DIR
    out.mkdir(parents=True, exist_ok=True)

    emotions = _load_emotions_171()
    topic_texts = _load_story_topics()

    if llm_call is None:
        llm_call = _default_ollama_llm_call()

    rng = random.Random(seed)
    counts = {"generated": 0, "skipped": 0, "failed": 0, "total_target": len(emotions) * n_per_emotion}

    for idx, emo in enumerate(emotions):
        name = emo["name"]
        slug = _slugify_emotion(name)
        emo_dir = out / slug
        emo_dir.mkdir(parents=True, exist_ok=True)

        for n in range(n_per_emotion):
            story_path = emo_dir / f"story_{n:02d}.txt"
            if skip_existing and story_path.exists() and story_path.stat().st_size >= min_bytes:
                counts["skipped"] += 1
                continue

            story_text = ""
            for attempt in range(max_retries_per_story):
                topic = rng.choice(topic_texts)
                prompt = _build_story_prompt(name, topic)
                try:
                    raw = llm_call(prompt)
                except Exception as e:
                    logger.warning("LLM call failed for %s/story_%d (attempt %d): %s",
                                   slug, n, attempt + 1, e)
                    continue
                cleaned = _clean_generated_story(raw, name)
                if not cleaned:
                    continue
                if _story_contains_emotion_word(cleaned, name):
                    logger.debug("Leak detected for '%s' in story_%d, retrying", name, n)
                    continue
                if len(cleaned.encode("utf-8")) < min_bytes:
                    logger.debug("Too short (%d bytes < %d) for '%s' in story_%d, retrying",
                                 len(cleaned.encode("utf-8")), min_bytes, name, n)
                    continue
                story_text = cleaned
                break

            if not story_text:
                logger.warning("Failed to generate clean story %s/story_%d after %d attempts",
                               slug, n, max_retries_per_story)
                counts["failed"] += 1
                continue

            story_path.write_text(story_text, encoding="utf-8")
            counts["generated"] += 1

        if (idx + 1) % 10 == 0:
            logger.info("Progress: %d/%d emotions done (generated=%d, skipped=%d, failed=%d)",
                        idx + 1, len(emotions), counts["generated"],
                        counts["skipped"], counts["failed"])

    logger.info("Story generation complete: %s", counts)
    return counts


# ============================================================================
# RESIDUUM F2.3 — Dual probes dataset (Person1 / Person2 dialogues)
# ============================================================================
#
# The paper documents two orthogonal probe families: "present speaker" (the
# emotion of the speaker currently producing tokens) and "other speaker" (the
# emotion of the interlocutor). To extract both, we need a dataset where two
# characters express DISTINCT emotions in the same context — that is what
# dialogues_171.json provides.
#
# Schema and pairing rules are documented in cambios.txt lines 2117-2170.
# Output: src/pathos/steering_data/dialogues_171.json (single file, not a dir,
# because each dialogue belongs to TWO emotions).


_DIALOGUE_SYSTEM_PROMPT = (
    "You are a dialogue writer. Output ONLY the requested dialogue. "
    "Start your response immediately with 'Person1:'. NEVER explain, "
    "NEVER reason aloud, NEVER add preamble or commentary. "
    "Just write the 4 lines in the requested format."
)
_DIALOGUE_ASSISTANT_PREFIX = "Person1:"


def _build_dialogue_prompt(emo_a: str, emo_b: str, topic: str) -> str:
    """Build a generation prompt for a 4-turn spoken dialogue with two emotions.

    Anti-leak rule applies to BOTH emotion words (Person1's and Person2's).
    The prompt is paired with a system message + assistant pre-fill 'Person1:'
    at the LLM call layer; see _default_ollama_llm_call.
    """
    return (
        f"Write a 4-turn spoken dialogue between Person1 and Person2 about: "
        f"'{topic}'. Person1 is feeling {emo_a}. Person2 is feeling {emo_b}.\n\n"
        "Rules:\n"
        "- Each line is what the character SAYS OUT LOUD (15-40 words).\n"
        "- Show emotion through word choice, hesitation, and subtext — "
        "NOT through stage directions in asterisks or parentheses.\n"
        f"- The words '{emo_a}' and '{emo_b}' and their direct forms "
        "(conjugations, -ness, -ing, -ed, -ly) must NEVER appear.\n\n"
        "Output exactly this format (only 4 lines, no preamble, no comments):\n"
        "Person1: <spoken line>\n"
        "Person2: <spoken line>\n"
        "Person1: <spoken line>\n"
        "Person2: <spoken line>"
    )


def _clean_generated_dialogue(raw: str) -> str:
    """Strip preamble/reasoning that precedes the first 'Person1:' line.

    qwen3 (and other thinking models) sometimes ignore /no_think and emit
    chain-of-thought before the actual dialogue. We discard everything before
    the first labelled turn so the parser only sees the dialogue body.

    Also handles:
      - <think>...</think> blocks (full and orphaned closing tags)
      - markdown bold around speaker labels (**Person1**:)
      - leading code fences
    """
    import re

    text = raw.strip()
    # Full <think>...</think> blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Orphaned closing think tag: drop everything up to and including it.
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>"):].strip()
    # Strip surrounding code fences.
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines and lines[-1].startswith("```") else lines[1:])
    # Strip bold markdown around speaker labels.
    text = re.sub(r"\*\*\s*Person\s*([12])\s*\*\*\s*:", r"Person\1:", text, flags=re.IGNORECASE)
    # Discard any preamble before the first 'Person1:' label.
    m = re.search(r"^\s*Person\s*1\s*:", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        text = text[m.start():]
    return text.strip()


def _parse_dialogue_turns(raw: str) -> list[dict[str, str]] | None:
    """Parse alternating 'Person1: ...' / 'Person2: ...' turns.

    Accepts 'Person 1', 'Person1', 'PERSON1:', etc. The expected pattern is
    P1, P2, P1, P2. If MORE than 4 labelled turns are present (e.g. the
    model leaked thinking after the dialogue and produced extra labels),
    the first 4 are kept provided they alternate correctly. Returns None
    when fewer than 4 valid turns are found or the first 4 do not alternate
    P1/P2/P1/P2.

    The body of the final captured turn is also clipped at any trailing
    '<think>' block or markdown fence to avoid contaminating the residual
    extraction with reasoning text.
    """
    import re

    text = raw.strip()
    pattern = re.compile(
        r"^\s*Person\s*([12])\s*:\s*(.*?)(?=^\s*Person\s*[12]\s*:|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    turns: list[dict[str, str]] = []
    for m in pattern.finditer(text):
        speaker_idx = m.group(1)
        body = m.group(2).strip()
        # Strip trailing reasoning/markdown that the model sometimes appends
        # after the 4-turn dialogue body. Cut at the earliest sentinel.
        for sentinel in ("<think>", "</think>", "```"):
            cut = body.lower().find(sentinel.lower())
            if cut != -1:
                body = body[:cut].rstrip()
        if not body:
            return None
        turns.append({"speaker": f"Person{speaker_idx}", "text": body})

    if len(turns) < 4:
        return None
    turns = turns[:4]
    if [t["speaker"] for t in turns] != ["Person1", "Person2", "Person1", "Person2"]:
        return None
    return turns


def _dialogue_contains_either_emotion(
    turns: list[dict[str, str]], emo_a: str, emo_b: str
) -> bool:
    """True if EITHER emotion stem appears anywhere in the dialogue.

    Reuses _story_contains_emotion_word (morphological match) for each.
    """
    full_text = " ".join(t["text"] for t in turns)
    return (
        _story_contains_emotion_word(full_text, emo_a)
        or _story_contains_emotion_word(full_text, emo_b)
    )


def _load_existing_dialogues(out_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load already-generated dialogues for resumability. Empty dict if missing."""
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse %s (%s) — starting from scratch", out_path, e)
        return {}
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    for d in data.get("dialogues", []):
        try:
            key = (d["person1_emotion"], int(d["idx_within_block"]))
        except (KeyError, TypeError, ValueError):
            continue
        existing[key] = d
    return existing


def _save_dialogues_atomic(
    out_path: Path,
    dialogues: list[dict[str, Any]],
    n_per_emotion: int,
) -> None:
    """Write dialogues to a .tmp file and atomically replace the target."""
    payload = {
        "_meta": {
            "n_per_emotion_as_person1": n_per_emotion,
            "total_dialogues": len(dialogues),
            "structure": "Person1 = emotion_A, Person2 = emotion_B distinct",
            "turns_per_dialogue": 4,
            "topics_source": "story_topics.json",
            "emotions_source": "emotions_171.json",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "dialogues": dialogues,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def generate_dialogues_for_emotions(
    n_per_emotion: int = 5,
    llm_call: Callable[[str], str] | None = None,
    output_path: Path | None = None,
    skip_existing: bool = True,
    max_retries_per_dialogue: int = 5,
    seed: int = 42,
    min_chars_per_turn: int = 20,
    emotions: list[dict[str, Any]] | None = None,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    """Generate N dialogues per emotion (as Person1) offline using the active LLM.

    For each of the 171 emotions A, generates n_per_emotion dialogues where
    Person1 expresses A and Person2 expresses a different emotion B sampled
    without replacement from the other 170. Each dialogue has exactly 4 turns
    (P1 -> P2 -> P1 -> P2). Both emotion words must be absent from the final
    text (double anti-leak).

    The output is a SINGLE JSON file (not a directory) because each dialogue
    belongs to two emotions. Schema documented in cambios.txt L2117-2140.

    Args:
        n_per_emotion: Dialogues per emotion as Person1 (default 5 -> 855 total).
        llm_call: Sync callable (prompt) -> text. Defaults to Ollama qwen3:4b.
        output_path: Override output file location.
        skip_existing: If True, re-use already-generated dialogues from a prior run.
            Each dialogue is keyed by (person1_emotion, idx_within_block).
        max_retries_per_dialogue: Retry attempts on parse failure or anti-leak hit.
        seed: Master seed for pairing and topic sampling (reproducible).
        min_chars_per_turn: Minimum stripped length per turn to accept.

    Returns:
        Dict with counts: {"generated", "skipped", "failed", "total_target"}.
    """
    import random

    out = output_path or _DIALOGUES_171_PATH
    if emotions is None:
        emotions = _load_emotions_171()
    if topics is None:
        topics = _load_story_topics()

    if llm_call is None:
        # Use the strict dialogue-mode defaults: system message + assistant
        # pre-fill 'Person1:'. These two together suppress qwen3's
        # chain-of-thought preamble and force a direct 4-turn output.
        llm_call = _default_ollama_llm_call(
            temperature=0.7,
            num_predict=600,
            system_message=_DIALOGUE_SYSTEM_PROMPT,
            assistant_prefix=_DIALOGUE_ASSISTANT_PREFIX,
        )

    existing = _load_existing_dialogues(out) if skip_existing else {}

    counts = {
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "total_target": len(emotions) * n_per_emotion,
    }
    dialogues: list[dict[str, Any]] = []
    next_id = 0

    for idx_a, emo_a in enumerate(emotions):
        name_a = emo_a["name"]
        # Deterministic per-block RNG: same seed yields same pairing for emotion A.
        block_rng = random.Random(seed * 100_003 + idx_a)
        others = [e["name"] for e in emotions if e["name"] != name_a]
        # Sample n_per_emotion DISTINCT P2 partners for this block.
        if n_per_emotion > len(others):
            raise ValueError(
                f"n_per_emotion={n_per_emotion} exceeds available distinct "
                f"partners ({len(others)})"
            )
        sampled_p2 = block_rng.sample(others, n_per_emotion)
        # Sticky topic per dialogue (same topic across retries). Rotating the
        # topic on each retry was discarding successful context: if a combo
        # like (afraid, perplexed) needs 3 attempts to produce a clean
        # dialogue, all 3 attempts should share the same topic so the model
        # can refine the same scene, not start cold each time.
        sampled_topics = [block_rng.choice(topics) for _ in range(n_per_emotion)]

        for n in range(n_per_emotion):
            name_b = sampled_p2[n]
            topic = sampled_topics[n]
            key = (name_a, n)

            # Resumability: reuse a valid existing dialogue if it passes anti-leak.
            if key in existing:
                ex = existing[key]
                ex_turns = ex.get("turns", [])
                if (
                    isinstance(ex_turns, list)
                    and len(ex_turns) == 4
                    and ex.get("person2_emotion") == name_b
                    and not _dialogue_contains_either_emotion(ex_turns, name_a, name_b)
                ):
                    dialogues.append(
                        {
                            "id": next_id,
                            "person1_emotion": name_a,
                            "person2_emotion": name_b,
                            "idx_within_block": n,
                            "topic": ex.get("topic", ""),
                            "turns": ex_turns,
                        }
                    )
                    next_id += 1
                    counts["skipped"] += 1
                    continue

            turns: list[dict[str, str]] | None = None
            chosen_topic: str = ""
            for attempt in range(max_retries_per_dialogue):
                prompt = _build_dialogue_prompt(name_a, name_b, topic)
                try:
                    raw = llm_call(prompt)
                except Exception as e:
                    logger.warning(
                        "LLM call failed for dialogue %s/%s n=%d (attempt %d): %s",
                        name_a, name_b, n, attempt + 1, e,
                    )
                    continue
                cleaned = _clean_generated_dialogue(raw)
                parsed = _parse_dialogue_turns(cleaned)
                if parsed is None:
                    logger.debug(
                        "Parse failed for %s/%s n=%d (attempt %d)",
                        name_a, name_b, n, attempt + 1,
                    )
                    continue
                if _dialogue_contains_either_emotion(parsed, name_a, name_b):
                    logger.debug(
                        "Anti-leak hit for %s/%s n=%d (attempt %d)",
                        name_a, name_b, n, attempt + 1,
                    )
                    continue
                if any(len(t["text"]) < min_chars_per_turn for t in parsed):
                    logger.debug(
                        "Turn too short for %s/%s n=%d (attempt %d)",
                        name_a, name_b, n, attempt + 1,
                    )
                    continue
                turns = parsed
                chosen_topic = topic
                break

            if turns is None:
                logger.warning(
                    "Failed dialogue %s/%s n=%d after %d attempts",
                    name_a, name_b, n, max_retries_per_dialogue,
                )
                counts["failed"] += 1
                continue

            dialogues.append(
                {
                    "id": next_id,
                    "person1_emotion": name_a,
                    "person2_emotion": name_b,
                    "idx_within_block": n,
                    "topic": chosen_topic,
                    "turns": turns,
                }
            )
            next_id += 1
            counts["generated"] += 1

        # Incremental save after each emotion block (resumability).
        _save_dialogues_atomic(out, dialogues, n_per_emotion)

        if (idx_a + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d emotions done (generated=%d, skipped=%d, failed=%d)",
                idx_a + 1, len(emotions),
                counts["generated"], counts["skipped"], counts["failed"],
            )

    logger.info("Dialogue generation complete: %s", counts)
    return counts


def _load_model_for_residuum(model_id: str, device: str) -> tuple[Any, Any, str, Any, str]:
    """Load tokenizer + model for 171-probe extraction. Returns (model, tokenizer, source, dtype, device_map)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "171-probe extraction requires torch + transformers."
        ) from e

    from pathos.llm.transformers_provider import find_ollama_gguf, _ollama_to_hf_id

    load_path = model_id
    gguf_file: str | None = None
    source = "huggingface"
    force_cpu = False

    use_gpu_quant = torch.cuda.is_available() and device in ("auto", "cuda")

    if ":" in model_id and "/" not in model_id:
        hf_id = _ollama_to_hf_id(model_id)
        if use_gpu_quant and hf_id:
            load_path = hf_id
            source = "huggingface_mapped"
            logger.info("CUDA detected — HF '%s' with 4-bit nf4 quant", hf_id)
        else:
            gguf_path = find_ollama_gguf(model_id)
            if gguf_path is not None:
                load_path = str(gguf_path.parent)
                gguf_file = gguf_path.name
                source = "ollama_gguf"
                force_cpu = True
                logger.info("Found Ollama GGUF: %s (CPU load)", gguf_path)
            elif hf_id:
                load_path = hf_id
                source = "huggingface_mapped"
            else:
                logger.warning("Cannot resolve '%s'. Trying as HF ID.", model_id)

    if Path(load_path).is_dir() or Path(load_path).is_file():
        source = "local_path"

    if force_cpu or device == "cpu":
        device_map, dtype = "cpu", torch.float32
    elif device == "auto":
        if torch.cuda.is_available():
            device_map, dtype = "auto", torch.float16
        else:
            device_map, dtype = "cpu", torch.float32
    else:
        device_map, dtype = device, torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    except Exception:
        hf_id = _ollama_to_hf_id(model_id)
        if hf_id:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if gguf_file:
        load_kwargs["gguf_file"] = gguf_file
        load_kwargs["torch_dtype"] = dtype
    elif use_gpu_quant and not force_cpu:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
    model.eval()
    return model, tokenizer, source, dtype, str(device_map)


def _mean_residual_for_text(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int,
    token_start: int,
    max_seq_len: int,
) -> "Any | None":
    """Run text through the model and return the mean residual at layer over tokens [token_start:].

    Returns None if the tokenized text has no tokens past token_start.
    """
    import numpy as np
    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_seq_len,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (num_layers+1) tensors. Index 0 = embeddings.
    # Layer N lives at index N+1.
    hidden = outputs.hidden_states
    if layer + 1 >= len(hidden):
        raise ValueError(f"Layer {layer} out of range for {len(hidden) - 1} layers.")

    layer_h = hidden[layer + 1][0]  # (seq_len, hidden)
    seq_len = layer_h.shape[0]
    if seq_len <= token_start:
        start = max(0, seq_len - 1)
    else:
        start = token_start
    mean_vec = layer_h[start:].mean(dim=0).float().cpu().numpy()
    return np.asarray(mean_vec, dtype=np.float32)


def _compute_neutral_pcs(
    activations: "Any",
    variance_threshold: float,
) -> "Any":
    """Compute top principal components of neutral activations until variance_threshold covered.

    Args:
        activations: (N_neutral, hidden_size) array.
        variance_threshold: Accumulated explained variance to cover [0, 1].

    Returns:
        (n_pcs, hidden_size) array of unit-norm PC vectors.
    """
    import numpy as np

    X = activations - activations.mean(axis=0, keepdims=True)
    # SVD: X = U S V^T. Right singular vectors V are the PCs.
    # For wide matrices (hidden_size >> N_neutral), full_matrices=False keeps it cheap.
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    variances = s ** 2
    total = variances.sum()
    if total <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    cum = np.cumsum(variances) / total
    # Include all PCs up to (and including) the first that crosses the threshold.
    n_pcs = int(np.searchsorted(cum, variance_threshold) + 1)
    n_pcs = min(n_pcs, vt.shape[0])
    pcs = vt[:n_pcs].astype(np.float32)
    # vt rows are already unit-norm from SVD.
    return pcs


def _project_out(vec: "Any", basis: "Any") -> "Any":
    """Project a vector out of a basis of unit vectors.

    Args:
        vec: (hidden_size,) array.
        basis: (n_pcs, hidden_size) array of unit-norm vectors (assumed).

    Returns:
        (hidden_size,) array with components along basis removed.
    """
    import numpy as np

    if basis.size == 0:
        return vec.copy()
    # Assume basis rows are ~orthonormal from SVD. Subtract projections.
    coeffs = basis @ vec  # (n_pcs,)
    return vec - coeffs @ basis


def extract_171_probes(
    model_id: str,
    layer: int | None = None,
    device: str = "auto",
    stories_per_emotion: int = 15,
    neutral_variance: float = 0.50,
    token_start: int = 50,
    max_seq_len: int = 512,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract 171 emotion probes from stories with neutral PC projection.

    Pipeline:
      1. For each emotion, load its stories (stories/<emotion>/story_*.txt).
      2. Mean residual at `layer` over tokens [token_start:], per story.
      3. probe_raw[e] = mean(activations_e) - mean(all_activations).
      4. Run neutral transcripts, PCA over them, keep PCs covering
         neutral_variance of variance.
      5. Project those PCs out of each probe_raw, then normalize.
      6. Save as cached_vectors/<model>_171.npz with arrays:
           probes: (171, hidden_size)
           emotion_names: (171,) str
           clusters: (171,) str
           neutral_pcs: (n_pcs, hidden_size)
           metadata: (1,) str (JSON)

    Args:
        model_id: Model identifier (Ollama, HF, or local path).
        layer: Target layer. None = auto-select ~2/3 of depth.
        device: "auto" | "cpu" | "cuda".
        stories_per_emotion: Max stories to read per emotion (defaults 15).
        neutral_variance: Variance fraction for neutral PC cutoff.
        token_start: First token index considered in mean (skips preface).
        max_seq_len: Max tokens per input.
        seed: Random seed (used only for tie-breaks; extraction is deterministic).
        output_dir: Override cache directory.

    Returns:
        Dict with metadata about the extraction.
    """
    try:
        import numpy as np
        import torch
    except ImportError as e:
        raise ImportError("171-probe extraction requires numpy + torch.") from e

    from pathos.models.residuum import ExtractionConfig

    t_start = time.perf_counter()

    out = output_dir or _CACHED_VECTORS_DIR_171
    out.mkdir(parents=True, exist_ok=True)

    emotions = _load_emotions_171()
    if len(emotions) != 171:
        raise ValueError(f"Expected 171 emotions, got {len(emotions)}")
    emotion_names = [e["name"] for e in emotions]
    clusters = [e["cluster"] for e in emotions]
    neutral_transcripts = _load_neutral_transcripts()

    # --- Collect per-emotion stories (fail fast if missing) ---
    stories_by_emotion: dict[str, list[str]] = {}
    missing: list[str] = []
    for e in emotions:
        name = e["name"]
        emo_dir = _STORIES_DIR / _slugify_emotion(name)
        if not emo_dir.is_dir():
            missing.append(name)
            continue
        files = sorted(emo_dir.glob("story_*.txt"))[:stories_per_emotion]
        texts = [p.read_text(encoding="utf-8").strip() for p in files if p.stat().st_size > 0]
        if not texts:
            missing.append(name)
            continue
        stories_by_emotion[name] = texts

    if missing:
        raise FileNotFoundError(
            f"Missing stories for {len(missing)} emotions (first 5: {missing[:5]}). "
            "Run with --generate-stories first."
        )

    total_stories = sum(len(v) for v in stories_by_emotion.values())
    logger.info("Stories loaded: %d emotions x ~%d = %d total",
                len(stories_by_emotion), stories_per_emotion, total_stories)

    # --- Load model ---
    model, tokenizer, source, dtype, device_map = _load_model_for_residuum(model_id, device)
    t_load = time.perf_counter()
    logger.info("Model loaded in %.1fs (source=%s, device=%s)",
                t_load - t_start, source, device_map)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info("GPU memory used: %.2f GB", gpu_mem)

    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    if num_layers <= 0 or hidden_size <= 0:
        raise RuntimeError("Cannot detect num_hidden_layers/hidden_size from model config.")

    if layer is None:
        layer = max(1, min(num_layers - 1, int(round(num_layers * 2 / 3))))
        logger.info("Auto-selected layer %d of %d (~2/3 depth)", layer, num_layers)
    elif not (0 <= layer < num_layers):
        raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}]")

    # --- Extract per-emotion activations ---
    emotion_means = np.zeros((171, hidden_size), dtype=np.float32)
    story_counts: list[int] = []
    all_story_activations: list[np.ndarray] = []

    for i, name in enumerate(emotion_names):
        stories = stories_by_emotion[name]
        story_vecs: list[np.ndarray] = []
        for s in stories:
            try:
                v = _mean_residual_for_text(model, tokenizer, s, layer, token_start, max_seq_len)
                if v is not None:
                    story_vecs.append(v)
                    all_story_activations.append(v)
            except Exception as e:
                logger.warning("Skip story for %s: %s", name, e)
        if not story_vecs:
            raise RuntimeError(f"No usable activations for emotion '{name}'")
        emotion_means[i] = np.mean(np.stack(story_vecs, axis=0), axis=0)
        story_counts.append(len(story_vecs))
        if (i + 1) % 20 == 0 or (i + 1) == 171:
            logger.info("Emotion activations: %d/171", i + 1)

    t_emo = time.perf_counter()
    logger.info("Emotion activations extracted in %.1fs", t_emo - t_load)

    # --- Raw probes: probe_raw[e] = mean(emotion stories) - mean(ALL stories) ---
    all_story_stack = np.stack(all_story_activations, axis=0)
    all_mean = np.mean(all_story_stack, axis=0)
    probes_raw = emotion_means - all_mean[np.newaxis, :]
    norms_before = np.linalg.norm(probes_raw, axis=1)

    # F4.0 — Norma tipica del residual stream en el layer objetivo.
    # Computed over the same activations the probes are built from, so it
    # reflects the model's actual residual magnitude at this depth. Used by
    # F4 granular steering as the reference for MAX_STEERING_FRACTION (the
    # steering vector L2 is capped as a fraction of this value).
    residual_norm_typical = float(np.mean(np.linalg.norm(all_story_stack, axis=1)))
    logger.info("Residual norm typical (layer %d): %.3f", layer, residual_norm_typical)

    # --- Neutral activations + PCA ---
    logger.info("Extracting neutral activations (%d transcripts)...", len(neutral_transcripts))
    neutral_activations: list[np.ndarray] = []
    for t in neutral_transcripts:
        try:
            v = _mean_residual_for_text(model, tokenizer, t, layer, token_start, max_seq_len)
            if v is not None:
                neutral_activations.append(v)
        except Exception as e:
            logger.warning("Skip neutral transcript: %s", e)
    neutral_stack = np.stack(neutral_activations, axis=0)
    logger.info("Neutral activations: %d x %d", *neutral_stack.shape)

    pcs = _compute_neutral_pcs(neutral_stack, neutral_variance)
    logger.info("Neutral PCs kept: %d (variance threshold=%.2f)", pcs.shape[0], neutral_variance)

    # --- Project out and normalize ---
    probes = np.zeros_like(probes_raw)
    for i in range(171):
        projected = _project_out(probes_raw[i], pcs)
        n = np.linalg.norm(projected)
        if n < 1e-8:
            logger.warning("Degenerate probe for '%s' (norm ~ 0 after projection)", emotion_names[i])
            probes[i] = projected  # zero-ish; caller should flag
        else:
            probes[i] = projected / n

    norms_after = np.linalg.norm(probes_raw - (probes_raw @ pcs.T) @ pcs, axis=1)

    # --- Save ---
    safe_id = model_id.replace("/", "_").replace(":", "_").replace("\\", "_")
    save_path = out / f"{safe_id}_171.npz"

    ext_config = ExtractionConfig(
        model_id=model_id,
        layer=layer,
        stories_per_emotion=stories_per_emotion,
        neutral_pc_variance_threshold=neutral_variance,
        token_start_index=token_start,
        device=device_map,
        dtype=str(dtype),
        seed=seed,
        extra={
            "source": source,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_neutral_pcs": int(pcs.shape[0]),
            "total_stories_used": int(sum(story_counts)),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            # F4.0 — typical residual L2 norm at the target layer (single family).
            "residual_norm_typical": residual_norm_typical,
        },
    )

    np.savez(
        save_path,
        probes=probes.astype(np.float32),
        emotion_names=np.array(emotion_names),
        clusters=np.array(clusters),
        neutral_pcs=pcs.astype(np.float32),
        norms_before=norms_before.astype(np.float32),
        norms_after=norms_after.astype(np.float32),
        story_counts=np.array(story_counts, dtype=np.int32),
        metadata=np.array([ext_config.model_dump_json()]),
    )
    logger.info("Saved 171 probes to %s", save_path)

    # --- Cleanup ---
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start

    result = {
        "model_id": model_id,
        "source": source,
        "device": device_map,
        "dtype": str(dtype),
        "layer": layer,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_probes": 171,
        "num_neutral_pcs": int(pcs.shape[0]),
        "mean_norm_before": float(np.mean(norms_before)),
        "mean_norm_after": float(np.mean(norms_after)),
        "norm_reduction_ratio": float(np.mean(norms_after) / np.mean(norms_before))
        if np.mean(norms_before) > 0
        else 0.0,
        "cache_path": str(save_path),
        "total_time_s": round(total_time, 1),
    }

    logger.info("171-probe extraction complete in %.1fs. Norm reduction: %.3f",
                total_time, result["norm_reduction_ratio"])
    return result


# ============================================================================
# RESIDUUM F2.3.3 — Dual Probe Extraction (Present vs Other Speaker)
# ============================================================================
#
# The paper (Lindsey et al. 2026) documents two partially-orthogonal emotion
# direction families: "present speaker" (the emotion of whoever is currently
# producing tokens) and "other speaker" (the emotion of the interlocutor).
# F1.2 produced ONE library trained on solo stories. F2.3 produces TWO new
# libraries trained on dialogues where two characters express DISTINCT
# emotions in the same context (dialogues_171.json from F2.3.2).
#
# Per dialogue, each of the 4 turns contributes one mean-residual vector
# computed over the tokens belonging to that turn:
#   - turn 0 (P1, emotion A): sample for present[A] AND for other[B]
#   - turn 1 (P2, emotion B): sample for present[B] AND for other[A]
#   - turn 2 (P1, emotion A): sample for present[A] AND for other[B]
#   - turn 3 (P2, emotion B): sample for present[B] AND for other[A]
# This yields per emotion: ~2 samples per dialogue where it appears as P1
# plus ~2 samples per dialogue where it appears as P2, both for present
# and for other. With ~10 dialogue appearances per emotion (5 as P1 + ~5
# as P2), each family ends with ~20 samples per emotion.
#
# Neutral PC projection: reuses the PCs from the single library qwen3_4b_171.npz
# (F1.2 output) for consistency. Both families are projected against the
# same neutral subspace before normalization.


def _tokenize_dialogue_with_boundaries(
    tokenizer: Any,
    dialogue: dict[str, Any],
    max_seq_len: int = 512,
) -> tuple[Any, list[int]]:
    """Tokenize a full 4-turn dialogue and map each token to its turn index.

    Builds the text as:
      Person1: <turn 0 text>\\nPerson2: <turn 1 text>\\nPerson1: <turn 2 text>\\nPerson2: <turn 3 text>
    Tokens belonging to a turn's body (NOT the 'PersonX: ' prefix) are mapped
    to that turn's index. Special tokens, the speaker labels, and the newline
    separators map to -1.

    Args:
        tokenizer: HF fast tokenizer (supports return_offsets_mapping=True).
        dialogue: dict with key 'turns' = list of 4 {'speaker', 'text'} dicts.
        max_seq_len: truncation cap.

    Returns:
        (encoded_inputs, token_to_turn): encoded_inputs is the tokenizer
        output dict (with input_ids, attention_mask). token_to_turn is a
        list[int] of length seq_len mapping each token position to its
        turn index (0..3) or -1 if it belongs to a label/separator/special.
    """
    turns = dialogue["turns"]
    if len(turns) != 4:
        raise ValueError(f"Expected 4 turns, got {len(turns)}")

    # Build text incrementally and record character offsets where each turn's
    # body STARTS and ENDS (exclusive of trailing newline).
    text_parts: list[str] = []
    body_char_ranges: list[tuple[int, int]] = []
    cursor = 0
    for k, t in enumerate(turns):
        if k > 0:
            text_parts.append("\n")
            cursor += 1
        prefix = f"{t['speaker']}: "
        body = str(t["text"])
        text_parts.append(prefix)
        cursor += len(prefix)
        body_start = cursor
        text_parts.append(body)
        cursor += len(body)
        body_end = cursor
        body_char_ranges.append((body_start, body_end))

    full_text = "".join(text_parts)

    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    token_to_turn: list[int] = [-1] * len(offsets)
    for tok_idx, (s, e) in enumerate(offsets):
        if s == e:
            continue  # special token (BOS/EOS/PAD)
        for turn_idx, (ts, te) in enumerate(body_char_ranges):
            if ts <= s < te:
                token_to_turn[tok_idx] = turn_idx
                break
    return enc, token_to_turn


def _dual_turn_means(
    model: Any,
    tokenizer: Any,
    dialogue: dict[str, Any],
    layer: int,
    max_seq_len: int = 512,
) -> "list[Any] | None":
    """Compute the mean residual at `layer` for each of the 4 turns.

    Returns a list of 4 numpy arrays (one per turn) or None if any turn ends
    up with no tokens mapped to it after tokenization (e.g. extreme truncation).
    """
    import numpy as np
    import torch

    enc, token_to_turn = _tokenize_dialogue_with_boundaries(
        tokenizer, dialogue, max_seq_len=max_seq_len,
    )
    device = next(model.parameters()).device
    inputs = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states
    if layer + 1 >= len(hidden):
        raise ValueError(f"Layer {layer} out of range for {len(hidden) - 1} layers.")
    layer_h = hidden[layer + 1][0]  # (seq_len, hidden)

    results: list[Any] = []
    for turn_idx in range(4):
        positions = [i for i, t in enumerate(token_to_turn) if t == turn_idx]
        if not positions:
            return None
        mean_vec = layer_h[positions].mean(dim=0).float().cpu().numpy()
        results.append(np.asarray(mean_vec, dtype=np.float32))
    return results


def _accumulate_dual_samples(
    turn_means: list[Any],
    p1_emotion: str,
    p2_emotion: str,
    present_samples: dict[str, list[Any]],
    other_samples: dict[str, list[Any]],
) -> None:
    """Distribute the 4 turn means into present/other sample dicts per emotion.

    P1 turns (0, 2):
      - 'present' from P1's perspective: emotion = p1_emotion
      - 'other'   from P2's perspective: emotion = p2_emotion
    P2 turns (1, 3):
      - 'present' from P2's perspective: emotion = p2_emotion
      - 'other'   from P1's perspective: emotion = p1_emotion
    """
    for k, vec in enumerate(turn_means):
        if k in (0, 2):
            present_samples.setdefault(p1_emotion, []).append(vec)
            other_samples.setdefault(p2_emotion, []).append(vec)
        else:
            present_samples.setdefault(p2_emotion, []).append(vec)
            other_samples.setdefault(p1_emotion, []).append(vec)


def _compute_family_probes(
    emotion_names: list[str],
    samples_by_emotion: dict[str, list[Any]],
    neutral_pcs: Any,
) -> tuple[Any, Any, Any, Any, float]:
    """Aggregate per-emotion samples into normalized probes for one family.

    Args:
        emotion_names: ordered 171 emotion names.
        samples_by_emotion: dict name -> list of (hidden,) vectors.
        neutral_pcs: (n_pcs, hidden) basis projected out before normalization.

    Returns:
        (probes, norms_before, norms_after, sample_counts, residual_norm_typical):
          probes:        (171, hidden) unit-norm post-projection.
          norms_before:  norms of raw (mean-centered) probes before projection.
          norms_after:   norms after projecting out neutral_pcs.
          sample_counts: (171,) ints — samples available per emotion.
          residual_norm_typical: mean L2 norm of the raw residual samples
            (F4.0: used as the cap reference for MAX_STEERING_FRACTION).
    """
    import numpy as np

    hidden = neutral_pcs.shape[1] if neutral_pcs.ndim == 2 else 0
    if hidden == 0:
        # If neutral_pcs is empty, fall back to first sample's hidden size.
        for samples in samples_by_emotion.values():
            if samples:
                hidden = samples[0].shape[0]
                break
    if hidden == 0:
        raise RuntimeError("Cannot infer hidden size: no samples and no neutral_pcs.")

    emotion_means = np.zeros((len(emotion_names), hidden), dtype=np.float32)
    sample_counts = np.zeros((len(emotion_names),), dtype=np.int32)
    all_samples: list[Any] = []

    for i, name in enumerate(emotion_names):
        samples = samples_by_emotion.get(name, [])
        sample_counts[i] = len(samples)
        if not samples:
            continue
        emotion_means[i] = np.mean(np.stack(samples, axis=0), axis=0)
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("No samples accumulated across all emotions.")
    grand_mean = np.mean(np.stack(all_samples, axis=0), axis=0)

    probes_raw = emotion_means - grand_mean[np.newaxis, :]
    norms_before = np.linalg.norm(probes_raw, axis=1)

    probes = np.zeros_like(probes_raw)
    for i in range(len(emotion_names)):
        if sample_counts[i] == 0:
            continue
        projected = _project_out(probes_raw[i], neutral_pcs)
        n = np.linalg.norm(projected)
        if n < 1e-8:
            probes[i] = projected
        else:
            probes[i] = projected / n
    norms_after = np.linalg.norm(
        probes_raw - (probes_raw @ neutral_pcs.T) @ neutral_pcs, axis=1,
    )
    # F4.0 — typical residual norm at this layer for the dual family.
    residual_norm_typical = float(
        np.mean(np.linalg.norm(np.stack(all_samples, axis=0), axis=1))
    )
    return probes, norms_before, norms_after, sample_counts, residual_norm_typical


def extract_171_probes_dual(
    model_id: str,
    layer: int | None = None,
    device: str = "auto",
    max_seq_len: int = 512,
    seed: int = 42,
    output_dir: Path | None = None,
    single_library_path: Path | None = None,
    dialogues_path: Path | None = None,
) -> dict[str, Any]:
    """Extract present + other probes from dialogues_171.json.

    Pipeline:
      1. Load dialogues_171.json (855 dialogues, 171 emotions x 5 as P1).
      2. Load the single library NPZ (F1.2) for its neutral_pcs + emotion_names.
      3. For each dialogue, run the model once and compute mean residual
         per turn at the target layer.
      4. Distribute the 4 means into present/other sample dicts (see
         _accumulate_dual_samples for assignment rules).
      5. For each family separately:
         probe_raw[e] = mean(samples_e) - mean(all_samples_of_family)
         probe[e]     = normalize(project_out(probe_raw[e], neutral_pcs))
      6. Save two NPZ files alongside the single library:
           cached_vectors/<model>_171_present.npz
           cached_vectors/<model>_171_other.npz
         Same schema as the single library.

    Args:
        model_id: Model identifier (Ollama name, HF id, or local path).
        layer: Target layer. None = auto-select ~2/3 of depth.
        device: "auto" | "cpu" | "cuda".
        max_seq_len: Max tokens per dialogue (default 512 — fits a typical
            4-turn dialogue with margin).
        seed: For metadata only (extraction is deterministic).
        output_dir: Override cache directory.
        single_library_path: Override the single-library NPZ to read
            neutral_pcs from. Default: <output_dir>/<safe_id>_171.npz.
        dialogues_path: Override dialogues_171.json location.

    Returns:
        Dict with extraction metadata for both families.
    """
    try:
        import numpy as np
        import torch
    except ImportError as e:
        raise ImportError("Dual probe extraction requires numpy + torch.") from e

    from pathos.models.residuum import ExtractionConfig

    t_start = time.perf_counter()

    out = output_dir or _CACHED_VECTORS_DIR_171
    out.mkdir(parents=True, exist_ok=True)

    safe_id = model_id.replace("/", "_").replace(":", "_").replace("\\", "_")
    single_path = single_library_path or (out / f"{safe_id}_171.npz")
    if not single_path.is_file():
        raise FileNotFoundError(
            f"Single library NPZ not found at {single_path}. "
            "Run --extract-171 first (F1.2)."
        )

    dlg_path = dialogues_path or _DIALOGUES_171_PATH
    if not dlg_path.is_file():
        raise FileNotFoundError(
            f"dialogues_171.json not found at {dlg_path}. "
            "Run --generate-dialogues first (F2.3.1/F2.3.2)."
        )

    # --- Load neutral PCs + canonical emotion ordering from F1.2 library ---
    single_data = np.load(single_path, allow_pickle=False)
    neutral_pcs = np.asarray(single_data["neutral_pcs"], dtype=np.float32)
    emotion_names = [str(x) for x in single_data["emotion_names"]]
    clusters = [str(x) for x in single_data["clusters"]]
    if len(emotion_names) != 171:
        raise ValueError(
            f"Expected 171 emotion names in single library, got {len(emotion_names)}"
        )
    logger.info("Loaded neutral_pcs (%d x %d) from single library",
                neutral_pcs.shape[0], neutral_pcs.shape[1])

    # --- Load dialogues ---
    with dlg_path.open(encoding="utf-8") as f:
        dlg_data = json.load(f)
    dialogues = dlg_data.get("dialogues", [])
    if not dialogues:
        raise RuntimeError(f"No dialogues in {dlg_path}")
    logger.info("Loaded %d dialogues from %s", len(dialogues), dlg_path)

    # --- Load model ---
    model, tokenizer, source, dtype, device_map = _load_model_for_residuum(model_id, device)
    t_load = time.perf_counter()
    logger.info("Model loaded in %.1fs (source=%s, device=%s)",
                t_load - t_start, source, device_map)

    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    if num_layers <= 0 or hidden_size <= 0:
        raise RuntimeError("Cannot detect num_hidden_layers/hidden_size from model config.")

    if layer is None:
        layer = max(1, min(num_layers - 1, int(round(num_layers * 2 / 3))))
        logger.info("Auto-selected layer %d of %d (~2/3 depth)", layer, num_layers)
    elif not (0 <= layer < num_layers):
        raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}]")

    if neutral_pcs.shape[1] != hidden_size:
        raise ValueError(
            f"neutral_pcs hidden size {neutral_pcs.shape[1]} does not match model "
            f"hidden_size {hidden_size}. Single library was extracted from a "
            "different model — extract that first."
        )

    # --- Capture mean residuals per turn for every dialogue ---
    present_samples: dict[str, list[Any]] = {}
    other_samples: dict[str, list[Any]] = {}
    skipped = 0

    for di, d in enumerate(dialogues):
        p1 = d["person1_emotion"]
        p2 = d["person2_emotion"]
        try:
            turn_means = _dual_turn_means(model, tokenizer, d, layer, max_seq_len)
        except Exception as e:
            logger.warning("Dialogue %d (%s/%s) error: %s", d.get("id", di), p1, p2, e)
            skipped += 1
            continue
        if turn_means is None:
            logger.warning("Dialogue %d (%s/%s) had a turn with no tokens — skipping",
                           d.get("id", di), p1, p2)
            skipped += 1
            continue
        _accumulate_dual_samples(turn_means, p1, p2, present_samples, other_samples)
        if (di + 1) % 50 == 0 or (di + 1) == len(dialogues):
            logger.info("Dialogue capture: %d/%d", di + 1, len(dialogues))

    t_capture = time.perf_counter()
    logger.info("Dialogue residual capture done in %.1fs (skipped=%d)",
                t_capture - t_load, skipped)

    # --- Compute probes per family ---
    (
        present_probes, present_norms_before, present_norms_after,
        present_counts, present_residual_norm,
    ) = _compute_family_probes(emotion_names, present_samples, neutral_pcs)
    (
        other_probes, other_norms_before, other_norms_after,
        other_counts, other_residual_norm,
    ) = _compute_family_probes(emotion_names, other_samples, neutral_pcs)
    logger.info(
        "Residual norm typical (layer %d): present=%.3f, other=%.3f",
        layer, present_residual_norm, other_residual_norm,
    )

    # --- Save both families ---
    present_path = out / f"{safe_id}_171_present.npz"
    other_path = out / f"{safe_id}_171_other.npz"

    def _save_family(
        path: Path,
        family: str,
        probes: Any,
        norms_b: Any,
        norms_a: Any,
        counts: Any,
        residual_norm: float,
    ) -> None:
        ext_config = ExtractionConfig(
            model_id=model_id,
            layer=layer,
            # F1.2 schema repurposed: "stories_per_emotion" stores the
            # n_per_emotion target of the dialogue dataset (Person1 anchor),
            # which seeds the per-emotion sample budget.
            stories_per_emotion=5,
            neutral_pc_variance_threshold=0.0,  # reused from single library
            token_start_index=0,
            device=device_map,
            dtype=str(dtype),
            seed=seed,
            extra={
                "family": family,
                "source": source,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "num_neutral_pcs": int(neutral_pcs.shape[0]),
                "num_dialogues_used": int(len(dialogues) - skipped),
                "num_dialogues_skipped": int(skipped),
                "single_library_path": str(single_path),
                "dialogues_path": str(dlg_path),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                # F4.0 — typical residual L2 at the target layer for this family.
                "residual_norm_typical": residual_norm,
            },
        )
        np.savez(
            path,
            probes=probes.astype(np.float32),
            emotion_names=np.array(emotion_names),
            clusters=np.array(clusters),
            neutral_pcs=neutral_pcs.astype(np.float32),
            norms_before=norms_b.astype(np.float32),
            norms_after=norms_a.astype(np.float32),
            sample_counts=counts.astype(np.int32),
            metadata=np.array([ext_config.model_dump_json()]),
        )
        logger.info("Saved %s probes to %s (mean samples/emo=%.1f)",
                    family, path, float(counts.mean()))

    _save_family(present_path, "present", present_probes,
                 present_norms_before, present_norms_after, present_counts,
                 present_residual_norm)
    _save_family(other_path, "other", other_probes,
                 other_norms_before, other_norms_after, other_counts,
                 other_residual_norm)

    # --- Cleanup ---
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Orthogonality check (matched-emotion cosine between families) ---
    # The paper claims present[E] and other[E] are partially orthogonal.
    # We report the mean cosine across all 171 emotions for QA. Threshold
    # in tests is 0.5 (see RESIDUUMREWORK.txt L387-388).
    cosines = []
    for i in range(171):
        if present_counts[i] == 0 or other_counts[i] == 0:
            continue
        a = present_probes[i]
        b = other_probes[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > 1e-8 and nb > 1e-8:
            cosines.append(float(a @ b) / (na * nb))
    mean_cosine = float(np.mean(cosines)) if cosines else 0.0

    total_time = time.perf_counter() - t_start
    result = {
        "model_id": model_id,
        "source": source,
        "device": device_map,
        "dtype": str(dtype),
        "layer": layer,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_dialogues_used": int(len(dialogues) - skipped),
        "num_dialogues_skipped": int(skipped),
        "present_cache_path": str(present_path),
        "other_cache_path": str(other_path),
        "mean_samples_per_emotion_present": float(present_counts.mean()),
        "mean_samples_per_emotion_other": float(other_counts.mean()),
        "mean_cosine_present_vs_other_same_emotion": mean_cosine,
        "total_time_s": round(total_time, 1),
    }
    logger.info(
        "Dual probe extraction complete in %.1fs. Mean cosine(present[E], other[E]) = %.3f",
        total_time, mean_cosine,
    )
    return result


def main() -> None:
    """CLI entry point for steering vector extraction."""
    parser = argparse.ArgumentParser(
        description="Extract emotional steering vectors from a model (offline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --model qwen3:4b                    # From Ollama GGUF
  %(prog)s --model Qwen/Qwen3-4B --device cpu  # From HuggingFace on CPU
  %(prog)s --model qwen3:4b --layers 9 18 27   # Specific layers
  %(prog)s --list                               # Show available models
""",
    )
    parser.add_argument("--model", type=str, help="Model to extract from")
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device: auto (GPU if available), cpu, or cuda (default: auto)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Specific layer indices to extract (default: auto-select early/mid/late)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for extraction (reduce if OOM, default: 4)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=64,
        help="Max tokens per prompt (default: 64)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and their steering vector status",
    )
    # RESIDUUM F1.2 options
    parser.add_argument(
        "--generate-stories", action="store_true",
        help="[RESIDUUM F1.2] Generate N stories per emotion using the active LLM (Ollama)",
    )
    parser.add_argument(
        "--extract-171", action="store_true",
        help="[RESIDUUM F1.2] Extract 171 probes from stories (requires --generate-stories first)",
    )
    parser.add_argument(
        "--stories-per-emotion", type=int, default=15,
        help="Stories per emotion for F1.2 (default 15)",
    )
    parser.add_argument(
        "--generate-dialogues", action="store_true",
        help="[RESIDUUM F2.3] Generate N dialogues per emotion as Person1 "
             "(Person2 = distinct random emotion) using the active LLM",
    )
    parser.add_argument(
        "--dialogues-per-emotion", type=int, default=5,
        help="Dialogues per emotion (as Person1) for F2.3 (default 5 -> 855 total)",
    )
    parser.add_argument(
        "--extract-171-dual", action="store_true",
        help="[RESIDUUM F2.3.3] Extract present + other probes from "
             "dialogues_171.json (requires single library NPZ from --extract-171)",
    )
    parser.add_argument(
        "--ollama-model", type=str, default="qwen3:4b",
        help="Ollama model for story generation (default qwen3:4b)",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama base URL (default http://localhost:11434)",
    )
    parser.add_argument(
        "--residuum-layer", type=int, default=None,
        help="Layer for 171-probe extraction (default: ~2/3 depth)",
    )
    parser.add_argument(
        "--neutral-variance", type=float, default=0.50,
        help="Variance threshold for neutral PC projection (default 0.50)",
    )
    parser.add_argument(
        "--token-start", type=int, default=50,
        help="First token index in residual mean (skips preface, default 50)",
    )
    parser.add_argument(
        "--residuum-max-seq-len", type=int, default=512,
        help="Max tokens per story for 171-probe extraction (default 512)",
    )
    parser.add_argument(
        "--residuum-seed", type=int, default=42,
        help="Seed for story topic sampling (default 42)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list:
        models = list_available_models()
        if not models:
            print("No Ollama models found. Download models with: ollama pull <model>")
            return
        print(f"\n{'Model':<25} {'Status':<15} {'Details'}")
        print("-" * 60)
        for m in models:
            details = ""
            if m["status"] == "cached":
                details = f"{m['dimensions']} dims × {m['layers']} layers"
            print(f"{m['model_id']:<25} {m['status']:<15} {details}")
        return

    # --- RESIDUUM F1.2 flows ---
    if args.generate_stories:
        logger.info("Generating stories (model=%s, url=%s)...",
                    args.ollama_model, args.ollama_url)
        llm_call = _default_ollama_llm_call(
            base_url=args.ollama_url,
            model=args.ollama_model,
        )
        counts = generate_stories_for_emotions(
            n_per_emotion=args.stories_per_emotion,
            llm_call=llm_call,
            seed=args.residuum_seed,
        )
        print(f"\n{'='*50}")
        print("Story generation summary:")
        print(f"  Generated: {counts['generated']}")
        print(f"  Skipped (existing): {counts['skipped']}")
        print(f"  Failed: {counts['failed']}")
        print(f"  Target: {counts['total_target']}")
        print(f"{'='*50}")
        if not args.extract_171:
            return

    if args.extract_171_dual:
        if not args.model:
            print("Error: --extract-171-dual requires --model <id>")
            sys.exit(2)
        logger.info("Extracting dual probes (model=%s)...", args.model)
        result = extract_171_probes_dual(
            model_id=args.model,
            layer=args.residuum_layer,
            device=args.device,
            max_seq_len=args.residuum_max_seq_len,
            seed=args.residuum_seed,
        )
        print(f"\n{'='*50}")
        print("Dual probe extraction summary:")
        print(f"  Dialogues used:    {result['num_dialogues_used']}")
        print(f"  Dialogues skipped: {result['num_dialogues_skipped']}")
        print(f"  Mean samples/emo present: {result['mean_samples_per_emotion_present']:.1f}")
        print(f"  Mean samples/emo other:   {result['mean_samples_per_emotion_other']:.1f}")
        print(f"  Mean cosine(present[E], other[E]): {result['mean_cosine_present_vs_other_same_emotion']:.3f}")
        print(f"  Present NPZ: {result['present_cache_path']}")
        print(f"  Other NPZ:   {result['other_cache_path']}")
        print(f"  Total time:  {result['total_time_s']}s")
        print(f"{'='*50}")
        return

    if args.generate_dialogues:
        logger.info("Generating dialogues (model=%s, url=%s)...",
                    args.ollama_model, args.ollama_url)
        # Build LLM call with dialogue-mode strict defaults so qwen3:4b
        # does not razon-and-fail (system message + assistant 'Person1:'
        # prefix). The orchestrator only applies these when llm_call=None,
        # so the CLI must pass them explicitly.
        llm_call = _default_ollama_llm_call(
            base_url=args.ollama_url,
            model=args.ollama_model,
            temperature=0.7,
            num_predict=600,
            system_message=_DIALOGUE_SYSTEM_PROMPT,
            assistant_prefix=_DIALOGUE_ASSISTANT_PREFIX,
        )
        counts = generate_dialogues_for_emotions(
            n_per_emotion=args.dialogues_per_emotion,
            llm_call=llm_call,
            seed=args.residuum_seed,
        )
        print(f"\n{'='*50}")
        print("Dialogue generation summary:")
        print(f"  Generated: {counts['generated']}")
        print(f"  Skipped (existing): {counts['skipped']}")
        print(f"  Failed: {counts['failed']}")
        print(f"  Target: {counts['total_target']}")
        print(f"{'='*50}")
        return

    if args.extract_171:
        if not args.model:
            parser.error("--model is required for --extract-171")
        logger.info("Extracting 171 probes for '%s'...", args.model)
        result = extract_171_probes(
            model_id=args.model,
            layer=args.residuum_layer,
            device=args.device,
            stories_per_emotion=args.stories_per_emotion,
            neutral_variance=args.neutral_variance,
            token_start=args.token_start,
            max_seq_len=args.residuum_max_seq_len,
            seed=args.residuum_seed,
        )
        print(f"\n{'='*50}")
        print(f"171-probe extraction complete for: {result['model_id']}")
        print(f"  Layer:         {result['layer']} of {result['num_layers']}")
        print(f"  Hidden size:   {result['hidden_size']}")
        print(f"  Probes:        {result['num_probes']}")
        print(f"  Neutral PCs:   {result['num_neutral_pcs']}")
        print(f"  Norm before:   {result['mean_norm_before']:.3f}")
        print(f"  Norm after:    {result['mean_norm_after']:.3f}")
        print(f"  Reduction:     {result['norm_reduction_ratio']:.3f}")
        print(f"  Time:          {result['total_time_s']}s")
        print(f"  Cached at:     {result['cache_path']}")
        print(f"{'='*50}")
        return

    if not args.model:
        parser.error("--model is required (or use --list)")

    result = extract_and_cache(
        model_id=args.model,
        device=args.device,
        layers=args.layers,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    print(f"\n{'='*50}")
    print(f"Extraction complete for: {result['model_id']}")
    print(f"  Source:     {result['source']}")
    print(f"  Device:     {result['device']} ({result['dtype']})")
    print(f"  Model:      {result['num_layers']} layers × {result['hidden_size']} hidden")
    print(f"  Vectors:    {result['total_vectors']} ({len(result['dimensions'])} dims × {len(result['layers_extracted'])} layers)")
    print(f"  Layers:     {result['layers_extracted']}")
    print(f"  Time:       {result['load_time_s']}s load + {result['extract_time_s']}s extract = {result['total_time_s']}s total")
    print(f"  Cached at:  {result['cache_path']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
