"""Steering Vector Extraction Utility — Offline batch extraction.

Extracts emotional direction vectors from a model and caches them for
runtime steering. This is an OFFLINE operation — run once per model,
then the cached vectors are loaded automatically at startup.

Usage as script:
    python -m pathos.engine.steering_extract --model qwen3:4b
    python -m pathos.engine.steering_extract --model Qwen/Qwen3-4B --device cpu
    python -m pathos.engine.steering_extract --model /path/to/model.gguf

Usage from code:
    from pathos.engine.steering_extract import extract_and_cache
    result = extract_and_cache("qwen3:4b", device="auto")

The script:
1. Locates the model (Ollama GGUF, HuggingFace, or local path)
2. Loads it via transformers (dequantized to fp16/fp32)
3. Runs contrastive pairs through the model
4. Extracts hidden states at early/mid/late layers
5. Computes direction vectors (mean difference)
6. Saves to src/pathos/steering_data/cached_vectors/<model>.npz
7. Unloads the model to free memory

Typical runtime: 2-5 min on GPU, 10-30 min on CPU (one-time cost).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    load_path = model_id
    gguf_file: str | None = None
    source = "huggingface"

    # Check if it's an Ollama model name
    if ":" in model_id and "/" not in model_id:
        gguf_path = find_ollama_gguf(model_id)
        if gguf_path is not None:
            load_path = str(gguf_path.parent)
            gguf_file = gguf_path.name
            source = "ollama_gguf"
            logger.info("Found Ollama GGUF: %s", gguf_path)
        else:
            # Try mapping to HuggingFace ID
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
    if device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"  # Split GPU/CPU as needed
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32
    elif device == "cpu":
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
