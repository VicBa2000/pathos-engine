"""Curated catalog of recommended models for Pathos Engine.

Organized by category with VRAM estimates at Q4_K_M / 4096 context.
Target hardware: GTX 1660 Super 6GB.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeaturedModel:
    name: str
    description: str
    size: str
    vram_estimate: str
    category: str


FEATURED_MODELS: list[FeaturedModel] = [
    # --- Small & Fast (< 2GB VRAM) ---
    FeaturedModel("qwen3:1.7b", "Fast reasoning, multilingual", "1.1GB", "~1.2GB", "Small & Fast"),
    FeaturedModel("llama3.2:1b", "Meta's compact model", "1.3GB", "~1.0GB", "Small & Fast"),
    FeaturedModel("gemma3:1b", "Google's lightweight model", "815MB", "~1.0GB", "Small & Fast"),
    FeaturedModel("phi4-mini:3.8b", "Microsoft reasoning, compact", "2.5GB", "~1.8GB", "Small & Fast"),
    FeaturedModel("smollm2:1.7b", "HuggingFace's small model", "1.0GB", "~1.2GB", "Small & Fast"),

    # --- Balanced (2-4GB VRAM) ---
    FeaturedModel("qwen3:4b", "Best balance for Pathos (default)", "2.6GB", "~2.5GB", "Balanced"),
    FeaturedModel("llama3.2:3b", "Meta's mid-size, great quality", "2.0GB", "~2.0GB", "Balanced"),
    FeaturedModel("gemma3:4b", "Google's balanced model", "3.0GB", "~2.5GB", "Balanced"),
    FeaturedModel("phi4-mini", "Microsoft reasoning, efficient", "2.5GB", "~2.5GB", "Balanced"),
    FeaturedModel("mistral:7b", "Mistral's classic 7B", "4.1GB", "~4.0GB", "Balanced"),

    # --- Large & Powerful (4-6GB VRAM) ---
    FeaturedModel("gemma3:12b", "Google's large, high quality", "8.1GB", "~5.5GB", "Large & Powerful"),
    FeaturedModel("llama3.1:8b", "Meta's 8B, strong reasoning", "4.7GB", "~4.5GB", "Large & Powerful"),
    FeaturedModel("qwen3:8b", "Alibaba's 8B, multilingual", "5.2GB", "~4.8GB", "Large & Powerful"),
    FeaturedModel("deepseek-r1:7b", "DeepSeek reasoning model", "4.7GB", "~4.5GB", "Large & Powerful"),
    FeaturedModel("mistral-small:24b", "Mistral's best (needs >6GB)", "15GB", "~8GB", "Large & Powerful"),

    # --- Code ---
    FeaturedModel("qwen2.5-coder:3b", "Alibaba code specialist", "1.9GB", "~1.8GB", "Code"),
    FeaturedModel("qwen2.5-coder:7b", "Strong code generation", "4.7GB", "~4.5GB", "Code"),
    FeaturedModel("deepseek-coder-v2:16b", "DeepSeek code v2", "8.9GB", "~5.5GB", "Code"),
    FeaturedModel("codellama:7b", "Meta's code LLaMA", "3.8GB", "~3.5GB", "Code"),
    FeaturedModel("starcoder2:3b", "BigCode's coder", "1.7GB", "~1.5GB", "Code"),

    # --- Vision ---
    FeaturedModel("llava:7b", "Vision + language (image input)", "4.7GB", "~4.5GB", "Vision"),
    FeaturedModel("llava:13b", "Larger vision model", "8.0GB", "~6GB+", "Vision"),
    FeaturedModel("moondream:1.8b", "Tiny vision model", "1.7GB", "~1.5GB", "Vision"),
    FeaturedModel("bakllava:7b", "BakLLaVA vision", "4.7GB", "~4.5GB", "Vision"),
    FeaturedModel("llava-phi3:3.8b", "Phi3 + vision, compact", "2.9GB", "~2.5GB", "Vision"),
]

CATEGORIES = ["Small & Fast", "Balanced", "Large & Powerful", "Code", "Vision"]


def get_featured_models() -> list[dict]:
    """Return featured models as dicts for API response."""
    return [
        {
            "name": m.name,
            "description": m.description,
            "size": m.size,
            "vram_estimate": m.vram_estimate,
            "category": m.category,
        }
        for m in FEATURED_MODELS
    ]
