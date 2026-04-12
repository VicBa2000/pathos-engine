"""Pathos Training — offline fine-tuning infrastructure.

Generates emotional dialogue datasets and trains QLoRA adapters
for emotional conditioning tokens (5.3b) and emotional adapters (5.2b).

Requires: peft, trl, datasets, bitsandbytes (optional, for 4-bit quantization)
Hardware: 24GB+ VRAM recommended (RTX 3090/4090, A100)

Usage:
  python -m pathos.training.generate_dataset --output training_data/
  python -m pathos.training.fine_tune --model qwen3:4b --dataset training_data/emotional.jsonl
"""
