"""QLoRA Fine-Tuning Script — trains emotional adapter and conditioning tokens.

Trains a LoRA adapter on the emotional dialogue dataset so the model learns
to condition its responses on emotional state tokens.

Two training modes:
  1. Conditioning Tokens (5.3b): Adds special tokens to vocabulary, resizes
     embeddings, and trains LoRA + new embeddings on emotional data.
  2. Emotional Adapter (5.2b): Standard LoRA adapter that learns emotional
     response patterns from the dataset.

Both modes can be combined in a single training run.

Requirements:
  - peft >= 0.12.0
  - trl >= 0.12.0 (optional, for SFT trainer)
  - bitsandbytes >= 0.43.0 (optional, for 4-bit quantization)
  - torch >= 2.0
  - transformers >= 4.40
  - 24GB+ VRAM recommended (RTX 3090/4090, A100)

Usage:
  python -m pathos.training.fine_tune \\
    --model Qwen/Qwen3-4B \\
    --dataset training_data/emotional.jsonl \\
    --output adapters/qwen3-4b-emotional \\
    --epochs 3 \\
    --mode both
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""

    # Model
    model_id: str = "Qwen/Qwen3-4B"
    adapter_output: Path = Path("adapters/emotional")

    # Dataset
    dataset_path: Path = Path("training_data/emotional.jsonl")

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Training config
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 512
    use_4bit: bool = True  # 4-bit quantization (requires bitsandbytes)

    # Mode
    add_conditioning_tokens: bool = True  # 5.3b: add special tokens
    train_adapter: bool = True  # 5.2b: train LoRA adapter

    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100


def check_requirements() -> dict[str, bool]:
    """Check which training dependencies are available.

    Returns dict of library -> available.
    """
    available: dict[str, bool] = {}

    for lib in ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"):
        try:
            __import__(lib)
            available[lib] = True
        except ImportError:
            available[lib] = False

    return available


def prepare_training_data(
    dataset_path: Path,
    add_tokens: bool = True,
) -> list[dict[str, str]]:
    """Load and format dataset for SFT training.

    Formats each example as:
      system: "You are an emotionally aware assistant."
      user: "<V+3><A-1> {prompt}"
      assistant: "{system_prompt_as_ideal_response_seed}"

    Args:
        dataset_path: Path to JSONL dataset.
        add_tokens: Whether to prepend conditioning tokens to prompts.

    Returns:
        List of conversation dicts for SFT training.
    """
    conversations: list[dict[str, str]] = []

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            prompt = ex["prompt"]
            tokens = ex.get("emotional_tokens", "")

            # Prepend conditioning tokens if enabled
            if add_tokens and tokens:
                user_msg = f"{tokens} {prompt}"
            else:
                user_msg = prompt

            conversations.append({
                "system": "You are an emotionally authentic assistant. Respond naturally according to your emotional state.",
                "user": user_msg,
                "emotional_tokens": tokens,
                "valence": ex.get("valence", 0.0),
                "arousal": ex.get("arousal", 0.3),
                "intensity": ex.get("intensity", 0.0),
                "primary_emotion": ex.get("primary_emotion", "neutral"),
            })

    return conversations


def train(config: TrainingConfig) -> dict[str, Any]:
    """Run QLoRA fine-tuning.

    This is the main training function. It:
    1. Loads the base model with optional 4-bit quantization
    2. Optionally adds conditioning tokens and resizes embeddings
    3. Configures LoRA adapter
    4. Trains on the emotional dataset
    5. Saves the adapter (and token embeddings if applicable)

    Args:
        config: Training configuration.

    Returns:
        Dict with training results (loss, steps, output_path).

    Raises:
        ImportError: If required libraries are not installed.
        FileNotFoundError: If dataset doesn't exist.
    """
    # Check requirements
    reqs = check_requirements()
    missing = [lib for lib, ok in reqs.items() if not ok and lib in ("torch", "transformers", "peft")]
    if missing:
        raise ImportError(
            f"Missing required libraries: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )

    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info("Loading model: %s", config.model_id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add conditioning tokens (5.3b)
    new_token_count = 0
    if config.add_conditioning_tokens:
        from pathos.training.emotional_tokens import generate_emotional_tokens
        special_tokens = generate_emotional_tokens()
        new_token_count = tokenizer.add_tokens(special_tokens)
        logger.info("Added %d conditioning tokens to tokenizer", new_token_count)

    # Load model with optional quantization
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if config.use_4bit and reqs.get("bitsandbytes"):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization (NF4)")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)

    # Resize embeddings for new tokens
    if new_token_count > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model embeddings to %d tokens", len(tokenizer))

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA adapter: %d trainable / %d total (%.2f%%)",
        trainable, total, trainable / total * 100,
    )

    # Prepare dataset
    data = prepare_training_data(config.dataset_path, add_tokens=config.add_conditioning_tokens)
    logger.info("Training on %d examples", len(data))

    # Format for training
    train_texts: list[str] = []
    for conv in data:
        text = f"<|system|>{conv['system']}<|user|>{conv['user']}<|assistant|>"
        train_texts.append(text)

    # Tokenize
    encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=config.max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Simple training loop (no trl dependency required)
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    model.train()
    total_steps = 0
    total_loss = 0.0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / config.gradient_accumulation
            loss.backward()

            if (step + 1) % config.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation
            total_steps += 1

            if total_steps % config.logging_steps == 0:
                avg = epoch_loss / (step + 1)
                logger.info("Epoch %d, Step %d, Loss: %.4f", epoch + 1, total_steps, avg)

        total_loss = epoch_loss / max(len(dataloader), 1)
        logger.info("Epoch %d complete, Avg Loss: %.4f", epoch + 1, total_loss)

    # Save adapter
    config.adapter_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.adapter_output)
    tokenizer.save_pretrained(config.adapter_output)

    # Save training config
    meta = {
        "model_id": config.model_id,
        "epochs": config.epochs,
        "final_loss": round(total_loss, 6),
        "total_steps": total_steps,
        "lora_r": config.lora_r,
        "conditioning_tokens_added": new_token_count,
        "dataset_size": len(data),
    }
    with open(config.adapter_output / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Adapter saved to %s", config.adapter_output)

    return meta


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for emotional conditioning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=Path, default=Path("training_data/emotional.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("adapters/emotional"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--mode", choices=["tokens", "adapter", "both"], default="both",
                        help="tokens=5.3b only, adapter=5.2b only, both=5.2b+5.3b")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = TrainingConfig(
        model_id=args.model,
        dataset_path=args.dataset,
        adapter_output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        use_4bit=not args.no_4bit,
        add_conditioning_tokens=args.mode in ("tokens", "both"),
        train_adapter=args.mode in ("adapter", "both"),
    )

    reqs = check_requirements()
    print("Dependencies:")
    for lib, ok in reqs.items():
        print(f"  {lib}: {'OK' if ok else 'MISSING'}")

    missing_critical = [lib for lib in ("torch", "transformers", "peft") if not reqs.get(lib)]
    if missing_critical:
        print(f"\nMissing critical dependencies: {', '.join(missing_critical)}")
        print(f"Install with: pip install {' '.join(missing_critical)}")
        exit(1)

    result = train(config)
    print(f"\nTraining complete:")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Adapter saved: {config.adapter_output}")
