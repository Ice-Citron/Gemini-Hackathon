#!/usr/bin/env python3
"""
SkyHammer LoRA Training Script
Fine-tunes Qwen 14B on RLAIF-collected security patching data.
Optimized for H200/H100 GPUs.
"""

import os
import json
import argparse
import torch
from datetime import datetime

# Check for required libraries
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("[!] Unsloth not available. Install with: pip install unsloth")

try:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("[!] TRL not available. Install with: pip install trl")

try:
    from peft import LoraConfig
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# Configuration
MODEL_NAME = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
OUTPUT_DIR = "checkpoints/skyhammer_14b_lora"
MAX_SEQ_LENGTH = 2048

# LoRA Configuration for Code Reasoning
LORA_CONFIG = {
    "r": 16,                # Higher rank for code tasks
    "lora_alpha": 32,       # Stability factor
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"       # FFN layers (critical for reasoning)
    ],
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
}

# Training Arguments (optimized for H200)
TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 60,        # Quick run for hackathon
    "learning_rate": 2e-4,
    "logging_steps": 1,
    "output_dir": "checkpoints",
    "optim": "adamw_8bit",
    "seed": 42,
    "save_steps": 30,
}


def load_dataset(data_path: str = "data/rlaif_14b_dataset.jsonl") -> Dataset:
    """Load the RLAIF dataset"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run rlaif_factory.py first.")

    # Load JSONL
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Format for chat training
                messages = entry.get("messages", [])
                if len(messages) >= 2:
                    # Convert to text format
                    text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
                    text += f"<|im_start|>assistant\n{messages[1]['content']}<|im_end|>"
                    data.append({"text": text, "score": entry.get("score", 0.5)})

    print(f"Loaded {len(data)} training examples")
    return Dataset.from_list(data)


def train_model(
    dataset: Dataset,
    model_name: str = MODEL_NAME,
    output_dir: str = OUTPUT_DIR,
    max_steps: int = 60
):
    """Train the LoRA adapter"""
    if not HAS_UNSLOTH or not HAS_TRL:
        raise RuntimeError("Required libraries not available. Install unsloth and trl.")

    print(f"\nLoading model: {model_name}")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapter
    print("Adding LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing=LORA_CONFIG["use_gradient_checkpointing"],
    )

    # Detect hardware capabilities
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    print(f"Hardware: bf16={use_bf16}, fp16={use_fp16}")

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
        warmup_steps=TRAINING_ARGS["warmup_steps"],
        max_steps=max_steps,
        learning_rate=TRAINING_ARGS["learning_rate"],
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=TRAINING_ARGS["logging_steps"],
        output_dir=TRAINING_ARGS["output_dir"],
        optim=TRAINING_ARGS["optim"],
        seed=TRAINING_ARGS["seed"],
        save_steps=TRAINING_ARGS["save_steps"],
        report_to="none",  # Disable wandb for now
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print(f"\nStarting training ({max_steps} steps)...")
    start_time = datetime.now()

    trainer.train()

    duration = datetime.now() - start_time
    print(f"\nTraining complete in {duration}")

    # Save model
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "model_name": model_name,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "training_duration": str(duration),
        "dataset_size": len(dataset),
        "lora_config": LORA_CONFIG,
        "trained_at": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print("Training info saved to: training_info.json")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SkyHammer LoRA Training")
    parser.add_argument("--data", type=str, default="data/rlaif_14b_dataset.jsonl", help="Dataset path")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Base model")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_steps", type=int, default=60, help="Max training steps")

    args = parser.parse_args()

    # Check requirements
    if not HAS_UNSLOTH:
        print("\n[ERROR] Unsloth not installed!")
        print("Install with: pip install unsloth")
        return

    if not HAS_TRL:
        print("\n[ERROR] TRL not installed!")
        print("Install with: pip install trl")
        return

    # Load dataset
    try:
        dataset = load_dataset(args.data)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run 'python src/rlaif_factory.py' first to generate training data.")
        return

    if len(dataset) == 0:
        print("\n[ERROR] Dataset is empty!")
        return

    # Train
    train_model(
        dataset=dataset,
        model_name=args.model,
        output_dir=args.output,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()
