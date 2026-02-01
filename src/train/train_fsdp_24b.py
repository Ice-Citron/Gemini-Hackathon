"""
SkyHammer FSDP Training - 24B Model (Verification Run)
======================================================

Purpose: Verify the FSDP "Rank 0 Loading" pipeline on a smaller model.
Logic:   Identical to 123B script, just different Model ID.

Usage:
    accelerate launch src/train/train_fsdp_24b.py
"""

import os
import gc
import torch

# --- 1. GLOBAL LOGGING (Must be at top) ---
os.environ["WANDB_PROJECT"] = "SkyHammer-Gemini-Hack"
os.environ["WANDB_ENTITY"] = "Imperial-College-London-SPQR"
os.environ["WANDB_WATCH"] = "false"

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM
)
from huggingface_hub import snapshot_download
from peft import LoraConfig
from accelerate import Accelerator

# --- CONFIGURATION ---
# FIX: Pointing to the actual 24B model now
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501" 
DATASET_HF = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_24b_test"
MAX_LENGTH = 4096
HF_MODEL_ID = "shng2025/SkyHammer-24B-mistral-v1"


def format_example(example):
    """Format into Mistral instruct format."""
    prompt = (
        f"You are a security expert. Fix the following {example['vulnerability_type']} vulnerability.\n\n"
        f"## Vulnerable Code\n```python\n{example['vulnerable_code']}\n```\n\n"
        "Provide the secure version and explain your fix."
    )
    response = (
        f"## Secure Code\n```python\n{example['secure_code']}\n```\n\n"
        f"## Reasoning\n{example['reasoning']}"
    )
    return {"text": f"[INST] {prompt} [/INST] {response}"}


def main():
    # 2. Initialize Accelerator
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if rank == 0:
        print(f"--- SkyHammer Verification Run (24B) ---")
        print(f"Model: {MODEL_NAME}")
        print(f"GPUs: {world_size}")

    # 3. Pre-download (Rank 0 only)
    if rank == 0:
        print("[Rank 0] Pre-downloading model files...")
        try:
            snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.ggml", "*.bin"])
        except Exception as e:
            print(f"Download note: {e}")

    accelerator.wait_for_everyone()

    # 4. Load Config
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.torch_dtype = torch.bfloat16

    # 5. LOAD MODEL (Rank 0 Logic - True Verification)
    # We use the same logic as 123B to verify the FSDP sharding works.
    if rank == 0:
        print(f"[Rank 0] Loading {MODEL_NAME} into CPU RAM...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu", # Important for Rank 0 loading
        )
        print("[Rank 0] Model loaded!")
    else:
        print(f"[Rank {rank}] Creating empty model skeleton...")
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Syncs the weights from Rank 0 to others
    accelerator.wait_for_everyone()
    
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    except Exception as e:
        if rank == 0: print(f"Tokenizer fallback: {e}")
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 7. Dataset
    dataset = load_dataset(DATASET_HF, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    if rank == 0:
        print(f"Dataset size: {len(dataset)} examples")

    # 8. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 9. Training Args
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        packing=False,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, 

        learning_rate=2e-5, # Standard LR for 24B
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.01,
        
        num_train_epochs=3,
        bf16=True,

        logging_steps=1,
        report_to="wandb",
        run_name="skyhammer-24b-verify",

        save_strategy="steps",
        save_steps=50,

        push_to_hub=True,
        hub_model_id=HF_MODEL_ID,
        hub_strategy="checkpoint",

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_timeout=7200,
    )

    # 10. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    if rank == 0:
        print("Starting training...")

    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("TRAINING COMPLETE!")

if __name__ == "__main__":
    main()