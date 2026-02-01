"""
SkyHammer FSDP Training - 123B Model
====================================

FIX: Load model on rank 0 only, then FSDP shards to other GPUs.
This prevents 8x memory usage during loading.

Usage:
    accelerate launch src/train/train_zero3_devstral.py
"""

import os
import gc
import torch
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
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512"
DATASET_HF = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_fsdp"
MAX_LENGTH = 4096


def format_example(example):
    prompt = f"VULNERABILITY: {example['vulnerability_type']}\nCODE:\n{example['vulnerable_code']}\nFIX THIS."
    response = f"SECURE CODE:\n{example['secure_code']}\nREASONING:\n{example['reasoning']}"
    return {"text": f"[INST] {prompt} [/INST] {response}"}


def main():
    # 1. Initialize Accelerator
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if rank == 0:
        print(f"--- Initializing FSDP Training for {MODEL_NAME} ---")
        print(f"--- World size: {world_size} GPUs ---")

    # 2. Pre-download on rank 0 only
    if rank == 0:
        print("[Rank 0] Pre-downloading model files...")
        try:
            snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.ggml", "*.bin"])
        except Exception as e:
            print(f"Download note: {e}")

    accelerator.wait_for_everyone()

    # 3. Load Config (all ranks - this is small)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.torch_dtype = torch.bfloat16

    # 4. LOAD MODEL - RANK 0 ONLY
    # This is the critical fix: only rank 0 loads the full model
    # FSDP will broadcast shards to other ranks

    if rank == 0:
        print("[Rank 0] Loading full model into CPU RAM...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Load to CPU, FSDP will shard to GPUs
        )
        print("[Rank 0] Model loaded successfully!")
    else:
        # Other ranks: create empty model structure (no weights loaded)
        print(f"[Rank {rank}] Creating empty model skeleton...")
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    model.config.use_cache = False

    # 5. Sync model state from rank 0 to all other ranks
    # FSDP will handle this when we pass sync_module_states=True
    accelerator.wait_for_everyone()

    if rank == 0:
        print("--- Model ready for FSDP sharding ---")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    gc.collect()
    torch.cuda.empty_cache()

    # 6. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    except:
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
        print(f"--- Dataset: {len(dataset)} examples ---")

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

        per_device_train_batch_size=1, # changed from 1 to 2 for greater A100 utilisation
        gradient_accumulation_steps=1,

        # STABILITY: Lower LR + gradient clipping to prevent explosion
        learning_rate=2e-6,        # Drastically lowered (was 1e-5)
        warmup_ratio=0.1,          # 10% warmup to gently wake up the weights
        max_grad_norm=0.3,         # Clamp gradients hard (default is 1.0)
        weight_decay=0.01,         # Standard stability regularization
        num_train_epochs=1,

        num_train_epochs=1,
        bf16=True,

        logging_steps=1,
        save_strategy="steps",
        save_steps=50,

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
        print("--- Starting Training ---")

    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        print(f"--- Saved to {OUTPUT_DIR} ---")


if __name__ == "__main__":
    main()
