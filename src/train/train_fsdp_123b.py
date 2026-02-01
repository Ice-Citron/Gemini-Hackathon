"""
SkyHammer 123B - The "Hail Mary" (QLoRA)
========================================

STRATEGY: 4-bit Loading + BF16 Adapter.
WHY:      1. Loads in < 5 mins (Bypasses the "Hang").
          2. Fits in RAM (Bypasses the "OOM").
          3. Produces a standard BF16 Adapter (vLLM Compatible).
"""

import os
import torch
import gc
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizerFast, 
    AutoConfig,
    BitsAndBytesConfig
)
from huggingface_hub import snapshot_download
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator

# --- GLOBAL LOGGING ---
os.environ["WANDB_PROJECT"] = "SkyHammer-Gemini-Hack"
os.environ["WANDB_ENTITY"] = "Imperial-College-London-SPQR"
os.environ["WANDB_WATCH"] = "false"

# --- CONFIGURATION ---
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512" 
DATASET_HF = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_123b_qlora"
MAX_LENGTH = 4096
HF_MODEL_ID = "shng2025/SkyHammer-123B-devstral2-v1"

def format_example(example):
    prompt = f"VULNERABILITY: {example['vulnerability_type']}\nCODE:\n{example['vulnerable_code']}\nFIX THIS."
    response = f"SECURE CODE:\n{example['secure_code']}\nREASONING:\n{example['reasoning']}"
    return {"text": f"[INST] {prompt} [/INST] {response}"}

def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    
    if rank == 0:
        print(f"--- SkyHammer 123B QLoRA Rescue ---")
        try:
            snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.ggml", "*.bin"])
        except Exception as e:
            print(f"Download note: {e}")

    accelerator.wait_for_everyone()

    # --- 1. CONFIG SURGERY (Fixes A100 Crash) ---
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        if rank == 0: print("--- Removing FP8 Requirement (A100 Fix) ---")
        delattr(config, "quantization_config")
    
    # --- 2. 4-BIT QUANTIZATION (Fixes RAM/Hang) ---
    # This shrinks the model from 250GB -> 70GB. 
    # It loads instantly on all GPUs in parallel.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 3. TOKENIZER ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    except:
        if rank == 0: print("Using Fallback Tokenizer")
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4. DATASET ---
    dataset = load_dataset(DATASET_HF, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # --- 5. LORA CONFIG ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --- 6. TRAINING ARGS ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        packing=False,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 

        learning_rate=2e-4,       # QLoRA likes higher LR
        warmup_ratio=0.1,
        max_grad_norm=0.3,        
        weight_decay=0.01,
        
        num_train_epochs=10,
        bf16=True,

        logging_steps=1,
        report_to="wandb",
        run_name="skyhammer-123b-qlora-rescue",

        save_strategy="steps",
        save_steps=10,
        push_to_hub=True,
        hub_model_id=HF_MODEL_ID,
        hub_strategy="checkpoint",
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_timeout=7200,
        
        # --- THE MAGIC LOADING CONFIG ---
        model_init_kwargs={
            "config": config,          # Sanitzed (No FP8)
            "quantization_config": bnb_config, # 4-bit (Fast Load)
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
    )

    # --- 7. TRAINER ---
    trainer = SFTTrainer(
        model=MODEL_NAME,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    if rank == 0:
        print("--- Starting Training (Should load in < 5 mins) ---")

    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("--- Done! ---")

if __name__ == "__main__":
    main()