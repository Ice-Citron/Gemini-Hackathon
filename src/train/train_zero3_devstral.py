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

# --- LOGGING & HUB ---
WANDB_PROJECT = "SkyHammer-Gemini-Hack"
WANDB_ENTITY = "Imperial-College-London-SPQR"
HF_MODEL_ID = "shng2025/SkyHammer-123B-devstral2-v1"


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
        print(f"--- W&B: {WANDB_ENTITY}/{WANDB_PROJECT} ---")
        print(f"--- HF Hub: {HF_MODEL_ID} ---")

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
    if rank == 0:
        print("[Rank 0] Loading full model into CPU RAM...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        print("[Rank 0] Model loaded successfully!")
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

    accelerator.wait_for_everyone()

    if rank == 0:
        print("--- Model ready for FSDP sharding ---")

    model.gradient_checkpointing_enable()

    gc.collect()
    torch.cuda.empty_cache()

    # 5. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    except:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # --- SAFETY CHECK: VOCABULARY MISMATCH ---
    vocab_size_model = model.config.vocab_size
    vocab_size_tokenizer = len(tokenizer)
    
    if rank == 0:
        print(f"--- VOCAB CHECK ---")
        print(f"Model Vocab: {vocab_size_model}")
        print(f"Tokenizer Vocab: {vocab_size_tokenizer}")
    
    if vocab_size_tokenizer > vocab_size_model:
        if rank == 0:
            print("!!! CRITICAL WARNING: Tokenizer is larger than Model !!!")
            print("Resizing model embeddings to avoid NaN crash...")
        
        # This expands the model's input layer to handle the extra tokens
        # (FSDP will handle the synchronization of this)
        model.resize_token_embeddings(vocab_size_tokenizer)
        
        # We must ensure the new embeddings require gradients so LoRA works
        model.get_input_embeddings().requires_grad_(True)
    # -----------------------------------------



    # 6. Dataset
    dataset = load_dataset(DATASET_HF, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    if rank == 0:
        print(f"--- Dataset: {len(dataset)} examples ---")

    # 7. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 8. Training Args
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        packing=False,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        # STABILITY
        learning_rate=5e-7,
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        weight_decay=0.01,

        num_train_epochs=10,
        bf16=True,

        # FRONTIER LAB TRICKS
        adam_beta2=0.95,                # "Forget" the explosion faster
        adam_epsilon=1e-8,              # Standard stability

        # LOGGING - W&B
        logging_steps=1,
        report_to="wandb",
        run_name="skyhammer-123b-sft-v1",

        # CHECKPOINTING
        save_strategy="steps",
        save_steps=50,

        # HUGGINGFACE HUB
        push_to_hub=True,
        hub_model_id=HF_MODEL_ID,
        hub_strategy="checkpoint",  # Push at each checkpoint

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        ddp_timeout=7200,
    )

    # 9. Set W&B environment (rank 0 only logs)
    if rank == 0:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT
        os.environ["WANDB_ENTITY"] = WANDB_ENTITY
        os.environ["WANDB_WATCH"] = "false"  # Optional: Prevents logging gradients (saves RAM)

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

    # 11. Save & Push
    if rank == 0:
        print(f"--- Saving to {OUTPUT_DIR} ---")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Final push to hub
        print(f"--- Pushing to HuggingFace: {HF_MODEL_ID} ---")
        trainer.push_to_hub(commit_message="Final SFT checkpoint")
        print("--- Done! ---")


if __name__ == "__main__":
    main()
