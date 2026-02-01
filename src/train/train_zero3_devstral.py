import os
import gc
import torch
import torch.distributed as dist
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

# --- CONFIGURATION ---
# MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512"

DATASET_HF = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_fsdp"
MAX_LENGTH = 4096

def format_example(example):
    prompt = f"VULNERABILITY: {example['vulnerability_type']}\nCODE:\n{example['vulnerable_code']}\nFIX THIS."
    response = f"SECURE CODE:\n{example['secure_code']}\nREASONING:\n{example['reasoning']}"
    return {"text": f"[INST] {prompt} [/INST] {response}"}

def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank == 0:
        print(f"--- [GPU {rank}] Initializing FSDP Training (Size-Based Wrapping) ---")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # 1. PRE-DOWNLOAD
    if rank == 0:
        try:
            snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.ggml", "*.bin"])
        except:
            pass
    if world_size > 1:
        dist.barrier()

    # 2. LOAD CONFIG
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.torch_dtype = torch.bfloat16

    # 3. LOAD BASE MODEL (No PEFT wrapping here!)
    print(f"--- [GPU {rank}] Loading base model ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Critical settings for FSDP
    model.config.use_cache = False 
    model.gradient_checkpointing_enable() 

    # 4. TOKENIZER
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    except:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 5. DATASET
    dataset = load_dataset(DATASET_HF, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # 6. LORA CONFIG (Passed to Trainer, NOT model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 7. TRAINING ARGS
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        ddp_timeout=7200,
    )

    # 8. TRAINER (The Fix)
    # SFTTrainer will wrap the model with LoRA *inside* the FSDP context safely.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config, # <--- THIS PREVENTS THE DEADLOCK
    )

    print(f"--- [GPU {rank}] Starting Training ---")
    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()