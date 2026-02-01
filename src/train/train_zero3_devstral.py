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
from accelerate import Accelerator  # <--- WE IMPORT THIS NOW

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
    # 1. Initialize Accelerator MANUALLY to control memory loading
    accelerator = Accelerator()
    rank = accelerator.process_index  # Use accelerator's rank detection
    
    if rank == 0:
        print(f"--- [GPU {rank}] Initializing FSDP Training (Memory Optimized) ---")

    # 2. Pre-download (Rank 0 only)
    if rank == 0:
        try:
            snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.ggml", "*.bin"])
        except:
            pass
    accelerator.wait_for_everyone() # Replaces dist.barrier()

    # 3. Load Config
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.torch_dtype = torch.bfloat16
    
    # 4. LOAD MODEL (The Critical Fix)
    # We let SFTTrainer handle the loading internally, but we set the config right.
    # We do NOT manually load model = AutoModel... here anymore if we want FSDP to work perfectly with low_mem.
    # However, to be 100% sure we don't OOM, we pass the model_id string to SFTTrainer
    # and let it use the accelerator to lazy-load.
    
    # BUT, since we need specific configs (no cache, etc), we load with the empty_init context.
    print(f"--- [GPU {rank}] Loading Model Skeleton... ---")
    
    # This context manager forces the model to load as "Empty" (Meta Device) first
    # Then FSDP materializes only the shard this GPU needs.
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True, 
            # device_map="cpu" # We DON'T use this, we rely on low_cpu_mem_usage + FSDP
        )

    model.config.use_cache = False 
    model.gradient_checkpointing_enable() 

    # 5. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    except:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 6. Dataset
    dataset = load_dataset(DATASET_HF, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # 7. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
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
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        ddp_timeout=7200,
    )

    # 9. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    print(f"--- [GPU {rank}] Starting Training ---")
    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()