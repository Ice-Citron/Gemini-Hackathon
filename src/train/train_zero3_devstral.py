import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from peft import LoraConfig


# --- CONFIGURATION ---
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512"
DATASET_HF = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_devstral_zero3"
MAX_LENGTH = 4096 

def format_example(example):
    prompt = (
        f"You are a security expert. Fix the following {example['vulnerability_type']} vulnerability.\n\n"
        "## Vulnerable Code\n"
        "```python\n"
        f"{example['vulnerable_code']}\n"
        "```\n\n"
        "Provide the secure version of the code and explain your fix."
    )
    response = (
        "## Secure Code\n"
        "```python\n"
        f"{example['secure_code']}\n"
        "```\n\n"
        "## Patch\n"
        "```diff\n"
        f"{example['patch_diff']}\n"
        "```\n\n"
        "## Reasoning\n"
        f"{example['reasoning']}"
    )
    return {"text": f"[INST] {prompt} [/INST] {response}"}

def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"--- [GPU {rank}] Initializing ZeRO-3 Training ---")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # 1. Pre-download (Avoids race conditions)
    if rank == 0:
        print(f"[Rank 0] Pre-downloading model: {MODEL_NAME}")
        snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.ggml"])

    if world_size > 1:
        dist.barrier()

    # 2. Config Sanitization (THE FIX for FP8 Error)
    # We load the config and explicitly delete the quantization metadata.
    # This tricks the system into loading the weights as standard Bfloat16.
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        if rank == 0:
            print("--- [Detected FP8 Config] Stripping quantization metadata to bypass Trainer check ---")
        del config.quantization_config
        config.quantization_method = None
    
    # Force Bfloat16
    config.torch_dtype = torch.bfloat16

    # 3. Manual Model Load (THE FIX for kwargs Error)
    # We load the model manually so we can pass our sanitized 'config'.
    # 'low_cpu_mem_usage=True' allows Accelerate/DeepSpeed to shard it instantly.
    print(f"--- [GPU {rank}] Loading Model Weights (Sharded) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,             # Pass the sanitized config
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,    # Critical for ZeRO-3 loading
    )
    # Enable gradient checkpointing manually on the loaded model
    model.gradient_checkpointing_enable()

    # 4. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, use_fast=True
        )
    except Exception as e:
        if rank == 0: print(f"AutoTokenizer failed ({e}). Using Manual Mistral Tekken fallback...")
        from huggingface_hub import hf_hub_download
        tekken_file = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tekken_file)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Dataset
    if rank == 0: print(f"Loading dataset: {DATASET_HF}")
    dataset = load_dataset(DATASET_HF, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # 6. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 7. Training Args
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        ddp_timeout=7200,
        # Gradient checkpointing is handled manually above, but we keep this flag for Trainer compatibility
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 8. Trainer
    # We pass the loaded 'model' object directly, bypassing the internal loader that was causing issues.
    trainer = SFTTrainer(
        model=model,                 # <--- Manual Object Passed Here
        train_dataset=dataset,
        processing_class=tokenizer,  
        args=training_args,
        peft_config=peft_config,
    )

    print(f"--- [GPU {rank}] Starting Training ---")
    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        print(f"--- Model saved to {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()