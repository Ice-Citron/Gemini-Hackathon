import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

# --- CONFIGURATION ---
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512" 
OUTPUT_DIR = "checkpoints/skyhammer_devstral_zero3"
MAX_SEQ_LENGTH = 4096

def main():
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    print(f"--- [GPU {rank}] Initializing ZeRO-3 Training ---")

    # 1. Load Tokenizer (With Tekken Fallback)
    try:
        # Standard load attempt
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        if rank == 0:
            print(f"AutoTokenizer failed ({e}). Using Manual Mistral Tekken fallback...")
        
        # Fallback: Download the tokenizer file manually
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from huggingface_hub import hf_hub_download
        
        # Fetch tekken.json from the small model if the big one is gated
        tekken_file = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tekken_file)

    # Ensure special tokens exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset
    try:
        dataset = load_dataset("shng2025/SkyHammer-Gemini-Dataset", split="train")
    except:
        if rank == 0: print("Warning: Dataset not found. Using dummy data.")
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": ["def hello(): pass"] * 100})

    # 3. Training Arguments (FIXED: dataset_text_field moved here)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",  # <--- MOVED HERE (Fixes TypeError)
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=1,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        ddp_timeout=7200,
        max_seq_length=MAX_SEQ_LENGTH, # Also moved to config in newer TRL
        packing=False,                 # Explicitly disable packing to avoid conflicts
    )

    # 4. Trainer
    trainer = SFTTrainer(
        model=MODEL_NAME,   
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        # Removed 'dataset_text_field' & 'max_seq_length' from here as they are now in args
    )

    print(f"--- [GPU {rank}] Starting Training ---")
    trainer.train()
    
    if rank == 0:
        trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()