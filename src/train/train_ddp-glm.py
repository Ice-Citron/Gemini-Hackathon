import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# USE THE OFFICIAL NVIDIA/MISTRAL REPO (NOT MLX!)
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512"
OUTPUT_DIR = "checkpoints/skyhammer_devstral_123b"

def main():
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset
    dataset = load_dataset("shng2025/SkyHammer-Gemini-Dataset", split="train")

    # 3. Setup Config (DeepSpeed handles the model loading!)
    # We DO NOT load the model manually here with .from_pretrained()
    # SFTTrainer + DeepSpeed integration will handle the sharding automatically.
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        learning_rate=1e-5, # Lower LR for massive models
        bf16=True,          # Use BFloat16 (Native L40S support)
        logging_steps=1,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=50,
        ddp_timeout=5400,   # Increase timeout for massive model syncing
        report_to="none",
    )

    # 4. Trainer
    trainer = SFTTrainer(
        model=MODEL_NAME,   # Pass string; Accelerate loads it sharded
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("--- Starting ZeRO-3 Training (123B Params) ---")
    trainer.train()
    
    # Save (ZeRO-3 will gather weights from all GPUs automatically)
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()