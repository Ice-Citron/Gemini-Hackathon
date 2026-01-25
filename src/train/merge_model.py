import os
import torch
from unsloth import FastLanguageModel

# --- CONFIG ---
MODEL_NAME = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
ADAPTER_DIR = "checkpoints/skyhammer_32b_lora" # Corrected path
OUTPUT_DIR = "models/SkyHammer-32B-v1"
HF_REPO = "shng2025/SkyHammer-32B-v1" 

def main():
    print(f"🦥 Loading Adapter from {ADAPTER_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTER_DIR, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print("⚡ Merging LoRA adapters into base model...")
    # This saves to local disk first
    model.save_pretrained_merged(
        OUTPUT_DIR, 
        tokenizer, 
        save_method = "merged_4bit_forced", 
    )
    print(f"✅ Local merge saved to {OUTPUT_DIR}")
    
    # --- PUSH TO HUB SECTION ---
    print(f"🚀 Pushing to Hugging Face: {HF_REPO}...")
    try:
        # This uploads the merged model directly
        model.push_to_hub_merged(
            HF_REPO, 
            tokenizer, 
            save_method = "merged_4bit_forced", 
            token = os.getenv("HF_TOKEN") # Uses your cached token if env var is missing
        )
        print("✅ Push Complete!")
    except Exception as e:
        print(f"❌ Push Failed (You can still upload manually with CLI): {e}")

if __name__ == "__main__":
    main()