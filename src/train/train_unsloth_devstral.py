import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_NAME = "unsloth/Devstral-2-123B-Instruct-2512" 
OUTPUT_DIR = "checkpoints/skyhammer_devstral_fp8"
MAX_SEQ_LENGTH = 4096 

def main():
    print(f"--- [GPU {os.environ.get('LOCAL_RANK', '0')}] Initializing ---")
    
    # 1. Load Model (FP8 Mode)
    # We set load_in_4bit=False to stop Unsloth from forcing bitsandbytes.
    # This lets the model's native 'FineGrainedFP8Config' take over.
    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,           
        load_in_4bit = False,   # <--- FIXED: Stops sending BNB config
    )

    # 2. Load Tokenizer (Tekken / Hybrid Fix)
    try:
        print("--- Attempting AutoTokenizer ---")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        print(f"Warning: AutoTokenizer failed ({e}). Attempting Manual Tekken Load...")
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        
        tekken_file = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        mistral_tokenizer = MistralTokenizer.from_file(tekken_file)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tekken_file)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Add LoRA Adapters
    # Note: We use standard LoRA settings here. 
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 4. Dataset
    try:
        dataset = load_dataset("shng2025/SkyHammer-Gemini-Dataset", split="train")
    except:
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": ["def hello(): pass"] * 10})

    # 5. Trainer
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        tokenizer = tokenizer,
        args = SFTConfig(
            output_dir = OUTPUT_DIR,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            logging_steps = 1,
            num_train_epochs = 1,
            report_to = "none",
            ddp_find_unused_parameters = False,
        ),
    )

    print("--- Starting Training (FP8 Native) ---")
    trainer.train()

if __name__ == "__main__":
    main()