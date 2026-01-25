import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported

# --- CONFIG ---
MODEL_NAME = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
DATASET_NAME = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_32b_ddp"
FINAL_REPO = "shng2025/SkyHammer-32B-v1"  # Where the merges get pushed

PROMPT_TEMPLATE = """### Instruction:
You are SkyHammer, an automated security agent.
Fix the following vulnerability.
Vulnerability Type: {vuln_type}

### Vulnerable Code:
```python
{code}

```

### Response:

{reasoning}

```diff
{patch}
```"""
EOS_TOKEN = "<|endoftext|>"

# --- CUSTOM CALLBACK FOR AUTO-MERGE ---
class SkyHammerCheckpointCallback(TrainerCallback):
    """
    Automates the Merge -> Save -> Push workflow at the end of every epoch.
    Only Rank 0 executes this to prevent race conditions.
    """
    def __init__(self, model, tokenizer, repo_id):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_id = repo_id

    def on_epoch_end(self, args, state, control, **kwargs):
        # DDP Check: Only the main process (Rank 0) should merge/upload
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            epoch = int(state.epoch)
            print(f"\n⚡ [Epoch {epoch}] Starting Auto-Merge & Push Protocol...")
            
            # 1. Save Merged Model Locally
            # We create a specific folder for this epoch to avoid overwrites
            epoch_dir = f"models/SkyHammer-32B-Epoch-{epoch}"
            print(f"   💾 Merging to {epoch_dir}...")
            self.model.save_pretrained_merged(
                epoch_dir, 
                self.tokenizer, 
                save_method = "merged_4bit_forced"
            )
            
            # 2. Push to Hugging Face
            print(f"   🚀 Uploading to {self.repo_id} (Branch/Tag: main)...")
            try:
                self.model.push_to_hub_merged(
                    self.repo_id, 
                    self.tokenizer, 
                    save_method = "merged_4bit_forced", 
                    token = os.getenv("HF_TOKEN"),
                    commit_message = f"SkyHammer Auto-Checkpoint: Epoch {epoch}"
                )
                print(f"   ✅ [Epoch {epoch}] Securely stored on Hugging Face.")
            except Exception as e:
                print(f"   ⚠️ [Epoch {epoch}] Upload failed (Local file is safe): {e}")

# --- HELPER FUNCTIONS ---
def formatting_prompts_func(examples):
    texts = []
    for instruction, input_code, reasoning, patch in zip(
        examples["vulnerability_type"], examples["vulnerable_code"],
        examples["reasoning"], examples["patch_diff"]):
        
        text = PROMPT_TEMPLATE.format(
            vuln_type=instruction, code=input_code, 
            reasoning=reasoning, patch=patch
        ) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def main():
    # --- SETUP ---
    os.environ["WANDB_PROJECT"] = "SkyHammer-Gemini-Hack"
    os.environ["WANDB_WATCH"] = "false"
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_device_map = {"": local_rank}
    
    if local_rank == 0:
        print(f"🚀 Initializing SkyHammer DDP (Rank {local_rank})")

    # --- MODEL LOADING ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        device_map = ddp_device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    dataset = load_dataset(DATASET_NAME, split = "train")
    
    with torch.device(f"cuda:{local_rank}"):
        dataset = dataset.map(formatting_prompts_func, batched = True)

    # --- TRAINER ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 1,  # Safe for 32B
            gradient_accumulation_steps = 2,  
            num_train_epochs = 3,             # Changed from steps to epochs for the callback
            warmup_steps = 5,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            report_to = "wandb",
            ddp_find_unused_parameters = False, 
            ddp_timeout = 1800,
            run_name = f"skyhammer_32b_ddp_rank{local_rank}",
            save_strategy = "no", # We handle saving via our Custom Callback
        ),
    )

    # Register our new Auto-Merge Callback
    trainer.add_callback(SkyHammerCheckpointCallback(model, tokenizer, FINAL_REPO))

    if local_rank == 0:
        print("🔥 Starting Training with Auto-Checkpointing...")

    trainer.train()
    
    # Final Save (Just in case the callback didn't catch the very last step)
    if local_rank == 0:
        print("🏁 Training Finished. Performing Final Push...")
        model.push_to_hub_merged(
            FINAL_REPO, 
            tokenizer, 
            save_method = "merged_4bit_forced", 
            token = os.getenv("HF_TOKEN"),
            commit_message = "SkyHammer Final Release"
        )

if __name__ == "__main__":
    main()