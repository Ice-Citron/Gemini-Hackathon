import os
import gc
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported

# --- CONFIGURATION ---
MODEL_NAME = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
            # unsloth/Qwen3-32B-unsloth-bnb-4bit
DATASET_NAME = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_32b_ddp"
FINAL_REPO = "shng2025/SkyHammer-32B-v1"

# TRAINING HYPERPARAMETERS
NUM_EPOCHS = 3          # Total training duration
SAVE_EVERY_EPOCHS = 1   # Checkpoint frequency (1 = every epoch, 2 = every 2 epochs, etc.)

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
    def __init__(self, model, tokenizer, repo_id, save_freq):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_id = repo_id
        self.save_freq = save_freq

    def on_epoch_end(self, args, state, control, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        epoch = int(state.epoch)

        if local_rank == 0 and (epoch % self.save_freq == 0):
            print(f"\n⚡ [Epoch {epoch}] Saving Adapters (Freq: {self.save_freq})...")
            
            # 1. Save ADAPTERS Locally (Fast & Light)
            epoch_dir = f"models/SkyHammer-32B-Epoch-{epoch}-Adapter"
            print(f"   💾 Saving LoRA adapters to {epoch_dir}...")
            
            try:
                # CHANGED: We now use .save_pretrained() instead of .save_pretrained_merged()
                self.model.save_pretrained(epoch_dir)
                self.tokenizer.save_pretrained(epoch_dir)
                
                # 2. Push ADAPTERS to Hub
                print(f"   🚀 Uploading Adapters to {self.repo_id}...")
                self.model.push_to_hub(
                    self.repo_id, 
                    token = os.getenv("HF_TOKEN"),
                    commit_message = f"SkyHammer Adapter: Epoch {epoch}"
                )
                # Also push tokenizer so the repo is usable
                self.tokenizer.push_to_hub(
                     self.repo_id,
                     token = os.getenv("HF_TOKEN")
                )
                print(f"   ✅ [Epoch {epoch}] Adapters securely stored.")
                
            except Exception as e:
                print(f"   ⚠️ [Epoch {epoch}] Upload failed: {e}")

            # 3. Light Cleanup (No heavy fragmentation to fix)
            gc.collect()
            torch.cuda.empty_cache()
            print("   ✨ Resuming training...")

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
    os.environ["WANDB_PROJECT"] = "SkyHammer-Gemini-Hack"
    os.environ["WANDB_WATCH"] = "false"
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_device_map = {"": local_rank}
    
    if local_rank == 0:
        print(f"🚀 Initializing SkyHammer DDP (Rank {local_rank})")
        print(f"🎯 Config: {NUM_EPOCHS} Epochs, Saving every {SAVE_EVERY_EPOCHS} Epochs.")

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

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 2,  
            num_train_epochs = NUM_EPOCHS,      # Uses top-level config
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
            # run_name removed -> WandB will auto-generate one
            save_strategy = "no", 
        ),
    )

    # Pass the frequency config to the callback
    trainer.add_callback(SkyHammerCheckpointCallback(model, tokenizer, FINAL_REPO, SAVE_EVERY_EPOCHS))

    trainer.train()
    
    # Final cleanup & save
    if local_rank == 0:
        print("🏁 Training Finished. Performing Final Push...")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model.push_to_hub_merged(
                FINAL_REPO, 
                tokenizer, 
                save_method = "merged_4bit_forced", 
                token = os.getenv("HF_TOKEN"),
                commit_message = "SkyHammer Final Release"
            )
        except Exception as e:
            print(f"⚠️ Final Push Failed: {e}")

if __name__ == "__main__":
    main()