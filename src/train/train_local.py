import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- CONFIGURATION ---
# We use Qwen 2.5 Coder 7B (Instruct) - The best coding model for its size
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None # None = auto detection. Float16 for Tesla T4, Bfloat16 for Ampere+
LOAD_IN_4BIT = True 

# Your Dataset on Hugging Face (Auto-detected from Data Factory)
DATASET_NAME = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_DIR = "checkpoints/skyhammer_lora"

# --- PROMPT TEMPLATE ---
# We force the model to output the "Chain of Thought" reasoning first, then the patch.
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

EOS_TOKEN = "<|endoftext|>" # Will be auto-patched by Unsloth

def formatting_prompts_func(examples):
    instructions = examples["vulnerability_type"]
    inputs       = examples["vulnerable_code"]
    reasonings   = examples["reasoning"]
    patches      = examples["patch_diff"]
    texts = []
    
    for instruction, input_code, reasoning, patch in zip(instructions, inputs, reasonings, patches):
        text = PROMPT_TEMPLATE.format(
            vuln_type=instruction,
            code=input_code,
            reasoning=reasoning,
            patch=patch
        ) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def train():
    print(f"🚀 Loading Model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # Add LoRA adapters (Efficient Fine-Tuning)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,
        loftq_config = None, 
    )

    print(f"📥 Loading Dataset: {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split = "train")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Tip: Ensure 'src/data/generator.py' finished and uploaded the data.")
        return

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("🔥 Starting Training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can set to True for speed boost
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # SHORT RUN for testing. Increase to 300+ for real training.
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
        ),
    )

    trainer_stats = trainer.train()
    print("✅ Training Complete!")

    # Save the LoRA adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()