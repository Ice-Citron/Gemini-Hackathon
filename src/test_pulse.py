import torch
from unsloth import FastLanguageModel

# Load 32B Coder in 4-bit (Requires ~20GB VRAM)
model_name = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"

print(f"🚀 Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

prompt = """### Instruction:
You are SkyHammer. Fix this code.
Vulnerability: SQL Injection

### Vulnerable Code:
```python
query = f"SELECT * FROM users WHERE user = '{username}'"

```

### Response:

"""

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
print("⚡ Generating...")
outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print("\n" + tokenizer.batch_decode(outputs)[0])