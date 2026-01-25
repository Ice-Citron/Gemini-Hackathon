#!/usr/bin/env python3
"""
SkyHammer RLAIF Factory
Collects training data by having Qwen 14B generate patches and Gemini judge them.
Outputs high-quality (prompt, response) pairs for LoRA fine-tuning.
"""

import os
import json
import argparse
import torch
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

# Import judge
from judge_gemini import compute_reward

# Check for GPU training environment
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("[!] Unsloth not available. Will use transformers fallback.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Configuration
DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 4096
N_ATTEMPTS_PER_SCENARIO = 4  # Generate 4 patches, pick best
MIN_SCORE_THRESHOLD = 0.5    # Only keep patches with score >= 0.5


def load_benchmark_set(benchmark_path: str = "data/benchmark_set/train_set.json") -> List[Dict]:
    """Load the training benchmark set"""
    if not os.path.exists(benchmark_path):
        print(f"[!] Benchmark set not found at {benchmark_path}")
        print("[!] Run 'python src/factory.py' first to generate benchmarks")
        return []

    with open(benchmark_path, "r") as f:
        return json.load(f)


def load_student_model():
    """Load the student model (Qwen 14B or fallback)"""
    if HAS_UNSLOTH:
        print(f"Loading {DEFAULT_MODEL} with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=DEFAULT_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"

    elif HAS_TRANSFORMERS:
        print(f"Loading {FALLBACK_MODEL} with Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer, "transformers"

    else:
        raise RuntimeError("No model loading library available. Install unsloth or transformers.")


def generate_patch(model, tokenizer, vulnerable_code: str, n_attempts: int = 4, backend: str = "unsloth") -> List[str]:
    """Generate multiple patch attempts from the student model"""
    prompt = f"""Fix the SQL injection vulnerability in this code. Use parameterized queries with placeholders (?) instead of string formatting.

VULNERABLE CODE:
```python
{vulnerable_code}
```

Output ONLY the fixed Python code, nothing else. Do not explain, just output the secure version:"""

    # Format for chat model
    if backend == "unsloth":
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        formatted_prompt = prompt

    inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt"
    ).to(model.device if hasattr(model, 'device') else "cuda")

    # Generate multiple attempts
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8,
        do_sample=True,
        num_return_sequences=n_attempts,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode outputs
    patches = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        # Extract just the response part
        if "assistant" in decoded:
            patch = decoded.split("assistant")[-1].strip()
        else:
            patch = decoded.replace(prompt, "").strip()

        # Clean up code blocks if present
        if "```python" in patch:
            patch = patch.split("```python")[-1].split("```")[0].strip()
        elif "```" in patch:
            patch = patch.split("```")[1].split("```")[0].strip()

        patches.append(patch)

    return patches


def collect_rlaif_data(
    scenarios: List[Dict],
    model,
    tokenizer,
    backend: str,
    output_path: str = "data/rlaif_14b_dataset.jsonl",
    n_attempts: int = N_ATTEMPTS_PER_SCENARIO,
    min_score: float = MIN_SCORE_THRESHOLD
) -> List[Dict]:
    """
    Main RLAIF data collection loop.

    For each scenario:
    1. Generate n_attempts patches from student model
    2. Score each with Gemini judge
    3. Keep the best one if score >= min_score
    """
    training_data = []
    stats = {"total": 0, "accepted": 0, "rejected": 0}

    print(f"\nStarting RLAIF data collection...")
    print(f"  - Scenarios: {len(scenarios)}")
    print(f"  - Attempts per scenario: {n_attempts}")
    print(f"  - Min score threshold: {min_score}")

    for scenario in tqdm(scenarios, desc="Collecting"):
        stats["total"] += 1
        vulnerable_code = scenario["vulnerable_code"]

        # Generate patches
        try:
            patches = generate_patch(model, tokenizer, vulnerable_code, n_attempts, backend)
        except Exception as e:
            print(f"\n[!] Generation error for {scenario['id']}: {e}")
            stats["rejected"] += 1
            continue

        # Score each patch
        best_patch = None
        best_score = -1.0
        best_reason = ""

        for patch in patches:
            if not patch.strip():
                continue

            score, reason = compute_reward(vulnerable_code, patch, use_gemini=True)

            if score > best_score:
                best_score = score
                best_patch = patch
                best_reason = reason

        # Keep if above threshold
        if best_score >= min_score and best_patch:
            prompt = f"Fix the SQL injection vulnerability in this code:\n\n{vulnerable_code}"

            training_data.append({
                "id": scenario["id"],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": best_patch}
                ],
                "score": best_score,
                "reason": best_reason,
                "collected_at": datetime.now().isoformat()
            })
            stats["accepted"] += 1
            tqdm.write(f"  [{scenario['id']}] Score: {best_score:.2f}")
        else:
            stats["rejected"] += 1
            tqdm.write(f"  [{scenario['id']}] REJECTED (best score: {best_score:.2f})")

    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nCollection complete!")
    print(f"  - Total scenarios: {stats['total']}")
    print(f"  - Accepted (score >= {min_score}): {stats['accepted']}")
    print(f"  - Rejected: {stats['rejected']}")
    print(f"  - Output: {output_path}")

    return training_data


def main():
    parser = argparse.ArgumentParser(description="RLAIF Data Factory")
    parser.add_argument("--n_scenarios", type=int, default=20, help="Number of scenarios to process")
    parser.add_argument("--n_attempts", type=int, default=4, help="Attempts per scenario")
    parser.add_argument("--min_score", type=float, default=0.5, help="Minimum score threshold")
    parser.add_argument("--output", type=str, default="data/rlaif_14b_dataset.jsonl", help="Output path")
    parser.add_argument("--benchmark", type=str, default="data/benchmark_set/train_set.json", help="Benchmark path")

    args = parser.parse_args()

    # Load benchmark set
    scenarios = load_benchmark_set(args.benchmark)
    if not scenarios:
        print("[!] No scenarios loaded. Generate benchmarks first with: python src/factory.py")
        return

    # Limit to requested number
    scenarios = scenarios[:args.n_scenarios]

    # Load model
    model, tokenizer, backend = load_student_model()

    # Collect data
    collect_rlaif_data(
        scenarios=scenarios,
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        output_path=args.output,
        n_attempts=args.n_attempts,
        min_score=args.min_score
    )


if __name__ == "__main__":
    main()
