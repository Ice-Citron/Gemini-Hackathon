import os
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoConfig
from huggingface_hub import hf_hub_download

# --- CONFIG ---
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512" 
DATASET_HF = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_REPORT = "debug_tokenizer_report.txt"
OUTPUT_IDS = "debug_ids.txt"
OUTPUT_DECODED = "debug_decoded.txt"

def format_example(example):
    prompt = f"VULNERABILITY: {example['vulnerability_type']}\nCODE:\n{example['vulnerable_code']}\nFIX THIS."
    response = f"SECURE CODE:\n{example['secure_code']}\nREASONING:\n{example['reasoning']}"
    return {"text": f"[INST] {prompt} [/INST] {response}"}

def main():
    print(f"--- 🔍 STARTING DEEP TOKENIZER AUDIT ---")
    
    # 1. Load Model Config
    print(f"Loading Config for {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    real_vocab_limit = config.vocab_size
    print(f"✅ Model Vocab Size Limit: {real_vocab_limit}")

    # 2. Load Tokenizer (Exact Logic from Train Script)
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
        tokenizer_source = "Main (Devstral-2-123B)"
    except:
        print("⚠️ Main tokenizer failed. Using Fallback (Tekken/Small)...")
        f = hf_hub_download(repo_id="mistralai/Devstral-Small-2505", filename="tekken.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f)
        tokenizer_source = "Fallback (Devstral-Small/Tekken)"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer_vocab_size = len(tokenizer)
    print(f"✅ Tokenizer Loaded from: {tokenizer_source}")
    print(f"📊 Tokenizer Vocab Size: {tokenizer_vocab_size}")

    # 3. Load Data
    print(f"\nDownloading Dataset {DATASET_HF}...")
    dataset = load_dataset(DATASET_HF, split="train")
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"Processing {len(formatted_dataset)} examples...")

    # 4. Scan & Dump
    max_id_found = 0
    
    # Open all file handles
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f_rep, \
         open(OUTPUT_IDS, "w", encoding="utf-8") as f_ids, \
         open(OUTPUT_DECODED, "w", encoding="utf-8") as f_dec:

        # Write Headers
        f_rep.write(f"TOKENIZER AUDIT REPORT\n======================\nModel Limit: {real_vocab_limit}\nTokenizer: {tokenizer_source}\n\n")
        f_ids.write(f"RAW TOKEN IDS DUMP\n==================\n")
        f_dec.write(f"DECODED TEXT DUMP (Sanity Check)\n================================\n")

        for idx, example in enumerate(formatted_dataset):
            text = example['text']
            
            # A. Tokenize (Text -> Numbers)
            tokens = tokenizer(text, return_tensors="pt").input_ids[0]
            
            # B. Decode (Numbers -> Text)
            decoded_text = tokenizer.decode(tokens)

            # C. Check Limits
            current_max = tokens.max().item()
            if current_max > max_id_found:
                max_id_found = current_max
            
            out_of_bounds = tokens[tokens >= real_vocab_limit]
            
            # D. Write to ID File
            f_ids.write(f"--- Example {idx} ---\n")
            f_ids.write(f"{tokens.tolist()}\n\n")

            # E. Write to Decoded File
            f_dec.write(f"--- Example {idx} ---\n")
            f_dec.write(f"{decoded_text}\n")
            f_dec.write("-" * 40 + "\n")

            # F. Write to Report (Only errors or first 3)
            if idx < 3 or len(out_of_bounds) > 0:
                status = "✅ OK" if len(out_of_bounds) == 0 else "❌ CRITICAL FAIL"
                f_rep.write(f"Example {idx} [{status}]\n")
                f_rep.write(f"Length: {len(tokens)} tokens\n")
                f_rep.write(f"Max ID: {current_max}\n")
                if len(out_of_bounds) > 0:
                    f_rep.write(f"⚠️ ILLEGAL IDS: {out_of_bounds.tolist()}\n")
                f_rep.write("\n")

    # 5. Final Verdict
    print(f"\n--- AUDIT RESULTS ---")
    print(f"Max Token ID Found: {max_id_found}")
    print(f"Model Limit:        {real_vocab_limit}")
    
    if max_id_found >= real_vocab_limit:
        print(f"\n❌❌ FAIL: Data contains IDs >= Model Limit.")
        print(f"    This CONFIRMS the cause of the NaN crash.")
        print(f"    The training script will fix this via 'resize_token_embeddings'.")
    else:
        print(f"\n✅✅ PASS: All tokens valid. Tokenizer is compatible.")
        print(f"    If it crashes, double check Learning Rate ({5e-7} recommended).")

    print(f"\nFiles generated:")
    print(f"1. {OUTPUT_REPORT} (Summary)")
    print(f"2. {OUTPUT_IDS}    (Raw numbers - check for weird patterns)")
    print(f"3. {OUTPUT_DECODED} (Read this to ensure code looks like code!)")

if __name__ == "__main__":
    main()