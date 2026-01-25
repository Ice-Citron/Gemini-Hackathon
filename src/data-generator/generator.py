import os
import sys
import json
import time
import threading
import concurrent.futures
from google import genai            # <--- NEW LIBRARY
from google.genai import types      # <--- NEW TYPES
from huggingface_hub import HfApi, create_repo
from typing import List, Dict

# --- 1. SETUP PATHS & CONFIG ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from secretsConfig import GEMINI_API_KEY, HF_TOKEN
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found! Set it in secretsConfig.py")

if not HF_TOKEN:
    print("⚠️  HF_TOKEN not found in secretsConfig.py. Upload will be skipped.")

# NOTE: No global genai.configure() needed anymore. We use a Client instance.

# --- 2. CONFIGURATION ---
HF_REPO_ID = "shng2025/SkyHammer-Gemini-Dataset"
OUTPUT_FILE = "data/raw_dataset.jsonl"
os.makedirs("data", exist_ok=True)

TEACHER_MODEL = "gemini-3-pro-preview"

# --- 3. PROMPTS ---
VULN_TYPES = [
    "SQL Injection in a login function",
    "Reflected XSS in a search query parameter",
    "Command Injection in a system administration tool",
    "Path Traversal in a file download endpoint",
    "Insecure Direct Object Reference (IDOR)",
    "Hardcoded API keys in a config loader",
    "Use of insecure MD5 hashing for passwords",
    "Unsafe deserialization using pickle",
    "XML External Entity (XXE) in a document parser",
    "Race condition in a coupon redemption system"
]

SYSTEM_PROMPT = """You are a Senior Security Engineer generating training data for a junior AI.
For each request, output a SINGLE JSON object with this exact structure:
{
    "task_id": "unique_id_string",
    "vulnerability_type": "string",
    "vulnerable_code": "The full python code containing the bug",
    "secure_code": "The full python code with the bug fixed",
    "patch_diff": "The unified diff between vulnerable and secure code",
    "reasoning": "A step-by-step explanation of how you found and fixed the bug"
}
Constraints:
1. Code must be self-contained Python.
2. The 'patch_diff' must be a valid unified diff string.
3. Do NOT output markdown fencing. Just raw JSON.
"""

# Global lock for file writing
file_lock = threading.Lock()

def generate_sample(args) -> Dict:
    """Wrapper function for threading"""
    vuln_type, index = args
    
    # --- NEW CLIENT INITIALIZATION ---
    # It is safe to initialize this per thread or pass a global one.
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Unique ID generation
    short_code = "".join([w[0] for w in vuln_type.split()[:3]]).lower()
    task_id = f"{short_code}_{index:03d}"
    
    prompt = f"Generate a training sample for: {vuln_type}. Task ID: {task_id}"
    
    print(f"[*] Requesting {task_id}...")
    try:
        # --- NEW GENERATION METHOD ---
        response = client.models.generate_content(
            model=TEACHER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,  # System prompt goes here now
                response_mime_type="application/json"
            )
        )
        
        data = json.loads(response.text)
        # Ensure task_id matches our format
        data["task_id"] = task_id
        return data
    except Exception as e:
        print(f"[!] Error on {task_id}: {e}")
        return None

def upload_dataset():
    """Uploads the generated file to Hugging Face Hub."""
    if not HF_TOKEN:
        return

    print(f"\n🚀 Uploading to Hugging Face: {HF_REPO_ID}...")
    api = HfApi(token=HF_TOKEN)
    
    try:
        create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=OUTPUT_FILE,
            path_in_repo="raw_dataset.jsonl",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"✅ Upload Complete! View at: https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

def main():
    print(f"🚀 Starting Parallel Data Factory using {TEACHER_MODEL}")
    print(f"🎯 Configuration: {len(VULN_TYPES)} vulnerability types x 5 samples each = 50 total.")
    
    # 1. Prepare the Task List
    tasks = []
    for vuln in VULN_TYPES:
        for i in range(1, 2):  # 10 samples per type (updated range from your script comment which said 5)
            tasks.append((vuln, i))
            
    successful_count = 0
    
    # 2. Run with ThreadPool (Concurrency)
    # max_workers=5 keeps us safe from rate limits while being fast
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(generate_sample, task): task for task in tasks}
        
        with open(OUTPUT_FILE, "a") as f:
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                
                if result and "vulnerable_code" in result:
                    # Thread-safe write
                    with file_lock:
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        successful_count += 1
                    print(f"   ✅ Saved {result['task_id']}")
                
                # Tiny sleep to jitter the requests slightly
                time.sleep(0.5)

    print(f"\n✨ Generation Complete. {successful_count} samples saved.")
    
    # 3. Trigger Auto-Upload
    upload_dataset()

if __name__ == "__main__":
    main()