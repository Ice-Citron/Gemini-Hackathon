# SkyHammer - Project Status & Context

**Last Updated: January 17, 2026 (RLAIF PIVOT)**
**For: GDM Geminiathon**

---

## 1. PROJECT OVERVIEW

### What Is This Project?

SkyHammer is a **cybersecurity AI system** powered by Gemini (GDM) with two main capabilities:

1. **ATTACK Mode**: AI agent that can find and exploit security vulnerabilities in web applications
2. **DEFENSE Mode**: AI agent that scans code for vulnerabilities, generates patches, and verifies fixes

### The Vision (For GDM Pitch)

- **RLAIF Training**: Use Gemini as the Judge to train smaller models (Qwen 14B) to write secure code
- **Benchmark Suite**: SQLi, XSS, Command Injection, Path Traversal benchmarks for quantitative results
- **Cost-Effective Security**: Train local models that approach SOTA performance at fraction of the cost

### Current Hackathon (January 2026 - Geminiathon)

Repo: `/Users/administrator/Imperial-College-London/Projects/2025/2026-01 January/GDM Geminiathon/`

**PRIORITY: RLAIF + Benchmarks (SQLi first)**

---

## 2. RLAIF BATTLE PLAN (FULL THRUST)

### 2.1 The Architecture: "Gemini-Guided Evolution"

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLAIF TRAINING LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   STUDENT    │     │   TEACHER    │     │   BENCHMARK  │   │
│   │ (Qwen 14B)   │────>│ (Gemini API)   │────>│  (50 Vulns)  │   │
│   │   Local      │     │   Judges     │     │  SQLi/XSS    │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                    │                     │            │
│         │    Score 0-1       │                     │            │
│         ▼                    ▼                     ▼            │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │               GRPO WEIGHT UPDATE                         │  │
│   │   Generate 4 patches → Gemini scores → Update LoRA         │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Pipeline

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `src/factory.py` | Generate 50 synthetic SQLi vulnerabilities |
| 2 | `src/judge_gemini.py` | Gemini-based reward function (0.0 - 1.0) |
| 3 | `src/rlaif_factory.py` | Collect trajectories: Student tries → Teacher scores |
| 4 | `src/train_14b.py` | GRPO + LoRA training on H200 |
| 5 | `src/benchmark_runner.py` | Run tournament across models |

### 2.3 The Benchmark Matrix (Target Output)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     SQLi VULNERABILITY BENCHMARK                            │
├───────────────────────┬──────────────┬──────────────┬──────────────────────┤
│ Model                 │ Success Rate │ False Pos %  │ Cost ($/1k runs)     │
├───────────────────────┼──────────────┼──────────────┼──────────────────────┤
│ GPT-4o (Oracle)       │ 98%          │ 2%           │ High                 │
│ Gemini Beta (Oracle)    │ 95%          │ 4%           │ Medium               │
│ Claude 3.5 Sonnet     │ 93%          │ 5%           │ Medium               │
│ Qwen-14B (Base)       │ 40%          │ 30%          │ Low                  │
│ SkyHammer (RLAIF)     │ 85%+         │ <10%         │ Low                  │
└───────────────────────┴──────────────┴──────────────┴──────────────────────┘
```

**The Narrative:**
> "We proved that a small, local model (SkyHammer), when trained with RLAIF using
> Gemini as a teacher, can outperform its base model by +45% and approach the
> performance of massive frontier models for a fraction of the cost."

---

## 3. BENCHMARK CATEGORIES

### 3.1 SQL Injection (SQLi) - PRIMARY FOCUS

| ID | Template | Description |
|----|----------|-------------|
| sqli_001 | Flask F-String | `f"SELECT * FROM users WHERE name='{name}'"` |
| sqli_002 | Django Raw | `cursor.execute("SELECT * FROM " + table)` |
| sqli_003 | FastAPI ORM | Unsafe parameterization |
| sqli_004 | PHP Legacy | `mysql_query($query)` |
| ... | ... | 50 variations with randomized names |

### 3.2 XSS (Cross-Site Scripting) - NEXT

| ID | Template | Description |
|----|----------|-------------|
| xss_001 | Reflected | Direct echo of user input |
| xss_002 | Stored | Database retrieval without sanitization |
| xss_003 | DOM-Based | Client-side injection |

### 3.3 Command Injection - FUTURE

| ID | Template | Description |
|----|----------|-------------|
| cmd_001 | os.system | `os.system("ping " + ip)` |
| cmd_002 | subprocess | `subprocess.call(cmd, shell=True)` |

### 3.4 Path Traversal - FUTURE

| ID | Template | Description |
|----|----------|-------------|
| path_001 | File Read | `open(f"uploads/{filename}")` |
| path_002 | Include | `include($_GET['page'])` |

---

## 4. EXECUTION PLAN (1.25 Hours)

### Immediate Actions (H200 Instance)

```bash
# 1. Boot H200 Instance (Vast.ai or Lambda)
#    Min specs: 80GB VRAM, 64GB RAM

# 2. Install dependencies
pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes openai

# 3. Set API keys
export GDM_API_KEY="xai-..."

# 4. Run the factory (generates training data)
python src/rlaif_factory.py --n_scenarios 20

# 5. Train the model (LoRA fine-tuning)
python src/train_14b.py

# 6. Run benchmarks
python src/benchmark_runner.py --models "gemini-beta,qwen-14b-base,skyhammer-rlaif"
```

### File Structure for RLAIF

```
GDM Geminiathon/
├── src/
│   ├── factory.py           # SQLi vulnerability generator
│   ├── judge_gemini.py        # Gemini-based reward function
│   ├── rlaif_factory.py     # Data collection loop
│   ├── train_14b.py         # LoRA training script
│   └── benchmark_runner.py  # Model comparison
│
├── data/
│   ├── benchmark_set/       # 50 synthetic vulns (10 test, 40 train)
│   └── rlaif_14b_dataset.jsonl  # Collected trajectories
│
└── checkpoints/
    └── skyhammer_14b_lora/  # Trained LoRA adapter
```

---

## 5. REWARD FUNCTION (judge_gemini.py)

### Hybrid Scoring

```python
def compute_reward(vulnerable_code, patched_code, test_result):
    score = 0.0

    # 1. HARD GATE: Did the exploit fail after patching?
    if "EXPLOIT FAILED" in test_result:
        score += 0.5
    elif "Server failed to start" in test_result:
        return -0.5  # Penalty for breaking the app

    # 2. SOFT CHECK: Gemini rates code quality
    # - Must use parameterized queries
    # - Must NOT just use regex/sanitization
    # - Must preserve original logic
    gemini_score = gemini_judge(vulnerable_code, patched_code)
    score += gemini_score * 0.5

    return score  # Range: -0.5 to 1.0
```

### Gemini Judge Prompt

```
You are a Senior Security Engineer. Rate this patch on a scale of 0.0 to 1.0.

ORIGINAL:
{vulnerable_code}

PATCH:
{patched_code}

CRITERIA:
- Must use parameterized queries (prepared statements)
- Must NOT just use regex/sanitization (brittle)
- Must preserve original logic

Output ONLY JSON: {"score": 0.9, "reason": "..."}
```

---

## 6. TRAINING CONFIG (train_14b.py)

### LoRA Configuration

```python
lora_config = LoraConfig(
    r = 16,                # Higher rank for code tasks
    lora_alpha = 32,       # Stability
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN (Critical for reasoning)
    ],
    task_type = "CAUSAL_LM",
    lora_dropout = 0,
    bias = "none",
)
```

### Training Arguments

```python
TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 60,        # Fast run for hackathon
    learning_rate = 2e-4,
    bf16 = True,           # Use bfloat16 on H200
    logging_steps = 1,
    output_dir = "checkpoints",
    optim = "adamw_8bit",
)
```

---

## 7. CURRENT ARCHITECTURE (SkyHammer CLI)

### Entry Points

| File | Description | Status |
|------|-------------|--------|
| `gemini_code-base.py` | Main entry point (NO CAI) - modular, imports from src/ | READY |
| `gemini_code.py` | Legacy monolithic version (1500+ lines) | DEPRECATED |

### Modular Structure (src/)

```
GDM Geminiathon/
├── gemini_code-base.py          # Main entry point (907 lines, modular)
├── secretsConfig.py           # API keys (GDM_API_KEY)
│
├── src/
│   ├── theme.py               # Evangelion/Cyberpunk color constants
│   ├── logging_utils.py       # Session logging to runs/
│   ├── ui_components.py       # NERV-style UI functions
│   ├── tools.py               # Tool definitions and execution
│   ├── attack.py              # Attack CLI tool
│   ├── patcher.py             # Blue team patch generator
│   └── generator.py           # Synthetic vulnerable app generator
│
├── runs/                      # Session logs (auto-generated)
│   └── gemini_code_YYYY-MM-DD_HH-MM-SS/
│       ├── session.jsonl
│       └── reports/           # SkyHammer reports saved here
│
└── data/                      # RLAIF data (NEW)
    ├── benchmark_set/
    └── rlaif_14b_dataset.jsonl
```

---

## 8. HOW TO RUN (Current CLI)

### Quick Start

```bash
cd "/Users/administrator/Imperial-College-London/Projects/2025/2026-01 January/GDM Geminiathon"

# Interactive mode
python3 gemini_code-base.py

# Single command mode
python3 gemini_code-base.py "create a Flask app with login"
```

### SkyHammer Flow

```
1. Start:     python3 gemini_code-base.py
2. Create:    "create a Flask app with SQL injection vulnerability"
3. Engage:    /skyhammer → Select target
4. Watch:     Gemini scans → exploits → reports → patches
5. Approve:   Review diff → EXECUTE
```

---

## 9. DEPENDENCIES

### CLI (gemini_code-base.py)

```bash
pip install rich questionary openai httpx pydantic
```

### RLAIF Training (H200)

```bash
pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes openai torch
```

---

## 10. API KEYS NEEDED

| Key | Purpose | Location |
|-----|---------|----------|
| `GDM_API_KEY` | Gemini API (Teacher) | `secretsConfig.py` or `export GDM_API_KEY=...` |
| `OPENAI_API_KEY` | GPT-4o benchmarks | Optional |
| `ANTHROPIC_API_KEY` | Claude benchmarks | Optional |

---

## 11. QUICK REFERENCE FOR FUTURE CLAUDE

**PRIORITY: RLAIF + SQLi Benchmarks**

**Current state:**
- `gemini_code-base.py` is the CLI (Evangelion UI)
- Need to build RLAIF pipeline on H200 instance
- Focus on SQLi first, then expand to XSS

**Next scripts to create:**
```
src/factory.py          # Generate 50 SQLi vulnerabilities
src/judge_gemini.py       # Gemini-based reward (0.0 - 1.0)
src/rlaif_factory.py    # Data collection with Qwen 14B
src/train_14b.py        # LoRA fine-tuning
src/benchmark_runner.py # Compare models
```

**Key insight:**
> Without benchmarks, this is just a demo.
> With benchmarks showing SkyHammer-RLAIF beating base Qwen-14B by +45%
> and approaching Gemini/GPT-4o performance → winning project.

---

## 12. GEMINI MODELS FOR JUDGING

| Model ID | Price (in/out per 1M) | Use Case |
|----------|----------------------|----------|
| `gemini-beta` | $5 / $15 | Primary Judge |
| `gemini-4-1-fast-reasoning` | $0.20 / $0.50 | Fast evaluation |

---

## 13. SAFETY CONSTRAINTS

**CRITICAL:** Since training an offensive model, add hard constraints:

```python
BLOCKED_PATTERNS = [
    "rm -rf /",
    ":(){ :|:& };:",  # Fork bomb
    "DROP DATABASE",
    "malicious IP addresses"
]

def safety_filter(output):
    for pattern in BLOCKED_PATTERNS:
        if pattern in output:
            return -10.0  # Hard penalty
    return 0.0
```

---

## 14. SUCCESS CRITERIA

| Metric | Target |
|--------|--------|
| SQLi Benchmark (10 test cases) | >80% success rate |
| Improvement over base Qwen-14B | >+40% |
| Training time on H200 | <30 minutes |
| Benchmark runs against SOTA | 5+ models compared |

---

**END OF DOCUMENT**
