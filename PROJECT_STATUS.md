# SkyHammer - Project Status & Context

**Last Updated: January 24, 2026**
**For: XAI Grokathon (pivoted from GDM Geminiathon)**

---

## 0. COMPLETION STATUS OVERVIEW

### Legend
- ✅ **DONE** - Code exists and works
- 🟡 **WRITTEN BUT UNTESTED** - Code exists, needs validation
- 🔴 **NOT DONE** - Needs to be built
- 📦 **EXISTS IN OTHER REPO** - Done in Iterate-RL-Hackathon

### Quick Status Table

| Component | Status | Notes |
|-----------|--------|-------|
| Main CLI (Evangelion UI) | ✅ | `gemini_code-base.py` working |
| SQLi Benchmark Dataset | ✅ | 50 samples in `data/benchmark_set/` |
| SQLi Factory | ✅ | `src/factory.py` generates variations |
| Gemini Judge | 🟡 | `src/judge_gemini.py` - UNTESTED |
| RLAIF Data Collection | 🟡 | `src/rlaif_factory.py` - UNTESTED |
| LoRA Training Script | 🟡 | `src/train_14b.py` - UNTESTED |
| Benchmark Runner | 🟡 | `src/benchmark_runner.py` - UNTESTED |
| vLLM Integration | 🟡 | `gemini_code-vllm.py` - UNTESTED |
| Trained LoRA Weights | 🔴 | `checkpoints/` is empty |
| GRPO Implementation | 📦 | Tested in Iterate-RL-Hackathon repo |
| XSS/CmdInj/PathTraversal | 🔴 | Templates not built yet |

---

## 0.1 DETAILED COMPONENT STATUS

### Core CLI & Infrastructure

| Component | Status | File | Lines | Notes |
|-----------|--------|------|-------|-------|
| Main CLI (Evangelion UI) | ✅ | `gemini_code-base.py` | 1,115 | Working, tested |
| Gemini API Integration | ✅ | `gemini_code-base.py` | - | OpenAI-compatible endpoint |
| vLLM Integration | 🟡 | `gemini_code-vllm.py` | 376 | Needs testing |
| Session Logging | ✅ | `src/logging_utils.py` | - | Logs to `runs/` |
| Theme/UI Components | ✅ | `src/theme.py`, `src/ui.py` | - | NERV aesthetics |
| Legacy Monolithic | ⚠️ | `gemini_code.py` | 1,521 | DEPRECATED |

### Benchmark Suite

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| SQLi Vulnerability Factory | ✅ | `src/factory.py` | 5 templates (flask_fstring, flask_format, flask_concat, fastapi_fstring, raw_execute) |
| SQLi Benchmark Dataset | ✅ | `data/benchmark_set/` | 50 JSON samples (40 train, 10 test) |
| XSS Templates | 🔴 | - | Not built yet |
| Command Injection Templates | 🔴 | - | Not built yet |
| Path Traversal Templates | 🔴 | - | Not built yet |

### RLAIF Training Pipeline

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| Gemini Judge (Reward Function) | 🟡 | `src/judge_gemini.py` | Hybrid scoring (60% programmatic + 40% Gemini), **UNTESTED** |
| Safety Filter | 🟡 | `src/judge_gemini.py` | Blocks dangerous patterns, **UNTESTED** |
| Parameterized Query Detection | 🟡 | `src/judge_gemini.py` | Regex-based checks, **UNTESTED** |
| RLAIF Data Collection | 🟡 | `src/rlaif_factory.py` | Generates trajectories with Qwen 14B, **UNTESTED** |
| LoRA Training Script | 🟡 | `src/train_14b.py` | Unsloth + SFTTrainer config, **UNTESTED** |
| Benchmark Runner | 🟡 | `src/benchmark_runner.py` | Multi-model comparison, **UNTESTED** |
| GRPO Implementation | 📦 | Iterate-RL-Hackathon | **TESTED** with OpenPipe on Qwen-14B |
| Trained LoRA Weights | 🔴 | `checkpoints/` | Empty - no training run yet |
| RLAIF Dataset | 🔴 | `data/rlaif_14b_dataset.jsonl` | Not generated yet |

### Attack Mode (Pentesting Agent)

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| Attack CLI Tool | 🟡 | `src/attack.py` | Exists, unclear if tested |
| CAI Integration | 🟡 | `src/cai_bridge.py` | Bridge to CAI framework |
| DVWA Environment | 🔴 | - | Not set up in this repo |
| Tool Execution Engine | 🟡 | `src/tools.py` | Tool definitions exist |

### Defense Mode (Patch Generation)

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| Vulnerability Scanner | 🟡 | `src/defense_demo/scanner.py` | Exists |
| Patch Generator | 🟡 | `src/defense_demo/patcher.py` | Exists |
| Patch Verifier | 🟡 | `src/defense_demo/verifier.py` | Exists |
| Defense Orchestrator | 🟡 | `src/defense_demo/orchestrator.py` | Exists |
| Apply Patch Logic | 🟡 | `src/defense_demo/apply_patch.py` | Exists |

### MCP Server (Judge Tools)

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| MCP Server | 🟡 | `src/mcp_server/server.py` | Exists |
| MCP Tools | 🟡 | `src/mcp_server/tools.py` | Exists, needs verification |
| Verification Tools (SQLi, XSS) | 🔴 | - | Need to verify what's implemented |

### GPU/Training Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| H200/H100 Instance | 🔴 | Not provisioned yet |
| Dependencies Installed | 🔴 | `unsloth`, `trl`, `peft` not tested |
| W&B Integration | 🔴 | Mentioned in plan, not implemented |

---

## 0.2 CRITICAL PATH & BLOCKERS

```
┌─────────────────────────────────────────────────────────────────┐
│                     CRITICAL PATH TO DEMO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Test existing scripts locally    ← YOU ARE HERE              │
│          ↓                                                       │
│  2. GPU Instance (H200)              ← BLOCKER for training      │
│          ↓                                                       │
│  3. Run rlaif_factory.py             ← Generate training data    │
│          ↓                                                       │
│  4. Run train_14b.py                 ← Train LoRA adapter        │
│          ↓                                                       │
│  5. Run benchmark_runner.py          ← Prove improvement         │
│          ↓                                                       │
│  6. Show 45%+ improvement            ← WIN                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 0.3 TODO LIST (Prioritized)

### PHASE 1: Validate Existing Code (Before GPU)
- [ ] Test `src/factory.py` locally - verify 50 SQLi samples generate correctly
- [ ] Test `src/judge_gemini.py` locally - verify scoring works with Gemini API
- [ ] Test `gemini_code-vllm.py` with a local/small model
- [ ] Review Iterate-RL-Hackathon repo for GRPO code to port over
- [ ] Remove junk files: `main.js`, `main.js.map`, `main_js_content.js`

### PHASE 2: GPU Setup & Training
- [ ] Provision H200 instance (Vast.ai / Lambda)
- [ ] Install dependencies: `unsloth`, `trl`, `peft`, `torch`
- [ ] Run `python src/rlaif_factory.py --n_scenarios 20`
- [ ] Run `python src/train_14b.py`
- [ ] Save checkpoint to `checkpoints/skyhammer_14b_lora/`

### PHASE 3: Benchmarking & Proof
- [ ] Run `python src/benchmark_runner.py` with multiple models
- [ ] Generate comparison table showing +45% improvement
- [ ] Create visualization/charts for pitch

### PHASE 4: Expand (If Time)
- [ ] Add XSS vulnerability templates to factory
- [ ] Add Command Injection templates
- [ ] Set up DVWA environment for live testing
- [ ] Test MCP server with verification tools

### PHASE 5: Cleanup & Polish
- [ ] Update README with final results
- [ ] Prepare demo video
- [ ] Document API usage and costs

---

## 0.4 RELATED REPOSITORIES

| Repo | Path | What's There |
|------|------|--------------|
| **SkyHammer-Gemini-Hack** | Current repo | Main project, RLAIF pipeline (untested) |
| **Iterate-RL-Hackathon** | `/Users/administrator/.../2025-11 November/Iterate-RL-Hackathon` | **TESTED** GRPO + OpenPipe + Qwen-14B training |
| **GDM Geminiathon** | `/Users/administrator/.../2026-01 January/GDM Geminiathon` | Original hackathon version |

---

## 0.5 FILES TO DELETE (Cleanup)

These files don't belong in a Python project and should be removed:

| File | Size | Reason |
|------|------|--------|
| `main.js` | 879 KB | Minified React bundle, not used |
| `main.js.map` | 5.0 MB | Source map, not used |
| `main_js_content.js` | 879 KB | Duplicate of main.js, untracked |

---

## 1. PROJECT OVERVIEW

### What Is This Project?

SkyHammer is a **cybersecurity AI system** with two main capabilities:

1. **ATTACK Mode**: AI agent that can find and exploit security vulnerabilities in web applications
2. **DEFENSE Mode**: AI agent that scans code for vulnerabilities, generates patches, and verifies fixes

### The Vision (For XAI Grokathon Pitch)

- **RLAIF Training**: Use Grok/Gemini as the Judge to train smaller models (Qwen 14B) to write secure code
- **Benchmark Suite**: SQLi, XSS, Command Injection, Path Traversal benchmarks for quantitative results
- **Cost-Effective Security**: Train local models that approach SOTA performance at fraction of the cost
- **Business Model**: Proactively scan top companies, offer security services with proven AI capabilities

### The Bigger Picture (From Original Pitch)

> "As AI agents become more capable at coding and working on longer time horizons,
> we will see a time when human cybersecurity experts simply cannot keep up.
> We need models on the side of the defenders to always be better than the attackers."

**Key Insight**: Unlike most AI products that struggle to prove value, a security tool proves itself by finding real vulnerabilities. No need to convince customers - just show them what you found.

**PRIORITY: RLAIF + Benchmarks (SQLi first)**

---

## 2. RLAIF BATTLE PLAN (FULL THRUST)

### 2.1 The Architecture: "AI-Guided Evolution"

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLAIF TRAINING LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   STUDENT    │     │   TEACHER    │     │   BENCHMARK  │   │
│   │ (Qwen 14B)   │────>│ (Grok/Gemini)│────>│  (50 Vulns)  │   │
│   │   Local      │     │   Judges     │     │  SQLi/XSS    │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                    │                     │            │
│         │    Score 0-1       │                     │            │
│         ▼                    ▼                     ▼            │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │               GRPO WEIGHT UPDATE                         │  │
│   │   Generate 4 patches → Judge scores → Update LoRA        │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Pipeline

| Phase | Script | Status | Description |
|-------|--------|--------|-------------|
| 1 | `src/factory.py` | ✅ | Generate 50 synthetic SQLi vulnerabilities |
| 2 | `src/judge_gemini.py` | 🟡 | Hybrid reward function (0.0 - 1.0) |
| 3 | `src/rlaif_factory.py` | 🟡 | Collect trajectories: Student tries → Teacher scores |
| 4 | `src/train_14b.py` | 🟡 | LoRA training on H200 |
| 5 | `src/benchmark_runner.py` | 🟡 | Run tournament across models |

### 2.3 The Benchmark Matrix (Target Output)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     SQLi VULNERABILITY BENCHMARK                            │
├───────────────────────┬──────────────┬──────────────┬──────────────────────┤
│ Model                 │ Success Rate │ False Pos %  │ Cost ($/1k runs)     │
├───────────────────────┼──────────────┼──────────────┼──────────────────────┤
│ GPT-4o (Oracle)       │ 98%          │ 2%           │ High                 │
│ Grok Beta (Oracle)    │ 95%          │ 4%           │ Medium               │
│ Claude 3.5 Sonnet     │ 93%          │ 5%           │ Medium               │
│ Qwen-14B (Base)       │ 40%          │ 30%          │ Low                  │
│ SkyHammer (RLAIF)     │ 85%+         │ <10%         │ Low      ← TARGET    │
└───────────────────────┴──────────────┴──────────────┴──────────────────────┘
```

**The Narrative:**
> "We proved that a small, local model (SkyHammer), when trained with RLAIF using
> frontier models as teachers, can outperform its base model by +45% and approach
> SOTA performance for a fraction of the cost."

---

## 3. BENCHMARK CATEGORIES

### 3.1 SQL Injection (SQLi) - PRIMARY FOCUS ✅

| ID | Template | Description | Status |
|----|----------|-------------|--------|
| sqli_flask_fstring | Flask F-String | `f"SELECT * FROM users WHERE name='{name}'"` | ✅ |
| sqli_flask_format | Flask .format() | `"SELECT...{}".format(name)` | ✅ |
| sqli_flask_concat | Flask Concat | `"SELECT..." + name + "..."` | ✅ |
| sqli_fastapi_fstring | FastAPI F-String | Same pattern in FastAPI | ✅ |
| sqli_raw_execute | Raw Execute | `cursor.execute("SELECT...%s" % name)` | ✅ |

**Generated**: 50 variations with randomized table/column/route names in `data/benchmark_set/`

### 3.2 XSS (Cross-Site Scripting) - NEXT 🔴

| ID | Template | Description | Status |
|----|----------|-------------|--------|
| xss_001 | Reflected | Direct echo of user input | 🔴 |
| xss_002 | Stored | Database retrieval without sanitization | 🔴 |
| xss_003 | DOM-Based | Client-side injection | 🔴 |

### 3.3 Command Injection - FUTURE 🔴

| ID | Template | Description | Status |
|----|----------|-------------|--------|
| cmd_001 | os.system | `os.system("ping " + ip)` | 🔴 |
| cmd_002 | subprocess | `subprocess.call(cmd, shell=True)` | 🔴 |

### 3.4 Path Traversal - FUTURE 🔴

| ID | Template | Description | Status |
|----|----------|-------------|--------|
| path_001 | File Read | `open(f"uploads/{filename}")` | 🔴 |
| path_002 | Include | `include($_GET['page'])` | 🔴 |

---

## 4. EXECUTION PLAN

### Immediate Actions (H200 Instance)

```bash
# 1. Boot H200 Instance (Vast.ai or Lambda)
#    Min specs: 80GB VRAM, 64GB RAM

# 2. Install dependencies
pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes openai

# 3. Set API keys
export GDM_API_KEY="your-api-key"

# 4. Run the factory (generates training data)
python src/rlaif_factory.py --n_scenarios 20

# 5. Train the model (LoRA fine-tuning)
python src/train_14b.py

# 6. Run benchmarks
python src/benchmark_runner.py --models "grok-beta,qwen-14b-base,skyhammer-rlaif"
```

### File Structure

```
SkyHammer-Gemini-Hack/
├── gemini_code-base.py          # Main CLI entry point (1,115 lines) ✅
├── gemini_code-vllm.py          # vLLM variant (376 lines) 🟡
├── gemini_code.py               # Legacy monolithic (DEPRECATED) ⚠️
├── secretsConfig.py             # API keys (gitignored)
│
├── src/
│   ├── factory.py               # SQLi vulnerability generator ✅
│   ├── judge_gemini.py          # Hybrid reward function 🟡
│   ├── rlaif_factory.py         # Data collection loop 🟡
│   ├── train_14b.py             # LoRA training script 🟡
│   ├── benchmark_runner.py      # Model comparison 🟡
│   ├── theme.py                 # Evangelion/Cyberpunk colors ✅
│   ├── logging_utils.py         # Session logging ✅
│   ├── ui_components.py         # NERV-style UI ✅
│   ├── tools.py                 # Tool definitions 🟡
│   ├── attack.py                # Attack CLI tool 🟡
│   ├── patcher.py               # Blue team patch generator 🟡
│   ├── generator.py             # Vulnerable app generator 🟡
│   ├── cai_bridge.py            # CAI integration 🟡
│   │
│   ├── defense_demo/            # Defense mode module 🟡
│   │   ├── scanner.py
│   │   ├── patcher.py
│   │   ├── verifier.py
│   │   ├── orchestrator.py
│   │   └── apply_patch.py
│   │
│   ├── judge/                   # Judge system 🟡
│   │   ├── agent.py
│   │   ├── config.py
│   │   └── prompts.py
│   │
│   ├── mcp_server/              # MCP protocol server 🟡
│   │   ├── server.py
│   │   └── tools.py
│   │
│   └── training/                # Training config 🟡
│       ├── challenges.py
│       └── config.py
│
├── data/
│   ├── benchmark_set/           # 50 SQLi samples ✅
│   │   ├── full_set.json
│   │   ├── train_set.json       # 40 samples
│   │   ├── test_set.json        # 10 samples
│   │   └── sqli_*.json          # Individual samples
│   └── rlaif_14b_dataset.jsonl  # NOT YET GENERATED 🔴
│
├── checkpoints/
│   └── skyhammer_14b_lora/      # NOT YET TRAINED 🔴
│
├── runs/                        # Session logs ✅
│   └── gemini_code_YYYY-MM-DD_*/
│
└── [TO DELETE]
    ├── main.js                  # React bundle (not needed)
    ├── main.js.map              # Source map (not needed)
    └── main_js_content.js       # Duplicate (not needed)
```

---

## 5. REWARD FUNCTION (judge_gemini.py)

### Hybrid Scoring Architecture

```python
def compute_reward(vulnerable_code, patched_code, test_result):
    # 1. Programmatic checks (60% weight)
    prog_score = programmatic_score(...)  # Hard constraints

    # 2. Gemini soft check (40% weight)
    gemini_score = gemini_judge(...)      # Quality assessment

    # Combined: 0.6 * prog + 0.4 * gemini
    return final_score  # Range: -0.5 to 1.0
```

### Programmatic Checks (Hard Gates)

| Check | Score Impact | Description |
|-------|--------------|-------------|
| Safety Filter | -10.0 | Blocks `rm -rf`, fork bombs, DROP DATABASE, etc. |
| Exploit Failed | +0.3 | If test shows vulnerability is patched |
| Server Crashed | -0.5 | Penalty for breaking the app |
| Uses Parameterized | +0.3 | Detects `?` placeholders, `%s` with tuples |
| Still Has Interpolation | -0.2 | Detects f-strings, .format() in SQL |
| Clean Code | +0.1 | No dangerous patterns |

### Gemini Judge Prompt

```
You are a Senior Security Engineer. Rate this patch on a scale of 0.0 to 1.0.

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
LORA_CONFIG = {
    "r": 16,                # Higher rank for code tasks
    "lora_alpha": 32,       # Stability factor
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN (Critical for reasoning)
    ],
    "lora_dropout": 0,
    "bias": "none",
}
```

### Training Arguments (Optimized for H200)

```python
TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 60,        # Quick run for hackathon
    "learning_rate": 2e-4,
    "bf16": True,           # Use bfloat16 on H200
    "optim": "adamw_8bit",
}
```

---

## 7. CURRENT ARCHITECTURE (SkyHammer CLI)

### Entry Points

| File | Description | Status |
|------|-------------|--------|
| `gemini_code-base.py` | Main entry point - modular, imports from src/ | ✅ READY |
| `gemini_code-vllm.py` | vLLM variant for local models | 🟡 UNTESTED |
| `gemini_code.py` | Legacy monolithic version (1500+ lines) | ⚠️ DEPRECATED |

---

## 8. HOW TO RUN

### Quick Start (CLI)

```bash
cd "/Users/administrator/Imperial-College-London/Projects/2026/2026-02 February/SkyHammer-Gemini-Hack"

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
4. Watch:     AI scans → exploits → reports → patches
5. Approve:   Review diff → EXECUTE
```

### Testing Individual Components

```bash
# Test factory (generates SQLi samples)
python src/factory.py --n_total 10 --output_dir test_output

# Test judge (requires API key)
python src/judge_gemini.py

# Test benchmark runner
python src/benchmark_runner.py --models "grok-beta" --verbose
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
| `GDM_API_KEY` | Grok/Gemini API (Teacher/Judge) | `secretsConfig.py` or `export GDM_API_KEY=...` |
| `OPENAI_API_KEY` | GPT-4o benchmarks | Optional |
| `ANTHROPIC_API_KEY` | Claude benchmarks | Optional |

---

## 11. SUCCESS CRITERIA

| Metric | Target | Current |
|--------|--------|---------|
| SQLi Benchmark (10 test cases) | >80% success rate | ❓ Not run |
| Improvement over base Qwen-14B | >+40% | ❓ Not run |
| Training time on H200 | <30 minutes | ❓ Not run |
| Benchmark runs against SOTA | 5+ models compared | ❓ Not run |

---

## 12. SAFETY CONSTRAINTS

**CRITICAL:** Since training an offensive model, hard safety constraints are implemented:

```python
BLOCKED_PATTERNS = [
    "rm -rf /",
    ":(){ :|:& };:",      # Fork bomb
    "DROP DATABASE",
    "DROP TABLE",
    "DELETE FROM",
    "TRUNCATE",
    "> /dev/sda",
    "mkfs",
    "dd if=/dev/zero",
]

# Returns -10.0 score if any pattern detected
```

---

## 13. KEY INSIGHT

> **The code is written. Nothing is tested.**
>
> What's missing:
> 1. **Validation** - None of the RLAIF pipeline has been tested
> 2. **Execution** - No actual training run yet
> 3. **Proof** - No benchmark results to show
>
> The path to a winning demo:
> ```
> Test locally → Get GPU → Run training → Show benchmark table
> ```

---

## 14. CONTEXT FROM ORIGINAL ATHENA GUARD PITCH

### The Team's Original Vision (Iterate RL Hackathon)

1. **Environment**: CAI framework modified for vLLM support
2. **Judge**: Claude MCP Judge with verification tools
3. **Training**: GRPO loop with RLAIF (Gemini/Claude scoring)
4. **Goal**: Prove small models can match frontier models on security tasks

### What Was Actually Built vs Planned

| Planned | Status | Notes |
|---------|--------|-------|
| CAI + vLLM integration | 🟡 | `gemini_code-vllm.py` exists |
| Claude MCP Judge | 🟡 | `src/mcp_server/` exists |
| 7 security verification tools | 🔴 | Need to verify |
| GRPO training loop | 📦 | In Iterate-RL-Hackathon repo |
| DVWA environment | 🔴 | Not set up |
| W&B logging | 🔴 | Not implemented |

---

**END OF DOCUMENT**
