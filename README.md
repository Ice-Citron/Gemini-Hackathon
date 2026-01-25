# SkyHammer

**AI-Powered Cybersecurity Agent for Vulnerability Detection, Exploitation, and Remediation**

Built for the **GDM Geminiathon 2026** | Powered by **Gemini**

*Per Aspera Ad Astra*

---

## Overview

SkyHammer is an autonomous cybersecurity AI system that combines offensive security testing with defensive code remediation. It uses Gemini (GDM's frontier model) as its reasoning engine to:

1. **ATTACK**: Automatically discover and exploit security vulnerabilities in web applications
2. **DEFEND**: Generate secure patches and verify fixes
3. **LEARN**: Train smaller models via RLAIF (Reinforcement Learning from AI Feedback) to write more secure code

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SKYHAMMER SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │   ATTACK    │     │   DEFEND    │     │   LEARN     │              │
│   │  Mode       │     │  Mode       │     │  Mode       │              │
│   ├─────────────┤     ├─────────────┤     ├─────────────┤              │
│   │ • Scan code │     │ • Generate  │     │ • RLAIF     │              │
│   │ • Find vulns│     │   patches   │     │   training  │              │
│   │ • Exploit   │     │ • Verify    │     │ • Benchmark │              │
│   │ • Report    │     │   fixes     │     │   models    │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                                         │
│                    Powered by Gemini (GDM)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Interactive CLI with Evangelion/Cyberpunk Aesthetic

- NERV-style system banners and alerts
- Real-time progress spinners
- Split-pane diff visualization
- Color-coded vulnerability reports (RED = vulnerable, GREEN = secure)

### Security Testing Capabilities

| Vulnerability Type | Detection | Exploitation | Auto-Patch |
|-------------------|-----------|--------------|------------|
| SQL Injection | ✅ | ✅ | ✅ |
| Cross-Site Scripting (XSS) | ✅ | ✅ | ✅ |
| Command Injection | ✅ | ✅ | ✅ |
| Path Traversal | ✅ | ✅ | ✅ |
| Hardcoded Secrets | ✅ | - | ✅ |

### Tool Suite

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents from workspace |
| `write_file` | Create or modify files (with diff preview) |
| `run_command` | Execute shell commands (sandboxed) |
| `list_files` | Directory listing with glob patterns |
| `search_code` | Grep-style code search |
| `security_scan` | Deep vulnerability analysis |
| `patch_vulnerability` | Generate secure code patches |

### Safety Features

- **Permission Prompts**: Every file write and command execution requires approval
- **Auto-Backup**: Modified files are automatically backed up (.bak)
- **Instant Rollback**: `/undo` command restores previous versions
- **Workspace Sandboxing**: Operations restricted to designated directory
- **Split-Pane Diff**: Visual comparison before approving changes

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python3 --version

# Install dependencies
pip install rich questionary openai httpx pydantic
```

### API Key Setup

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

Create `secretsConfig.py` in the project root:

```python
GEMINI_API_KEY = "AIza..."  # Your Google Gemini API key
```

Or set as environment variable:

```bash
export GEMINI_API_KEY="AIza..."
```

### Running SkyHammer

```bash
# Interactive mode (recommended)
python3 gemini_code-base.py

# Single command mode
python3 gemini_code-base.py "create a Flask app with user authentication"

# Security scan mode
python3 gemini_code-base.py --scan
```

---

## Usage Guide

### Basic Workflow

```
1. Launch:    python3 gemini_code-base.py

2. Create:    "create a Flask app with a login form that queries a database"
              → Gemini creates the file using write_file tool
              → Shows MISSION COMPLETE banner
              → Recommends security scan

3. Scan:      /skyhammer
              → Select target type (file, directory, or URL)
              → Choose the file to scan

4. Review:    → Gemini analyzes code for vulnerabilities
              → Attempts actual exploits (curl commands)
              → Generates detailed report with:
                - Vulnerability severity [CRITICAL/HIGH/MEDIUM/LOW]
                - BEFORE code (vulnerable) in red context
                - AFTER code (secure) in green context
                - Hacker's Note explaining the exploit

5. Patch:     → Review split-pane diff
              → Approve with EXECUTE
              → Original file is patched in place

6. PR:        /pr "Fixed SQL injection vulnerability"
              → Creates GitHub PR with security fixes
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Display command reference |
| `/skyhammer` | Engage security scanning mode |
| `/model` | Switch Gemini models (with pricing info) |
| `/auto` | Toggle auto-approve mode |
| `/workspace PATH` | Change working directory |
| `/undo` | Rollback file changes |
| `/pr [title]` | Create GitHub Pull Request |
| `/goals` | Show current task status |
| `/clear` | Reset conversation memory |
| `/apikey` | Update API key |
| `/exit` | Terminate session |
| `!command` | Direct bash execution (e.g., `!ls -la`) |

### Authorization Options

When SkyHammer requests permission for an action:

| Option | Description |
|--------|-------------|
| **EXECUTE** | Allow this specific action |
| **EXECUTE ALL** | Auto-approve all actions for this session |
| **DENY** | Skip this action |
| **REVISE** | Provide alternative instructions |

---

## Available Gemini Models

| Model ID | Price (per 1M tokens) | Best For |
|----------|----------------------|----------|
| `gemini-3-flash-preview` | $0.50 / $3.00 | **Default** - Latest, 3x faster than 2.5 Pro |
| `gemini-3-pro-preview` | $2.00 / $12.00 | Most capable, complex reasoning |
| `gemini-2.5-pro` | $1.25 / $10.00 | Strong reasoning, 1M+ context |
| `gemini-2.5-flash` | $0.15 / $0.60 | Fast with optional thinking mode |
| `gemini-2.5-flash-lite` | $0.10 / $0.40 | Most economical |
| `gemini-2.0-flash` | $0.10 / $0.40 | Fast, multimodal |

Switch models interactively with `/model` command.

---

## Vulnerability Report Format

SkyHammer generates CLI-friendly reports (no markdown):

```
============================================================
VULNERABILITY REPORT: app.py
============================================================

[CRITICAL] SQL Injection in /login endpoint
  Status: EXPLOITED
  Proof: curl -X POST "http://localhost:5000/login" \
         -d "username=admin' OR '1'='1'--&password=x"

BEFORE (vulnerable):
  query = f"SELECT * FROM users WHERE username = '{username}'"
  cursor.execute(query)

AFTER (secure):
  query = "SELECT * FROM users WHERE username = ?"
  cursor.execute(query, (username,))

Hacker's Note:
  The attacker input "admin' OR '1'='1'--" manipulates the SQL query.
  The OR '1'='1' condition is always TRUE, causing the query to return
  all users regardless of the password. The -- comments out the rest
  of the query, bypassing the password check entirely.

============================================================
```

---

## RLAIF Training Pipeline

SkyHammer includes infrastructure for training smaller models to write secure code using Reinforcement Learning from AI Feedback (RLAIF).

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLAIF TRAINING LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   STUDENT    │     │   TEACHER    │     │   BENCHMARK  │   │
│   │ (Qwen 14B)   │────▶│ (Gemini API)   │────▶│  (50 Vulns)  │   │
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

### Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `src/factory.py` | Generate synthetic SQLi vulnerabilities |
| `src/judge_gemini.py` | Gemini-based reward function (0.0 - 1.0) |
| `src/rlaif_factory.py` | Collect training data: Student generates, Teacher judges |
| `src/train_14b.py` | LoRA fine-tuning on GPU |
| `src/benchmark_runner.py` | Compare models across test set |

### Running RLAIF Training (GPU Required)

```bash
# 1. Install GPU dependencies
pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 2. Generate benchmark set
python src/factory.py --n_total 50 --n_test 10

# 3. Collect training data (requires GPU)
python src/rlaif_factory.py --n_scenarios 20

# 4. Train LoRA adapter
python src/train_14b.py --max_steps 60

# 5. Run benchmarks
python src/benchmark_runner.py --models "gemini-beta,gemini-4-1-fast-reasoning"
```

### Reward Function

The hybrid reward function combines:

1. **Programmatic Checks** (60%)
   - Uses parameterized queries? (+0.3)
   - No string interpolation? (+0.1)
   - Exploit failed after patch? (+0.3)
   - Broke the application? (-0.5)

2. **Gemini Judge** (40%)
   - Code quality assessment
   - Best practices evaluation
   - Logic preservation check

```python
def compute_reward(vulnerable_code, patched_code, test_result):
    # Programmatic: 60%
    prog_score = check_parameterized() + check_no_interpolation() + check_exploit_failed()

    # Gemini Judge: 40%
    gemini_score = gemini_api_evaluate(vulnerable_code, patched_code)

    return (prog_score * 0.6) + (gemini_score * 0.4)
```

### Benchmark Results Target

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

---

## Project Structure

```
GDM-Geminiathon/
├── gemini_code-base.py          # Main CLI entry point
├── gemini_code.py               # Legacy monolithic version
├── secretsConfig.py           # API keys (gitignored)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── PROJECT_STATUS.md          # Detailed project documentation
│
├── src/
│   ├── __init__.py
│   ├── theme.py               # Evangelion/Cyberpunk color palette
│   ├── logging_utils.py       # Session logging to runs/
│   ├── ui_components.py       # NERV-style UI components
│   ├── tools.py               # Tool definitions and execution
│   │
│   ├── factory.py             # SQLi benchmark generator
│   ├── judge_gemini.py          # Gemini-based reward function
│   ├── rlaif_factory.py       # RLAIF data collection
│   ├── train_14b.py           # LoRA training script
│   ├── benchmark_runner.py    # Model comparison
│   │
│   ├── attack.py              # Standalone attack tool
│   ├── patcher.py             # Standalone patching tool
│   └── generator.py           # Vulnerable app generator
│
├── data/
│   ├── benchmark_set/         # Generated SQLi test cases
│   │   ├── train_set.json     # 40 training samples
│   │   ├── test_set.json      # 10 test samples
│   │   └── sqli_*.json        # Individual samples
│   └── rlaif_14b_dataset.jsonl # RLAIF training data
│
├── runs/                      # Session logs (auto-generated)
│   └── gemini_code_YYYY-MM-DD_HH-MM-SS/
│       ├── session.jsonl      # Full conversation log
│       └── reports/           # Saved vulnerability reports
│
├── checkpoints/               # Trained model weights
│   └── skyhammer_14b_lora/    # LoRA adapter
│
└── testing-ground/            # Test vulnerable applications
    ├── mock_dvwa.py
    └── synthetic_*.py
```

---

## Demo Video Script

### Suggested Flow (3-5 minutes)

1. **Intro** (30s)
   - Show the NERV-style banner
   - Explain: "SkyHammer is an AI security agent powered by Gemini"

2. **Create Vulnerable App** (60s)
   - Type: "create a Flask app with a login form that checks credentials against a SQLite database"
   - Show Gemini calling `write_file` tool
   - Show the MISSION COMPLETE banner recommending `/skyhammer`

3. **SkyHammer Attack** (90s)
   - Type: `/skyhammer`
   - Select "Local file" → choose the created app
   - Watch Gemini:
     - Read the file
     - Identify SQL injection
     - Start the server in background
     - Run curl exploit command
     - Generate vulnerability report

4. **Review & Patch** (60s)
   - Show the BEFORE/AFTER code comparison
   - Show the split-pane diff
   - Approve the patch
   - Show the patched file

5. **Benchmark Results** (30s)
   - Show the benchmark comparison table
   - Explain RLAIF training potential

---

## API Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | No | For GPT model benchmarks |
| `ANTHROPIC_API_KEY` | No | For Claude model benchmarks |

### Tool Schemas

```python
# read_file
{"path": "string"}  # File path to read

# write_file
{"path": "string", "content": "string"}  # Path and content

# run_command
{"command": "string"}  # Shell command to execute

# list_files
{"path": "string", "pattern": "string"}  # Directory and glob

# search_code
{"pattern": "string", "file_pattern": "string"}  # Search term and file glob

# security_scan
{"target": "string", "scan_type": "sqli|xss|cmd|lfi|all"}

# patch_vulnerability
{"file_path": "string", "vulnerability": "string"}
```

---

## Troubleshooting

### Common Issues

**"GEMINI_API_KEY missing"**
```bash
# Set in environment
export GEMINI_API_KEY="AIza..."

# Or create secretsConfig.py
echo 'GEMINI_API_KEY = "AIza..."' > secretsConfig.py
```

**"Rich library required"**
```bash
pip install rich questionary
```

**"Access denied - path outside workspace"**
- SkyHammer sandboxes file operations to the workspace directory
- Use `/workspace /path/to/dir` to change the working directory

**"Command blocked for safety"**
- Certain dangerous commands are blocked (rm -rf /, fork bombs, etc.)
- This is intentional for safety

### GPU Training Issues

**"CUDA out of memory"**
- Reduce batch size in `src/train_14b.py`
- Use 4-bit quantization (already enabled)

**"Unsloth not available"**
```bash
pip install unsloth "xformers<0.0.27"
```

---

## Security Considerations

SkyHammer is designed for **authorized security testing only**. It includes:

- Workspace sandboxing
- Dangerous command blocking
- Permission prompts for all actions
- Full audit logging

**Do not use SkyHammer on systems you do not own or have explicit authorization to test.**

---

## Contributing

This project was built for the GDM Geminiathon 2026. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **GDM** for Gemini API access and hosting the Geminiathon
- **Anthropic** for Claude (benchmark comparison)
- **OpenAI** for GPT-4o (benchmark comparison)
- **Unsloth** for efficient LoRA training
- **Rich** for the beautiful terminal UI

---

## Contact

Built by the SkyHammer team for GDM Geminiathon 2026 in Central London.

---

**"Security through Intelligence, Defense through Understanding"**

```
 ███████╗██╗  ██╗██╗   ██╗██╗  ██╗ █████╗ ███╗   ███╗███╗   ███╗███████╗██████╗
 ██╔════╝██║ ██╔╝╚██╗ ██╔╝██║  ██║██╔══██╗████╗ ████║████╗ ████║██╔════╝██╔══██╗
 ███████╗█████╔╝  ╚████╔╝ ███████║███████║██╔████╔██║██╔████╔██║█████╗  ██████╔╝
 ╚════██║██╔═██╗   ╚██╔╝  ██╔══██║██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══╝  ██╔══██╗
 ███████║██║  ██╗   ██║   ██║  ██║██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║███████╗██║  ██║
 ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
                         Powered by Gemini | GDM Geminiathon 2026
```
