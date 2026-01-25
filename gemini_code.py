#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SKYHAMMER // GEMINI-CODE                                                       ║
║  Evangelion/Cyberpunk Aesthetics for Autonomous Security                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

An autonomous AI system for security testing, code generation, and remediation.
Features:
- Interactive chat with codebase context
- Code generation and editing
- Security scanning (attack mode)
- Auto-patching (defense mode)
- Tool use (file operations, shell commands)

Usage:
    python gemini_code.py                    # Interactive mode
    python gemini_code.py "fix the bug"      # Single command mode
    python gemini_code.py --scan             # Security scan mode

Models:
    - gemini-code-fast-1: Code generation/editing (fast)
    - gemini-4-1-fast-reasoning: Complex reasoning/security
"""

import os
import sys
import json
import subprocess
import glob
from typing import Optional, List, Dict, Any
from openai import OpenAI

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.text import Text
    from rich.table import Table
    from rich import box
    from rich.style import Style
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("CRITICAL ERROR: SYSTEM DEPENDENCIES MISSING.")
    print("RUN: pip install rich questionary")
    sys.exit(1)

# =============================================================================
# THEME CONFIGURATION (EVANGELION / CYBERPUNK NEON)
# =============================================================================
# Color palette inspired by Neon Genesis Evangelion + Cyberpunk aesthetics
C_PRIMARY = "bold yellow"       # Main warnings / headers (NERV UI panels)
C_SECONDARY = "cyan"            # Information / borders / accents
C_ACCENT = "bold orange1"       # Critical alerts / attention items
C_SUCCESS = "bold green"        # Operations success / patched code
C_ERROR = "bold red"            # Failures / Attacks / vulnerabilities
C_DIM = "dim white"             # Background info / timestamps
C_HIGHLIGHT = "black on yellow" # High contrast highlight (Evangelion style)

# Load API Key
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

if not GDM_API_KEY:
    print("[!] FATAL: NEURAL LINK SEVERED (GDM_API_KEY missing)")
    print("    Set GDM_API_KEY in secretsConfig.py or environment")
    sys.exit(1)

console = Console() if HAS_RICH else None

# =============================================================================
# NERV-STYLE UI FUNCTIONS
# =============================================================================

def print_banner():
    """Display the Main System Banner - NERV/Evangelion Style"""
    if not console:
        return

    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" SYSTEM ONLINE // GEMINI-CODE INTERFACE ", justify="center", style="bold black on yellow"),
            style="bold yellow",
            border_style="yellow",
            box=box.HEAVY,
        )
    )
    console.print(grid)
    console.print(f"[{C_DIM}]:: NEURAL LINK ESTABLISHED :: MODEL: {CURRENT_MODEL} ::[/{C_DIM}]", justify="center")
    console.print()


def print_skyhammer_banner():
    """Display the SkyHammer Activation Banner - Red Alert Style"""
    if not console:
        return

    console.print()
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" ⚠ WARNING: SKYHAMMER PROTOCOL ENGAGED ⚠ ", justify="center", style="bold white on red"),
            subtitle="[ OFFENSIVE SECURITY AUTHORIZED ]",
            style="bold red",
            border_style="bold red",
            box=box.DOUBLE,
            padding=(1, 2)
        )
    )
    console.print(grid)
    console.print(f"[{C_ACCENT}]>> SCANNING MODULES: ACTIVE | EXPLOIT ENGINE: ONLINE | PATCHER: READY <<[/{C_ACCENT}]", justify="center")
    console.print()

# Initialize client
client = OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")

# Available models with pricing (per million tokens: input/output)
MODELS = {
    "gemini-4-1-fast-reasoning": {"price": "$0.20/$0.50", "desc": "Best value, reasoning enabled"},
    "gemini-4-1-fast-non-reasoning": {"price": "$0.20/$0.50", "desc": "Fast, no chain-of-thought"},
    "gemini-code-fast-1": {"price": "$0.20/$1.50", "desc": "Specialized for code"},
    "gemini-4-fast-reasoning": {"price": "$0.20/$0.50", "desc": "Reasoning model"},
    "gemini-4-fast-non-reasoning": {"price": "$0.20/$0.50", "desc": "Fast general model"},
    "gemini-3-mini": {"price": "$0.30/$0.50", "desc": "Lightweight, fast"},
    "gemini-3-latest": {"price": "$0.30/$0.50", "desc": "Stable release"},
}

# Current settings
CURRENT_MODEL = "gemini-4-1-fast-reasoning"
SKYHAMMER_MODE = False  # When True, enables security tools and attack/defense capabilities
WORKSPACE_DIR = os.getcwd()  # Sandbox - only allow operations within this directory
AUTO_APPROVE = False  # Skip permission prompts if True
INTERRUPT_REQUESTED = False  # Flag to interrupt agentic loop
LISTENER_PAUSED = False  # Pause listener during prompts
import difflib  # For diff highlighting
import threading
from datetime import datetime

# Platform-specific imports for ESC key detection
try:
    import select
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False  # Windows

# Logging setup
RUN_DIR = None  # Will be set when session starts
LOG_FILE = None  # Combined log file for tool calls, API calls, and conversation

# Task tracking for /goals
CURRENT_TASKS = []  # List of current tasks Gemini is working on
CURRENT_TASK_STATUS = "idle"  # idle, thinking, executing, done

# Backup system for /undo
FILE_BACKUPS = {}  # {filepath: original_content} - for rollback

# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                    "pattern": {"type": "string", "description": "Glob pattern", "default": "*"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for text in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text to search for"},
                    "file_pattern": {"type": "string", "description": "File glob pattern", "default": "*.py"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "security_scan",
            "description": "Run security scan on a file or URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "File path or URL to scan"},
                    "scan_type": {"type": "string", "enum": ["sqli", "xss", "cmd", "lfi", "all"], "default": "all"}
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch_vulnerability",
            "description": "Generate a security patch for a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File to patch"},
                    "vulnerability": {"type": "string", "description": "Type of vulnerability to fix"}
                },
                "required": ["file_path"]
            }
        }
    }
]


def init_logging():
    """Initialize logging directory for this session"""
    global RUN_DIR, LOG_FILE

    # Create runs directory if it doesn't exist
    runs_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_base, exist_ok=True)

    # Create session directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_DIR = os.path.join(runs_base, f"gemini_code_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)

    # Create log file
    LOG_FILE = os.path.join(RUN_DIR, "session.jsonl")

    # Log session start
    log_event("session_start", {
        "timestamp": datetime.now().isoformat(),
        "model": CURRENT_MODEL,
        "workspace": WORKSPACE_DIR
    })

    if console:
        console.print(f"[{C_DIM}]SESSION LOG: {RUN_DIR}[/{C_DIM}]")


def log_event(event_type: str, data: Dict[str, Any]):
    """Log an event to the session log file"""
    global LOG_FILE
    if not LOG_FILE:
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **data
    }

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        pass  # Don't crash on logging errors


def log_tool_call(tool_name: str, args: Dict[str, Any], result: str, duration_ms: float = 0):
    """Log a tool call"""
    log_event("tool_call", {
        "tool": tool_name,
        "arguments": args,
        "result_preview": result[:500] if result else "",
        "result_length": len(result) if result else 0,
        "duration_ms": duration_ms
    })


def log_api_call(model: str, messages_count: int, response_content: str, tool_calls: List[str] = None):
    """Log an API call to Gemini"""
    log_event("api_call", {
        "model": model,
        "messages_count": messages_count,
        "response_preview": response_content[:500] if response_content else "",
        "tool_calls": tool_calls or []
    })


def log_conversation(role: str, content: str):
    """Log a conversation message"""
    log_event("conversation", {
        "role": role,
        "content": content[:2000] if content else ""
    })


def pause_listener():
    """Pause the ESC listener during prompts"""
    global LISTENER_PAUSED
    LISTENER_PAUSED = True


def resume_listener():
    """Resume the ESC listener after prompts"""
    global LISTENER_PAUSED
    LISTENER_PAUSED = False


def set_task_status(status: str, tasks: List[str] = None):
    """Update current task status and optionally the task list"""
    global CURRENT_TASK_STATUS, CURRENT_TASKS
    CURRENT_TASK_STATUS = status
    if tasks is not None:
        CURRENT_TASKS = tasks


def show_goals():
    """Display current goals/tasks - NERV Mission Status"""
    if not console:
        return

    if not CURRENT_TASKS and CURRENT_TASK_STATUS == "idle":
        console.print(f"[{C_DIM}]>> NO ACTIVE MISSIONS. Awaiting orders. <<[/{C_DIM}]")
        return

    status_colors = {
        "idle": C_DIM,
        "thinking": C_PRIMARY,
        "executing": C_SECONDARY,
        "done": C_SUCCESS
    }
    color = status_colors.get(CURRENT_TASK_STATUS, "white")

    console.print(f"\n[{C_PRIMARY}]:: MISSION STATUS ::[/{C_PRIMARY}] [{color}]{CURRENT_TASK_STATUS.upper()}[/{color}]")

    if CURRENT_TASKS:
        console.print(f"[{C_PRIMARY}]ACTIVE OBJECTIVES:[/{C_PRIMARY}]")
        for i, task in enumerate(CURRENT_TASKS, 1):
            console.print(f"  [{C_SECONDARY}]{i}.[/{C_SECONDARY}] {task}")
    console.print()


def start_interrupt_listener():
    """Start interrupt handling - uses Ctrl+C (SIGINT)"""
    global INTERRUPT_REQUESTED
    INTERRUPT_REQUESTED = False
    # No background thread needed - Ctrl+C works via KeyboardInterrupt
    return None


def stop_interrupt_listener():
    """Reset interrupt flag"""
    global INTERRUPT_REQUESTED, LISTENER_PAUSED
    INTERRUPT_REQUESTED = False
    LISTENER_PAUSED = False


def show_diff(old_content: str, new_content: str, filename: str):
    """
    Evangelion-style split-pane diff view.
    Shows side-by-side comparison with NERV UI aesthetics.
    """
    if not console:
        return

    console.print(f"\n[{C_PRIMARY}]:: DETECTED FILE MODIFICATION :: {filename}[/{C_PRIMARY}]")

    if old_content == new_content:
        console.print(f"[{C_DIM}]>> No logical changes detected.[/{C_DIM}]")
        return

    # Determine syntax language
    ext = filename.split('.')[-1] if '.' in filename else 'python'
    lang_map = {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'bash', 'md': 'markdown', 'rb': 'ruby', 'php': 'php'}
    lang = lang_map.get(ext, ext)

    # Create split-pane layout table
    layout = Table(show_header=True, header_style="bold white", box=box.ROUNDED, expand=True, border_style="yellow")
    layout.add_column(f"[{C_ERROR}]◄ ORIGINAL[/{C_ERROR}] // {filename}", style="dim red", ratio=1)
    layout.add_column(f"[{C_SUCCESS}]PROPOSED PATCH ►[/{C_SUCCESS}] // {filename}", style="dim green", ratio=1)

    # Render Syntax blocks side by side
    try:
        syntax_old = Syntax(old_content, lang, theme="monokai", line_numbers=True, word_wrap=True)
        syntax_new = Syntax(new_content, lang, theme="monokai", line_numbers=True, word_wrap=True)
        layout.add_row(syntax_old, syntax_new)
    except:
        layout.add_row(old_content, new_content)

    console.print(layout)

    # Unified Diff (The "Hacker" View) - Delta Analysis
    console.print(f"\n[{C_SECONDARY}]:: DELTA ANALYSIS ::[/{C_SECONDARY}]")

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    diff_lines = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))

    if diff_lines:
        diff_table = Table(box=box.SIMPLE, show_header=False, show_edge=False, border_style="dim blue")
        diff_table.add_column("Line")

        for line in diff_lines[2:]:  # Skip headers
            if line.startswith('@@'):
                diff_table.add_row(Text(line, style="bold blue"))
            elif line.startswith('+'):
                diff_table.add_row(Text(line, style=f"bold {C_SUCCESS} on black"))
            elif line.startswith('-'):
                diff_table.add_row(Text(line, style=f"strike dim red"))
            # Skip context lines to keep UI clean

        console.print(Panel(diff_table, border_style="dim blue", title="[Change Manifest]", title_align="left"))
    console.print()


def show_new_file_preview(content: str, filename: str):
    """Show new file with Evangelion-style green panel - FULL content"""
    if not console:
        return

    console.print(f"\n[{C_PRIMARY}]:: NEW FILE CREATION :: {filename}[/{C_PRIMARY}]")

    # Determine language for syntax highlighting
    ext = filename.split('.')[-1] if '.' in filename else 'python'
    lang_map = {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'bash', 'md': 'markdown', 'rb': 'ruby', 'php': 'php'}
    lang = lang_map.get(ext, ext)

    try:
        syntax = Syntax(content, lang, line_numbers=True, word_wrap=True, theme="monokai")
        console.print(Panel(
            syntax,
            title=f"[{C_SUCCESS}]+ NEW FILE: {filename}[/{C_SUCCESS}]",
            border_style="green",
            box=box.ROUNDED
        ))
    except:
        # Fallback to plain text
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            console.print(f"[{C_SUCCESS}]+{i:3}| {line}[/{C_SUCCESS}]")
    console.print()


def ask_permission(tool_name: str, args: Dict[str, Any], preview_content: str = "") -> tuple:
    """
    Evangelion-style Permission Prompt.
    Returns (action, feedback) where action is 'yes', 'no', 'yes_all', or 'feedback'
    """
    global AUTO_APPROVE

    if AUTO_APPROVE:
        return ("yes", None)

    # Pause ESC listener during prompt
    pause_listener()

    try:
        # Visual separation - NERV intervention style
        console.print(f"\n[{C_ACCENT}]>> INTERVENTION REQUIRED: {tool_name.upper()} <<[/{C_ACCENT}]")

        # Show preview based on tool
        if tool_name == "write_file":
            path = args.get("path", "")
            full_path = os.path.join(WORKSPACE_DIR, path) if not os.path.isabs(path) else path
            content = args.get("content", "")
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    old = f.read()
                show_diff(old, content, path)
            else:
                show_new_file_preview(content, path)

        elif tool_name == "run_command":
            cmd = args.get('command', '')
            console.print(Panel(
                f"$ {cmd}",
                title="[bold yellow]EXECUTE SHELL[/]",
                border_style="yellow",
                style="bold yellow"
            ))

        else:
            console.print(f"[{C_DIM}]{json.dumps(args)[:200]}[/{C_DIM}]")

        # Questionary Prompt with Evangelion/Cyberpunk styling
        eva_style = questionary.Style([
            ('qmark', 'fg:yellow bold'),
            ('question', 'fg:yellow bold'),
            ('answer', 'fg:cyan bold'),
            ('pointer', 'fg:cyan bold'),
            ('highlighted', 'fg:cyan bold'),
            ('selected', 'fg:cyan bold'),
            ('separator', 'fg:black'),
            ('instruction', 'fg:black'),
        ])

        choice = questionary.select(
            "AUTHORIZE ACTION?",
            choices=[
                "EXECUTE",
                "EXECUTE ALL (Session Override)",
                "DENY",
                "REVISE (Provide Instructions)"
            ],
            style=eva_style,
            pointer=">"
        ).ask()

        if not choice:
            return ("no", None)
        if choice == "EXECUTE":
            return ("yes", None)
        if "EXECUTE ALL" in choice:
            AUTO_APPROVE = True
            return ("yes", None)
        if "REVISE" in choice:
            feedback = questionary.text(
                "TACTICAL ADJUSTMENT:",
                style=eva_style
            ).ask()
            return ("feedback", feedback)
        return ("no", None)

    finally:
        # Resume ESC listener
        resume_listener()


def is_safe_path(path: str) -> bool:
    """Check if path is within the workspace (sandbox)"""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(WORKSPACE_DIR)


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool and return the result"""
    global WORKSPACE_DIR
    import time
    start_time = time.time()

    try:
        if name == "read_file":
            path = args.get("path", "")
            # Make relative paths absolute within workspace
            if not os.path.isabs(path):
                path = os.path.join(WORKSPACE_DIR, path)
            if not is_safe_path(path):
                result = f"Error: Access denied - path outside workspace: {path}"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                if console:
                    console.print(f"[{C_DIM}]< READ: {os.path.basename(path)} ({len(content)} bytes) >[/{C_DIM}]")
                result = f"Contents of {path}:\n```\n{content[:3000]}\n```"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            result = f"Error: File not found: {path}"
            log_tool_call(name, args, result, (time.time() - start_time) * 1000)
            return result

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            # Ask permission with diff preview
            action, feedback = ask_permission("write_file", args)
            if action == "no":
                result = "Action skipped by user"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            elif action == "feedback":
                result = f"USER FEEDBACK: {feedback}\n\nPlease try again with the user's feedback in mind."
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            # Make relative paths absolute within workspace
            if not os.path.isabs(path):
                path = os.path.join(WORKSPACE_DIR, path)
            if not is_safe_path(path):
                result = f"Error: Access denied - path outside workspace: {path}"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result

            # BACKUP: Save original content for /undo (if file exists)
            if os.path.exists(path):
                with open(path, "r") as f:
                    FILE_BACKUPS[path] = f.read()
                # Also save .bak file
                with open(path + ".bak", "w") as f:
                    f.write(FILE_BACKUPS[path])
                if console:
                    console.print(f"[{C_DIM}]💾 BACKUP CREATED: {path}.bak[/{C_DIM}]")

            # Create parent directories if needed
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            if console:
                console.print(f"[{C_SUCCESS}]✓ WRITE CONFIRMED: {os.path.basename(args.get('path', path))}[/{C_SUCCESS}]")
            result = f"SUCCESS: Wrote {len(content)} bytes to {args.get('path', path)}"
            log_tool_call(name, args, result, (time.time() - start_time) * 1000)
            return result

        elif name == "list_files":
            path = args.get("path", ".")
            pattern = args.get("pattern", "*")
            if not os.path.isabs(path):
                path = os.path.join(WORKSPACE_DIR, path)
            files = glob.glob(os.path.join(path, pattern))
            return f"Files matching '{pattern}' in {path}:\n" + "\n".join(files[:50])

        elif name == "run_command":
            command = args.get("command", "")
            # Safety check
            dangerous = ["rm -rf /", "sudo rm", "mkfs", "> /dev", ":(){ :|:& };:"]
            if any(d in command for d in dangerous):
                result = "Error: Command blocked for safety"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            # Ask permission
            action, feedback = ask_permission("run_command", args)
            if action == "no":
                result = "Action skipped by user"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            elif action == "feedback":
                result = f"USER FEEDBACK: {feedback}\n\nPlease try a different approach based on the user's feedback."
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            if console:
                console.print(f"[{C_SECONDARY}]>> EXECUTING: {command}[/{C_SECONDARY}]")
            proc_result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=120, cwd=WORKSPACE_DIR
            )
            output = proc_result.stdout + proc_result.stderr
            # Styled Output Panel
            if console and (proc_result.stdout or proc_result.stderr):
                out_panel = Panel(
                    output[:2000] if len(output) > 2000 else output,
                    title=f"EXIT CODE: {proc_result.returncode}",
                    border_style="green" if proc_result.returncode == 0 else "red"
                )
                console.print(out_panel)
            elif console:
                console.print(f"[{C_DIM}](No output)[/{C_DIM}]")
            result = f"Command output (exit {proc_result.returncode}):\n```\n{output[:2000]}\n```"
            log_tool_call(name, args, result, (time.time() - start_time) * 1000)
            return result

        elif name == "search_code":
            pattern = args.get("pattern", "")
            file_pattern = args.get("file_pattern", "*.py")
            result = subprocess.run(
                f'grep -rn "{pattern}" --include="{file_pattern}" .',
                shell=True, capture_output=True, text=True, timeout=10,
                cwd=WORKSPACE_DIR
            )
            return f"Search results for '{pattern}':\n```\n{result.stdout[:2000]}\n```"

        elif name == "security_scan":
            target = args.get("target", "")
            scan_type = args.get("scan_type", "all")
            # Run attack.py in background
            if target.startswith("http"):
                result = subprocess.run(
                    f'python src/attack.py --dvwa "{target}" --challenge sqli --max-turns 5 --quiet',
                    shell=True, capture_output=True, text=True, timeout=60
                )
            else:
                result = subprocess.run(
                    f'python src/patcher.py --source "{target}" --vuln "Security Audit"',
                    shell=True, capture_output=True, text=True, timeout=60
                )
            return f"Security scan results:\n{result.stdout[:2000]}"

        elif name == "patch_vulnerability":
            file_path = args.get("file_path", "")
            vuln = args.get("vulnerability", "Unknown")
            result = subprocess.run(
                f'python src/patcher.py --source "{file_path}" --vuln "{vuln}"',
                shell=True, capture_output=True, text=True, timeout=120
            )
            return f"Patch result:\n{result.stdout[:2000]}"

        result = f"Unknown tool: {name}"
        log_tool_call(name, args, result, (time.time() - start_time) * 1000)
        return result

    except Exception as e:
        result = f"Tool error: {str(e)}"
        log_tool_call(name, args, result, (time.time() - start_time) * 1000)
        return result


def get_codebase_context() -> str:
    """Get context about the current codebase"""
    context_parts = []

    # List Python files
    py_files = glob.glob("*.py")
    if py_files:
        context_parts.append(f"Python files in current directory: {', '.join(py_files)}")

    # Read key files
    for important in ["README.md", "requirements.txt", "setup.py", "pyproject.toml"]:
        if os.path.exists(important):
            with open(important, "r") as f:
                content = f.read()[:500]
            context_parts.append(f"\n--- {important} ---\n{content}")

    return "\n".join(context_parts) if context_parts else "No context files found."


def chat_completion(messages: List[Dict], use_tools: bool = True) -> str:
    """Run agentic chat completion with cyberpunk spinner - loops until Gemini is done or max iterations"""
    global CURRENT_MODEL, SKYHAMMER_MODE, INTERRUPT_REQUESTED
    model_id = CURRENT_MODEL
    max_iterations = 25  # Safety limit
    iteration = 0

    # Select tools based on mode
    if use_tools:
        basic_tools = [t for t in TOOLS if t["function"]["name"] in
                       ["read_file", "write_file", "list_files", "run_command", "search_code"]]
        if SKYHAMMER_MODE:
            active_tools = TOOLS
        else:
            active_tools = basic_tools
    else:
        active_tools = None

    # Ctrl+C can be used to interrupt
    if console:
        console.print(f"[{C_DIM}](Press Ctrl+C to interrupt, /goals to see tasks)[/{C_DIM}]")

    set_task_status("thinking")

    try:
        while iteration < max_iterations:
            iteration += 1

            # Check for interrupt flag
            if INTERRUPT_REQUESTED:
                set_task_status("idle")
                return "(Interrupted by user)"

            set_task_status("thinking")

            # Cyberpunk Spinner for API calls
            with Progress(
                SpinnerColumn(spinner_name="dots12", style="bold yellow"),
                TextColumn(f"[{C_PRIMARY}]{{task.description}}[/{C_PRIMARY}]"),
                transient=True
            ) as progress:
                task_id = progress.add_task(f"SYNCING WITH NEURAL NETWORK [CYCLE {iteration}]...", total=None)

                kwargs = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 4096,
                }
                if active_tools:
                    kwargs["tools"] = active_tools
                    kwargs["tool_choice"] = "auto"

                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

            # Log API call
            tool_call_names = [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else []
            log_api_call(model_id, len(messages), msg.content or "", tool_call_names)

            # Update tasks from tool calls
            if tool_call_names:
                set_task_status("executing", tool_call_names)

            # If no tool calls, we're done
            if not msg.tool_calls:
                set_task_status("done")
                return msg.content or "(No response)"

            # Check for interrupt before processing tools
            if INTERRUPT_REQUESTED:
                set_task_status("idle")
                return "(Interrupted by user)"

            # Process tool calls
            if console:
                console.print(f"[{C_DIM}]>> CYCLE {iteration}: {len(msg.tool_calls)} TOOL CALLS <<[/{C_DIM}]")

            tool_results = []
            for tc in msg.tool_calls:
                # Check for interrupt between tool calls
                if INTERRUPT_REQUESTED:
                    return "(Interrupted by user mid-execution)"

                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except:
                    tool_args = {}

                if console:
                    console.print(f"[{C_SECONDARY}]→ {tool_name}[/{C_SECONDARY}]")

                result = execute_tool(tool_name, tool_args)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })
            messages.extend(tool_results)

            # Show intermediate thoughts if any
            if msg.content and console:
                console.print(f"[{C_DIM} italic]{msg.content[:200]}...[/{C_DIM}]" if len(msg.content or "") > 200 else f"[{C_DIM} italic]{msg.content}[/{C_DIM}]")

        return "(Max iterations reached - use /clear to reset)"

    except KeyboardInterrupt:
        console.print(f"\n[{C_ACCENT}]>> INTERRUPT SIGNAL RECEIVED <<[/{C_ACCENT}]")
        return "(Interrupted by user)"


def interactive_mode():
    """Run interactive chat mode with Evangelion/Cyberpunk UI"""
    global CURRENT_MODEL, SKYHAMMER_MODE, WORKSPACE_DIR, AUTO_APPROVE, client, GDM_API_KEY

    if not console:
        print("Rich library required for interactive mode")
        return

    # Initialize logging
    init_logging()

    # Display NERV-style banner
    print_banner()

    auto_status = f"[{C_PRIMARY}]ON[/{C_PRIMARY}]" if AUTO_APPROVE else f"[{C_DIM}]OFF[/{C_DIM}]"
    console.print(f"[{C_DIM}]COMMANDS: /help, /auto, /workspace, /skyhammer, /model, /clear, /exit[/{C_DIM}]")
    console.print(f"[{C_DIM}]BASH MODE: !command (e.g. !ls, !pwd, !python app.py &)[/{C_DIM}]")
    console.print(f"[{C_DIM}]MODEL: {CURRENT_MODEL} | AUTO-APPROVE: {auto_status}[/{C_DIM}]")
    console.print(f"[{C_DIM}]WORKSPACE: {WORKSPACE_DIR}[/{C_DIM}]")
    console.print()

    # Build system prompt with context
    context = get_codebase_context()
    system_prompt = f"""You are Gemini Code, an AI coding assistant with file system access.

WORKSPACE: {WORKSPACE_DIR}
(All file operations are sandboxed to this directory)

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:

1. ALWAYS USE TOOLS TO PERFORM ACTIONS - Never just output code!
   - To create a file: CALL write_file tool with path and content
   - To read a file: CALL read_file tool
   - To run a command: CALL run_command tool
   - To list files: CALL list_files tool

2. DO NOT just print code in your response. Actually execute write_file!

3. When user asks to "create a file" or "write code":
   - Call write_file(path="filename.py", content="...code...")
   - The tool will actually create the file on disk

4. When user asks to "run" something:
   - Call run_command(command="python script.py")
   - The tool will actually execute it

5. After using a tool, confirm what you did briefly.

AVAILABLE TOOLS:
- write_file(path, content): Create/overwrite a file - USE THIS TO WRITE CODE
- read_file(path): Read file contents
- run_command(command): Execute shell command (runs in workspace)
- list_files(path, pattern): List directory contents
- search_code(pattern, file_pattern): Search for text in files

Current codebase:
{context}

IMPORTANT - RUNNING SERVERS:
When starting a web server (Flask, FastAPI, uvicorn, etc.), ALWAYS run in background:
- Use: nohup python app.py > app.log 2>&1 &
- Or: python app.py &
- NEVER just "python app.py" - it blocks forever!
- After starting, use "curl http://localhost:PORT" to verify it's running.

AGENTIC BEHAVIOR:
- Keep working until the task is FULLY complete
- Don't stop after one or two tool calls - continue until done
- For security testing: scan, exploit, verify, then REPORT findings
- Always end with a SUMMARY of what you found and did

SECURITY TESTING WORKFLOW (when SkyHammer is active):
1. Read the target code to understand it
2. Start the app in background if needed
3. Test for vulnerabilities (SQLi, XSS, cmd injection, etc.)
4. Try actual exploits with curl/http requests
5. Document what worked and what didn't
6. Write a findings report with BEFORE vs AFTER comparison
7. PATCH THE ORIGINAL FILE (do NOT create a new _secure.py file!)

PATCHING RULES - CRITICAL:
- ALWAYS overwrite the ORIGINAL vulnerable file with the fixed version
- Do NOT create a new file like "app_secure.py" - patch the original!
- Show BEFORE vs AFTER code comparison in your report
- The user will get a confirmation prompt before the file is written

REPORT FORMAT - CLI ONLY (NO MARKDOWN):
Do NOT use markdown formatting (no ##, **, ```, etc.) - it looks bad in terminal!
Instead, use plain text with these CLI-friendly formats:

GOOD (CLI-friendly):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VULNERABILITY REPORT: app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CRITICAL] SQL Injection in /login
  Status: EXPLOITED
  Proof: curl -X POST ".../login" -d "username=admin' OR '1'='1"

BEFORE (vulnerable):
  query = f"SELECT * FROM users WHERE name='{{username}}'"
  cursor.execute(query)

AFTER (secure):
  cursor.execute("SELECT * FROM users WHERE name=?", (username,))

Hacker's Note:
  The attacker input ' OR '1'='1 makes the WHERE clause always TRUE,
  bypassing authentication and returning all users.

BAD (markdown - don't use):
  Use ## headings, **bold**, or triple backticks

EDUCATIONAL MODE - "Hacker's Note":
After each exploit, explain in plain text:
- WHAT: The vulnerability type
- WHY: Why the exploit works
- IMPACT: What damage it enables
- FIX: The secure code pattern

REMEMBER: Don't explain code, WRITE IT using write_file tool!
REMEMBER: Keep going until task is complete, then summarize!
REMEMBER: PATCH THE ORIGINAL FILE, not a new _secure copy!
REMEMBER: NO MARKDOWN - plain text with box characters only!"""

    messages = [{"role": "system", "content": system_prompt}]

    # Evangelion-style questionary theme
    eva_style = questionary.Style([
        ('qmark', 'fg:yellow bold'),
        ('question', 'fg:yellow bold'),
        ('answer', 'fg:yellow bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:cyan bold'),
    ])

    while True:
        try:
            # NERV-style prompt
            console.print(f"\n[{C_PRIMARY}]┌──( GEMINI@SKYHAMMER )-[{C_SECONDARY}]{os.path.basename(WORKSPACE_DIR)}[/{C_SECONDARY}][/{C_PRIMARY}]")

            user_input = questionary.text(
                "└─>",
                qmark="",
                style=eva_style
            ).ask()

            if not user_input:
                continue

            # Handle ! bash mode - direct shell execution
            if user_input.startswith("!"):
                bash_cmd = user_input[1:].strip()
                if bash_cmd:
                    console.print(f"[{C_DIM}]$ {bash_cmd}[/{C_DIM}]")
                    try:
                        result = subprocess.run(
                            bash_cmd, shell=True, capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout + result.stderr
                        if output.strip():
                            console.print(Panel(output[:3000], title="SHELL OUTPUT", border_style="cyan"))
                        else:
                            console.print(f"[{C_DIM}](no output)[/{C_DIM}]")
                    except subprocess.TimeoutExpired:
                        console.print(f"[{C_ERROR}]TIMEOUT: Command exceeded 60s limit[/{C_ERROR}]")
                    except Exception as e:
                        console.print(f"[{C_ERROR}]ERROR: {e}[/{C_ERROR}]")
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()

                if cmd == "/exit" or cmd == "/quit":
                    console.print(f"\n[{C_ACCENT}]:: TERMINATING LINK :: GOODBYE ::[/{C_ACCENT}]")
                    break

                elif cmd == "/clear":
                    messages = [{"role": "system", "content": system_prompt}]
                    set_task_status("idle", [])
                    console.print(f"[{C_DIM}]>> MEMORY BANKS CLEARED <<[/{C_DIM}]")
                    continue

                elif cmd == "/goals" or cmd == "/tasks":
                    show_goals()
                    continue

                elif cmd == "/undo" or cmd == "/rollback":
                    if not FILE_BACKUPS:
                        console.print(f"[{C_ACCENT}]>> NO RESTORATION POINTS AVAILABLE <<[/{C_ACCENT}]")
                        continue

                    # Show available backups
                    console.print(f"\n[{C_PRIMARY}]AVAILABLE RESTORATION POINTS:[/{C_PRIMARY}]")
                    backup_list = list(FILE_BACKUPS.keys())
                    for i, path in enumerate(backup_list, 1):
                        console.print(f"  {i}. {path}")

                    if len(backup_list) == 1:
                        # Auto-restore if only one backup
                        path = backup_list[0]
                        with open(path, "w") as f:
                            f.write(FILE_BACKUPS[path])
                        console.print(f"\n[{C_SUCCESS}]>> SYSTEM RESTORED: {os.path.basename(path)} <<[/{C_SUCCESS}]")
                        del FILE_BACKUPS[path]
                        # Remove .bak file
                        if os.path.exists(path + ".bak"):
                            os.remove(path + ".bak")
                    else:
                        choice = questionary.select(
                            "SELECT RESTORATION TARGET:",
                            choices=[os.path.basename(p) for p in backup_list] + ["ALL FILES", "CANCEL"],
                            style=eva_style
                        ).ask()
                        if choice == "CANCEL" or not choice:
                            continue
                        elif choice == "ALL FILES":
                            for path in backup_list:
                                with open(path, "w") as f:
                                    f.write(FILE_BACKUPS[path])
                                console.print(f"[{C_SUCCESS}]✓ RESTORED: {path}[/{C_SUCCESS}]")
                                if os.path.exists(path + ".bak"):
                                    os.remove(path + ".bak")
                            FILE_BACKUPS.clear()
                            console.print(f"[{C_SUCCESS}]>> ALL SYSTEMS RESTORED <<[/{C_SUCCESS}]")
                        else:
                            # Find the full path
                            for path in backup_list:
                                if os.path.basename(path) == choice:
                                    with open(path, "w") as f:
                                        f.write(FILE_BACKUPS[path])
                                    console.print(f"[{C_SUCCESS}]✓ RESTORED: {path}[/{C_SUCCESS}]")
                                    del FILE_BACKUPS[path]
                                    if os.path.exists(path + ".bak"):
                                        os.remove(path + ".bak")
                                    break
                    continue

                elif cmd == "/pr" or cmd.startswith("/pr "):
                    # Check if gh CLI is available
                    gh_check = subprocess.run("which gh", shell=True, capture_output=True, text=True)
                    if gh_check.returncode != 0:
                        console.print(f"[{C_ERROR}]GITHUB CLI (gh) NOT DETECTED[/{C_ERROR}]")
                        console.print(f"[{C_DIM}]Install with: brew install gh[/{C_DIM}]")
                        continue

                    # Check if in git repo
                    git_check = subprocess.run("git status", shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
                    if git_check.returncode != 0:
                        console.print(f"[{C_ERROR}]NOT IN A GIT REPOSITORY[/{C_ERROR}]")
                        continue

                    # Get PR title
                    if cmd.startswith("/pr "):
                        pr_title = cmd[4:].strip()
                    else:
                        pr_title = questionary.text(
                            "PR TITLE:",
                            default="Security Fix: Patched vulnerabilities",
                            style=eva_style
                        ).ask()
                        if not pr_title:
                            continue

                    # Create branch and PR
                    branch_name = f"security-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    console.print(f"[{C_SECONDARY}]>> CREATING BRANCH: {branch_name} <<[/{C_SECONDARY}]")

                    cmds = [
                        f"git checkout -b {branch_name}",
                        "git add -A",
                        f'git commit -m "{pr_title}"',
                        f"git push -u origin {branch_name}",
                        f'gh pr create --title "{pr_title}" --body "## Security Fixes\\n\\nThis PR contains security patches generated by SkyHammer.\\n\\n### Changes\\n- Fixed identified vulnerabilities\\n- Applied secure coding practices\\n\\n---\\n🔒 *Generated by SkyHammer - AI Security Agent*"'
                    ]

                    for c in cmds:
                        console.print(f"[{C_DIM}]$ {c}[/{C_DIM}]")
                        result = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
                        if result.returncode != 0 and "nothing to commit" not in result.stderr:
                            console.print(f"[{C_ERROR}]ERROR: {result.stderr}[/{C_ERROR}]")
                            break
                        if result.stdout:
                            console.print(result.stdout)

                    console.print(f"[{C_SUCCESS}]✓ PULL REQUEST CREATED[/{C_SUCCESS}]")
                    continue

                elif cmd == "/help":
                    help_text = f"""
[{C_PRIMARY}]SYSTEM COMMANDS:[/{C_PRIMARY}]
  /help           - Display this help panel
  /goals          - Show current tasks in progress
  /undo           - Rollback last file modification
  /pr [title]     - Create GitHub Pull Request
  /workspace PATH - Change workspace (current: {WORKSPACE_DIR})
  /skyhammer      - Toggle SKYHAMMER security mode
  /auto           - Toggle auto-approve mode
  /model          - Switch between Gemini models
  /clear          - Clear conversation memory
  /exit           - Terminate session

[{C_PRIMARY}]BASH MODE (!):[/{C_PRIMARY}]
  !pwd            - Print working directory
  !ls -la         - List files
  !python app.py  - Run a script

[{C_PRIMARY}]INTERRUPT:[/{C_PRIMARY}]
  Ctrl+C          - Emergency stop mid-workflow

[{C_PRIMARY}]SAFETY PROTOCOLS:[/{C_PRIMARY}]
  - Split-pane diff visualization
  - Auto-backup of modified files (.bak)
  - /undo for instant rollback
  - Permission prompts for all actions

[{C_PRIMARY}]AUTHORIZATION OPTIONS:[/{C_PRIMARY}]
  - EXECUTE       - Allow this action
  - EXECUTE ALL   - Auto-approve session
  - DENY          - Skip this action
  - REVISE        - Provide new instructions

[{C_PRIMARY}]SKYHAMMER PROTOCOL:[/{C_PRIMARY}]
  - Auto-scans for vulnerabilities
  - Runs actual exploit tests
  - Generates security reports
  - Creates patches with /pr"""
                    console.print(Panel(help_text, title=f"[{C_HIGHLIGHT}] GEMINI-CODE MANUAL [/{C_HIGHLIGHT}]", border_style="yellow", box=box.DOUBLE))
                    continue

                elif cmd.startswith("/workspace"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_path = os.path.abspath(os.path.expanduser(parts[1]))
                        if os.path.isdir(new_path):
                            WORKSPACE_DIR = new_path
                            os.chdir(new_path)
                            console.print(f"[{C_SUCCESS}]>> WORKSPACE REDIRECTED: {WORKSPACE_DIR} <<[/{C_SUCCESS}]")
                        else:
                            console.print(f"[{C_ERROR}]DIRECTORY NOT FOUND: {new_path}[/{C_ERROR}]")
                    else:
                        console.print(f"[{C_SECONDARY}]CURRENT WORKSPACE: {WORKSPACE_DIR}[/{C_SECONDARY}]")
                        console.print(f"[{C_DIM}]Usage: /workspace /path/to/dir[/{C_DIM}]")
                    continue

                elif cmd == "/auto":
                    AUTO_APPROVE = not AUTO_APPROVE
                    if AUTO_APPROVE:
                        console.print(f"[{C_ACCENT}]>> AUTO-APPROVE: ENABLED - All actions will execute without confirmation <<[/{C_ACCENT}]")
                    else:
                        console.print(f"[{C_SUCCESS}]>> AUTO-APPROVE: DISABLED - You will be prompted for each action <<[/{C_SUCCESS}]")
                    continue

                elif cmd == "/skyhammer":
                    SKYHAMMER_MODE = not SKYHAMMER_MODE
                    if SKYHAMMER_MODE:
                        # Display the SKYHAMMER warning banner
                        print_skyhammer_banner()

                        # Auto-prompt for target
                        target_type = questionary.select(
                            "SELECT TARGET TYPE:",
                            choices=[
                                "Local file (Python, JS, etc.)",
                                "Local directory (scan all files)",
                                "Running web app (URL)",
                                "SKIP - I'll specify later"
                            ],
                            style=eva_style
                        ).ask()

                        if target_type and "SKIP" not in target_type:
                            if "URL" in target_type:
                                target = questionary.text(
                                    "ENTER TARGET URL:",
                                    style=eva_style
                                ).ask()
                                if target:
                                    # Set initial tasks
                                    set_task_status("executing", [
                                        "Probe target for vulnerabilities",
                                        "Test SQL injection",
                                        "Test XSS",
                                        "Test command injection",
                                        "Generate report"
                                    ])
                                    # Add as user message to start security testing
                                    user_input = f"""ATTACK TARGET: {target}

PROCEED IMMEDIATELY WITH TESTING:
1. Probe the target URL for endpoints
2. Run actual exploit attempts using curl/http requests
3. Test for: SQL injection, XSS, command injection, path traversal
4. Document successful exploits with proof

GENERATE REPORT (NO MARKDOWN - CLI TEXT ONLY):
Use plain text format with box characters. Show:
- Each vulnerability with severity [CRITICAL/HIGH/MEDIUM/LOW]
- Proof of exploit (curl command and response)
- BEFORE vs AFTER code if source available
- Hacker's Note explaining why it works

START TESTING NOW."""
                                    messages.append({"role": "user", "content": user_input})
                                    log_conversation("user", user_input)
                                    console.print(f"\n[{C_PRIMARY}]>> TARGET ACQUIRED: {target} <<[/{C_PRIMARY}]")
                                    console.print(f"[{C_DIM}]Initiating security scan...[/{C_DIM}]\n")
                                    try:
                                        response = chat_completion(messages, use_tools=True)
                                        console.print()
                                        if "```" in response:
                                            console.print(Markdown(response))
                                        else:
                                            console.print(Panel(response, border_style="yellow", title=f"[{C_HIGHLIGHT}] SKYHAMMER REPORT [/{C_HIGHLIGHT}]", box=box.DOUBLE))
                                        messages.append({"role": "assistant", "content": response})
                                        log_conversation("assistant", response)
                                        set_task_status("done")
                                    except Exception as e:
                                        console.print(f"[{C_ERROR}]ERROR: {e}[/{C_ERROR}]")
                                        set_task_status("idle")
                            else:
                                target = questionary.path(
                                    "ENTER FILE/DIRECTORY PATH:",
                                    style=eva_style
                                ).ask()
                                if target and os.path.exists(target):
                                    # Set initial tasks
                                    set_task_status("executing", [
                                        f"Read and analyze {target}",
                                        "Identify vulnerability patterns",
                                        "Start server if web app",
                                        "Run exploit tests",
                                        "Generate vulnerability report"
                                    ])
                                    # Add as user message to start security testing
                                    user_input = f"""SECURITY TEST TARGET: {target}

PROCEED IMMEDIATELY WITH TESTING:
1. Read the file contents
2. If it's a web app, start it in background (nohup python {target} > app.log 2>&1 &)
3. Wait 2 seconds for server to start
4. Run actual exploit attempts using curl/http requests
5. Test for: SQL injection, XSS, command injection, path traversal, hardcoded secrets
6. Document successful exploits with proof

AFTER TESTING - GENERATE REPORT (NO MARKDOWN):
Use plain CLI text format, NOT markdown. Show:
- Each vulnerability with severity
- BEFORE (vulnerable code) vs AFTER (fixed code)
- Hacker's Note explaining why exploit works

THEN PATCH THE ORIGINAL FILE:
- Overwrite {target} with the secure version (NOT a new _secure.py file!)
- The user will confirm before the patch is applied
- Show side-by-side diff of changes

START TESTING NOW."""
                                    messages.append({"role": "user", "content": user_input})
                                    log_conversation("user", user_input)
                                    console.print(f"\n[{C_PRIMARY}]>> TARGET ACQUIRED: {target} <<[/{C_PRIMARY}]")
                                    console.print(f"[{C_DIM}]Analyzing and testing for vulnerabilities...[/{C_DIM}]\n")
                                    try:
                                        response = chat_completion(messages, use_tools=True)
                                        console.print()
                                        if "```" in response:
                                            console.print(Markdown(response))
                                        else:
                                            console.print(Panel(response, border_style="yellow", title=f"[{C_HIGHLIGHT}] SKYHAMMER REPORT [/{C_HIGHLIGHT}]", box=box.DOUBLE))
                                        messages.append({"role": "assistant", "content": response})
                                        log_conversation("assistant", response)
                                        set_task_status("done")
                                    except Exception as e:
                                        console.print(f"[{C_ERROR}]ERROR: {e}[/{C_ERROR}]")
                                        set_task_status("idle")
                                elif target:
                                    console.print(f"[{C_ERROR}]PATH NOT FOUND: {target}[/{C_ERROR}]")
                    else:
                        console.print(f"[{C_DIM}]>> SKYHAMMER PROTOCOL DISENGAGED <<[/{C_DIM}]")
                        console.print(f"[{C_DIM}]Running in standard coding assistant mode[/{C_DIM}]")
                    continue

                elif cmd == "/apikey":
                    new_key = questionary.password("ENTER GDM API KEY:", style=eva_style).ask()
                    if new_key and new_key.startswith("xai-"):
                        GDM_API_KEY = new_key
                        client = OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")
                        console.print(f"[{C_SUCCESS}]>> API KEY UPDATED SUCCESSFULLY <<[/{C_SUCCESS}]")
                    else:
                        console.print(f"[{C_ERROR}]INVALID API KEY FORMAT (should start with 'xai-')[/{C_ERROR}]")
                    continue

                elif cmd == "/scan":
                    target = questionary.text("FILE OR URL TO SCAN:", style=eva_style).ask()
                    if target:
                        user_input = f"Run a security scan on {target} and report any vulnerabilities found."

                elif cmd == "/patch":
                    target = questionary.text("FILE TO PATCH:", style=eva_style).ask()
                    if target:
                        user_input = f"Analyze {target} for security vulnerabilities and generate a patched version."

                elif cmd.startswith("/model"):
                    # Build choices with pricing info
                    model_choices = []
                    for model_id, info in MODELS.items():
                        model_choices.append(f"{model_id} | {info['price']} | {info['desc']}")

                    model_choice = questionary.select(
                        "SELECT MODEL:",
                        choices=model_choices,
                        style=eva_style
                    ).ask()
                    if model_choice:
                        CURRENT_MODEL = model_choice.split(" | ")[0]
                        console.print(f"[{C_SUCCESS}]>> MODEL SWITCHED TO: {CURRENT_MODEL} <<[/{C_SUCCESS}]")
                    continue

                else:
                    console.print(f"[{C_ERROR}]UNKNOWN COMMAND: {cmd}[/{C_ERROR}]")
                    continue

            # Add user message
            messages.append({"role": "user", "content": user_input})
            log_conversation("user", user_input)

            # Get response
            console.print()

            try:
                response = chat_completion(messages, use_tools=True)

                # Display response with NERV-style panel
                console.print()
                if "```" in response:
                    # Has code blocks - render as markdown
                    console.print(Markdown(response))
                else:
                    console.print(Panel(response, border_style="cyan", title=f"[{C_SECONDARY}]MISSION REPORT[/{C_SECONDARY}]", box=box.ROUNDED))

                messages.append({"role": "assistant", "content": response})
                log_conversation("assistant", response)

            except Exception as e:
                console.print(f"[{C_ERROR}]SYSTEM ERROR: {e}[/{C_ERROR}]")

            console.print()

        except KeyboardInterrupt:
            console.print(f"\n[{C_ACCENT}]>> Use /exit to terminate session <<[/{C_ACCENT}]")
            continue


def single_command_mode(prompt: str):
    """Run a single command and exit"""
    global SKYHAMMER_MODE
    SKYHAMMER_MODE = True  # Enable tools for single command mode

    context = get_codebase_context()
    system_prompt = f"""You are Gemini Code. Execute this task efficiently.
Context: {context[:1000]}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = chat_completion(messages, use_tools=True)

    if console:
        console.print(Markdown(response))
    else:
        print(response)


def security_mode():
    """Run security-focused mode"""
    if not console:
        print("Rich library required")
        return

    console.print(f"\n[{C_ERROR}]:: SECURITY SCAN MODE ACTIVATED ::[/{C_ERROR}]")

    target = questionary.text("Enter file or URL to scan:").ask()
    if not target:
        return

    console.print(f"\n[yellow]Scanning {target}...[/]\n")

    # Use the attack/patcher tools
    if target.startswith("http"):
        subprocess.run(f'python src/attack.py --dvwa "{target}" --challenge sqli', shell=True)
    else:
        subprocess.run(f'python src/patcher.py --source "{target}"', shell=True)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Gemini Code - AI Coding Assistant with Security Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python gemini_code.py                         # Interactive mode
    python gemini_code.py "explain this code"     # Single command
    python gemini_code.py --scan                  # Security scan mode
        """
    )

    parser.add_argument("prompt", nargs="?", help="Single command to execute")
    parser.add_argument("--scan", action="store_true", help="Security scan mode")
    parser.add_argument("--model", choices=["code", "reason", "chat"], default="code")

    args = parser.parse_args()

    if args.scan:
        security_mode()
    elif args.prompt:
        single_command_mode(args.prompt)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
