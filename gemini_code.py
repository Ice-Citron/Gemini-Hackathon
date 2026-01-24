#!/usr/bin/env python3
"""
SkyHammer - Gemini-Powered Security & Coding CLI

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
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Install rich & questionary: pip install rich questionary")

# Load API Key
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

if not GDM_API_KEY:
    print("[!] Error: GDM_API_KEY not found")
    sys.exit(1)

console = Console() if HAS_RICH else None

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
        console.print(f"[dim]Logging to: {RUN_DIR}[/]")


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
    """Display current goals/tasks"""
    if not console:
        return

    if not CURRENT_TASKS and CURRENT_TASK_STATUS == "idle":
        console.print("[dim]No active tasks. Give Gemini something to do![/]")
        return

    status_colors = {
        "idle": "dim",
        "thinking": "yellow",
        "executing": "cyan",
        "done": "green"
    }
    color = status_colors.get(CURRENT_TASK_STATUS, "white")

    console.print(f"\n[bold]Current Status:[/] [{color}]{CURRENT_TASK_STATUS.upper()}[/]")

    if CURRENT_TASKS:
        console.print("[bold]Tasks:[/]")
        for i, task in enumerate(CURRENT_TASKS, 1):
            console.print(f"  {i}. {task}")
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
    """Show side-by-side diff with color highlighting"""
    from rich.table import Table
    from rich.syntax import Syntax

    if old_content == new_content:
        console.print("[dim]No changes[/]")
        return

    # Create side-by-side table
    table = Table(title=f"📝 Changes to {filename}", show_lines=True, expand=True)
    table.add_column("Before (Original)", style="red", width=50)
    table.add_column("After (Patched)", style="green", width=50)

    # Truncate for display
    old_display = old_content[:2000] if len(old_content) > 2000 else old_content
    new_display = new_content[:2000] if len(new_content) > 2000 else new_content

    # Add syntax highlighted code
    try:
        ext = filename.split('.')[-1] if '.' in filename else 'python'
        lang = {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'rb': 'ruby', 'php': 'php'}.get(ext, ext)
        old_syntax = Syntax(old_display, lang, line_numbers=True, word_wrap=True)
        new_syntax = Syntax(new_display, lang, line_numbers=True, word_wrap=True)
        table.add_row(old_syntax, new_syntax)
    except:
        table.add_row(old_display, new_display)

    console.print(table)

    # Also show unified diff below for detailed changes
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))

    if diff:
        console.print("\n[bold]Line-by-line changes:[/]")
        for line in diff[2:30]:  # Skip headers, limit output
            if line.startswith('+'):
                console.print(f"[bright_green]{line}[/]")
            elif line.startswith('-'):
                console.print(f"[bright_red]{line}[/]")
            elif line.startswith('@@'):
                console.print(f"[yellow]{line}[/]")
        if len(diff) > 32:
            console.print(f"[dim]  ... +{len(diff) - 32} more lines[/]")
    console.print()


def show_new_file_preview(content: str, filename: str):
    """Show new file with green highlighting"""
    console.print(f"\n[bold]New file: {filename}[/]")
    lines = content.splitlines()
    for i, line in enumerate(lines[:25], 1):
        console.print(f"[bright_green]+{i:3}| {line}[/]")
    if len(lines) > 25:
        console.print(f"[dim]  ... +{len(lines) - 25} more lines[/]")
    console.print()


def ask_permission(tool_name: str, args: Dict[str, Any], preview_content: str = "") -> tuple:
    """Ask user permission. Returns (action, feedback) where action is 'yes', 'no', 'yes_all', or 'feedback'"""
    global AUTO_APPROVE

    if AUTO_APPROVE:
        return ("yes", None)

    # Pause ESC listener during prompt
    pause_listener()

    try:
        console.print(f"\n[bold yellow]⚡ {tool_name}[/]")

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
            console.print(f"[cyan]$ {args.get('command', '')}[/]")

        else:
            console.print(f"[dim]{json.dumps(args)[:200]}[/]")

        # Ask
        choice = questionary.select(
            "Allow?",
            choices=["Yes", "Yes to all (session)", "No (skip)", "Tell Gemini to do otherwise"],
            style=questionary.Style([('selected', 'fg:cyan bold')])
        ).ask()

        if not choice:
            return ("no", None)
        if "Yes to all" in choice:
            AUTO_APPROVE = True
            return ("yes", None)
        if "Yes" == choice:
            return ("yes", None)
        if "otherwise" in choice.lower():
            feedback = questionary.text(
                "What should Gemini do differently?",
                style=questionary.Style([('answer', 'fg:yellow')])
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
                    console.print(f"[green]✓ Read {len(content)} bytes from {path}[/]")
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
                    console.print(f"[dim]💾 Backup saved: {path}.bak[/]")

            # Create parent directories if needed
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            if console:
                console.print(f"[green]✓ Wrote {len(content)} bytes to {args.get('path', path)}[/]")
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
                console.print(f"[yellow]$ {command}[/]")
            proc_result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=120, cwd=WORKSPACE_DIR
            )
            output = proc_result.stdout + proc_result.stderr
            if console:
                console.print(f"[green]✓ Exit code: {proc_result.returncode}[/]")
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
    """Run agentic chat completion - loops until Gemini is done or max iterations"""
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
        console.print("[dim](Press Ctrl+C to interrupt, /goals to see tasks)[/]")

    set_task_status("thinking")

    try:
        while iteration < max_iterations:
            iteration += 1

            # Check for interrupt flag
            if INTERRUPT_REQUESTED:
                set_task_status("idle")
                return "(Interrupted by user)"

            set_task_status("thinking")
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
                console.print(f"[dim]Turn {iteration}: {len(msg.tool_calls)} tool calls[/]")

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
                    console.print(f"[cyan]→ {tool_name}[/]")

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
                console.print(f"[dim italic]{msg.content[:200]}...[/]" if len(msg.content or "") > 200 else f"[dim italic]{msg.content}[/]")

        return "(Max iterations reached - use /clear to reset)"

    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Interrupted by Ctrl+C[/]")
        return "(Interrupted by user)"


def interactive_mode():
    """Run interactive chat mode"""
    global CURRENT_MODEL, SKYHAMMER_MODE, WORKSPACE_DIR, AUTO_APPROVE, client, GDM_API_KEY

    if not console:
        print("Rich library required for interactive mode")
        return

    # Initialize logging
    init_logging()

    console.print("\n[bold cyan]╔══════════════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║             Gemini Code - Powered by GDM                    ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════╝[/]")
    console.print()

    skyhammer_status = "[bold red]OFF[/]" if not SKYHAMMER_MODE else "[bold green]ON[/]"
    auto_status = "[yellow]ON[/]" if AUTO_APPROVE else "[dim]OFF[/]"
    console.print(f"[dim]Commands: /help, /auto, /workspace, /skyhammer, /model, /clear, /exit[/]")
    console.print(f"[dim]Bash mode: !command (e.g. !ls, !pwd, !python app.py &)[/]")
    console.print(f"[dim]Model: {CURRENT_MODEL} | Auto-approve: {auto_status}[/]")
    console.print(f"[dim]Workspace: {WORKSPACE_DIR}[/]")
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
  query = f"SELECT * FROM users WHERE name='{username}'"
  cursor.execute(query)

AFTER (secure):
  cursor.execute("SELECT * FROM users WHERE name=?", (username,))

Hacker's Note:
  The attacker input ' OR '1'='1 makes the WHERE clause always TRUE,
  bypassing authentication and returning all users.

BAD (markdown - don't use):
  ## Vulnerabilities
  **SQL Injection** in `/login`
  ```python
  query = f"SELECT..."
  ```

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

    while True:
        try:
            user_input = questionary.text(
                "You >",
                style=questionary.Style([('answer', 'fg:cyan')])
            ).ask()

            if not user_input:
                continue

            # Handle ! bash mode - direct shell execution
            if user_input.startswith("!"):
                bash_cmd = user_input[1:].strip()
                if bash_cmd:
                    console.print(f"[dim]$ {bash_cmd}[/]")
                    try:
                        result = subprocess.run(
                            bash_cmd, shell=True, capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout + result.stderr
                        if output.strip():
                            console.print(Panel(output[:3000], title="Output", border_style="blue"))
                        else:
                            console.print("[dim](no output)[/]")
                    except subprocess.TimeoutExpired:
                        console.print("[red]Command timed out (60s)[/]")
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/]")
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()

                if cmd == "/exit" or cmd == "/quit":
                    console.print("[yellow]Goodbye![/]")
                    break

                elif cmd == "/clear":
                    messages = [{"role": "system", "content": system_prompt}]
                    set_task_status("idle", [])
                    console.print("[dim]Conversation cleared.[/]")
                    continue

                elif cmd == "/goals" or cmd == "/tasks":
                    show_goals()
                    continue

                elif cmd == "/undo" or cmd == "/rollback":
                    if not FILE_BACKUPS:
                        console.print("[yellow]No backups available to restore.[/]")
                        continue

                    # Show available backups
                    console.print("\n[bold]Available backups:[/]")
                    backup_list = list(FILE_BACKUPS.keys())
                    for i, path in enumerate(backup_list, 1):
                        console.print(f"  {i}. {path}")

                    if len(backup_list) == 1:
                        # Auto-restore if only one backup
                        path = backup_list[0]
                        with open(path, "w") as f:
                            f.write(FILE_BACKUPS[path])
                        console.print(f"\n[green]✓ Restored {path} to original state[/]")
                        del FILE_BACKUPS[path]
                        # Remove .bak file
                        if os.path.exists(path + ".bak"):
                            os.remove(path + ".bak")
                    else:
                        choice = questionary.select(
                            "Which file to restore?",
                            choices=[os.path.basename(p) for p in backup_list] + ["All files", "Cancel"]
                        ).ask()
                        if choice == "Cancel" or not choice:
                            continue
                        elif choice == "All files":
                            for path in backup_list:
                                with open(path, "w") as f:
                                    f.write(FILE_BACKUPS[path])
                                console.print(f"[green]✓ Restored {path}[/]")
                                if os.path.exists(path + ".bak"):
                                    os.remove(path + ".bak")
                            FILE_BACKUPS.clear()
                            console.print("[green]All files restored![/]")
                        else:
                            # Find the full path
                            for path in backup_list:
                                if os.path.basename(path) == choice:
                                    with open(path, "w") as f:
                                        f.write(FILE_BACKUPS[path])
                                    console.print(f"[green]✓ Restored {path}[/]")
                                    del FILE_BACKUPS[path]
                                    if os.path.exists(path + ".bak"):
                                        os.remove(path + ".bak")
                                    break
                    continue

                elif cmd == "/pr" or cmd.startswith("/pr "):
                    # Check if gh CLI is available
                    gh_check = subprocess.run("which gh", shell=True, capture_output=True, text=True)
                    if gh_check.returncode != 0:
                        console.print("[red]GitHub CLI (gh) not installed.[/]")
                        console.print("[dim]Install with: brew install gh[/]")
                        continue

                    # Check if in git repo
                    git_check = subprocess.run("git status", shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
                    if git_check.returncode != 0:
                        console.print("[red]Not in a git repository.[/]")
                        continue

                    # Get PR title
                    if cmd.startswith("/pr "):
                        pr_title = cmd[4:].strip()
                    else:
                        pr_title = questionary.text(
                            "PR Title:",
                            default="Security Fix: Patched vulnerabilities"
                        ).ask()
                        if not pr_title:
                            continue

                    # Create branch and PR
                    branch_name = f"security-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    console.print(f"[cyan]Creating branch: {branch_name}[/]")

                    cmds = [
                        f"git checkout -b {branch_name}",
                        "git add -A",
                        f'git commit -m "{pr_title}"',
                        f"git push -u origin {branch_name}",
                        f'gh pr create --title "{pr_title}" --body "## Security Fixes\\n\\nThis PR contains security patches generated by SkyHammer.\\n\\n### Changes\\n- Fixed identified vulnerabilities\\n- Applied secure coding practices\\n\\n---\\n🔒 *Generated by SkyHammer - AI Security Agent*"'
                    ]

                    for c in cmds:
                        console.print(f"[dim]$ {c}[/]")
                        result = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
                        if result.returncode != 0 and "nothing to commit" not in result.stderr:
                            console.print(f"[red]Error: {result.stderr}[/]")
                            break
                        if result.stdout:
                            console.print(result.stdout)

                    console.print("[green]✓ Pull request created![/]")
                    continue

                elif cmd == "/help":
                    console.print(Panel(f"""
[bold]Commands:[/]
  /help           - Show this help
  /goals          - Show current tasks Gemini is working on
  /undo           - Rollback last file change (safety feature)
  /pr [title]     - Create GitHub PR with security fixes
  /workspace PATH - Change sandbox directory (current: {WORKSPACE_DIR})
  /skyhammer      - Toggle SkyHammer security mode
  /auto           - Toggle auto-approve (skip permission prompts)
  /model          - Switch between Gemini models
  /clear          - Clear conversation
  /exit           - Exit

[bold]Bash Mode (!):[/]
  !pwd            - Print working directory
  !ls -la         - List files
  !python app.py  - Run a script

[bold]Interrupt:[/]
  Ctrl+C          - Stop Gemini mid-workflow

[bold]Safety Features:[/]
  - Side-by-side diff before changes
  - Auto-backup of modified files (.bak)
  - /undo to instantly rollback
  - Permission prompts for all actions

[bold]Permission Prompts:[/]
  - Yes           - Allow this action
  - Yes to all    - Auto-approve session
  - No            - Skip this action
  - Tell Gemini...  - Give different instructions

[bold]Security Mode (/skyhammer):[/]
  - Auto-scans for vulnerabilities
  - Runs actual exploit tests
  - Generates security reports
  - Creates patches with /pr
                    """, title="Gemini Code Help", border_style="cyan"))
                    continue

                elif cmd.startswith("/workspace"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_path = os.path.abspath(os.path.expanduser(parts[1]))
                        if os.path.isdir(new_path):
                            WORKSPACE_DIR = new_path
                            os.chdir(new_path)
                            console.print(f"[green]Workspace changed to: {WORKSPACE_DIR}[/]")
                        else:
                            console.print(f"[red]Directory not found: {new_path}[/]")
                    else:
                        console.print(f"[cyan]Current workspace: {WORKSPACE_DIR}[/]")
                        console.print("[dim]Usage: /workspace /path/to/dir[/]")
                    continue

                elif cmd == "/auto":
                    AUTO_APPROVE = not AUTO_APPROVE
                    if AUTO_APPROVE:
                        console.print("[yellow]Auto-approve ON - tools will execute without prompts[/]")
                    else:
                        console.print("[green]Auto-approve OFF - you'll be asked before each action[/]")
                    continue

                elif cmd == "/skyhammer":
                    SKYHAMMER_MODE = not SKYHAMMER_MODE
                    if SKYHAMMER_MODE:
                        console.print("\n[bold green]╔═══════════════════════════════════════╗[/]")
                        console.print("[bold green]║      SKYHAMMER MODE ACTIVATED         ║[/]")
                        console.print("[bold green]╚═══════════════════════════════════════╝[/]\n")
                        console.print("[dim]Security tools: scan, exploit, patch, shell[/]\n")

                        # Auto-prompt for target
                        target_type = questionary.select(
                            "What do you want to test?",
                            choices=[
                                "Local file (Python, JS, etc.)",
                                "Local directory (scan all files)",
                                "Running web app (URL)",
                                "Skip - I'll specify later"
                            ],
                            style=questionary.Style([('selected', 'fg:green bold')])
                        ).ask()

                        if target_type and "Skip" not in target_type:
                            if "URL" in target_type:
                                target = questionary.text(
                                    "Enter URL (e.g., http://localhost:5000):",
                                    style=questionary.Style([('answer', 'fg:cyan')])
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
                                    console.print(f"\n[yellow]Target: {target}[/]")
                                    console.print("[dim]Starting security scan...[/]\n")
                                    try:
                                        response = chat_completion(messages, use_tools=True)
                                        console.print()
                                        if "```" in response:
                                            console.print(Markdown(response))
                                        else:
                                            console.print(Panel(response, border_style="green", title="SkyHammer Report"))
                                        messages.append({"role": "assistant", "content": response})
                                        log_conversation("assistant", response)
                                        set_task_status("done")
                                    except Exception as e:
                                        console.print(f"[red]Error: {e}[/]")
                                        set_task_status("idle")
                            else:
                                target = questionary.path(
                                    "Enter file/directory path:",
                                    style=questionary.Style([('answer', 'fg:cyan')])
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
                                    console.print(f"\n[yellow]Target: {target}[/]")
                                    console.print("[dim]Analyzing and testing for vulnerabilities...[/]\n")
                                    try:
                                        response = chat_completion(messages, use_tools=True)
                                        console.print()
                                        if "```" in response:
                                            console.print(Markdown(response))
                                        else:
                                            console.print(Panel(response, border_style="green", title="SkyHammer Report"))
                                        messages.append({"role": "assistant", "content": response})
                                        log_conversation("assistant", response)
                                        set_task_status("done")
                                    except Exception as e:
                                        console.print(f"[red]Error: {e}[/]")
                                        set_task_status("idle")
                                elif target:
                                    console.print(f"[red]Path not found: {target}[/]")
                    else:
                        console.print("[bold red]SkyHammer Mode DEACTIVATED[/]")
                        console.print("[dim]Running in standard coding assistant mode[/]")
                    continue

                elif cmd == "/apikey":
                    new_key = questionary.password("Enter your GDM API key:").ask()
                    if new_key and new_key.startswith("xai-"):
                        GDM_API_KEY = new_key
                        client = OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")
                        console.print("[green]API key updated successfully![/]")
                    else:
                        console.print("[red]Invalid API key format (should start with 'xai-')[/]")
                    continue

                elif cmd == "/scan":
                    target = questionary.text("File or URL to scan:").ask()
                    if target:
                        user_input = f"Run a security scan on {target} and report any vulnerabilities found."

                elif cmd == "/patch":
                    target = questionary.text("File to patch:").ask()
                    if target:
                        user_input = f"Analyze {target} for security vulnerabilities and generate a patched version."

                elif cmd.startswith("/model"):
                    # Build choices with pricing info
                    model_choices = []
                    for model_id, info in MODELS.items():
                        model_choices.append(f"{model_id} | {info['price']} | {info['desc']}")

                    model_choice = questionary.select(
                        "Select model:",
                        choices=model_choices
                    ).ask()
                    if model_choice:
                        CURRENT_MODEL = model_choice.split(" | ")[0]
                        console.print(f"[green]Switched to {CURRENT_MODEL}[/]")
                    continue

                else:
                    console.print(f"[red]Unknown command: {cmd}[/]")
                    continue

            # Add user message
            messages.append({"role": "user", "content": user_input})
            log_conversation("user", user_input)

            # Get response
            console.print()
            console.print("[dim]Gemini is thinking...[/]")

            try:
                response = chat_completion(messages, use_tools=True)

                # Display response
                console.print()
                if "```" in response:
                    # Has code blocks - render as markdown
                    console.print(Markdown(response))
                else:
                    console.print(Panel(response, border_style="green", title="Gemini"))

                messages.append({"role": "assistant", "content": response})
                log_conversation("assistant", response)

            except Exception as e:
                console.print(f"[red]Error: {e}[/]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit[/]")
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

    console.print("\n[bold red]Security Scan Mode[/]")

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
