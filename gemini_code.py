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
import difflib  # For diff highlighting

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


def show_diff(old_content: str, new_content: str, filename: str):
    """Show diff with green/red highlighting"""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}"))

    if not diff:
        console.print("[dim]No changes[/]")
        return

    console.print(f"\n[bold]Diff for {filename}:[/]")
    for line in diff[:50]:  # Limit output
        line = line.rstrip()
        if line.startswith('+') and not line.startswith('+++'):
            console.print(f"[green]{line}[/]")
        elif line.startswith('-') and not line.startswith('---'):
            console.print(f"[red]{line}[/]")
        elif line.startswith('@@'):
            console.print(f"[cyan]{line}[/]")
        else:
            console.print(f"[dim]{line}[/]")
    console.print()


def show_new_file_preview(content: str, filename: str):
    """Show new file with green highlighting"""
    console.print(f"\n[bold]New file: {filename}[/]")
    lines = content.splitlines()
    for i, line in enumerate(lines[:25], 1):
        console.print(f"[green]+{i:3}| {line}[/]")
    if len(lines) > 25:
        console.print(f"[dim]  ... +{len(lines) - 25} more lines[/]")
    console.print()


def ask_permission(tool_name: str, args: Dict[str, Any], preview_content: str = "") -> str:
    """Ask user permission. Returns 'yes', 'no', 'yes_all', or 'edit'"""
    global AUTO_APPROVE

    if AUTO_APPROVE:
        return "yes"

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
        choices=["Yes", "Yes to all (session)", "No (skip)", "Edit args"],
        style=questionary.Style([('selected', 'fg:cyan bold')])
    ).ask()

    if not choice:
        return "no"
    if "Yes to all" in choice:
        AUTO_APPROVE = True
        return "yes"
    if "Yes" in choice:
        return "yes"
    if "Edit" in choice:
        return "edit"
    return "no"


def is_safe_path(path: str) -> bool:
    """Check if path is within the workspace (sandbox)"""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(WORKSPACE_DIR)


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool and return the result"""
    global WORKSPACE_DIR

    try:
        if name == "read_file":
            path = args.get("path", "")
            # Make relative paths absolute within workspace
            if not os.path.isabs(path):
                path = os.path.join(WORKSPACE_DIR, path)
            if not is_safe_path(path):
                return f"Error: Access denied - path outside workspace: {path}"
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                if console:
                    console.print(f"[green]✓ Read {len(content)} bytes from {path}[/]")
                return f"Contents of {path}:\n```\n{content[:3000]}\n```"
            return f"Error: File not found: {path}"

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            # Ask permission with diff preview
            permission = ask_permission("write_file", args)
            if permission == "no":
                return "Action skipped by user"
            elif permission == "edit":
                edited = questionary.text("Edit content:", default=content[:500]).ask()
                if edited:
                    content = edited
            # Make relative paths absolute within workspace
            if not os.path.isabs(path):
                path = os.path.join(WORKSPACE_DIR, path)
            if not is_safe_path(path):
                return f"Error: Access denied - path outside workspace: {path}"
            # Create parent directories if needed
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            if console:
                console.print(f"[green]✓ Wrote {len(content)} bytes to {args.get('path', path)}[/]")
            return f"SUCCESS: Wrote {len(content)} bytes to {args.get('path', path)}"

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
                return "Error: Command blocked for safety"
            # Ask permission
            permission = ask_permission("run_command", args)
            if permission == "no":
                return "Action skipped by user"
            elif permission == "edit":
                edited = questionary.text("Edit command:", default=command).ask()
                if edited:
                    command = edited
            if console:
                console.print(f"[yellow]$ {command}[/]")
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=120, cwd=WORKSPACE_DIR
            )
            output = result.stdout + result.stderr
            if console:
                console.print(f"[green]✓ Exit code: {result.returncode}[/]")
            return f"Command output (exit {result.returncode}):\n```\n{output[:2000]}\n```"

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
                    f'python attack.py --dvwa "{target}" --challenge sqli --max-turns 5 --quiet',
                    shell=True, capture_output=True, text=True, timeout=60
                )
            else:
                result = subprocess.run(
                    f'python patcher.py --source "{target}" --vuln "Security Audit"',
                    shell=True, capture_output=True, text=True, timeout=60
                )
            return f"Security scan results:\n{result.stdout[:2000]}"

        elif name == "patch_vulnerability":
            file_path = args.get("file_path", "")
            vuln = args.get("vulnerability", "Unknown")
            result = subprocess.run(
                f'python patcher.py --source "{file_path}" --vuln "{vuln}"',
                shell=True, capture_output=True, text=True, timeout=120
            )
            return f"Patch result:\n{result.stdout[:2000]}"

        return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error: {str(e)}"


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
    global CURRENT_MODEL, SKYHAMMER_MODE
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

    while iteration < max_iterations:
        iteration += 1

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

        # If no tool calls, we're done
        if not msg.tool_calls:
            return msg.content or "(No response)"

        # Process tool calls
        if console:
            console.print(f"[dim]Turn {iteration}: {len(msg.tool_calls)} tool calls[/]")

        tool_results = []
        for tc in msg.tool_calls:
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


def interactive_mode():
    """Run interactive chat mode"""
    global CURRENT_MODEL, SKYHAMMER_MODE, WORKSPACE_DIR, AUTO_APPROVE, client, GDM_API_KEY

    if not console:
        print("Rich library required for interactive mode")
        return

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
6. Write a findings report

REMEMBER: Don't explain code, WRITE IT using write_file tool!
REMEMBER: Keep going until task is complete, then summarize!"""

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
                    console.print("[dim]Conversation cleared.[/]")
                    continue

                elif cmd == "/help":
                    console.print(Panel(f"""
[bold]Commands:[/]
  /help           - Show this help
  /workspace PATH - Change sandbox directory (current: {WORKSPACE_DIR})
  /skyhammer      - Toggle SkyHammer security mode
  /model          - Switch between Gemini models
  /clear          - Clear conversation
  /exit           - Exit

[bold]Bash Mode (!):[/]
  !pwd            - Print working directory
  !ls -la         - List files
  !python app.py  - Run a script
  !uvicorn app:app --port 8000

[bold]Tool Calling:[/]
  Gemini will ACTUALLY execute these (not just show code):
  - write_file: Creates real files on disk
  - read_file: Reads file contents
  - run_command: Runs shell commands
  - list_files: Lists directory
  - search_code: Searches code

[bold]Examples:[/]
  "create a vulnerable flask app called vuln.py"
  "run python vuln.py in background"
  "read the mock_dvwa.py file and explain it"
  "find all SQL queries in *.py files"

[bold]Security Mode (/skyhammer):[/]
  Adds: security_scan, patch_vulnerability tools
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
                        console.print("[bold green]SkyHammer Mode ACTIVATED[/]")
                        console.print("[dim]Security tools now available: scan, patch, shell, file ops[/]")
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
        subprocess.run(f'python attack.py --dvwa "{target}" --challenge sqli', shell=True)
    else:
        subprocess.run(f'python patcher.py --source "{target}"', shell=True)


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
