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


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool and return the result"""
    try:
        if name == "read_file":
            path = args.get("path", "")
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                return f"Contents of {path}:\n```\n{content[:3000]}\n```"
            return f"Error: File not found: {path}"

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {path}"

        elif name == "list_files":
            path = args.get("path", ".")
            pattern = args.get("pattern", "*")
            files = glob.glob(os.path.join(path, pattern))
            return f"Files matching '{pattern}' in {path}:\n" + "\n".join(files[:50])

        elif name == "run_command":
            command = args.get("command", "")
            # Safety check
            dangerous = ["rm -rf", "sudo", "mkfs", "> /dev"]
            if any(d in command for d in dangerous):
                return "Error: Command blocked for safety"
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            return f"Command output:\n```\n{output[:2000]}\n```"

        elif name == "search_code":
            pattern = args.get("pattern", "")
            file_pattern = args.get("file_pattern", "*.py")
            result = subprocess.run(
                f'grep -rn "{pattern}" --include="{file_pattern}" .',
                shell=True, capture_output=True, text=True, timeout=10
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
    """Run a chat completion with optional tool use"""
    global CURRENT_MODEL, SKYHAMMER_MODE
    model_id = CURRENT_MODEL

    kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    # Only include security tools if SkyHammer mode is active
    if use_tools and SKYHAMMER_MODE:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    # Handle tool calls
    if msg.tool_calls:
        tool_results = []
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except:
                tool_args = {}

            if console:
                console.print(f"[dim]Calling tool: {tool_name}({json.dumps(tool_args)[:50]}...)[/]")

            result = execute_tool(tool_name, tool_args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        # Add tool results and get final response
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})
        messages.extend(tool_results)

        # Get follow-up response
        follow_up = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        return follow_up.choices[0].message.content or ""

    return msg.content or ""


def interactive_mode():
    """Run interactive chat mode"""
    global CURRENT_MODEL, SKYHAMMER_MODE, client, GDM_API_KEY

    if not console:
        print("Rich library required for interactive mode")
        return

    console.print("\n[bold cyan]╔══════════════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║             Gemini Code - Powered by GDM                    ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════╝[/]")
    console.print()

    skyhammer_status = "[bold red]OFF[/]" if not SKYHAMMER_MODE else "[bold green]ON[/]"
    console.print(f"[dim]Commands: /help, /skyhammer, /model, /apikey, /clear, /exit[/]")
    console.print(f"[dim]Model: {CURRENT_MODEL} | SkyHammer: {skyhammer_status}[/]")
    console.print()

    # Build system prompt with context
    context = get_codebase_context()
    system_prompt = f"""You are Gemini Code, an AI coding assistant with security expertise.

You have access to the following tools:
- read_file: Read file contents
- write_file: Write/create files
- list_files: List directory contents
- run_command: Execute shell commands (safely)
- search_code: Search for patterns in code
- security_scan: Run security vulnerability scan
- patch_vulnerability: Generate security patches

Current codebase context:
{context}

Guidelines:
1. Use tools when needed to accomplish tasks
2. Be concise but thorough
3. For security issues, always recommend fixes
4. When writing code, follow best practices
5. Ask clarifying questions if needed"""

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = questionary.text(
                "You >",
                style=questionary.Style([('answer', 'fg:cyan')])
            ).ask()

            if not user_input:
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
                    console.print(Panel("""
[bold]Commands:[/]
  /help       - Show this help
  /skyhammer  - Toggle SkyHammer security mode (attack/defense tools)
  /model      - Switch between Gemini models
  /apikey     - Set your own GDM API key
  /scan       - Run security scan (requires SkyHammer mode)
  /patch      - Generate security patch (requires SkyHammer mode)
  /clear      - Clear conversation
  /exit       - Exit

[bold]SkyHammer Mode:[/]
  When activated, Gemini Code gains access to security tools:
  - File read/write, shell commands, code search
  - Security scanning and vulnerability detection
  - Auto-patching and remediation

[bold]Examples:[/]
  "read the mock_dvwa.py file"
  "find all SQL queries in the code"
  "create a new Flask app with user auth"
  "scan this file for vulnerabilities"
  "fix the SQL injection in mock_dvwa.py"
                    """, title="Gemini Code Help", border_style="cyan"))
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
