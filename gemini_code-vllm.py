#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SKYHAMMER // GEMINI-CODE (BASE)                                                ║
║  Evangelion/Cyberpunk Aesthetics for Autonomous Security                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python gemini_code-base.py
"""

import os
import sys
import json
import subprocess
import glob
from typing import List, Dict, Any
from datetime import datetime

# =============================================================================
# DEPENDENCIES & THEME
# =============================================================================
try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.table import Table
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.style import Style
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("CRITICAL ERROR: SYSTEM DEPENDENCIES MISSING.")
    print("RUN: pip install rich questionary google-generativeai")
    sys.exit(1)

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("CRITICAL ERROR: Google Generative AI SDK missing.")
    print("RUN: pip install google-generativeai")
    sys.exit(1)

# NERV / CYBERPUNK THEME
C_PRIMARY = "bold yellow"       
C_SECONDARY = "cyan"            
C_ACCENT = "bold orange1"       
C_SUCCESS = "bold green"        
C_ERROR = "bold red"            
C_DIM = "dim white"             
C_HIGHLIGHT = "black on yellow" 

# =============================================================================
# CONFIGURATION
# =============================================================================
try:
    from secretsConfig import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

if not GEMINI_API_KEY:
    print("[!] FATAL: NEURAL LINK SEVERED (GEMINI_API_KEY missing)")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
console = Console()

# Available Gemini models with pricing (per million tokens: input/output)
MODELS = {
    "gemini-3-flash-preview": {"price": "$0.50/$3.00", "desc": "Gemini 3 Flash - Latest, 3x faster than 2.5 Pro"},
    "gemini-3-pro-preview": {"price": "$2.00/$12.00", "desc": "Gemini 3 Pro - Most capable"},
    "gemini-2.5-pro": {"price": "$1.25/$10.00", "desc": "Strong reasoning, long context"},
    "gemini-2.5-flash": {"price": "$0.15/$0.60", "desc": "Fast with thinking mode"},
    "gemini-2.5-flash-lite": {"price": "$0.10/$0.40", "desc": "Most economical"},
    "gemini-2.0-flash": {"price": "$0.10/$0.40", "desc": "Fast, multimodal"},
}

CURRENT_MODEL = "gemini-3-flash-preview"
WORKSPACE_DIR = os.getcwd()
AUTO_APPROVE = False
INTERRUPT_REQUESTED = False
LAST_MODIFIED_FILE = None # Tracks file for recommendation

# =============================================================================
# UI COMPONENTS
# =============================================================================

def print_banner():
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" SYSTEM ONLINE // GEMINI-CODE BASE ", justify="center", style="bold black on yellow"),
            style="bold yellow",
            border_style="yellow",
            box=box.HEAVY,
        )
    )
    console.print(grid)
    console.print(f"[{C_DIM}]:: NEURAL LINK ESTABLISHED :: MODEL: {CURRENT_MODEL} ::[/{C_DIM}]", justify="center")
    console.print()

def print_skyhammer_banner():
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
    console.print()

def print_recommendation(filename):
    """Suggest SkyHammer after coding"""
    console.print()
    panel = Panel(
        Text(f"Target Modified: {filename}\nRecommend engaging SKYHAMMER protocol to audit for vulnerabilities.", justify="center"),
        title=f"[{C_ACCENT}] MISSION COMPLETE [{C_ACCENT}]",
        border_style="orange1",
        box=box.ROUNDED
    )
    console.print(panel)
    console.print(f"[{C_DIM}]Type '/skyhammer' to scan this target.[/{C_DIM}]")

# =============================================================================
# TOOLS
# =============================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write/Create file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": []
            }
        }
    }
]

def execute_tool(name, args):
    try:
        if name == "read_file":
            path = args.get("path")
            if os.path.exists(path):
                with open(path, "r") as f: return f"File Content:\n{f.read()}"
            return "Error: File not found."
        
        elif name == "write_file":
            path = args.get("path")
            content = args.get("content")
            
            # Interactive Approval
            if not AUTO_APPROVE:
                console.print(f"\n[{C_ACCENT}]>> PROPOSED WRITE: {path} <<[/{C_ACCENT}]")
                if not questionary.confirm("Authorize write operation?").ask():
                    return "Action denied by user."
            
            with open(path, "w") as f: f.write(content)
            global LAST_MODIFIED_FILE
            LAST_MODIFIED_FILE = path
            return f"Success: Wrote to {path}"
        
        elif name == "run_command":
            cmd = args.get("command")
            
            if not AUTO_APPROVE:
                console.print(Panel(f"$ {cmd}", title="EXECUTE SHELL", border_style="yellow"))
                if not questionary.confirm("Authorize execution?").ask():
                    return "Action denied by user."

            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return f"Exit: {res.returncode}\nOut: {res.stdout}\nErr: {res.stderr}"
            
        elif name == "list_files":
            return str(os.listdir(args.get("path", ".")))
            
    except Exception as e:
        return f"Error: {e}"
    return "Unknown Tool"

def get_context():
    files = glob.glob("*.py") + ["requirements.txt", "README.md"]
    ctx = []
    for f in files:
        if os.path.exists(f):
            with open(f) as file: ctx.append(f"--- {f} ---\n{file.read()[:800]}")
    return "\n".join(ctx)

# =============================================================================
# CORE LOOP
# =============================================================================
def chat_completion(messages):
    global LAST_MODIFIED_FILE
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style="bold yellow"),
        TextColumn(f"[{C_PRIMARY}]NEURAL SYNC...[/{C_PRIMARY}]"),
        transient=True
    ) as progress:
        progress.add_task("waiting", total=None)
        
        response = client.chat.completions.create(
            model=CURRENT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3
        )
        msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_name = tc.function.name
            
            console.print(f"[{C_SECONDARY}]-> TOOL: {tool_name}[/{C_SECONDARY}]")
            result = execute_tool(tool_name, args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
            
            # Print Tool Output Preview
            console.print(f"[{C_DIM}]< {result[:100].replace(chr(10), ' ')}... >[/{C_DIM}]")
            
        return chat_completion(messages)

    # Check for recommendation
    if LAST_MODIFIED_FILE:
        print_recommendation(LAST_MODIFIED_FILE)
        LAST_MODIFIED_FILE = None

    return msg.content

# =============================================================================
# MAIN
# =============================================================================
def main():
    print_banner()
    
    system_prompt = f"""You are Gemini Code.
    
    GUIDELINES:
    1. USE TOOLS. Don't just print code, write it.
    2. Be concise.
    
    REPORTING (When auditing/fixing):
    - Use Markdown 'diff' blocks to show changes.
    - This allows the user to see RED (Removed) and GREEN (Added) lines.
    
    Example:
    ```diff
    - vulnerable_query = f"SELECT * FROM users WHERE id={{id}}"
    + secure_query = "SELECT * FROM users WHERE id=?"
    ```
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Eva Style for Questionary
    eva_style = questionary.Style([
        ('qmark', 'fg:yellow bold'),
        ('question', 'fg:yellow bold'),
        ('answer', 'fg:cyan bold'),
    ])

    while True:
        try:
            console.print(f"\n[{C_PRIMARY}]┌──( GEMINI )-[{C_SECONDARY}]{os.path.basename(WORKSPACE_DIR)}[/{C_SECONDARY}][/{C_PRIMARY}]")
            user_input = questionary.text("└─>", style=eva_style).ask()
            
            if not user_input: continue
            
            # COMMANDS
            if user_input.lower() == "/exit": break
            if user_input.lower() == "/clear":
                messages = [{"role": "system", "content": system_prompt}]
                console.print(f"[{C_DIM}]Memory Cleared.[/{C_DIM}]")
                continue
                
            if user_input.lower() == "/skyhammer":
                print_skyhammer_banner()
                
                # FLEXIBLE TARGET SELECTION
                target_type = questionary.select(
                    "SELECT TARGET TYPE:",
                    choices=["URL (Web App)", "File (Source Code)", "Directory (Scan All)"],
                    style=eva_style
                ).ask()
                
                target = ""
                if "URL" in target_type:
                    target = questionary.text("ENTER URL:", style=eva_style).ask()
                    prompt = f"SKYHAMMER ATTACK: Audit {target}. Test SQLi, XSS, CMD Injection. Show PROOF of exploits."
                elif "File" in target_type:
                    target = questionary.path("ENTER FILE PATH:", style=eva_style).ask()
                    prompt = f"SKYHAMMER AUDIT: Analyze {target}. Show BEFORE (Red) and AFTER (Green) diffs for fixes."
                else:
                    target = "."
                    prompt = "SKYHAMMER SCAN: Audit all python files in current dir."
                
                if not target: continue
                
                user_input = prompt
                console.print(f"[{C_ACCENT}]>> TARGET ACQUIRED: {target} <<[/{C_ACCENT}]")

            # CHAT
            messages.append({"role": "user", "content": f"Context:\n{get_context()}\n\n{user_input}"})
            response = chat_completion(messages)
            
            console.print()
            console.print(Markdown(response))
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted.[/red]")
            break

if __name__ == "__main__":
    main()