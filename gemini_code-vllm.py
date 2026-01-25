#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   SKYHAMMER // LOCAL VLLM CLIENT                                              ║
║   Evangelion/Cyberpunk Aesthetics for Autonomous Security                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python gemini_code-vllm.py
"""

import os
import sys
import json
import time
from openai import OpenAI  # <--- NEW: vLLM is OpenAI compatible

# =============================================================================
# DEPENDENCIES & THEME
# =============================================================================
try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.table import Table
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    print("CRITICAL ERROR: SYSTEM DEPENDENCIES MISSING.")
    print("RUN: pip install rich questionary openai")
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
# Connect to local vLLM server
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "skyhammer"  # <--- FIXED: Matches your launch command

try:
    client = OpenAI(
        base_url=VLLM_API_BASE,
        api_key=VLLM_API_KEY,
    )
except Exception as e:
    print(f"[!] FATAL: COULD NOT CONNECT TO LOCAL GPU SERVER: {e}")
    sys.exit(1)

console = Console()

# We will auto-detect models from the server
CURRENT_MODEL = "unknown" 
WORKSPACE_DIR = os.getcwd()

# =============================================================================
# UI COMPONENTS
# =============================================================================

def print_banner():
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" SYSTEM ONLINE // SKYHAMMER LOCAL ", justify="center", style="bold black on yellow"),
            style="bold yellow",
            border_style="yellow",
            box=box.HEAVY,
        )
    )
    console.print(grid)
    console.print(f"[{C_DIM}]:: NEURAL LINK ESTABLISHED :: LOCALHOST:8000 ::[/{C_DIM}]", justify="center")
    console.print()

# =============================================================================
# CORE LOOP
# =============================================================================

def get_available_models():
    """Fetch models loaded in vLLM"""
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        # Return the specific error message for debugging
        return [f"Error: {str(e)}"]

def chat_completion(messages):
    global CURRENT_MODEL
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style="bold yellow"),
        TextColumn(f"[{C_PRIMARY}]NEURAL SYNC ({CURRENT_MODEL})...[/{C_PRIMARY}]"),
        transient=True
    ) as progress:
        progress.add_task("waiting", total=None)
        
        try:
            # Auto-detect model if not set (handle restart/reconnect)
            if CURRENT_MODEL == "unknown":
                models = client.models.list()
                CURRENT_MODEL = models.data[0].id

            response = client.chat.completions.create(
                model=CURRENT_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                stop=["<|endoftext|>"],
                extra_body={"stop_token_ids": [151645]} 
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ MODEL ERROR: {str(e)}"

# =============================================================================
# MAIN
# =============================================================================
def main():
    global CURRENT_MODEL
    print_banner()
    
    # 1. Select Model (Base vs SkyHammer)
    available_models = get_available_models()
    
    if "Error" in available_models[0]:
        console.print(f"[{C_ERROR}]CRITICAL: Connection Failed.[/{C_ERROR}]")
        console.print(f"[{C_DIM}]Details: {available_models[0]}[/{C_DIM}]")
        console.print("\n[yellow]Troubleshooting:[/]")
        console.print("1. Is vLLM running? (ps aux | grep vllm)")
        console.print("2. Did you use the right API Key? (Script expects: 'skyhammer')")
        console.print("3. Is the port correct? (Script expects: 8000)")
        sys.exit(1)
        
    CURRENT_MODEL = questionary.select(
        "SELECT ACTIVE NEURAL MODEL:",
        choices=available_models,
        style=questionary.Style([('qmark', 'fg:yellow bold'),('question', 'fg:yellow bold'),('answer', 'fg:cyan bold')])
    ).ask()
    
    system_prompt = f"""You are SkyHammer, an automated security agent.
    GUIDELINES:
    1. You are an expert in Python security.
    2. When asked to audit code, output the VULNERABILITY, EXPLANATION, and FIXED CODE.
    3. Use Markdown blocks for code.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Eva Style
    eva_style = questionary.Style([
        ('qmark', 'fg:yellow bold'),
        ('question', 'fg:yellow bold'),
        ('answer', 'fg:cyan bold'),
    ])

    while True:
        try:
            console.print(f"\n[{C_PRIMARY}]┌──( SKYHAMMER )-[{C_SECONDARY}]LOCAL-GPU[/{C_SECONDARY}][/{C_PRIMARY}]")
            user_input = questionary.text("└─>", style=eva_style).ask()
            
            if not user_input: continue
            if user_input.lower() == "/exit": break
            
            # --- VULNERABILITY TEST MODE ---
            if user_input.lower() == "/test":
                user_input = """
                Audit this code:
                def login(u, p):
                    cursor.execute(f"SELECT * FROM users WHERE user='{u}' AND pass='{p}'")
                """
                console.print(f"[{C_ACCENT}]>> INJECTING TEST VULNERABILITY (SQLi)... <<[/{C_ACCENT}]")

            # CHAT
            messages.append({"role": "user", "content": user_input})
            response = chat_completion(messages)
            
            console.print()
            console.print(Markdown(response))
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted.[/red]")
            break

if __name__ == "__main__":
    main()