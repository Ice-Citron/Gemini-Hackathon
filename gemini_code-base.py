#!/usr/bin/env python3
"""
SKYHAMMER // GEMINI-CODE (Base Version - No CAI)
Evangelion/Cyberpunk Aesthetics for Autonomous Security

An autonomous AI system for security testing, code generation, and remediation.
Features:
- Interactive chat with codebase context
- Code generation and editing
- Security scanning (attack mode)
- Auto-patching (defense mode)
- Tool use (file operations, shell commands)

Usage:
    python gemini_code-base.py                    # Interactive mode
    python gemini_code-base.py "fix the bug"      # Single command mode
    python gemini_code-base.py --scan             # Security scan mode

Models:
    - gemini-2.0-flash: Fast, multimodal (recommended)
    - gemini-2.5-pro-preview-05-06: Most capable reasoning
    - gemini-1.5-pro: Balanced performance
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# Import from modules
from src.theme import (
    C_PRIMARY, C_SECONDARY, C_ACCENT, C_SUCCESS, C_ERROR, C_DIM, C_HIGHLIGHT,
    get_eva_style
)
from src.logging_utils import (
    init_logging, log_event, log_tool_call, log_api_call, log_conversation, get_run_dir, save_report
)
from src.ui_components import (
    print_banner, print_skyhammer_banner, print_mission_complete_banner,
    show_diff, show_new_file_preview, show_report_diff, show_goals, get_console
)
from src.tools import (
    TOOLS, BASIC_TOOLS, execute_tool, get_codebase_context, is_safe_path
)

# Load API Key
try:
    from secretsConfig import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

if not GEMINI_API_KEY:
    print("[!] FATAL: NEURAL LINK SEVERED (GEMINI_API_KEY missing)")
    print("    Set GEMINI_API_KEY in secretsConfig.py or environment")
    sys.exit(1)

# Initialize Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)
console = Console() if HAS_RICH else None

# Available Gemini models with pricing (per million tokens: input/output)
MODELS = {
    "gemini-3-flash-preview": {"price": "$0.50/$3.00", "desc": "Gemini 3 Flash - Latest, 3x faster than 2.5 Pro"},
    "gemini-3-pro-preview": {"price": "$2.00/$12.00", "desc": "Gemini 3 Pro - Most capable"},
    "gemini-2.5-pro": {"price": "$1.25/$10.00", "desc": "Strong reasoning, long context"},
    "gemini-2.5-flash": {"price": "$0.15/$0.60", "desc": "Fast with thinking mode"},
    "gemini-2.5-flash-lite": {"price": "$0.10/$0.40", "desc": "Most economical"},
    "gemini-2.0-flash": {"price": "$0.10/$0.40", "desc": "Fast, multimodal"},
}

# Current settings
CURRENT_MODEL = "gemini-3-flash-preview"
SKYHAMMER_MODE = False  # When True, enables security tools
WORKSPACE_DIR = os.getcwd()
AUTO_APPROVE = False
INTERRUPT_REQUESTED = False
LISTENER_PAUSED = False

# Task tracking
CURRENT_TASKS = []
CURRENT_TASK_STATUS = "idle"

# Backup system for /undo
FILE_BACKUPS = {}

# Track if we just created a file (for recommendation)
LAST_CREATED_FILE = None


def pause_listener():
    global LISTENER_PAUSED
    LISTENER_PAUSED = True


def resume_listener():
    global LISTENER_PAUSED
    LISTENER_PAUSED = False


def set_task_status(status: str, tasks: List[str] = None):
    global CURRENT_TASK_STATUS, CURRENT_TASKS
    CURRENT_TASK_STATUS = status
    if tasks is not None:
        CURRENT_TASKS = tasks


def ask_permission(tool_name: str, args: Dict[str, Any], preview_content: str = "") -> tuple:
    """
    Evangelion-style Permission Prompt.
    Returns (action, feedback) where action is 'yes', 'no', 'yes_all', or 'feedback'
    """
    global AUTO_APPROVE

    if AUTO_APPROVE:
        return ("yes", None)

    pause_listener()

    try:
        console.print(f"\n[{C_ACCENT}]>> INTERVENTION REQUIRED: {tool_name.upper()} <<[/]")

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
            console.print(f"[{C_DIM}]{json.dumps(args)[:200]}[/]")

        eva_style = get_eva_style()

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
        resume_listener()


def convert_tools_to_gemini(openai_tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI-style tool definitions to Gemini format"""
    gemini_tools = []
    for tool in openai_tools:
        func = tool.get("function", {})
        params = func.get("parameters", {})

        # Convert parameters to Gemini schema format
        properties = params.get("properties", {})
        required = params.get("required", [])

        gemini_props = {}
        for prop_name, prop_def in properties.items():
            gemini_prop = {"type": prop_def.get("type", "string").upper()}
            if "description" in prop_def:
                gemini_prop["description"] = prop_def["description"]
            if "enum" in prop_def:
                gemini_prop["enum"] = prop_def["enum"]
            gemini_props[prop_name] = gemini_prop

        gemini_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": {
                "type": "OBJECT",
                "properties": gemini_props,
                "required": required
            }
        })

    return gemini_tools


def parse_json_tool_calls(text: str) -> List[Dict]:
    """
    Parse JSON tool calls from text response as a fallback.
    Handles cases where Gemini outputs tool calls as JSON in text instead of using function calling.
    """
    import re
    tool_calls = []

    # Pattern to match JSON objects with "action" and "command" or tool-like structures
    patterns = [
        r'\{\s*"action"\s*:\s*"([^"]+)"\s*,\s*"command"\s*:\s*"([^"]+)"\s*\}',
        r'\{\s*"action"\s*:\s*"([^"]+)"\s*,\s*"path"\s*:\s*"([^"]+)"\s*\}',
        r'\{\s*"action"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"\s*\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            action = match[0]
            arg = match[1]

            # Map action names to tool names
            tool_map = {
                "run_command": ("run_command", {"command": arg}),
                "read_file": ("read_file", {"path": arg}),
                "write_file": ("write_file", {"path": arg}),
                "list_files": ("list_files", {"path": arg}),
                "search_code": ("search_code", {"pattern": arg}),
                "security_scan": ("security_scan", {"target": arg}),
            }

            if action in tool_map:
                tool_name, tool_args = tool_map[action]
                tool_calls.append({"name": tool_name, "args": tool_args})

    return tool_calls


def chat_completion(messages: List[Dict], use_tools: bool = True) -> str:
    """Run agentic chat completion with Gemini and cyberpunk spinner"""
    global CURRENT_MODEL, SKYHAMMER_MODE, INTERRUPT_REQUESTED, LAST_CREATED_FILE
    model_id = CURRENT_MODEL
    max_iterations = 100
    iteration = 0

    # Select tools based on mode
    if use_tools:
        if SKYHAMMER_MODE:
            active_tools = TOOLS
        else:
            active_tools = BASIC_TOOLS
    else:
        active_tools = None

    if console:
        console.print(f"[{C_DIM}](Press Ctrl+C to interrupt, /goals to see tasks)[/]")

    set_task_status("thinking")

    # Convert tools to Gemini format
    gemini_tools = None
    if active_tools:
        gemini_tool_defs = convert_tools_to_gemini(active_tools)
        gemini_tools = [genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            k: genai.protos.Schema(
                                type=getattr(genai.protos.Type, v.get("type", "STRING")),
                                description=v.get("description", ""),
                                enum=v.get("enum", []) if "enum" in v else None
                            ) for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", [])
                    )
                ) for t in gemini_tool_defs
            ]
        )]

    # Initialize model with safety settings disabled for security testing
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Configure tool behavior - AUTO means model decides when to call tools
    tool_config = None
    if gemini_tools:
        tool_config = genai.protos.ToolConfig(
            function_calling_config=genai.protos.FunctionCallingConfig(
                mode=genai.protos.FunctionCallingConfig.Mode.AUTO
            )
        )

    model = genai.GenerativeModel(
        model_name=model_id,
        safety_settings=safety_settings,
        tools=gemini_tools if gemini_tools else None,
        tool_config=tool_config
    )

    # Convert messages to Gemini format
    # Extract system prompt and build history
    system_instruction = None
    gemini_history = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content
        elif role == "user":
            gemini_history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            gemini_history.append({"role": "model", "parts": [content]})
        elif role == "tool":
            # Tool results go as function responses
            pass  # Handled in the loop below

    # Create chat with system instruction
    if system_instruction:
        model = genai.GenerativeModel(
            model_name=model_id,
            safety_settings=safety_settings,
            tools=gemini_tools if gemini_tools else None,
            tool_config=tool_config,
            system_instruction=system_instruction
        )

    chat = model.start_chat(history=gemini_history[:-1] if gemini_history else [])

    # Get the last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    try:
        while iteration < max_iterations:
            iteration += 1

            if INTERRUPT_REQUESTED:
                set_task_status("idle")
                return "(Interrupted by user)"

            set_task_status("thinking")

            with Progress(
                SpinnerColumn(spinner_name="dots12", style="bold yellow"),
                TextColumn(f"[{C_PRIMARY}]{{task.description}}[/]"),
                transient=True
            ) as progress:
                task_id = progress.add_task(f"SYNCING WITH NEURAL NETWORK [CYCLE {iteration}]...", total=None)

                if iteration == 1:
                    # First iteration: send the user message
                    response = chat.send_message(last_user_msg)
                else:
                    # Subsequent iterations: send tool results
                    response = chat.send_message(tool_response_parts)

            # Check for function calls
            function_calls = []
            text_response = ""

            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
                elif hasattr(part, 'text') and part.text:
                    text_response += part.text

            # Fallback: parse JSON tool calls from text if no function calls detected
            parsed_tool_calls = []
            if not function_calls and text_response and use_tools:
                parsed_tool_calls = parse_json_tool_calls(text_response)
                if parsed_tool_calls and console:
                    console.print(f"[{C_ACCENT}]>> DETECTED {len(parsed_tool_calls)} TOOL CALLS IN TEXT (fallback) <<[/]")

            tool_call_names = [fc.name for fc in function_calls] if function_calls else [tc["name"] for tc in parsed_tool_calls]
            log_api_call(model_id, len(messages), text_response, tool_call_names)

            if tool_call_names:
                set_task_status("executing", tool_call_names)

            if not function_calls and not parsed_tool_calls:
                set_task_status("done")

                # Check if we created a file and should recommend SkyHammer
                if LAST_CREATED_FILE and not SKYHAMMER_MODE:
                    print_mission_complete_banner(LAST_CREATED_FILE)
                    LAST_CREATED_FILE = None

                return text_response or "(No response)"

            if INTERRUPT_REQUESTED:
                set_task_status("idle")
                return "(Interrupted by user)"

            total_calls = len(function_calls) + len(parsed_tool_calls)
            if console:
                console.print(f"[{C_DIM}]>> CYCLE {iteration}: {total_calls} TOOL CALLS <<[/]")

            # Execute tools and collect results
            tool_response_parts = []
            tool_results_text = []

            # Handle native Gemini function calls
            for fc in function_calls:
                if INTERRUPT_REQUESTED:
                    return "(Interrupted by user mid-execution)"

                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                if console:
                    console.print(f"[{C_SECONDARY}]-> {tool_name}[/]")

                result = execute_tool(
                    tool_name, tool_args, WORKSPACE_DIR,
                    console, ask_permission, FILE_BACKUPS
                )

                # Track file creation for recommendation
                if tool_name == "write_file" and "SUCCESS" in result:
                    LAST_CREATED_FILE = tool_args.get("path", "")

                # Build function response for Gemini
                tool_response_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result}
                        )
                    )
                )

            # Handle parsed JSON tool calls (fallback)
            for tc in parsed_tool_calls:
                if INTERRUPT_REQUESTED:
                    return "(Interrupted by user mid-execution)"

                tool_name = tc["name"]
                tool_args = tc["args"]

                if console:
                    console.print(f"[{C_SECONDARY}]-> {tool_name} (parsed)[/]")

                result = execute_tool(
                    tool_name, tool_args, WORKSPACE_DIR,
                    console, ask_permission, FILE_BACKUPS
                )

                # Track file creation for recommendation
                if tool_name == "write_file" and "SUCCESS" in result:
                    LAST_CREATED_FILE = tool_args.get("path", "")

                tool_results_text.append(f"Tool {tool_name} result: {result}")

            # If we used parsed tool calls, send results as text (since we can't use function_response)
            if parsed_tool_calls and not function_calls:
                tool_response_parts = "\n\n".join(tool_results_text) + "\n\nContinue with the task based on these results. Use function calls (not JSON in text) for any additional tool usage."

            if text_response and console:
                preview = text_response[:200] + "..." if len(text_response) > 200 else text_response
                console.print(f"[{C_DIM} italic]{preview}[/]")

        return "(Max iterations reached - use /clear to reset)"

    except KeyboardInterrupt:
        console.print(f"\n[{C_ACCENT}]>> INTERRUPT SIGNAL RECEIVED <<[/]")
        return "(Interrupted by user)"
    except Exception as e:
        if console:
            console.print(f"[{C_ERROR}]API Error: {str(e)}[/]")
        return f"(Error: {str(e)})"


def interactive_mode():
    """Run interactive chat mode with Evangelion/Cyberpunk UI"""
    global CURRENT_MODEL, SKYHAMMER_MODE, WORKSPACE_DIR, AUTO_APPROVE, GEMINI_API_KEY

    if not console:
        print("Rich library required for interactive mode")
        return

    # Initialize logging
    run_dir = init_logging(CURRENT_MODEL, WORKSPACE_DIR)
    console.print(f"[{C_DIM}]SESSION LOG: {run_dir}[/]")

    # Display NERV-style banner
    print_banner(CURRENT_MODEL)

    auto_status = f"[{C_PRIMARY}]ON[/]" if AUTO_APPROVE else f"[{C_DIM}]OFF[/]"
    console.print(f"[{C_DIM}]COMMANDS: /help, /auto, /workspace, /skyhammer, /model, /clear, /exit[/]")
    console.print(f"[{C_DIM}]BASH MODE: !command (e.g. !ls, !pwd, !python app.py &)[/]")
    console.print(f"[{C_DIM}]MODEL: {CURRENT_MODEL} | AUTO-APPROVE: {auto_status}[/]")
    console.print(f"[{C_DIM}]WORKSPACE: {WORKSPACE_DIR}[/]")
    console.print()

    # Build system prompt
    context = get_codebase_context(WORKSPACE_DIR)
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
- Always end with a SUMMARY of what you did

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
----------------------------------------
VULNERABILITY REPORT: app.py
----------------------------------------

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
    eva_style = get_eva_style()

    while True:
        try:
            console.print(f"\n[{C_PRIMARY}]----( GEMINI@SKYHAMMER )-[{C_SECONDARY}]{os.path.basename(WORKSPACE_DIR)}[/][/]")

            user_input = questionary.text(
                "-->",
                qmark="",
                style=eva_style
            ).ask()

            if not user_input:
                continue

            # Handle ! bash mode
            if user_input.startswith("!"):
                bash_cmd = user_input[1:].strip()
                if bash_cmd:
                    console.print(f"[{C_DIM}]$ {bash_cmd}[/]")
                    try:
                        result = subprocess.run(
                            bash_cmd, shell=True, capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout + result.stderr
                        if output.strip():
                            console.print(Panel(output[:3000], title="SHELL OUTPUT", border_style="cyan"))
                        else:
                            console.print(f"[{C_DIM}](no output)[/]")
                    except subprocess.TimeoutExpired:
                        console.print(f"[{C_ERROR}]TIMEOUT: Command exceeded 60s limit[/]")
                    except Exception as e:
                        console.print(f"[{C_ERROR}]ERROR: {e}[/]")
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()

                if cmd == "/exit" or cmd == "/quit":
                    console.print(f"\n[{C_ACCENT}]:: TERMINATING LINK :: GOODBYE ::[/]")
                    break

                elif cmd == "/clear":
                    messages = [{"role": "system", "content": system_prompt}]
                    set_task_status("idle", [])
                    console.print(f"[{C_DIM}]>> MEMORY BANKS CLEARED <<[/]")
                    continue

                elif cmd == "/goals" or cmd == "/tasks":
                    show_goals(CURRENT_TASKS, CURRENT_TASK_STATUS)
                    continue

                elif cmd == "/undo" or cmd == "/rollback":
                    if not FILE_BACKUPS:
                        console.print(f"[{C_ACCENT}]>> NO RESTORATION POINTS AVAILABLE <<[/]")
                        continue

                    console.print(f"\n[{C_PRIMARY}]AVAILABLE RESTORATION POINTS:[/]")
                    backup_list = list(FILE_BACKUPS.keys())
                    for i, path in enumerate(backup_list, 1):
                        console.print(f"  {i}. {path}")

                    if len(backup_list) == 1:
                        path = backup_list[0]
                        with open(path, "w") as f:
                            f.write(FILE_BACKUPS[path])
                        console.print(f"\n[{C_SUCCESS}]>> SYSTEM RESTORED: {os.path.basename(path)} <<[/]")
                        del FILE_BACKUPS[path]
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
                                console.print(f"[{C_SUCCESS}]RESTORED: {path}[/]")
                                if os.path.exists(path + ".bak"):
                                    os.remove(path + ".bak")
                            FILE_BACKUPS.clear()
                            console.print(f"[{C_SUCCESS}]>> ALL SYSTEMS RESTORED <<[/]")
                        else:
                            for path in backup_list:
                                if os.path.basename(path) == choice:
                                    with open(path, "w") as f:
                                        f.write(FILE_BACKUPS[path])
                                    console.print(f"[{C_SUCCESS}]RESTORED: {path}[/]")
                                    del FILE_BACKUPS[path]
                                    if os.path.exists(path + ".bak"):
                                        os.remove(path + ".bak")
                                    break
                    continue

                elif cmd == "/pr" or cmd.startswith("/pr "):
                    gh_check = subprocess.run("which gh", shell=True, capture_output=True, text=True)
                    if gh_check.returncode != 0:
                        console.print(f"[{C_ERROR}]GITHUB CLI (gh) NOT DETECTED[/]")
                        console.print(f"[{C_DIM}]Install with: brew install gh[/]")
                        continue

                    git_check = subprocess.run("git status", shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
                    if git_check.returncode != 0:
                        console.print(f"[{C_ERROR}]NOT IN A GIT REPOSITORY[/]")
                        continue

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

                    branch_name = f"security-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    console.print(f"[{C_SECONDARY}]>> CREATING BRANCH: {branch_name} <<[/]")

                    cmds = [
                        f"git checkout -b {branch_name}",
                        "git add -A",
                        f'git commit -m "{pr_title}"',
                        f"git push -u origin {branch_name}",
                        f'gh pr create --title "{pr_title}" --body "## Security Fixes\\n\\nThis PR contains security patches generated by SkyHammer.\\n\\n### Changes\\n- Fixed identified vulnerabilities\\n- Applied secure coding practices\\n\\n---\\nGenerated by SkyHammer - AI Security Agent"'
                    ]

                    for c in cmds:
                        console.print(f"[{C_DIM}]$ {c}[/]")
                        result = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=WORKSPACE_DIR)
                        if result.returncode != 0 and "nothing to commit" not in result.stderr:
                            console.print(f"[{C_ERROR}]ERROR: {result.stderr}[/]")
                            break
                        if result.stdout:
                            console.print(result.stdout)

                    console.print(f"[{C_SUCCESS}]PULL REQUEST CREATED[/]")
                    continue

                elif cmd == "/help":
                    help_text = f"""
[{C_PRIMARY}]SYSTEM COMMANDS:[/]
  /help           - Display this help panel
  /goals          - Show current tasks in progress
  /undo           - Rollback last file modification
  /pr [title]     - Create GitHub Pull Request
  /workspace PATH - Change workspace (current: {WORKSPACE_DIR})
  /skyhammer      - Engage SKYHAMMER security mode
  /auto           - Toggle auto-approve mode
  /model          - Switch between Gemini models
  /clear          - Clear conversation memory
  /exit           - Terminate session

[{C_PRIMARY}]BASH MODE (!):[/]
  !pwd            - Print working directory
  !ls -la         - List files
  !python app.py  - Run a script

[{C_PRIMARY}]INTERRUPT:[/]
  Ctrl+C          - Emergency stop mid-workflow

[{C_PRIMARY}]SAFETY PROTOCOLS:[/]
  - Split-pane diff visualization
  - Auto-backup of modified files (.bak)
  - /undo for instant rollback
  - Permission prompts for all actions

[{C_PRIMARY}]AUTHORIZATION OPTIONS:[/]
  - EXECUTE       - Allow this action
  - EXECUTE ALL   - Auto-approve session
  - DENY          - Skip this action
  - REVISE        - Provide new instructions

[{C_PRIMARY}]SKYHAMMER PROTOCOL:[/]
  - Scans for vulnerabilities
  - Runs actual exploit tests
  - Generates security reports
  - Creates patches with /pr"""
                    console.print(Panel(help_text, title=f"[{C_HIGHLIGHT}] GEMINI-CODE MANUAL [/]", border_style="yellow", box=box.DOUBLE))
                    continue

                elif cmd.startswith("/workspace"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_path = os.path.abspath(os.path.expanduser(parts[1]))
                        if os.path.isdir(new_path):
                            WORKSPACE_DIR = new_path
                            os.chdir(new_path)
                            console.print(f"[{C_SUCCESS}]>> WORKSPACE REDIRECTED: {WORKSPACE_DIR} <<[/]")
                        else:
                            console.print(f"[{C_ERROR}]DIRECTORY NOT FOUND: {new_path}[/]")
                    else:
                        console.print(f"[{C_SECONDARY}]CURRENT WORKSPACE: {WORKSPACE_DIR}[/]")
                        console.print(f"[{C_DIM}]Usage: /workspace /path/to/dir[/]")
                    continue

                elif cmd == "/auto":
                    AUTO_APPROVE = not AUTO_APPROVE
                    if AUTO_APPROVE:
                        console.print(f"[{C_ACCENT}]>> AUTO-APPROVE: ENABLED - All actions will execute without confirmation <<[/]")
                    else:
                        console.print(f"[{C_SUCCESS}]>> AUTO-APPROVE: DISABLED - You will be prompted for each action <<[/]")
                    continue

                elif cmd == "/skyhammer":
                    # SKYHAMMER MODE - Manual activation with target selection
                    SKYHAMMER_MODE = True
                    print_skyhammer_banner()

                    # Prompt for target type
                    target_type = questionary.select(
                        "SELECT TARGET TYPE:",
                        choices=[
                            "Local file (Python, JS, etc.)",
                            "Local directory (scan all files)",
                            "URL (paste a web app URL)",
                            "SKIP - I'll specify later"
                        ],
                        style=eva_style
                    ).ask()

                    if target_type and "SKIP" not in target_type:
                        if "URL" in target_type:
                            # Allow pasting URL
                            target = questionary.text(
                                "ENTER TARGET URL (paste link):",
                                style=eva_style
                            ).ask()
                            if target:
                                set_task_status("executing", [
                                    "Probe target for vulnerabilities",
                                    "Test SQL injection",
                                    "Test XSS",
                                    "Test command injection",
                                    "Generate report"
                                ])
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
- BEFORE vs AFTER code if source available (BEFORE in red context, AFTER in green context)
- Hacker's Note explaining why it works

START TESTING NOW."""
                                messages.append({"role": "user", "content": user_input})
                                log_conversation("user", user_input)
                                console.print(f"\n[{C_PRIMARY}]>> TARGET ACQUIRED: {target} <<[/]")
                                console.print(f"[{C_DIM}]Initiating security scan...[/]\n")
                                try:
                                    response = chat_completion(messages, use_tools=True)
                                    console.print()
                                    console.print(Panel(response, border_style="yellow", title=f"[{C_HIGHLIGHT}] SKYHAMMER REPORT [/]", box=box.DOUBLE))
                                    messages.append({"role": "assistant", "content": response})
                                    log_conversation("assistant", response)
                                    # Save report to logs
                                    report_path = save_report(response, target, "skyhammer")
                                    if report_path:
                                        console.print(f"[{C_DIM}]Report saved: {report_path}[/]")
                                    set_task_status("done")
                                except Exception as e:
                                    console.print(f"[{C_ERROR}]ERROR: {e}[/]")
                                    set_task_status("idle")
                        else:
                            # Local file/directory
                            target = questionary.path(
                                "ENTER FILE/DIRECTORY PATH:",
                                style=eva_style
                            ).ask()
                            if target and os.path.exists(target):
                                set_task_status("executing", [
                                    f"Read and analyze {target}",
                                    "Identify vulnerability patterns",
                                    "Start server if web app",
                                    "Run exploit tests",
                                    "Generate vulnerability report"
                                ])
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
- BEFORE (vulnerable code) - show in context that this is the BAD/RED code
- AFTER (fixed code) - show in context that this is the GOOD/GREEN code
- Hacker's Note explaining why exploit works

THEN PATCH THE ORIGINAL FILE:
- Overwrite {target} with the secure version (NOT a new _secure.py file!)
- The user will confirm before the patch is applied
- Show side-by-side diff of changes

START TESTING NOW."""
                                messages.append({"role": "user", "content": user_input})
                                log_conversation("user", user_input)
                                console.print(f"\n[{C_PRIMARY}]>> TARGET ACQUIRED: {target} <<[/]")
                                console.print(f"[{C_DIM}]Analyzing and testing for vulnerabilities...[/]\n")
                                try:
                                    response = chat_completion(messages, use_tools=True)
                                    console.print()
                                    console.print(Panel(response, border_style="yellow", title=f"[{C_HIGHLIGHT}] SKYHAMMER REPORT [/]", box=box.DOUBLE))
                                    messages.append({"role": "assistant", "content": response})
                                    log_conversation("assistant", response)
                                    # Save report to logs
                                    report_path = save_report(response, target, "skyhammer")
                                    if report_path:
                                        console.print(f"[{C_DIM}]Report saved: {report_path}[/]")
                                    set_task_status("done")
                                except Exception as e:
                                    console.print(f"[{C_ERROR}]ERROR: {e}[/]")
                                    set_task_status("idle")
                            elif target:
                                console.print(f"[{C_ERROR}]PATH NOT FOUND: {target}[/]")
                    else:
                        console.print(f"[{C_DIM}]>> SKYHAMMER MODE ACTIVE - specify target in your next message <<[/]")
                    continue

                elif cmd == "/apikey":
                    new_key = questionary.password("ENTER GEMINI API KEY:", style=eva_style).ask()
                    if new_key and new_key.startswith("AIza"):
                        GEMINI_API_KEY = new_key
                        genai.configure(api_key=GEMINI_API_KEY)
                        console.print(f"[{C_SUCCESS}]>> API KEY UPDATED SUCCESSFULLY <<[/]")
                    else:
                        console.print(f"[{C_ERROR}]INVALID API KEY FORMAT (should start with 'AIza')[/]")
                    continue

                elif cmd.startswith("/model"):
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
                        console.print(f"[{C_SUCCESS}]>> MODEL SWITCHED TO: {CURRENT_MODEL} <<[/]")
                    continue

                else:
                    console.print(f"[{C_ERROR}]UNKNOWN COMMAND: {cmd}[/]")
                    continue

            # Add user message
            messages.append({"role": "user", "content": user_input})
            log_conversation("user", user_input)

            console.print()

            try:
                response = chat_completion(messages, use_tools=True)

                console.print()
                if "```" in response:
                    console.print(Markdown(response))
                else:
                    console.print(Panel(response, border_style="cyan", title=f"[{C_SECONDARY}]MISSION REPORT[/]", box=box.ROUNDED))

                messages.append({"role": "assistant", "content": response})
                log_conversation("assistant", response)

            except Exception as e:
                console.print(f"[{C_ERROR}]SYSTEM ERROR: {e}[/]")

            console.print()

        except KeyboardInterrupt:
            console.print(f"\n[{C_ACCENT}]>> Use /exit to terminate session <<[/]")
            continue


def single_command_mode(prompt: str):
    """Run a single command and exit"""
    global SKYHAMMER_MODE
    SKYHAMMER_MODE = True

    context = get_codebase_context(WORKSPACE_DIR)
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

    console.print(f"\n[{C_ERROR}]:: SECURITY SCAN MODE ACTIVATED ::[/]")

    target = questionary.text("Enter file or URL to scan:").ask()
    if not target:
        return

    console.print(f"\n[yellow]Scanning {target}...[/]\n")

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
    python gemini_code-base.py                    # Interactive mode
    python gemini_code-base.py "explain this code"     # Single command
    python gemini_code-base.py --scan                  # Security scan mode
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
