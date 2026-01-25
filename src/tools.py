#!/usr/bin/env python3
"""
SkyHammer Tool Definitions and Execution
Function calling tools for Gemini
"""

import os
import json
import glob
import subprocess
import time
from typing import Dict, Any, Callable, Optional

from .theme import C_DIM, C_SUCCESS, C_SECONDARY, C_ERROR
from .logging_utils import log_tool_call

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

# Get basic tools (non-security)
BASIC_TOOLS = [t for t in TOOLS if t["function"]["name"] in
               ["read_file", "write_file", "list_files", "run_command", "search_code"]]


def is_safe_path(path: str, workspace_dir: str) -> bool:
    """Check if path is within the workspace (sandbox)"""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(workspace_dir)


def execute_tool(
    name: str,
    args: Dict[str, Any],
    workspace_dir: str,
    console,
    ask_permission_fn: Callable,
    file_backups: Dict[str, str]
) -> str:
    """Execute a tool and return the result"""
    start_time = time.time()

    try:
        if name == "read_file":
            path = args.get("path", "")
            # Make relative paths absolute within workspace
            if not os.path.isabs(path):
                path = os.path.join(workspace_dir, path)
            if not is_safe_path(path, workspace_dir):
                result = f"Error: Access denied - path outside workspace: {path}"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                if console:
                    console.print(f"[{C_DIM}]< READ: {os.path.basename(path)} ({len(content)} bytes) >[/]")
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
            action, feedback = ask_permission_fn("write_file", args)
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
                path = os.path.join(workspace_dir, path)
            if not is_safe_path(path, workspace_dir):
                result = f"Error: Access denied - path outside workspace: {path}"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result

            # BACKUP: Save original content for /undo (if file exists)
            if os.path.exists(path):
                with open(path, "r") as f:
                    file_backups[path] = f.read()
                # Also save .bak file
                with open(path + ".bak", "w") as f:
                    f.write(file_backups[path])
                if console:
                    console.print(f"[{C_DIM}]BACKUP CREATED: {path}.bak[/]")

            # Create parent directories if needed
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            if console:
                console.print(f"[{C_SUCCESS}]WRITE CONFIRMED: {os.path.basename(args.get('path', path))}[/]")
            result = f"SUCCESS: Wrote {len(content)} bytes to {args.get('path', path)}"
            log_tool_call(name, args, result, (time.time() - start_time) * 1000)
            return result

        elif name == "list_files":
            path = args.get("path", ".")
            pattern = args.get("pattern", "*")
            if not os.path.isabs(path):
                path = os.path.join(workspace_dir, path)
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
            action, feedback = ask_permission_fn("run_command", args)
            if action == "no":
                result = "Action skipped by user"
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            elif action == "feedback":
                result = f"USER FEEDBACK: {feedback}\n\nPlease try a different approach based on the user's feedback."
                log_tool_call(name, args, result, (time.time() - start_time) * 1000)
                return result
            if console:
                console.print(f"[{C_SECONDARY}]>> EXECUTING: {command}[/]")
            from rich.panel import Panel
            proc_result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=120, cwd=workspace_dir
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
                console.print(f"[{C_DIM}](No output)[/]")
            result = f"Command output (exit {proc_result.returncode}):\n```\n{output[:2000]}\n```"
            log_tool_call(name, args, result, (time.time() - start_time) * 1000)
            return result

        elif name == "search_code":
            pattern = args.get("pattern", "")
            file_pattern = args.get("file_pattern", "*.py")
            result = subprocess.run(
                f'grep -rn "{pattern}" --include="{file_pattern}" .',
                shell=True, capture_output=True, text=True, timeout=10,
                cwd=workspace_dir
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


def get_codebase_context(workspace_dir: str) -> str:
    """Get context about the current codebase"""
    context_parts = []
    original_dir = os.getcwd()

    try:
        os.chdir(workspace_dir)

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

    finally:
        os.chdir(original_dir)

    return "\n".join(context_parts) if context_parts else "No context files found."
