#!/usr/bin/env python3
"""
SkyHammer CAI Bridge

Bridges Alias Robotics' CAI Framework tools to Gemini's OpenAI function calling schema.
Provides access to professional-grade security tools: nmap, curl, netcat, etc.

Usage:
    from cai_bridge import CAIBridge
    bridge = CAIBridge()
    tools = bridge.get_tool_definitions()
    result = bridge.execute("nmap", {"args": "-sV", "target": "127.0.0.1"})
"""

import subprocess
import sys
import os
from typing import List, Dict, Any, Optional

# Add CAI to path
CAI_PATH = "/Users/administrator/Imperial-College-London/Projects/2025/2025-11 November/Iterate-RL-Hackathon/cai-vllm/src"
if CAI_PATH not in sys.path:
    sys.path.insert(0, CAI_PATH)

try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def log(msg, style="white"):
    if HAS_RICH and console:
        console.print(f"[{style}]{msg}[/]")
    else:
        print(msg)


class CAIBridge:
    """
    Bridges CAI Framework tools to OpenAI/Gemini function calling format.
    """

    # Tool definitions - maps to CAI tools
    TOOL_CATALOG = {
        "nmap": {
            "description": "Network scanner for host discovery, port scanning, and service detection. Use for reconnaissance.",
            "params": {
                "args": {"type": "string", "description": "Nmap arguments (e.g., '-sV -p 80,443', '-sC -sV', '-A')"},
                "target": {"type": "string", "description": "Target IP, hostname, or CIDR range"}
            },
            "required": ["target"],
            "command": "nmap {args} {target}"
        },
        "curl": {
            "description": "HTTP client for making web requests. Use for testing endpoints, sending payloads, and web exploitation.",
            "params": {
                "args": {"type": "string", "description": "Curl arguments (e.g., '-X POST', '-d data', '-H header')"},
                "target": {"type": "string", "description": "Target URL"}
            },
            "required": ["target"],
            "command": "curl {args} {target}"
        },
        "wget": {
            "description": "Download files from web servers. Use for retrieving resources and testing file access.",
            "params": {
                "args": {"type": "string", "description": "Wget arguments"},
                "target": {"type": "string", "description": "Target URL"}
            },
            "required": ["target"],
            "command": "wget {args} {target} -O -"
        },
        "netcat": {
            "description": "Network utility for reading/writing data across connections. Use for reverse shells, port scanning, banner grabbing.",
            "params": {
                "args": {"type": "string", "description": "Netcat arguments (e.g., '-v', '-z', '-l')"},
                "target": {"type": "string", "description": "Target host"},
                "port": {"type": "string", "description": "Target port"}
            },
            "required": ["target", "port"],
            "command": "nc {args} {target} {port}"
        },
        "sqlmap": {
            "description": "Automatic SQL injection detection and exploitation tool.",
            "params": {
                "args": {"type": "string", "description": "Sqlmap arguments"},
                "target": {"type": "string", "description": "Target URL with injection point"}
            },
            "required": ["target"],
            "command": "sqlmap {args} -u {target} --batch"
        },
        "gobuster": {
            "description": "Directory/file brute-forcer. Use for discovering hidden paths and files.",
            "params": {
                "mode": {"type": "string", "description": "Mode: dir, dns, vhost"},
                "target": {"type": "string", "description": "Target URL"},
                "wordlist": {"type": "string", "description": "Wordlist path (default: /usr/share/wordlists/dirb/common.txt)"}
            },
            "required": ["target"],
            "command": "gobuster {mode} -u {target} -w {wordlist}"
        },
        "nikto": {
            "description": "Web server scanner for vulnerabilities, misconfigurations, and dangerous files.",
            "params": {
                "args": {"type": "string", "description": "Nikto arguments"},
                "target": {"type": "string", "description": "Target URL or host"}
            },
            "required": ["target"],
            "command": "nikto {args} -h {target}"
        },
        "hydra": {
            "description": "Password brute-force tool for various protocols (SSH, FTP, HTTP, etc.).",
            "params": {
                "args": {"type": "string", "description": "Hydra arguments including service type"},
                "target": {"type": "string", "description": "Target host"}
            },
            "required": ["target", "args"],
            "command": "hydra {args} {target}"
        },
        "ffuf": {
            "description": "Fast web fuzzer for directory discovery and parameter fuzzing.",
            "params": {
                "args": {"type": "string", "description": "Ffuf arguments"},
                "target": {"type": "string", "description": "Target URL with FUZZ keyword"}
            },
            "required": ["target"],
            "command": "ffuf {args} -u {target}"
        },
        "whatweb": {
            "description": "Web technology fingerprinting tool. Identifies CMS, frameworks, server software.",
            "params": {
                "args": {"type": "string", "description": "WhatWeb arguments"},
                "target": {"type": "string", "description": "Target URL"}
            },
            "required": ["target"],
            "command": "whatweb {args} {target}"
        },
        "wpscan": {
            "description": "WordPress security scanner. Finds vulnerabilities, plugins, themes.",
            "params": {
                "args": {"type": "string", "description": "WPScan arguments"},
                "target": {"type": "string", "description": "Target WordPress URL"}
            },
            "required": ["target"],
            "command": "wpscan {args} --url {target}"
        },
        "shell_command": {
            "description": "Execute arbitrary shell commands. Use for general system interaction.",
            "params": {
                "command": {"type": "string", "description": "Shell command to execute"}
            },
            "required": ["command"],
            "command": "{command}"
        },
        "python_exec": {
            "description": "Execute Python code. Use for scripting, automation, and custom exploits.",
            "params": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
            "command": "python3 -c \"{code}\""
        },
        "base64_decode": {
            "description": "Decode base64 encoded strings.",
            "params": {
                "data": {"type": "string", "description": "Base64 encoded string"}
            },
            "required": ["data"],
            "command": "echo '{data}' | base64 -d"
        },
        "grep_file": {
            "description": "Search for patterns in files.",
            "params": {
                "pattern": {"type": "string", "description": "Pattern to search for"},
                "file": {"type": "string", "description": "File or path to search"}
            },
            "required": ["pattern", "file"],
            "command": "grep -r '{pattern}' {file}"
        },
        "find_files": {
            "description": "Find files by name or pattern.",
            "params": {
                "path": {"type": "string", "description": "Starting path"},
                "pattern": {"type": "string", "description": "Filename pattern"}
            },
            "required": ["pattern"],
            "command": "find {path} -name '{pattern}' 2>/dev/null"
        },
        "cat_file": {
            "description": "Read file contents.",
            "params": {
                "file": {"type": "string", "description": "File path to read"}
            },
            "required": ["file"],
            "command": "cat {file}"
        },
        "ls_dir": {
            "description": "List directory contents.",
            "params": {
                "path": {"type": "string", "description": "Directory path"},
                "args": {"type": "string", "description": "ls arguments (e.g., '-la')"}
            },
            "required": [],
            "command": "ls {args} {path}"
        }
    }

    def __init__(self, timeout: int = 60):
        """Initialize the CAI bridge."""
        self.timeout = timeout
        self.execution_log = []

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Convert CAI tools to OpenAI function calling schema.

        Returns:
            List of tool definitions in OpenAI format
        """
        definitions = []

        for tool_name, tool_info in self.TOOL_CATALOG.items():
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool_info["params"],
                        "required": tool_info["required"]
                    }
                }
            }
            definitions.append(schema)

        return definitions

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a CAI tool.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            Tool output as string
        """
        if tool_name not in self.TOOL_CATALOG:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self.TOOL_CATALOG.keys())}"

        tool_info = self.TOOL_CATALOG[tool_name]
        command_template = tool_info["command"]

        # Build command with arguments
        try:
            # Fill in template with provided args, use empty string for missing optional args
            filled_args = {}
            for param in tool_info["params"]:
                filled_args[param] = args.get(param, "")

            command = command_template.format(**filled_args)
            command = " ".join(command.split())  # Clean up extra spaces

            log(f"[CAI] Executing: {command[:80]}...", "bold red")

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"

            # Log execution
            self.execution_log.append({
                "tool": tool_name,
                "args": args,
                "command": command,
                "output_length": len(output)
            })

            # Truncate if too long
            if len(output) > 5000:
                output = output[:5000] + "\n...[TRUNCATED]..."

            return output if output else "(No output)"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout}s"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.TOOL_CATALOG.keys())

    def get_execution_log(self) -> List[Dict]:
        """Get log of all tool executions."""
        return self.execution_log


# Convenience function for quick tool execution
def run_cai_tool(tool_name: str, **kwargs) -> str:
    """
    Quick helper to run a CAI tool.

    Example:
        result = run_cai_tool("nmap", args="-sV", target="127.0.0.1")
    """
    bridge = CAIBridge()
    return bridge.execute(tool_name, kwargs)


if __name__ == "__main__":
    # Demo
    bridge = CAIBridge()

    print("Available CAI Tools:")
    for tool in bridge.get_available_tools():
        print(f"  - {tool}")

    print("\nTool definitions (OpenAI schema):")
    import json
    print(json.dumps(bridge.get_tool_definitions()[:2], indent=2))
