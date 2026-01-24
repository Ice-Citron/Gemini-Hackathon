#!/usr/bin/env python3
"""
SkyHammer Demo UI

A rich terminal UI that orchestrates the full attack -> patch -> verify loop.

Usage:
    python ui.py

Prerequisites:
    - Run 'sudo python mock_dvwa.py' in another terminal first
    - pip install rich
"""

import time
import subprocess
import os
import shutil
import sys

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.table import Table
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("[!] Error: 'rich' library not found. Install with: pip install rich")
    sys.exit(1)

console = Console()

# == CONFIGURATION ==
VULNERABLE_FILE = "mock_dvwa.py"
SECURE_FILE = "mock_dvwa_secure.py"
TEMP_BACKUP = "mock_dvwa.bak"


def make_layout():
    """Create the terminal layout"""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


class DemoOrchestrator:
    """Orchestrates the SkyHammer demo"""

    def __init__(self):
        self.layout = make_layout()
        self.logs = []
        self.status = "SYSTEM ONLINE"
        self.attack_success = None

    def update_header(self):
        """Update the header panel with current status"""
        if "SECURED" in self.status:
            color = "green"
        elif "ATTACK" in self.status or "INTRUSION" in self.status or "CRITICAL" in self.status:
            color = "red"
        elif "PATCH" in self.status or "ANALYZING" in self.status:
            color = "yellow"
        else:
            color = "blue"

        title = Text("SkyHammer | Autonomous Defense System", style="bold white")
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_row(title, Text(f"Status: {self.status}", style=f"bold {color}"))

        border_color = color if color != "blue" else "cyan"
        return Panel(grid, style=f"white on {color}" if color == "red" else f"white on black", border_style=border_color)

    def add_log(self, message, style="white"):
        """Add a log entry"""
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[dim]{timestamp}[/dim] [{style}]{message}[/]")
        if len(self.logs) > 18:
            self.logs.pop(0)

    def get_log_panel(self):
        """Get the log panel"""
        log_text = "\n".join(self.logs) if self.logs else "[dim]Waiting for activity...[/dim]"
        return Panel(log_text, title="[bold cyan]Operation Logs[/]", border_style="cyan", box=box.ROUNDED)

    def get_code_panel(self, file_path, title, highlight_line=None):
        """Get a code panel showing the file"""
        if not os.path.exists(file_path):
            return Panel(f"[red]File not found: {file_path}[/]", title=title)

        with open(file_path, "r") as f:
            code = f.read()

        # Show only the sqli function (the interesting part)
        if "def sqli():" in code:
            start = code.find("def sqli():")
            end = code.find("\nif __name__", start)
            if end == -1:
                end = start + 800
            code = code[start:end]

        syntax = Syntax(code, "python", theme="monokai", line_numbers=True, word_wrap=True)
        return Panel(syntax, title=f"[bold yellow]{title}[/]", border_style="yellow", box=box.ROUNDED)

    def run_attack(self, quiet=True):
        """Run the attack and capture result"""
        cmd = "python attack.py --challenge sqli"
        if quiet:
            cmd += " --quiet"

        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        success = False
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue

            if "Tool:" in line:
                tool_info = line.split("Tool:")[1][:40] if "Tool:" in line else line[:40]
                self.add_log(f"AGENT: {tool_info}...", "blue")
            elif "Login SUCCESS" in line:
                self.add_log("AUTH: Session established", "cyan")
            elif "Success: True" in line:
                self.add_log("CRITICAL: VULNERABILITY EXPLOITED!", "bold red")
                success = True
            elif "Success: False" in line:
                self.add_log("DEFENSE: Attack BLOCKED!", "bold green")
                success = False
            elif "SUBMITTED" in line:
                self.add_log("AGENT: Submitting findings...", "yellow")

        process.wait()
        return success

    def update_display(self, live):
        """Refresh all panels"""
        self.layout["header"].update(self.update_header())
        self.layout["left"].update(self.get_log_panel())
        live.refresh()

    def run_demo(self):
        """Run the full demo sequence"""
        console.clear()

        # Check prerequisites
        if not os.path.exists(VULNERABLE_FILE):
            console.print(f"[red]Error: {VULNERABLE_FILE} not found![/]")
            return
        if not os.path.exists(SECURE_FILE):
            console.print(f"[red]Error: {SECURE_FILE} not found![/]")
            return

        with Live(self.layout, refresh_per_second=4, console=console) as live:
            # Initialize display
            self.layout["header"].update(self.update_header())
            self.layout["left"].update(self.get_log_panel())
            self.layout["right"].update(self.get_code_panel(VULNERABLE_FILE, "Target Code (Vulnerable)"))
            live.refresh()

            time.sleep(1)
            self.add_log("System initialized. Monitoring active.", "green")
            self.update_display(live)
            time.sleep(1)

            # ================================================================
            # PHASE 1: RED TEAM ATTACK
            # ================================================================
            self.status = "INTRUSION DETECTED"
            self.add_log("", "white")
            self.add_log("=" * 40, "red")
            self.add_log("PHASE 1: RED TEAM ATTACK", "bold red")
            self.add_log("=" * 40, "red")
            self.update_display(live)
            time.sleep(0.5)

            self.add_log("ALERT: Suspicious traffic on /vulnerabilities/sqli/", "red")
            self.update_display(live)
            time.sleep(0.5)

            self.add_log("Launching Gemini attack agent...", "yellow")
            self.update_display(live)

            # Run the actual attack
            attack_success = self.run_attack(quiet=True)
            self.update_display(live)

            if attack_success:
                self.status = "CRITICAL: SYSTEM COMPROMISED"
                self.add_log("Data exfiltrated: admin credentials exposed!", "bold red")
            else:
                self.add_log("Attack failed on first attempt.", "yellow")

            self.update_display(live)
            time.sleep(2)

            # ================================================================
            # PHASE 2: AI PATCHING
            # ================================================================
            self.status = "ANALYZING VULNERABILITY"
            self.add_log("", "white")
            self.add_log("=" * 40, "yellow")
            self.add_log("PHASE 2: GEMINI REMEDIATION", "bold yellow")
            self.add_log("=" * 40, "yellow")
            self.update_display(live)
            time.sleep(1)

            self.add_log("Engaging Gemini-4 for code analysis...", "magenta")
            self.update_display(live)
            time.sleep(1)

            self.add_log("Analyzing attack pattern: UNION-based SQLi", "cyan")
            self.update_display(live)
            time.sleep(1)

            self.add_log("Identifying vulnerable code path...", "cyan")
            self.update_display(live)
            time.sleep(1)

            self.status = "GENERATING PATCH"
            self.add_log("Generating secure implementation...", "magenta")
            self.update_display(live)
            time.sleep(1)

            # Show the secure code
            self.layout["right"].update(self.get_code_panel(SECURE_FILE, "Proposed Patch (Secure)"))
            live.refresh()

            self.add_log("PATCH: Using int() for strict type validation", "green")
            self.update_display(live)
            time.sleep(0.5)

            self.add_log("PATCH: Adding escape() for XSS prevention", "green")
            self.update_display(live)
            time.sleep(1)

            # ================================================================
            # PHASE 3: DEPLOY & VERIFY
            # ================================================================
            self.status = "DEPLOYING PATCH"
            self.add_log("", "white")
            self.add_log("=" * 40, "blue")
            self.add_log("PHASE 3: DEPLOY & VERIFY", "bold blue")
            self.add_log("=" * 40, "blue")
            self.update_display(live)
            time.sleep(1)

            # Swap the files
            self.add_log("Hot-swapping vulnerable module...", "yellow")
            self.update_display(live)

            shutil.copy(VULNERABLE_FILE, TEMP_BACKUP)
            shutil.copy(SECURE_FILE, VULNERABLE_FILE)

            self.add_log("Module replaced. Flask reloading...", "yellow")
            self.update_display(live)
            time.sleep(2)  # Give Flask time to reload

            self.status = "VERIFYING SECURITY"
            self.add_log("Re-running attack to verify patch...", "cyan")
            self.update_display(live)

            # Run attack again (should fail now)
            verify_success = self.run_attack(quiet=True)
            self.update_display(live)

            if not verify_success:
                self.status = "SYSTEM SECURED"
                self.add_log("", "white")
                self.add_log("=" * 40, "green")
                self.add_log("THREAT NEUTRALIZED", "bold green")
                self.add_log("System hardened. Zero-day patched.", "bold green")
                self.add_log("=" * 40, "green")
            else:
                self.status = "PATCH FAILED"
                self.add_log("WARNING: Patch verification failed!", "bold red")

            self.update_display(live)
            time.sleep(3)

            # Show summary
            self.add_log("", "white")
            self.add_log("Demo complete. Press Ctrl+C to exit.", "dim")
            self.update_display(live)

            # Wait for user
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

            # Cleanup: restore vulnerable file for next demo
            if os.path.exists(TEMP_BACKUP):
                shutil.move(TEMP_BACKUP, VULNERABLE_FILE)


def main():
    console.print("\n[bold cyan]SkyHammer Demo[/]")
    console.print("[dim]Autonomous AI-Powered Defense System[/]\n")

    # Check if mock_dvwa.py is running
    console.print("[yellow]Prerequisites:[/]")
    console.print("  1. Run 'sudo python mock_dvwa.py' in another terminal")
    console.print("  2. Make sure mock_dvwa_secure.py exists")
    console.print()

    if not os.path.exists(VULNERABLE_FILE):
        console.print(f"[red]Error: {VULNERABLE_FILE} not found![/]")
        return 1

    if not os.path.exists(SECURE_FILE):
        console.print(f"[red]Error: {SECURE_FILE} not found![/]")
        console.print("[yellow]Tip: Run 'python patcher.py' first to generate the secure version.[/]")
        return 1

    console.print("[green]All files found. Starting demo in 3 seconds...[/]")
    time.sleep(3)

    try:
        orchestrator = DemoOrchestrator()
        orchestrator.run_demo()
    except KeyboardInterrupt:
        # Cleanup on interrupt
        if os.path.exists(TEMP_BACKUP):
            shutil.move(TEMP_BACKUP, VULNERABLE_FILE)
        console.print("\n[yellow]Demo interrupted. Files restored.[/]")
    except Exception as e:
        if os.path.exists(TEMP_BACKUP):
            shutil.move(TEMP_BACKUP, VULNERABLE_FILE)
        console.print(f"\n[red]Error: {e}[/]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
