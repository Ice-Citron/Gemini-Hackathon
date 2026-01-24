#!/usr/bin/env python3
"""
AthenaGuard Command Center v2.2

Interactive CLI for autonomous security testing and remediation.
Features arrow-key menus, synthetic app generation, and live attack visualization.

Usage:
    python cli.py

Requirements:
    pip install questionary rich
"""

import sys
import os
import time
import subprocess
import signal

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.table import Table
    from rich import box
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install questionary rich")
    sys.exit(1)

console = Console()

# == GLOBAL STATE ==
TARGET_HOST = "http://127.0.0.1:80"
TARGET_FILE = "mock_dvwa.py"
SECURE_FILE = "mock_dvwa_secure.py"
STATUS = "IDLE"
LAST_ATTACK_RESULT = None
SYNTHETIC_SERVER = None


def render_header():
    """Render the header panel with current status"""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="right")

    # Status color logic
    if "ATTACK" in STATUS or "COMPROMISED" in STATUS:
        style = "bold red"
        icon = "🔴"
    elif "PATCH" in STATUS or "GENERATING" in STATUS:
        style = "bold yellow"
        icon = "🟡"
    elif "SECURE" in STATUS or "BLOCKED" in STATUS:
        style = "bold green"
        icon = "🟢"
    elif "GENERATE" in STATUS:
        style = "bold magenta"
        icon = "🧪"
    else:
        style = "bold cyan"
        icon = "🔵"

    grid.add_row(
        Text("AthenaGuard", style="bold cyan"),
        Text("Command Center v2.2", style="dim"),
        Text(f"{icon} {STATUS}", style=style)
    )
    return Panel(grid, style="white on black", box=box.DOUBLE)


def show_logs(process, title="Running Process"):
    """Stream subprocess output with live updates"""
    logs = []
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1)
    )

    with Live(layout, refresh_per_second=8, console=console):
        while True:
            layout["header"].update(render_header())

            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                clean = line.strip()
                if clean:
                    # Color code different log types
                    if "Tool:" in clean:
                        logs.append(f"[blue]{clean[:80]}[/]")
                    elif "Success: True" in clean:
                        logs.append(f"[bold red]>>> {clean} <<<[/]")
                    elif "Success: False" in clean:
                        logs.append(f"[bold green]>>> {clean} <<<[/]")
                    elif "Error" in clean or "error" in clean:
                        logs.append(f"[red]{clean}[/]")
                    elif "PATCH" in clean or "Generating" in clean:
                        logs.append(f"[magenta]{clean}[/]")
                    else:
                        logs.append(clean)

                    if len(logs) > 25:
                        logs.pop(0)

            log_content = "\n".join(logs) if logs else "[dim]Waiting...[/]"
            layout["body"].update(Panel(log_content, title=f"[bold]{title}[/]", border_style="cyan", box=box.ROUNDED))

    return process.poll()


def run_generator():
    """Generate a synthetic vulnerable application"""
    global STATUS
    STATUS = "GENERATING APP"

    console.print("\n[bold magenta]Synthetic Vulnerability Generator[/]")
    console.print("[dim]Creates a new vulnerable Flask app using Gemini[/]\n")

    vuln = questionary.select(
        "Select vulnerability type to generate:",
        choices=[
            "SQL Injection",
            "Command Injection",
            "Reflected XSS",
            "Path Traversal"
        ],
        style=questionary.Style([
            ('selected', 'fg:cyan bold'),
            ('pointer', 'fg:cyan bold'),
        ])
    ).ask()

    if not vuln:
        STATUS = "IDLE"
        return

    console.print(f"\n[cyan]Generating {vuln} challenge...[/]")

    cmd = f'python generator.py "{vuln}"'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    show_logs(process, title=f"Gemini Synthetic Generator ({vuln})")

    STATUS = "APP GENERATED"
    console.print("[green]Synthetic app created! Select it in Attack menu.[/]")


def run_attack_module():
    """Run the Gemini attack agent"""
    global STATUS, LAST_ATTACK_RESULT, SYNTHETIC_SERVER
    STATUS = "RED TEAM ATTACK"

    console.print("\n[bold red]Red Team Attack Module[/]")
    console.print("[dim]Gemini-powered autonomous exploitation[/]\n")

    # Find available targets
    targets = []
    if os.path.exists("mock_dvwa.py"):
        targets.append("mock_dvwa.py (Port 80 - requires sudo)")

    # Find synthetic apps
    for f in os.listdir("."):
        if f.startswith("synthetic_") and f.endswith(".py"):
            targets.append(f"{f} (Port 5001 - auto-start)")

    if not targets:
        console.print("[red]No target applications found![/]")
        STATUS = "IDLE"
        return

    target_choice = questionary.select(
        "Select target application:",
        choices=targets,
        style=questionary.Style([
            ('selected', 'fg:red bold'),
            ('pointer', 'fg:red bold'),
        ])
    ).ask()

    if not target_choice:
        STATUS = "IDLE"
        return

    target_app = target_choice.split(" ")[0]

    # Determine port and challenge type
    if "synthetic" in target_app:
        port = "5001"
        # Guess challenge from filename
        if "sql" in target_app.lower():
            challenge = "sqli"
        elif "command" in target_app.lower():
            challenge = "cmd"
        elif "xss" in target_app.lower():
            challenge = "xss"
        else:
            challenge = "sqli"

        # Start synthetic server
        console.print(f"[yellow]Starting {target_app} on port {port}...[/]")
        SYNTHETIC_SERVER = subprocess.Popen(
            ["python", target_app],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2)  # Wait for Flask
    else:
        port = "80"
        # Let user pick challenge for mock_dvwa
        challenge = questionary.select(
            "Select attack vector:",
            choices=["sqli (SQL Injection)", "xss (Cross-Site Scripting)", "cmd (Command Injection)"]
        ).ask()
        challenge = challenge.split(" ")[0] if challenge else "sqli"

    target_url = f"http://127.0.0.1:{port}"

    console.print(f"\n[red]Targeting: {target_url}[/]")
    console.print(f"[red]Challenge: {challenge}[/]\n")

    # Run attack
    cmd = f"python attack.py --challenge {challenge} --dvwa {target_url}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    exit_code = show_logs(process, title=f"Attacking {target_app}")

    # Cleanup synthetic server
    if SYNTHETIC_SERVER:
        SYNTHETIC_SERVER.terminate()
        SYNTHETIC_SERVER = None
        console.print("[yellow]Synthetic server stopped.[/]")

    # Determine result
    STATUS = "ATTACK COMPLETE"


def run_patch_module():
    """Run the Gemini patcher"""
    global STATUS
    STATUS = "AI PATCHING"

    console.print("\n[bold yellow]AI Patcher Module[/]")
    console.print("[dim]Gemini-powered vulnerability remediation[/]\n")

    # Find patchable files
    files = []
    if os.path.exists("mock_dvwa.py"):
        files.append("mock_dvwa.py")
    for f in os.listdir("."):
        if f.startswith("synthetic_") and f.endswith(".py") and "_secure" not in f:
            files.append(f)

    if not files:
        console.print("[red]No files to patch![/]")
        STATUS = "IDLE"
        return

    target = questionary.select(
        "Select file to patch:",
        choices=files,
        style=questionary.Style([
            ('selected', 'fg:yellow bold'),
            ('pointer', 'fg:yellow bold'),
        ])
    ).ask()

    if not target:
        STATUS = "IDLE"
        return

    console.print(f"\n[yellow]Analyzing {target} with Gemini...[/]\n")

    cmd = f'python patcher.py --source "{target}"'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    show_logs(process, title=f"Patching {target}")

    STATUS = "PATCH READY"


def run_deploy():
    """Deploy the secure version"""
    global STATUS
    STATUS = "DEPLOYING"

    console.print("\n[bold green]Deploy Module[/]")
    console.print("[dim]Hot-swap vulnerable code with patched version[/]\n")

    # Find secure versions
    secure_files = []
    for f in os.listdir("."):
        if "_secure.py" in f:
            original = f.replace("_secure.py", ".py")
            if os.path.exists(original):
                secure_files.append(f"{original} -> {f}")

    if not secure_files:
        console.print("[red]No patched files found! Run Patcher first.[/]")
        STATUS = "IDLE"
        return

    choice = questionary.select(
        "Select deployment:",
        choices=secure_files + ["Cancel"],
        style=questionary.Style([
            ('selected', 'fg:green bold'),
            ('pointer', 'fg:green bold'),
        ])
    ).ask()

    if not choice or choice == "Cancel":
        STATUS = "IDLE"
        return

    original = choice.split(" -> ")[0]
    secure = choice.split(" -> ")[1]

    confirmed = questionary.confirm(
        f"Hot-swap {original} with {secure}?",
        default=True
    ).ask()

    if confirmed:
        # Backup
        backup = f"{original}.bak"
        subprocess.run(f'cp "{original}" "{backup}"', shell=True)
        console.print(f"[dim]Backup saved: {backup}[/]")

        # Swap
        subprocess.run(f'cp "{secure}" "{original}"', shell=True)
        console.print(f"[green]Deployed! {original} is now secure.[/]")

        STATUS = "SYSTEM SECURED"
    else:
        STATUS = "IDLE"


def run_reset():
    """Reset to vulnerable state"""
    global STATUS

    console.print("\n[bold yellow]Reset Module[/]")
    console.print("[dim]Restore vulnerable versions for testing[/]\n")

    # Find backups
    backups = [f for f in os.listdir(".") if f.endswith(".bak")]

    if not backups:
        console.print("[yellow]No backups found.[/]")
        return

    for bak in backups:
        original = bak.replace(".bak", "")
        subprocess.run(f'mv "{bak}" "{original}"', shell=True)
        console.print(f"[yellow]Restored: {original}[/]")

    STATUS = "VULNERABLE (RESET)"
    console.print("\n[red]System restored to VULNERABLE state.[/]")


def run_auto_demo():
    """Run the full automated demo"""
    global STATUS

    console.print("\n[bold cyan]Automated Demo Mode[/]")
    console.print("[dim]Runs the full attack -> patch -> verify loop[/]\n")

    confirmed = questionary.confirm(
        "Run full automated demo? (Make sure mock_dvwa.py is running on port 80)",
        default=True
    ).ask()

    if not confirmed:
        return

    # Run the ui.py demo
    os.system("python ui.py")


def show_status():
    """Show current system status"""
    console.print("\n")
    console.print(render_header())

    table = Table(title="System Status", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Check files
    table.add_row("mock_dvwa.py", "✓ Found" if os.path.exists("mock_dvwa.py") else "✗ Missing")
    table.add_row("mock_dvwa_secure.py", "✓ Found" if os.path.exists("mock_dvwa_secure.py") else "✗ Missing")

    # Count synthetic apps
    synthetic = [f for f in os.listdir(".") if f.startswith("synthetic_") and f.endswith(".py")]
    table.add_row("Synthetic Apps", f"{len(synthetic)} generated")

    # Check dependencies
    table.add_row("Gemini API", "✓ Configured" if os.path.exists("secretsConfig.py") else "✗ Missing")

    console.print(table)


def main_menu():
    """Main interactive menu"""
    global STATUS

    while True:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Show header
        console.print(render_header())
        console.print()

        choice = questionary.select(
            "Select operation:",
            choices=[
                "1. 🧪 Generate Synthetic Challenge",
                "2. 🔴 Run Red Team Attack",
                "3. 🔵 Run AI Patcher (Fix Vulnerabilities)",
                "4. 🚀 Deploy Patch (Hot-Swap)",
                "5. ♻️  Reset Environment",
                "6. 🎬 Run Automated Demo",
                "7. 📊 Show Status",
                "8. 🚪 Exit"
            ],
            style=questionary.Style([
                ('selected', 'fg:cyan bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan'),
            ])
        ).ask()

        if not choice:
            continue

        if "1." in choice:
            run_generator()
        elif "2." in choice:
            run_attack_module()
        elif "3." in choice:
            run_patch_module()
        elif "4." in choice:
            run_deploy()
        elif "5." in choice:
            run_reset()
        elif "6." in choice:
            run_auto_demo()
        elif "7." in choice:
            show_status()
        elif "8." in choice:
            console.print("\n[cyan]Goodbye![/]")
            sys.exit(0)

        input("\n[Press Enter to continue...]")


def main():
    """Entry point"""
    console.print("\n[bold cyan]AthenaGuard Command Center[/]")
    console.print("[dim]Autonomous AI-Powered Security System[/]")
    console.print("[dim]Powered by Gemini-4[/]\n")

    try:
        main_menu()
    except KeyboardInterrupt:
        # Cleanup
        if SYNTHETIC_SERVER:
            SYNTHETIC_SERVER.terminate()
        console.print("\n[yellow]Interrupted. Cleaning up...[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
