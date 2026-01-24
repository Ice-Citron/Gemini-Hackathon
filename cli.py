#!/usr/bin/env python3
"""
SkyHammer Command Center v3.0

Interactive CLI for autonomous security testing and remediation.
Features:
- Attack any URL
- Generate synthetic vulnerable apps
- Auto-patch with Gemini
- Auto-dockerize repos
- Full demo mode

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
    from rich.text import Text
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install questionary rich")
    sys.exit(1)

console = Console()

# == GLOBAL STATE ==
STATUS = "IDLE"
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
    elif "PATCH" in STATUS:
        style = "bold yellow"
        icon = "🛡️"
    elif "SECURE" in STATUS or "BLOCKED" in STATUS:
        style = "bold green"
        icon = "🟢"
    elif "GENERATE" in STATUS or "DOCKER" in STATUS:
        style = "bold magenta"
        icon = "🧪"
    else:
        style = "bold cyan"
        icon = "🔵"

    grid.add_row(
        Text("SkyHammer", style="bold cyan"),
        Text("Command Center v3.0", style="dim"),
        Text(f"{icon} {STATUS}", style=style)
    )
    return Panel(grid, style="white on black", box=box.DOUBLE)


def run_process_with_logs(cmd, title="Running..."):
    """Stream subprocess output with live updates"""
    logs = []
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1)
    )

    process = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
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
                        logs.append(f"[bold red]>>> EXPLOITED: {clean} <<<[/]")
                    elif "Success: False" in clean:
                        logs.append(f"[bold green]>>> BLOCKED: {clean} <<<[/]")
                    elif "Error" in clean or "error" in clean:
                        logs.append(f"[red]{clean}[/]")
                    elif "PATCH" in clean or "[+]" in clean:
                        logs.append(f"[green]{clean}[/]")
                    elif "Generating" in clean or "Generated" in clean:
                        logs.append(f"[magenta]{clean}[/]")
                    else:
                        logs.append(clean)

                    if len(logs) > 25:
                        logs.pop(0)

            log_content = "\n".join(logs) if logs else "[dim]Waiting...[/]"
            layout["body"].update(Panel(log_content, title=f"[bold]{title}[/]", border_style="cyan", box=box.ROUNDED))

    return process.poll()


def attack_url():
    """Attack any URL target"""
    global STATUS
    STATUS = "ATTACKING EXTERNAL TARGET"

    console.print("\n[bold red]Attack External Target[/]")
    console.print("[dim]Attack any web application by URL[/]\n")

    url = questionary.text(
        "Enter target URL:",
        default="http://127.0.0.1:80",
        style=questionary.Style([('answer', 'fg:red bold')])
    ).ask()

    if not url:
        STATUS = "IDLE"
        return

    vuln = questionary.select(
        "Select attack vector:",
        choices=[
            "sqli - SQL Injection",
            "xss - Cross-Site Scripting",
            "cmd - Command Injection",
            "lfi - Local File Inclusion"
        ],
        style=questionary.Style([
            ('selected', 'fg:red bold'),
            ('pointer', 'fg:red bold'),
        ])
    ).ask()

    if not vuln:
        STATUS = "IDLE"
        return

    challenge = vuln.split(" - ")[0]

    console.print(f"\n[red]Target: {url}[/]")
    console.print(f"[red]Vector: {challenge}[/]\n")

    cmd = f'python attack.py --challenge {challenge} --dvwa "{url}"'
    run_process_with_logs(cmd, f"Attacking {url}")

    STATUS = "ATTACK COMPLETE"


def attack_synthetic():
    """Generate and attack a synthetic app"""
    global STATUS, SYNTHETIC_SERVER
    STATUS = "GENERATING CHALLENGE"

    console.print("\n[bold magenta]Synthetic Challenge[/]")
    console.print("[dim]Generate a new vulnerable app and attack it[/]\n")

    # Run generator interactively
    subprocess.run("python generator.py", shell=True)

    # Find the latest synthetic file
    synthetic_files = sorted([f for f in os.listdir(".") if f.startswith("synthetic_") and f.endswith(".py")])

    if not synthetic_files:
        console.print("[red]No synthetic app generated![/]")
        STATUS = "IDLE"
        return

    latest_file = synthetic_files[-1]

    # Ask if user wants to attack it
    attack_now = questionary.confirm(
        f"Attack {latest_file} now?",
        default=True
    ).ask()

    if not attack_now:
        STATUS = "IDLE"
        return

    STATUS = "ATTACKING SYNTHETIC"

    # Determine challenge type from filename
    if "sql" in latest_file.lower():
        challenge = "sqli"
    elif "command" in latest_file.lower():
        challenge = "cmd"
    elif "xss" in latest_file.lower():
        challenge = "xss"
    elif "lfi" in latest_file.lower() or "file" in latest_file.lower():
        challenge = "lfi"
    else:
        challenge = "sqli"

    console.print(f"\n[yellow]Starting {latest_file} on port 5001...[/]")

    # Start the synthetic server
    SYNTHETIC_SERVER = subprocess.Popen(
        ["python", latest_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)  # Wait for Flask to start

    # Attack it
    cmd = f"python attack.py --challenge {challenge} --dvwa http://127.0.0.1:5001"
    run_process_with_logs(cmd, f"Attacking {latest_file}")

    # Cleanup
    if SYNTHETIC_SERVER:
        SYNTHETIC_SERVER.terminate()
        SYNTHETIC_SERVER = None
        console.print("[yellow]Synthetic server stopped.[/]")

    STATUS = "ATTACK COMPLETE"


def run_patcher():
    """Patch a vulnerable file"""
    global STATUS
    STATUS = "AI PATCHING"

    console.print("\n[bold yellow]AI Patcher[/]")
    console.print("[dim]Fix vulnerabilities with Gemini[/]\n")

    # Find patchable files
    files = ["mock_dvwa.py"]
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

    cmd = f'python patcher.py --source "{target}"'
    run_process_with_logs(cmd, f"Patching {target}")

    STATUS = "PATCH COMPLETE"


def run_deploy():
    """Deploy a patched version"""
    global STATUS
    STATUS = "DEPLOYING"

    console.print("\n[bold green]Deploy Patch[/]")
    console.print("[dim]Hot-swap vulnerable code with secure version[/]\n")

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
        backup = f"{original}.bak"
        subprocess.run(f'cp "{original}" "{backup}"', shell=True)
        console.print(f"[dim]Backup: {backup}[/]")

        subprocess.run(f'cp "{secure}" "{original}"', shell=True)
        console.print(f"[green]Deployed! {original} is now secure.[/]")

        STATUS = "SYSTEM SECURED"
    else:
        STATUS = "IDLE"


def run_dockerizer():
    """Auto-generate Docker configs"""
    global STATUS
    STATUS = "DOCKERIZING"

    console.print("\n[bold blue]Auto-Dockerizer[/]")
    console.print("[dim]Generate Docker configs with Gemini[/]\n")

    subprocess.run("python dockerizer.py", shell=True)

    STATUS = "DOCKER COMPLETE"


def run_reset():
    """Reset to vulnerable state"""
    global STATUS

    console.print("\n[bold yellow]Reset Environment[/]")

    backups = [f for f in os.listdir(".") if f.endswith(".bak")]

    if not backups:
        console.print("[yellow]No backups found.[/]")
        return

    for bak in backups:
        original = bak.replace(".bak", "")
        subprocess.run(f'mv "{bak}" "{original}"', shell=True)
        console.print(f"[yellow]Restored: {original}[/]")

    STATUS = "VULNERABLE (RESET)"


def run_auto_demo():
    """Run the automated demo"""
    global STATUS

    console.print("\n[bold cyan]Automated Demo[/]")
    console.print("[dim]Full attack -> patch -> verify loop[/]\n")

    confirmed = questionary.confirm(
        "Run automated demo? (Requires mock_dvwa.py running on port 80)",
        default=True
    ).ask()

    if confirmed:
        os.system("python ui.py")


def show_status():
    """Show system status"""
    console.print("\n")
    console.print(render_header())

    table = Table(title="System Status", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Check files
    table.add_row("mock_dvwa.py", "✓ Found" if os.path.exists("mock_dvwa.py") else "✗ Missing")
    table.add_row("mock_dvwa_secure.py", "✓ Found" if os.path.exists("mock_dvwa_secure.py") else "○ Not generated")

    # Count synthetic apps
    synthetic = [f for f in os.listdir(".") if f.startswith("synthetic_") and f.endswith(".py") and "_secure" not in f]
    table.add_row("Synthetic Apps", f"{len(synthetic)} generated")

    # Check patched versions
    patched = [f for f in os.listdir(".") if "_secure.py" in f]
    table.add_row("Patched Versions", f"{len(patched)} available")

    # Check Docker
    table.add_row("Dockerfile", "✓ Found" if os.path.exists("Dockerfile") else "○ Not generated")

    # Check API
    table.add_row("Gemini API", "✓ Configured" if os.path.exists("secretsConfig.py") else "✗ Missing")

    console.print(table)


def run_gemini_code():
    """Launch Gemini Code interactive mode"""
    global STATUS
    STATUS = "GEMINI CODE MODE"
    os.system("python gemini_code.py")
    STATUS = "IDLE"


def main_menu():
    """Main interactive menu"""
    global STATUS

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(render_header())
        console.print()

        choice = questionary.select(
            "Select operation:",
            choices=[
                "1. 💬 Gemini Code + SkyHammer: Interactive AI Assistant",
                "2. ⚔️  Attack: Target URL (Paste Link)",
                "3. 🧪 Attack: Synthetic Challenge (Generate & Attack)",
                "4. 🛡️  Defense: AI Patcher (Fix Code)",
                "5. 🚀 Deploy: Hot-Swap Patch",
                "6. 🐳 DevOps: Auto-Dockerize",
                "7. ♻️  Reset: Restore Vulnerable State",
                "8. 🎬 Demo: Run Automated Loop",
                "9. 📊 Status: Show System Info",
                "0. 🚪 Exit"
            ],
            style=questionary.Style([
                ('selected', 'fg:cyan bold'),
                ('pointer', 'fg:cyan bold'),
            ])
        ).ask()

        if not choice:
            continue

        if "1." in choice:
            run_gemini_code()
        elif "2." in choice:
            attack_url()
        elif "3." in choice:
            attack_synthetic()
        elif "4." in choice:
            run_patcher()
        elif "5." in choice:
            run_deploy()
        elif "6." in choice:
            run_dockerizer()
        elif "7." in choice:
            run_reset()
        elif "8." in choice:
            run_auto_demo()
        elif "9." in choice:
            show_status()
        elif "0." in choice:
            console.print("\n[cyan]Goodbye![/]")
            sys.exit(0)

        input("\n[Press Enter to continue...]")


def main():
    """Entry point"""
    console.print("\n[bold cyan]SkyHammer Command Center[/]")
    console.print("[dim]Autonomous AI-Powered Security System[/]")
    console.print("[dim]Powered by Gemini-4[/]\n")

    try:
        main_menu()
    except KeyboardInterrupt:
        if SYNTHETIC_SERVER:
            SYNTHETIC_SERVER.terminate()
        console.print("\n[yellow]Interrupted. Exiting...[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
