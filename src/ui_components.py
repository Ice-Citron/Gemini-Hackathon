#!/usr/bin/env python3
"""
SkyHammer UI Components
Evangelion/NERV-style banners, diffs, and panels
"""

import difflib
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .theme import C_PRIMARY, C_SECONDARY, C_ACCENT, C_SUCCESS, C_ERROR, C_DIM, C_HIGHLIGHT

# Console instance
console: Optional[Console] = Console() if HAS_RICH else None


def print_banner(model: str):
    """Display the Main System Banner - NERV/Evangelion Style"""
    if not console:
        return

    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" SYSTEM ONLINE // GEMINI-CODE INTERFACE ", justify="center", style="bold black on yellow"),
            style="bold yellow",
            border_style="yellow",
            box=box.HEAVY,
        )
    )
    console.print(grid)
    console.print(f"[{C_DIM}]:: NEURAL LINK ESTABLISHED :: MODEL: {model} ::[/]", justify="center")
    console.print()


def print_skyhammer_banner():
    """Display the SkyHammer Activation Banner - Red Alert Style"""
    if not console:
        return

    console.print()
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" WARNING: SKYHAMMER PROTOCOL ENGAGED ", justify="center", style="bold white on red"),
            subtitle="[ OFFENSIVE SECURITY AUTHORIZED ]",
            style="bold red",
            border_style="bold red",
            box=box.DOUBLE,
            padding=(1, 2)
        )
    )
    console.print(grid)
    console.print(f"[{C_ACCENT}]>> SCANNING MODULES: ACTIVE | EXPLOIT ENGINE: ONLINE | PATCHER: READY <<[/]", justify="center")
    console.print()


def print_mission_complete_banner(created_file: str = None):
    """Display mission complete banner with SkyHammer recommendation"""
    if not console:
        return

    console.print()
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(
        Panel(
            Text(" MISSION COMPLETE ", justify="center", style="bold black on green"),
            style="bold green",
            border_style="green",
            box=box.DOUBLE,
        )
    )
    console.print(grid)

    # Recommend SkyHammer
    console.print()
    console.print(f"[{C_PRIMARY}]RECOMMENDATION:[/]")
    if created_file:
        console.print(f"[{C_SECONDARY}]  File created: {created_file}[/]")
    console.print(f"[{C_SECONDARY}]  To scan for security vulnerabilities, run:[/]")
    console.print(f"[{C_ACCENT}]    /skyhammer[/]")
    console.print(f"[{C_DIM}]  This will analyze the code for SQL injection, XSS, command injection, and more.[/]")
    console.print()


def show_diff(old_content: str, new_content: str, filename: str):
    """
    Evangelion-style split-pane diff view.
    Shows side-by-side comparison with NERV UI aesthetics.
    """
    if not console:
        return

    console.print(f"\n[{C_PRIMARY}]:: DETECTED FILE MODIFICATION :: {filename}[/]")

    if old_content == new_content:
        console.print(f"[{C_DIM}]>> No logical changes detected.[/]")
        return

    # Determine syntax language
    ext = filename.split('.')[-1] if '.' in filename else 'python'
    lang_map = {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'bash', 'md': 'markdown', 'rb': 'ruby', 'php': 'php'}
    lang = lang_map.get(ext, ext)

    # Create split-pane layout table
    layout = Table(show_header=True, header_style="bold white", box=box.ROUNDED, expand=True, border_style="yellow")
    layout.add_column(f"[{C_ERROR}] ORIGINAL[/] // {filename}", style="dim red", ratio=1)
    layout.add_column(f"[{C_SUCCESS}]PROPOSED PATCH [/] // {filename}", style="dim green", ratio=1)

    # Render Syntax blocks side by side
    try:
        syntax_old = Syntax(old_content, lang, theme="monokai", line_numbers=True, word_wrap=True)
        syntax_new = Syntax(new_content, lang, theme="monokai", line_numbers=True, word_wrap=True)
        layout.add_row(syntax_old, syntax_new)
    except Exception:
        layout.add_row(old_content, new_content)

    console.print(layout)

    # Unified Diff (The "Hacker" View) - Delta Analysis
    console.print(f"\n[{C_SECONDARY}]:: DELTA ANALYSIS ::[/]")

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    diff_lines = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))

    if diff_lines:
        diff_table = Table(box=box.SIMPLE, show_header=False, show_edge=False, border_style="dim blue")
        diff_table.add_column("Line")

        for line in diff_lines[2:]:  # Skip headers
            if line.startswith('@@'):
                diff_table.add_row(Text(line, style="bold blue"))
            elif line.startswith('+'):
                diff_table.add_row(Text(line, style=f"bold green on black"))
            elif line.startswith('-'):
                diff_table.add_row(Text(line, style=f"strike dim red"))
            # Skip context lines to keep UI clean

        console.print(Panel(diff_table, border_style="dim blue", title="[Change Manifest]", title_align="left"))
    console.print()


def show_new_file_preview(content: str, filename: str):
    """Show new file with Evangelion-style green panel - FULL content"""
    if not console:
        return

    console.print(f"\n[{C_PRIMARY}]:: NEW FILE CREATION :: {filename}[/]")

    # Determine language for syntax highlighting
    ext = filename.split('.')[-1] if '.' in filename else 'python'
    lang_map = {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'bash', 'md': 'markdown', 'rb': 'ruby', 'php': 'php'}
    lang = lang_map.get(ext, ext)

    try:
        syntax = Syntax(content, lang, line_numbers=True, word_wrap=True, theme="monokai")
        console.print(Panel(
            syntax,
            title=f"[{C_SUCCESS}]+ NEW FILE: {filename}[/]",
            border_style="green",
            box=box.ROUNDED
        ))
    except Exception:
        # Fallback to plain text
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            console.print(f"[{C_SUCCESS}]+{i:3}| {line}[/]")
    console.print()


def show_report_diff(before_code: str, after_code: str, vuln_type: str = ""):
    """
    Show BEFORE/AFTER code comparison with red/green highlighting for mission reports.
    This is specifically for the final vulnerability report.
    """
    if not console:
        return

    console.print(f"\n[{C_PRIMARY}]:: CODE COMPARISON ::[/]")
    if vuln_type:
        console.print(f"[{C_DIM}]Vulnerability: {vuln_type}[/]")

    # BEFORE section - RED
    console.print(f"\n[{C_ERROR}]BEFORE (vulnerable):[/]")
    for line in before_code.strip().splitlines():
        console.print(f"[red]  {line}[/red]")

    # AFTER section - GREEN
    console.print(f"\n[{C_SUCCESS}]AFTER (secure):[/]")
    for line in after_code.strip().splitlines():
        console.print(f"[green]  {line}[/green]")

    console.print()


def show_goals(tasks: list, status: str):
    """Display current goals/tasks - NERV Mission Status"""
    if not console:
        return

    if not tasks and status == "idle":
        console.print(f"[{C_DIM}]>> NO ACTIVE MISSIONS. Awaiting orders. <<[/]")
        return

    status_colors = {
        "idle": C_DIM,
        "thinking": C_PRIMARY,
        "executing": C_SECONDARY,
        "done": C_SUCCESS
    }
    color = status_colors.get(status, "white")

    console.print(f"\n[{C_PRIMARY}]:: MISSION STATUS ::[/] [{color}]{status.upper()}[/{color}]")

    if tasks:
        console.print(f"[{C_PRIMARY}]ACTIVE OBJECTIVES:[/]")
        for i, task in enumerate(tasks, 1):
            console.print(f"  [{C_SECONDARY}]{i}.[/] {task}")
    console.print()


def get_console() -> Optional[Console]:
    """Get the rich console instance"""
    return console
