#!/usr/bin/env python3
"""
SkyHammer CI/CD Security Scanner

A GitGuardian-style security scanner that uses Gemini to detect vulnerabilities
in source code. Designed for CI/CD pipeline integration.

Features:
- Scans Python files for security vulnerabilities
- Uses Gemini-4 for intelligent code analysis
- Outputs structured JSON reports
- Exit codes for CI/CD integration (0=pass, 1=fail)
- Auto-fix suggestions

Usage:
    python ci_scan.py                     # Scan current directory
    python ci_scan.py --path ./src        # Scan specific directory
    python ci_scan.py --fix               # Generate fix suggestions
    python ci_scan.py --json              # Output JSON report
    python ci_scan.py --severity high     # Only report high severity

GitHub Actions Integration:
    - name: Security Scan
      run: python ci_scan.py --json > security_report.json
      continue-on-error: false
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from openai import AsyncOpenAI
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install openai rich")
    sys.exit(1)

# Load secrets
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

console = Console()

# Configuration
MODEL = "gemini-4-1-fast-reasoning"
MAX_FILE_SIZE = 50000  # 50KB max per file
SCAN_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".php", ".rb", ".go", ".java"}


@dataclass
class Vulnerability:
    """Represents a detected vulnerability"""
    file: str
    line: Optional[int]
    severity: str  # low, medium, high, critical
    category: str
    description: str
    code_snippet: str
    fix_suggestion: Optional[str] = None
    cwe_id: Optional[str] = None


@dataclass
class ScanResult:
    """Result of a security scan"""
    scan_time: str
    files_scanned: int
    vulnerabilities_found: int
    vulnerabilities: List[Vulnerability]
    passed: bool


class SecurityScanner:
    """Gemini-powered security scanner"""

    def __init__(self, generate_fixes: bool = False):
        self.client = AsyncOpenAI(
            api_key=GDM_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        self.generate_fixes = generate_fixes
        self.vulnerabilities: List[Vulnerability] = []

    def find_files(self, path: Path) -> List[Path]:
        """Find all scannable files in path"""
        files = []

        if path.is_file():
            if path.suffix in SCAN_EXTENSIONS:
                files.append(path)
        else:
            for ext in SCAN_EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))

        # Filter by size
        files = [f for f in files if f.stat().st_size < MAX_FILE_SIZE]

        # Exclude common non-source directories
        exclude_dirs = {"node_modules", "venv", ".venv", "__pycache__", ".git", "dist", "build"}
        files = [f for f in files if not any(ex in f.parts for ex in exclude_dirs)]

        return files

    async def scan_file(self, file_path: Path) -> List[Vulnerability]:
        """Scan a single file for vulnerabilities"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/]")
            return []

        if len(code) < 10:
            return []

        fix_instruction = ""
        if self.generate_fixes:
            fix_instruction = """
For each vulnerability, also provide a "fix_suggestion" with the corrected code."""

        prompt = f"""Analyze this source code for security vulnerabilities.

FILE: {file_path.name}
LANGUAGE: {file_path.suffix}

```
{code[:8000]}
```

Return a JSON object with this exact structure:
{{
    "vulnerabilities": [
        {{
            "line": <line number or null>,
            "severity": "<low|medium|high|critical>",
            "category": "<SQL Injection|XSS|Command Injection|Path Traversal|Hardcoded Secret|Insecure Deserialization|SSRF|XXE|Other>",
            "description": "<brief description>",
            "code_snippet": "<the vulnerable code>",
            "cwe_id": "<CWE-XXX or null>",
            "fix_suggestion": "<fixed code or null>"
        }}
    ]
}}

Focus on:
1. SQL Injection (string concatenation in queries)
2. Command Injection (os.system, subprocess with user input)
3. XSS (unescaped output)
4. Path Traversal (user input in file paths)
5. Hardcoded Secrets (API keys, passwords)
6. Insecure Deserialization (pickle, yaml.load)
7. SSRF (user-controlled URLs)

If no vulnerabilities found, return: {{"vulnerabilities": []}}
{fix_instruction}"""

        try:
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a security code auditor. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            vulns = []

            for v in result.get("vulnerabilities", []):
                vuln = Vulnerability(
                    file=str(file_path),
                    line=v.get("line"),
                    severity=v.get("severity", "medium"),
                    category=v.get("category", "Other"),
                    description=v.get("description", ""),
                    code_snippet=v.get("code_snippet", "")[:200],
                    fix_suggestion=v.get("fix_suggestion") if self.generate_fixes else None,
                    cwe_id=v.get("cwe_id")
                )
                vulns.append(vuln)

            return vulns

        except Exception as e:
            console.print(f"[red]Error scanning {file_path}: {e}[/]")
            return []

    async def scan_directory(self, path: Path, min_severity: str = "low") -> ScanResult:
        """Scan all files in a directory"""
        files = self.find_files(path)

        if not files:
            console.print(f"[yellow]No scannable files found in {path}[/]")
            return ScanResult(
                scan_time=datetime.now().isoformat(),
                files_scanned=0,
                vulnerabilities_found=0,
                vulnerabilities=[],
                passed=True
            )

        console.print(f"[cyan]Scanning {len(files)} files...[/]")

        all_vulns = []
        for i, file_path in enumerate(files):
            console.print(f"  [{i+1}/{len(files)}] {file_path.name}...", end="")
            vulns = await self.scan_file(file_path)
            if vulns:
                console.print(f" [red]{len(vulns)} issues[/]")
                all_vulns.extend(vulns)
            else:
                console.print(" [green]OK[/]")

        # Filter by severity
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_order.get(min_severity, 0)
        filtered_vulns = [v for v in all_vulns if severity_order.get(v.severity, 0) >= min_level]

        # Determine pass/fail
        # Fail if any high or critical vulnerabilities
        has_critical = any(v.severity in ["high", "critical"] for v in filtered_vulns)

        return ScanResult(
            scan_time=datetime.now().isoformat(),
            files_scanned=len(files),
            vulnerabilities_found=len(filtered_vulns),
            vulnerabilities=filtered_vulns,
            passed=not has_critical
        )


def print_results(result: ScanResult, json_output: bool = False):
    """Print scan results"""
    if json_output:
        output = {
            "scan_time": result.scan_time,
            "files_scanned": result.files_scanned,
            "vulnerabilities_found": result.vulnerabilities_found,
            "passed": result.passed,
            "vulnerabilities": [asdict(v) for v in result.vulnerabilities]
        }
        print(json.dumps(output, indent=2))
        return

    console.print("\n")
    console.print(Panel("[bold]SkyHammer Security Scan Results[/]", style="cyan"))

    # Summary
    status = "[bold green]PASSED[/]" if result.passed else "[bold red]FAILED[/]"
    console.print(f"\nStatus: {status}")
    console.print(f"Files Scanned: {result.files_scanned}")
    console.print(f"Vulnerabilities Found: {result.vulnerabilities_found}")

    if not result.vulnerabilities:
        console.print("\n[green]No vulnerabilities detected.[/]")
        return

    # Detailed table
    table = Table(title="Vulnerabilities", box=box.ROUNDED)
    table.add_column("File", style="cyan", max_width=30)
    table.add_column("Line", justify="right", width=6)
    table.add_column("Severity", justify="center", width=10)
    table.add_column("Category", width=20)
    table.add_column("Description", max_width=40)

    severity_styles = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "blue"
    }

    for v in result.vulnerabilities:
        severity_display = f"[{severity_styles.get(v.severity, 'white')}]{v.severity.upper()}[/]"
        table.add_row(
            Path(v.file).name,
            str(v.line) if v.line else "-",
            severity_display,
            v.category,
            v.description[:40]
        )

    console.print(table)

    # Show fix suggestions if available
    fixes = [v for v in result.vulnerabilities if v.fix_suggestion]
    if fixes:
        console.print("\n[bold]Fix Suggestions:[/]")
        for v in fixes[:5]:  # Show max 5 fixes
            console.print(f"\n[cyan]{Path(v.file).name}:{v.line or '?'}[/] - {v.category}")
            console.print(f"[red]Before:[/] {v.code_snippet}")
            console.print(f"[green]After:[/] {v.fix_suggestion}")


async def main():
    parser = argparse.ArgumentParser(
        description="SkyHammer CI/CD Security Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ci_scan.py                     # Scan current directory
  python ci_scan.py --path ./src        # Scan specific path
  python ci_scan.py --fix               # Include fix suggestions
  python ci_scan.py --json              # JSON output for CI
  python ci_scan.py --severity high     # Only high/critical
        """
    )

    parser.add_argument(
        "--path", "-p",
        type=str,
        default=".",
        help="Path to scan (file or directory)"
    )
    parser.add_argument(
        "--fix", "-f",
        action="store_true",
        help="Generate fix suggestions"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON format"
    )
    parser.add_argument(
        "--severity", "-s",
        type=str,
        default="low",
        choices=["low", "medium", "high", "critical"],
        help="Minimum severity to report"
    )

    args = parser.parse_args()

    if not GDM_API_KEY:
        console.print("[red]Error: GDM_API_KEY not set[/]")
        console.print("Add it to secretsConfig.py or set as environment variable")
        sys.exit(1)

    if not args.json:
        console.print("\n[bold cyan]SkyHammer CI/CD Scanner[/]")
        console.print("[dim]Powered by Gemini-4[/]\n")

    scanner = SecurityScanner(generate_fixes=args.fix)
    path = Path(args.path)

    if not path.exists():
        console.print(f"[red]Error: Path not found: {path}[/]")
        sys.exit(1)

    result = await scanner.scan_directory(path, min_severity=args.severity)

    print_results(result, json_output=args.json)

    # Exit code for CI/CD
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
