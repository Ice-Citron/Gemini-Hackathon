#!/usr/bin/env python3
"""
SkyHammer Synthetic Vulnerability Generator v2.0

Uses Gemini to generate new vulnerable Flask applications on-the-fly.
Supports: SQL Injection, Command Injection, XSS, LFI, Path Traversal, RCE

Usage:
    python generator.py                    # Interactive menu
    python generator.py "SQL Injection"    # Direct generation
"""

import os
import sys
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

# Load API Key
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

if not GDM_API_KEY:
    print("[!] Error: GDM_API_KEY not found")
    sys.exit(1)

client = OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")

# Vulnerability templates for better generation
VULN_TEMPLATES = {
    "SQL Injection": {
        "endpoint": "/vulnerabilities/sqli/",
        "param": "id",
        "description": "Use sqlite3 in-memory database with 'users' table containing admin/password. The endpoint should use string formatting to build SQL query (vulnerable to UNION injection)."
    },
    "Command Injection": {
        "endpoint": "/vulnerabilities/exec/",
        "param": "ip",
        "description": "Endpoint accepts IP for ping command. Use os.system() or subprocess with shell=True WITHOUT sanitization. Should be exploitable with ; or | or &&."
    },
    "Reflected XSS": {
        "endpoint": "/vulnerabilities/xss_r/",
        "param": "name",
        "description": "Reflect the 'name' parameter directly in HTML response WITHOUT escaping. Should allow <script>alert(1)</script> to execute."
    },
    "LFI (File Inclusion)": {
        "endpoint": "/vulnerabilities/fi/",
        "param": "page",
        "description": "Open and read files based on 'page' parameter. Should be exploitable with ../../../etc/passwd to read system files."
    },
    "Path Traversal": {
        "endpoint": "/vulnerabilities/traversal/",
        "param": "file",
        "description": "Serve files from a 'uploads' directory but vulnerable to ../ traversal to access files outside the directory."
    },
    "SSTI (Template Injection)": {
        "endpoint": "/vulnerabilities/ssti/",
        "param": "name",
        "description": "Use Jinja2 render_template_string with user input. Should be exploitable with {{7*7}} or {{config}}."
    }
}


def log(msg, style="white"):
    """Print with optional rich formatting"""
    if HAS_RICH and console:
        console.print(f"[{style}]{msg}[/]")
    else:
        print(msg)


def generate_app(vuln_type: str) -> str:
    """
    Generate a synthetic vulnerable Flask application.
    """
    log(f"\n{'='*60}", "cyan")
    log("SkyHammer Synthetic Generator v2.0", "bold cyan")
    log(f"{'='*60}", "cyan")

    log(f"\n[+] Vulnerability Type: {vuln_type}", "yellow")

    # Get template info
    template = VULN_TEMPLATES.get(vuln_type, {
        "endpoint": "/vulnerable/",
        "param": "input",
        "description": f"Create a {vuln_type} vulnerability that is exploitable."
    })

    log(f"[+] Endpoint: {template['endpoint']}", "dim")
    log(f"[+] Parameter: {template['param']}", "dim")
    log("[+] Generating with Gemini...", "yellow")

    prompt = f"""Write a complete, single-file Python Flask application.

REQUIREMENTS:
1. Run on port 5001: app.run(host='0.0.0.0', port=5001, debug=True)
2. Must have a REAL, EXPLOITABLE vulnerability of type: {vuln_type}
3. NO markdown formatting. NO explanation. ONLY raw Python code.
4. Import all necessary modules at the top.

ROUTES TO INCLUDE:
- GET / : Welcome page with link to vulnerable endpoint
- GET/POST /login.php : Simple login that always succeeds, sets cookies (PHPSESSID, security=low)
- GET/POST {template['endpoint']} : The vulnerable endpoint using parameter '{template['param']}'

VULNERABILITY DETAILS:
{template['description']}

IMPORTANT:
- The vulnerability must be REAL and EXPLOITABLE, not simulated
- Include sample data if needed (like a users table for SQLi)
- Make sure the app actually runs without errors

OUTPUT:
Just the Python code, starting with 'from flask import'. No markdown, no explanations.
"""

    try:
        response = client.chat.completions.create(
            model="gemini-4-1-fast-reasoning",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python code generator for security testing. Output only valid Python code. No markdown fences, no explanations, just code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        code = response.choices[0].message.content

        # Clean up markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1]
                if code.startswith("python\n"):
                    code = code[7:]
                code = code.split("```")[0]

        code = code.strip()

        # Generate filename
        vuln_slug = vuln_type.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        vuln_slug = ''.join(c for c in vuln_slug if c.isalnum() or c == '_')
        filename = f"synthetic_{vuln_slug}.py"

        # Save
        with open(filename, "w") as f:
            f.write(code)

        log(f"\n[+] Generated: {filename}", "bold green")
        log(f"[+] Lines: {len(code.splitlines())}", "green")
        log(f"[+] Port: 5001", "green")
        log(f"[+] Endpoint: {template['endpoint']}?{template['param']}=", "green")

        # Show preview
        if HAS_RICH and console:
            log("\n[Preview]", "dim")
            preview = code[:600] + "\n..." if len(code) > 600 else code
            syntax = Syntax(preview, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Generated Code", border_style="green"))

        log(f"\n{'='*60}", "cyan")
        log("NEXT STEPS:", "bold cyan")
        log(f"{'='*60}", "cyan")
        log(f"1. Start the app:  python {filename}", "white")
        log(f"2. Attack it:      python attack.py --dvwa http://127.0.0.1:5001 --challenge sqli", "white")
        log(f"3. Patch it:       python patcher.py --source {filename}", "white")
        log(f"{'='*60}\n", "cyan")

        return filename

    except Exception as e:
        log(f"\n[!] Error generating app: {e}", "red")
        return None


def interactive_menu():
    """Show interactive menu for vulnerability selection"""
    if not HAS_RICH:
        print("Install 'questionary' for interactive menu: pip install questionary")
        return generate_app("SQL Injection")

    log("\n[bold cyan]Synthetic Vulnerability Generator[/]", "bold cyan")
    log("[dim]Create new vulnerable apps for testing[/]\n", "dim")

    choice = questionary.select(
        "Choose vulnerability type to generate:",
        choices=list(VULN_TEMPLATES.keys()),
        style=questionary.Style([
            ('selected', 'fg:magenta bold'),
            ('pointer', 'fg:magenta bold'),
        ])
    ).ask()

    if choice:
        return generate_app(choice)
    return None


def main():
    if len(sys.argv) > 1:
        vuln_type = " ".join(sys.argv[1:])
        generate_app(vuln_type)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
