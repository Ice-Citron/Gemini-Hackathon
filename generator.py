#!/usr/bin/env python3
"""
AthenaGuard Synthetic Vulnerability Generator

Uses Gemini to generate new vulnerable Flask applications on-the-fly.
This proves the system can handle unknown/unseen vulnerabilities.

Usage:
    python generator.py "SQL Injection"
    python generator.py "Command Injection"
    python generator.py "Reflected XSS"
"""

import os
import sys
import time
from openai import OpenAI

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    console = Console()
except ImportError:
    console = None

# Load API Key
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

if not GDM_API_KEY:
    print("[!] Error: GDM_API_KEY not found")
    sys.exit(1)

client = OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")


def log(msg, style="white"):
    """Print with optional rich formatting"""
    if console:
        console.print(f"[{style}]{msg}[/]")
    else:
        print(msg)


def generate_app(vuln_type: str) -> str:
    """
    Generate a synthetic vulnerable Flask application.

    Args:
        vuln_type: Type of vulnerability (e.g., "SQL Injection", "XSS")

    Returns:
        Path to the generated file
    """
    log(f"\n{'='*60}", "cyan")
    log("AthenaGuard Synthetic Generator", "bold cyan")
    log(f"{'='*60}", "cyan")

    log(f"\n[+] Vulnerability Type: {vuln_type}", "yellow")
    log("[+] Generating with Gemini-4...", "yellow")

    prompt = f"""Write a complete, single-file Python Flask application.

REQUIREMENTS:
1. Run on port 5001: app.run(host='0.0.0.0', port=5001, debug=True)
2. Must have a REAL, EXPLOITABLE vulnerability of type: {vuln_type}
3. The vulnerability must be exploitable via HTTP GET or POST parameters
4. Include a login route at /login.php that always succeeds (for compatibility)
5. Include a home route at / that shows a welcome message
6. NO markdown formatting. NO explanation. ONLY raw Python code.
7. Import all necessary modules at the top.

VULNERABILITY SPECIFICS:
- If SQL Injection:
  * Create an in-memory SQLite database with a 'users' table
  * Populate with test data (admin/password, guest/guest123)
  * Create /vulnerabilities/sqli/ endpoint that takes 'id' parameter
  * Use string formatting (f-string or %) to build query - THIS IS THE VULNERABILITY
  * Return user data in HTML format

- If Command Injection:
  * Create /vulnerabilities/exec/ endpoint that takes 'ip' parameter via POST
  * Use os.system() or subprocess with shell=True - THIS IS THE VULNERABILITY
  * Intended for "ping" but exploitable with ; or |

- If Reflected XSS:
  * Create /vulnerabilities/xss_r/ endpoint that takes 'name' parameter
  * Reflect the parameter directly in HTML without escaping - THIS IS THE VULNERABILITY
  * Return HTML that includes the user input

- If Path Traversal:
  * Create /vulnerabilities/fi/ endpoint that takes 'page' parameter
  * Use open() with user input to read files - THIS IS THE VULNERABILITY
  * Allow reading files outside web directory with ../

OUTPUT FORMAT:
Just the Python code, starting with imports. No markdown, no explanations.
"""

    try:
        response = client.chat.completions.create(
            model="gemini-4-1-fast-reasoning",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python code generator for security testing. Output only valid Python code, no markdown fences or explanations."
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

        # Generate filename based on vuln type
        vuln_slug = vuln_type.lower().replace(" ", "_").replace("-", "_")
        vuln_slug = ''.join(c for c in vuln_slug if c.isalnum() or c == '_')
        filename = f"synthetic_{vuln_slug}.py"

        # Save
        with open(filename, "w") as f:
            f.write(code)

        log(f"\n[+] Generated: {filename}", "green")
        log(f"[+] Lines: {len(code.splitlines())}", "green")
        log(f"[+] Port: 5001", "green")

        # Show preview
        if console:
            log("\n[Preview]", "dim")
            preview = code[:500] + "..." if len(code) > 500 else code
            syntax = Syntax(preview, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Generated Code", border_style="green"))

        log(f"\n{'='*60}", "cyan")
        log("NEXT STEPS:", "bold cyan")
        log(f"{'='*60}", "cyan")
        log(f"1. Start the app:  python {filename}", "white")
        log(f"2. Attack it:      python attack.py --dvwa http://127.0.0.1:5001", "white")
        log(f"3. Patch it:       python patcher.py --source {filename}", "white")
        log(f"{'='*60}\n", "cyan")

        return filename

    except Exception as e:
        log(f"\n[!] Error generating app: {e}", "red")
        return None


def main():
    if len(sys.argv) > 1:
        vuln_type = " ".join(sys.argv[1:])
    else:
        # Default
        vuln_type = "SQL Injection"

    generate_app(vuln_type)


if __name__ == "__main__":
    main()
