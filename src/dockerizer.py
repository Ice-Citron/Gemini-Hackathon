#!/usr/bin/env python3
"""
SkyHammer Auto-Dockerizer

Point it at a repo/folder and Gemini generates Dockerfile + docker-compose.yml.
Can also clone a GitHub repo and set it up automatically.

Usage:
    python dockerizer.py                    # Dockerize current directory
    python dockerizer.py /path/to/repo      # Dockerize specific path
    python dockerizer.py https://github.com/user/repo  # Clone and dockerize
"""

import os
import sys
import subprocess
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


def log(msg, style="white"):
    """Print with optional rich formatting"""
    if HAS_RICH and console:
        console.print(f"[{style}]{msg}[/]")
    else:
        print(msg)


def clone_repo(url: str) -> str:
    """Clone a GitHub repo and return the path"""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    target_path = f"./cloned_{repo_name}"

    if os.path.exists(target_path):
        log(f"[!] Directory {target_path} already exists. Using it.", "yellow")
        return target_path

    log(f"[+] Cloning {url}...", "cyan")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, target_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log(f"[!] Clone failed: {result.stderr}", "red")
        return None

    log(f"[+] Cloned to {target_path}", "green")
    return target_path


def scan_directory(path: str) -> dict:
    """Scan directory and gather context for Gemini"""
    info = {
        "files": [],
        "requirements": "",
        "package_json": "",
        "main_file": None,
        "language": "python"
    }

    for item in os.listdir(path):
        if item.startswith("."):
            continue
        info["files"].append(item)

        # Read key files
        full_path = os.path.join(path, item)
        if item == "requirements.txt":
            with open(full_path) as f:
                info["requirements"] = f.read()
        elif item == "package.json":
            with open(full_path) as f:
                info["package_json"] = f.read()
                info["language"] = "node"
        elif item in ["app.py", "main.py", "server.py", "index.py"]:
            info["main_file"] = item
        elif item == "Cargo.toml":
            info["language"] = "rust"
        elif item == "go.mod":
            info["language"] = "go"

    return info


def auto_dockerize(target_path: str = ".") -> bool:
    """Generate Dockerfile and docker-compose.yml using Gemini"""
    log(f"\n{'='*60}", "cyan")
    log("SkyHammer Auto-Dockerizer", "bold cyan")
    log(f"{'='*60}", "cyan")

    # Handle GitHub URLs
    if target_path.startswith("http"):
        target_path = clone_repo(target_path)
        if not target_path:
            return False

    if not os.path.isdir(target_path):
        log(f"[!] Not a directory: {target_path}", "red")
        return False

    log(f"\n[+] Scanning: {target_path}", "yellow")

    # Gather context
    info = scan_directory(target_path)

    log(f"[+] Files found: {len(info['files'])}", "dim")
    log(f"[+] Language detected: {info['language']}", "dim")
    if info["main_file"]:
        log(f"[+] Main file: {info['main_file']}", "dim")

    log("[+] Generating Docker config with Gemini...", "yellow")

    prompt = f"""You are a DevOps Expert. Generate a Dockerfile and docker-compose.yml for this project.

PROJECT INFO:
- Language: {info['language']}
- Main file: {info['main_file'] or 'Unknown'}
- Files in repo: {', '.join(info['files'][:20])}

REQUIREMENTS.TXT:
{info['requirements'] or 'Not found'}

PACKAGE.JSON:
{info['package_json'][:500] if info['package_json'] else 'Not found'}

INSTRUCTIONS:
1. First output the Dockerfile content
2. Then output exactly this line: ### DELIMITER ###
3. Then output the docker-compose.yml content
4. NO markdown fences. Just raw file contents.
5. For Python: use python:3.11-slim, install requirements, expose port 5000 or 80
6. For Node: use node:20-slim, npm install, expose port 3000
7. Include health checks and proper WORKDIR

OUTPUT FORMAT:
[Dockerfile content]
### DELIMITER ###
[docker-compose.yml content]
"""

    try:
        response = client.chat.completions.create(
            model="gemini-4-1-fast-reasoning",  # $0.20/M tokens - spam it!
            messages=[
                {
                    "role": "system",
                    "content": "You are a DevOps expert. Output only raw file contents, no markdown."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        content = response.choices[0].message.content

        # Parse output
        if "### DELIMITER ###" not in content:
            log("[!] Unexpected output format. Trying to parse...", "yellow")
            # Try to find Dockerfile and docker-compose sections
            dockerfile_content = content
            compose_content = ""
        else:
            parts = content.split("### DELIMITER ###")
            dockerfile_content = parts[0].strip()
            compose_content = parts[1].strip() if len(parts) > 1 else ""

        # Clean up markdown if present
        for marker in ["```dockerfile", "```yaml", "```"]:
            dockerfile_content = dockerfile_content.replace(marker, "")
            compose_content = compose_content.replace(marker, "")

        dockerfile_content = dockerfile_content.strip()
        compose_content = compose_content.strip()

        # Write files
        dockerfile_path = os.path.join(target_path, "Dockerfile")
        compose_path = os.path.join(target_path, "docker-compose.yml")

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        log(f"[+] Created: {dockerfile_path}", "green")

        if compose_content:
            with open(compose_path, "w") as f:
                f.write(compose_content)
            log(f"[+] Created: {compose_path}", "green")

        # Show preview
        if HAS_RICH and console:
            log("\n[Dockerfile Preview]", "dim")
            preview = dockerfile_content[:400] + "\n..." if len(dockerfile_content) > 400 else dockerfile_content
            console.print(Panel(preview, title="Dockerfile", border_style="blue"))

        log(f"\n{'='*60}", "cyan")
        log("NEXT STEPS:", "bold cyan")
        log(f"{'='*60}", "cyan")
        log(f"1. cd {target_path}", "white")
        log("2. docker compose up --build", "white")
        log("3. Access at http://localhost:5000 or http://localhost:80", "white")
        log(f"{'='*60}\n", "cyan")

        return True

    except Exception as e:
        log(f"\n[!] Error: {e}", "red")
        return False


def interactive_menu():
    """Interactive menu for dockerization"""
    if not HAS_RICH:
        return auto_dockerize(".")

    log("\n[bold cyan]Auto-Dockerizer[/]", "bold cyan")
    log("[dim]Generate Docker configs with Gemini[/]\n", "dim")

    choice = questionary.select(
        "What do you want to dockerize?",
        choices=[
            "Current directory (.)",
            "Enter a local path",
            "Clone from GitHub URL",
            "Cancel"
        ],
        style=questionary.Style([
            ('selected', 'fg:blue bold'),
            ('pointer', 'fg:blue bold'),
        ])
    ).ask()

    if choice == "Current directory (.)":
        return auto_dockerize(".")
    elif choice == "Enter a local path":
        path = questionary.text("Enter path:").ask()
        if path:
            return auto_dockerize(path)
    elif choice == "Clone from GitHub URL":
        url = questionary.text("Enter GitHub URL:").ask()
        if url:
            return auto_dockerize(url)

    return False


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        auto_dockerize(target)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
