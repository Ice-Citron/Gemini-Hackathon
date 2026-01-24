#!/usr/bin/env python3
"""
SkyHammer Patcher - Blue Team Defense

Takes vulnerable code + attack evidence and generates a secure version using Gemini.

Usage:
    python patcher.py                           # Patch mock_dvwa.py
    python patcher.py --source app.py           # Patch custom file
    python patcher.py --evidence "UNION SELECT" # Custom attack evidence
"""

import argparse
import os
import sys
from openai import OpenAI

# Load API key
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

if not GDM_API_KEY:
    print("[!] Error: GDM_API_KEY not found in secretsConfig.py or environment")
    sys.exit(1)

# Setup Gemini client
client = OpenAI(
    api_key=GDM_API_KEY,
    base_url="https://api.x.ai/v1",
)


def patch_code(source_file: str, vulnerability_type: str, attack_evidence: str) -> str:
    """
    Use Gemini to generate a secure version of vulnerable code.

    Args:
        source_file: Path to the vulnerable source code
        vulnerability_type: Type of vulnerability (e.g., "SQL Injection")
        attack_evidence: Evidence from the attacker showing the exploit

    Returns:
        Path to the patched file
    """
    print(f"\n{'='*60}")
    print("SkyHammer Patcher")
    print("Gemini-Powered Security Remediation")
    print(f"{'='*60}")

    print(f"\n[+] Reading source code from: {source_file}")
    with open(source_file, 'r') as f:
        source_code = f.read()

    print(f"[+] Source code: {len(source_code)} bytes, {len(source_code.splitlines())} lines")
    print(f"[+] Vulnerability: {vulnerability_type}")
    print(f"[+] Evidence: {attack_evidence[:100]}...")

    print(f"\n[+] Generating secure patch with Gemini...")

    prompt = f"""You are an expert Secure Code Repair Agent.

TASK: Fix the security vulnerability in the provided Python code.

VULNERABILITY TYPE: {vulnerability_type}

ATTACK EVIDENCE (what the attacker exploited):
{attack_evidence}

INSTRUCTIONS:
1. Analyze the code to find the exact lines allowing the exploit.
2. Rewrite the code to be SECURE:
   - For SQL Injection: Use parameterized queries or strict input validation
   - For XSS: Escape/sanitize all user input before rendering
   - For Command Injection: Whitelist allowed inputs, never pass raw input to shell
3. DO NOT change the functionality - the app must still work for valid inputs.
4. Add a comment "# PATCHED: [reason]" where you made security fixes.
5. Return ONLY the complete valid Python code. No markdown fences, no explanation.

VULNERABLE SOURCE CODE:
{source_code}
"""

    response = client.chat.completions.create(
        model="gemini-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a senior security engineer. Output only raw Python code, no markdown."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=4096,
    )

    fixed_code = response.choices[0].message.content

    # Clean up markdown if Gemini includes it
    if "```python" in fixed_code:
        fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
    elif "```" in fixed_code:
        parts = fixed_code.split("```")
        if len(parts) >= 2:
            fixed_code = parts[1].strip()
            if fixed_code.startswith("python\n"):
                fixed_code = fixed_code[7:]

    # Generate output filename
    base, ext = os.path.splitext(source_file)
    output_file = f"{base}_secure{ext}"

    with open(output_file, 'w') as f:
        f.write(fixed_code)

    print(f"\n[+] Patch generated successfully!")
    print(f"[+] Secure code saved to: {output_file}")
    print(f"[+] Lines changed: analyzing...")

    # Quick diff summary
    original_lines = set(source_code.splitlines())
    patched_lines = set(fixed_code.splitlines())
    added = len(patched_lines - original_lines)
    removed = len(original_lines - patched_lines)
    print(f"    - Lines added: {added}")
    print(f"    - Lines removed/modified: {removed}")

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Stop the vulnerable server (Ctrl+C)")
    print(f"2. Start the secure server:")
    print(f"   sudo python {output_file}")
    print(f"3. Run the attack again:")
    print(f"   python attack.py --challenge sqli")
    print(f"4. The attack should FAIL (system is now secure)")
    print(f"{'='*60}\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="SkyHammer Patcher - Auto-fix security vulnerabilities with Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python patcher.py                              # Patch mock_dvwa.py (default)
    python patcher.py --source myapp.py            # Patch custom file
    python patcher.py --vuln "XSS"                 # Specify vulnerability type
    python patcher.py --evidence "UNION SELECT..." # Custom attack evidence
        """
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        default="mock_dvwa.py",
        help="Source file to patch (default: mock_dvwa.py)"
    )

    parser.add_argument(
        "--vuln", "-v",
        type=str,
        default="SQL Injection (UNION-based)",
        help="Vulnerability type"
    )

    parser.add_argument(
        "--evidence", "-e",
        type=str,
        default="""Payload: id=999 UNION SELECT user_id, first_name, last_name, user, password FROM users --

Extracted data:
- user_id: 1, first_name: admin, last_name: admin, user: admin, password: 5f4dcc3b5aa765d61d8327deb882cf99
- user_id: 2, first_name: Gordon, last_name: Brown, user: gordonb, password: e99a18c428cb38d5f260853678922e03

The attacker used UNION-based SQL injection to extract all usernames and password hashes from the database.""",
        help="Attack evidence from the red team"
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"[!] Error: Source file not found: {args.source}")
        sys.exit(1)

    patched_file = patch_code(args.source, args.vuln, args.evidence)

    return 0


if __name__ == "__main__":
    sys.exit(main())
