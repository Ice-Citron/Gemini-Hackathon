#!/usr/bin/env python3
"""
SkyHammer Gemini Judge
Reward function using Gemini API to score security patches.
Returns scores 0.0 - 1.0 for RLAIF training.
"""

import os
import json
import re
from typing import Optional, Tuple

from openai import OpenAI

# Load API key
try:
    from secretsConfig import GDM_API_KEY
except ImportError:
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")

if not GDM_API_KEY:
    print("[!] Warning: GDM_API_KEY not found. Gemini judge will not work.")
    client = None
else:
    client = OpenAI(api_key=GDM_API_KEY, base_url="https://api.x.ai/v1")


# Safety patterns - hard penalty
BLOCKED_PATTERNS = [
    "rm -rf /",
    ":(){ :|:& };:",  # Fork bomb
    "DROP DATABASE",
    "DROP TABLE",
    "DELETE FROM",
    "TRUNCATE",
    "> /dev/sda",
    "mkfs",
    "dd if=/dev/zero",
]


def safety_filter(code: str) -> float:
    """
    Check for dangerous patterns.
    Returns -10.0 if blocked pattern found, 0.0 otherwise.
    """
    code_upper = code.upper()
    for pattern in BLOCKED_PATTERNS:
        if pattern.upper() in code_upper:
            return -10.0
    return 0.0


def check_parameterized_query(code: str) -> bool:
    """
    Check if code uses parameterized queries.
    Returns True if parameterized patterns found.
    """
    # Common patterns for parameterized queries
    parameterized_patterns = [
        r"execute\s*\([^,]+,\s*\(",    # cursor.execute(query, (param,))
        r"execute\s*\([^,]+,\s*\[",    # cursor.execute(query, [param])
        r"\?\s*\)",                     # WHERE x = ?)
        r"%s\s*\)",                     # WHERE x = %s) with tuple
        r":[\w]+",                      # :param style (SQLAlchemy)
        r"\$\d+",                       # $1, $2 style (PostgreSQL)
    ]

    for pattern in parameterized_patterns:
        if re.search(pattern, code):
            return True
    return False


def check_string_interpolation(code: str) -> bool:
    """
    Check if code still has dangerous string interpolation.
    Returns True if vulnerable patterns found.
    """
    vulnerable_patterns = [
        r'f"[^"]*SELECT[^"]*\{',        # f-string with SELECT
        r"f'[^']*SELECT[^']*\{",        # f-string with SELECT
        r'\.format\([^)]*\)',           # .format() usage
        r'%\s*["\'][^"\']*SELECT',      # % formatting with SELECT
        r'\+\s*["\']',                   # String concatenation
        r'["\']\s*\+',                   # String concatenation
    ]

    for pattern in vulnerable_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return True
    return False


def programmatic_score(vulnerable_code: str, patched_code: str, test_result: str = "") -> Tuple[float, str]:
    """
    Programmatic scoring (hard checks).
    Returns (score, reason).
    """
    score = 0.0
    reasons = []

    # 1. Safety check
    safety = safety_filter(patched_code)
    if safety < 0:
        return safety, "BLOCKED: Dangerous pattern detected"

    # 2. Did exploit fail? (Binary hard gate)
    if test_result:
        if "EXPLOIT FAILED" in test_result or "error" in test_result.lower():
            score += 0.3
            reasons.append("Exploit appears to have failed")
        elif "Server failed" in test_result or "crash" in test_result.lower():
            return -0.5, "PENALTY: Patch broke the application"

    # 3. Uses parameterized queries?
    if check_parameterized_query(patched_code):
        score += 0.3
        reasons.append("Uses parameterized queries")
    else:
        reasons.append("Does not use parameterized queries")

    # 4. Still has string interpolation?
    if check_string_interpolation(patched_code):
        score -= 0.2
        reasons.append("Still contains string interpolation")
    else:
        score += 0.1
        reasons.append("No string interpolation detected")

    return score, "; ".join(reasons)


def gemini_judge(vulnerable_code: str, patched_code: str, model: str = "gemini-beta") -> Tuple[float, str]:
    """
    Use Gemini to evaluate patch quality.
    Returns (score, reason).
    """
    if not client:
        return 0.5, "Gemini API not available (no API key)"

    prompt = f"""You are a Senior Security Engineer. Rate this security patch on a scale of 0.0 to 1.0.

ORIGINAL VULNERABLE CODE:
```python
{vulnerable_code}
```

PATCHED CODE:
```python
{patched_code}
```

EVALUATION CRITERIA:
1. MUST use parameterized queries (prepared statements with ? or %s placeholders)
2. MUST NOT use string formatting/interpolation in SQL queries
3. MUST NOT use regex/sanitization alone (these are brittle)
4. MUST preserve the original application logic
5. Code should be clean and maintainable

SCORING GUIDE:
- 0.0: Patch does not fix the vulnerability or introduces new issues
- 0.3: Patch partially fixes but uses weak methods (regex, sanitization)
- 0.5: Patch fixes issue but code quality is poor
- 0.7: Good patch with parameterized queries
- 0.9: Excellent patch, clean code, best practices
- 1.0: Perfect, production-ready security patch

Output ONLY a JSON object:
{{"score": 0.8, "reason": "Brief explanation of score"}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        # Try to extract JSON if wrapped in other text
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            return float(result.get("score", 0.5)), result.get("reason", "No reason provided")
        else:
            # Fallback: try direct parse
            result = json.loads(content)
            return float(result.get("score", 0.5)), result.get("reason", "No reason provided")

    except json.JSONDecodeError:
        return 0.5, f"Failed to parse Gemini response: {content[:100]}"
    except Exception as e:
        return 0.5, f"Gemini API error: {str(e)}"


def compute_reward(
    vulnerable_code: str,
    patched_code: str,
    test_result: str = "",
    use_gemini: bool = True
) -> Tuple[float, str]:
    """
    Hybrid reward function combining programmatic and Gemini scoring.

    Returns:
        Tuple of (score, reason) where score is in range [-0.5, 1.0]
    """
    # 1. Programmatic checks (hard constraints)
    prog_score, prog_reason = programmatic_score(vulnerable_code, patched_code, test_result)

    # Early exit on safety violation
    if prog_score <= -1.0:
        return prog_score, prog_reason

    # 2. Gemini soft check (if enabled)
    if use_gemini and client:
        gemini_score, gemini_reason = gemini_judge(vulnerable_code, patched_code)
        # Weighted combination: 60% programmatic, 40% Gemini
        final_score = (prog_score * 0.6) + (gemini_score * 0.4)
        final_reason = f"Programmatic ({prog_score:.2f}): {prog_reason} | Gemini ({gemini_score:.2f}): {gemini_reason}"
    else:
        final_score = prog_score
        final_reason = f"Programmatic only: {prog_reason}"

    # Clamp to valid range
    final_score = max(-0.5, min(1.0, final_score))

    return final_score, final_reason


# Example usage and testing
if __name__ == "__main__":
    # Test case
    vulnerable = '''
query = f"SELECT * FROM users WHERE name = '{username}'"
cursor.execute(query)
'''

    patched_good = '''
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (username,))
'''

    patched_bad = '''
username = username.replace("'", "''")  # Sanitization
query = f"SELECT * FROM users WHERE name = '{username}'"
cursor.execute(query)
'''

    print("Testing Reward Function...")
    print("\n1. Good patch (parameterized):")
    score, reason = compute_reward(vulnerable, patched_good, use_gemini=True)
    print(f"   Score: {score:.2f}")
    print(f"   Reason: {reason}")

    print("\n2. Bad patch (sanitization only):")
    score, reason = compute_reward(vulnerable, patched_bad, use_gemini=True)
    print(f"   Score: {score:.2f}")
    print(f"   Reason: {reason}")
