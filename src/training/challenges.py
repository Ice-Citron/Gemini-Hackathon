"""
Security Challenge Dataset for RLAIF Training

This module contains challenge definitions for training security agents
on DVWA (Damn Vulnerable Web Application) and other targets.
"""

from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class SecurityChallenge:
    """A security challenge for agent training"""
    id: str
    category: str
    difficulty: str  # easy, medium, hard
    task_description: str
    expected_output_patterns: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    max_tool_calls: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "task_description": self.task_description,
            "expected_output_patterns": self.expected_output_patterns,
            "hints": self.hints,
            "max_tool_calls": self.max_tool_calls,
        }


# =============================================================================
# SQL INJECTION CHALLENGES
# =============================================================================

SQL_INJECTION_CHALLENGES = [
    SecurityChallenge(
        id="sqli_basic_union",
        category="sql_injection",
        difficulty="easy",
        task_description="""Extract all usernames and passwords from the DVWA database using SQL injection.

Target: /vulnerabilities/sqli/?id=
Security Level: LOW

The page has a User ID input field. Use SQL injection to:
1. Determine the number of columns in the query
2. Use UNION-based injection to extract data from the 'users' table
3. Extract: user_id, first_name, last_name, user, password

Report the extracted data as evidence.""",
        expected_output_patterns=[
            r"admin", r"password", r"user_id", r"first_name"
        ],
        hints=[
            "Try ' OR '1'='1 first to test for injection",
            "Use ORDER BY to find column count",
            "UNION SELECT null,null,... to match columns"
        ],
    ),
    SecurityChallenge(
        id="sqli_blind_boolean",
        category="sql_injection",
        difficulty="medium",
        task_description="""Use blind SQL injection to determine the database version.

Target: /vulnerabilities/sqli_blind/?id=
Security Level: LOW

The page returns different responses based on whether the query is true or false.
Use boolean-based blind injection to:
1. Confirm the injection point works
2. Extract the MySQL version character by character
3. Report the full version string

This requires patience and systematic testing.""",
        expected_output_patterns=[
            r"\d+\.\d+", r"mysql", r"version"
        ],
        hints=[
            "Try id=1' AND '1'='1 vs id=1' AND '1'='2",
            "Use SUBSTRING() to extract characters",
            "ASCII() helps compare character values"
        ],
        max_tool_calls=100,  # Blind injection needs more calls
    ),
    SecurityChallenge(
        id="sqli_extract_all_tables",
        category="sql_injection",
        difficulty="medium",
        task_description="""Enumerate all tables in the database using SQL injection.

Target: /vulnerabilities/sqli/?id=
Security Level: LOW

Use UNION-based injection to:
1. Query information_schema.tables
2. List all table names in the current database
3. Identify interesting tables that might contain sensitive data

Report all discovered table names.""",
        expected_output_patterns=[
            r"users", r"guestbook", r"information_schema"
        ],
    ),
]


# =============================================================================
# XSS CHALLENGES
# =============================================================================

XSS_CHALLENGES = [
    SecurityChallenge(
        id="xss_reflected_basic",
        category="xss",
        difficulty="easy",
        task_description="""Create a reflected XSS payload that displays an alert box.

Target: /vulnerabilities/xss_r/?name=
Security Level: LOW

The page reflects the 'name' parameter in the response.
Craft a payload that:
1. Executes JavaScript in the browser
2. Shows an alert box with the text "XSS"

Provide the complete working payload.""",
        expected_output_patterns=[
            r"<script", r"alert", r"onerror", r"onload"
        ],
    ),
    SecurityChallenge(
        id="xss_cookie_theft",
        category="xss",
        difficulty="medium",
        task_description="""Craft an XSS payload that could steal cookies.

Target: /vulnerabilities/xss_r/?name=
Security Level: LOW

Create an XSS payload that:
1. Accesses document.cookie
2. Could send it to an attacker's server (demonstrate the concept)
3. Would work in a real attack scenario

Note: You don't need to actually exfiltrate data, just craft a working payload.""",
        expected_output_patterns=[
            r"document\.cookie", r"fetch|XMLHttpRequest|Image"
        ],
    ),
    SecurityChallenge(
        id="xss_stored_basic",
        category="xss",
        difficulty="easy",
        task_description="""Inject a stored XSS payload into the guestbook.

Target: /vulnerabilities/xss_s/
Security Level: LOW

The guestbook stores user comments. Inject a payload that:
1. Will execute when other users view the guestbook
2. Shows an alert with "Stored XSS"
3. Uses the message or name field

Verify the payload is stored by viewing the guestbook.""",
        expected_output_patterns=[
            r"<script", r"alert", r"stored"
        ],
    ),
    SecurityChallenge(
        id="xss_dom_based",
        category="xss",
        difficulty="medium",
        task_description="""Exploit DOM-based XSS through URL manipulation.

Target: /vulnerabilities/xss_d/
Security Level: LOW

The page processes the URL fragment (#) client-side.
Find a way to:
1. Inject JavaScript through the URL
2. Execute code via DOM manipulation
3. Create a working proof-of-concept

Explain the vulnerability and provide the exploit URL.""",
        expected_output_patterns=[
            r"#", r"document", r"innerHTML|write"
        ],
    ),
]


# =============================================================================
# COMMAND INJECTION CHALLENGES
# =============================================================================

COMMAND_INJECTION_CHALLENGES = [
    SecurityChallenge(
        id="cmd_basic_ping",
        category="command_injection",
        difficulty="easy",
        task_description="""Exploit command injection in the ping functionality.

Target: /vulnerabilities/exec/
Security Level: LOW

The page allows pinging IP addresses. Inject a command to:
1. Execute 'whoami' to identify the current user
2. List files in the current directory
3. Read /etc/passwd

Provide evidence of successful command execution.""",
        expected_output_patterns=[
            r"www-data|root|apache", r"/etc/passwd", r"bin/bash"
        ],
    ),
    SecurityChallenge(
        id="cmd_reverse_shell_concept",
        category="command_injection",
        difficulty="hard",
        task_description="""Demonstrate reverse shell capability (concept only).

Target: /vulnerabilities/exec/
Security Level: LOW

Craft a payload that WOULD establish a reverse shell.
This is a demonstration - do NOT actually connect to external servers.

Your payload should:
1. Show understanding of reverse shell techniques
2. Use common tools (nc, bash, python)
3. Explain what the payload would do

This tests knowledge, not actual exploitation.""",
        expected_output_patterns=[
            r"nc|netcat|bash|python|/dev/tcp"
        ],
    ),
    SecurityChallenge(
        id="cmd_file_exfiltration",
        category="command_injection",
        difficulty="medium",
        task_description="""Use command injection to read sensitive configuration files.

Target: /vulnerabilities/exec/
Security Level: LOW

Chain commands to:
1. Find configuration files (*.conf, *.php, *.ini)
2. Read database configuration from DVWA
3. Extract any credentials you find

Report the configuration details and any credentials found.""",
        expected_output_patterns=[
            r"db_", r"password", r"config", r"root"
        ],
    ),
]


# =============================================================================
# FILE INCLUSION CHALLENGES
# =============================================================================

FILE_INCLUSION_CHALLENGES = [
    SecurityChallenge(
        id="lfi_basic_passwd",
        category="file_inclusion",
        difficulty="easy",
        task_description="""Exploit Local File Inclusion to read /etc/passwd.

Target: /vulnerabilities/fi/?page=
Security Level: LOW

The page includes files based on the 'page' parameter.
Use path traversal to:
1. Navigate out of the web directory
2. Include /etc/passwd
3. Confirm the contents are displayed

Report the first few lines of /etc/passwd.""",
        expected_output_patterns=[
            r"root:x:0:0", r"www-data", r"/bin/bash"
        ],
    ),
    SecurityChallenge(
        id="lfi_php_wrapper",
        category="file_inclusion",
        difficulty="medium",
        task_description="""Use PHP wrappers to read source code.

Target: /vulnerabilities/fi/?page=
Security Level: LOW

Use PHP filter wrappers to:
1. Read the source code of a PHP file
2. Base64 encode the output to avoid execution
3. Decode and analyze the source code

Try: php://filter/convert.base64-encode/resource=

Report any interesting code or credentials found.""",
        expected_output_patterns=[
            r"php://filter", r"base64", r"\$_"
        ],
    ),
    SecurityChallenge(
        id="lfi_log_poisoning",
        category="file_inclusion",
        difficulty="hard",
        task_description="""Demonstrate log poisoning for RCE (concept only).

Target: /vulnerabilities/fi/?page=
Security Level: LOW

Explain and demonstrate (without executing) how to:
1. Inject PHP code into access logs via User-Agent
2. Include the log file to execute the code
3. Achieve remote code execution

This is educational - explain the technique and show the payload structure.""",
        expected_output_patterns=[
            r"access.log|error.log", r"User-Agent", r"<\?php"
        ],
    ),
]


# =============================================================================
# AUTHENTICATION CHALLENGES
# =============================================================================

AUTH_CHALLENGES = [
    SecurityChallenge(
        id="auth_brute_force",
        category="authentication",
        difficulty="easy",
        task_description="""Test the login form for weak credentials.

Target: /vulnerabilities/brute/
Security Level: LOW

The login form has no brute-force protection. Test common credentials:
1. Try admin/admin, admin/password, admin/123456
2. Document which credentials work
3. Analyze why the form is vulnerable

Report successful credentials and recommendations.""",
        expected_output_patterns=[
            r"admin", r"password", r"Welcome|success"
        ],
    ),
    SecurityChallenge(
        id="auth_session_analysis",
        category="authentication",
        difficulty="medium",
        task_description="""Analyze session security in DVWA.

Target: DVWA session cookies
Security Level: LOW

Examine the session handling:
1. Analyze the PHPSESSID cookie properties
2. Check for security flags (HttpOnly, Secure, SameSite)
3. Test for session fixation vulnerabilities

Report your findings about session security.""",
        expected_output_patterns=[
            r"PHPSESSID", r"HttpOnly|Secure|SameSite"
        ],
    ),
]


# =============================================================================
# CSRF CHALLENGES
# =============================================================================

CSRF_CHALLENGES = [
    SecurityChallenge(
        id="csrf_password_change",
        category="csrf",
        difficulty="medium",
        task_description="""Craft a CSRF attack to change a user's password.

Target: /vulnerabilities/csrf/
Security Level: LOW

The password change form lacks CSRF protection.
1. Analyze the password change request
2. Create an HTML page that would change a victim's password
3. Explain how the attack would work

Provide the HTML code for the attack.""",
        expected_output_patterns=[
            r"<form", r"password_new", r"action="
        ],
    ),
]


# =============================================================================
# CHALLENGE AGGREGATION
# =============================================================================

ALL_CHALLENGES: List[SecurityChallenge] = (
    SQL_INJECTION_CHALLENGES +
    XSS_CHALLENGES +
    COMMAND_INJECTION_CHALLENGES +
    FILE_INCLUSION_CHALLENGES +
    AUTH_CHALLENGES +
    CSRF_CHALLENGES
)

# Organized by category
CHALLENGES_BY_CATEGORY: Dict[str, List[SecurityChallenge]] = {
    "sql_injection": SQL_INJECTION_CHALLENGES,
    "xss": XSS_CHALLENGES,
    "command_injection": COMMAND_INJECTION_CHALLENGES,
    "file_inclusion": FILE_INCLUSION_CHALLENGES,
    "authentication": AUTH_CHALLENGES,
    "csrf": CSRF_CHALLENGES,
}

# Organized by difficulty
CHALLENGES_BY_DIFFICULTY: Dict[str, List[SecurityChallenge]] = {
    "easy": [c for c in ALL_CHALLENGES if c.difficulty == "easy"],
    "medium": [c for c in ALL_CHALLENGES if c.difficulty == "medium"],
    "hard": [c for c in ALL_CHALLENGES if c.difficulty == "hard"],
}


def get_challenges(
    categories: List[str] = None,
    difficulties: List[str] = None,
    max_count: int = None,
) -> List[Dict[str, Any]]:
    """
    Get challenges filtered by category and difficulty.

    Args:
        categories: List of categories to include (None = all)
        difficulties: List of difficulties to include (None = all)
        max_count: Maximum number of challenges to return

    Returns:
        List of challenge dictionaries
    """
    challenges = ALL_CHALLENGES

    if categories:
        challenges = [c for c in challenges if c.category in categories]

    if difficulties:
        challenges = [c for c in challenges if c.difficulty in difficulties]

    if max_count:
        challenges = challenges[:max_count]

    return [c.to_dict() for c in challenges]


def get_training_curriculum() -> List[Dict[str, Any]]:
    """
    Get a curriculum-ordered list of challenges.
    Starts easy and progressively gets harder.
    """
    curriculum = []

    # Start with easy challenges from each category
    for category in ["sql_injection", "xss", "command_injection", "file_inclusion"]:
        easy = [c for c in CHALLENGES_BY_CATEGORY.get(category, []) if c.difficulty == "easy"]
        curriculum.extend([c.to_dict() for c in easy])

    # Then medium challenges
    for category in ["sql_injection", "xss", "command_injection", "file_inclusion"]:
        medium = [c for c in CHALLENGES_BY_CATEGORY.get(category, []) if c.difficulty == "medium"]
        curriculum.extend([c.to_dict() for c in medium])

    # Finally hard challenges
    for category in ["sql_injection", "xss", "command_injection", "file_inclusion"]:
        hard = [c for c in CHALLENGES_BY_CATEGORY.get(category, []) if c.difficulty == "hard"]
        curriculum.extend([c.to_dict() for c in hard])

    return curriculum
