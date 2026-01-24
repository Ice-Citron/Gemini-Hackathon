"""
Security Scanner Module

Uses Semgrep for SAST scanning to find vulnerabilities.
Deterministic, fast, and language-agnostic.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional
import tempfile

from .schemas import Finding


class SecurityScanner:
    """
    Security scanner using Semgrep for static analysis.

    Semgrep is:
    - Fast and deterministic
    - Language-agnostic (Python, JS, Go, etc.)
    - Has extensive security rule packs
    """

    def __init__(
        self,
        rules: str = "p/security-audit",
        additional_rules: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the scanner.

        Args:
            rules: Semgrep rule pack or path to rules
            additional_rules: Additional rule packs to include
            exclude_patterns: Patterns to exclude from scanning
        """
        self.rules = rules
        self.additional_rules = additional_rules or []
        self.exclude_patterns = exclude_patterns or [
            "node_modules",
            "venv",
            ".git",
            "__pycache__",
            "*.min.js",
        ]

    def scan(self, target_path: str) -> List[Finding]:
        """
        Scan a file or directory for vulnerabilities.

        Args:
            target_path: Path to file or directory to scan

        Returns:
            List of Finding objects
        """
        target = Path(target_path)
        if not target.exists():
            raise FileNotFoundError(f"Target not found: {target_path}")

        # Build semgrep command
        cmd = [
            "semgrep",
            "--json",
            "--config", self.rules,
        ]

        # Add additional rules
        for rule in self.additional_rules:
            cmd.extend(["--config", rule])

        # Add exclusions
        for pattern in self.exclude_patterns:
            cmd.extend(["--exclude", pattern])

        # Add target
        cmd.append(str(target))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse JSON output
            if result.returncode == 0 or result.stdout:
                return self._parse_semgrep_output(result.stdout)
            else:
                print(f"Semgrep error: {result.stderr}")
                return []

        except subprocess.TimeoutExpired:
            print("Semgrep scan timed out")
            return []
        except FileNotFoundError:
            print("Semgrep not installed. Install with: pip install semgrep")
            return self._fallback_scan(target_path)

    def _parse_semgrep_output(self, output: str) -> List[Finding]:
        """Parse Semgrep JSON output into Finding objects"""
        findings = []

        try:
            data = json.loads(output)
            results = data.get("results", [])

            for result in results:
                finding = Finding(
                    file_path=result.get("path", ""),
                    rule_id=result.get("check_id", "unknown"),
                    title=result.get("extra", {}).get("message", "Security finding"),
                    severity=self._map_severity(result.get("extra", {}).get("severity", "INFO")),
                    snippet=result.get("extra", {}).get("lines", ""),
                    line_start=result.get("start", {}).get("line", 0),
                    line_end=result.get("end", {}).get("line", 0),
                    message=result.get("extra", {}).get("message", ""),
                    cwe=self._extract_cwe(result),
                    owasp=self._extract_owasp(result),
                )
                findings.append(finding)

        except json.JSONDecodeError as e:
            print(f"Failed to parse Semgrep output: {e}")

        return findings

    def _map_severity(self, semgrep_severity: str) -> str:
        """Map Semgrep severity to our standard levels"""
        mapping = {
            "ERROR": "high",
            "WARNING": "medium",
            "INFO": "low",
        }
        return mapping.get(semgrep_severity.upper(), "info")

    def _extract_cwe(self, result: dict) -> Optional[str]:
        """Extract CWE from Semgrep metadata"""
        metadata = result.get("extra", {}).get("metadata", {})
        cwe = metadata.get("cwe", [])
        if isinstance(cwe, list) and cwe:
            return cwe[0]
        return cwe if isinstance(cwe, str) else None

    def _extract_owasp(self, result: dict) -> Optional[str]:
        """Extract OWASP category from Semgrep metadata"""
        metadata = result.get("extra", {}).get("metadata", {})
        owasp = metadata.get("owasp", [])
        if isinstance(owasp, list) and owasp:
            return owasp[0]
        return owasp if isinstance(owasp, str) else None

    def _fallback_scan(self, target_path: str) -> List[Finding]:
        """
        Fallback scanner using regex patterns when Semgrep unavailable.
        This is less accurate but works without dependencies.
        """
        import re

        findings = []
        target = Path(target_path)

        # Common vulnerability patterns
        patterns = {
            "sql_injection": (
                r'execute\s*\(\s*f["\']|execute\s*\(\s*["\'].*%s|cursor\.execute\s*\([^,]+\+',
                "SQL Injection vulnerability",
                "high",
                "CWE-89",
            ),
            "xss": (
                r'innerHTML\s*=|document\.write\s*\(|\.html\s*\([^)]*\+',
                "Potential XSS vulnerability",
                "medium",
                "CWE-79",
            ),
            "command_injection": (
                r'os\.system\s*\(|subprocess\.call\s*\([^,]+\+|eval\s*\(',
                "Command injection vulnerability",
                "high",
                "CWE-78",
            ),
            "hardcoded_secret": (
                r'password\s*=\s*["\'][^"\']+["\']|api_key\s*=\s*["\'][^"\']+["\']',
                "Hardcoded credential",
                "high",
                "CWE-798",
            ),
        }

        files = [target] if target.is_file() else target.rglob("*")

        for file_path in files:
            if not file_path.is_file():
                continue
            if any(p in str(file_path) for p in self.exclude_patterns):
                continue
            if file_path.suffix not in [".py", ".js", ".ts", ".php", ".rb", ".go", ".java"]:
                continue

            try:
                content = file_path.read_text(errors='ignore')
                lines = content.split('\n')

                for rule_id, (pattern, message, severity, cwe) in patterns.items():
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append(Finding(
                                file_path=str(file_path),
                                rule_id=f"fallback.{rule_id}",
                                title=message,
                                severity=severity,
                                snippet=line.strip(),
                                line_start=i,
                                line_end=i,
                                message=message,
                                cwe=cwe,
                            ))
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")

        return findings

    def scan_for_specific_rules(
        self,
        target_path: str,
        rule_ids: List[str],
    ) -> List[Finding]:
        """
        Scan for specific vulnerabilities only.

        Args:
            target_path: Path to scan
            rule_ids: List of rule IDs to check

        Returns:
            Filtered findings matching the rule IDs
        """
        all_findings = self.scan(target_path)
        return [f for f in all_findings if f.rule_id in rule_ids]

    def rescan_file(self, file_path: str) -> List[Finding]:
        """
        Rescan a single file (useful after patching).

        Args:
            file_path: Path to the file

        Returns:
            List of findings in that file
        """
        return self.scan(file_path)
