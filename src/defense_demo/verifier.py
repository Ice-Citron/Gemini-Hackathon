"""
Security Verifier Module

Verifies that patches are correct using tests and SAST.
This is the "truth gate" - deterministic verification only.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .schemas import Finding, VerifyResult
from .scanner import SecurityScanner


class SecurityVerifier:
    """
    Verifies patches using deterministic checks:
    1. Unit tests (pytest)
    2. SAST rescan (Semgrep)
    3. Optional linting
    """

    def __init__(
        self,
        test_command: Optional[str] = None,
        scanner: Optional[SecurityScanner] = None,
    ):
        """
        Initialize the verifier.

        Args:
            test_command: Command to run tests (default: pytest)
            scanner: Scanner instance for SAST verification
        """
        self.test_command = test_command or "pytest -x -q"
        self.scanner = scanner or SecurityScanner()

    def verify(
        self,
        target_path: str,
        original_findings: List[Finding],
    ) -> VerifyResult:
        """
        Verify that a patch is correct.

        Args:
            target_path: Path to the patched code
            original_findings: The findings that were patched

        Returns:
            VerifyResult with verification status
        """
        errors = []
        warnings = []
        details = {}

        # 1. Run tests
        tests_pass, test_details = self._run_tests(target_path)
        details["tests"] = test_details

        if not tests_pass:
            errors.append(f"Tests failed: {test_details.get('error', 'Unknown error')}")

        # 2. Run SAST rescan
        sast_pass, sast_details = self._run_sast(target_path, original_findings)
        details["sast"] = sast_details

        if not sast_pass:
            warnings.append("SAST still reports findings")

        # 3. Optional: Run linter
        lint_pass, lint_details = self._run_linter(target_path)
        details["linter"] = lint_details

        if not lint_pass:
            warnings.append(f"Linter warnings: {lint_details.get('warning_count', 0)}")

        return VerifyResult(
            tests_pass=tests_pass,
            sast_pass=sast_pass,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    def _run_tests(self, target_path: str) -> tuple[bool, dict]:
        """Run the test suite"""
        try:
            # Try to find test directory
            target = Path(target_path)
            test_dir = None

            for possible in ["tests", "test", "spec"]:
                if (target / possible).exists():
                    test_dir = target / possible
                    break
                if (target.parent / possible).exists():
                    test_dir = target.parent / possible
                    break

            if not test_dir:
                return True, {"status": "skipped", "reason": "No test directory found"}

            # Run tests
            result = subprocess.run(
                self.test_command.split(),
                cwd=str(target.parent) if target.is_file() else str(target),
                capture_output=True,
                text=True,
                timeout=120,
            )

            passed = result.returncode == 0

            return passed, {
                "status": "passed" if passed else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
                "error": result.stderr[:200] if not passed else None,
            }

        except subprocess.TimeoutExpired:
            return False, {"status": "timeout", "error": "Tests timed out after 120s"}
        except FileNotFoundError:
            return True, {"status": "skipped", "reason": "pytest not installed"}
        except Exception as e:
            return False, {"status": "error", "error": str(e)}

    def _run_sast(
        self,
        target_path: str,
        original_findings: List[Finding],
    ) -> tuple[bool, dict]:
        """Run SAST and check if findings are fixed"""
        try:
            # Get the file paths from original findings
            files_to_check = set(f.file_path for f in original_findings)

            # Rescan each file
            remaining_findings = []
            fixed_findings = []

            for file_path in files_to_check:
                if not Path(file_path).exists():
                    continue

                new_findings = self.scanner.rescan_file(file_path)

                # Check if original findings are still present
                original_rules = {f.rule_id for f in original_findings if f.file_path == file_path}
                new_rules = {f.rule_id for f in new_findings}

                for finding in original_findings:
                    if finding.file_path != file_path:
                        continue
                    if finding.rule_id in new_rules:
                        remaining_findings.append(finding)
                    else:
                        fixed_findings.append(finding)

            all_fixed = len(remaining_findings) == 0

            return all_fixed, {
                "status": "clean" if all_fixed else "findings_remain",
                "original_count": len(original_findings),
                "remaining_count": len(remaining_findings),
                "fixed_count": len(fixed_findings),
                "remaining_rules": [f.rule_id for f in remaining_findings],
            }

        except Exception as e:
            return False, {"status": "error", "error": str(e)}

    def _run_linter(self, target_path: str) -> tuple[bool, dict]:
        """Run linter checks (optional)"""
        try:
            target = Path(target_path)

            # Determine linter based on file type
            if target.suffix == ".py" or (target.is_dir() and list(target.glob("*.py"))):
                cmd = ["ruff", "check", str(target), "--quiet"]
            elif target.suffix in [".js", ".ts"]:
                cmd = ["eslint", str(target), "--quiet"]
            else:
                return True, {"status": "skipped", "reason": "No linter for file type"}

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            passed = result.returncode == 0
            warning_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

            return passed, {
                "status": "passed" if passed else "warnings",
                "warning_count": warning_count,
                "output": result.stdout[:300] if result.stdout else "",
            }

        except FileNotFoundError:
            return True, {"status": "skipped", "reason": "Linter not installed"}
        except Exception as e:
            return True, {"status": "skipped", "error": str(e)}

    def quick_verify(self, file_path: str, original_finding: Finding) -> bool:
        """
        Quick verification for a single file/finding.

        Args:
            file_path: Path to the patched file
            original_finding: The finding that was patched

        Returns:
            True if the finding is fixed
        """
        new_findings = self.scanner.rescan_file(file_path)
        return not any(f.rule_id == original_finding.rule_id for f in new_findings)
