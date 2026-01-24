"""
Defense Orchestrator

Main loop: scan -> patch -> apply -> verify
With iteration support for failed patches.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import Finding, PatchProposal, VerifyResult, RunTrace
from .gemini_client import GeminiClient
from .scanner import SecurityScanner
from .patcher import PatchGenerator
from .apply_patch import PatchApplier
from .verifier import SecurityVerifier


class DefenseOrchestrator:
    """
    Orchestrates the complete defense loop:
    1. Scan for vulnerabilities
    2. Generate patches using Gemini
    3. Apply patches safely
    4. Verify with tests and SAST
    5. Iterate if verification fails (max 2 iterations)
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        scanner: Optional[SecurityScanner] = None,
        max_iterations: int = 2,
        runs_dir: str = "runs",
    ):
        """
        Initialize the orchestrator.

        Args:
            gemini_client: Gemini client for patch generation
            scanner: Security scanner instance
            max_iterations: Maximum patch refinement iterations
            runs_dir: Directory to store run traces
        """
        self.gemini = gemini_client or GeminiClient()
        self.scanner = scanner or SecurityScanner()
        self.patcher = PatchGenerator(self.gemini)
        self.applier = PatchApplier()
        self.verifier = SecurityVerifier(scanner=self.scanner)
        self.max_iterations = max_iterations
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)

    def run(self, target_path: str) -> RunTrace:
        """
        Run the complete defense loop on a target.

        Args:
            target_path: Path to file or directory to defend

        Returns:
            RunTrace with complete execution history
        """
        run_id = str(uuid.uuid4())[:8]
        trace = RunTrace(
            id=run_id,
            target_path=target_path,
        )

        print(f"\n{'='*60}")
        print(f"Defense Run: {run_id}")
        print(f"Target: {target_path}")
        print(f"{'='*60}\n")

        # Step 1: Scan
        print("[1/4] Scanning for vulnerabilities...")
        findings = self.scanner.scan(target_path)
        trace.findings = findings
        trace.findings_count = len(findings)

        if not findings:
            print("No vulnerabilities found!")
            trace.success = True
            trace.final_score = 1.0
            trace.summary = "No vulnerabilities detected"
            trace.completed_at = datetime.now()
            self._save_trace(trace)
            return trace

        print(f"Found {len(findings)} vulnerabilities:")
        for f in findings[:5]:  # Show first 5
            print(f"  - [{f.severity.upper()}] {f.title} in {f.file_path}:{f.line_start}")
        if len(findings) > 5:
            print(f"  ... and {len(findings) - 5} more")

        # Process each finding
        fixed_count = 0
        for i, finding in enumerate(findings):
            print(f"\n[2/4] Processing finding {i+1}/{len(findings)}: {finding.rule_id}")

            success = self._process_finding(finding, trace)
            if success:
                fixed_count += 1

        # Final summary
        trace.success = fixed_count == len(findings)
        trace.final_score = fixed_count / len(findings) if findings else 1.0
        trace.summary = f"Fixed {fixed_count}/{len(findings)} vulnerabilities"
        trace.completed_at = datetime.now()

        print(f"\n{'='*60}")
        print(f"Run Complete: {run_id}")
        print(f"Result: {'SUCCESS' if trace.success else 'PARTIAL'}")
        print(f"Fixed: {fixed_count}/{len(findings)} ({trace.final_score:.0%})")
        print(f"{'='*60}\n")

        self._save_trace(trace)
        return trace

    def _process_finding(self, finding: Finding, trace: RunTrace) -> bool:
        """Process a single finding with iteration support"""
        iteration = 0
        last_error = None

        while iteration < self.max_iterations:
            iteration += 1
            trace.iterations += 1
            print(f"  Iteration {iteration}/{self.max_iterations}")

            # Generate or refine patch
            try:
                if iteration == 1:
                    print("  Generating patch with Gemini...")
                    patch = self.patcher.generate_patch(finding)
                else:
                    print("  Refining patch...")
                    patch = self.patcher.refine_patch(finding, patch, last_error)

                trace.patches.append(patch)
                print(f"  Patch confidence: {patch.confidence:.0%}")

            except Exception as e:
                print(f"  Patch generation failed: {e}")
                last_error = str(e)
                continue

            # Apply patch
            print("  Applying patch...")
            success, message = self.applier.apply(patch)

            if not success:
                print(f"  Apply failed: {message}")
                last_error = message
                continue

            # Verify
            print("  Verifying...")
            verify_result = self.verifier.verify(
                finding.file_path,
                [finding],
            )
            trace.verify_results.append(verify_result)

            if verify_result.success:
                print(f"  SUCCESS: {finding.rule_id} fixed!")
                self.applier.clear_backups()
                return True
            else:
                print(f"  Verification failed: {verify_result.errors}")
                last_error = "; ".join(verify_result.errors)
                self.applier.rollback()

        print(f"  FAILED: Could not fix {finding.rule_id} after {self.max_iterations} iterations")
        return False

    def _save_trace(self, trace: RunTrace):
        """Save run trace to disk"""
        trace_path = self.runs_dir / f"{trace.id}.json"
        trace.to_json_file(str(trace_path))
        print(f"Trace saved to: {trace_path}")

    def run_single_finding(self, finding: Finding) -> tuple[bool, VerifyResult]:
        """
        Process a single finding (for testing/demo).

        Args:
            finding: The finding to fix

        Returns:
            Tuple of (success, verification_result)
        """
        trace = RunTrace(
            id=str(uuid.uuid4())[:8],
            target_path=finding.file_path,
            findings=[finding],
        )

        success = self._process_finding(finding, trace)

        return success, trace.verify_results[-1] if trace.verify_results else None

    def dry_run(self, target_path: str) -> list[Finding]:
        """
        Scan only, no patches applied (useful for previewing).

        Args:
            target_path: Path to scan

        Returns:
            List of findings
        """
        print(f"Dry run scan of: {target_path}")
        findings = self.scanner.scan(target_path)
        print(f"Found {len(findings)} vulnerabilities")
        return findings
