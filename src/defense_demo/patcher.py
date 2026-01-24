"""
Patch Generator Module

Uses Gemini to generate security patches from findings.
"""

from pathlib import Path
from typing import Optional

from .schemas import Finding, PatchProposal, GeminiPatchResponse
from .gemini_client import GeminiClient


class PatchGenerator:
    """
    Generates patches for security vulnerabilities using Gemini.
    """

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize the patch generator.

        Args:
            gemini_client: Optional pre-configured Gemini client
        """
        self.gemini = gemini_client or GeminiClient()

    def generate_patch(
        self,
        finding: Finding,
        context: Optional[str] = None,
    ) -> PatchProposal:
        """
        Generate a patch for a security finding.

        Args:
            finding: The vulnerability to fix
            context: Optional additional context

        Returns:
            PatchProposal with unified diff
        """
        # Read the file content
        file_path = Path(finding.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {finding.file_path}")

        file_content = file_path.read_text()

        # Call Gemini to generate patch
        response: GeminiPatchResponse = self.gemini.generate_patch(
            finding=finding,
            file_content=file_content,
            context=context,
        )

        # Add finding reference to patch
        patch = response.patch
        patch.finding_id = finding.rule_id

        return patch

    def refine_patch(
        self,
        finding: Finding,
        failed_patch: PatchProposal,
        error_message: str,
    ) -> PatchProposal:
        """
        Refine a patch that failed to apply or verify.

        Args:
            finding: Original finding
            failed_patch: The patch that failed
            error_message: Why it failed

        Returns:
            Refined PatchProposal
        """
        # Read current file content
        file_path = Path(finding.file_path)
        file_content = file_path.read_text() if file_path.exists() else ""

        # Call Gemini to refine
        response: GeminiPatchResponse = self.gemini.refine_patch(
            finding=finding,
            previous_patch=failed_patch,
            error_message=error_message,
            file_content=file_content,
        )

        return response.patch

    def generate_patches_for_findings(
        self,
        findings: list[Finding],
        max_patches: int = 10,
    ) -> list[PatchProposal]:
        """
        Generate patches for multiple findings.

        Args:
            findings: List of findings to patch
            max_patches: Maximum number of patches to generate

        Returns:
            List of patch proposals
        """
        patches = []

        for finding in findings[:max_patches]:
            try:
                patch = self.generate_patch(finding)
                patches.append(patch)
            except Exception as e:
                print(f"Failed to generate patch for {finding.rule_id}: {e}")

        return patches
