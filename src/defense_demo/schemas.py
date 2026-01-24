"""
Pydantic Schemas for Defense Demo

Strict type definitions for the scan -> patch -> verify loop.
These schemas enable structured outputs from Gemini API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A security vulnerability finding from the scanner"""
    file_path: str = Field(..., description="Path to the vulnerable file")
    rule_id: str = Field(..., description="Identifier for the security rule triggered")
    title: str = Field(..., description="Human-readable title of the finding")
    severity: str = Field(..., description="Severity level: critical, high, medium, low, info")
    snippet: str = Field(..., description="Code snippet showing the vulnerability")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    message: str = Field("", description="Detailed explanation of the vulnerability")
    cwe: Optional[str] = Field(None, description="CWE identifier if applicable")
    owasp: Optional[str] = Field(None, description="OWASP category if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "src/app.py",
                "rule_id": "python.sqli.raw-query",
                "title": "SQL Injection vulnerability",
                "severity": "high",
                "snippet": "cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")",
                "line_start": 42,
                "line_end": 42,
                "message": "User input directly interpolated into SQL query",
                "cwe": "CWE-89",
                "owasp": "A03:2021-Injection"
            }
        }


class PatchProposal(BaseModel):
    """A proposed patch to fix a vulnerability"""
    diff_unified: str = Field(..., description="Unified diff format patch")
    rationale: str = Field(..., description="Explanation of why this fix works")
    risks: List[str] = Field(default_factory=list, description="Potential risks of applying this patch")
    files_touched: List[str] = Field(default_factory=list, description="Files modified by this patch")
    finding_id: Optional[str] = Field(None, description="ID of the finding this patch addresses")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence score for this patch")

    class Config:
        json_schema_extra = {
            "example": {
                "diff_unified": """--- a/src/app.py
+++ b/src/app.py
@@ -40,3 +40,3 @@
-    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
+    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
""",
                "rationale": "Use parameterized queries to prevent SQL injection",
                "risks": ["Query parameters may need type conversion"],
                "files_touched": ["src/app.py"],
                "confidence": 0.95
            }
        }


class VerifyResult(BaseModel):
    """Result of verifying a patch"""
    tests_pass: bool = Field(..., description="Whether all tests passed")
    sast_pass: bool = Field(..., description="Whether SAST scan is clean/improved")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed verification results")
    artifact_paths: List[str] = Field(default_factory=list, description="Paths to verification artifacts")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")

    @property
    def success(self) -> bool:
        """Overall verification success"""
        return self.tests_pass and self.sast_pass and len(self.errors) == 0


class RunTrace(BaseModel):
    """Complete trace of a defense run"""
    id: str = Field(..., description="Unique run identifier")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Input
    target_path: str = Field(..., description="Path to target code/repo")

    # Findings
    findings: List[Finding] = Field(default_factory=list)
    findings_count: int = Field(0)

    # Patch attempts
    patches: List[PatchProposal] = Field(default_factory=list)
    iterations: int = Field(0, description="Number of patch iterations")

    # Verification
    verify_results: List[VerifyResult] = Field(default_factory=list)

    # Final status
    success: bool = Field(False)
    final_score: float = Field(0.0, ge=0.0, le=1.0)
    summary: str = Field("")

    def to_json_file(self, path: str):
        """Save trace to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.model_dump(mode='json'), f, indent=2, default=str)

    @classmethod
    def from_json_file(cls, path: str) -> "RunTrace":
        """Load trace from JSON file"""
        import json
        with open(path, 'r') as f:
            return cls.model_validate(json.load(f))


# =============================================================================
# Gemini Structured Output Schemas (for API calls)
# =============================================================================

class GeminiPatchResponse(BaseModel):
    """Schema for Gemini's patch generation response"""
    thinking: str = Field(..., description="Gemini's reasoning about the fix")
    patch: PatchProposal = Field(..., description="The proposed patch")
    alternative_approaches: List[str] = Field(
        default_factory=list,
        description="Other possible approaches considered"
    )


class GeminiAnalysisResponse(BaseModel):
    """Schema for Gemini's vulnerability analysis response"""
    vulnerability_summary: str = Field(..., description="Summary of the vulnerability")
    root_cause: str = Field(..., description="Root cause analysis")
    impact: str = Field(..., description="Potential impact if exploited")
    fix_strategy: str = Field(..., description="Recommended fix strategy")
    severity_assessment: str = Field(..., description="Severity assessment with reasoning")
