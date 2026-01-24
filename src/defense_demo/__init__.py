# Defense Demo Module - Gemini-powered scan/patch/verify loop
from .schemas import Finding, PatchProposal, VerifyResult, RunTrace
from .gemini_client import GeminiClient
from .scanner import SecurityScanner
from .patcher import PatchGenerator
from .apply_patch import PatchApplier
from .verifier import SecurityVerifier
from .orchestrator import DefenseOrchestrator

__all__ = [
    "Finding",
    "PatchProposal",
    "VerifyResult",
    "RunTrace",
    "GeminiClient",
    "SecurityScanner",
    "PatchGenerator",
    "PatchApplier",
    "SecurityVerifier",
    "DefenseOrchestrator",
]
