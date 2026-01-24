"""
Gemini API Client with Structured Outputs

Uses GDM's OpenAI-compatible REST API for structured patch generation.
https://api.x.ai/v1 with OpenAI SDK compatibility.
"""

import os
import json
from typing import Any, Dict, Optional, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel

from .schemas import Finding, PatchProposal, GeminiPatchResponse, GeminiAnalysisResponse

T = TypeVar('T', bound=BaseModel)


class GeminiClient:
    """
    Gemini API client using GDM's OpenAI-compatible endpoint.

    Uses structured outputs via response_format for reliable JSON parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-latest",
        base_url: str = "https://api.x.ai/v1",
    ):
        self.api_key = api_key or os.environ.get("GDM_API_KEY")
        if not self.api_key:
            raise ValueError("GDM_API_KEY is required. Set it in environment or pass to constructor.")

        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

    def _call_with_schema(
        self,
        messages: list,
        response_schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> T:
        """
        Make a Gemini API call with structured output schema.

        Args:
            messages: Chat messages
            response_schema: Pydantic model for response validation
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Parsed and validated response as Pydantic model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                }
            }
        )

        content = response.choices[0].message.content
        return response_schema.model_validate_json(content)

    def generate_patch(
        self,
        finding: Finding,
        file_content: str,
        context: Optional[str] = None,
    ) -> GeminiPatchResponse:
        """
        Generate a patch proposal for a security finding.

        Args:
            finding: The security vulnerability to fix
            file_content: Current content of the file
            context: Optional additional context about the codebase

        Returns:
            GeminiPatchResponse with patch proposal
        """
        system_prompt = """You are a security engineer expert at fixing vulnerabilities.
Your task is to generate a minimal, safe patch that fixes the security issue.

RULES:
1. Modify as little code as possible
2. Do NOT change behavior except to fix the vulnerability
3. Do NOT add new dependencies
4. Keep formatting stable (match existing style)
5. Output a valid unified diff format
6. Explain your reasoning clearly"""

        user_prompt = f"""Fix this security vulnerability:

## Finding
- File: {finding.file_path}
- Rule: {finding.rule_id}
- Title: {finding.title}
- Severity: {finding.severity}
- Lines: {finding.line_start}-{finding.line_end}
- Message: {finding.message}

## Vulnerable Code Snippet
```
{finding.snippet}
```

## Full File Content
```
{file_content}
```

{f"## Additional Context\n{context}" if context else ""}

Generate a patch to fix this vulnerability. Return a unified diff."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self._call_with_schema(messages, GeminiPatchResponse)

    def analyze_vulnerability(
        self,
        finding: Finding,
        file_content: str,
    ) -> GeminiAnalysisResponse:
        """
        Analyze a vulnerability in detail.

        Args:
            finding: The security finding to analyze
            file_content: Content of the vulnerable file

        Returns:
            GeminiAnalysisResponse with detailed analysis
        """
        system_prompt = """You are a security analyst expert at vulnerability assessment.
Analyze the given vulnerability and provide a detailed assessment."""

        user_prompt = f"""Analyze this vulnerability:

## Finding
- File: {finding.file_path}
- Rule: {finding.rule_id}
- Title: {finding.title}
- Severity: {finding.severity}
- Lines: {finding.line_start}-{finding.line_end}

## Code
```
{finding.snippet}
```

## Full File
```
{file_content[:2000]}...
```

Provide a detailed analysis including root cause, impact, and fix strategy."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self._call_with_schema(messages, GeminiAnalysisResponse)

    def refine_patch(
        self,
        finding: Finding,
        previous_patch: PatchProposal,
        error_message: str,
        file_content: str,
    ) -> GeminiPatchResponse:
        """
        Refine a patch that failed verification.

        Args:
            finding: Original finding
            previous_patch: The patch that failed
            error_message: Why it failed
            file_content: Current file content

        Returns:
            GeminiPatchResponse with refined patch
        """
        system_prompt = """You are a security engineer fixing a failed patch.
The previous patch attempt failed. Generate a corrected version.

RULES:
1. Understand why the previous patch failed
2. Fix the issues while still addressing the vulnerability
3. Keep changes minimal
4. Ensure the diff applies cleanly"""

        user_prompt = f"""Previous patch failed. Please fix it.

## Original Finding
- File: {finding.file_path}
- Rule: {finding.rule_id}
- Title: {finding.title}

## Previous Patch (FAILED)
```diff
{previous_patch.diff_unified}
```

## Error Message
{error_message}

## Current File Content
```
{file_content}
```

Generate a corrected patch that addresses both the vulnerability and the error."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self._call_with_schema(messages, GeminiPatchResponse)

    def test_connection(self) -> bool:
        """Test that the Gemini API is accessible"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'connected' if you can read this."}],
                max_tokens=10,
            )
            return "connected" in response.choices[0].message.content.lower()
        except Exception as e:
            print(f"Gemini connection test failed: {e}")
            return False


# Convenience function for quick testing
def call_gemini(prompt: str, api_key: Optional[str] = None) -> str:
    """Quick helper to call Gemini with a simple prompt"""
    client = GeminiClient(api_key=api_key)
    response = client.client.chat.completions.create(
        model=client.model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
