#!/usr/bin/env python3
"""
SkyHammer Run Logger

Creates timestamped run directories with full audit trails:
- api_calls.jsonl     - Every Gemini API call
- tool_calls.jsonl    - Every tool invocation
- transcript.json     - Full conversation
- summary.md          - Gemini-generated exploit writeup

Usage:
    from run_logger import RunLogger

    logger = RunLogger("sqli_attack")
    logger.log_api_call(request, response)
    logger.log_tool_call("nmap", args, result)
    logger.save_transcript(messages)
    await logger.generate_summary(client)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

# Base directory for all runs
RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class APICallLog:
    """Log entry for an API call"""
    timestamp: str
    model: str
    messages_count: int
    tools_count: int
    response_id: str
    tokens_prompt: int
    tokens_completion: int
    finish_reason: str
    tool_calls: List[str]
    duration_ms: float


@dataclass
class ToolCallLog:
    """Log entry for a tool invocation"""
    timestamp: str
    tool_name: str
    arguments: Dict[str, Any]
    result_preview: str
    result_length: int
    success: bool
    duration_ms: float


class RunLogger:
    """
    Manages logging for a single attack run.
    Creates a timestamped directory with full audit trail.
    """

    def __init__(self, run_name: str, challenge_id: str = "unknown"):
        """
        Initialize a new run logger.

        Args:
            run_name: Descriptive name for the run
            challenge_id: ID of the challenge being attacked
        """
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = run_name
        self.challenge_id = challenge_id

        # Create run directory
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
        self.run_dir = RUNS_DIR / f"{self.timestamp}_{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths
        self.api_log_path = self.run_dir / "api_calls.jsonl"
        self.tool_log_path = self.run_dir / "tool_calls.jsonl"
        self.transcript_path = self.run_dir / "transcript.json"
        self.summary_path = self.run_dir / "summary.md"
        self.metadata_path = self.run_dir / "metadata.json"

        # In-memory logs
        self.api_calls: List[APICallLog] = []
        self.tool_calls: List[ToolCallLog] = []
        self.messages: List[Dict] = []

        # Write initial metadata
        self._write_metadata()

        print(f"[Logger] Run directory: {self.run_dir}")

    def _write_metadata(self):
        """Write run metadata"""
        metadata = {
            "run_name": self.run_name,
            "challenge_id": self.challenge_id,
            "start_time": self.start_time.isoformat(),
            "run_dir": str(self.run_dir),
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def log_api_call(
        self,
        model: str,
        messages: List[Dict],
        tools: List[Dict],
        response: Any,
        duration_ms: float
    ):
        """
        Log a Gemini API call.

        Args:
            model: Model used
            messages: Messages sent
            tools: Tools provided
            response: API response object
            duration_ms: Call duration in milliseconds
        """
        try:
            choice = response.choices[0]
            tool_call_names = []
            if choice.message.tool_calls:
                tool_call_names = [tc.function.name for tc in choice.message.tool_calls]

            log_entry = APICallLog(
                timestamp=datetime.now().isoformat(),
                model=model,
                messages_count=len(messages),
                tools_count=len(tools),
                response_id=response.id if hasattr(response, 'id') else "unknown",
                tokens_prompt=response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                tokens_completion=response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                finish_reason=choice.finish_reason,
                tool_calls=tool_call_names,
                duration_ms=duration_ms
            )

            self.api_calls.append(log_entry)

            # Append to JSONL file
            with open(self.api_log_path, "a") as f:
                f.write(json.dumps(asdict(log_entry)) + "\n")

        except Exception as e:
            print(f"[Logger] Warning: Failed to log API call: {e}")

    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        duration_ms: float,
        success: bool = True
    ):
        """
        Log a tool invocation.

        Args:
            tool_name: Name of the tool
            arguments: Arguments passed to the tool
            result: Tool output
            duration_ms: Execution duration in milliseconds
            success: Whether the tool succeeded
        """
        try:
            log_entry = ToolCallLog(
                timestamp=datetime.now().isoformat(),
                tool_name=tool_name,
                arguments=arguments,
                result_preview=result[:500] if result else "",
                result_length=len(result) if result else 0,
                success=success,
                duration_ms=duration_ms
            )

            self.tool_calls.append(log_entry)

            # Append to JSONL file
            with open(self.tool_log_path, "a") as f:
                f.write(json.dumps(asdict(log_entry)) + "\n")

        except Exception as e:
            print(f"[Logger] Warning: Failed to log tool call: {e}")

    def save_transcript(self, messages: List[Dict]):
        """
        Save the full conversation transcript.

        Args:
            messages: List of conversation messages
        """
        self.messages = messages

        # Clean messages for JSON serialization
        clean_messages = []
        for msg in messages:
            clean_msg = {
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", ""),
            }
            if "tool_calls" in msg and msg["tool_calls"]:
                clean_msg["tool_calls"] = [
                    {
                        "name": tc.get("function", {}).get("name", "") if isinstance(tc, dict)
                               else tc.function.name,
                        "arguments": tc.get("function", {}).get("arguments", "") if isinstance(tc, dict)
                                    else tc.function.arguments
                    }
                    for tc in msg["tool_calls"]
                ]
            if "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            clean_messages.append(clean_msg)

        transcript = {
            "run_name": self.run_name,
            "challenge_id": self.challenge_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_api_calls": len(self.api_calls),
            "total_tool_calls": len(self.tool_calls),
            "messages": clean_messages
        }

        with open(self.transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)

    async def generate_summary(self, client, model: str = "gemini-4-1-fast-reasoning"):
        """
        Use Gemini to generate a summary of the attack.

        Args:
            client: OpenAI client
            model: Model to use for summary
        """
        # Build context for summary
        tool_usage = {}
        for tc in self.tool_calls:
            tool_usage[tc.tool_name] = tool_usage.get(tc.tool_name, 0) + 1

        successful_tools = [tc for tc in self.tool_calls if tc.success]

        # Extract key findings from messages
        findings = []
        for msg in self.messages:
            content = msg.get("content", "")
            if content and msg.get("role") == "assistant":
                if len(content) > 100:
                    findings.append(content[:500])

        prompt = f"""You are a security researcher writing a post-engagement report.

## Attack Summary Request

**Challenge:** {self.challenge_id}
**Run Name:** {self.run_name}
**Duration:** {self.start_time.isoformat()} to {datetime.now().isoformat()}

**Tools Used:**
{json.dumps(tool_usage, indent=2)}

**Total API Calls:** {len(self.api_calls)}
**Total Tool Invocations:** {len(self.tool_calls)}
**Successful Tool Calls:** {len(successful_tools)}

**Key Agent Observations:**
{chr(10).join(findings[:5])}

---

Write a professional security assessment summary in Markdown format. Include:

1. **Executive Summary** - One paragraph overview
2. **Methodology** - What tools and techniques were used
3. **Findings** - What vulnerabilities were discovered
4. **Exploitation Chain** - Step-by-step how the exploit worked
5. **Recommendations** - How to fix the vulnerabilities
6. **Tools Effectiveness** - Which tools were most useful

Be concise but thorough. This report is for developers to understand how the vulnerability was found.
"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional security researcher writing engagement reports."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            summary = response.choices[0].message.content

            # Write summary with header
            with open(self.summary_path, "w") as f:
                f.write(f"# SkyHammer Security Assessment Report\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n")
                f.write(f"**Challenge:** {self.challenge_id}\n")
                f.write(f"**Run Directory:** `{self.run_dir}`\n\n")
                f.write("---\n\n")
                f.write(summary)

            print(f"[Logger] Summary saved to {self.summary_path}")
            return summary

        except Exception as e:
            print(f"[Logger] Warning: Failed to generate summary: {e}")

            # Write a basic summary
            with open(self.summary_path, "w") as f:
                f.write(f"# SkyHammer Run Report\n\n")
                f.write(f"**Challenge:** {self.challenge_id}\n")
                f.write(f"**Start:** {self.start_time.isoformat()}\n")
                f.write(f"**API Calls:** {len(self.api_calls)}\n")
                f.write(f"**Tool Calls:** {len(self.tool_calls)}\n\n")
                f.write("## Tools Used\n\n")
                for name, count in tool_usage.items():
                    f.write(f"- {name}: {count} calls\n")

            return None

    def finalize(self):
        """
        Finalize the run and update metadata.
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # Update metadata
        metadata = {
            "run_name": self.run_name,
            "challenge_id": self.challenge_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_api_calls": len(self.api_calls),
            "total_tool_calls": len(self.tool_calls),
            "run_dir": str(self.run_dir),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Logger] Run finalized. Duration: {duration:.1f}s")
        print(f"[Logger] Logs saved to: {self.run_dir}")

    def get_run_path(self) -> Path:
        """Return the run directory path"""
        return self.run_dir


def list_runs() -> List[Path]:
    """List all previous runs"""
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.iterdir(), reverse=True)


if __name__ == "__main__":
    # Demo
    print("SkyHammer Run Logger")
    print(f"Runs directory: {RUNS_DIR}")

    runs = list_runs()
    if runs:
        print(f"\nPrevious runs ({len(runs)}):")
        for run in runs[:5]:
            print(f"  - {run.name}")
    else:
        print("\nNo previous runs found.")
