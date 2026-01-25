#!/usr/bin/env python3
"""
SkyHammer Logging Utilities
Session logging to runs/ directory
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Global logging state
RUN_DIR: Optional[str] = None
LOG_FILE: Optional[str] = None


def init_logging(model: str, workspace: str, base_path: str = None):
    """Initialize logging directory for this session"""
    global RUN_DIR, LOG_FILE

    # Create runs directory if it doesn't exist
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_base = os.path.join(base_path, "runs")
    os.makedirs(runs_base, exist_ok=True)

    # Create session directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_DIR = os.path.join(runs_base, f"gemini_code_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)

    # Create log file
    LOG_FILE = os.path.join(RUN_DIR, "session.jsonl")

    # Log session start
    log_event("session_start", {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "workspace": workspace
    })

    return RUN_DIR


def log_event(event_type: str, data: Dict[str, Any]):
    """Log an event to the session log file"""
    global LOG_FILE
    if not LOG_FILE:
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **data
    }

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Don't crash on logging errors


def log_tool_call(tool_name: str, args: Dict[str, Any], result: str, duration_ms: float = 0):
    """Log a tool call"""
    log_event("tool_call", {
        "tool": tool_name,
        "arguments": args,
        "result_preview": result[:500] if result else "",
        "result_length": len(result) if result else 0,
        "duration_ms": duration_ms
    })


def log_api_call(model: str, messages_count: int, response_content: str, tool_calls: List[str] = None):
    """Log an API call to Gemini"""
    log_event("api_call", {
        "model": model,
        "messages_count": messages_count,
        "response_preview": response_content[:500] if response_content else "",
        "tool_calls": tool_calls or []
    })


def log_conversation(role: str, content: str):
    """Log a conversation message"""
    log_event("conversation", {
        "role": role,
        "content": content[:2000] if content else ""
    })


def get_run_dir() -> Optional[str]:
    """Get the current run directory"""
    return RUN_DIR


def save_report(report_content: str, target: str, report_type: str = "skyhammer") -> Optional[str]:
    """
    Save a SkyHammer report to the session log directory.

    Args:
        report_content: The full report text
        target: The target that was scanned (file path or URL)
        report_type: Type of report (skyhammer, vulnerability, patch)

    Returns:
        Path to the saved report file, or None if logging not initialized
    """
    global RUN_DIR
    if not RUN_DIR:
        return None

    # Create reports subdirectory
    reports_dir = os.path.join(RUN_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Generate filename from target
    timestamp = datetime.now().strftime("%H-%M-%S")
    target_name = os.path.basename(target) if not target.startswith("http") else target.replace("://", "_").replace("/", "_")[:50]
    filename = f"{report_type}_{target_name}_{timestamp}.txt"
    filepath = os.path.join(reports_dir, filename)

    # Save report
    try:
        with open(filepath, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"SKYHAMMER REPORT\n")
            f.write(f"{'='*60}\n")
            f.write(f"Target: {target}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(report_content)

        # Also log the event
        log_event("report_saved", {
            "report_type": report_type,
            "target": target,
            "filepath": filepath
        })

        return filepath
    except Exception as e:
        return None
