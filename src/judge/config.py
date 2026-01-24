"""
Configuration for the LLM-judge agent

Supports both Anthropic (Claude) and GDM (Gemini) as judge models.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Load secrets from secretsConfig.py if available
try:
    from secretsConfig import ANTHROPIC_API_KEY, GDM_API_KEY
    os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    os.environ.setdefault("GDM_API_KEY", GDM_API_KEY)
except ImportError:
    pass  # Fall back to environment variable


class JudgeConfig(BaseModel):
    """Configuration for the LLM-judge agent"""

    # API settings - supports both Anthropic and GDM
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    xai_api_key: str = Field(
        default_factory=lambda: os.getenv("GDM_API_KEY", "")
    )

    # Model selection - "claude" or "gemini"
    judge_provider: str = Field(
        default_factory=lambda: os.getenv("JUDGE_PROVIDER", "claude")
    )
    model: str = Field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "claude-sonnet-4-20250514")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3-latest")
    )

    max_tokens: int = 4096
    temperature: float = 0.0  # Use 0 for deterministic judging

    # MCP server settings
    mcp_server_command: str = Field(
        default="python -m src.mcp_server.server",
        description="Command to start the MCP server"
    )

    # Judge behavior settings
    max_tool_calls: int = 10  # Maximum tool calls per evaluation
    strict_mode: bool = True  # If True, requires explicit verification of all criteria

    def validate_config(self) -> bool:
        """Validate that required configuration is present"""
        if self.judge_provider == "claude" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using Claude as judge")
        if self.judge_provider == "gemini" and not self.xai_api_key:
            raise ValueError("GDM_API_KEY is required when using Gemini as judge")
        return True
