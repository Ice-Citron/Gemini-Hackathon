"""
Training Configuration for Security Agent RLAIF

Centralized configuration for all training parameters.
Enhanced for GDM Geminiathon with Gemini model support.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Load secrets from secretsConfig.py
try:
    from secretsConfig import (
        ANTHROPIC_API_KEY,
        GDM_API_KEY,
        WANDB_API_KEY,
        HF_TOKEN,
    )
    # Set environment variables from secrets
    os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    os.environ.setdefault("GDM_API_KEY", GDM_API_KEY)
    os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
except ImportError:
    print("Warning: secretsConfig.py not found. Using environment variables.")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    GDM_API_KEY = os.environ.get("GDM_API_KEY", "")
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")


@dataclass
class TrainingConfig:
    """Centralized training configuration"""

    # === Model Configuration ===
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    model_name: str = "security-agent-v1"

    # === Gemini Configuration (for Geminiathon) ===
    use_gemini: bool = True  # Use Gemini as the primary model
    gemini_model: str = "gemini-3-latest"

    # === Training Loop ===
    groups_per_step: int = 2          # Challenges per training step
    rollouts_per_group: int = 10      # Trajectories per challenge
    num_epochs: int = 3
    max_steps: int = 100

    # === LoRA Configuration ===
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5

    # === CAI/Rollout Configuration ===
    max_tool_calls: int = 50          # Max tool calls per rollout
    max_rollout_failures: int = 5     # Failures before skipping group
    cai_path: str = field(default_factory=lambda: os.environ.get(
        "CAI_PATH", "/workspace/main_dir/cai_env"
    ))

    # === Target Configuration ===
    dvwa_url: str = field(default_factory=lambda: os.environ.get(
        "DVWA_URL", "http://31.97.117.123"
    ))

    # === Judge Configuration ===
    judge_url: str = "http://localhost:8088"
    judge_model: str = "claude-sonnet-4-20250514"
    judge_max_tool_calls: int = 20

    # === W&B Configuration ===
    wandb_project: str = "geminiathon-security-agent"
    wandb_entity: Optional[str] = None

    # === Checkpointing ===
    checkpoint_every: int = 10
    hf_repo_id: str = "geminiathon/athenaguard-security-agent"
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    # === GPU Configuration ===
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    load_in_4bit: bool = True
    max_seq_length: int = 8192

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for W&B logging"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in ['hf_token']
        }

    def validate(self) -> bool:
        """Validate configuration"""
        errors = []

        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GDM_API_KEY"):
            errors.append("Either ANTHROPIC_API_KEY or GDM_API_KEY is required")

        if not os.environ.get("WANDB_API_KEY"):
            errors.append("WANDB_API_KEY environment variable not set")

        if self.hf_repo_id and not self.hf_token:
            errors.append("HF_TOKEN required for HuggingFace push")

        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

        return True


# === Pre-configured Profiles ===

H100_CONFIG = TrainingConfig(
    # Optimized for single H100 80GB
    groups_per_step=4,
    rollouts_per_group=10,
    max_steps=200,
    gpu_memory_utilization=0.85,
    load_in_4bit=True,
    checkpoint_every=10,
)

DEV_CONFIG = TrainingConfig(
    # Minimal config for testing
    groups_per_step=1,
    rollouts_per_group=2,
    max_steps=3,
    max_tool_calls=10,
    checkpoint_every=1,
    model_name="security-agent-dev",
)

TINY_TEST_CONFIG = TrainingConfig(
    # Single rollout test
    groups_per_step=1,
    rollouts_per_group=1,
    max_steps=1,
    max_tool_calls=5,
    checkpoint_every=1,
    model_name="security-agent-tiny-test",
)
