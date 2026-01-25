from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TaskSpec:
    task_id: str
    workspace_dir: str  # Path to the specific task folder (e.g., tasks/train/sqli_01)
    target_files: List[str]  # Files the AI is allowed to modify
    verify_cmd: str  # Command to run the exploit/test (e.g., "pytest test_exploit.py")
    issue_summary: str # The prompt for the AI