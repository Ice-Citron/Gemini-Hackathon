import subprocess
import os
import time
from dataclasses import dataclass
from src.types import TaskSpec

@dataclass
class VerificationResult:
    passed: bool
    stdout: str
    stderr: str
    runtime_ms: float

def run_verification(task: TaskSpec, timeout_sec: int = 10) -> VerificationResult:
    """
    Runs the verification command in the task's directory.
    Returns PASS only if the return code is 0 (Success).
    """
    start_time = time.time()
    
    try:
        # We run the command INSIDE the workspace directory
        result = subprocess.run(
            task.verify_cmd,
            shell=True,
            cwd=task.workspace_dir,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        
        runtime = (time.time() - start_time) * 1000
        
        return VerificationResult(
            passed=(result.returncode == 0),
            stdout=result.stdout,
            stderr=result.stderr,
            runtime_ms=runtime
        )
        
    except subprocess.TimeoutExpired:
        return VerificationResult(False, "", "Timed out", (time.time() - start_time) * 1000)
    except Exception as e:
        return VerificationResult(False, "", str(e), 0.0)

# Quick self-test if run directly
if __name__ == "__main__":
    print("Verifier module ready. Run integration test to verify.")