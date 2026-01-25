import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from src.core.types import TaskSpec

@dataclass
class PatchResult:
    success: bool
    message: str

class PatchManager:
    def __init__(self, task: TaskSpec):
        self.task = task
        self.backup_dir = os.path.join(task.workspace_dir, ".skyhammer_backup")

    def create_backup(self):
        """Creates a backup of all target files before applying changes."""
        if os.path.exists(self.backup_dir):
            shutil.rmtree(self.backup_dir)
        os.makedirs(self.backup_dir)
        
        for filename in self.task.target_files:
            src = os.path.join(self.task.workspace_dir, filename)
            dst = os.path.join(self.backup_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    def restore_backup(self):
        """Restores files from backup, effectively undoing the patch."""
        if not os.path.exists(self.backup_dir):
            return
            
        for filename in self.task.target_files:
            src = os.path.join(self.backup_dir, filename)
            dst = os.path.join(self.task.workspace_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Cleanup
        shutil.rmtree(self.backup_dir)

    def apply_diff(self, diff_content: str) -> PatchResult:
        """
        Applies a unified diff string to the workspace.
        """
        self.create_backup()

        # Write diff to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.diff', delete=False) as tmp:
            tmp.write(diff_content)
            tmp_path = tmp.name

        try:
            # Use git apply as it's robust for unified diffs
            cmd = ["git", "apply", "--ignore-space-change", "--ignore-whitespace", tmp_path]
            
            result = subprocess.run(
                cmd,
                cwd=self.task.workspace_dir,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return PatchResult(True, "Patch applied successfully")
            else:
                self.restore_backup() # Fail safe
                return PatchResult(False, f"Git Apply Failed: {result.stderr}")

        except Exception as e:
            self.restore_backup()
            return PatchResult(False, str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    print("PatchManager ready.")