"""
Safe Patch Application Module

Applies patches with backup and rollback capabilities.
"""

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from .schemas import PatchProposal


class PatchApplier:
    """
    Safely applies patches with backup and rollback support.

    Uses git apply when available, falls back to Python implementation.
    """

    def __init__(self, backup_dir: Optional[str] = None):
        """
        Initialize the patch applier.

        Args:
            backup_dir: Directory for backups (default: temp directory)
        """
        self.backup_dir = Path(backup_dir) if backup_dir else Path(tempfile.gettempdir()) / "patch_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._backup_stack: list[Tuple[str, str]] = []  # (original_path, backup_path)

    def apply(self, patch: PatchProposal, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Apply a patch to the codebase.

        Args:
            patch: The patch to apply
            dry_run: If True, only test if patch would apply

        Returns:
            Tuple of (success, message)
        """
        if not patch.diff_unified.strip():
            return False, "Empty patch"

        # Create backup of affected files
        if not dry_run:
            for file_path in patch.files_touched:
                self._backup_file(file_path)

        # Try git apply first (most reliable)
        success, message = self._apply_with_git(patch.diff_unified, dry_run)

        if not success:
            # Fallback to Python patch application
            success, message = self._apply_with_python(patch.diff_unified, dry_run)

        if not success and not dry_run:
            # Rollback on failure
            self.rollback()

        return success, message

    def _apply_with_git(self, diff: str, dry_run: bool) -> Tuple[bool, str]:
        """Apply patch using git apply"""
        try:
            # Write diff to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(diff)
                patch_file = f.name

            cmd = ["git", "apply"]
            if dry_run:
                cmd.append("--check")
            cmd.append(patch_file)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            os.unlink(patch_file)

            if result.returncode == 0:
                return True, "Patch applied successfully with git"
            else:
                return False, f"git apply failed: {result.stderr}"

        except FileNotFoundError:
            return False, "git not available"
        except subprocess.TimeoutExpired:
            return False, "git apply timed out"
        except Exception as e:
            return False, f"git apply error: {e}"

    def _apply_with_python(self, diff: str, dry_run: bool) -> Tuple[bool, str]:
        """Apply patch using Python (fallback)"""
        try:
            # Parse unified diff
            changes = self._parse_unified_diff(diff)

            if not changes:
                return False, "Could not parse diff"

            for file_path, hunks in changes.items():
                if not Path(file_path).exists():
                    # Handle new files
                    if not dry_run:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(file_path).write_text(self._reconstruct_new_file(hunks))
                    continue

                # Read original content
                original = Path(file_path).read_text().splitlines(keepends=True)

                # Apply hunks
                modified = self._apply_hunks(original, hunks)

                if modified is None:
                    return False, f"Failed to apply hunks to {file_path}"

                if not dry_run:
                    Path(file_path).write_text(''.join(modified))

            return True, "Patch applied successfully with Python"

        except Exception as e:
            return False, f"Python patch error: {e}"

    def _parse_unified_diff(self, diff: str) -> dict:
        """Parse a unified diff into a structured format"""
        changes = {}
        current_file = None
        current_hunks = []

        for line in diff.splitlines():
            if line.startswith('--- '):
                # Start of a new file
                if current_file and current_hunks:
                    changes[current_file] = current_hunks
                current_hunks = []
            elif line.startswith('+++ '):
                # Target file
                path = line[4:].strip()
                if path.startswith('b/'):
                    path = path[2:]
                current_file = path
            elif line.startswith('@@'):
                # Hunk header
                current_hunks.append({'header': line, 'lines': []})
            elif current_hunks and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                current_hunks[-1]['lines'].append(line)

        if current_file and current_hunks:
            changes[current_file] = current_hunks

        return changes

    def _apply_hunks(self, original: list, hunks: list) -> Optional[list]:
        """Apply hunks to original content"""
        result = original.copy()
        offset = 0

        for hunk in hunks:
            # Parse hunk header to get line numbers
            header = hunk['header']
            # Format: @@ -start,count +start,count @@
            parts = header.split(' ')
            try:
                old_start = int(parts[1].split(',')[0].replace('-', ''))
                new_start = int(parts[2].split(',')[0].replace('+', ''))
            except (IndexError, ValueError):
                return None

            # Apply changes
            idx = old_start - 1 + offset
            for line in hunk['lines']:
                if line.startswith('-'):
                    if idx < len(result):
                        result.pop(idx)
                        offset -= 1
                elif line.startswith('+'):
                    result.insert(idx, line[1:] + '\n')
                    offset += 1
                    idx += 1
                else:  # Context line
                    idx += 1

        return result

    def _reconstruct_new_file(self, hunks: list) -> str:
        """Reconstruct a new file from hunks"""
        lines = []
        for hunk in hunks:
            for line in hunk['lines']:
                if line.startswith('+'):
                    lines.append(line[1:])
        return '\n'.join(lines)

    def _backup_file(self, file_path: str):
        """Create a backup of a file"""
        src = Path(file_path)
        if not src.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src.name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(src, backup_path)
        self._backup_stack.append((str(src), str(backup_path)))

    def rollback(self) -> bool:
        """Rollback all applied patches"""
        success = True
        while self._backup_stack:
            original, backup = self._backup_stack.pop()
            try:
                shutil.copy2(backup, original)
            except Exception as e:
                print(f"Rollback failed for {original}: {e}")
                success = False
        return success

    def clear_backups(self):
        """Clear all backups (call after successful verification)"""
        for _, backup in self._backup_stack:
            try:
                os.unlink(backup)
            except Exception:
                pass
        self._backup_stack.clear()
