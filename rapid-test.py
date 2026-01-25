import os
import subprocess
import difflib
from src.types import TaskSpec
from src.apply_patch import PatchManager

# 1. SETUP PATHS
workspace = os.path.abspath("tasks/test/dummy_task")
app_path = os.path.join(workspace, "app.py")
patch_path = os.path.join(workspace, "fix.patch")
os.makedirs(workspace, exist_ok=True)

# 2. DEFINE CONTENT
original_code = """# Vulnerable code
def check_password(input_pass):
    # Hardcoded = Bad
    return input_pass == "secret123"

if __name__ == "__main__":
    pass
"""

fixed_code = """# Vulnerable code
def check_password(input_pass):
    # Hardcoded = Bad
    return input_pass == "newpassword"

if __name__ == "__main__":
    pass
"""

# 3. WRITE ORIGINAL FILE
with open(app_path, "w") as f:
    f.write(original_code)
print(f"✅ Reset app.py")

# 4. GENERATE PATCH DYNAMICALLY (The Fix)
# We use difflib to create a perfectly formatted unified diff
diff_lines = difflib.unified_diff(
    original_code.splitlines(keepends=True),
    fixed_code.splitlines(keepends=True),
    fromfile="a/app.py",
    tofile="b/app.py"
)
patch_content = "".join(diff_lines)

with open(patch_path, "w") as f:
    f.write(patch_content)
print(f"✅ Generated valid fix.patch via difflib")

# 5. RUN MANAGER
task = TaskSpec(
    task_id="dummy_01",
    workspace_dir=workspace,
    target_files=["app.py"],
    verify_cmd="true",
    issue_summary="test"
)
manager = PatchManager(task)

print(f"\n--- Applying Patch ---")
result = manager.apply_diff(patch_content)
print(f"Manager Result: {result.success} - {result.message}")

# 6. VERIFY CONTENT
print(f"\n--- Verifying Content ---")
with open(app_path, "r") as f:
    content = f.read()
    if "newpassword" in content:
        print("✅ SUCCESS: File updated to 'newpassword'")
    else:
        print("❌ FAIL: File did not update")
        # Debug: Print git apply error if manager failed silently
        subprocess.run(["git", "apply", "-v", patch_path], cwd=workspace)

# 7. ROLLBACK
print(f"\n--- Rolling Back ---")
manager.restore_backup()
with open(app_path, "r") as f:
    if "secret123" in f.read():
        print("✅ SUCCESS: Rollback worked")
    else:
        print("❌ FAIL: Rollback failed")