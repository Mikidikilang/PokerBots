#!/usr/bin/env python3
"""Complete workspace cleanup and git commit."""

import subprocess
from pathlib import Path

def run_cmd(cmd, description=""):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.stdout:
            print(result.stdout)
        if result.stderr and "warning" not in result.stderr.lower():
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

proj_root = Path.cwd()
print(f"Working directory: {proj_root}")
print("=" * 70)

# Step 1: Remove temporary files
print("\nStep 1: Cleaning up temporary files...")
temp_files = [
    "organize_workspace.py",
    "cleanup.py",
    "PHASE_4_4_HOW_TO_RUN.sh"
]

for fname in temp_files:
    fpath = proj_root / fname
    if fpath.exists():
        try:
            fpath.unlink()
            print(f"  ✓ Removed: {fname}")
        except Exception as e:
            print(f"  ✗ Failed to remove {fname}: {e}")

# Step 2: Git add all changes
print("\nStep 2: Staging changes with git add...")
if run_cmd("git add -A", "Staging"):
    print("  ✓ All changes staged")

# Step 3: Git diff to show what changed
print("\nStep 3: Git status summary...")
run_cmd("git status --short | head -40")

# Step 4: Commit
print("\nStep 4: Committing final state...")
commit_cmd = 'git commit -m "chore: Phase 5 complete, VR-DeepPDCFR+ engine finalized, workspace sanitized"'
if run_cmd(commit_cmd):
    print("\n" + "=" * 70)
    print("✓ COMMIT SUCCESSFUL!")
    print("=" * 70)
    
    # Show commit info
    print("\nCommit summary:")
    run_cmd("git log --oneline -n 1")
    print("\nGit status (should be clean):")
    run_cmd("git status")
else:
    print("\n⚠ Commit may have failed or working tree is clean")
    print("Checking current git status:")
    run_cmd("git status")

print("\n" + "=" * 70)
print("Phase 5 Workspace Cleanup Complete!")
print("=" * 70)
