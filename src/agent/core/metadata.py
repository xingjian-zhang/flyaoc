"""Experiment metadata capture for reproducibility.

This module captures environment and version information to ensure
experiment runs can be traced back to specific code/data versions.
"""

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def get_git_info(repo_path: Path | None = None) -> dict:
    """Get git repository state.

    Args:
        repo_path: Path to git repository (defaults to cwd)

    Returns:
        Dict with commit, branch, dirty, and tag info
    """
    cwd = str(repo_path) if repo_path else None

    try:
        # Get commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Get short commit
        commit_short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Get branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status)

        # Get most recent tag (if any)
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            tag = None

        return {
            "commit": commit,
            "commit_short": commit_short,
            "branch": branch,
            "dirty": dirty,
            "tag": tag,
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": None,
            "commit_short": None,
            "branch": None,
            "dirty": None,
            "tag": None,
            "error": "Not a git repository or git not available",
        }


def get_lock_hash(lock_path: Path | None = None) -> str | None:
    """Get hash of uv.lock file for dependency tracking.

    Args:
        lock_path: Path to uv.lock (defaults to cwd/uv.lock)

    Returns:
        MD5 hash of lock file, or None if not found
    """
    if lock_path is None:
        lock_path = Path.cwd() / "uv.lock"

    if not lock_path.exists():
        return None

    content = lock_path.read_bytes()
    return hashlib.md5(content).hexdigest()


def capture_metadata(
    command_args: list[str] | None = None,
    repo_path: Path | None = None,
) -> dict:
    """Capture full experiment metadata.

    Args:
        command_args: CLI arguments (defaults to sys.argv)
        repo_path: Path to git repository

    Returns:
        Dict with all metadata fields
    """
    if command_args is None:
        command_args = sys.argv

    git_info = get_git_info(repo_path)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git": git_info,
        "python_version": sys.version,
        "command": command_args,
        "uv_lock_hash": get_lock_hash(),
    }


def format_version_string(metadata: dict) -> str:
    """Format metadata as a short version string for logging.

    Args:
        metadata: Metadata dict from capture_metadata()

    Returns:
        Short string like "abc1234 (dirty)" or "abc1234"
    """
    git = metadata.get("git", {})
    commit = git.get("commit_short", "unknown")
    dirty = git.get("dirty", False)

    if dirty:
        return f"{commit} (dirty)"
    return commit
