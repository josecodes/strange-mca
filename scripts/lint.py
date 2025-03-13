#!/usr/bin/env python
"""
Linting script for the Strange MCA codebase.

This script runs both Ruff and Black on the codebase to maintain code quality.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional


def run_command(cmd: List[str], description: str) -> int:
    """Run a command and return its exit code.

    Args:
        cmd: The command to run as a list of strings.
        description: A description of the command for logging.

    Returns:
        The exit code of the command.
    """
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_ruff(
    target_dirs: List[str], fix: bool = False, unsafe_fixes: bool = False
) -> int:
    """Run Ruff on the specified directories.

    Args:
        target_dirs: List of directories to run Ruff on.
        fix: Whether to automatically fix issues.
        unsafe_fixes: Whether to apply unsafe fixes.

    Returns:
        The exit code of the Ruff command.
    """
    cmd = ["poetry", "run", "ruff", "check"]
    
    if fix:
        cmd.append("--fix")
    
    if unsafe_fixes:
        cmd.append("--unsafe-fixes")
    
    cmd.extend(target_dirs)
    
    return run_command(cmd, "Running Ruff linter")


def run_black(target_dirs: List[str], check: bool = False) -> int:
    """Run Black on the specified directories.

    Args:
        target_dirs: List of directories to run Black on.
        check: Whether to check if files would be reformatted without actually reformatting them.

    Returns:
        The exit code of the Black command.
    """
    cmd = ["poetry", "run", "black"]
    
    if check:
        cmd.append("--check")
    
    cmd.extend(target_dirs)
    
    return run_command(cmd, "Running Black formatter")


def main():
    """Run the linting script."""
    parser = argparse.ArgumentParser(description="Run linters on the Strange MCA codebase")
    parser.add_argument(
        "--target", 
        nargs="+", 
        default=["src", "tests", "examples"], 
        help="Directories to lint (default: src tests examples)"
    )
    parser.add_argument(
        "--ruff-only", 
        action="store_true", 
        help="Run only Ruff, not Black"
    )
    parser.add_argument(
        "--black-only", 
        action="store_true", 
        help="Run only Black, not Ruff"
    )
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Fix issues automatically with Ruff"
    )
    parser.add_argument(
        "--unsafe-fixes", 
        action="store_true", 
        help="Apply unsafe fixes with Ruff (implies --fix)"
    )
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check if files would be reformatted with Black without actually reformatting them"
    )
    
    args = parser.parse_args()
    
    # Ensure target directories exist
    for target_dir in args.target:
        if not os.path.exists(target_dir):
            print(f"Warning: Target directory '{target_dir}' does not exist.")
    
    exit_code = 0
    
    # If unsafe-fixes is specified, ensure fix is also set
    if args.unsafe_fixes:
        args.fix = True
    
    # Run Ruff if not black-only
    if not args.black_only:
        ruff_exit = run_ruff(args.target, args.fix, args.unsafe_fixes)
        exit_code = max(exit_code, ruff_exit)
    
    # Run Black if not ruff-only
    if not args.ruff_only:
        black_exit = run_black(args.target, args.check)
        exit_code = max(exit_code, black_exit)
    
    if exit_code == 0:
        print("\n✅ All linting checks passed!")
    else:
        print("\n❌ Some linting checks failed. See above for details.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 