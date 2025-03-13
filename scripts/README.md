# Strange MCA Scripts

This directory contains utility scripts for the Strange MCA project.

## Linting Scripts

### `lint.py`

A Python script that runs both Ruff and Black on the codebase to maintain code quality.

Usage:
```bash
python scripts/lint.py [options]
```

Options:
- `--target DIR1 [DIR2 ...]`: Directories to lint (default: src tests arena)
- `--ruff-only`: Run only Ruff, not Black
- `--black-only`: Run only Black, not Ruff
- `--fix`: Fix issues automatically with Ruff
- `--unsafe-fixes`: Apply unsafe fixes with Ruff (implies --fix)
- `--check`: Check if files would be reformatted with Black without actually reformatting them

Examples:
```bash
# Run both Ruff and Black on the default directories
python scripts/lint.py

# Run only Ruff with automatic fixes
python scripts/lint.py --ruff-only --fix

# Run only Black on a specific directory
python scripts/lint.py --black-only --target src/strange_mca

# Check if files would be reformatted with Black without actually reformatting them
python scripts/lint.py --black-only --check
```

### `lint.sh`

A shell script wrapper for `lint.py` that can be used as a shortcut.

Usage:
```bash
./scripts/lint.sh [options]
```

The options are the same as for `lint.py`.

## Pre-commit Hook

You can set up a pre-commit hook to run the linting scripts automatically before each commit:

1. Create a file named `.git/hooks/pre-commit` with the following content:
```bash
#!/bin/bash

# Run linting checks
./scripts/lint.sh --check

# If the linting checks fail, abort the commit
if [ $? -ne 0 ]; then
  echo "Linting checks failed. Please fix the issues before committing."
  exit 1
fi
```

2. Make the hook executable:
```bash
chmod +x .git/hooks/pre-commit
```

This will run the linting checks in "check" mode before each commit, and abort the commit if there are any issues.

## Setting Up the Pre-commit Hook Manually

If you're having trouble with the pre-commit hook, you can set it up manually:

1. Copy the pre-commit script to the Git hooks directory:
```bash
cp scripts/pre-commit .git/hooks/
```

2. Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

Alternatively, you can create a symbolic link:
```bash
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
``` 