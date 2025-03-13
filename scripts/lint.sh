#!/bin/bash
# Simple wrapper script for lint.py

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python linting script with all arguments passed to this script
python "$SCRIPT_DIR/lint.py" "$@" 