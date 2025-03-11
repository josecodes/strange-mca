"""
Script to run a single task with custom parameters.

This script provides a simple way to run a single task with the Strange MCA system,
allowing you to specify custom parameters.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strange_mca.run_strange_mca import run_strange_mca


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a single task with the Strange MCA system.")
    
    parser.add_argument("--task", type=str, required=True, help="The task to run.")
    parser.add_argument("--child_per_parent", type=int, default=2, help="Number of children per parent node.")
    parser.add_argument("--depth", type=int, default=2, help="Depth of the tree.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use.")
    parser.add_argument("--log_level", type=str, default="info", help="Log level.")
    parser.add_argument("--viz", action="store_true", help="Generate visualizations.")
    parser.add_argument("--no_viz", action="store_true", help="Do not generate visualizations.")
    parser.add_argument("--only_local_logs", action="store_true", help="Only use local logs.")
    parser.add_argument("--print_details", action="store_true", help="Print detailed information.")
    parser.add_argument("--no_print_details", action="store_true", help="Do not print detailed information.")
    parser.add_argument("--output_dir", type=str, help="Output directory to use.")
    
    args = parser.parse_args()
    
    # Handle conflicting flags
    if args.viz and args.no_viz:
        print("Warning: Both --viz and --no_viz specified. Using --viz.")
        args.no_viz = False
    
    if args.print_details and args.no_print_details:
        print("Warning: Both --print_details and --no_print_details specified. Using --print_details.")
        args.no_print_details = False
    
    return args


def main():
    """Run a single task with custom parameters."""
    args = parse_args()
    
    # Determine viz and print_details values
    viz = True
    if args.no_viz:
        viz = False
    elif args.viz:
        viz = True
    
    print_details = True
    if args.no_print_details:
        print_details = False
    elif args.print_details:
        print_details = True
    
    # Run the task
    result = run_strange_mca(
        task=args.task,
        child_per_parent=args.child_per_parent,
        depth=args.depth,
        model=args.model,
        log_level=args.log_level,
        viz=viz,
        only_local_logs=args.only_local_logs,
        print_details=print_details,
        output_dir=args.output_dir,
    )
    
    return result


if __name__ == "__main__":
    main() 