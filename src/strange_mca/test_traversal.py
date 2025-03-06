"""Test script for tree traversal functions."""

import argparse
import logging

from src.strange_mca.graph import test_tree_traversal
from src.strange_mca.logging_utils import setup_detailed_logging

# Set up logging
logger = logging.getLogger("strange_mca")

def main():
    """Run the tree traversal test."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test tree traversal functions")
    parser.add_argument("--child_per_parent", type=int, default=2,
                        help="Number of children per parent node (default: 2)")
    parser.add_argument("--depth", type=int, default=3,
                        help="Depth of the tree (default: 3)")
    parser.add_argument("--log_level", type=str, default="debug",
                        choices=["debug", "info", "warn"],
                        help="Logging level (default: debug)")
    parser.add_argument("--only_local_logs", action="store_true",
                        help="Only show logs from the strange_mca logger")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_detailed_logging(log_level=args.log_level, only_local_logs=args.only_local_logs)
    
    # Run the test
    logger.info("Starting tree traversal test")
    visited, processed = test_tree_traversal(
        child_per_parent=args.child_per_parent,
        depth=args.depth
    )
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tree structure: {args.child_per_parent} children per parent, {args.depth} levels deep")
    print(f"Nodes visited in downward pass: {len(visited)}")
    print(f"Nodes processed in upward pass: {len(processed)}")
    
    logger.info("Tree traversal test completed")

if __name__ == "__main__":
    main() 