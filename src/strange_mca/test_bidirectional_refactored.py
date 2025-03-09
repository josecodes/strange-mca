#!/usr/bin/env python
"""
Test script for the refactored bidirectional graph implementation.

This script demonstrates the use of the refactored bidirectional graph with the AgentTree
traversal callbacks. It creates a bidirectional graph with the specified parameters and
runs it on a given task and context.
"""

import argparse
import json
import logging
import os
import pprint
from typing import Optional
from dotenv import load_dotenv

from src.strange_mca.graph import create_bidirectional_graph, run_bidirectional_graph
from src.strange_mca.logging_utils import setup_detailed_logging

# Set up logger
logger = logging.getLogger("strange_mca")

def main():
    """Run the test script."""
    # Load environment variables from .env file if it exists
    # This ensures the OpenAI API key is available
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the refactored bidirectional graph implementation")
    parser.add_argument("--task", type=str, default="Explain the concept of recursion in programming",
                        help="The task to perform")
    parser.add_argument("--context", type=str, default="",
                        help="The context for the task")
    parser.add_argument("--child_per_parent", type=int, default=2,
                        help="The number of children each non-leaf node has")
    parser.add_argument("--depth", type=int, default=2,
                        help="The number of levels in the tree")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="The name of the LLM model to use")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warn", "error"],
                        help="The log level to use")
    parser.add_argument("--only_local_logs", action="store_true",
                        help="Only show logs from this script, not from dependencies")
    parser.add_argument("--viz_dir", type=str, default="output",
                        help="Directory to save visualizations")
    parser.add_argument("--verbose", action="store_true",
                        help="Display the full State at the end of execution")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    if args.only_local_logs:
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=log_level)
    
    # Create visualization directory if it doesn't exist
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # Print test parameters
    logger.info(f"Running bidirectional graph test with the following parameters:")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Context: {args.context}")
    logger.info(f"  Children per parent: {args.child_per_parent}")
    logger.info(f"  Depth: {args.depth}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Visualization directory: {args.viz_dir}")
    logger.info(f"  Verbose output: {args.verbose}")
    
    # Create the bidirectional graph
    logger.info("Creating bidirectional graph...")
    graph = create_bidirectional_graph(
        child_per_parent=args.child_per_parent,
        depth=args.depth,
        model_name=args.model_name,
        langgraph_viz_dir=args.viz_dir
    )
    
    # Run the graph
    logger.info("Running bidirectional graph...")
    result = run_bidirectional_graph(
        graph=graph,
        task=args.task,
        log_level=args.log_level,
        only_local_logs=args.only_local_logs,
        langgraph_viz_dir=args.viz_dir
    )
    
    # Print the final response
    logger.info("Graph execution completed")
    print("\nFinal Response:")
    print("=" * 80)
    print(result.get("final_response", "No final response generated"))
    print("=" * 80)
    
    # Print the full State if verbose is enabled
    if args.verbose:
        print("\nFull State:")
        print("=" * 80)
        # Create a copy of the result to avoid modifying the original
        state_copy = dict(result)
        
        # Format nodes dictionary for better readability
        if "nodes" in state_copy:
            for node_name, node_data in state_copy["nodes"].items():
                # Truncate long responses for display
                if "response" in node_data and len(node_data["response"]) > 500:
                    state_copy["nodes"][node_name]["response"] = node_data["response"][:500] + "... [truncated]"
                
                # Truncate long tasks for display
                if "task" in node_data and len(node_data["task"]) > 500:
                    state_copy["nodes"][node_name]["task"] = node_data["task"][:500] + "... [truncated]"
                
                # Truncate long contexts for display
                if "context" in node_data and len(node_data["context"]) > 500:
                    state_copy["nodes"][node_name]["context"] = node_data["context"][:500] + "... [truncated]"
        
        # Pretty print the state
        pp = pprint.PrettyPrinter(indent=2, width=100)
        pp.pprint(state_copy)
        print("=" * 80)
        
        # Save the full state to a JSON file
        state_file = os.path.join(args.viz_dir, "final_state.json")
        with open(state_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Full state saved to: {state_file}")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tree structure: {args.child_per_parent} children per parent, {args.depth} levels deep")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Visualization saved to: {args.viz_dir}")
    
    logger.info("Bidirectional graph test completed")

if __name__ == "__main__":
    main() 