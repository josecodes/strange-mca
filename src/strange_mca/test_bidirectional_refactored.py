"""Test script for refactored bidirectional graph traversal."""

import argparse
import logging
import os
from dotenv import load_dotenv

from src.strange_mca.graph import create_bidirectional_graph, run_bidirectional_graph
from src.strange_mca.logging_utils import setup_detailed_logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger("strange_mca")

def main():
    """Run the bidirectional graph test with refactored implementation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test refactored bidirectional graph traversal")
    parser.add_argument("--task", type=str, default="Explain what a neural network is",
                        help="The task to perform (default: 'Explain what a neural network is')")
    parser.add_argument("--context", type=str, default="",
                        help="The context for the task (default: '')")
    parser.add_argument("--child_per_parent", type=int, default=2,
                        help="Number of children per parent node (default: 2)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Depth of the tree (default: 2)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="The name of the LLM model to use (default: 'gpt-3.5-turbo')")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warn"],
                        help="Logging level (default: info)")
    parser.add_argument("--only_local_logs", action="store_true",
                        help="Only show logs from the strange_mca logger")
    parser.add_argument("--viz_dir", type=str, default="output/langgraph_viz",
                        help="Directory to save visualization of the graph (default: output/langgraph_viz)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_detailed_logging(log_level=args.log_level, only_local_logs=args.only_local_logs)
    
    # Create the visualization directory if specified
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    
    # Create the bidirectional graph
    logger.info("Creating bidirectional graph with refactored implementation")
    graph = create_bidirectional_graph(
        child_per_parent=args.child_per_parent,
        depth=args.depth,
        model_name=args.model_name,
        langgraph_viz_dir=args.viz_dir
    )
    
    # Run the graph
    logger.info(f"Running bidirectional graph with task: {args.task}")
    result = run_bidirectional_graph(
        graph=graph,
        task=args.task,
        context=args.context,
        log_level=args.log_level,
        only_local_logs=args.only_local_logs,
        langgraph_viz_dir=args.viz_dir
    )
    
    # Print the final response
    print("\nFinal Response:")
    print(result.get("final_response", "No response available."))
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tree structure: {args.child_per_parent} children per parent, {args.depth} levels deep")
    print(f"Total nodes in responses: {len(result.get('node_responses', {}))}")
    
    logger.info("Refactored bidirectional graph test completed")

if __name__ == "__main__":
    main() 