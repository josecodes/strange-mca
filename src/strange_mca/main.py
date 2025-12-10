#!/usr/bin/env python
"""
Main script for the Strange MCA (Multi-Agent Conversation Architecture).

This script creates and runs a multi-agent system using a bidirectional graph
for task decomposition and response synthesis.
"""

import argparse
import copy
import datetime
import json
import logging
import os

from dotenv import load_dotenv

from src.strange_mca.graph import (
    create_execution_graph,
    run_execution_graph,
    total_nodes,
)
from src.strange_mca.visualization import (
    print_agent_tree,
    visualize_agent_tree,
    visualize_langgraph,
)

# Set up logger
logger = logging.getLogger("strange_mca")


def create_output_dir(child_per_parent: int, depth: int, model: str) -> str:
    """Generate an output directory name based on timestamp, child_per_parent, depth, and model.

    Args:
        child_per_parent: Number of children per parent.
        depth: Depth of the tree.
        model: Model name.

    Returns:
        Path to the output directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("-")[-1] if "-" in model else model
    dir_name = f"{timestamp}_c{child_per_parent}_d{depth}_{model_short}"
    output_dir = os.path.join("output", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    """Run the main script."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the Strange MCA multi-agent system"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Explain the concept of multi-agent systems",
        help="The task to perform",
    )
    parser.add_argument(
        "--child_per_parent",
        type=int,
        default=3,
        help="The number of children each non-leaf node has",
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="The number of levels in the tree"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="The name of the LLM model to use",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Logging level (debug, info, warning, error, critical)",
    )
    parser.add_argument(
        "--local-logs-only",
        action="store_true",
        help="Show only logs from strange_mca, suppress dependency logs",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Generate visualizations of the agent tree and execution graph",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't run the system, just show the configuration",
    )
    parser.add_argument(
        "--print_tree", action="store_true", help="Print the agent tree structure"
    )
    parser.add_argument(
        "--print_details", action="store_true", help="Print details about each agent"
    )
    parser.add_argument(
        "--domain_specific_instructions",
        type=str,
        default="",
        help="Domain-specific instructions to include in the strange loop prompt",
    )
    parser.add_argument(
        "--strange_loop_count",
        type=int,
        default=0,
        help="Number of strange loop iterations to perform",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    if args.local_logs_only:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=log_level)

    # Generate output directory
    output_dir = create_output_dir(args.child_per_parent, args.depth, args.model)
    logger.info(f"Output directory: {output_dir}")

    # Print configuration
    logger.info("Running with the following parameters:")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Children per parent: {args.child_per_parent}")
    logger.info(f"  Depth: {args.depth}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Output directory: {output_dir}")

    # Calculate the total number of agents
    num_agents = total_nodes(args.child_per_parent, args.depth)
    logger.info(f"Total agents: {num_agents}")

    # Print the agent tree if requested
    if args.print_tree:
        print_agent_tree(args.child_per_parent, args.depth)
        print()

    # Visualize the agent tree if requested
    if args.viz:
        output_file = visualize_agent_tree(
            cpp=args.child_per_parent,
            depth=args.depth,
            output_path=os.path.join(output_dir, "agent_tree"),
            format="png",
        )
        if output_file:
            logger.info(f"Agent tree visualization saved to {output_file}")

    # Exit if this is a dry run
    if args.dry_run:
        logger.info("Dry run completed")
        return

    # Create the execution graph
    logger.info("Creating execution graph...")
    graph = create_execution_graph(
        child_per_parent=args.child_per_parent,
        depth=args.depth,
        model_name=args.model,
        langgraph_viz_dir=None,  # We'll handle visualization separately
        domain_specific_instructions=args.domain_specific_instructions,
        strange_loop_count=args.strange_loop_count,
    )

    # Generate LangGraph visualization if requested
    if args.viz:
        visualize_langgraph(graph, output_dir, args.child_per_parent, args.depth)

    # Run the execution graph
    logger.info("Running execution graph...")
    result = run_execution_graph(
        execution_graph=graph,
        task=args.task,
        log_level=args.log_level,
        only_local_logs=args.local_logs_only,
        langgraph_viz_dir=None,  # We've already handled visualization
    )

    # Print the final response
    logger.info("Graph execution completed")
    print("\nFinal Response:")
    print("=" * 80)
    print(result.get("final_response", "No final response generated"))
    print("=" * 80)

    if args.print_details:
        print("\nFull State:")
        print("=" * 80)
        # Create a DEEP copy of the result to avoid modifying the original
        state_copy = copy.deepcopy(result)
        print(json.dumps(state_copy, indent=2))
        print("=" * 80)

    # Save the full state to a JSON file
    state_file = os.path.join(output_dir, "final_state.json")
    with open(state_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full state saved to: {state_file}")

    # Print summary
    print("\nExecution Summary:")
    print(
        f"Tree structure: {args.child_per_parent} children per parent, {args.depth} levels deep"
    )
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Output saved to: {output_dir}")

    logger.info("Execution completed")


if __name__ == "__main__":
    main()
