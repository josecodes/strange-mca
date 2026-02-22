#!/usr/bin/env python
"""
Main script for the Strange MCA (Multiscale Competency Architecture).

This script creates and runs the emergent MCA system using a flat LangGraph
with round-based bottom-up processing.
"""

import argparse
import datetime
import logging
import os

from dotenv import load_dotenv

from src.strange_mca.tree_helpers import total_nodes
from src.strange_mca.visualization import (
    print_agent_tree,
)

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
    load_dotenv()

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
        default="gpt-4o-mini",
        help="The name of the LLM model to use",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum number of rounds for convergence",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for convergence (0-1)",
    )
    parser.add_argument(
        "--enable_downward_signals",
        action="store_true",
        default=True,
        help="Enable parent-to-child signals (default: on)",
    )
    parser.add_argument(
        "--no_downward_signals",
        action="store_true",
        help="Disable parent-to-child signals",
    )
    parser.add_argument(
        "--perspectives",
        nargs="+",
        type=str,
        default=None,
        help="Custom perspectives for leaf agents",
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

    # Handle downward signals flag
    enable_downward_signals = not args.no_downward_signals

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
    logger.info(f"  Max rounds: {args.max_rounds}")
    logger.info(f"  Convergence threshold: {args.convergence_threshold}")
    logger.info(f"  Downward signals: {enable_downward_signals}")
    logger.info(f"  Output directory: {output_dir}")

    num_agents = total_nodes(args.child_per_parent, args.depth)
    logger.info(f"Total agents: {num_agents}")

    # Print the agent tree if requested
    if args.print_tree:
        print_agent_tree(args.child_per_parent, args.depth)
        print()

    # Exit if dry run
    if args.dry_run:
        logger.info("Dry run completed")
        return

    # Import here to avoid circular imports and allow dry_run without API key
    from src.strange_mca.run_strange_mca import run_strange_mca

    result = run_strange_mca(
        task=args.task,
        child_per_parent=args.child_per_parent,
        depth=args.depth,
        model=args.model,
        max_rounds=args.max_rounds,
        convergence_threshold=args.convergence_threshold,
        enable_downward_signals=enable_downward_signals,
        perspectives=args.perspectives,
        strange_loop_count=args.strange_loop_count,
        domain_specific_instructions=args.domain_specific_instructions,
        log_level=args.log_level,
        viz=args.viz,
        local_logs_only=args.local_logs_only,
        print_details=args.print_details,
        output_dir=output_dir,
    )

    # Print final response
    print("\nFinal Response:")
    print("=" * 80)
    print(result.get("final_response", "No final response generated"))
    print("=" * 80)

    # Print summary metrics
    convergence_scores = result.get("convergence_scores", [])
    current_round = result.get("current_round", 1)
    print("\nExecution Summary:")
    print(
        f"Tree structure: {args.child_per_parent} children per parent, {args.depth} levels deep"
    )
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Rounds used: {current_round - 1}")
    print(f"Converged: {result.get('converged', False)}")
    if convergence_scores:
        print(f"Convergence scores: {convergence_scores}")
    print(f"Output saved to: {output_dir}")

    logger.info("Execution completed")


if __name__ == "__main__":
    main()
