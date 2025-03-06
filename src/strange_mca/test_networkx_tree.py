#!/usr/bin/env python3
"""Test script for the NetworkX-based agent tree."""

import argparse
import logging
import os
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv

from src.strange_mca.agents import create_agent_tree
from src.strange_mca.logging_utils import setup_detailed_logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger("strange_mca")


def main():
    """Run the test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the NetworkX-based agent tree")
    parser.add_argument(
        "--children", type=int, default=2, help="Number of children per parent (default: 2)"
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Depth of the tree (default: 3)"
    )
    parser.add_argument(
        "--log-level", type=str, default="info", choices=["debug", "info", "warn"],
        help="Logging level (default: info)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory to save visualizations (default: output)"
    )
    args = parser.parse_args()
    
    # Set up logging
    setup_detailed_logging(log_level=args.log_level, only_local_logs=True)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log the test parameters
    logger.info(f"Testing NetworkX-based agent tree with:")
    logger.info(f"  Children per parent: {args.children}")
    logger.info(f"  Tree depth: {args.depth}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Create the agent tree
    logger.info("Creating agent tree...")
    agent_tree = create_agent_tree(args.children, args.depth)
    
    # Log the tree structure
    logger.info("Tree structure:")
    for name, config in agent_tree.get_configs().items():
        parent = agent_tree.get_parent(name)
        children = agent_tree.get_children(name)
        parent_info = f"parent={parent}" if parent else "root"
        children_info = f"children={children}" if children else "leaf"
        logger.info(f"  {name}: {parent_info}, {children_info}")
    
    # Visualize the tree
    logger.info("Visualizing the tree...")
    viz_path = agent_tree.visualize(output_dir=str(output_dir), filename="agent_tree.png")
    logger.info(f"Tree visualization saved to: {viz_path}")
    
    # Perform downward traversal
    logger.info("Performing downward traversal...")
    downward_nodes = agent_tree.perform_down_traversal()
    logger.info(f"Downward traversal visited {len(downward_nodes)} nodes:")
    logger.info(f"  {downward_nodes}")
    
    # Get leaf nodes
    leaf_nodes = agent_tree.get_leaf_nodes()
    logger.info(f"Found {len(leaf_nodes)} leaf nodes:")
    logger.info(f"  {leaf_nodes}")
    
    # Perform upward traversal
    logger.info("Performing upward traversal...")
    upward_nodes = agent_tree.perform_up_traversal()
    logger.info(f"Upward traversal processed {len(upward_nodes)} nodes:")
    logger.info(f"  {upward_nodes}")
    
    # Verify that all nodes were visited
    all_nodes = set(agent_tree.graph.nodes())
    if set(downward_nodes) == all_nodes and set(upward_nodes) == all_nodes:
        logger.info("All nodes were visited in both traversals")
    else:
        missing_down = all_nodes - set(downward_nodes)
        missing_up = all_nodes - set(upward_nodes)
        if missing_down:
            logger.warning(f"Nodes not visited in downward traversal: {missing_down}")
        if missing_up:
            logger.warning(f"Nodes not visited in upward traversal: {missing_up}")
    
    # Test graph properties
    logger.info("Testing graph properties...")
    
    # Check if the graph is a directed acyclic graph (DAG)
    is_dag = agent_tree.graph.is_directed() and nx.is_directed_acyclic_graph(agent_tree.graph)
    logger.info(f"Is the graph a DAG? {'Yes' if is_dag else 'No'}")
    
    # Check if the graph is a tree
    is_tree = nx.is_tree(agent_tree.graph.to_undirected())
    logger.info(f"Is the graph a tree? {'Yes' if is_tree else 'No'}")
    
    # Calculate the diameter of the graph (longest shortest path)
    try:
        diameter = nx.diameter(agent_tree.graph.to_undirected())
        logger.info(f"Graph diameter: {diameter}")
    except nx.NetworkXError:
        logger.warning("Could not calculate graph diameter (graph may not be connected)")
    
    # Calculate the average shortest path length
    try:
        avg_path_length = nx.average_shortest_path_length(agent_tree.graph)
        logger.info(f"Average shortest path length: {avg_path_length:.2f}")
    except nx.NetworkXError:
        logger.warning("Could not calculate average shortest path length (graph may not be connected)")
    
    logger.info("NetworkX-based agent tree test completed successfully")


if __name__ == "__main__":
    main() 