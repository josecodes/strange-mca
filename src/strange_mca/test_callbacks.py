"""Test script for demonstrating callbacks in AgentTree traversals."""

import argparse
import logging
import os
from typing import Dict, List, Callable, Any

from src.strange_mca.agents import create_agent_tree, AgentTree
from src.strange_mca.logging_utils import setup_detailed_logging

# Set up logging
logger = logging.getLogger("strange_mca")

def downward_callback(node: str, tree: AgentTree) -> None:
    """Callback function for downward traversal.
    
    Args:
        node: The name of the current node.
        tree: The AgentTree instance.
    """
    parent = tree.get_parent(node)
    children = tree.get_children(node)
    
    parent_info = f"parent={parent}" if parent else "root"
    children_info = f"children={children}" if children else "leaf"
    
    logger.info(f"Downward traversal at node {node}: {parent_info}, {children_info}")
    
    # Example of custom logic that could be performed at each node
    if not parent:  # Root node
        logger.info(f"Root node {node} will coordinate the overall task")
    elif not children:  # Leaf node
        logger.info(f"Leaf node {node} will perform a specific subtask")
    else:  # Middle node
        logger.info(f"Middle node {node} will decompose tasks for {len(children)} children")

def upward_callback(node: str, tree: AgentTree) -> None:
    """Callback function for upward traversal.
    
    Args:
        node: The name of the current node.
        tree: The AgentTree instance.
    """
    parent = tree.get_parent(node)
    children = tree.get_children(node)
    
    parent_info = f"parent={parent}" if parent else "root"
    children_info = f"children={children}" if children else "leaf"
    
    logger.info(f"Upward traversal at node {node}: {parent_info}, {children_info}")
    
    # Example of custom logic that could be performed at each node
    if not parent:  # Root node
        logger.info(f"Root node {node} will synthesize all results")
    elif not children:  # Leaf node
        logger.info(f"Leaf node {node} has completed its task and is sending results up")
    else:  # Middle node
        logger.info(f"Middle node {node} is synthesizing results from {len(children)} children")

def collect_node_data() -> Dict[str, List[str]]:
    """Collect data about nodes visited during traversals.
    
    Returns:
        A dictionary with keys 'downward' and 'upward', each containing a list of node names.
    """
    node_data = {
        'downward': [],
        'upward': []
    }
    
    def downward_collector(node: str, tree: AgentTree) -> None:
        node_data['downward'].append(node)
        downward_callback(node, tree)
    
    def upward_collector(node: str, tree: AgentTree) -> None:
        node_data['upward'].append(node)
        upward_callback(node, tree)
    
    return node_data, downward_collector, upward_collector

def main():
    """Run the test for AgentTree callbacks."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test AgentTree traversal callbacks")
    parser.add_argument("--child_per_parent", type=int, default=2,
                        help="Number of children per parent node (default: 2)")
    parser.add_argument("--depth", type=int, default=3,
                        help="Depth of the tree (default: 3)")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warn"],
                        help="Logging level (default: info)")
    parser.add_argument("--only_local_logs", action="store_true",
                        help="Only show logs from the strange_mca logger")
    parser.add_argument("--viz_dir", type=str, default="output/agent_tree_viz",
                        help="Directory to save visualization of the tree (default: output/agent_tree_viz)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_detailed_logging(log_level=args.log_level, only_local_logs=args.only_local_logs)
    
    # Create the visualization directory if specified
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    
    # Create agent tree
    logger.info(f"Creating agent tree with {args.child_per_parent} children per parent and {args.depth} levels")
    agent_tree = create_agent_tree(args.child_per_parent, args.depth)
    
    # Visualize the tree
    agent_tree.visualize(output_dir=args.viz_dir, filename="agent_tree")
    
    # Collect node data and get collector callbacks
    node_data, downward_collector, upward_collector = collect_node_data()
    
    # Perform downward traversal with callback
    logger.info("Starting downward traversal with callback")
    downward_nodes = agent_tree.perform_down_traversal(node_callback=downward_collector)
    
    # Get leaf nodes
    leaf_nodes = agent_tree.get_leaf_nodes()
    
    # Perform upward traversal with callback
    logger.info("Starting upward traversal with callback")
    upward_nodes = agent_tree.perform_up_traversal(leaf_nodes, node_callback=upward_collector)
    
    # Log the results
    logger.info(f"Downward traversal visited {len(downward_nodes)} nodes: {downward_nodes}")
    logger.info(f"Found {len(leaf_nodes)} leaf nodes: {leaf_nodes}")
    logger.info(f"Upward traversal processed {len(upward_nodes)} nodes: {upward_nodes}")
    
    # Verify that all nodes were visited
    all_nodes = set(agent_tree.graph.nodes())
    if set(downward_nodes) == all_nodes and set(upward_nodes) == all_nodes:
        logger.info("All nodes were visited in both traversals")
    else:
        missing_down = all_nodes - set(downward_nodes)
        missing_up = all_nodes - set(upward_nodes)
        if missing_down:
            logger.warning(f"Nodes missed in downward traversal: {missing_down}")
        if missing_up:
            logger.warning(f"Nodes missed in upward traversal: {missing_up}")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tree structure: {args.child_per_parent} children per parent, {args.depth} levels deep")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Nodes visited in downward traversal: {len(node_data['downward'])}")
    print(f"Nodes visited in upward traversal: {len(node_data['upward'])}")
    
    logger.info("AgentTree callback test completed")

if __name__ == "__main__":
    main() 