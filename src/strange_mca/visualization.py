"""Visualization utilities for the multiagent system."""

import logging
import os
from typing import Any, Optional

import graphviz

# Set up logger
logger = logging.getLogger("strange_mca")


# =============================================================================
# Tree Helper Functions (copied from graph.py for independence)
# =============================================================================


def _parse_node_name(node_name: str) -> tuple[int, int]:
    """Parse 'L{level}N{num}' into (level, num)."""
    parts = node_name.split("N")
    level = int(parts[0][1:])
    num = int(parts[1])
    return level, num


def _make_node_name(level: int, num: int) -> str:
    """Create node name from level and node number."""
    return f"L{level}N{num}"


def _get_children(node_name: str, cpp: int, depth: int) -> list[str]:
    """Get child node names for a given node."""
    level, num = _parse_node_name(node_name)
    if level >= depth:
        return []
    child_level = level + 1
    start = (num - 1) * cpp + 1
    return [_make_node_name(child_level, start + i) for i in range(cpp)]


def _get_parent(node_name: str, cpp: int) -> Optional[str]:
    """Get parent node name."""
    level, num = _parse_node_name(node_name)
    if level == 1:
        return None
    parent_num = (num - 1) // cpp + 1
    return _make_node_name(level - 1, parent_num)


def _count_nodes_at_level(level: int, cpp: int) -> int:
    """Count nodes at a given level."""
    return cpp ** (level - 1)


def _generate_all_nodes(cpp: int, depth: int) -> list[str]:
    """Generate all node names in level order."""
    nodes = []
    for level in range(1, depth + 1):
        count = _count_nodes_at_level(level, cpp)
        for node_num in range(1, count + 1):
            nodes.append(_make_node_name(level, node_num))
    return nodes


# =============================================================================
# Visualization Functions
# =============================================================================


def visualize_agent_tree(
    cpp: int,
    depth: int,
    output_path: str = None,
    format: str = "png",
) -> Optional[str]:
    """Visualize the agent tree structure.

    Args:
        cpp: Children per parent.
        depth: Total tree depth.
        output_path: The path to save the visualization.
        format: The format to save the visualization in.

    Returns:
        The path to the saved visualization.
    """
    # Ensure the output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a new graph
    dot = graphviz.Digraph(
        "Agent Tree",
        comment="Visualization of the multiagent system",
        format=format,
    )

    # Generate all nodes
    all_nodes = _generate_all_nodes(cpp, depth)

    # Add nodes
    for node_name in all_nodes:
        level, _ = _parse_node_name(node_name)

        # Create a label with the agent's name and level
        label = f"{node_name}\nLevel {level}"

        # Use different colors for different levels
        if level == 1:
            color = "lightblue"
        elif level == depth:
            color = "lightgreen"
        else:
            color = "lightyellow"

        # Add the node
        dot.node(node_name, label, style="filled", fillcolor=color)

    # Add edges
    for node_name in all_nodes:
        parent = _get_parent(node_name, cpp)
        if parent:
            dot.edge(parent, node_name)

    # Render the graph
    if output_path:
        try:
            output_file = dot.render(output_path, cleanup=True)
            return output_file
        except Exception as dot_e:
            logger.warning(f"Could not convert dot to image using Graphviz: {dot_e}")
            logger.info(
                f"For example: dot -Tpng {output_path}.dot -o {output_path}.png"
            )
            return f"{output_path}.dot"
    else:
        return None


def print_agent_tree(cpp: int, depth: int) -> None:
    """Print the agent tree structure.

    Args:
        cpp: Children per parent.
        depth: Total tree depth.
    """

    def print_node(node_name: str, indent: int = 0) -> None:
        """Print a node and its children recursively."""
        level, _ = _parse_node_name(node_name)
        print(f"{'  ' * indent}└─ {node_name} (Level {level})")

        for child in _get_children(node_name, cpp, depth):
            print_node(child, indent + 1)

    print("Agent Tree:")
    print_node("L1N1")


def visualize_langgraph(
    graph: Any,
    output_dir: str,
    cpp: int = 2,
    depth: int = 2,
    filename: str = "execution_graph_lg",
) -> Optional[str]:
    """Visualize the agent tree structure (since nested subgraphs don't visualize well).

    This creates a visualization of the logical agent tree structure,
    which mirrors the nested subgraph hierarchy.

    Args:
        graph: The compiled LangGraph (unused, kept for API compatibility).
        output_dir: Directory to save the visualization.
        cpp: Children per parent.
        depth: Total tree depth.
        filename: Base filename for the visualization (without extension).

    Returns:
        The path to the saved visualization file, or None if visualization failed.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create output path
    output_path = os.path.join(output_dir, filename)

    try:
        # Create a new Graphviz graph
        dot = graphviz.Digraph(
            "Execution Graph",
            comment="Visualization of the agent tree structure",
            format="png",
        )

        # Set graph attributes
        dot.attr(rankdir="TB", size="10,10", ratio="fill")
        dot.attr("node", shape="box", style="filled", fontname="Arial", fontsize="10")
        dot.attr("edge", arrowsize="0.7", fontname="Arial", fontsize="9")

        # Add title
        total = sum(_count_nodes_at_level(level, cpp) for level in range(1, depth + 1))
        dot.attr(
            label=f"Agent Tree\nDepth: {depth}, CPP: {cpp}\nTotal Agents: {total}",
            fontsize="14",
            fontname="Arial Bold",
        )

        # Generate all nodes
        all_nodes = _generate_all_nodes(cpp, depth)

        # Add nodes with appropriate styling
        for node_name in all_nodes:
            level, _ = _parse_node_name(node_name)

            if level == 1:
                color = "#e6f7ff"  # Light blue (root)
                label = f"{node_name}\n(Root)"
            elif level == depth:
                color = "#e6ffe6"  # Light green (leaf)
                label = f"{node_name}\n(Leaf)"
            else:
                color = "#fff2e6"  # Light orange (internal)
                label = f"{node_name}\n(Internal)"

            dot.node(node_name, label, style="filled", fillcolor=color)

        # Add edges
        for node_name in all_nodes:
            parent = _get_parent(node_name, cpp)
            if parent:
                dot.edge(parent, node_name, dir="forward")

        # Render the graph
        output_file = dot.render(output_path, cleanup=True)
        logger.info(f"Agent tree visualization saved to {output_file}")
        return output_file

    except Exception as e:
        logger.warning(f"Error visualizing agent tree: {e}")
        return None
