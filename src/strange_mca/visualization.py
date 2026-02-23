"""Visualization utilities for the multiagent system."""

import logging
import os
from typing import Any, Optional

import graphviz

from src.strange_mca.tree_helpers import (
    count_nodes_at_level,
    generate_all_nodes,
    get_children,
    get_parent,
    get_siblings,
    parse_node_name,
)

logger = logging.getLogger("strange_mca")


def visualize_agent_tree(
    cpp: int,
    depth: int,
    output_path: Optional[str] = None,
    format: str = "png",
) -> Optional[str]:
    """Visualize the agent tree structure with lateral edges.

    Args:
        cpp: Children per parent.
        depth: Total tree depth.
        output_path: The path to save the visualization.
        format: The format to save the visualization in.

    Returns:
        The path to the saved visualization.
    """
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dot = graphviz.Digraph(
        "Agent Tree",
        comment="Visualization of the MCA agent hierarchy",
        format=format,
    )

    all_nodes = generate_all_nodes(cpp, depth)

    for node_name in all_nodes:
        level, num = parse_node_name(node_name)

        if level == 1:
            color = "lightblue"
            role = "Integrator"
            label = f"{node_name}\n{role}"
        elif level == depth:
            color = "lightgreen"
            role = "Specialist"
            label = f"{node_name}\n{role}"
        else:
            color = "lightyellow"
            role = "Coordinator"
            label = f"{node_name}\n{role}"

        dot.node(node_name, label, style="filled", fillcolor=color)

    # Add parent-child edges (solid)
    for node_name in all_nodes:
        parent = get_parent(node_name, cpp)
        if parent:
            dot.edge(parent, node_name)

    # Add lateral edges (dashed) between siblings
    seen_pairs = set()
    for node_name in all_nodes:
        siblings = get_siblings(node_name, cpp, depth)
        for sibling in siblings:
            pair = tuple(sorted([node_name, sibling]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                dot.edge(
                    node_name,
                    sibling,
                    style="dashed",
                    dir="none",
                    color="gray",
                    constraint="false",
                )

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
        level, _ = parse_node_name(node_name)
        print(f"{'  ' * indent}└─ {node_name} (Level {level})")

        for child in get_children(node_name, cpp, depth):
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
    """Visualize the agent tree structure.

    Args:
        graph: The compiled LangGraph. Reserved for future use when actual
            graph introspection is implemented. Currently unused because
            nested subgraphs don't produce clean visualizations.
        output_dir: Directory to save the visualization.
        cpp: Children per parent.
        depth: Total tree depth.
        filename: Base filename for the visualization (without extension).

    Returns:
        The path to the saved visualization file, or None if visualization failed.
    """
    _ = graph  # Reserved for future graph introspection
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        dot = graphviz.Digraph(
            "Execution Graph",
            comment="Visualization of the MCA agent tree structure",
            format="png",
        )

        dot.attr(rankdir="TB", size="10,10", ratio="fill")
        dot.attr("node", shape="box", style="filled", fontname="Arial", fontsize="10")
        dot.attr("edge", arrowsize="0.7", fontname="Arial", fontsize="9")

        total = sum(count_nodes_at_level(level, cpp) for level in range(1, depth + 1))
        dot.attr(
            label=f"MCA Agent Tree\nDepth: {depth}, CPP: {cpp}\nTotal Agents: {total}",
            fontsize="14",
            fontname="Arial Bold",
        )

        all_nodes = generate_all_nodes(cpp, depth)

        for node_name in all_nodes:
            level, _ = parse_node_name(node_name)

            if level == 1:
                color = "#e6f7ff"
                label = f"{node_name}\n(Integrator)"
            elif level == depth:
                color = "#e6ffe6"
                label = f"{node_name}\n(Specialist)"
            else:
                color = "#fff2e6"
                label = f"{node_name}\n(Coordinator)"

            dot.node(node_name, label, style="filled", fillcolor=color)

        # Parent-child edges
        for node_name in all_nodes:
            parent = get_parent(node_name, cpp)
            if parent:
                dot.edge(parent, node_name, dir="forward")

        # Lateral edges
        seen_pairs = set()
        for node_name in all_nodes:
            siblings = get_siblings(node_name, cpp, depth)
            for sibling in siblings:
                pair = tuple(sorted([node_name, sibling]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    dot.edge(
                        node_name,
                        sibling,
                        style="dashed",
                        dir="none",
                        color="gray",
                        constraint="false",
                    )

        output_file = dot.render(output_path, cleanup=True)
        logger.info(f"Agent tree visualization saved to {output_file}")
        return output_file

    except Exception as e:
        logger.warning(f"Error visualizing agent tree: {e}")
        return None
