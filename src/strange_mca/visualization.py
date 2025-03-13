"""Visualization utilities for the multiagent system."""

import logging
import os
from typing import Any, Optional

import graphviz

from src.strange_mca.agents import AgentConfig

# Set up logger
logger = logging.getLogger("strange_mca")


def visualize_agent_graph(
    agent_configs: dict[str, AgentConfig],
    output_path: str = None,
    format: str = "png",
) -> Optional[str]:
    """Visualize the agent graph.

    Args:
        agent_configs: Dictionary mapping agent names to their configurations.
        output_path: The path to save the visualization.
        format: The format to save the visualization in.

    Returns:
        The path to the saved visualization.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a new graph
    dot = graphviz.Digraph(
        "Agent Tree",
        comment="Visualization of the multiagent system",
        format=format,
    )

    # Extract child_per_parent and depth from the agent_configs
    # We need these to create an AgentTree
    max_level = max(config.level for config in agent_configs.values())
    depth = max_level

    # Count children for the first parent to determine child_per_parent
    # Assuming all parents have the same number of children
    level_1_nodes = [
        name for name, config in agent_configs.items() if config.level == 1
    ]
    if level_1_nodes:
        root_node = level_1_nodes[0]
        level_2_nodes = [
            name for name, config in agent_configs.items() if config.level == 2
        ]
        child_per_parent = len(level_2_nodes)
    else:
        # Default to 2 if we can't determine
        child_per_parent = 2

    # Create an AgentTree to get parent-child relationships
    from src.strange_mca.agents import AgentTree

    agent_tree = AgentTree(child_per_parent, depth)

    # Add nodes
    for name, config in agent_configs.items():
        # Create a label with the agent's name and level
        label = f"{name}\nLevel {config.level}"

        # Use different colors for different levels
        color = "lightblue" if config.level == 1 else "lightgreen"

        # Add the node
        dot.node(name, label, style="filled", fillcolor=color)

    # Add edges using the AgentTree
    for name in agent_configs:
        parent = agent_tree.get_parent(name)
        if parent:
            dot.edge(parent, name)

    # Render the graph
    try:
        output_file = dot.render(output_path, cleanup=True)
        return output_file
    except Exception as dot_e:
        logger.warning(f"Could not convert dot to image using Graphviz: {dot_e}")
        logger.info(
            "You can manually convert the dot file using Graphviz or an online converter."
        )
        logger.info(f"For example: dot -Tpng {output_path}.dot -o {output_path}.png")
        return f"{output_path}.dot"


def print_agent_tree(agent_configs: dict[str, AgentConfig]) -> None:
    """Print the agent tree structure.

    Args:
        agent_configs: Dictionary mapping agent names to their configurations.
    """
    # Extract child_per_parent and depth from the agent_configs
    max_level = max(config.level for config in agent_configs.values())
    depth = max_level

    # Count children for the first parent to determine child_per_parent
    level_1_nodes = [
        name for name, config in agent_configs.items() if config.level == 1
    ]
    if level_1_nodes:
        root_node = level_1_nodes[0]
        level_2_nodes = [
            name for name, config in agent_configs.items() if config.level == 2
        ]
        child_per_parent = len(level_2_nodes)
    else:
        # Default to 2 if we can't determine
        child_per_parent = 2
        root_node = list(agent_configs.keys())[0]  # Just use the first node

    # Create an AgentTree to get parent-child relationships
    from src.strange_mca.agents import AgentTree

    agent_tree = AgentTree(child_per_parent, depth)

    def print_node(name: str, indent: int = 0) -> None:
        """Print a node and its children recursively.

        Args:
            name: The name of the node.
            indent: The indentation level.
        """
        config = agent_configs[name]
        print(f"{'  ' * indent}└─ {name} (Level {config.level})")

        for child in agent_tree.get_children(name):
            print_node(child, indent + 1)

    print("Agent Tree:")
    print_node(root_node)


def print_agent_details(agent_configs: dict[str, AgentConfig]) -> None:
    """Print details about each agent in the system.

    Args:
        agent_configs: Dictionary mapping agent names to their configurations.
    """
    print("Agent Details:")
    print("=" * 80)

    # Create a dictionary to map agents to their parent
    agent_parents = {}
    agent_children = {}

    # Initialize empty children lists for all agents
    for name in agent_configs:
        agent_children[name] = []

    # Build the parent-child relationships from the agent configs
    for name, config in agent_configs.items():
        # For non-root nodes, determine the parent based on naming convention
        if config.level > 1:
            # Calculate parent node number
            parent_node_number = (
                (config.node_number - 1) // (len(agent_configs) // config.level)
            ) + 1
            parent_name = f"L{config.level - 1}N{parent_node_number}"
            agent_parents[name] = parent_name

            # Add this node as a child of its parent
            if parent_name in agent_children:
                agent_children[parent_name].append(name)
        else:
            agent_parents[name] = None

    # Print details for each agent, sorted by level and node number
    for name, config in sorted(
        agent_configs.items(), key=lambda x: (x[1].level, x[1].node_number)
    ):
        print(f"Name: {name}")
        print(f"Level: {config.level}")
        print(f"Node Number: {config.node_number}")
        print(f"Parent: {agent_parents.get(name, 'None')}")
        children = agent_children.get(name, [])
        print(f"Children: {', '.join(children) if children else 'None'}")
        print(f"System Prompt: {config.system_prompt}")
        print("-" * 80)


def visualize_langgraph(
    graph: Any, output_dir: str, filename: str = "execution_graph_lg"
) -> Optional[str]:
    """Visualize a LangGraph structure using Graphviz.

    This function creates a visualization of the LangGraph structure using Graphviz,
    similar to how agent trees are visualized.

    Args:
        graph: The compiled LangGraph to visualize.
        output_dir: Directory to save the visualization.
        filename: Base filename for the visualization (without extension).

    Returns:
        The path to the saved visualization file, or None if visualization failed.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create langgraph-specific output path
    output_path = os.path.join(output_dir, filename)

    try:
        # Create a new Graphviz graph
        dot = graphviz.Digraph(
            "Execution Graph",
            comment="Visualization of the LangGraph structure",
            format="png",
        )

        # Get the nodes and edges from the LangGraph
        lg_graph = graph.get_graph()

        # Set graph attributes for better visualization
        dot.attr(rankdir="TB", size="10,10", ratio="fill")
        dot.attr("node", shape="box", style="filled", fontname="Arial", fontsize="10")
        dot.attr("edge", arrowsize="0.7", fontname="Arial", fontsize="9")

        # Add a title to the graph
        try:
            depth = len(
                {
                    n.split("_")[0][1:2]
                    for n in lg_graph.nodes
                    if n not in ["__start__", "__end__"]
                }
            )
            dot.attr(
                label=f"Execution Graph\nDepth: {depth}\nNodes: {len(lg_graph.nodes)}",
                fontsize="14",
                fontname="Arial Bold",
            )
        except Exception:
            # Fallback if we can't determine the depth
            dot.attr(
                label=f"Execution Graph\nNodes: {len(lg_graph.nodes)}",
                fontsize="14",
                fontname="Arial Bold",
            )

        # Add nodes
        for node in lg_graph.nodes:
            # Use different colors for different node types
            if node == "__start__":
                color = "#e6f7ff"  # Light blue
                shape = "oval"
                label = "START"
            elif node == "__end__":
                color = "#e6ffe6"  # Light green
                shape = "oval"
                label = "END"
            elif node.endswith("_down"):
                color = "#fff2e6"  # Light orange
                shape = "box"
                agent_name = node.split("_down")[0]
                label = f"{agent_name}\n(Down Pass)"
            elif node.endswith("_up"):
                color = "#f7e6ff"  # Light purple
                shape = "box"
                agent_name = node.split("_up")[0]
                label = f"{agent_name}\n(Up Pass)"
            else:
                color = "#f2f2f2"  # Light gray
                shape = "box"
                label = node

            # Add the node with custom attributes
            dot.node(node, label, style="filled", fillcolor=color, shape=shape)

        # Add edges
        for edge in lg_graph.edges:
            # Handle different edge formats
            if isinstance(edge, tuple) and len(edge) == 2:
                source, target = edge
                # Use solid line for regular edges
                dot.edge(source, target, dir="forward")
            elif isinstance(edge, tuple) and len(edge) == 3:
                source, target, _ = edge
                # Use solid line for regular edges
                dot.edge(source, target, dir="forward")
            else:
                # Try to handle Edge objects
                try:
                    source = getattr(edge, "source", None)
                    target = getattr(edge, "target", None)
                    conditional = getattr(edge, "conditional", False)

                    if source and target:
                        # Use dashed line for conditional edges
                        if conditional:
                            dot.edge(source, target, style="dashed", dir="forward")
                        else:
                            dot.edge(source, target, dir="forward")
                    else:
                        logger.warning(
                            f"Could not extract source and target from edge: {edge}"
                        )
                except Exception as e:
                    logger.warning(f"Error processing edge {edge}: {e}")

        # Render the graph
        output_file = dot.render(output_path, cleanup=True)
        logger.info(f"Execution graph visualization saved to {output_file}")
        return output_file
    except Exception as e:
        logger.warning(f"Error visualizing execution graph: {e}")
        return None
