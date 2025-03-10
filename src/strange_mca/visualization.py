"""Visualization utilities for the multiagent system."""

import os
import logging
from typing import Dict, List, Optional, Any

import graphviz

from src.strange_mca.agents import AgentConfig

# Set up logger
logger = logging.getLogger("strange_mca")


def visualize_agent_graph(
    agent_configs: Dict[str, AgentConfig],
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
    
    # Build a tree structure from the agent_configs
    tree = {}
    for name, config in agent_configs.items():
        level = config.level
        if level not in tree:
            tree[level] = []
        tree[level].append(name)
    
    # Build parent-child relationships
    children = {}
    for level in sorted(tree.keys())[1:]:  # Skip level 1 (root)
        parent_level = level - 1
        for name in tree[level]:
            # Extract parent information from the name
            # Assuming names follow the pattern "L{level}N{number}"
            node_number = int(name.split('N')[1])
            parent_number = (node_number + 1) // 2  # Simple formula to find parent
            parent_name = f"L{parent_level}N{parent_number}"
            
            if parent_name not in children:
                children[parent_name] = []
            children[parent_name].append(name)
    
    # Create a new graph
    dot = graphviz.Digraph(
        "Agent Graph",
        comment="Visualization of the multiagent system",
        format=format,
    )
    
    # Add nodes
    for name, config in agent_configs.items():
        # Create a label with the agent's name and level
        label = f"{name}\nLevel {config.level}"
        
        # Use different colors for different levels
        color = "lightblue" if config.level == 1 else "lightgreen"
        
        # Add the node
        dot.node(name, label, style="filled", fillcolor=color)
    
    # Add edges
    for parent, child_list in children.items():
        for child in child_list:
            dot.edge(parent, child)
    
    # Render the graph
    try:
        output_file = dot.render(output_path, cleanup=True)
        return output_file
    except Exception as dot_e:
        logger.warning(f"Could not convert dot to image using Graphviz: {dot_e}")
        logger.info("You can manually convert the dot file using Graphviz or an online converter.")
        logger.info(f"For example: dot -Tpng {output_path}.dot -o {output_path}.png")
        return f"{output_path}.dot"


def print_agent_tree(agent_configs: Dict[str, AgentConfig]) -> None:
    """Print the agent tree structure.
    
    Args:
        agent_configs: Dictionary mapping agent names to their configurations.
    """
    # Build a tree structure from the agent_configs
    tree = {}
    for name, config in agent_configs.items():
        level = config.level
        if level not in tree:
            tree[level] = []
        tree[level].append(name)
    
    # Find the root node (the only node at level 1)
    if 1 not in tree or len(tree[1]) != 1:
        print("Error: Could not find a unique root node at level 1")
        return
    
    root = tree[1][0]
    
    # Build parent-child relationships
    children = {}
    for level in sorted(tree.keys())[1:]:  # Skip level 1 (root)
        parent_level = level - 1
        for name in tree[level]:
            # Extract parent information from the name
            # Assuming names follow the pattern "L{level}N{number}"
            node_number = int(name.split('N')[1])
            parent_number = (node_number + 1) // 2  # Simple formula to find parent
            parent_name = f"L{parent_level}N{parent_number}"
            
            if parent_name not in children:
                children[parent_name] = []
            children[parent_name].append(name)
    
    def print_node(name: str, indent: int = 0) -> None:
        """Print a node and its children recursively.
        
        Args:
            name: The name of the node.
            indent: The indentation level.
        """
        config = agent_configs[name]
        print(f"{'  ' * indent}└─ {name} (Level {config.level})")
        
        for child in children.get(name, []):
            print_node(child, indent + 1)
    
    print("Agent Tree:")
    print_node(root)


def print_agent_details(agent_configs: Dict[str, AgentConfig]) -> None:
    """Print details about each agent.
    
    Args:
        agent_configs: The agent configurations.
    """
    print("Agent Details:")
    print("=" * 80)
    
    for name, config in sorted(
        agent_configs.items(),
        key=lambda x: (x[1].level, x[1].node_number)
    ):
        print(f"Name: {name}")
        print(f"Level: {config.level}")
        print(f"Node Number: {config.node_number}")
        print(f"Parent: {config.parent or 'None'}")
        print(f"Children: {', '.join(config.children) or 'None'}")
        print(f"System Prompt: {config.system_prompt}")
        print("-" * 80)


def visualize_langgraph(
    graph: Any,
    output_dir: str,
    filename: str = "langgraph_structure",
    use_local_rendering: bool = True  # Keep for backward compatibility
) -> Optional[str]:
    """Visualize a LangGraph structure using Graphviz.
    
    This function creates a visualization of the LangGraph structure using Graphviz,
    similar to how agent trees are visualized.
    
    Args:
        graph: The compiled LangGraph to visualize.
        output_dir: Directory to save the visualization.
        filename: Base filename for the visualization (without extension).
        use_local_rendering: Deprecated. Always uses local Graphviz rendering now.
        
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
            "LangGraph Structure",
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
            depth = len(set([n.split('_')[0][1:2] for n in lg_graph.nodes if n not in ['__start__', '__end__']]))
            dot.attr(label=f"LangGraph Structure\nDepth: {depth}\nNodes: {len(lg_graph.nodes)}", fontsize="14", fontname="Arial Bold")
        except Exception:
            # Fallback if we can't determine the depth
            dot.attr(label=f"LangGraph Structure\nNodes: {len(lg_graph.nodes)}", fontsize="14", fontname="Arial Bold")
        
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
                    source = getattr(edge, 'source', None)
                    target = getattr(edge, 'target', None)
                    conditional = getattr(edge, 'conditional', False)
                    
                    if source and target:
                        # Use dashed line for conditional edges
                        if conditional:
                            dot.edge(source, target, style="dashed", dir="forward")
                        else:
                            dot.edge(source, target, dir="forward")
                    else:
                        logger.warning(f"Could not extract source and target from edge: {edge}")
                except Exception as e:
                    logger.warning(f"Error processing edge {edge}: {e}")
        
        # Render the graph
        output_file = dot.render(output_path, cleanup=True)
        logger.info(f"LangGraph visualization saved to {output_file}")
        
