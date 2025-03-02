"""Visualization utilities for the multiagent system."""

import os
from typing import Dict, List, Optional

import graphviz

from src.strange_mca.agents import AgentConfig


def visualize_agent_graph(
    agent_configs: Dict[str, AgentConfig],
    output_path: str = None,
    format: str = "png",
) -> Optional[str]:
    """Visualize the agent graph.
    
    Args:
        agent_configs: The agent configurations.
        output_path: The path to save the visualization.
        format: The format to save the visualization in.
        
    Returns:
        The path to the saved visualization.
    """
    # Ensure the viz_outputs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
    for name, config in agent_configs.items():
        for child in config.children:
            dot.edge(name, child)
    
    # Render the graph
    output_file = dot.render(output_path, cleanup=True)
    
    return output_file


def print_agent_tree(agent_configs: Dict[str, AgentConfig]) -> None:
    """Print the agent tree in a hierarchical format.
    
    Args:
        agent_configs: The agent configurations.
    """
    # Find the root node
    root = next((name for name, config in agent_configs.items() if config.level == 1), None)
    
    if not root:
        print("No root node found.")
        return
    
    def print_node(name: str, indent: int = 0) -> None:
        """Print a node and its children recursively.
        
        Args:
            name: The name of the node.
            indent: The indentation level.
        """
        config = agent_configs[name]
        print(f"{'  ' * indent}└─ {name} (Level {config.level})")
        
        for child in config.children:
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
        print(f"System Prompt: {config.system_prompt[:100]}...")
        print("-" * 80) 