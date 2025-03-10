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
    output_file = dot.render(output_path, cleanup=True)
    
    return output_file


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