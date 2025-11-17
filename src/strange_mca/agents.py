"""Agent definitions for the multiagent system."""

import os
from typing import Optional

import graphviz  # Use graphviz directly
import networkx as nx
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for an agent in the system."""

    name: str
    level: int
    node_number: int
    system_prompt: Optional[str] = ""

    @property
    def full_name(self) -> str:
        """Return the full name of the agent (e.g., L1N1)."""
        return f"L{self.level}N{self.node_number}"


class Agent:
    """An agent in the multiagent system."""

    def __init__(self, config: AgentConfig, model_name: str = "gpt-3.5-turbo"):
        """Initialize the agent.

        Args:
            config: The agent configuration.
            model_name: The name of the LLM model to use.
        """
        self.config = config
        # Use a real OpenAI LLM
        # GPT-5 series only supports default temperature (1), so explicitly set it to 1.0
        model_name_normalized = model_name.strip().lower()
        if model_name_normalized.startswith("gpt-5"):
            self.llm = ChatOpenAI(model_name=model_name, temperature=1.0)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

        # Create the prompt template for this agent
        self.prompt_template = PromptTemplate.from_template(
            template=("Task: {task}\n\n" "Your response:")
        )

    def run(self, task: str) -> str:
        """Run the agent on a task.

        Args:
            task: The task to perform.

        Returns:
            The agent's response.
        """
        # Format the prompt
        prompt = self.prompt_template.format(task=task)

        # Create messages
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=prompt),
        ]

        # Invoke the LLM
        response = self.llm.invoke(messages)
        return response.content

    def __repr__(self) -> str:
        return f"Agent({self.config.full_name})"


class AgentTree:
    """A tree of agents using NetworkX for structure management."""

    def __init__(self, child_per_parent: int, depth: int):
        """Initialize the agent tree.

        Args:
            child_per_parent: The number of children each non-leaf node has.
            depth: The number of levels in the tree.
        """
        self.child_per_parent = child_per_parent
        self.depth = depth
        self.mca_graph = nx.DiGraph()

        # Build the tree structure
        self._build_tree()

    def _build_tree(self):
        """Build the tree structure with agent configurations."""
        # Create the root node
        root_name = "L1N1"
        root_config = AgentConfig(name=root_name, level=1, node_number=1)
        # Add the root node to the graph with its config as a node attribute
        self.mca_graph.add_node(root_name, config=root_config)

        # Create the rest of the tree
        for level in range(2, self.depth + 1):
            parent_level = level - 1
            parent_count = self.child_per_parent ** (parent_level - 1)

            for parent_idx in range(1, parent_count + 1):
                parent_name = f"L{parent_level}N{parent_idx}"

                for child_idx in range(1, self.child_per_parent + 1):
                    node_number = ((parent_idx - 1) * self.child_per_parent) + child_idx
                    child_name = f"L{level}N{node_number}"

                    child_config = AgentConfig(
                        name=child_name, level=level, node_number=node_number
                    )

                    # Add the child node to the graph with its config as a node attribute
                    self.mca_graph.add_node(child_name, config=child_config)

                    # Add the edge from parent to child
                    self.mca_graph.add_edge(parent_name, child_name)

        # Validate the graph
        if not nx.is_directed_acyclic_graph(self.mca_graph):
            raise ValueError("The tree structure contains cycles, which is not allowed")

    def get_config(self, node_name: str) -> AgentConfig:
        """Get the configuration for a node."""
        return self.mca_graph.nodes[node_name]["config"]

    def get_configs(self) -> dict[str, AgentConfig]:
        """Get all configurations as a dictionary."""
        return {node: data["config"] for node, data in self.mca_graph.nodes(data=True)}

    def is_leaf(self, node_name: str) -> bool:
        """Return whether a node is a leaf node."""
        return len(list(self.mca_graph.successors(node_name))) == 0

    def is_root(self, node_name: str) -> bool:
        """Return whether a node is the root node."""
        return len(list(self.mca_graph.predecessors(node_name))) == 0

    def get_root(self) -> str:
        """Get the name of the root node."""
        for node in self.mca_graph.nodes():
            if self.is_root(node):
                return node
        return None

    def get_leaf_nodes(self) -> list[str]:
        """Get the names of all leaf nodes."""
        return [node for node in self.mca_graph.nodes() if self.is_leaf(node)]

    def get_children(self, node_name: str) -> list[str]:
        """Get the children of a node."""
        return list(self.mca_graph.successors(node_name))

    def get_parent(self, node_name: str) -> Optional[str]:
        """Get the parent of a node."""
        parents = list(self.mca_graph.predecessors(node_name))
        return parents[0] if parents else None

    def perform_down_traversal(
        self, start_node: Optional[str] = None, node_callback: Optional[callable] = None
    ) -> list[str]:
        """Perform a downward traversal of the tree.

        Args:
            start_node: The node to start from. If None, starts from the root.
            node_callback: Optional callback function to execute at each node.
                           The callback should accept two arguments: the node name and the predecessor node.

        Returns:
            List of nodes in traversal order.
        """
        if start_node is None:
            start_node = self.get_root()
            if start_node is None:
                raise ValueError("No root node found in the tree")

        # Use NetworkX's BFS to traverse the tree downward
        traversal = list(nx.bfs_tree(self.mca_graph, source=start_node))

        # Execute callback for each node if provided
        if node_callback:
            predecessor_node = None
            for node in traversal:
                node_callback(node, predecessor_node, "down")
                predecessor_node = node

        return traversal

    def perform_up_traversal(
        self, node_callback: Optional[callable] = None
    ) -> list[str]:
        """Perform an upward traversal of the tree using breadth-first walk.

        Args:
            node_callback: Optional callback function to execute at each node.
                           The callback should accept three arguments: the node name and the predecessor node.

        Returns:
            List of nodes in traversal order.
        """
        leaf_nodes = self.get_leaf_nodes()

        # Create a reversed graph for upward traversal
        reversed_graph = self.mca_graph.reverse()

        # Track processed nodes and their order
        processed = []
        visited = set()

        # Initialize queue with all leaf nodes
        from collections import deque

        queue = deque(leaf_nodes)

        # Add all leaf nodes to visited set
        for leaf in leaf_nodes:
            visited.add(leaf)

        # Perform breadth-first traversal
        predecessor_node = None
        while queue:
            current = queue.popleft()
            processed.append(current)

            # Execute callback for the current node if provided
            if node_callback:
                node_callback(current, predecessor_node, "up")
                predecessor_node = current
            # Get the parent in the original graph (successor in reversed graph)
            for parent in reversed_graph.successors(current):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)

        return processed

    def visualize(self, output_dir: str = None, filename: str = "agent_tree"):
        """Visualize the tree structure using graphviz.

        Args:
            output_dir: Directory to save the visualization. If None, just displays it.
            filename: Name of the file to save (without extension).

        Returns:
            Path to the saved visualization file if output_dir is provided, None otherwise.
        """
        # Create a graphviz Digraph
        dot = graphviz.Digraph(comment="Agent Tree")

        # Add nodes
        for node_name in self.get_configs():
            # Determine node shape and color based on type
            if self.is_root(node_name):
                node_shape = "doubleoctagon"
                node_color = "lightblue"
                node_label = f"{node_name}\n(Root)"
            elif self.is_leaf(node_name):
                node_shape = "box"
                node_color = "lightgreen"
                node_label = f"{node_name}\n(Leaf)"
            else:
                node_shape = "ellipse"
                node_color = "lightyellow"
                node_label = f"{node_name}\n(Internal)"

            # Add the node with appropriate attributes
            dot.node(
                node_name,
                label=node_label,
                shape=node_shape,
                style="filled",
                fillcolor=node_color,
            )

        # Add edges
        for edge in self.mca_graph.edges():
            source, target = edge
            dot.edge(source, target)

        # Set graph attributes for better layout
        dot.graph_attr["rankdir"] = "TB"  # Top to bottom layout
        dot.graph_attr["nodesep"] = "0.5"
        dot.graph_attr["ranksep"] = "0.7"
        dot.graph_attr["splines"] = "ortho"  # Orthogonal edges

        # Add a title
        dot.attr(
            label=f"Agent Tree (Depth: {self.depth}, Children per Parent: {self.child_per_parent})"
        )
        dot.attr(fontsize="20")

        # Render the graph
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            dot.render(output_path, format="png", cleanup=True)
            return f"{output_path}.png"
        else:
            # Just display the graph
            dot.view(cleanup=True)
            return None


def create_agent_configs(child_per_parent: int, depth: int) -> dict[str, AgentConfig]:
    """Create agent configurations for a tree with the given parameters.

    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.

    Returns:
        A dictionary mapping agent names to their configurations.
    """
    # Create an agent tree and return its configs
    tree = AgentTree(child_per_parent, depth)
    return tree.get_configs()


def create_agent_tree(child_per_parent: int, depth: int) -> AgentTree:
    """Create an agent tree with the given parameters.

    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.

    Returns:
        An AgentTree object.
    """
    return AgentTree(child_per_parent, depth)
