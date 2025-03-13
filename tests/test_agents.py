"""Tests for the agents module."""

import networkx as nx
import pytest

from src.strange_mca.agents import (
    Agent,
    AgentConfig,
    AgentTree,
    create_agent_configs,
    create_agent_tree,
)


def test_agent_config():
    """Test the AgentConfig class."""
    # Test basic initialization
    config = AgentConfig(name="test_agent", level=1, node_number=1)
    assert config.name == "test_agent"
    assert config.level == 1
    assert config.node_number == 1
    assert config.system_prompt == ""

    # Test with system prompt
    config = AgentConfig(
        name="test_agent",
        level=1,
        node_number=1,
        system_prompt="You are a helpful assistant.",
    )
    assert config.system_prompt == "You are a helpful assistant."

    # Test full_name property
    assert config.full_name == "L1N1"


def test_agent_tree_structure():
    """Test the structure of the AgentTree."""
    # Test with child_per_parent=2, depth=2
    tree = AgentTree(child_per_parent=2, depth=2)

    # Check that the graph is a directed acyclic graph
    assert nx.is_directed_acyclic_graph(tree.mca_graph)

    # Check that we have the correct number of nodes
    assert len(tree.mca_graph.nodes) == 3  # 1 root + 2 children

    # Check that the root node exists and has the correct name
    root = tree.get_root()
    assert root == "L1N1"

    # Check that the root node has the correct number of children
    children = tree.get_children(root)
    assert len(children) == 2
    assert "L2N1" in children
    assert "L2N2" in children

    # Test with child_per_parent=3, depth=3
    tree = AgentTree(child_per_parent=3, depth=3)

    # Check that we have the correct number of nodes
    assert len(tree.mca_graph.nodes) == 13  # 1 + 3 + 9

    # Check leaf nodes
    leaf_nodes = tree.get_leaf_nodes()
    assert len(leaf_nodes) == 9
    for node in leaf_nodes:
        assert node.startswith("L3")


def test_agent_tree_traversal():
    """Test the traversal methods of the AgentTree."""
    tree = AgentTree(child_per_parent=2, depth=3)

    # Test downward traversal
    down_traversal = tree.perform_down_traversal()
    assert len(down_traversal) == 7  # 1 + 2 + 4
    assert down_traversal[0] == "L1N1"  # Root should be first

    # Test upward traversal
    up_traversal = tree.perform_up_traversal()
    assert len(up_traversal) == 7  # 1 + 2 + 4

    # Leaf nodes should come first in upward traversal
    for node in up_traversal[:4]:
        assert node.startswith("L3")

    # Root should be last in upward traversal
    assert up_traversal[-1] == "L1N1"


def test_create_agent_configs():
    """Test the create_agent_configs function."""
    # Test with child_per_parent=2, depth=2
    configs = create_agent_configs(child_per_parent=2, depth=2)

    # Check that we have the correct number of agents
    assert len(configs) == 3  # 1 root + 2 children

    # Check that all configs are AgentConfig objects
    for name, config in configs.items():
        assert isinstance(config, AgentConfig)
        assert config.name == name

    # Test with child_per_parent=3, depth=3
    configs = create_agent_configs(child_per_parent=3, depth=3)
    assert len(configs) == 13  # 1 + 3 + 9


def test_create_agent_tree():
    """Test the create_agent_tree function."""
    tree = create_agent_tree(child_per_parent=2, depth=2)
    assert isinstance(tree, AgentTree)
    assert tree.child_per_parent == 2
    assert tree.depth == 2
    assert len(tree.mca_graph.nodes) == 3  # 1 root + 2 children


@pytest.mark.skip(reason="Requires OpenAI API key and makes actual API calls")
def test_agent_run():
    """Test the Agent.run method."""
    # This test is skipped by default as it requires an OpenAI API key
    # and makes actual API calls
    config = AgentConfig(name="test_agent", level=1, node_number=1)
    agent = Agent(config, model_name="gpt-3.5-turbo")
    response = agent.run(task="Say hello")
    assert isinstance(response, str)
    assert len(response) > 0
