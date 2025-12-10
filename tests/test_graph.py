"""Tests for the graph module."""

from unittest.mock import MagicMock, patch

import pytest

from src.strange_mca.graph import (
    State,
    create_execution_graph,
    run_execution_graph,
    parse_node_name,
    make_node_name,
    get_children,
    is_leaf,
    is_root,
    count_nodes_at_level,
    total_nodes,
)


# =============================================================================
# Tree Helper Function Tests
# =============================================================================


def test_parse_node_name():
    """Test parsing node names."""
    assert parse_node_name("L1N1") == (1, 1)
    assert parse_node_name("L2N3") == (2, 3)
    assert parse_node_name("L3N7") == (3, 7)
    assert parse_node_name("L10N100") == (10, 100)


def test_make_node_name():
    """Test making node names."""
    assert make_node_name(1, 1) == "L1N1"
    assert make_node_name(2, 3) == "L2N3"
    assert make_node_name(3, 7) == "L3N7"


def test_get_children():
    """Test getting children of a node."""
    # Root with 2 children per parent, depth 3
    children = get_children("L1N1", cpp=2, depth=3)
    assert children == ["L2N1", "L2N2"]

    # L2N1 with 2 children per parent, depth 3
    children = get_children("L2N1", cpp=2, depth=3)
    assert children == ["L3N1", "L3N2"]

    # L2N2 with 2 children per parent, depth 3
    children = get_children("L2N2", cpp=2, depth=3)
    assert children == ["L3N3", "L3N4"]

    # Leaf node (no children)
    children = get_children("L3N1", cpp=2, depth=3)
    assert children == []

    # Root with 3 children per parent
    children = get_children("L1N1", cpp=3, depth=2)
    assert children == ["L2N1", "L2N2", "L2N3"]


def test_is_leaf():
    """Test leaf node detection."""
    assert is_leaf(level=3, depth=3) is True
    assert is_leaf(level=2, depth=3) is False
    assert is_leaf(level=1, depth=3) is False
    assert is_leaf(level=1, depth=1) is True


def test_is_root():
    """Test root node detection."""
    assert is_root(level=1) is True
    assert is_root(level=2) is False
    assert is_root(level=3) is False


def test_count_nodes_at_level():
    """Test counting nodes at each level."""
    # cpp=2
    assert count_nodes_at_level(level=1, cpp=2) == 1
    assert count_nodes_at_level(level=2, cpp=2) == 2
    assert count_nodes_at_level(level=3, cpp=2) == 4

    # cpp=3
    assert count_nodes_at_level(level=1, cpp=3) == 1
    assert count_nodes_at_level(level=2, cpp=3) == 3
    assert count_nodes_at_level(level=3, cpp=3) == 9


def test_total_nodes():
    """Test total node count."""
    # cpp=2, depth=2: 1 + 2 = 3
    assert total_nodes(cpp=2, depth=2) == 3

    # cpp=2, depth=3: 1 + 2 + 4 = 7
    assert total_nodes(cpp=2, depth=3) == 7

    # cpp=3, depth=2: 1 + 3 = 4
    assert total_nodes(cpp=3, depth=2) == 4

    # cpp=3, depth=3: 1 + 3 + 9 = 13
    assert total_nodes(cpp=3, depth=3) == 13


# =============================================================================
# State Tests
# =============================================================================


def test_state_type():
    """Test the State type."""
    # Create a valid State (depth/cpp are build-time constants, not in state)
    state: State = {
        "task": "Test task",
        "original_task": "Test task",
        "response": "",
    }

    # Check that the state has the expected keys
    assert "task" in state
    assert "original_task" in state


# =============================================================================
# Graph Creation Tests
# =============================================================================


@pytest.mark.skip(reason="Requires LangGraph and makes actual API calls")
def test_create_execution_graph():
    """Test the create_execution_graph function."""
    # This test is skipped by default as it requires LangGraph and makes actual API calls
    graph = create_execution_graph(
        child_per_parent=2, depth=2, model_name="gpt-3.5-turbo"
    )

    # Check that the graph has the expected attributes
    assert hasattr(graph, "get_graph")
    assert hasattr(graph, "invoke")


@patch("src.strange_mca.graph.Agent")
def test_create_execution_graph_structure(mock_agent_class):
    """Test the structure creation in create_execution_graph."""
    # Mock the Agent class
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Test response"
    mock_agent_class.return_value = mock_agent

    # Create the graph
    graph = create_execution_graph(
        child_per_parent=2, depth=2, model_name="gpt-3.5-turbo"
    )

    # Check that the graph was created
    assert graph is not None

    # Check that Agent was instantiated (3 agents for cpp=2, depth=2)
    assert mock_agent_class.call_count == 3


@patch("src.strange_mca.graph.setup_detailed_logging")
def test_run_execution_graph_setup(mock_setup_logging):
    """Test the setup phase of run_execution_graph."""
    # Create a mock graph
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"response": "Test response", "final_response": "Test response"}

    # Run the function
    result = run_execution_graph(
        execution_graph=mock_graph,
        task="Test task",
        log_level="info",
        only_local_logs=True,
    )

    # Check that logging was set up
    mock_setup_logging.assert_called_once_with(log_level="info", only_local_logs=True)

    # Check that the graph was invoked
    mock_graph.invoke.assert_called_once()

    # Check that the result contains the expected keys
    assert "final_response" in result
    assert result["final_response"] == "Test response"
