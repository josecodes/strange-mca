"""Tests for the graph module."""

from unittest.mock import MagicMock, patch

import pytest

from src.strange_mca.graph import State, create_execution_graph, run_execution_graph


def test_state_type():
    """Test the State type."""
    # Create a valid State
    state = State(
        original_task="Test task",
        nodes={"L1N1_down": {"task": "Test task"}},
        current_node="L1N1_down",
        final_response="",
        strange_loops=[],
    )

    # Check that the state has the expected keys
    assert "original_task" in state
    assert "nodes" in state
    assert "current_node" in state
    assert "final_response" in state
    assert "strange_loops" in state


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


@patch("src.strange_mca.graph.create_execution_graph")
@patch("src.strange_mca.graph.setup_detailed_logging")
def test_run_execution_graph_setup(mock_setup_logging, mock_create_graph):
    """Test the setup phase of run_execution_graph."""
    # Create a mock graph
    mock_graph = MagicMock()
    mock_graph.get_graph.return_value.nodes = ["L1N1_down"]
    mock_graph.invoke.return_value = {"final_response": "Test response"}
    mock_create_graph.return_value = mock_graph

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


@patch("src.strange_mca.graph.create_agent_tree")
def test_create_execution_graph_structure(mock_create_agent_tree):
    """Test the structure creation in create_execution_graph."""
    # Create a mock agent tree
    mock_tree = MagicMock()
    mock_tree.get_root.return_value = "L1N1"
    mock_tree.perform_down_traversal.return_value = ["L1N1", "L2N1", "L2N2"]
    mock_tree.perform_up_traversal.return_value = ["L2N1", "L2N2", "L1N1"]
    mock_tree.mca_graph.nodes.return_value = ["L1N1", "L2N1", "L2N2"]
    mock_create_agent_tree.return_value = mock_tree

    # Mock the StateGraph
    with patch("src.strange_mca.graph.StateGraph") as mock_state_graph:
        # Mock the graph builder
        mock_builder = MagicMock()
        mock_state_graph.return_value = mock_builder

        # Call the function
        create_execution_graph(child_per_parent=2, depth=2, model_name="gpt-3.5-turbo")

        # Check that the agent tree was created
        mock_create_agent_tree.assert_called_once_with(2, 2)

        # Check that the graph builder was created
        mock_state_graph.assert_called_once()

        # Check that the entry point was set
        mock_builder.set_entry_point.assert_called_once_with("L1N1_down")

        # Check that the graph was compiled
        mock_builder.compile.assert_called_once()
