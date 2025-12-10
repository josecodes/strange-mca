"""Tests for the run_strange_mca module."""

from unittest.mock import ANY, MagicMock, mock_open, patch

from src.strange_mca.run_strange_mca import run_strange_mca


@patch("src.strange_mca.run_strange_mca.create_output_dir")
@patch("src.strange_mca.run_strange_mca.total_nodes")
@patch("src.strange_mca.run_strange_mca.visualize_agent_tree")
@patch("src.strange_mca.run_strange_mca.create_execution_graph")
@patch("src.strange_mca.run_strange_mca.run_execution_graph")
@patch("src.strange_mca.run_strange_mca.json.dump")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_run_strange_mca(
    mock_makedirs,
    mock_file_open,
    mock_json_dump,
    mock_run_graph,
    mock_create_graph,
    mock_viz_agent_tree,
    mock_total_nodes,
    mock_create_dir,
):
    """Test the run_strange_mca function."""
    # Mock the output directory
    mock_create_dir.return_value = "output/test_dir"

    # Mock total_nodes
    mock_total_nodes.return_value = 3

    # Skip visualization to avoid TypeError with MagicMock
    mock_viz_agent_tree.return_value = "output/test_dir/agent_tree.png"

    # Mock the execution graph
    mock_graph = MagicMock()
    mock_create_graph.return_value = mock_graph

    # Mock the execution result
    mock_result = {"final_response": "Test response"}
    mock_run_graph.return_value = mock_result

    # Call the function
    result = run_strange_mca(
        task="Test task",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=True,
        local_logs_only=False,
        print_details=False,
        domain_specific_instructions="Be concise.",
        strange_loop_count=1,
    )

    # Check that the output directory was created
    mock_create_dir.assert_called_once_with(2, 2, "gpt-3.5-turbo")

    # Check that total_nodes was called
    mock_total_nodes.assert_called_once_with(2, 2)

    # Check that the execution graph was created
    mock_create_graph.assert_called_once_with(
        child_per_parent=2,
        depth=2,
        model_name="gpt-3.5-turbo",
        langgraph_viz_dir=None,
        domain_specific_instructions="Be concise.",
        strange_loop_count=1,
    )

    # Check that the execution graph was run
    mock_run_graph.assert_called_once_with(
        execution_graph=mock_graph,
        task="Test task",
        log_level="info",
        only_local_logs=False,
        langgraph_viz_dir=None,
    )

    # Check that the result is correct
    assert result == mock_result

    # Check that json.dump was called with the result
    mock_json_dump.assert_called_once_with(mock_result, ANY, indent=2)


@patch("src.strange_mca.run_strange_mca.create_output_dir")
@patch("src.strange_mca.run_strange_mca.total_nodes")
@patch("src.strange_mca.run_strange_mca.visualize_agent_tree")
@patch("src.strange_mca.run_strange_mca.create_execution_graph")
@patch("src.strange_mca.run_strange_mca.visualize_langgraph")
@patch("src.strange_mca.run_strange_mca.run_execution_graph")
@patch("src.strange_mca.run_strange_mca.json.dump")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_run_strange_mca_with_viz(
    mock_makedirs,
    mock_file_open,
    mock_json_dump,
    mock_run_graph,
    mock_viz_langgraph,
    mock_create_graph,
    mock_viz_agent_tree,
    mock_total_nodes,
    mock_create_dir,
):
    """Test the run_strange_mca function with visualization enabled."""
    # Mock the output directory
    mock_create_dir.return_value = "output/test_dir"

    # Mock total_nodes
    mock_total_nodes.return_value = 3

    # Mock the agent tree visualization
    mock_viz_agent_tree.return_value = "output/test_dir/agent_tree.png"

    # Mock the execution graph
    mock_graph = MagicMock()
    mock_create_graph.return_value = mock_graph

    # Mock the execution result
    mock_result = {"final_response": "Test response"}
    mock_run_graph.return_value = mock_result

    # Call the function with viz=True
    result = run_strange_mca(
        task="Test task", child_per_parent=2, depth=2, model="gpt-3.5-turbo", viz=True
    )

    # Check that the agent tree visualization was created
    mock_viz_agent_tree.assert_called_once()
    args, kwargs = mock_viz_agent_tree.call_args
    assert kwargs["cpp"] == 2
    assert kwargs["depth"] == 2
    assert kwargs["format"] == "png"

    # Check that the LangGraph visualization was created
    mock_viz_langgraph.assert_called_once_with(mock_graph, "output/test_dir", 2, 2)


@patch("src.strange_mca.run_strange_mca.create_output_dir")
@patch("src.strange_mca.run_strange_mca.total_nodes")
@patch("src.strange_mca.run_strange_mca.create_execution_graph")
@patch("src.strange_mca.run_strange_mca.run_execution_graph")
@patch("src.strange_mca.run_strange_mca.json.dump")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_run_strange_mca_with_custom_output_dir(
    mock_makedirs,
    mock_file_open,
    mock_json_dump,
    mock_run_graph,
    mock_create_graph,
    mock_total_nodes,
    mock_create_dir,
):
    """Test the run_strange_mca function with a custom output directory."""
    # Mock total_nodes
    mock_total_nodes.return_value = 3

    # Mock the execution graph
    mock_graph = MagicMock()
    mock_create_graph.return_value = mock_graph

    # Mock the execution result
    mock_result = {"final_response": "Test response"}
    mock_run_graph.return_value = mock_result

    # Call the function with a custom output directory
    result = run_strange_mca(task="Test task", output_dir="custom_output_dir")

    # Check that create_output_dir was not called
    mock_create_dir.assert_not_called()

    # Check that json.dump was called with the result
    mock_json_dump.assert_called_once_with(mock_result, ANY, indent=2)
