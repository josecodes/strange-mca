"""Tests for the main module."""

import os
from unittest.mock import MagicMock, mock_open, patch

from src.strange_mca.main import create_output_dir, main


def test_create_output_dir():
    """Test the create_output_dir function."""
    with patch("os.makedirs") as mock_makedirs, patch(
        "datetime.datetime"
    ) as mock_datetime:
        # Mock the datetime to return a fixed timestamp
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        # Call the function
        output_dir = create_output_dir(
            child_per_parent=2, depth=3, model="gpt-3.5-turbo"
        )

        # Check that the output directory has the expected format
        expected_dir = os.path.join("output", "20240101_120000_c2_d3_turbo")
        assert output_dir == expected_dir

        # Check that os.makedirs was called with the expected arguments
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)


@patch("src.strange_mca.main.argparse.ArgumentParser")
@patch("src.strange_mca.main.load_dotenv")
@patch("src.strange_mca.main.create_output_dir")
@patch("src.strange_mca.main.total_nodes")
@patch("src.strange_mca.main.create_execution_graph")
@patch("src.strange_mca.main.run_execution_graph")
@patch("builtins.open", new_callable=mock_open)
@patch("os.path.join", return_value="output/test_dir/final_state.json")
@patch("os.makedirs")
def test_main_function(
    mock_makedirs,
    mock_path_join,
    mock_file_open,
    mock_run_graph,
    mock_create_graph,
    mock_total_nodes,
    mock_create_dir,
    mock_load_dotenv,
    mock_arg_parser,
):
    """Test the main function."""
    # Mock the argument parser
    mock_parser = MagicMock()
    mock_arg_parser.return_value = mock_parser

    # Mock the parsed arguments
    mock_args = MagicMock()
    mock_args.task = "Test task"
    mock_args.child_per_parent = 2
    mock_args.depth = 2
    mock_args.model = "gpt-3.5-turbo"
    mock_args.log_level = "info"
    mock_args.local_logs_only = False
    mock_args.viz = False
    mock_args.dry_run = False
    mock_args.print_tree = False
    mock_args.print_details = False
    mock_args.domain_specific_instructions = ""
    mock_args.strange_loop_count = 0
    mock_parser.parse_args.return_value = mock_args

    # Mock total_nodes
    mock_total_nodes.return_value = 3

    # Mock the output directory
    mock_create_dir.return_value = "output/test_dir"

    # Mock the execution graph
    mock_graph = MagicMock()
    mock_create_graph.return_value = mock_graph

    # Mock the execution result
    mock_run_graph.return_value = {"final_response": "Test response"}

    # Call the main function
    with patch("builtins.print") as mock_print:
        main()

    # Check that dotenv was loaded
    mock_load_dotenv.assert_called_once()

    # Check that the argument parser was created and used
    mock_arg_parser.assert_called_once()
    mock_parser.parse_args.assert_called_once()

    # Check that the output directory was created
    mock_create_dir.assert_called_once_with(
        mock_args.child_per_parent, mock_args.depth, mock_args.model
    )

    # Check that the execution graph was created
    mock_create_graph.assert_called_once_with(
        child_per_parent=mock_args.child_per_parent,
        depth=mock_args.depth,
        model_name=mock_args.model,
        langgraph_viz_dir=None,
        domain_specific_instructions=mock_args.domain_specific_instructions,
        strange_loop_count=mock_args.strange_loop_count,
    )

    # Check that the execution graph was run
    mock_run_graph.assert_called_once_with(
        execution_graph=mock_graph,
        task=mock_args.task,
        log_level=mock_args.log_level,
        only_local_logs=mock_args.local_logs_only,
        langgraph_viz_dir=None,
    )

    # Check that the final response was printed
    mock_print.assert_any_call("\nFinal Response:")
    mock_print.assert_any_call("=" * 80)
    mock_print.assert_any_call("Test response")

    # Check that the file was opened for writing
    mock_file_open.assert_called_once_with("output/test_dir/final_state.json", "w")


@patch("src.strange_mca.main.argparse.ArgumentParser")
@patch("src.strange_mca.main.load_dotenv")
@patch("src.strange_mca.main.create_output_dir")
@patch("src.strange_mca.main.total_nodes")
def test_main_dry_run(
    mock_total_nodes, mock_create_dir, mock_load_dotenv, mock_arg_parser
):
    """Test the main function with dry_run=True."""
    # Mock the argument parser
    mock_parser = MagicMock()
    mock_arg_parser.return_value = mock_parser

    # Mock the parsed arguments
    mock_args = MagicMock()
    mock_args.task = "Test task"
    mock_args.child_per_parent = 2
    mock_args.depth = 2
    mock_args.model = "gpt-3.5-turbo"
    mock_args.log_level = "info"
    mock_args.local_logs_only = False
    mock_args.viz = False
    mock_args.dry_run = True  # Set dry_run to True
    mock_args.print_tree = False
    mock_args.print_details = False
    mock_parser.parse_args.return_value = mock_args

    # Mock total_nodes
    mock_total_nodes.return_value = 3

    # Mock the output directory
    mock_create_dir.return_value = "output/test_dir"

    # Call the main function
    with patch("src.strange_mca.main.logger") as mock_logger:
        main()

    # Check that the logger reported a dry run
    mock_logger.info.assert_any_call("Dry run completed")
