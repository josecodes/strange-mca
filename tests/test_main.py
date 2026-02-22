"""Tests for the main module."""

import os
from unittest.mock import MagicMock, patch

from src.strange_mca.main import create_output_dir, main


def test_create_output_dir():
    """Test the create_output_dir function."""
    with patch("os.makedirs") as mock_makedirs, patch(
        "datetime.datetime"
    ) as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        output_dir = create_output_dir(child_per_parent=2, depth=3, model="gpt-4o-mini")

        expected_dir = os.path.join("output", "20240101_120000_c2_d3_mini")
        assert output_dir == expected_dir
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)


@patch("src.strange_mca.main.argparse.ArgumentParser")
@patch("src.strange_mca.main.load_dotenv")
@patch("src.strange_mca.main.create_output_dir")
@patch("src.strange_mca.main.total_nodes")
def test_main_dry_run(
    mock_total_nodes, mock_create_dir, mock_load_dotenv, mock_arg_parser
):
    """Test the main function with dry_run=True."""
    mock_parser = MagicMock()
    mock_arg_parser.return_value = mock_parser

    mock_args = MagicMock()
    mock_args.task = "Test task"
    mock_args.child_per_parent = 2
    mock_args.depth = 2
    mock_args.model = "gpt-4o-mini"
    mock_args.max_rounds = 3
    mock_args.convergence_threshold = 0.85
    mock_args.no_downward_signals = False
    mock_args.perspectives = None
    mock_args.log_level = "info"
    mock_args.local_logs_only = False
    mock_args.viz = False
    mock_args.dry_run = True
    mock_args.print_tree = False
    mock_args.print_details = False
    mock_args.domain_specific_instructions = ""
    mock_args.strange_loop_count = 0
    mock_parser.parse_args.return_value = mock_args

    mock_total_nodes.return_value = 4
    mock_create_dir.return_value = "output/test_dir"

    with patch("src.strange_mca.main.logger") as mock_logger:
        main()

    mock_logger.info.assert_any_call("Dry run completed")


@patch("src.strange_mca.main.argparse.ArgumentParser")
@patch("src.strange_mca.main.load_dotenv")
@patch("src.strange_mca.main.create_output_dir")
@patch("src.strange_mca.main.total_nodes")
def test_main_function(
    mock_total_nodes, mock_create_dir, mock_load_dotenv, mock_arg_parser
):
    """Test the main function with full execution."""
    mock_parser = MagicMock()
    mock_arg_parser.return_value = mock_parser

    mock_args = MagicMock()
    mock_args.task = "Test task"
    mock_args.child_per_parent = 3
    mock_args.depth = 2
    mock_args.model = "gpt-4o-mini"
    mock_args.max_rounds = 3
    mock_args.convergence_threshold = 0.85
    mock_args.no_downward_signals = False
    mock_args.perspectives = None
    mock_args.log_level = "info"
    mock_args.local_logs_only = False
    mock_args.viz = False
    mock_args.dry_run = False
    mock_args.print_tree = False
    mock_args.print_details = False
    mock_args.domain_specific_instructions = ""
    mock_args.strange_loop_count = 0
    mock_parser.parse_args.return_value = mock_args

    mock_total_nodes.return_value = 4
    mock_create_dir.return_value = "output/test_dir"

    mock_result = {
        "final_response": "Test response",
        "converged": True,
        "convergence_scores": [0.9],
        "current_round": 2,
    }

    with patch(
        "src.strange_mca.run_strange_mca.run_strange_mca", return_value=mock_result
    ) as mock_run, patch("builtins.print") as mock_print:
        main()

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["task"] == "Test task"
    assert call_kwargs["child_per_parent"] == 3
    assert call_kwargs["depth"] == 2
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["max_rounds"] == 3
    assert call_kwargs["enable_downward_signals"] is True

    mock_print.assert_any_call("\nFinal Response:")
    mock_print.assert_any_call("Test response")
