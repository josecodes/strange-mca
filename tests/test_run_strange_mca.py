"""Tests for the run_strange_mca module."""

from unittest.mock import MagicMock, mock_open, patch

from src.strange_mca.run_strange_mca import build_mca_report, run_strange_mca

# =============================================================================
# build_mca_report Tests
# =============================================================================


def test_build_mca_report():
    """Test building an MCA report from results."""
    result = {
        "agent_history": {
            "L1N1": [{"response": "root response", "revised": False}],
            "L2N1": [
                {
                    "response": "leaf 1",
                    "lateral_response": "leaf 1 revised",
                    "revised": True,
                }
            ],
            "L2N2": [
                {"response": "leaf 2", "lateral_response": "leaf 2", "revised": False}
            ],
        },
        "convergence_scores": [],
        "converged": True,
        "final_response": "root response",
    }
    config = {"cpp": 2, "depth": 2, "model": "gpt-4o-mini"}

    report = build_mca_report(result, "Test task", config)

    assert report["task"] == "Test task"
    assert report["config"] == config
    assert len(report["rounds"]) == 1
    assert report["convergence"]["converged"] is True
    assert report["convergence"]["rounds_used"] == 1
    assert report["final_response"] == "root response"
    assert "summary_metrics" in report
    assert report["summary_metrics"]["per_agent_revision_counts"]["L2N1"] == 1
    assert report["summary_metrics"]["per_agent_revision_counts"]["L2N2"] == 0


# =============================================================================
# run_strange_mca Tests
# =============================================================================


@patch("src.strange_mca.run_strange_mca.create_output_dir")
@patch("src.strange_mca.run_strange_mca.total_nodes")
@patch("src.strange_mca.run_strange_mca.build_agent_tree")
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
    mock_build_tree,
    mock_total_nodes,
    mock_create_dir,
):
    """Test the run_strange_mca function."""
    mock_create_dir.return_value = "output/test_dir"
    mock_total_nodes.return_value = 4

    # Mock agent tree
    mock_agents = {
        "L1N1": MagicMock(),
        "L2N1": MagicMock(),
        "L2N2": MagicMock(),
        "L2N3": MagicMock(),
    }
    for name, agent in mock_agents.items():
        agent.config.perspective = "analytical" if name != "L1N1" else ""
    mock_build_tree.return_value = mock_agents

    # Mock graph
    mock_graph = MagicMock()
    mock_create_graph.return_value = (mock_graph, 50)

    # Mock result
    mock_result = {
        "final_response": "Test response",
        "agent_history": {},
        "convergence_scores": [],
        "converged": True,
    }
    mock_run_graph.return_value = mock_result

    result = run_strange_mca(
        task="Test task",
        child_per_parent=3,
        depth=2,
        model="gpt-4o-mini",
        max_rounds=3,
        convergence_threshold=0.85,
        enable_downward_signals=True,
    )

    mock_create_dir.assert_called_once_with(3, 2, "gpt-4o-mini")
    mock_total_nodes.assert_called_once_with(3, 2)
    mock_build_tree.assert_called_once_with(
        cpp=3,
        depth=2,
        model_name="gpt-4o-mini",
        perspectives=None,
    )
    mock_create_graph.assert_called_once()
    mock_run_graph.assert_called_once()
    assert result == mock_result
    # json.dump called twice: report + state
    assert mock_json_dump.call_count == 2


@patch("src.strange_mca.run_strange_mca.create_output_dir")
@patch("src.strange_mca.run_strange_mca.total_nodes")
@patch("src.strange_mca.run_strange_mca.build_agent_tree")
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
    mock_build_tree,
    mock_total_nodes,
    mock_create_dir,
):
    """Test run_strange_mca with a custom output directory."""
    mock_total_nodes.return_value = 4
    mock_agents = {"L1N1": MagicMock()}
    mock_agents["L1N1"].config.perspective = ""
    mock_build_tree.return_value = mock_agents
    mock_create_graph.return_value = (MagicMock(), 50)
    mock_run_graph.return_value = {
        "final_response": "Test response",
        "agent_history": {},
        "convergence_scores": [],
        "converged": True,
    }

    result = run_strange_mca(task="Test task", output_dir="custom_output_dir")

    mock_create_dir.assert_not_called()
    assert mock_json_dump.call_count == 2
