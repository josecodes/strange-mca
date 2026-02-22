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


# =============================================================================
# build_mca_report Edge Case Tests
# =============================================================================


def test_build_mca_report_empty_history():
    """Test build_mca_report with empty agent history."""
    result = {
        "agent_history": {},
        "converged": False,
        "convergence_scores": [],
        "final_response": "",
    }
    report = build_mca_report(result, "Test task", {"cpp": 2, "depth": 2})

    assert len(report["rounds"]) == 0
    assert report["summary_metrics"]["total_llm_calls"] == 0
    assert report["summary_metrics"]["lateral_revision_rate"] == 0.0
    assert report["convergence"]["rounds_used"] == 0


def test_build_mca_report_with_strange_loops():
    """Test that strange_loops are included in report when present."""
    result = {
        "agent_history": {"L1N1": [{"response": "synth"}]},
        "converged": True,
        "convergence_scores": [],
        "final_response": "final",
        "strange_loops": [{"prompt": "loop prompt", "response": "loop response"}],
    }
    report = build_mca_report(result, "Test", {})

    assert "strange_loops" in report
    assert len(report["strange_loops"]) == 1


def test_build_mca_report_missing_optional_fields():
    """Test build_mca_report handles missing optional fields gracefully."""
    result = {
        "agent_history": {"L1N1": [{"response": "synth"}]},
        "final_response": "final",
    }
    report = build_mca_report(result, "Test", {})

    assert report["convergence"]["converged"] is False
    assert report["convergence"]["score_trajectory"] == []
    assert report["final_response"] == "final"
    assert "strange_loops" not in report


def test_build_mca_report_llm_call_counting():
    """Test that total_llm_calls is counted accurately from round data."""
    result = {
        "agent_history": {
            # Root: 1 observe call (response only, no lateral)
            "L1N1": [{"response": "root synth", "revised": False}],
            # Leaf with revision: 1 respond + 1 lateral = 2
            "L2N1": [
                {
                    "response": "initial",
                    "lateral_response": "revised",
                    "revised": True,
                }
            ],
            # Leaf without revision (no siblings, copied): 1 respond + 0 lateral = 1
            "L2N2": [
                {
                    "response": "initial",
                    "lateral_response": "initial",
                    "revised": False,
                }
            ],
            # Leaf with signal_sent: 1 respond + 1 signal = 2
            "L2N3": [
                {
                    "response": "initial",
                    "revised": False,
                    "signal_sent": "explore more",
                }
            ],
        },
        "converged": True,
        "convergence_scores": [],
        "final_response": "final",
    }
    report = build_mca_report(result, "Test", {})

    # L1N1: 1 (response)
    # L2N1: 1 (response) + 1 (revised=True) = 2
    # L2N2: 1 (response) + 0 (revised=False, lateral==response) = 1
    # L2N3: 1 (response) + 0 (revised=False) + 1 (signal_sent) = 2
    # Total: 1 + 2 + 1 + 2 = 6
    assert report["summary_metrics"]["total_llm_calls"] == 6
