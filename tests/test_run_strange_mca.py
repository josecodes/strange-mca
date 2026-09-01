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

    # Phase metrics computed because config carries cpp and depth
    phase = report["summary_metrics"]["phase_analysis"]
    # Siblings "leaf 1 revised" vs "leaf 2": intersection {leaf}, union 4 tokens
    assert phase["final_mean_leaf_sibling_similarity"] == 0.25
    assert phase["final_cross_group_similarity"] is None  # only one leaf group
    # No convergence scores -> convergence unmeasurable -> unknown phase
    assert phase["genuinely_converged"] is None
    assert phase["phase_classification"] == "unknown"
    assert report["rounds"][0]["phase_metrics"]["sibling_similarity_by_group"] == {
        "L1N1": 0.25
    }


def test_build_mca_report_no_phase_without_topology():
    """Phase analysis is omitted when config lacks cpp/depth."""
    result = {
        "agent_history": {"L1N1": [{"response": "synth"}]},
        "converged": True,
        "convergence_scores": [],
        "final_response": "final",
    }
    report = build_mca_report(result, "Test", {})

    assert "phase_analysis" not in report["summary_metrics"]
    assert "phase_metrics" not in report["rounds"][0]


def test_build_mca_report_phase_classification_multi_round():
    """Two-round run: trajectories, stability, and classification computed."""
    result = {
        "agent_history": {
            "L1N1": [
                {"response": "root response one"},
                {"response": "root response two"},
            ],
            "L2N1": [
                {"response": "alpha beta"},
                {"response": "alpha beta"},
            ],
            "L2N2": [
                {"response": "gamma delta"},
                {"response": "gamma delta"},
            ],
        },
        "convergence_scores": [0.9],
        "converged": True,
        "final_response": "root response two",
    }
    config = {"cpp": 2, "depth": 2, "convergence_threshold": 0.85}

    report = build_mca_report(result, "Test task", config)
    phase = report["summary_metrics"]["phase_analysis"]

    # Siblings are fully disjoint in both rounds
    assert phase["mean_leaf_sibling_similarity_trajectory"] == [0.0, 0.0]
    assert phase["final_mean_leaf_sibling_similarity"] == 0.0
    # Leaves identical across rounds; the root is excluded from stability
    assert phase["mean_agent_stability"] == 1.0
    # Final score 0.9 >= threshold 0.85 -> genuine convergence
    assert phase["genuinely_converged"] is True
    # Converged with diverse siblings -> the target phase
    assert phase["phase_classification"] == "converged_hierarchical"
    assert len(report["rounds"]) == 2
    assert all("phase_metrics" in rd for rd in report["rounds"])


def test_build_mca_report_round_cap_not_genuine_convergence():
    """The max_rounds cap (converged=True, low score) must not classify as converged."""
    result = {
        "agent_history": {
            "L1N1": [
                {"response": "the deadlock between perspectives persists"},
                {"response": "the deadlock between perspectives remains"},
            ],
            "L2N1": [
                {"response": "alpha beta"},
                {"response": "alpha beta"},
            ],
            "L2N2": [
                {"response": "gamma delta"},
                {"response": "gamma delta"},
            ],
        },
        # Root round-over-round similarity 4/6 — below the 0.85 threshold
        "convergence_scores": [round(4 / 6, 3)],
        "converged": True,  # forced by hitting max_rounds
        "final_response": "the deadlock between perspectives remains",
    }
    config = {"cpp": 2, "depth": 2, "convergence_threshold": 0.85}

    report = build_mca_report(result, "Test task", config)
    phase = report["summary_metrics"]["phase_analysis"]

    assert phase["genuinely_converged"] is False
    # Non-root agents fully settled (root excluded from stability) while the
    # root score stayed below threshold -> stuck (glassy signature)
    assert phase["mean_agent_stability"] == 1.0
    assert phase["phase_classification"] == "stuck"


def test_build_mca_report_oscillating_when_leaves_unstable():
    """Leaves still changing content while root unconverged -> oscillating."""
    result = {
        "agent_history": {
            "L1N1": [
                {"response": "one thing entirely"},
                {"response": "another matter altogether"},
            ],
            "L2N1": [
                {"response": "alpha beta"},
                {"response": "epsilon zeta"},
            ],
            "L2N2": [
                {"response": "gamma delta"},
                {"response": "eta theta"},
            ],
        },
        "convergence_scores": [0.0],
        "converged": True,  # forced by hitting max_rounds
        "final_response": "another matter altogether",
    }
    config = {"cpp": 2, "depth": 2, "convergence_threshold": 0.85}

    report = build_mca_report(result, "Test task", config)
    phase = report["summary_metrics"]["phase_analysis"]

    assert phase["genuinely_converged"] is False
    assert phase["mean_agent_stability"] == 0.0
    assert phase["phase_classification"] == "oscillating"


def test_build_mca_report_two_leaf_groups_depth3():
    """Depth-3 topology: cross-group similarity computed, coordinators excluded
    from the leaf sibling mean."""
    result = {
        "agent_history": {
            "L1N1": [{"response": "root synthesis"}],
            # Coordinators (siblings under L1N1) with disjoint texts — their
            # group similarity (0.0) must not enter the leaf mean
            "L2N1": [{"response": "coord one text"}],
            "L2N2": [{"response": "entirely different synthesis here"}],
            # Leaf group A under L2N1: identical texts -> similarity 1.0
            "L3N1": [{"response": "alpha beta"}],
            "L3N2": [{"response": "alpha beta"}],
            # Leaf group B under L2N2: identical texts -> similarity 1.0
            "L3N3": [{"response": "gamma delta"}],
            "L3N4": [{"response": "gamma delta"}],
        },
        "convergence_scores": [],
        "converged": False,
        "final_response": "root synthesis",
    }
    config = {"cpp": 2, "depth": 3, "convergence_threshold": 0.85}

    report = build_mca_report(result, "Test task", config)
    metrics = report["rounds"][0]["phase_metrics"]

    # All three sibling groups measured individually
    assert metrics["sibling_similarity_by_group"] == {
        "L1N1": 0.0,  # the two coordinators share no tokens
        "L2N1": 1.0,
        "L2N2": 1.0,
    }
    # Leaf mean covers only the two leaf groups, not the coordinator group
    assert metrics["mean_leaf_sibling_similarity"] == 1.0
    # Inter-group leaf pairs ("alpha beta" vs "gamma delta") are disjoint
    assert metrics["cross_group_similarity"] == 0.0

    phase = report["summary_metrics"]["phase_analysis"]
    assert phase["final_cross_group_similarity"] == 0.0


def test_build_mca_report_cpp1_no_sibling_groups():
    """cpp=1 chains have no sibling groups: converged runs must not claim the
    diversity-preserving target phase."""
    result = {
        "agent_history": {
            "L1N1": [
                {"response": "stable root"},
                {"response": "stable root"},
            ],
            "L2N1": [
                {"response": "lone leaf"},
                {"response": "lone leaf"},
            ],
        },
        "convergence_scores": [1.0],
        "converged": True,
        "final_response": "stable root",
    }
    config = {"cpp": 1, "depth": 2, "convergence_threshold": 0.85}

    report = build_mca_report(result, "Test task", config)
    phase = report["summary_metrics"]["phase_analysis"]

    assert phase["final_mean_leaf_sibling_similarity"] is None
    assert phase["genuinely_converged"] is True
    assert phase["phase_classification"] == "converged_unmeasured"


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
