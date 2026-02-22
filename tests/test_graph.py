"""Tests for the graph module (emergent MCA flat graph)."""

from unittest.mock import MagicMock, patch

from src.strange_mca.graph import (
    MCAState,
    _apply_strange_loop,
    create_execution_graph,
    merge_dicts,
    run_execution_graph,
)
from tests.conftest import build_mock_agents_depth2_cpp3, make_mock_agent

# =============================================================================
# merge_dicts Tests
# =============================================================================


def test_merge_dicts():
    """Test merge_dicts reducer."""
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
    assert merge_dicts(None, {"a": 1}) == {"a": 1}
    assert merge_dicts({"a": 1}, None) == {"a": 1}


# =============================================================================
# State Tests
# =============================================================================


def test_mca_state_type():
    """Test the MCAState type."""
    state: MCAState = {
        "original_task": "Test task",
        "agent_history": {},
        "current_round": 1,
    }
    assert "original_task" in state
    assert "agent_history" in state


# =============================================================================
# Graph Creation Tests
# =============================================================================


def test_create_execution_graph_depth2_cpp3():
    """Test graph creation for depth=2, cpp=3."""
    agents = build_mock_agents_depth2_cpp3()
    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
    )
    assert graph is not None
    assert recursion_limit >= 50


def test_create_execution_graph_depth3_cpp2():
    """Test graph creation for depth=3, cpp=2."""
    agents = {
        "L1N1": make_mock_agent("L1N1", 1, 1, 3, 2, children=["L2N1", "L2N2"]),
        "L2N1": make_mock_agent(
            "L2N1",
            2,
            1,
            3,
            2,
            siblings=["L2N2"],
            parent="L1N1",
            children=["L3N1", "L3N2"],
        ),
        "L2N2": make_mock_agent(
            "L2N2",
            2,
            2,
            3,
            2,
            siblings=["L2N1"],
            parent="L1N1",
            children=["L3N3", "L3N4"],
        ),
        "L3N1": make_mock_agent(
            "L3N1",
            3,
            1,
            3,
            2,
            siblings=["L3N2"],
            parent="L2N1",
            perspective="analytical",
        ),
        "L3N2": make_mock_agent(
            "L3N2", 3, 2, 3, 2, siblings=["L3N1"], parent="L2N1", perspective="creative"
        ),
        "L3N3": make_mock_agent(
            "L3N3", 3, 3, 3, 2, siblings=["L3N4"], parent="L2N2", perspective="critical"
        ),
        "L3N4": make_mock_agent(
            "L3N4",
            3,
            4,
            3,
            2,
            siblings=["L3N3"],
            parent="L2N2",
            perspective="practical",
        ),
    }
    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=2,
        depth=3,
    )
    assert graph is not None


# =============================================================================
# Integration Tests with Mock Agents
# =============================================================================


def test_full_execution_depth2_cpp3_one_round():
    """Test full execution with depth=2, cpp=3, max_rounds=1."""
    agents = build_mock_agents_depth2_cpp3()

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=1,
        enable_downward_signals=False,
    )

    result = graph.invoke(
        {"original_task": "Explain recursion"},
        config={"recursion_limit": recursion_limit},
    )

    assert "final_response" in result
    assert result["final_response"] == "root synthesis"
    assert result["converged"] is True  # max_rounds=1 forces convergence

    # Check agent history was populated
    history = result["agent_history"]
    assert len(history["L2N1"]) == 1
    assert len(history["L2N2"]) == 1
    assert len(history["L2N3"]) == 1
    assert len(history["L1N1"]) == 1

    # Leaves should have lateral responses
    assert "lateral_response" in history["L2N1"][0]


def test_full_execution_with_convergence():
    """Test full execution with convergence after 2 rounds."""
    agents = build_mock_agents_depth2_cpp3()

    # Root returns same response each round → should converge
    agents["L1N1"].invoke.return_value = "stable synthesis"

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=3,
        convergence_threshold=0.85,
        enable_downward_signals=False,
    )

    result = graph.invoke(
        {"original_task": "Explain recursion"},
        config={"recursion_limit": recursion_limit},
    )

    assert result["converged"] is True
    assert result["final_response"] == "stable synthesis"
    # Should have converged after round 2 (identical responses)
    assert len(result["convergence_scores"]) >= 1
    assert result["convergence_scores"][-1] == 1.0


def test_full_execution_with_signals():
    """Test full execution with downward signals enabled."""
    agents = build_mock_agents_depth2_cpp3()

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=1,
        enable_downward_signals=True,
    )

    result = graph.invoke(
        {"original_task": "Explain recursion"},
        config={"recursion_limit": recursion_limit},
    )

    assert "final_response" in result
    # Root should have sent a signal
    root_hist = result["agent_history"]["L1N1"]
    assert "signal_sent" in root_hist[-1]


def test_full_execution_with_strange_loop():
    """Test full execution with strange loop."""
    agents = build_mock_agents_depth2_cpp3()

    strange_response = """Review...

Final Response:
**************************************************
Refined after strange loop
**************************************************
"""
    agents["L1N1"].invoke.side_effect = ["root synthesis", strange_response]

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=1,
        enable_downward_signals=False,
        strange_loop_count=1,
    )

    result = graph.invoke(
        {"original_task": "Explain recursion"},
        config={"recursion_limit": recursion_limit},
    )

    assert result["final_response"] == "Refined after strange loop"
    assert "strange_loops" in result
    assert len(result["strange_loops"]) == 1


# =============================================================================
# _apply_strange_loop Tests
# =============================================================================


def test_apply_strange_loop_no_loops():
    """Test _apply_strange_loop with 0 loops."""
    agent = MagicMock()
    result, loops = _apply_strange_loop(agent, "response", "task", 0, "")
    assert result == "response"
    assert loops == []
    agent.invoke.assert_not_called()


def test_apply_strange_loop_with_domain_instructions():
    """Test _apply_strange_loop triggers extra loop for domain instructions."""
    agent = MagicMock()
    agent.invoke.return_value = """Final Response:
**************************************************
domain refined
**************************************************"""

    result, loops = _apply_strange_loop(agent, "response", "task", 0, "Be concise.")
    assert result == "domain refined"
    assert len(loops) == 1


# =============================================================================
# run_execution_graph Tests
# =============================================================================


@patch("src.strange_mca.graph.setup_detailed_logging")
def test_run_execution_graph(mock_setup_logging):
    """Test run_execution_graph sets up logging and invokes graph."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "final_response": "Test response",
        "agent_history": {},
    }

    result = run_execution_graph(
        graph=mock_graph,
        task="Test task",
        recursion_limit=50,
        log_level="info",
        only_local_logs=True,
    )

    mock_setup_logging.assert_called_once_with(log_level="info", only_local_logs=True)
    mock_graph.invoke.assert_called_once()
    assert result["final_response"] == "Test response"
