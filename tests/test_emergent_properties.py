"""Tests for emergent properties of the MCA system.

Deterministic mock-based tests that verify system mechanics produce correct
emergent behaviors: lateral revision, convergence, downward signals, hierarchy
processing, strange loops, and report structure.
"""

from src.strange_mca.graph import create_execution_graph
from src.strange_mca.run_strange_mca import build_mca_report
from tests.conftest import build_mock_agents_depth2_cpp3, make_mock_agent

# =============================================================================
# Lateral Communication Tests
# =============================================================================


def test_lateral_communication_causes_revision():
    """Lateral communication should detect and record when agents revise."""
    agents = build_mock_agents_depth2_cpp3()

    # Leaves return different text for initial vs lateral invocation
    for name in ["L2N1", "L2N2", "L2N3"]:
        agents[name].invoke.side_effect = [
            f"{name} initial response",  # leaf_respond
            f"{name} revised after seeing peers",  # leaf_lateral
        ]

    graph, recursion_limit = create_execution_graph(
        agents=agents, cpp=3, depth=2, max_rounds=1, enable_downward_signals=False
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    history = result["agent_history"]
    for name in ["L2N1", "L2N2", "L2N3"]:
        round_data = history[name][0]
        assert round_data["revised"] is True, f"{name} should be revised"
        assert "lateral_response" in round_data
        assert round_data["lateral_response"] != round_data["response"]


def test_lateral_no_revision_when_unchanged():
    """Revision detection should correctly identify no-change case."""
    agents = build_mock_agents_depth2_cpp3()

    # Leaves return identical text for both initial and lateral invocation
    for name in ["L2N1", "L2N2", "L2N3"]:
        agents[name].invoke.return_value = f"{name} same response"

    graph, recursion_limit = create_execution_graph(
        agents=agents, cpp=3, depth=2, max_rounds=1, enable_downward_signals=False
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    history = result["agent_history"]
    for name in ["L2N1", "L2N2", "L2N3"]:
        round_data = history[name][0]
        assert round_data["revised"] is False, f"{name} should not be revised"


# =============================================================================
# Convergence Tests
# =============================================================================


def test_convergence_over_multiple_rounds():
    """System should converge when root output stabilizes across rounds."""
    agents = build_mock_agents_depth2_cpp3()

    # Root returns different text round 1, then identical text rounds 2 and 3
    # Root is invoked once per round (observe_root)
    agents["L1N1"].invoke.side_effect = [
        "synthesis version one with unique content",  # round 1 observe
        "synthesis version two with new content",  # round 2 observe
        "synthesis version two with new content",  # round 3 observe (identical to round 2)
    ]

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=3,
        convergence_threshold=0.85,
        enable_downward_signals=False,
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    assert result["converged"] is True
    scores = result["convergence_scores"]
    assert len(scores) >= 1
    # Last score should be 1.0 (identical round 2→3 outputs)
    assert scores[-1] == 1.0

    # Root should have exactly 3 rounds of history
    root_history = result["agent_history"]["L1N1"]
    assert len(root_history) == 3


def test_no_convergence_forces_max_rounds():
    """System should exit after max_rounds even without convergence."""
    agents = build_mock_agents_depth2_cpp3()

    # Root returns completely different text each round
    agents["L1N1"].invoke.side_effect = [
        "completely different output round one",
        "totally new and unique output round two",
    ]

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=2,
        convergence_threshold=0.99,
        enable_downward_signals=False,
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    # Should be forced converged at max_rounds
    assert result["converged"] is True
    # All agents should have 2 rounds of history
    for name in ["L2N1", "L2N2", "L2N3"]:
        assert len(result["agent_history"][name]) == 2
    assert len(result["agent_history"]["L1N1"]) == 2

    # Should have convergence score(s) less than threshold
    scores = result["convergence_scores"]
    assert len(scores) >= 1
    assert any(s < 0.99 for s in scores)


# =============================================================================
# Downward Signal Tests
# =============================================================================


def test_downward_signals_propagate_to_children():
    """Signals should propagate from root to children when enabled."""
    agents = build_mock_agents_depth2_cpp3()

    # Root's invoke will be called for: observe (round 1), signal (round 1),
    # observe (round 2)
    agents["L1N1"].invoke.side_effect = [
        "root synthesis round 1",  # observe_root round 1
        "consider exploring edge cases",  # signal_down round 1
        "root synthesis round 2",  # observe_root round 2
        "signal round 2",  # signal_down round 2 (needed since max_rounds=2)
    ]

    # Each leaf is invoked for: respond (r1), lateral (r1), respond (r2), lateral (r2)
    for name in ["L2N1", "L2N2", "L2N3"]:
        agents[name].invoke.side_effect = [
            f"{name} response r1",
            f"{name} lateral r1",
            f"{name} response r2",
            f"{name} lateral r2",
        ]

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=2,
        enable_downward_signals=True,
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    history = result["agent_history"]

    # Root's round 1 history should have signal_sent
    assert "signal_sent" in history["L1N1"][0]
    assert history["L1N1"][0]["signal_sent"] == "consider exploring edge cases"

    # Each leaf's round 1 history should have signal_received
    for name in ["L2N1", "L2N2", "L2N3"]:
        assert "signal_received" in history[name][0]
        assert history[name][0]["signal_received"] == "consider exploring edge cases"


def test_signals_disabled_produces_no_signals():
    """No signals should appear when downward signals are disabled."""
    agents = build_mock_agents_depth2_cpp3()

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=2,
        enable_downward_signals=False,
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    history = result["agent_history"]
    for name, rounds in history.items():
        for rd in rounds:
            assert "signal_sent" not in rd, f"{name} should have no signal_sent"
            assert "signal_received" not in rd, f"{name} should have no signal_received"


# =============================================================================
# Root Synthesis Tests
# =============================================================================


def test_root_synthesis_uses_child_lateral_responses():
    """Root should observe post-lateral (revised) outputs, not pre-lateral."""
    agents = build_mock_agents_depth2_cpp3()

    # Each leaf returns distinct initial and lateral responses
    agents["L2N1"].invoke.side_effect = [
        "L2N1 initial",
        "L2N1 revised lateral",
    ]
    agents["L2N2"].invoke.side_effect = [
        "L2N2 initial",
        "L2N2 revised lateral",
    ]
    agents["L2N3"].invoke.side_effect = [
        "L2N3 initial",
        "L2N3 revised lateral",
    ]

    # Root captures the prompt it's called with
    root_prompts = []
    original_invoke = agents["L1N1"].invoke

    def capture_root_prompt(prompt):
        root_prompts.append(prompt)
        return "root synthesis"

    agents["L1N1"].invoke.side_effect = capture_root_prompt

    graph, recursion_limit = create_execution_graph(
        agents=agents, cpp=3, depth=2, max_rounds=1, enable_downward_signals=False
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    # Root should have been invoked with a prompt containing lateral responses
    assert len(root_prompts) >= 1
    observe_prompt = root_prompts[0]
    assert "L2N1 revised lateral" in observe_prompt
    assert "L2N2 revised lateral" in observe_prompt
    assert "L2N3 revised lateral" in observe_prompt


# =============================================================================
# Perspective Tests
# =============================================================================


def test_perspective_diversity_in_prompts():
    """Each leaf agent's prompt should contain its specific perspective."""
    agents = build_mock_agents_depth2_cpp3()

    # Capture prompts for each leaf
    leaf_prompts = {}

    for name in ["L2N1", "L2N2", "L2N3"]:

        def make_capturer(agent_name):
            def capture(prompt):
                if agent_name not in leaf_prompts:
                    leaf_prompts[agent_name] = []
                leaf_prompts[agent_name].append(prompt)
                return f"{agent_name} response"

            return capture

        agents[name].invoke.side_effect = make_capturer(name)

    graph, recursion_limit = create_execution_graph(
        agents=agents, cpp=3, depth=2, max_rounds=1, enable_downward_signals=False
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    # Check each leaf's initial response prompt contains its perspective
    assert "analytical" in leaf_prompts["L2N1"][0]
    assert "creative" in leaf_prompts["L2N2"][0]
    assert "critical" in leaf_prompts["L2N3"][0]


# =============================================================================
# Multi-Level Hierarchy Tests
# =============================================================================


def test_multi_level_hierarchy_depth3():
    """Internal coordinators should observe children and communicate laterally."""
    # depth=3, cpp=2: 7 agents (1 root, 2 coordinators, 4 leaves)
    agents = {
        "L1N1": make_mock_agent(
            "L1N1", 1, 1, 3, 2, children=["L2N1", "L2N2"], response="root synthesis"
        ),
        "L2N1": make_mock_agent(
            "L2N1",
            2,
            1,
            3,
            2,
            siblings=["L2N2"],
            parent="L1N1",
            children=["L3N1", "L3N2"],
            response="coordinator 1 observation",
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
            response="coordinator 2 observation",
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
            response="leaf 1",
        ),
        "L3N2": make_mock_agent(
            "L3N2",
            3,
            2,
            3,
            2,
            siblings=["L3N1"],
            parent="L2N1",
            perspective="creative",
            response="leaf 2",
        ),
        "L3N3": make_mock_agent(
            "L3N3",
            3,
            3,
            3,
            2,
            siblings=["L3N4"],
            parent="L2N2",
            perspective="critical",
            response="leaf 3",
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
            response="leaf 4",
        ),
    }

    graph, recursion_limit = create_execution_graph(
        agents=agents, cpp=2, depth=3, max_rounds=1, enable_downward_signals=False
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    history = result["agent_history"]

    # Coordinators should have history entries (they observed their children)
    assert len(history["L2N1"]) == 1
    assert len(history["L2N2"]) == 1
    assert "response" in history["L2N1"][0]
    assert "response" in history["L2N2"][0]

    # Coordinators should have lateral responses (they communicated with sibling)
    assert "lateral_response" in history["L2N1"][0]
    assert "lateral_response" in history["L2N2"][0]

    # Root observed coordinators, not leaves directly
    assert len(history["L1N1"]) == 1

    # All 4 leaves should have history
    for leaf in ["L3N1", "L3N2", "L3N3", "L3N4"]:
        assert len(history[leaf]) == 1
        assert "lateral_response" in history[leaf][0]


# =============================================================================
# Strange Loop Tests
# =============================================================================


def test_strange_loop_refines_final_response():
    """Strange loop should post-process the root's synthesis."""
    agents = build_mock_agents_depth2_cpp3()

    strange_response = """Review of the synthesis...

Final Response:
**************************************************
Refined output after strange loop reflection
**************************************************

Brief reasoning for revisions."""

    # Root: first call is observe_root, second call is strange loop
    agents["L1N1"].invoke.side_effect = [
        "raw root synthesis",
        strange_response,
    ]

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=1,
        enable_downward_signals=False,
        strange_loop_count=1,
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    assert result["final_response"] == "Refined output after strange loop reflection"
    assert "strange_loops" in result
    assert len(result["strange_loops"]) == 1
    assert "prompt" in result["strange_loops"][0]
    assert "response" in result["strange_loops"][0]


# =============================================================================
# Report Structure Tests
# =============================================================================


def test_report_structure_completeness():
    """MCA report should have all required keys and correct structure."""
    agents = build_mock_agents_depth2_cpp3()

    # Root returns same thing each round to converge after 2
    agents["L1N1"].invoke.return_value = "stable synthesis"

    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=3,
        depth=2,
        max_rounds=2,
        convergence_threshold=0.85,
        enable_downward_signals=False,
    )

    result = graph.invoke(
        {"original_task": "Test task"},
        config={"recursion_limit": recursion_limit},
    )

    config = {
        "cpp": 3,
        "depth": 2,
        "model": "gpt-4o-mini",
        "max_rounds": 2,
        "convergence_threshold": 0.85,
        "enable_downward_signals": False,
    }
    report = build_mca_report(result, "Test task", config)

    # Top-level keys
    assert "task" in report
    assert "config" in report
    assert "rounds" in report
    assert "convergence" in report
    assert "summary_metrics" in report
    assert "final_response" in report

    # Rounds structure
    assert len(report["rounds"]) >= 1
    for round_data in report["rounds"]:
        assert "round" in round_data
        assert "agents" in round_data
        assert "convergence_score" in round_data

    # Convergence section
    assert "converged" in report["convergence"]
    assert "rounds_used" in report["convergence"]
    assert "score_trajectory" in report["convergence"]

    # Summary metrics
    metrics = report["summary_metrics"]
    assert "total_llm_calls" in metrics
    assert "lateral_revision_rate" in metrics
    assert "per_agent_revision_counts" in metrics
    assert metrics["total_llm_calls"] > 0

    # Per-agent revision counts should exist for each agent
    for name in ["L1N1", "L2N1", "L2N2", "L2N3"]:
        assert name in metrics["per_agent_revision_counts"]
