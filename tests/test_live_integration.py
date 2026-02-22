"""Live integration tests that call a real LLM.

These tests are marked with @pytest.mark.live and are skipped by default.
Run them explicitly with: poetry run pytest -m live -v

Requires OPENAI_API_KEY in .env file.
"""

import json
import os
import shutil
import tempfile

import pytest

from src.strange_mca.run_strange_mca import run_strange_mca


@pytest.mark.live()
def test_live_end_to_end_basic():
    """Basic end-to-end test: system runs and produces valid structured output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_strange_mca(
            task="What are the main tradeoffs between microservices and monoliths?",
            child_per_parent=2,
            depth=2,
            model="gpt-4o-mini",
            max_rounds=2,
            enable_downward_signals=False,
            log_level="warn",
            output_dir=tmpdir,
        )

    # Structure assertions
    assert "final_response" in result
    assert isinstance(result["final_response"], str)
    assert len(result["final_response"]) > 0

    # All agents should have history
    history = result["agent_history"]
    assert "L1N1" in history
    assert "L2N1" in history
    assert "L2N2" in history

    # Each agent should have 1-2 rounds
    for name, rounds in history.items():
        assert 1 <= len(rounds) <= 2, f"{name} has {len(rounds)} rounds"

    assert result["converged"] is True

    # Content should be substantive (>50 words)
    word_count = len(result["final_response"].split())
    assert word_count > 50, f"Response only has {word_count} words"


@pytest.mark.live()
def test_live_lateral_communication_effect():
    """Lateral communication should have observable effect on at least one agent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_strange_mca(
            task="Explain the concept of emergence in complex systems.",
            child_per_parent=3,
            depth=2,
            model="gpt-4o-mini",
            max_rounds=1,
            enable_downward_signals=False,
            log_level="warn",
            output_dir=tmpdir,
        )

    history = result["agent_history"]
    leaf_names = ["L2N1", "L2N2", "L2N3"]

    # All leaves should have both response and lateral_response
    for name in leaf_names:
        rd = history[name][0]
        assert "response" in rd, f"{name} missing response"
        assert "lateral_response" in rd, f"{name} missing lateral_response"

    # At least one leaf should have revised (lateral had effect)
    revised_count = sum(1 for n in leaf_names if history[n][0].get("revised", False))
    assert revised_count >= 1, "No leaf revised after lateral communication"


@pytest.mark.live()
def test_live_convergence_trajectory():
    """Convergence scores should be valid floats between 0 and 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_strange_mca(
            task="What is recursion?",
            child_per_parent=2,
            depth=2,
            model="gpt-4o-mini",
            max_rounds=3,
            convergence_threshold=0.95,
            enable_downward_signals=False,
            log_level="warn",
            output_dir=tmpdir,
        )

    scores = result.get("convergence_scores", [])
    assert isinstance(scores, list)
    # With max_rounds=3 and threshold=0.95, we should get at least 1 score
    assert len(scores) >= 1, "No convergence scores recorded"

    for score in scores:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"


@pytest.mark.live()
def test_live_downward_signals():
    """Downward signals should appear in agent history when enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_strange_mca(
            task="Compare functional and object-oriented programming.",
            child_per_parent=2,
            depth=2,
            model="gpt-4o-mini",
            max_rounds=2,
            enable_downward_signals=True,
            log_level="warn",
            output_dir=tmpdir,
        )

    history = result["agent_history"]

    # Root should have signal_sent in round 1
    root_round_1 = history["L1N1"][0]
    assert "signal_sent" in root_round_1, "Root should have sent a signal in round 1"
    assert len(root_round_1["signal_sent"]) > 0

    # Leaves should have signal_received in round 1
    for name in ["L2N1", "L2N2"]:
        rd = history[name][0]
        assert "signal_received" in rd, f"{name} should have received a signal"
        assert len(rd["signal_received"]) > 0


@pytest.mark.live()
def test_live_report_output():
    """Report files should be written and contain valid JSON."""
    tmpdir = tempfile.mkdtemp()
    try:
        result = run_strange_mca(
            task="What are the main tradeoffs between microservices and monoliths?",
            child_per_parent=2,
            depth=2,
            model="gpt-4o-mini",
            max_rounds=2,
            enable_downward_signals=False,
            log_level="warn",
            output_dir=tmpdir,
        )

        # Check mca_report.json
        report_path = os.path.join(tmpdir, "mca_report.json")
        assert os.path.exists(report_path), "mca_report.json not created"
        with open(report_path) as f:
            report = json.load(f)
        assert "task" in report
        assert "config" in report
        assert "rounds" in report
        assert "convergence" in report
        assert "summary_metrics" in report
        assert "final_response" in report

        # Check final_state.json
        state_path = os.path.join(tmpdir, "final_state.json")
        assert os.path.exists(state_path), "final_state.json not created"
        with open(state_path) as f:
            state = json.load(f)
        assert "final_response" in state
        assert "agent_history" in state

    finally:
        shutil.rmtree(tmpdir)
