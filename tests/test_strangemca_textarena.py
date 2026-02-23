"""Tests for the TextArena integration."""

import importlib.util
import os
from unittest.mock import patch


def _import_strangemca_textarena():
    """Import strangemca_textarena from examples/arena without modifying sys.path."""
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "arena",
        "strangemca_textarena.py",
    )
    spec = importlib.util.spec_from_file_location("strangemca_textarena", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strangemca_textarena = _import_strangemca_textarena()
StrangeMCAAgent = strangemca_textarena.StrangeMCAAgent


@patch.object(strangemca_textarena, "run_strange_mca")
def test_strangemca_agent_init(mock_run_strange_mca):
    """Test the initialization of the StrangeMCAAgent class."""
    agent = StrangeMCAAgent()
    assert agent.child_per_parent == 3
    assert agent.depth == 2
    assert agent.model == "gpt-4o-mini"
    assert agent.max_rounds == 3
    assert agent.convergence_threshold == 0.85
    assert agent.enable_downward_signals is True
    assert agent.perspectives is None
    assert agent.viz is False
    assert agent.print_details is False
    assert agent.task_template is None
    assert agent.domain_specific_instructions == ""
    assert agent.strange_loop_count == 0

    agent = StrangeMCAAgent(
        child_per_parent=2,
        depth=3,
        model="gpt-4",
        max_rounds=5,
        convergence_threshold=0.9,
        enable_downward_signals=False,
        perspectives=["alpha", "beta"],
        viz=True,
        print_details=True,
        task_template="Task: {observation}",
        domain_specific_instructions="Be concise.",
        strange_loop_count=2,
    )
    assert agent.child_per_parent == 2
    assert agent.depth == 3
    assert agent.model == "gpt-4"
    assert agent.max_rounds == 5
    assert agent.convergence_threshold == 0.9
    assert agent.enable_downward_signals is False
    assert agent.perspectives == ["alpha", "beta"]
    assert agent.viz is True
    assert agent.print_details is True
    assert agent.task_template == "Task: {observation}"
    assert agent.domain_specific_instructions == "Be concise."
    assert agent.strange_loop_count == 2


@patch.object(strangemca_textarena, "run_strange_mca")
def test_strangemca_agent_call(mock_run_strange_mca):
    """Test the __call__ method of the StrangeMCAAgent class."""
    mock_run_strange_mca.return_value = {"final_response": "Test response"}

    agent = StrangeMCAAgent()
    response = agent("What is the capital of France?")

    mock_run_strange_mca.assert_called_once_with(
        task="What is the capital of France?",
        child_per_parent=3,
        depth=2,
        model="gpt-4o-mini",
        max_rounds=3,
        convergence_threshold=0.85,
        enable_downward_signals=True,
        perspectives=None,
        viz=False,
        print_details=False,
        domain_specific_instructions="",
        strange_loop_count=0,
    )

    assert response == "Test response"


@patch.object(strangemca_textarena, "run_strange_mca")
def test_strangemca_agent_with_template(mock_run_strange_mca):
    """Test the StrangeMCAAgent with a custom task template."""
    mock_run_strange_mca.return_value = {"final_response": "Paris"}

    agent = StrangeMCAAgent(
        task_template="Answer the following question: {observation}"
    )
    response = agent("What is the capital of France?")

    mock_run_strange_mca.assert_called_once()
    call_kwargs = mock_run_strange_mca.call_args[1]
    assert (
        call_kwargs["task"]
        == "Answer the following question: What is the capital of France?"
    )
    assert response == "Paris"
