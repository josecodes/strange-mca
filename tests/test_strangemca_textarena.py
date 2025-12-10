"""Tests for the TextArena integration."""

import os
import sys
from unittest.mock import patch

# Add the examples/arena directory to the Python path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples/arena")
)

from strangemca_textarena import StrangeMCAAgent


@patch("strangemca_textarena.run_strange_mca")
def test_strangemca_agent_init(mock_run_strange_mca):
    """Test the initialization of the StrangeMCAAgent class."""
    # Create an agent with default parameters
    agent = StrangeMCAAgent()

    # Check that the default parameters are set correctly
    assert agent.child_per_parent == 2
    assert agent.depth == 2
    assert agent.model == "gpt-3.5-turbo"
    assert agent.viz is False
    assert agent.print_details is False
    assert agent.task_template is None
    assert agent.domain_specific_instructions == ""
    assert agent.strange_loop_count == 0

    # Create an agent with custom parameters
    agent = StrangeMCAAgent(
        child_per_parent=3,
        depth=3,
        model="gpt-4",
        viz=True,
        print_details=True,
        task_template="Task: {observation}",
        domain_specific_instructions="Be concise.",
        strange_loop_count=2,
    )

    # Check that the custom parameters are set correctly
    assert agent.child_per_parent == 3
    assert agent.depth == 3
    assert agent.model == "gpt-4"
    assert agent.viz is True
    assert agent.print_details is True
    assert agent.task_template == "Task: {observation}"
    assert agent.domain_specific_instructions == "Be concise."
    assert agent.strange_loop_count == 2


@patch("strangemca_textarena.run_strange_mca")
def test_strangemca_agent_call(mock_run_strange_mca):
    """Test the __call__ method of the StrangeMCAAgent class."""
    # Mock the run_strange_mca function
    mock_run_strange_mca.return_value = {"final_response": "Test response"}

    # Create an agent
    agent = StrangeMCAAgent()

    # Call the agent with an observation
    response = agent("What is the capital of France?")

    # Check that run_strange_mca was called with the correct parameters
    mock_run_strange_mca.assert_called_once_with(
        task="What is the capital of France?",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        viz=False,
        print_details=False,
        domain_specific_instructions="",
        strange_loop_count=0,
    )

    # Check that the response is correct
    assert response == "Test response"


@patch("strangemca_textarena.run_strange_mca")
def test_strangemca_agent_with_template(mock_run_strange_mca):
    """Test the StrangeMCAAgent with a custom task template."""
    # Mock the run_strange_mca function
    mock_run_strange_mca.return_value = {"final_response": "Paris"}

    # Create an agent with a task template
    agent = StrangeMCAAgent(
        task_template="Answer the following question: {observation}"
    )

    # Call the agent with an observation
    response = agent("What is the capital of France?")

    # Check that run_strange_mca was called with the correct parameters
    mock_run_strange_mca.assert_called_once_with(
        task="Answer the following question: What is the capital of France?",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        viz=False,
        print_details=False,
        domain_specific_instructions="",
        strange_loop_count=0,
    )

    # Check that the response is correct
    assert response == "Paris"
