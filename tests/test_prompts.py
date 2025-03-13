"""Tests for the prompts module."""

from src.strange_mca.prompts import (
    create_strange_loop_prompt,
    create_synthesis_prompt,
    create_task_decomposition_prompt,
    parse_strange_loop_response,
)


def test_create_task_decomposition_prompt():
    """Test the create_task_decomposition_prompt function."""
    task = "Analyze the impact of AI on healthcare"
    context = "You are coordinating a task across 3 agents: Agent1, Agent2, Agent3."
    child_nodes = ["Agent1", "Agent2", "Agent3"]

    prompt = create_task_decomposition_prompt(task, context, child_nodes)

    # Check that the prompt contains the task
    assert task in prompt

    # Check that the prompt contains all child nodes
    for node in child_nodes:
        assert node in prompt

    # Check that the prompt contains the context
    assert context in prompt


def test_create_synthesis_prompt():
    """Test the create_synthesis_prompt function."""
    child_responses = {
        "Agent1": "AI can help with diagnosis.",
        "Agent2": "AI can improve patient monitoring.",
        "Agent3": "AI raises privacy concerns.",
    }

    prompt = create_synthesis_prompt(child_responses)

    # Check that the prompt contains all agent responses
    for agent, response in child_responses.items():
        assert agent in prompt
        assert response in prompt


def test_create_strange_loop_prompt():
    """Test the create_strange_loop_prompt function."""
    original_task = "Explain quantum computing"
    tentative_response = "Quantum computing uses qubits instead of bits."

    # Test without domain-specific instructions
    prompt = create_strange_loop_prompt(original_task, tentative_response)

    # Check that the prompt contains the original task and tentative response
    assert original_task in prompt
    assert tentative_response in prompt

    # Test with domain-specific instructions
    domain_specific_instructions = "Focus on practical applications."
    prompt = create_strange_loop_prompt(
        original_task, tentative_response, domain_specific_instructions
    )

    # Check that the prompt contains the domain-specific instructions
    assert domain_specific_instructions in prompt


def test_parse_strange_loop_response():
    """Test the parse_strange_loop_response function."""
    # Test with a response that exactly matches the expected format
    response = """
Final Response:
**************************************************
Quantum computing is a revolutionary technology that uses quantum mechanics to process information.
**************************************************
"""

    parsed = parse_strange_loop_response(response)
    print(f"Response:\n{response}")
    print(f"Parsed: {parsed}")
    assert (
        parsed
        == "Quantum computing is a revolutionary technology that uses quantum mechanics to process information."
    )

    # Test with a response that has additional text before and after
    response2 = """
I've reviewed my response and here's my final answer:

Final Response:
**************************************************
This is a different response format.
**************************************************

I made these improvements because...
"""

    parsed2 = parse_strange_loop_response(response2)
    print(f"Response2:\n{response2}")
    print(f"Parsed2: {parsed2}")
    assert parsed2 == "This is a different response format."

    # Test with an empty response
    assert parse_strange_loop_response("") == ""

    # Test with a response that doesn't contain the final response section
    response_without_section = "Quantum computing is cool."
    assert (
        parse_strange_loop_response(response_without_section)
        == response_without_section.strip()
    )
