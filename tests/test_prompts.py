"""Tests for the prompts module."""

from src.strange_mca.prompts import (
    create_competency_prompt,
    create_initial_response_prompt,
    create_lateral_prompt,
    create_observation_prompt,
    create_signal_prompt,
    create_strange_loop_prompt,
    parse_strange_loop_response,
)

# =============================================================================
# Competency Prompt Tests
# =============================================================================


def test_create_competency_prompt_specialist():
    """Test specialist competency prompt."""
    prompt = create_competency_prompt("specialist", "analytical")
    assert "specialist analyst" in prompt
    assert "analytical" in prompt
    assert "NOT being assigned a subtask" in prompt


def test_create_competency_prompt_coordinator():
    """Test coordinator competency prompt."""
    prompt = create_competency_prompt("coordinator")
    assert "coordinator" in prompt
    assert "emergent patterns" in prompt
    assert "do NOT assign tasks" in prompt


def test_create_competency_prompt_integrator():
    """Test integrator competency prompt."""
    prompt = create_competency_prompt("integrator")
    assert "integrator" in prompt
    assert "holistic" in prompt


# =============================================================================
# Initial Response Prompt Tests
# =============================================================================


def test_create_initial_response_prompt_round_1():
    """Test initial response prompt for round 1."""
    prompt = create_initial_response_prompt(
        task="Explain recursion",
        perspective="analytical",
        round_num=1,
    )
    assert "Explain recursion" in prompt
    assert "analytical" in prompt
    assert "independent analysis" in prompt
    assert "PREVIOUS RESPONSE" not in prompt


def test_create_initial_response_prompt_round_2():
    """Test initial response prompt for round 2 with previous response."""
    prompt = create_initial_response_prompt(
        task="Explain recursion",
        perspective="analytical",
        round_num=2,
        previous_response="Recursion is a self-referential process.",
    )
    assert "Explain recursion" in prompt
    assert "PREVIOUS RESPONSE" in prompt
    assert "Recursion is a self-referential process." in prompt
    assert "updated analysis" in prompt


def test_create_initial_response_prompt_with_parent_signal():
    """Test initial response prompt with parent signal."""
    prompt = create_initial_response_prompt(
        task="Explain recursion",
        perspective="analytical",
        round_num=2,
        previous_response="Previous analysis.",
        parent_signal="Consider base cases more carefully.",
    )
    assert "SIGNAL FROM YOUR COORDINATOR" in prompt
    assert "Consider base cases more carefully." in prompt


# =============================================================================
# Lateral Prompt Tests
# =============================================================================


def test_create_lateral_prompt():
    """Test lateral communication prompt."""
    prompt = create_lateral_prompt(
        task="Explain recursion",
        own_response="My analytical take.",
        peer_responses={
            "L2N2": "Creative perspective on recursion.",
            "L2N3": "Critical view of recursion.",
        },
        round_num=1,
    )
    assert "Explain recursion" in prompt
    assert "My analytical take." in prompt
    assert "L2N2" in prompt
    assert "Creative perspective" in prompt
    assert "MAINTAIN your unique perspective" in prompt


# =============================================================================
# Observation Prompt Tests
# =============================================================================


def test_create_observation_prompt():
    """Test observation prompt for parent agents."""
    prompt = create_observation_prompt(
        task="Explain recursion",
        child_responses={
            "L2N1": "Analytical response.",
            "L2N2": "Creative response.",
        },
    )
    assert "Explain recursion" in prompt
    assert "L2N1" in prompt
    assert "Analytical response." in prompt
    assert "EMERGES from the combination" in prompt
    assert "PREVIOUS SYNTHESIS" not in prompt


def test_create_observation_prompt_with_previous():
    """Test observation prompt with previous synthesis."""
    prompt = create_observation_prompt(
        task="Explain recursion",
        child_responses={"L2N1": "Response."},
        own_previous="Previous synthesis.",
        round_num=2,
    )
    assert "PREVIOUS SYNTHESIS (round 1)" in prompt
    assert "Previous synthesis." in prompt


# =============================================================================
# Signal Prompt Tests
# =============================================================================


def test_create_signal_prompt():
    """Test downward signal prompt."""
    prompt = create_signal_prompt(
        task="Explain recursion",
        child_responses={
            "L2N1": "Response A.",
            "L2N2": "Response B.",
        },
        own_synthesis="My synthesis.",
    )
    assert "Explain recursion" in prompt
    assert "SUGGESTIVE, not DIRECTIVE" in prompt
    assert "Response A." in prompt
    assert "My synthesis." in prompt


# =============================================================================
# Strange Loop Tests
# =============================================================================


def test_create_strange_loop_prompt():
    """Test the create_strange_loop_prompt function."""
    original_task = "Explain quantum computing"
    tentative_response = "Quantum computing uses qubits instead of bits."

    prompt = create_strange_loop_prompt(original_task, tentative_response)
    assert original_task in prompt
    assert tentative_response in prompt

    domain_specific_instructions = "Focus on practical applications."
    prompt = create_strange_loop_prompt(
        original_task, tentative_response, domain_specific_instructions
    )
    assert domain_specific_instructions in prompt


def test_parse_strange_loop_response():
    """Test the parse_strange_loop_response function."""
    response = """
Final Response:
**************************************************
Quantum computing is a revolutionary technology that uses quantum mechanics to process information.
**************************************************
"""
    parsed = parse_strange_loop_response(response)
    assert (
        parsed
        == "Quantum computing is a revolutionary technology that uses quantum mechanics to process information."
    )

    response2 = """
I've reviewed my response and here's my final answer:

Final Response:
**************************************************
This is a different response format.
**************************************************

I made these improvements because...
"""
    parsed2 = parse_strange_loop_response(response2)
    assert parsed2 == "This is a different response format."

    assert parse_strange_loop_response("") == ""

    response_without_section = "Quantum computing is cool."
    assert (
        parse_strange_loop_response(response_without_section)
        == response_without_section.strip()
    )
