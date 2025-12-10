"""Tests for the agents module."""

import pytest

from src.strange_mca.agents import Agent, AgentConfig


def test_agent_config():
    """Test the AgentConfig class."""
    # Test basic initialization
    config = AgentConfig(name="test_agent", level=1, node_number=1)
    assert config.name == "test_agent"
    assert config.level == 1
    assert config.node_number == 1
    assert config.system_prompt == ""

    # Test with system prompt
    config = AgentConfig(
        name="test_agent",
        level=1,
        node_number=1,
        system_prompt="You are a helpful assistant.",
    )
    assert config.system_prompt == "You are a helpful assistant."

    # Test full_name property
    assert config.full_name == "L1N1"


def test_agent_config_full_name():
    """Test the full_name property for various configurations."""
    # Root node
    config = AgentConfig(name="L1N1", level=1, node_number=1)
    assert config.full_name == "L1N1"

    # Level 2 node
    config = AgentConfig(name="L2N3", level=2, node_number=3)
    assert config.full_name == "L2N3"

    # Level 3 node
    config = AgentConfig(name="L3N7", level=3, node_number=7)
    assert config.full_name == "L3N7"


@pytest.mark.skip(reason="Requires OpenAI API key and makes actual API calls")
def test_agent_run():
    """Test the Agent.run method."""
    # This test is skipped by default as it requires an OpenAI API key
    # and makes actual API calls
    config = AgentConfig(name="test_agent", level=1, node_number=1)
    agent = Agent(config, model_name="gpt-3.5-turbo")
    response = agent.run(task="Say hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_agent_repr():
    """Test the Agent.__repr__ method."""
    config = AgentConfig(name="L2N3", level=2, node_number=3)
    agent = Agent(config, model_name="gpt-3.5-turbo")
    assert repr(agent) == "Agent(L2N3)"
