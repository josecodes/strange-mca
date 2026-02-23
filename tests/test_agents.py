"""Tests for the agents module."""

from unittest.mock import patch

from src.strange_mca.agents import PERSPECTIVES, Agent, AgentConfig, build_agent_tree

# =============================================================================
# AgentConfig Tests
# =============================================================================


def test_agent_config_basic():
    """Test basic AgentConfig initialization."""
    config = AgentConfig(name="L1N1", level=1, node_number=1, depth=2)
    assert config.name == "L1N1"
    assert config.level == 1
    assert config.node_number == 1
    assert config.depth == 2
    assert config.system_prompt == ""
    assert config.siblings == []
    assert config.children == []
    assert config.parent is None
    assert config.perspective == ""


def test_agent_config_full_name():
    """Test the full_name property."""
    assert AgentConfig(name="L1N1", level=1, node_number=1, depth=2).full_name == "L1N1"
    assert AgentConfig(name="L2N3", level=2, node_number=3, depth=3).full_name == "L2N3"


def test_agent_config_is_leaf():
    """Test the is_leaf property."""
    assert AgentConfig(name="L2N1", level=2, node_number=1, depth=2).is_leaf is True
    assert AgentConfig(name="L1N1", level=1, node_number=1, depth=2).is_leaf is False
    assert AgentConfig(name="L1N1", level=1, node_number=1, depth=1).is_leaf is True


def test_agent_config_is_root():
    """Test the is_root property."""
    assert AgentConfig(name="L1N1", level=1, node_number=1, depth=2).is_root is True
    assert AgentConfig(name="L2N1", level=2, node_number=1, depth=2).is_root is False


def test_agent_config_role():
    """Test the role property."""
    # Root is integrator
    assert (
        AgentConfig(name="L1N1", level=1, node_number=1, depth=3).role == "integrator"
    )
    # Internal is coordinator
    assert (
        AgentConfig(name="L2N1", level=2, node_number=1, depth=3).role == "coordinator"
    )
    # Leaf is specialist
    assert (
        AgentConfig(name="L3N1", level=3, node_number=1, depth=3).role == "specialist"
    )
    # Depth=1 root-leaf is integrator
    assert (
        AgentConfig(name="L1N1", level=1, node_number=1, depth=1).role == "integrator"
    )


def test_agent_config_topology():
    """Test AgentConfig with topology fields."""
    config = AgentConfig(
        name="L2N1",
        level=2,
        node_number=1,
        depth=3,
        siblings=["L2N2"],
        children=["L3N1", "L3N2"],
        parent="L1N1",
        perspective="analytical",
    )
    assert config.siblings == ["L2N2"]
    assert config.children == ["L3N1", "L3N2"]
    assert config.parent == "L1N1"
    assert config.perspective == "analytical"


# =============================================================================
# Agent Tests
# =============================================================================


@patch("src.strange_mca.agents.ChatOpenAI")
def test_agent_repr(mock_chat):
    """Test the Agent.__repr__ method."""
    config = AgentConfig(name="L2N3", level=2, node_number=3, depth=3)
    agent = Agent(config, model_name="gpt-4o-mini")
    assert repr(agent) == "Agent(L2N3)"


@patch("src.strange_mca.agents.ChatOpenAI")
def test_agent_system_prompt(mock_chat):
    """Test that Agent stores the system prompt from config."""
    config = AgentConfig(
        name="L1N1",
        level=1,
        node_number=1,
        depth=2,
        system_prompt="You are the integrator.",
    )
    agent = Agent(config)
    assert agent.system_prompt == "You are the integrator."


# =============================================================================
# build_agent_tree Tests
# =============================================================================


@patch("src.strange_mca.agents.ChatOpenAI")
def test_build_agent_tree_depth_2_cpp_3(mock_chat):
    """Test building agent tree with depth=2, cpp=3."""
    agents = build_agent_tree(cpp=3, depth=2, model_name="gpt-4o-mini")

    # Should have 4 agents: 1 root + 3 leaves
    assert len(agents) == 4
    assert "L1N1" in agents
    assert "L2N1" in agents
    assert "L2N2" in agents
    assert "L2N3" in agents

    # Root config
    root = agents["L1N1"]
    assert root.config.role == "integrator"
    assert root.config.is_root is True
    assert root.config.children == ["L2N1", "L2N2", "L2N3"]
    assert root.config.parent is None
    assert root.config.siblings == []

    # Leaf configs
    leaf1 = agents["L2N1"]
    assert leaf1.config.role == "specialist"
    assert leaf1.config.is_leaf is True
    assert leaf1.config.parent == "L1N1"
    assert leaf1.config.siblings == ["L2N2", "L2N3"]
    assert leaf1.config.children == []
    assert leaf1.config.perspective == "analytical"

    leaf2 = agents["L2N2"]
    assert leaf2.config.perspective == "creative"

    leaf3 = agents["L2N3"]
    assert leaf3.config.perspective == "critical"


@patch("src.strange_mca.agents.ChatOpenAI")
def test_build_agent_tree_depth_3_cpp_2(mock_chat):
    """Test building agent tree with depth=3, cpp=2."""
    agents = build_agent_tree(cpp=2, depth=3, model_name="gpt-4o-mini")

    # Should have 7 agents: 1 root + 2 internal + 4 leaves
    assert len(agents) == 7

    # Root
    root = agents["L1N1"]
    assert root.config.role == "integrator"
    assert root.config.children == ["L2N1", "L2N2"]

    # Internal coordinators
    coord1 = agents["L2N1"]
    assert coord1.config.role == "coordinator"
    assert coord1.config.children == ["L3N1", "L3N2"]
    assert coord1.config.siblings == ["L2N2"]
    assert coord1.config.parent == "L1N1"

    # Leaves
    leaf1 = agents["L3N1"]
    assert leaf1.config.role == "specialist"
    assert leaf1.config.siblings == ["L3N2"]
    assert leaf1.config.parent == "L2N1"
    assert leaf1.config.perspective == "analytical"


@patch("src.strange_mca.agents.ChatOpenAI")
def test_build_agent_tree_custom_perspectives(mock_chat):
    """Test building agent tree with custom perspectives."""
    custom = ["alpha", "beta", "gamma"]
    agents = build_agent_tree(
        cpp=3, depth=2, model_name="gpt-4o-mini", perspectives=custom
    )

    assert agents["L2N1"].config.perspective == "alpha"
    assert agents["L2N2"].config.perspective == "beta"
    assert agents["L2N3"].config.perspective == "gamma"


@patch("src.strange_mca.agents.ChatOpenAI")
def test_build_agent_tree_perspective_wrapping(mock_chat):
    """Test that perspectives wrap around when there are more leaves than perspectives."""
    agents = build_agent_tree(
        cpp=3,
        depth=2,
        model_name="gpt-4o-mini",
        perspectives=["A", "B"],
    )

    assert agents["L2N1"].config.perspective == "A"
    assert agents["L2N2"].config.perspective == "B"
    assert agents["L2N3"].config.perspective == "A"  # wraps


@patch("src.strange_mca.agents.ChatOpenAI")
def test_build_agent_tree_competency_prompts(mock_chat):
    """Test that agents get appropriate competency system prompts."""
    agents = build_agent_tree(cpp=2, depth=2, model_name="gpt-4o-mini")

    # Root should have integrator prompt
    assert "integrator" in agents["L1N1"].system_prompt

    # Leaves should have specialist prompt with perspective
    assert "specialist" in agents["L2N1"].system_prompt
    assert "analytical" in agents["L2N1"].system_prompt


def test_perspectives_list():
    """Test that PERSPECTIVES has 8 entries."""
    assert len(PERSPECTIVES) == 8
    assert "analytical" in PERSPECTIVES
    assert "creative" in PERSPECTIVES
    assert "critical" in PERSPECTIVES
    assert "practical" in PERSPECTIVES
