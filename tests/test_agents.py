"""Tests for the agent module."""

from strange_mca.agents import create_agent_configs


def test_create_agent_configs():
    """Test creating agent configurations."""
    # Test with default values (team_size=3, depth=2)
    configs = create_agent_configs(team_size=3, depth=2)
    
    # Check that we have the expected number of agents
    assert len(configs) == 4  # 1 supervisor + 3 team members
    
    # Check that the supervisor has the correct children
    assert configs["L1N1"].children == ["L2N1", "L2N2", "L2N3"]
    
    # Check that the children have the correct parent
    assert configs["L2N1"].parent == "L1N1"
    assert configs["L2N2"].parent == "L1N1"
    assert configs["L2N3"].parent == "L1N1"
    
    # Test with different values
    configs = create_agent_configs(team_size=2, depth=3)
    
    # Check that we have the expected number of agents
    assert len(configs) == 7  # 1 + 2 + 4 = 7
    
    # Check the level 2 nodes
    assert configs["L1N1"].children == ["L2N1", "L2N2"]
    assert configs["L2N1"].parent == "L1N1"
    assert configs["L2N2"].parent == "L1N1"
    
    # Check the level 3 nodes
    assert configs["L2N1"].children == ["L3N1", "L3N2"]
    assert configs["L2N2"].children == ["L3N3", "L3N4"]
    
    assert configs["L3N1"].parent == "L2N1"
    assert configs["L3N2"].parent == "L2N1"
    assert configs["L3N3"].parent == "L2N2"
    assert configs["L3N4"].parent == "L2N2" 