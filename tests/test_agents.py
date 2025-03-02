"""Tests for the agent module."""

from strange_mca.agents import create_agent_configs


def test_create_agent_configs():
    """Test the create_agent_configs function."""
    # Test with default values (child_per_parent=3, depth=2)
    configs = create_agent_configs(child_per_parent=3, depth=2)
    
    # Check that we have the correct number of agents
    assert len(configs) == 4  # 1 root + 3 children
    
    # Check that the root node has the correct children
    assert len(configs["L1N1"].children) == 3
    assert "L2N1" in configs["L1N1"].children
    assert "L2N2" in configs["L1N1"].children
    assert "L2N3" in configs["L1N1"].children
    
    # Test with different values
    configs = create_agent_configs(child_per_parent=2, depth=3)
    
    # Check that we have the correct number of agents
    assert len(configs) == 7  # 1 root + 2 children + 4 grandchildren
    
    # Check that the root node has the correct children
    assert len(configs["L1N1"].children) == 2
    assert "L2N1" in configs["L1N1"].children
    assert "L2N2" in configs["L1N1"].children
    
    # Check that the level 2 nodes have the correct children
    assert len(configs["L2N1"].children) == 2
    assert "L3N1" in configs["L2N1"].children
    assert "L3N2" in configs["L2N1"].children
    
    assert len(configs["L2N2"].children) == 2
    assert "L3N3" in configs["L2N2"].children
    assert "L3N4" in configs["L2N2"].children 