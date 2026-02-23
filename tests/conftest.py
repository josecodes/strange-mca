"""Shared test fixtures and helpers for the MCA test suite."""

from unittest.mock import MagicMock

import pytest

from src.strange_mca.agents import Agent, AgentConfig


def make_mock_agent(
    name,
    level,
    node_number,
    depth,
    cpp,
    siblings=None,
    children=None,
    parent=None,
    perspective="",
    response="mock response",
):
    """Create a mock Agent with proper config."""
    agent = MagicMock(spec=Agent)
    config = AgentConfig(
        name=name,
        level=level,
        node_number=node_number,
        depth=depth,
        siblings=siblings or [],
        children=children or [],
        parent=parent,
        perspective=perspective,
    )
    agent.config = config
    agent.invoke.return_value = response
    return agent


def build_mock_agents_depth2_cpp3():
    """Build mock agents for depth=2, cpp=3 (4 agents: 1 root + 3 leaves)."""
    return {
        "L1N1": make_mock_agent(
            "L1N1",
            1,
            1,
            2,
            3,
            children=["L2N1", "L2N2", "L2N3"],
            response="root synthesis",
        ),
        "L2N1": make_mock_agent(
            "L2N1",
            2,
            1,
            2,
            3,
            siblings=["L2N2", "L2N3"],
            parent="L1N1",
            perspective="analytical",
            response="leaf 1 response",
        ),
        "L2N2": make_mock_agent(
            "L2N2",
            2,
            2,
            2,
            3,
            siblings=["L2N1", "L2N3"],
            parent="L1N1",
            perspective="creative",
            response="leaf 2 response",
        ),
        "L2N3": make_mock_agent(
            "L2N3",
            2,
            3,
            2,
            3,
            siblings=["L2N1", "L2N2"],
            parent="L1N1",
            perspective="critical",
            response="leaf 3 response",
        ),
    }


def build_mock_agents_depth3_cpp2():
    """Build mock agents for depth=3, cpp=2 (7 agents).

    Tree structure:
        L1N1 (root)
        +-- L2N1 (coordinator)
        |   +-- L3N1 (leaf, analytical)
        |   +-- L3N2 (leaf, creative)
        +-- L2N2 (coordinator)
            +-- L3N3 (leaf, critical)
            +-- L3N4 (leaf, practical)
    """
    return {
        "L1N1": make_mock_agent(
            "L1N1",
            1,
            1,
            3,
            2,
            children=["L2N1", "L2N2"],
            response="root synthesis",
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
        ),
    }


@pytest.fixture()
def mock_agents_depth2_cpp3():
    """Fixture providing mock agents for depth=2, cpp=3."""
    return build_mock_agents_depth2_cpp3()


@pytest.fixture()
def mock_agents_depth3_cpp2():
    """Fixture providing mock agents for depth=3, cpp=2."""
    return build_mock_agents_depth3_cpp2()
