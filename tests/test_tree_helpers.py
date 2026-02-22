"""Tests for the tree_helpers module."""

from src.strange_mca.tree_helpers import (
    count_nodes_at_level,
    generate_all_nodes,
    get_children,
    get_parent,
    get_siblings,
    is_leaf,
    is_root,
    make_node_name,
    parse_node_name,
    total_nodes,
)


def test_parse_node_name():
    """Test parsing node names."""
    assert parse_node_name("L1N1") == (1, 1)
    assert parse_node_name("L2N3") == (2, 3)
    assert parse_node_name("L3N7") == (3, 7)
    assert parse_node_name("L10N100") == (10, 100)


def test_make_node_name():
    """Test making node names."""
    assert make_node_name(1, 1) == "L1N1"
    assert make_node_name(2, 3) == "L2N3"
    assert make_node_name(3, 7) == "L3N7"


def test_get_children():
    """Test getting children of a node."""
    # Root with 2 children per parent, depth 3
    assert get_children("L1N1", cpp=2, depth=3) == ["L2N1", "L2N2"]

    # L2N1 with 2 children per parent, depth 3
    assert get_children("L2N1", cpp=2, depth=3) == ["L3N1", "L3N2"]

    # L2N2 with 2 children per parent, depth 3
    assert get_children("L2N2", cpp=2, depth=3) == ["L3N3", "L3N4"]

    # Leaf node (no children)
    assert get_children("L3N1", cpp=2, depth=3) == []

    # Root with 3 children per parent
    assert get_children("L1N1", cpp=3, depth=2) == ["L2N1", "L2N2", "L2N3"]


def test_get_parent():
    """Test getting parent of a node."""
    # Root has no parent
    assert get_parent("L1N1", cpp=2) is None

    # L2N1's parent is L1N1
    assert get_parent("L2N1", cpp=2) == "L1N1"
    assert get_parent("L2N2", cpp=2) == "L1N1"

    # L3N1 and L3N2's parent is L2N1
    assert get_parent("L3N1", cpp=2) == "L2N1"
    assert get_parent("L3N2", cpp=2) == "L2N1"

    # L3N3 and L3N4's parent is L2N2
    assert get_parent("L3N3", cpp=2) == "L2N2"
    assert get_parent("L3N4", cpp=2) == "L2N2"

    # cpp=3
    assert get_parent("L2N1", cpp=3) == "L1N1"
    assert get_parent("L2N3", cpp=3) == "L1N1"


def test_get_siblings():
    """Test getting siblings of a node."""
    # Root has no siblings
    assert get_siblings("L1N1", cpp=2, depth=2) == []

    # L2N1 and L2N2 are siblings (cpp=2)
    assert get_siblings("L2N1", cpp=2, depth=2) == ["L2N2"]
    assert get_siblings("L2N2", cpp=2, depth=2) == ["L2N1"]

    # cpp=3 siblings
    assert get_siblings("L2N1", cpp=3, depth=2) == ["L2N2", "L2N3"]
    assert get_siblings("L2N2", cpp=3, depth=2) == ["L2N1", "L2N3"]
    assert get_siblings("L2N3", cpp=3, depth=2) == ["L2N1", "L2N2"]

    # Leaf siblings in deeper tree
    assert get_siblings("L3N1", cpp=2, depth=3) == ["L3N2"]
    assert get_siblings("L3N3", cpp=2, depth=3) == ["L3N4"]


def test_is_leaf():
    """Test leaf node detection."""
    assert is_leaf(level=3, depth=3) is True
    assert is_leaf(level=2, depth=3) is False
    assert is_leaf(level=1, depth=3) is False
    assert is_leaf(level=1, depth=1) is True


def test_is_root():
    """Test root node detection."""
    assert is_root(level=1) is True
    assert is_root(level=2) is False
    assert is_root(level=3) is False


def test_count_nodes_at_level():
    """Test counting nodes at each level."""
    assert count_nodes_at_level(level=1, cpp=2) == 1
    assert count_nodes_at_level(level=2, cpp=2) == 2
    assert count_nodes_at_level(level=3, cpp=2) == 4

    assert count_nodes_at_level(level=1, cpp=3) == 1
    assert count_nodes_at_level(level=2, cpp=3) == 3
    assert count_nodes_at_level(level=3, cpp=3) == 9


def test_total_nodes():
    """Test total node count."""
    assert total_nodes(cpp=2, depth=2) == 3
    assert total_nodes(cpp=2, depth=3) == 7
    assert total_nodes(cpp=3, depth=2) == 4
    assert total_nodes(cpp=3, depth=3) == 13


def test_generate_all_nodes():
    """Test generating all node names."""
    # depth=1: just root
    assert generate_all_nodes(cpp=2, depth=1) == ["L1N1"]

    # cpp=2, depth=2
    assert generate_all_nodes(cpp=2, depth=2) == ["L1N1", "L2N1", "L2N2"]

    # cpp=2, depth=3
    assert generate_all_nodes(cpp=2, depth=3) == [
        "L1N1",
        "L2N1",
        "L2N2",
        "L3N1",
        "L3N2",
        "L3N3",
        "L3N4",
    ]

    # cpp=3, depth=2
    assert generate_all_nodes(cpp=3, depth=2) == [
        "L1N1",
        "L2N1",
        "L2N2",
        "L2N3",
    ]
