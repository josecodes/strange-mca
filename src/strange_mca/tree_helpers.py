"""Tree helper functions for the MCA agent hierarchy.

Provides pure functions for computing node names, parent-child relationships,
sibling groups, and tree traversal. Used by agents.py, graph.py, and visualization.py.
"""

from typing import Optional


def parse_node_name(node_name: str) -> tuple[int, int]:
    """Parse 'L{level}N{num}' into (level, num).

    Args:
        node_name: Node name like 'L1N1', 'L2N3', etc.

    Returns:
        Tuple of (level, node_number).

    Example:
        >>> parse_node_name('L2N3')
        (2, 3)
    """
    parts = node_name.split("N")
    level = int(parts[0][1:])
    num = int(parts[1])
    return level, num


def make_node_name(level: int, num: int) -> str:
    """Create node name from level and node number.

    Args:
        level: Tree level (1 = root).
        num: Node number within level (1-indexed).

    Returns:
        Node name like 'L1N1'.
    """
    return f"L{level}N{num}"


def get_children(node_name: str, cpp: int, depth: int) -> list[str]:
    """Get child node names for a given node.

    Args:
        node_name: Parent node name.
        cpp: Children per parent.
        depth: Total tree depth.

    Returns:
        List of child node names, empty if leaf.

    Example:
        >>> get_children('L1N1', 2, 3)
        ['L2N1', 'L2N2']
        >>> get_children('L2N2', 2, 3)
        ['L3N3', 'L3N4']
    """
    level, num = parse_node_name(node_name)
    if level >= depth:
        return []
    child_level = level + 1
    start = (num - 1) * cpp + 1
    return [make_node_name(child_level, start + i) for i in range(cpp)]


def get_parent(node_name: str, cpp: int) -> Optional[str]:
    """Get parent node name.

    Args:
        node_name: Child node name.
        cpp: Children per parent.

    Returns:
        Parent node name, or None if root.
    """
    level, num = parse_node_name(node_name)
    if level == 1:
        return None
    parent_num = (num - 1) // cpp + 1
    return make_node_name(level - 1, parent_num)


def get_siblings(node_name: str, cpp: int, depth: int) -> list[str]:
    """Get sibling node names (children of same parent, excluding self).

    Args:
        node_name: The node whose siblings to find.
        cpp: Children per parent.
        depth: Total tree depth.

    Returns:
        List of sibling node names, empty if root.
    """
    parent = get_parent(node_name, cpp)
    if parent is None:
        return []
    children = get_children(parent, cpp, depth)
    return [c for c in children if c != node_name]


def is_leaf(level: int, depth: int) -> bool:
    """Check if a node at given level is a leaf."""
    return level == depth


def is_root(level: int) -> bool:
    """Check if a node at given level is root."""
    return level == 1


def count_nodes_at_level(level: int, cpp: int) -> int:
    """Count nodes at a given level."""
    return cpp ** (level - 1)


def total_nodes(cpp: int, depth: int) -> int:
    """Calculate total nodes in tree."""
    return sum(count_nodes_at_level(level, cpp) for level in range(1, depth + 1))


def generate_all_nodes(cpp: int, depth: int) -> list[str]:
    """Generate all node names in level order.

    Args:
        cpp: Children per parent.
        depth: Total tree depth.

    Returns:
        List of all node names from root to leaves, in level order.
    """
    nodes = []
    for level in range(1, depth + 1):
        count = count_nodes_at_level(level, cpp)
        for node_num in range(1, count + 1):
            nodes.append(make_node_name(level, node_num))
    return nodes
