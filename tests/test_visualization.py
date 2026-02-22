"""Tests for the visualization module."""

from unittest.mock import MagicMock, patch

from src.strange_mca.visualization import (
    print_agent_tree,
    visualize_agent_tree,
    visualize_langgraph,
)


@patch("src.strange_mca.visualization.graphviz.Digraph")
def test_visualize_agent_tree_creates_graph(mock_digraph_class, tmp_path):
    """Test that visualize_agent_tree creates a Digraph and calls render."""
    output_path = str(tmp_path / "test_tree")
    rendered_path = str(tmp_path / "test_tree.png")

    mock_digraph = MagicMock()
    mock_digraph.render.return_value = rendered_path
    mock_digraph_class.return_value = mock_digraph

    result = visualize_agent_tree(cpp=2, depth=2, output_path=output_path)

    # Digraph should be instantiated
    mock_digraph_class.assert_called_once()

    # render() should be called with the output path
    mock_digraph.render.assert_called_once_with(output_path, cleanup=True)

    # Result should be the rendered file path
    assert result == rendered_path


@patch("src.strange_mca.visualization.graphviz.Digraph")
def test_visualize_agent_tree_no_output_path(mock_digraph_class):
    """Test that visualize_agent_tree with output_path=None returns None."""
    mock_digraph = MagicMock()
    mock_digraph_class.return_value = mock_digraph

    result = visualize_agent_tree(cpp=2, depth=2, output_path=None)

    # Should return None when no output path
    assert result is None

    # render() should NOT be called
    mock_digraph.render.assert_not_called()


@patch("src.strange_mca.visualization.graphviz.Digraph")
def test_visualize_agent_tree_render_failure(mock_digraph_class, tmp_path):
    """Test that visualize_agent_tree handles render failure gracefully."""
    output_path = str(tmp_path / "test_tree")

    mock_digraph = MagicMock()
    mock_digraph.render.side_effect = Exception("Graphviz not installed")
    mock_digraph_class.return_value = mock_digraph

    result = visualize_agent_tree(cpp=2, depth=2, output_path=output_path)

    # Should return the .dot fallback path
    assert result == f"{output_path}.dot"


def test_print_agent_tree_basic(capsys):
    """Test that print_agent_tree outputs agent names."""
    print_agent_tree(cpp=2, depth=2)

    captured = capsys.readouterr()

    # Should contain agent names
    assert "L1N1" in captured.out
    assert "L2N1" in captured.out
    assert "L2N2" in captured.out

    # Should contain the tree header
    assert "Agent Tree:" in captured.out


@patch("src.strange_mca.visualization.graphviz.Digraph")
@patch("src.strange_mca.visualization.os.makedirs")
def test_visualize_langgraph_basic(mock_makedirs, mock_digraph_class, tmp_path):
    """Test that visualize_langgraph creates a graph and renders."""
    output_dir = str(tmp_path / "viz_output")
    rendered_path = str(tmp_path / "viz_output" / "execution_graph_lg.png")

    mock_digraph = MagicMock()
    mock_digraph.render.return_value = rendered_path
    mock_digraph_class.return_value = mock_digraph

    mock_graph = MagicMock()

    result = visualize_langgraph(
        graph=mock_graph, output_dir=output_dir, cpp=2, depth=2
    )

    # Digraph should be instantiated
    mock_digraph_class.assert_called_once()

    # render() should be called
    mock_digraph.render.assert_called_once()

    # Should return the rendered file path
    assert result == rendered_path

    # Output directory should be created
    mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
