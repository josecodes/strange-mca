"""Tests for the main module."""

from strange_mca.main import hello


def test_hello_default():
    """Test the hello function with default arguments."""
    assert hello() == "Hello, World!"


def test_hello_name():
    """Test the hello function with a custom name."""
    assert hello("Poetry") == "Hello, Poetry!" 