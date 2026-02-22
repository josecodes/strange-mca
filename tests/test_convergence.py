"""Tests for the convergence module."""

from src.strange_mca.convergence import compute_jaccard_similarity


def test_identical_texts():
    """Identical texts should return 1.0."""
    assert compute_jaccard_similarity("hello world", "hello world") == 1.0


def test_disjoint_texts():
    """Completely different texts should return 0.0."""
    assert compute_jaccard_similarity("hello world", "foo bar") == 0.0


def test_partial_overlap():
    """Partially overlapping texts should return between 0 and 1."""
    score = compute_jaccard_similarity("hello world foo", "hello world bar")
    # tokens: {hello, world, foo} and {hello, world, bar}
    # intersection: {hello, world} = 2
    # union: {hello, world, foo, bar} = 4
    assert score == 0.5


def test_both_empty():
    """Both empty texts should return 1.0."""
    assert compute_jaccard_similarity("", "") == 1.0


def test_one_empty():
    """One empty text should return 0.0."""
    assert compute_jaccard_similarity("hello", "") == 0.0
    assert compute_jaccard_similarity("", "hello") == 0.0


def test_case_insensitivity():
    """Similarity should be case-insensitive."""
    assert compute_jaccard_similarity("Hello World", "hello world") == 1.0


def test_whitespace_only():
    """Whitespace-only texts should be treated as empty."""
    assert compute_jaccard_similarity("   ", "   ") == 1.0
    assert compute_jaccard_similarity("   ", "hello") == 0.0


def test_duplicate_tokens():
    """Duplicate tokens within a text should not affect similarity (set-based)."""
    # "hello hello world" -> {hello, world}
    # "hello world" -> {hello, world}
    assert compute_jaccard_similarity("hello hello world", "hello world") == 1.0
