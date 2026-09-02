"""Tests for the convergence module."""

import pytest

from src.strange_mca.convergence import (
    classify_phase,
    compute_jaccard_similarity,
    mean_cross_group_similarity,
    mean_pairwise_similarity,
)


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


# =============================================================================
# mean_pairwise_similarity Tests
# =============================================================================


def test_mean_pairwise_similarity_too_few():
    """Fewer than 2 texts should return None."""
    assert mean_pairwise_similarity([]) is None
    assert mean_pairwise_similarity(["hello"]) is None


def test_mean_pairwise_similarity_identical():
    """Identical texts should return 1.0."""
    assert mean_pairwise_similarity(["a b", "a b", "a b"]) == 1.0


def test_mean_pairwise_similarity_mixed():
    """Mean over all pairs."""
    # Pairs: (ab, ab)=1.0, (ab, cd)=0.0, (ab, cd)=0.0 -> mean = 1/3
    score = mean_pairwise_similarity(["a b", "a b", "c d"])
    assert score == pytest.approx(1 / 3)


# =============================================================================
# mean_cross_group_similarity Tests
# =============================================================================


def test_mean_cross_group_similarity_too_few_groups():
    """Fewer than 2 non-empty groups should return None."""
    assert mean_cross_group_similarity([]) is None
    assert mean_cross_group_similarity([["a b"]]) is None
    assert mean_cross_group_similarity([["a b"], []]) is None


def test_mean_cross_group_similarity_identical_groups():
    """Groups with identical content should return 1.0."""
    assert mean_cross_group_similarity([["a b"], ["a b"]]) == 1.0


def test_mean_cross_group_similarity_disjoint_groups():
    """Groups with disjoint content should return 0.0."""
    assert mean_cross_group_similarity([["a b", "a c"], ["x y", "x z"]]) == 0.0


def test_mean_cross_group_similarity_ignores_within_group_pairs():
    """Only inter-group pairs are compared."""
    # Within-group pair ("a b", "a b") would score 1.0 but must not count:
    # inter-group pairs are ("a b","x y") and ("a b","x y") -> 0.0
    assert mean_cross_group_similarity([["a b", "a b"], ["x y"]]) == 0.0


# =============================================================================
# classify_phase Tests
# =============================================================================


def test_classify_phase_unknown():
    """No convergence measurement -> unknown."""
    assert classify_phase(None, 0.5, 0.5) == "unknown"


def test_classify_phase_converged_hierarchical():
    """Converged with diverse siblings -> the target phase."""
    assert classify_phase(True, 0.4, 0.9) == "converged_hierarchical"


def test_classify_phase_converged_unmeasured():
    """Converged but sibling diversity never measured -> no diversity claim."""
    assert classify_phase(True, None, None) == "converged_unmeasured"
    assert classify_phase(True, None, 0.9) == "converged_unmeasured"


def test_classify_phase_converged_collapsed():
    """Converged with near-identical siblings -> consensus mush."""
    assert classify_phase(True, 0.85, 0.9) == "converged_collapsed"


def test_classify_phase_stuck():
    """Not converged but agents settled -> glassy signature."""
    assert classify_phase(False, 0.4, 0.9) == "stuck"


def test_classify_phase_oscillating():
    """Not converged and agents still changing -> disordered."""
    assert classify_phase(False, 0.4, 0.3) == "oscillating"
    assert classify_phase(False, 0.4, None) == "oscillating"


def test_classify_phase_custom_thresholds():
    """Thresholds are configurable."""
    assert classify_phase(True, 0.5, 0.9, mush_threshold=0.4) == "converged_collapsed"
    assert classify_phase(False, 0.4, 0.5, stability_threshold=0.4) == "stuck"
