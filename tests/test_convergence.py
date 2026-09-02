"""Tests for the convergence module."""

import pytest

from src.strange_mca.convergence import (
    EmbeddingSimilarity,
    classify_phase,
    compute_jaccard_similarity,
    cosine_similarity,
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


# =============================================================================
# cosine_similarity / EmbeddingSimilarity Tests
# =============================================================================


def test_cosine_similarity_basic():
    """Identical direction 1.0, orthogonal 0.0, zero vector 0.0."""
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def fake_embed(texts):
    """Deterministic 2-d embedder: texts starting with 'a' -> x axis, else y."""
    return [
        [1.0, 0.0] if t.strip().lower().startswith("a") else [0.0, 1.0] for t in texts
    ]


def test_embedding_similarity_scores():
    """Same-direction texts score 1.0, orthogonal 0.0."""
    sim = EmbeddingSimilarity(embed_fn=fake_embed)
    assert sim("alpha one", "alpha two") == pytest.approx(1.0)
    assert sim("alpha", "beta") == 0.0


def test_embedding_similarity_blank_handling():
    """Blank texts follow Jaccard's convention: both blank 1.0, one blank 0.0."""
    sim = EmbeddingSimilarity(embed_fn=fake_embed)
    assert sim("", "   ") == 1.0
    assert sim("alpha", "") == 0.0


def test_embedding_similarity_warm_batches_and_caches():
    """warm() embeds unique non-blank texts once; later lookups hit the cache."""
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    sim = EmbeddingSimilarity(embed_fn=counting_embed)
    sim.warm(["alpha", "beta", "alpha", "", "  "])
    assert calls == [["alpha", "beta"]]  # deduplicated, blanks dropped
    sim("alpha", "beta")
    sim("beta", "alpha")
    assert len(calls) == 1  # served from cache
    sim("gamma", "alpha")  # new text -> one more call, for gamma only
    assert calls[1] == ["gamma"]


def test_group_metrics_accept_custom_similarity():
    """mean_pairwise / mean_cross_group use the injected similarity."""
    sim = EmbeddingSimilarity(embed_fn=fake_embed)
    # Under Jaccard these share no tokens (0.0); under the fake embedder both
    # are 'a'-texts (1.0).
    assert mean_pairwise_similarity(["alpha x", "apple y"]) == 0.0
    assert mean_pairwise_similarity(["alpha x", "apple y"], sim) == pytest.approx(1.0)
    assert mean_cross_group_similarity(
        [["alpha x"], ["apple y"]], sim
    ) == pytest.approx(1.0)
    assert mean_cross_group_similarity([["alpha x"], ["beta y"]], sim) == 0.0


def test_embedding_similarity_vector_count_mismatch_is_clear():
    """A short batch response raises a clear error, not a KeyError later."""
    sim = EmbeddingSimilarity(embed_fn=lambda texts: [[1.0, 0.0]])
    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        sim.warm(["alpha", "beta"])
