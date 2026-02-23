"""Convergence utilities for the MCA system.

Provides Jaccard token similarity for detecting stabilization of root output
across rounds.
"""


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard token similarity between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
