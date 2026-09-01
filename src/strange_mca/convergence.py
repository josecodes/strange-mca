"""Convergence utilities for the MCA system.

Provides Jaccard token similarity for detecting stabilization of root output
across rounds, plus phase-aware metrics (sibling-group similarity, cross-group
similarity, and phase classification) for evaluating which collective regime a
run is in. See docs/topology-learnings.md for the theory behind the phases.
"""

from itertools import combinations
from typing import Optional


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


def mean_pairwise_similarity(texts: list[str]) -> Optional[float]:
    """Compute the mean pairwise Jaccard similarity within a group of texts.

    Used as the intra-group order parameter: high values mean sibling agents
    are producing near-identical content (perspective diversity collapsing).

    Args:
        texts: The texts to compare.

    Returns:
        Mean pairwise similarity, or None if fewer than 2 texts.
    """
    if len(texts) < 2:
        return None
    scores = [compute_jaccard_similarity(a, b) for a, b in combinations(texts, 2)]
    return sum(scores) / len(scores)


def mean_cross_group_similarity(groups: list[list[str]]) -> Optional[float]:
    """Compute the mean Jaccard similarity between members of different groups.

    Used as the cross-group order parameter: low values mean sibling groups
    hold genuinely different positions (the mosaic the architecture targets),
    high values mean groups are aligning globally.

    Args:
        groups: A list of groups, each a list of member texts.

    Returns:
        Mean similarity over all inter-group text pairs, or None if fewer
        than 2 non-empty groups.
    """
    non_empty = [g for g in groups if g]
    if len(non_empty) < 2:
        return None
    scores = []
    for group_a, group_b in combinations(non_empty, 2):
        for text_a in group_a:
            for text_b in group_b:
                scores.append(compute_jaccard_similarity(text_a, text_b))
    return sum(scores) / len(scores)


def classify_phase(
    converged: Optional[bool],
    final_sibling_similarity: Optional[float],
    agent_stability: Optional[float],
    mush_threshold: float = 0.8,
    stability_threshold: float = 0.8,
) -> str:
    """Classify a run's trajectory into a collective phase.

    Phases (see docs/topology-learnings.md §4):
    - "converged_collapsed": root stabilized but sibling responses are
      near-identical — perspective diversity washed out (the fully ordered
      phase / consensus mush).
    - "converged_hierarchical": root stabilized while sibling responses stayed
      distinct — the target phase (internal coherence, cross-perspective
      diversity).
    - "converged_unmeasured": root stabilized but sibling diversity was never
      measurable (no sibling groups — e.g. cpp=1), so no diversity claim can
      be made.
    - "stuck": root did not stabilize although the non-root agents did —
      settled, unresolved disagreement the rounds cannot move (the glassy
      signature).
    - "oscillating": root did not stabilize and agents kept changing — the
      disordered phase.
    - "unknown": not enough rounds to judge (no convergence scores).

    Args:
        converged: Whether the root output genuinely stabilized (its final
            similarity score met the threshold), or None if unmeasurable.
        final_sibling_similarity: Mean leaf sibling similarity in the last
            round, or None.
        agent_stability: Mean round-over-round self-similarity across non-root
            agents, or None. The root is excluded because its instability is
            what "not converged" already measures; "stuck" specifically means
            the rest of the system settled while the root did not.
        mush_threshold: Sibling similarity above which diversity is considered
            collapsed.
        stability_threshold: Agent self-similarity above which agents are
            considered settled.

    Returns:
        One of the phase labels above.
    """
    if converged is None:
        return "unknown"
    if converged:
        if final_sibling_similarity is None:
            return "converged_unmeasured"
        if final_sibling_similarity >= mush_threshold:
            return "converged_collapsed"
        return "converged_hierarchical"
    if agent_stability is not None and agent_stability >= stability_threshold:
        return "stuck"
    return "oscillating"
