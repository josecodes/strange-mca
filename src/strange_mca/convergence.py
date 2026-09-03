"""Convergence utilities for the MCA system.

Provides Jaccard token similarity for detecting stabilization of root output
across rounds, plus phase-aware metrics (sibling-group similarity, cross-group
similarity, and phase classification) for evaluating which collective regime a
run is in. See docs/topology-learnings.md for the theory behind the phases.
"""

import math
from itertools import combinations
from typing import Callable, Optional

SimilarityFn = Callable[[str, str], float]

SIMILARITY_METHODS = ("jaccard", "embedding")

# Phase thresholds by similarity method. Jaccard values are the design doc's Q3
# numbers. Embedding values (cosine over text-embedding-3-small) are PROVISIONAL
# calibrations from three re-scored runs (see docs/experiment-log.md): cosine on
# this model is compressed into a narrow high band for same-task prose —
# independent responses from different perspectives already score ~0.93, and
# same-content paraphrases ~0.95-0.98. Absolute thresholds are therefore
# fragile; prefer the baseline-relative metrics (final_minus_baseline) and the
# within- vs cross-group ordering when reading embedding reports.
# "convergence" is the root round-over-round similarity that counts as genuine
# convergence; None means "use the run's own convergence_threshold" (the
# loop's Jaccard threshold).
PHASE_THRESHOLDS = {
    "jaccard": {"mush": 0.8, "stability": 0.8, "convergence": None},
    "embedding": {"mush": 0.97, "stability": 0.95, "convergence": 0.95},
}


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


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors (0.0 if either is all zeros)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingSimilarity:
    """Cosine similarity between texts via cached embeddings.

    Unlike Jaccard, this sees through paraphrase: two texts saying the same
    thing in different words score high. Embeddings are cached per text, and
    ``warm()`` embeds a whole batch in one call — call it with every text a
    report will compare so the report costs one embedding request.

    Args:
        embed_fn: Callable mapping a list of texts to a list of vectors. If
            None, a langchain ``OpenAIEmbeddings`` client is created lazily on
            first use (reads OPENAI_API_KEY from the environment).
        model: Embedding model name for the default client.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        model: str = "text-embedding-3-small",
    ):
        self._embed_fn = embed_fn
        self._model = model
        self._cache: dict[str, list[float]] = {}

    def _embedder(self) -> Callable[[list[str]], list[list[float]]]:
        if self._embed_fn is None:
            from langchain_openai import OpenAIEmbeddings

            self._embed_fn = OpenAIEmbeddings(model=self._model).embed_documents
        return self._embed_fn

    def warm(self, texts: list[str]) -> None:
        """Embed every uncached, non-blank text in a single batch call."""
        missing = [
            t for t in dict.fromkeys(texts) if t.strip() and t not in self._cache
        ]
        if not missing:
            return
        vectors = self._embedder()(missing)
        if len(vectors) != len(missing):
            raise RuntimeError(
                f"Embedding function returned {len(vectors)} vectors for "
                f"{len(missing)} texts"
            )
        for text, vector in zip(missing, vectors):
            self._cache[text] = vector

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self.warm([text])
        return self._cache[text]

    def __call__(self, text_a: str, text_b: str) -> float:
        blank_a, blank_b = not text_a.strip(), not text_b.strip()
        if blank_a and blank_b:
            return 1.0
        if blank_a or blank_b:
            return 0.0
        return cosine_similarity(self.embed(text_a), self.embed(text_b))


def mean_pairwise_similarity(
    texts: list[str], similarity: SimilarityFn = compute_jaccard_similarity
) -> Optional[float]:
    """Compute the mean pairwise similarity within a group of texts.

    Used as the intra-group order parameter: high values mean sibling agents
    are producing near-identical content (perspective diversity collapsing).

    Args:
        texts: The texts to compare.
        similarity: Pairwise similarity function (default: Jaccard).

    Returns:
        Mean pairwise similarity, or None if fewer than 2 texts.
    """
    if len(texts) < 2:
        return None
    scores = [similarity(a, b) for a, b in combinations(texts, 2)]
    return sum(scores) / len(scores)


def mean_cross_group_similarity(
    groups: list[list[str]], similarity: SimilarityFn = compute_jaccard_similarity
) -> Optional[float]:
    """Compute the mean similarity between members of different groups.

    Used as the cross-group order parameter: low values mean sibling groups
    hold genuinely different positions (the mosaic the architecture targets),
    high values mean groups are aligning globally.

    Args:
        groups: A list of groups, each a list of member texts.
        similarity: Pairwise similarity function (default: Jaccard).

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
                scores.append(similarity(text_a, text_b))
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
