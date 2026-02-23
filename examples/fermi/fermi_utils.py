"""Utility functions for Fermi estimation benchmark.

Provides numeric answer extraction from LLM responses, log-error scoring,
aggregate statistics, and summary formatting.
"""

import math
import re
import statistics

# Word multipliers for parsing "2.5 billion" style answers
WORD_MULTIPLIERS = {
    "hundred": 1e2,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
    "quadrillion": 1e15,
    "quintillion": 1e18,
}

# Pattern for numbers: optional sign, digits with optional commas, optional decimal,
# optional scientific notation (e.g., 3.7e13, 3.7 x 10^13)
_NUMBER_PATTERN = re.compile(
    r"[-+]?\d[\d,]*\.?\d*(?:\s*[xX×]\s*10\s*\^\s*[-+]?\d+|[eE][-+]?\d+)?"
)

# Pattern for "FINAL ESTIMATE:" line
_FINAL_ESTIMATE_PATTERN = re.compile(r"FINAL\s+ESTIMATE\s*:\s*(.+)", re.IGNORECASE)

# Pattern for word multipliers after a number
_WORD_MULTIPLIER_PATTERN = re.compile(
    r"([-+]?\d[\d,]*\.?\d*)\s+(hundred|thousand|million|billion|trillion|quadrillion|quintillion)",
    re.IGNORECASE,
)


def _parse_number_string(s: str) -> float | None:
    """Parse a single number string into a float.

    Handles: plain numbers, commas, scientific notation (3.7e13),
    and 'x 10^' notation (3.7 x 10^13).

    Args:
        s: A string containing a number.

    Returns:
        The parsed float, or None if parsing fails.
    """
    s = s.strip()
    if not s:
        return None

    # Handle "x 10^" notation: convert to scientific notation
    x_notation = re.match(r"([-+]?\d[\d,]*\.?\d*)\s*[xX×]\s*10\s*\^\s*([-+]?\d+)", s)
    if x_notation:
        base = x_notation.group(1).replace(",", "")
        exp = x_notation.group(2)
        try:
            return float(base) * (10 ** float(exp))
        except (ValueError, OverflowError):
            return None

    # Standard number (possibly with commas or scientific notation)
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_numeric_answer(text: str) -> float | None:
    """Extract a numeric answer from LLM response text.

    Uses a cascade of extraction strategies:
    1. Parse the FINAL ESTIMATE: line (primary)
    2. Regex fallback for last number near "estimate"/"approximately" keywords
    3. Last number in the response (final fallback)

    Handles scientific notation, word multipliers, commas, and x 10^ notation.

    Args:
        text: The LLM response text.

    Returns:
        The extracted numeric answer as a float, or None if no number found.
    """
    if not text or not text.strip():
        return None

    # Strategy 1: Look for "FINAL ESTIMATE:" line
    match = _FINAL_ESTIMATE_PATTERN.search(text)
    if match:
        estimate_text = match.group(1).strip()
        # Check for word multiplier in the estimate line
        word_match = _WORD_MULTIPLIER_PATTERN.search(estimate_text)
        if word_match:
            base = float(word_match.group(1).replace(",", ""))
            multiplier = WORD_MULTIPLIERS[word_match.group(2).lower()]
            return base * multiplier
        # Try to extract a number from the estimate line
        numbers = _NUMBER_PATTERN.findall(estimate_text)
        if numbers:
            result = _parse_number_string(numbers[0])
            if result is not None:
                return result

    # Strategy 2: Look for numbers near estimation keywords
    keyword_pattern = re.compile(
        r"(?:estimate|approximately|roughly|about|around)\s*(?:is\s*)?(?::|=)?\s*"
        r"([-+]?\d[\d,]*\.?\d*(?:\s*[xX×]\s*10\s*\^\s*[-+]?\d+|[eE][-+]?\d+)?)"
        r"(?:\s+(hundred|thousand|million|billion|trillion|quadrillion|quintillion))?",
        re.IGNORECASE,
    )
    keyword_matches = list(keyword_pattern.finditer(text))
    if keyword_matches:
        last_match = keyword_matches[-1]
        number_str = last_match.group(1)
        word_mult = last_match.group(2)
        result = _parse_number_string(number_str)
        if result is not None:
            if word_mult:
                result *= WORD_MULTIPLIERS[word_mult.lower()]
            return result

    # Strategy 3: Check for word multiplier patterns anywhere in text
    word_matches = list(_WORD_MULTIPLIER_PATTERN.finditer(text))
    if word_matches:
        last_word_match = word_matches[-1]
        base = float(last_word_match.group(1).replace(",", ""))
        multiplier = WORD_MULTIPLIERS[last_word_match.group(2).lower()]
        return base * multiplier

    # Strategy 4: Last number in the response
    all_numbers = _NUMBER_PATTERN.findall(text)
    if all_numbers:
        result = _parse_number_string(all_numbers[-1])
        if result is not None:
            return result

    return None


def log_error(estimate: float, actual: float) -> float | None:
    """Compute log error between estimate and actual value.

    Standard Fermi metric: |log10(estimate) - log10(actual)|
    where 0 = perfect, <1 = within an order of magnitude.

    Args:
        estimate: The estimated value.
        actual: The reference/actual value.

    Returns:
        The absolute log10 error, or None if inputs are invalid
        (zero or negative).
    """
    if estimate is None or actual is None:
        return None
    if estimate <= 0 or actual <= 0:
        return None
    return abs(math.log10(estimate) - math.log10(actual))


def compute_aggregate_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics from benchmark results.

    Args:
        results: List of per-question result dicts, each containing:
            - mca_log_error: float or None
            - baseline_log_error: float or None
            - mca_estimate: float or None
            - baseline_estimate: float or None

    Returns:
        Dict with aggregate statistics.
    """
    mca_errors = [
        r["mca_log_error"] for r in results if r.get("mca_log_error") is not None
    ]
    baseline_errors = [
        r["baseline_log_error"]
        for r in results
        if r.get("baseline_log_error") is not None
    ]

    mca_attempted = sum(1 for r in results if r.get("mca_estimate") is not None)
    baseline_attempted = sum(
        1 for r in results if r.get("baseline_estimate") is not None
    )

    total = len(results)

    def _stats_for(errors, attempted):
        if not errors:
            return {
                "mean_log_error": None,
                "median_log_error": None,
                "within_1_oom": 0.0,
                "within_0_5_oom": 0.0,
                "extraction_failure_rate": 1.0 if total > 0 else 0.0,
                "count": 0,
            }
        return {
            "mean_log_error": round(statistics.mean(errors), 3),
            "median_log_error": round(statistics.median(errors), 3),
            "within_1_oom": round(sum(1 for e in errors if e < 1.0) / len(errors), 3),
            "within_0_5_oom": round(sum(1 for e in errors if e < 0.5) / len(errors), 3),
            "extraction_failure_rate": round(
                1.0 - (attempted / total) if total > 0 else 0.0, 3
            ),
            "count": len(errors),
        }

    mca_stats = _stats_for(mca_errors, mca_attempted)
    baseline_stats = _stats_for(baseline_errors, baseline_attempted)

    # MCA win rate: questions where MCA had lower log error
    head_to_head = [
        r
        for r in results
        if r.get("mca_log_error") is not None
        and r.get("baseline_log_error") is not None
    ]
    if head_to_head:
        mca_wins = sum(
            1 for r in head_to_head if r["mca_log_error"] < r["baseline_log_error"]
        )
        ties = sum(
            1 for r in head_to_head if r["mca_log_error"] == r["baseline_log_error"]
        )
        mca_win_rate = round((mca_wins + 0.5 * ties) / len(head_to_head), 3)
    else:
        mca_win_rate = None

    return {
        "mca": mca_stats,
        "baseline": baseline_stats,
        "mca_win_rate": mca_win_rate,
        "total_questions": total,
    }


def format_summary_table(
    results: list[dict], mca_stats: dict, baseline_stats: dict
) -> str:
    """Format benchmark results as a human-readable summary table.

    Args:
        results: List of per-question result dicts.
        mca_stats: Aggregate stats for MCA system.
        baseline_stats: Aggregate stats for baseline.

    Returns:
        Formatted string for printing to stdout.
    """
    lines = []
    lines.append("=" * 90)
    lines.append("FERMI ESTIMATION BENCHMARK RESULTS")
    lines.append("=" * 90)
    lines.append("")

    # Per-question table
    header = f"{'Question':<30} {'Reference':>12} {'MCA Est.':>12} {'MCA Err':>8} {'Base Est.':>12} {'Base Err':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for r in results:
        q_id = r.get("question_id", "?")[:30]
        ref = _format_number(r.get("reference_answer"))
        mca_est = _format_number(r.get("mca_estimate"))
        mca_err = _format_error(r.get("mca_log_error"))
        base_est = _format_number(r.get("baseline_estimate"))
        base_err = _format_error(r.get("baseline_log_error"))
        lines.append(
            f"{q_id:<30} {ref:>12} {mca_est:>12} {mca_err:>8} {base_est:>12} {base_err:>8}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Aggregate stats
    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 45)
    stat_header = f"{'Metric':<30} {'MCA':>10} {'Baseline':>10}"
    lines.append(stat_header)
    lines.append("-" * 45)

    for label, key in [
        ("Mean Log Error", "mean_log_error"),
        ("Median Log Error", "median_log_error"),
        ("Within 1 OOM (%)", "within_1_oom"),
        ("Within 0.5 OOM (%)", "within_0_5_oom"),
        ("Extraction Failures (%)", "extraction_failure_rate"),
    ]:
        mca_val = mca_stats.get(key)
        base_val = baseline_stats.get(key)
        if key in ("within_1_oom", "within_0_5_oom", "extraction_failure_rate"):
            mca_str = f"{mca_val * 100:.1f}%" if mca_val is not None else "N/A"
            base_str = f"{base_val * 100:.1f}%" if base_val is not None else "N/A"
        else:
            mca_str = f"{mca_val:.3f}" if mca_val is not None else "N/A"
            base_str = f"{base_val:.3f}" if base_val is not None else "N/A"
        lines.append(f"{label:<30} {mca_str:>10} {base_str:>10}")

    lines.append("-" * 45)
    lines.append("")

    return "\n".join(lines)


def _format_number(n) -> str:
    """Format a number for display in the summary table."""
    if n is None:
        return "FAIL"
    if abs(n) >= 1e9:
        return f"{n:.2e}"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.1f}"


def _format_error(e) -> str:
    """Format a log error for display."""
    if e is None:
        return "N/A"
    return f"{e:.2f}"
