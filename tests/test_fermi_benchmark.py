"""Tests for Fermi benchmark utility functions.

Tests extraction, scoring, and aggregation — all deterministic, no LLM calls.
"""

import math

import pytest

from examples.fermi.fermi_utils import (
    compute_aggregate_stats,
    extract_numeric_answer,
    format_summary_table,
    log_error,
)
from examples.fermi.run_fermi_benchmark import extract_mca_estimate

# --- extract_numeric_answer tests ---


class TestExtractFinalEstimateLine:
    """Tests for the primary FINAL ESTIMATE: extraction path."""

    def test_plain_integer(self):
        text = "Some reasoning.\nFINAL ESTIMATE: 225"
        assert extract_numeric_answer(text) == 225.0

    def test_scientific_notation(self):
        text = "Lots of cells.\nFINAL ESTIMATE: 3.7e13"
        assert extract_numeric_answer(text) == 3.7e13

    def test_comma_separated(self):
        text = "My estimate.\nFINAL ESTIMATE: 1,500,000"
        assert extract_numeric_answer(text) == 1_500_000.0

    def test_x_notation(self):
        text = "The atmosphere.\nFINAL ESTIMATE: 5.15 x 10^18"
        assert extract_numeric_answer(text) == 5.15e18

    def test_case_insensitive(self):
        text = "final estimate: 42"
        assert extract_numeric_answer(text) == 42.0

    def test_with_extra_whitespace(self):
        text = "FINAL ESTIMATE:   300  "
        assert extract_numeric_answer(text) == 300.0


class TestExtractWordMultipliers:
    """Tests for word multiplier extraction (e.g., '2.5 billion')."""

    def test_billion_in_final_estimate(self):
        text = "FINAL ESTIMATE: 2.5 billion"
        assert extract_numeric_answer(text) == 2.5e9

    def test_million_in_final_estimate(self):
        text = "FINAL ESTIMATE: 3.7 million"
        assert extract_numeric_answer(text) == 3.7e6

    def test_trillion_in_final_estimate(self):
        text = "FINAL ESTIMATE: 27 trillion"
        assert extract_numeric_answer(text) == 27e12

    def test_thousand_in_text(self):
        text = "I estimate approximately 150 thousand gas stations."
        assert extract_numeric_answer(text) == 150_000.0


class TestExtractKeywordFallback:
    """Tests for keyword-based extraction fallback."""

    def test_approximately_keyword(self):
        text = "Based on my analysis, the answer is approximately 45000 flights."
        assert extract_numeric_answer(text) == 45000.0

    def test_estimate_keyword(self):
        text = "My estimate is 3700000 teachers in total."
        assert extract_numeric_answer(text) == 3_700_000.0

    def test_roughly_keyword(self):
        text = "There are roughly 100000 hairs on a head."
        assert extract_numeric_answer(text) == 100_000.0


class TestExtractLastNumberFallback:
    """Tests for the last-number-in-text fallback."""

    def test_last_number(self):
        text = "Step 1: 330M people. Step 2: 50M households. Step 3: 225 tuners."
        assert extract_numeric_answer(text) == 225.0

    def test_single_number(self):
        text = "42"
        assert extract_numeric_answer(text) == 42.0

    def test_scientific_notation_in_text(self):
        text = "The total is 7.5e18 grains of sand across all beaches."
        assert extract_numeric_answer(text) == 7.5e18


class TestExtractEdgeCases:
    """Tests for edge cases in extraction."""

    def test_empty_string(self):
        assert extract_numeric_answer("") is None

    def test_none_input(self):
        assert extract_numeric_answer(None) is None

    def test_no_numbers(self):
        assert extract_numeric_answer("no numbers here at all") is None

    def test_whitespace_only(self):
        assert extract_numeric_answer("   \n\t  ") is None

    def test_negative_number(self):
        text = "FINAL ESTIMATE: -500"
        # Should still extract, though negative doesn't make sense for Fermi
        assert extract_numeric_answer(text) == -500.0


# --- log_error tests ---


class TestLogError:
    """Tests for log error computation."""

    def test_perfect_match(self):
        assert log_error(225, 225) == 0.0

    def test_one_order_of_magnitude(self):
        result = log_error(100, 1000)
        assert result == pytest.approx(1.0)

    def test_two_orders_of_magnitude(self):
        result = log_error(10, 1000)
        assert result == pytest.approx(2.0)

    def test_half_order(self):
        # log10(sqrt(10)) ≈ 0.5
        result = log_error(math.sqrt(10), 1)
        assert result == pytest.approx(0.5)

    def test_overestimate_same_as_underestimate(self):
        # Symmetric: 10x over and 10x under should give same error
        assert log_error(1000, 100) == log_error(10, 100)

    def test_zero_estimate(self):
        assert log_error(0, 100) is None

    def test_zero_actual(self):
        assert log_error(100, 0) is None

    def test_negative_estimate(self):
        assert log_error(-100, 100) is None

    def test_none_estimate(self):
        assert log_error(None, 100) is None

    def test_none_actual(self):
        assert log_error(100, None) is None

    def test_large_numbers(self):
        # 3.7e13 vs 3.7e13 should be 0
        assert log_error(3.7e13, 3.7e13) == pytest.approx(0.0)

    def test_large_number_off_by_one_oom(self):
        result = log_error(3.7e14, 3.7e13)
        assert result == pytest.approx(1.0)


# --- compute_aggregate_stats tests ---


class TestComputeAggregateStats:
    """Tests for aggregate statistics computation."""

    def test_basic_stats(self):
        results = [
            {
                "mca_log_error": 0.5,
                "baseline_log_error": 1.0,
                "mca_estimate": 100,
                "baseline_estimate": 100,
            },
            {
                "mca_log_error": 0.3,
                "baseline_log_error": 0.8,
                "mca_estimate": 100,
                "baseline_estimate": 100,
            },
            {
                "mca_log_error": 1.5,
                "baseline_log_error": 0.4,
                "mca_estimate": 100,
                "baseline_estimate": 100,
            },
        ]
        stats = compute_aggregate_stats(results)

        assert stats["total_questions"] == 3
        assert stats["mca"]["count"] == 3
        assert stats["baseline"]["count"] == 3
        # MCA mean: (0.5 + 0.3 + 1.5) / 3 = 0.767
        assert stats["mca"]["mean_log_error"] == pytest.approx(0.767, abs=0.001)
        # Baseline mean: (1.0 + 0.8 + 0.4) / 3 = 0.733
        assert stats["baseline"]["mean_log_error"] == pytest.approx(0.733, abs=0.001)

    def test_within_oom_thresholds(self):
        results = [
            {
                "mca_log_error": 0.3,
                "baseline_log_error": 0.3,
                "mca_estimate": 1,
                "baseline_estimate": 1,
            },
            {
                "mca_log_error": 0.7,
                "baseline_log_error": 1.5,
                "mca_estimate": 1,
                "baseline_estimate": 1,
            },
            {
                "mca_log_error": 1.2,
                "baseline_log_error": 0.9,
                "mca_estimate": 1,
                "baseline_estimate": 1,
            },
        ]
        stats = compute_aggregate_stats(results)

        # MCA: 0.3 (<0.5 and <1), 0.7 (<1 but not <0.5), 1.2 (neither)
        assert stats["mca"]["within_1_oom"] == pytest.approx(2 / 3, abs=0.001)
        assert stats["mca"]["within_0_5_oom"] == pytest.approx(1 / 3, abs=0.001)

    def test_mca_win_rate(self):
        results = [
            {
                "mca_log_error": 0.5,
                "baseline_log_error": 1.0,
                "mca_estimate": 1,
                "baseline_estimate": 1,
            },
            {
                "mca_log_error": 0.3,
                "baseline_log_error": 0.8,
                "mca_estimate": 1,
                "baseline_estimate": 1,
            },
            {
                "mca_log_error": 1.5,
                "baseline_log_error": 0.4,
                "mca_estimate": 1,
                "baseline_estimate": 1,
            },
        ]
        stats = compute_aggregate_stats(results)
        # MCA wins 2 out of 3
        assert stats["mca_win_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_extraction_failures(self):
        results = [
            {
                "mca_log_error": 0.5,
                "baseline_log_error": None,
                "mca_estimate": 100,
                "baseline_estimate": None,
            },
            {
                "mca_log_error": None,
                "baseline_log_error": 0.8,
                "mca_estimate": None,
                "baseline_estimate": 100,
            },
        ]
        stats = compute_aggregate_stats(results)

        assert stats["mca"]["count"] == 1
        assert stats["baseline"]["count"] == 1
        assert stats["mca"]["extraction_failure_rate"] == 0.5
        assert stats["baseline"]["extraction_failure_rate"] == 0.5

    def test_empty_results(self):
        stats = compute_aggregate_stats([])
        assert stats["total_questions"] == 0
        assert stats["mca_win_rate"] is None

    def test_all_none_errors(self):
        results = [
            {
                "mca_log_error": None,
                "baseline_log_error": None,
                "mca_estimate": None,
                "baseline_estimate": None,
            },
        ]
        stats = compute_aggregate_stats(results)
        assert stats["mca"]["mean_log_error"] is None
        assert stats["baseline"]["mean_log_error"] is None
        assert stats["mca_win_rate"] is None


# --- format_summary_table tests ---


class TestFormatSummaryTable:
    """Tests for summary table formatting."""

    def test_produces_string(self):
        results = [
            {
                "question_id": "test_q",
                "reference_answer": 100,
                "mca_estimate": 90,
                "mca_log_error": 0.046,
                "baseline_estimate": 1000,
                "baseline_log_error": 1.0,
            },
        ]
        mca_stats = {
            "mean_log_error": 0.046,
            "median_log_error": 0.046,
            "within_1_oom": 1.0,
            "within_0_5_oom": 1.0,
            "extraction_failure_rate": 0.0,
        }
        baseline_stats = {
            "mean_log_error": 1.0,
            "median_log_error": 1.0,
            "within_1_oom": 0.0,
            "within_0_5_oom": 0.0,
            "extraction_failure_rate": 0.0,
        }

        table = format_summary_table(results, mca_stats, baseline_stats)
        assert isinstance(table, str)
        assert "FERMI ESTIMATION BENCHMARK RESULTS" in table
        assert "test_q" in table

    def test_handles_none_values(self):
        results = [
            {
                "question_id": "fail_q",
                "reference_answer": 100,
                "mca_estimate": None,
                "mca_log_error": None,
                "baseline_estimate": None,
                "baseline_log_error": None,
            },
        ]
        mca_stats = {
            "mean_log_error": None,
            "median_log_error": None,
            "within_1_oom": None,
            "within_0_5_oom": None,
            "extraction_failure_rate": 1.0,
        }
        baseline_stats = {
            "mean_log_error": None,
            "median_log_error": None,
            "within_1_oom": None,
            "within_0_5_oom": None,
            "extraction_failure_rate": 1.0,
        }

        table = format_summary_table(results, mca_stats, baseline_stats)
        assert "FAIL" in table
        assert "N/A" in table


# --- extract_mca_estimate tests ---


class TestExtractMcaEstimate:
    """Tests for MCA estimate extraction with fallback."""

    def test_extracts_from_final_response(self):
        mca_result = {
            "final_response": "Here is my synthesis.\n\nFINAL ESTIMATE: 225",
            "agent_history": {},
        }
        estimate, source = extract_mca_estimate(mca_result)
        assert estimate == 225.0
        assert source == "final_response"

    def test_falls_back_to_agent_geometric_mean(self):
        mca_result = {
            "final_response": "A meta-commentary with no numbers or estimate line.",
            "agent_history": {
                "L2N1": [
                    {
                        "response": "FINAL ESTIMATE: 100",
                        "lateral_response": "FINAL ESTIMATE: 100",
                    }
                ],
                "L2N2": [
                    {
                        "response": "FINAL ESTIMATE: 1000",
                        "lateral_response": "FINAL ESTIMATE: 1000",
                    }
                ],
                "L2N3": [
                    {
                        "response": "FINAL ESTIMATE: 10000",
                        "lateral_response": "FINAL ESTIMATE: 10000",
                    }
                ],
            },
        }
        estimate, source = extract_mca_estimate(mca_result)
        assert source == "agent_geometric_mean"
        # Geometric mean of 100, 1000, 10000 = 10^((2+3+4)/3) = 10^3 = 1000
        assert estimate == pytest.approx(1000.0, rel=0.01)

    def test_falls_back_when_final_response_has_number_but_no_marker(self):
        """When final_response has a number but no FINAL ESTIMATE marker,
        prefer agent geometric mean if available."""
        mca_result = {
            "final_response": "The team used 3 different perspectives to analyze this.",
            "agent_history": {
                "L2N1": [{"response": "FINAL ESTIMATE: 500"}],
                "L2N2": [{"response": "FINAL ESTIMATE: 5000"}],
            },
        }
        estimate, source = extract_mca_estimate(mca_result)
        assert source == "agent_geometric_mean"
        # Geometric mean of 500 and 5000 = sqrt(500*5000) = sqrt(2500000) ≈ 1581
        assert estimate == pytest.approx(math.sqrt(500 * 5000), rel=0.01)

    def test_uses_lateral_response_over_response(self):
        mca_result = {
            "final_response": "Meta commentary only.",
            "agent_history": {
                "L2N1": [
                    {
                        "response": "FINAL ESTIMATE: 100",
                        "lateral_response": "FINAL ESTIMATE: 200",
                    }
                ],
            },
        }
        estimate, source = extract_mca_estimate(mca_result)
        assert source == "agent_geometric_mean"
        assert estimate == pytest.approx(200.0, rel=0.01)

    def test_returns_none_when_no_estimates_anywhere(self):
        mca_result = {
            "final_response": "Pure meta-commentary, no numbers at all.",
            "agent_history": {
                "L1N1": [{"response": "Also no numbers here."}],
            },
        }
        estimate, source = extract_mca_estimate(mca_result)
        assert estimate is None
        assert source == "extraction_failed"

    def test_skips_zero_and_negative_agent_estimates(self):
        mca_result = {
            "final_response": "No final estimate here.",
            "agent_history": {
                "L2N1": [{"response": "FINAL ESTIMATE: 0"}],
                "L2N2": [{"response": "FINAL ESTIMATE: 1000"}],
            },
        }
        estimate, source = extract_mca_estimate(mca_result)
        assert source == "agent_geometric_mean"
        assert estimate == 1000.0

    def test_empty_result(self):
        mca_result = {"final_response": "", "agent_history": {}}
        estimate, source = extract_mca_estimate(mca_result)
        assert estimate is None
        assert source == "extraction_failed"
