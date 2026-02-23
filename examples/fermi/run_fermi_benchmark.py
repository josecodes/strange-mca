"""Fermi Estimation Benchmark for Strange MCA.

Runs Fermi estimation problems through both the MCA system and a single LLM
baseline, then compares accuracy using log error.
"""

import argparse
import datetime
import json
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add the project root to the Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from examples.fermi.fermi_utils import (
    compute_aggregate_stats,
    extract_numeric_answer,
    format_summary_table,
    log_error,
)
from src.strange_mca.run_strange_mca import run_strange_mca

load_dotenv()

logger = logging.getLogger("fermi_benchmark")


def extract_mca_estimate(mca_result: dict) -> tuple[float | None, str]:
    """Extract a numeric estimate from MCA results with fallback.

    The root integrator often produces meta-commentary without a concrete number.
    When the final_response doesn't contain a parseable estimate, fall back to
    extracting estimates from individual agent responses in the last round and
    computing their geometric mean.

    Args:
        mca_result: The full MCA result dict from run_strange_mca().

    Returns:
        Tuple of (estimate, source) where source describes where the estimate
        came from ("final_response" or "agent_geometric_mean").
    """
    final_response = mca_result.get("final_response", "")

    # Strategy 1: extract from final response
    estimate = extract_numeric_answer(final_response)
    if estimate is not None and "FINAL ESTIMATE" in final_response.upper():
        return estimate, "final_response"

    # Strategy 2: extract from individual agent responses in the last round,
    # compute geometric mean
    agent_history = mca_result.get("agent_history", {})
    agent_estimates = []
    for _name, rounds in agent_history.items():
        if not rounds:
            continue
        last_round = rounds[-1]
        # Try lateral_response first (post-peer-communication), then response
        text = last_round.get("lateral_response") or last_round.get("response", "")
        if text:
            agent_est = extract_numeric_answer(text)
            if agent_est is not None and agent_est > 0:
                agent_estimates.append(agent_est)

    if agent_estimates:
        import math

        log_mean = sum(math.log10(e) for e in agent_estimates) / len(agent_estimates)
        geo_mean = 10**log_mean
        logger.info(
            f"  Fallback: geometric mean of {len(agent_estimates)} agent estimates"
        )
        return geo_mean, "agent_geometric_mean"

    # Strategy 3: return whatever extract_numeric_answer found, even without
    # FINAL ESTIMATE marker
    if estimate is not None:
        return estimate, "final_response_fallback"

    return None, "extraction_failed"


FERMI_PROMPT_TEMPLATE = """You are solving a Fermi estimation problem. Think step by step, breaking the problem into smaller, estimable quantities. Show your reasoning clearly.

PROBLEM: {question}

After your reasoning, you MUST end your response with a line in exactly this format:
FINAL ESTIMATE: <number>

The number should be a plain numeric value (e.g., 225, 3700000, 3.7e13, 5.15e18).
Do not include units, words like "million", or any other text on the FINAL ESTIMATE line.
"""


def load_questions(questions_path: str | None = None) -> list[dict]:
    """Load Fermi questions from JSON file.

    Args:
        questions_path: Path to JSON file. Defaults to fermi_questions.json
            in the same directory as this script.

    Returns:
        List of question dicts.
    """
    if questions_path is None:
        questions_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fermi_questions.json"
        )
    with open(questions_path) as f:
        return json.load(f)


def run_baseline(question: str, model: str) -> str:
    """Run a single question through the baseline LLM.

    Args:
        question: The Fermi prompt (already formatted).
        model: The model name.

    Returns:
        The LLM response text.
    """
    llm = ChatOpenAI(model=model, temperature=0.7)
    response = llm.invoke(question)
    return response.content


def run_single_question(
    question_data: dict,
    model: str,
    child_per_parent: int,
    depth: int,
    max_rounds: int,
    convergence_threshold: float,
    enable_downward_signals: bool,
    output_dir: str,
    run_mca: bool = True,
    run_baseline_flag: bool = True,
) -> dict:
    """Run a single Fermi question through both systems.

    Args:
        question_data: Dict with question info from the JSON bank.
        model: LLM model name.
        child_per_parent: MCA tree branching factor.
        depth: MCA tree depth.
        max_rounds: MCA max convergence rounds.
        convergence_threshold: MCA convergence threshold.
        enable_downward_signals: Whether MCA uses downward signals.
        output_dir: Base output directory for this benchmark run.
        run_mca: Whether to run the MCA system.
        run_baseline_flag: Whether to run the baseline.

    Returns:
        Result dict with estimates, errors, and responses.
    """
    q_id = question_data["id"]
    question = question_data["question"]
    reference = question_data["reference_answer"]
    prompt = FERMI_PROMPT_TEMPLATE.format(question=question)

    result = {
        "question_id": q_id,
        "question": question,
        "reference_answer": reference,
        "unit": question_data.get("unit", ""),
        "category": question_data.get("category", ""),
    }

    # Run MCA
    if run_mca:
        logger.info(f"Running MCA for: {q_id}")
        mca_output_dir = os.path.join(output_dir, "mca_runs", q_id)
        os.makedirs(mca_output_dir, exist_ok=True)

        mca_result = run_strange_mca(
            task=prompt,
            child_per_parent=child_per_parent,
            depth=depth,
            model=model,
            max_rounds=max_rounds,
            convergence_threshold=convergence_threshold,
            enable_downward_signals=enable_downward_signals,
            log_level="warning",
            output_dir=mca_output_dir,
        )

        mca_response = mca_result.get("final_response", "")
        mca_estimate, estimate_source = extract_mca_estimate(mca_result)
        mca_err = log_error(mca_estimate, reference) if mca_estimate else None

        result["mca_response"] = mca_response
        result["mca_estimate"] = mca_estimate
        result["mca_estimate_source"] = estimate_source
        result["mca_log_error"] = mca_err
    else:
        result["mca_response"] = None
        result["mca_estimate"] = None
        result["mca_log_error"] = None

    # Run baseline
    if run_baseline_flag:
        logger.info(f"Running baseline for: {q_id}")
        baseline_response = run_baseline(prompt, model)
        baseline_estimate = extract_numeric_answer(baseline_response)
        baseline_err = (
            log_error(baseline_estimate, reference) if baseline_estimate else None
        )

        result["baseline_response"] = baseline_response
        result["baseline_estimate"] = baseline_estimate
        result["baseline_log_error"] = baseline_err
    else:
        result["baseline_response"] = None
        result["baseline_estimate"] = None
        result["baseline_log_error"] = None

    # Log result
    mca_str = (
        f"{result['mca_log_error']:.2f}"
        if result.get("mca_log_error") is not None
        else "N/A"
    )
    base_str = (
        f"{result['baseline_log_error']:.2f}"
        if result.get("baseline_log_error") is not None
        else "N/A"
    )
    logger.info(f"  {q_id}: MCA={mca_str}, Baseline={base_str}")

    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Fermi Estimation Benchmark: MCA vs single LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for both systems (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--child_per_parent",
        type=int,
        default=3,
        help="MCA tree branching factor (default: 3)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="MCA tree depth (default: 2)",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="MCA max convergence rounds (default: 3)",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.85,
        help="MCA convergence threshold (default: 0.85)",
    )
    parser.add_argument(
        "--no_downward_signals",
        action="store_true",
        help="Disable MCA parent-to-child signals",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        type=str,
        default=None,
        help="Filter by question IDs (space-separated)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="Filter by category (space-separated)",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Skip MCA runs, only run baseline",
    )
    parser.add_argument(
        "--mca_only",
        action="store_true",
        help="Skip baseline runs, only run MCA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level (default: warning)",
    )
    return parser.parse_args()


def main():
    """Run the Fermi estimation benchmark."""
    args = parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"fermi_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load and filter questions
    questions = load_questions()

    if args.questions:
        questions = [q for q in questions if q["id"] in args.questions]
    if args.categories:
        questions = [q for q in questions if q.get("category") in args.categories]

    if not questions:
        print("No questions matched the filters.")
        return

    print(f"Running Fermi benchmark with {len(questions)} questions")
    print(f"Model: {args.model}")
    print(
        f"MCA config: cpp={args.child_per_parent}, depth={args.depth}, "
        f"rounds={args.max_rounds}, threshold={args.convergence_threshold}"
    )
    print(f"Output: {output_dir}")
    print()

    run_mca = not args.baseline_only
    run_baseline_flag = not args.mca_only
    enable_downward_signals = not args.no_downward_signals

    # Run each question
    results = []
    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        result = run_single_question(
            question_data=q,
            model=args.model,
            child_per_parent=args.child_per_parent,
            depth=args.depth,
            max_rounds=args.max_rounds,
            convergence_threshold=args.convergence_threshold,
            enable_downward_signals=enable_downward_signals,
            output_dir=output_dir,
            run_mca=run_mca,
            run_baseline_flag=run_baseline_flag,
        )
        results.append(result)

        # Print per-question result
        mca_str = (
            f"{result['mca_log_error']:.2f}"
            if result.get("mca_log_error") is not None
            else "N/A"
        )
        base_str = (
            f"{result['baseline_log_error']:.2f}"
            if result.get("baseline_log_error") is not None
            else "N/A"
        )
        print(f"  Log error — MCA: {mca_str}, Baseline: {base_str}")
        print()

    # Compute aggregate stats
    agg_stats = compute_aggregate_stats(results)

    # Print summary
    summary_table = format_summary_table(
        results, agg_stats["mca"], agg_stats["baseline"]
    )
    print(summary_table)

    if agg_stats["mca_win_rate"] is not None:
        print(f"MCA Win Rate: {agg_stats['mca_win_rate'] * 100:.1f}%")
        print()

    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(agg_stats, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
