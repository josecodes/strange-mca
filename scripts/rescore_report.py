#!/usr/bin/env python
"""Re-score saved runs with a different similarity method.

Reads ``final_state.json`` and the original ``mca_report.json`` (for config)
from each run directory and writes ``mca_report_<method>.json`` with the
phase metrics recomputed under the requested similarity method. The run's
LLM outputs are untouched — this only re-analyzes them, so it costs at most
one embedding request per run.

Usage:
    poetry run python scripts/rescore_report.py --similarity_method embedding \
        output/<run_dir> [output/<run_dir> ...]
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strange_mca.run_strange_mca import build_mca_report  # noqa: E402


def rescore(run_dir: str, method: str) -> dict:
    with open(os.path.join(run_dir, "final_state.json")) as f:
        state = json.load(f)
    with open(os.path.join(run_dir, "mca_report.json")) as f:
        original = json.load(f)
    config = {**original["config"], "similarity_method": method}
    report = build_mca_report(
        state, original["task"], config, similarity_method=method
    )
    out_path = os.path.join(run_dir, f"mca_report_{method}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Re-score saved MCA runs")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument(
        "--similarity_method",
        choices=["jaccard", "embedding"],
        default="embedding",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        report = rescore(run_dir, args.similarity_method)
        phase = report["summary_metrics"].get("phase_analysis")
        if phase is None:
            print(f"{run_dir}: no phase analysis (config lacks cpp/depth)")
            continue
        print(
            f"{run_dir}: sibling {phase['final_mean_leaf_sibling_similarity']} "
            f"(baseline {phase['baseline_sibling_similarity']}, "
            f"Δ {phase['final_minus_baseline']}) | cross "
            f"{phase['final_cross_group_similarity']} | root trajectory "
            f"{phase['root_similarity_trajectory']} | stability "
            f"{phase['mean_agent_stability']} | converged "
            f"{phase['genuinely_converged']} | {phase['phase_classification']}"
        )


if __name__ == "__main__":
    main()
