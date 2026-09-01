#!/usr/bin/env python
"""Topology comparison experiment for the emergent MCA system.

Tests the prediction from docs/topology-learnings.md §4: at comparable agent
counts, many small sibling groups should preserve perspective diversity better
than one large group. The default comparison is cpp=6/depth=2 (7 agents, one
6-leaf group) against cpp=2/depth=3 (7 agents, two 2-leaf groups), scored by
the phase metrics in mca_report.json.

Each (config, task) pair is one full MCA run, so this costs real LLM calls.
Use --dry_run to preview the plan.

Usage:
    poetry run python scripts/topology_experiment.py [options]

Options:
    --configs CPP,DEPTH [CPP,DEPTH ...]   Topologies to compare (default: 6,2 2,3)
    --tasks TASK [TASK ...]               Task battery (default: 3 built-in tasks)
    --model MODEL                         LLM model (default: gpt-4o-mini)
    --max_rounds N                        Rounds per run (default: 3)
    --lateral_pressure LEVEL              maintain | balanced | integrate
    --output_dir DIR                      Experiment output root
    --dry_run                             Print the plan without running
"""

import argparse
import datetime
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strange_mca.tree_helpers import total_nodes  # noqa: E402

DEFAULT_CONFIGS = ["6,2", "2,3"]

DEFAULT_TASKS = [
    "What are the most important trade-offs in designing a public transit "
    "system for a mid-sized city?",
    "Propose a novel approach to reducing food waste in urban households.",
    "Should social media platforms be liable for content recommended by "
    "their algorithms? Argue the strongest cases.",
]


def parse_config(spec: str) -> tuple[int, int]:
    """Parse and validate 'CPP,DEPTH' into (cpp, depth)."""
    try:
        cpp_str, depth_str = spec.split(",")
        cpp, depth = int(cpp_str), int(depth_str)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid config {spec!r}: expected 'CPP,DEPTH' (e.g. '3,2')"
        ) from exc
    # Validate up front, before any paid runs: the system requires depth >= 2,
    # and cpp >= 2 is needed for sibling groups (the experiment's subject).
    if depth < 2:
        raise SystemExit(f"Invalid config {spec!r}: depth must be >= 2")
    if cpp < 2:
        raise SystemExit(
            f"Invalid config {spec!r}: cpp must be >= 2 (single-child trees "
            "have no sibling groups to measure)"
        )
    return cpp, depth


def summarize_runs(run_reports: list[dict]) -> dict:
    """Aggregate phase metrics across a config's runs."""
    finals_sibling = []
    finals_cross = []
    classifications: dict[str, int] = {}
    rounds_used = []

    for report in run_reports:
        rounds_used.append(report.get("convergence", {}).get("rounds_used", 0))
        phase = report.get("summary_metrics", {}).get("phase_analysis")
        if not phase:
            continue
        sim = phase.get("final_mean_leaf_sibling_similarity")
        if sim is not None:
            finals_sibling.append(sim)
        cross = phase.get("final_cross_group_similarity")
        if cross is not None:
            finals_cross.append(cross)
        label = phase.get("phase_classification", "unknown")
        classifications[label] = classifications.get(label, 0) + 1

    def mean(values: list[float]):
        return round(sum(values) / len(values), 3) if values else None

    return {
        "runs": len(run_reports),
        "mean_final_leaf_sibling_similarity": mean(finals_sibling),
        "mean_final_cross_group_similarity": mean(finals_cross),
        "phase_classifications": classifications,
        "mean_rounds_used": mean([float(r) for r in rounds_used]),
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compare MCA topologies via phase metrics"
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument(
        "--lateral_pressure",
        type=str,
        choices=["maintain", "balanced", "integrate"],
        default="balanced",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    configs = [parse_config(spec) for spec in args.configs]

    print("Topology experiment plan:")
    for cpp, depth in configs:
        agents = total_nodes(cpp, depth)
        print(
            f"  cpp={cpp} depth={depth}: {agents} agents, "
            f"{len(args.tasks)} tasks, max {args.max_rounds} rounds each"
        )
    total_runs = len(configs) * len(args.tasks)
    print(f"  Total runs: {total_runs} (each run makes many LLM calls)")

    if args.dry_run:
        print("Dry run complete — nothing executed.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    # Import here so --dry_run works without API-dependent imports.
    from src.strange_mca.run_strange_mca import run_strange_mca  # noqa: E402

    if args.output_dir:
        exp_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join("output", f"{timestamp}_topology_experiment")
    os.makedirs(exp_dir, exist_ok=True)

    summary: dict = {
        "model": args.model,
        "max_rounds": args.max_rounds,
        "lateral_pressure": args.lateral_pressure,
        "tasks": args.tasks,
        "configs": {},
    }

    for cpp, depth in configs:
        config_key = f"cpp{cpp}_depth{depth}"
        run_reports = []
        for task_idx, task in enumerate(args.tasks, start=1):
            run_dir = os.path.join(exp_dir, f"{config_key}_task{task_idx}")
            print(f"\nRunning {config_key} task {task_idx}/{len(args.tasks)}...")
            run_strange_mca(
                task=task,
                child_per_parent=cpp,
                depth=depth,
                model=args.model,
                max_rounds=args.max_rounds,
                lateral_pressure=args.lateral_pressure,
                log_level="warning",
                output_dir=run_dir,
            )
            report_path = os.path.join(run_dir, "mca_report.json")
            with open(report_path) as f:
                run_reports.append(json.load(f))

        summary["configs"][config_key] = summarize_runs(run_reports)

    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("Topology experiment summary")
    print("=" * 72)
    header = (
        f"{'config':<16}{'sibling sim':>12}{'cross-group':>12}"
        f"{'rounds':>8}  phases"
    )
    print(header)
    for config_key, stats in summary["configs"].items():
        sib = stats["mean_final_leaf_sibling_similarity"]
        cross = stats["mean_final_cross_group_similarity"]
        print(
            f"{config_key:<16}"
            f"{sib if sib is not None else '—':>12}"
            f"{cross if cross is not None else '—':>12}"
            f"{stats['mean_rounds_used'] if stats['mean_rounds_used'] is not None else '—':>8}"
            f"  {stats['phase_classifications']}"
        )
    print(
        "\nLower sibling similarity = diversity retained; "
        "prediction: smaller groups score lower."
    )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
