"""
Script to run the Strange MCA system programmatically.

Provides a function to run the emergent MCA system with specified parameters.
"""

import copy
import json
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

from src.strange_mca.agents import build_agent_tree
from src.strange_mca.convergence import (
    classify_phase,
    compute_jaccard_similarity,
    mean_cross_group_similarity,
    mean_pairwise_similarity,
)
from src.strange_mca.graph import create_execution_graph, run_execution_graph
from src.strange_mca.main import create_output_dir
from src.strange_mca.tree_helpers import (
    generate_all_nodes,
    get_children,
    is_leaf,
    parse_node_name,
    total_nodes,
)
from src.strange_mca.visualization import visualize_agent_tree, visualize_langgraph

load_dotenv()

logger = logging.getLogger("strange_mca")


def _compute_phase_analysis(
    agent_history: dict,
    config: dict,
    convergence_scores: list[float],
    rounds: list[dict],
) -> Optional[dict]:
    """Compute phase-aware metrics and attach per-round data to ``rounds``.

    Requires ``config`` to carry integer ``cpp`` and ``depth`` describing the
    topology; returns None (and attaches nothing) when they are absent so that
    callers with partial configs keep working.

    Adds a ``phase_metrics`` entry to each round dict (sibling-group
    similarity, mean leaf sibling similarity, cross-group similarity) and
    returns the summary phase analysis. See docs/topology-learnings.md §4 for
    what these metrics mean.
    """
    cpp = config.get("cpp")
    depth = config.get("depth")
    if (
        not isinstance(cpp, int)
        or not isinstance(depth, int)
        or cpp < 1
        or depth < 2
        or not agent_history
    ):
        return None

    # Sibling groups keyed by parent node; a group is a parent's children.
    sibling_groups: dict[str, list[str]] = {}
    leaf_group_parents: list[str] = []
    for node in generate_all_nodes(cpp, depth):
        level, _ = parse_node_name(node)
        if is_leaf(level, depth):
            continue
        children = get_children(node, cpp, depth)
        if len(children) < 2:
            continue
        sibling_groups[node] = children
        if is_leaf(level + 1, depth):
            leaf_group_parents.append(node)

    def round_text(name: str, round_idx: int) -> Optional[str]:
        history = agent_history.get(name, [])
        if round_idx < len(history):
            rd = history[round_idx]
            return rd.get("lateral_response", rd.get("response"))
        return None

    def opt_round(value: Optional[float]) -> Optional[float]:
        return round(value, 3) if value is not None else None

    mean_leaf_trajectory: list[Optional[float]] = []
    cross_group_trajectory: list[Optional[float]] = []
    for round_idx in range(len(rounds)):
        by_group: dict[str, Optional[float]] = {}
        for parent, members in sibling_groups.items():
            texts = [
                t for t in (round_text(m, round_idx) for m in members) if t is not None
            ]
            by_group[parent] = mean_pairwise_similarity(texts)

        leaf_sims = [
            by_group[p] for p in leaf_group_parents if by_group.get(p) is not None
        ]
        mean_leaf = sum(leaf_sims) / len(leaf_sims) if leaf_sims else None

        leaf_group_texts = []
        for parent in leaf_group_parents:
            texts = [
                t
                for t in (round_text(m, round_idx) for m in sibling_groups[parent])
                if t is not None
            ]
            leaf_group_texts.append(texts)
        cross = mean_cross_group_similarity(leaf_group_texts)

        mean_leaf_trajectory.append(mean_leaf)
        cross_group_trajectory.append(cross)
        rounds[round_idx]["phase_metrics"] = {
            "sibling_similarity_by_group": {
                parent: opt_round(sim) for parent, sim in by_group.items()
            },
            "mean_leaf_sibling_similarity": opt_round(mean_leaf),
            "cross_group_similarity": opt_round(cross),
        }

    # Agent stability: mean similarity of each non-root agent's consecutive
    # round texts. The root is excluded — its instability is what the
    # convergence score already measures, and "stuck" specifically means the
    # rest of the system settled while the root did not.
    stabilities = []
    for name in agent_history:
        if name == "L1N1":
            continue
        finals = [
            t
            for t in (round_text(name, i) for i in range(len(rounds)))
            if t is not None
        ]
        if len(finals) >= 2:
            per_agent = [
                compute_jaccard_similarity(a, b) for a, b in zip(finals, finals[1:])
            ]
            stabilities.append(sum(per_agent) / len(per_agent))
    agent_stability = sum(stabilities) / len(stabilities) if stabilities else None

    # Genuine convergence: the final root score met the threshold. The state's
    # "converged" flag also goes True on the max_rounds cap, which must not
    # count as convergence for phase classification.
    threshold = config.get("convergence_threshold")
    genuinely_converged = None
    if convergence_scores and isinstance(threshold, (int, float)):
        genuinely_converged = convergence_scores[-1] >= threshold

    final_sibling = next(
        (s for s in reversed(mean_leaf_trajectory) if s is not None), None
    )
    final_cross = next(
        (s for s in reversed(cross_group_trajectory) if s is not None), None
    )

    # Classify on the same rounded values the report publishes, so a consumer
    # re-deriving the phase from the report's numbers gets the same label.
    final_sibling_rounded = opt_round(final_sibling)
    agent_stability_rounded = opt_round(agent_stability)

    return {
        "mean_leaf_sibling_similarity_trajectory": [
            opt_round(s) for s in mean_leaf_trajectory
        ],
        "cross_group_similarity_trajectory": [
            opt_round(s) for s in cross_group_trajectory
        ],
        "final_mean_leaf_sibling_similarity": final_sibling_rounded,
        "final_cross_group_similarity": opt_round(final_cross),
        "mean_agent_stability": agent_stability_rounded,
        "genuinely_converged": genuinely_converged,
        "phase_classification": classify_phase(
            genuinely_converged, final_sibling_rounded, agent_stability_rounded
        ),
    }


def build_mca_report(result: dict, task: str, config: dict) -> dict:
    """Build an MCA report from execution results.

    Args:
        result: The execution result state.
        task: The original task.
        config: Configuration parameters.

    Returns:
        Report dictionary suitable for JSON serialization.
    """
    agent_history = result.get("agent_history", {})
    convergence_scores = result.get("convergence_scores", [])

    # Build per-round agent data
    max_round_count = 0
    for _name, history in agent_history.items():
        max_round_count = max(max_round_count, len(history))

    rounds = []
    for round_idx in range(max_round_count):
        round_data = {"round": round_idx + 1, "agents": {}}
        for name, history in agent_history.items():
            if round_idx < len(history):
                round_data["agents"][name] = history[round_idx]
        # Add convergence score (scores start from round 2)
        score_idx = round_idx - 1
        if 0 <= score_idx < len(convergence_scores):
            round_data["convergence_score"] = convergence_scores[score_idx]
        else:
            round_data["convergence_score"] = None
        rounds.append(round_data)

    # Count total LLM calls precisely by inspecting round data fields:
    # - "response" present -> 1 call (initial respond or observe)
    # - "revised" is True OR lateral_response differs from response -> 1 call
    #   (agent was actually invoked for lateral communication)
    # - "signal_sent" present -> 1 call (signal generation)
    total_llm_calls = 0
    for _name, h in agent_history.items():
        for rd in h:
            if "response" in rd:
                total_llm_calls += 1
            if rd.get("revised", False) or (
                "lateral_response" in rd
                and rd["lateral_response"] != rd.get("response")
            ):
                total_llm_calls += 1
            if "signal_sent" in rd:
                total_llm_calls += 1
    revision_counts = {}
    total_lateral_phases = 0
    total_revised = 0
    for name, history in agent_history.items():
        rev_count = sum(1 for rd in history if rd.get("revised", False))
        revision_counts[name] = rev_count
        total_lateral_phases += len(history)
        total_revised += rev_count

    lateral_revision_rate = (
        total_revised / total_lateral_phases if total_lateral_phases > 0 else 0.0
    )

    phase_analysis = _compute_phase_analysis(
        agent_history, config, convergence_scores, rounds
    )

    summary_metrics = {
        "total_llm_calls": total_llm_calls,
        "lateral_revision_rate": round(lateral_revision_rate, 3),
        "per_agent_revision_counts": revision_counts,
    }
    if phase_analysis is not None:
        summary_metrics["phase_analysis"] = phase_analysis

    report = {
        "task": task,
        "config": config,
        "rounds": rounds,
        "convergence": {
            "converged": result.get("converged", False),
            "rounds_used": max_round_count,
            "score_trajectory": convergence_scores,
        },
        "summary_metrics": summary_metrics,
        "final_response": result.get("final_response", ""),
    }

    if result.get("strange_loops"):
        report["strange_loops"] = result["strange_loops"]

    return report


def run_strange_mca(
    task: str,
    child_per_parent: int = 3,
    depth: int = 2,
    model: str = "gpt-4o-mini",
    max_rounds: int = 3,
    convergence_threshold: float = 0.85,
    enable_downward_signals: bool = True,
    perspectives: Optional[list[str]] = None,
    strange_loop_count: int = 0,
    domain_specific_instructions: str = "",
    lateral_pressure: str = "balanced",
    log_level: str = "info",
    viz: bool = False,
    local_logs_only: bool = False,
    print_details: bool = False,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Run the Strange MCA system.

    Args:
        task: The task to run.
        child_per_parent: Children per parent node.
        depth: Tree depth.
        model: LLM model name.
        max_rounds: Maximum rounds for convergence.
        convergence_threshold: Jaccard similarity threshold (0-1).
        enable_downward_signals: Enable parent-to-child signals.
        perspectives: Custom perspectives for leaf agents.
        strange_loop_count: Strange loop iterations at finalization.
        domain_specific_instructions: Domain-specific instructions for strange loop.
        lateral_pressure: How strongly lateral prompts push toward peer
            agreement — "maintain", "balanced", or "integrate".
        log_level: Logging level.
        viz: Generate visualizations.
        local_logs_only: Suppress dependency logs.
        print_details: Print detailed state.
        output_dir: Output directory (auto-generated if None).

    Returns:
        The execution result dict.
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    if local_logs_only:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=numeric_level)

    # Create output directory
    if output_dir is None:
        output_dir = create_output_dir(child_per_parent, depth, model)
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info("Running with the following parameters:")
    logger.info(f"  Task: {task}")
    logger.info(f"  Children per parent: {child_per_parent}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Max rounds: {max_rounds}")
    logger.info(f"  Convergence threshold: {convergence_threshold}")
    logger.info(f"  Downward signals: {enable_downward_signals}")
    logger.info(f"  Lateral pressure: {lateral_pressure}")

    num_agents = total_nodes(child_per_parent, depth)
    logger.info(f"Total agents: {num_agents}")

    # Visualize agent tree if requested
    if viz:
        output_file = visualize_agent_tree(
            cpp=child_per_parent,
            depth=depth,
            output_path=os.path.join(output_dir, "agent_tree"),
            format="png",
        )
        if output_file:
            logger.info(f"Agent tree visualization saved to {output_file}")

    # Build agent tree
    logger.info("Building agent tree...")
    agents = build_agent_tree(
        cpp=child_per_parent,
        depth=depth,
        model_name=model,
        perspectives=perspectives,
    )

    # Create execution graph
    logger.info("Creating execution graph...")
    graph, recursion_limit = create_execution_graph(
        agents=agents,
        cpp=child_per_parent,
        depth=depth,
        max_rounds=max_rounds,
        convergence_threshold=convergence_threshold,
        enable_downward_signals=enable_downward_signals,
        strange_loop_count=strange_loop_count,
        domain_specific_instructions=domain_specific_instructions,
        lateral_pressure=lateral_pressure,
    )

    # Visualize LangGraph if requested
    if viz:
        visualize_langgraph(graph, output_dir, child_per_parent, depth)

    # Run execution graph
    logger.info("Running execution graph...")
    result = run_execution_graph(
        graph=graph,
        task=task,
        recursion_limit=recursion_limit,
        log_level=log_level,
        only_local_logs=local_logs_only,
    )

    logger.info("Graph execution completed")

    if print_details:
        print("\nFull State:")
        print("=" * 80)
        state_copy = copy.deepcopy(result)
        import pprint

        pp = pprint.PrettyPrinter(indent=2, width=100)
        pp.pprint(state_copy)
        print("=" * 80)

    # Build and save MCA report
    report_config = {
        "cpp": child_per_parent,
        "depth": depth,
        "model": model,
        "max_rounds": max_rounds,
        "convergence_threshold": convergence_threshold,
        "enable_downward_signals": enable_downward_signals,
        "lateral_pressure": lateral_pressure,
        "perspectives": perspectives
        or [
            agents[n].config.perspective
            for n in sorted(agents.keys())
            if agents[n].config.perspective
        ],
    }
    report = build_mca_report(result, task, report_config)

    report_file = os.path.join(output_dir, "mca_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"MCA report saved to {report_file}")

    # Also save raw state
    state_file = os.path.join(output_dir, "final_state.json")
    with open(state_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = run_strange_mca(
        task="Explain the concept of recursion",
        child_per_parent=3,
        depth=2,
        model="gpt-4o-mini",
        log_level="info",
        viz=False,
        max_rounds=2,
    )
