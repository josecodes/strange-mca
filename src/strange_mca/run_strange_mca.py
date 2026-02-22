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
from src.strange_mca.graph import create_execution_graph, run_execution_graph
from src.strange_mca.main import create_output_dir
from src.strange_mca.tree_helpers import total_nodes
from src.strange_mca.visualization import visualize_agent_tree, visualize_langgraph

load_dotenv()

logger = logging.getLogger("strange_mca")


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

    report = {
        "task": task,
        "config": config,
        "rounds": rounds,
        "convergence": {
            "converged": result.get("converged", False),
            "rounds_used": max_round_count,
            "score_trajectory": convergence_scores,
        },
        "summary_metrics": {
            "total_llm_calls": total_llm_calls,
            "lateral_revision_rate": round(lateral_revision_rate, 3),
            "per_agent_revision_counts": revision_counts,
        },
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

    logger.info(f"Output directory: {output_dir}")
    logger.info("Running with the following parameters:")
    logger.info(f"  Task: {task}")
    logger.info(f"  Children per parent: {child_per_parent}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Max rounds: {max_rounds}")
    logger.info(f"  Convergence threshold: {convergence_threshold}")
    logger.info(f"  Downward signals: {enable_downward_signals}")

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
