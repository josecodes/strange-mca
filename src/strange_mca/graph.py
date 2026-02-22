"""Flat LangGraph implementation of the emergent MCA system.

A single flat StateGraph with round-based bottom-up processing:
  init → leaf_respond → leaf_lateral → [observe/lateral per internal level]
  → observe_root → signal_down → check_convergence → [loop or finalize] → END
"""

import copy
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from src.strange_mca.agents import Agent
from src.strange_mca.convergence import compute_jaccard_similarity
from src.strange_mca.logging_utils import (
    DetailedLoggingCallbackHandler,
    setup_detailed_logging,
)
from src.strange_mca.prompts import (
    create_initial_response_prompt,
    create_lateral_prompt,
    create_observation_prompt,
    create_signal_prompt,
    create_strange_loop_prompt,
    parse_strange_loop_response,
)
from src.strange_mca.tree_helpers import (
    count_nodes_at_level,
    generate_all_nodes,
    is_leaf,
    make_node_name,
    parse_node_name,
)

logger = logging.getLogger("strange_mca")


# =============================================================================
# State Schema
# =============================================================================


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right overwriting left.

    WARNING: This is a shallow merge — keys in ``right`` completely replace the
    corresponding keys in ``left``.  Graph nodes that update ``agent_history``
    must therefore return **complete** agent history lists for every agent they
    touch, not partial updates, to avoid data loss.
    """
    if left is None:
        return right or {}
    if right is None:
        return left or {}
    return {**left, **right}


class AgentRoundData(TypedDict, total=False):
    response: str
    lateral_response: str
    revised: bool
    signal_sent: str
    signal_received: str


class MCAState(TypedDict, total=False):
    original_task: str
    agent_history: Annotated[dict, merge_dicts]
    current_round: int
    max_rounds: int
    convergence_threshold: float
    enable_downward_signals: bool
    converged: bool
    convergence_scores: list[float]
    final_response: str
    strange_loops: list[dict[str, str]]


# =============================================================================
# Strange Loop Helper
# =============================================================================


def _apply_strange_loop(
    agent: Agent,
    response: str,
    original_task: str,
    strange_loop_count: int,
    domain_instructions: str,
) -> tuple[str, list[dict[str, str]]]:
    """Apply strange loop processing to a response.

    Args:
        agent: The agent to use for strange loop iterations.
        response: The current response to refine.
        original_task: The original task for context.
        strange_loop_count: Number of strange loop iterations.
        domain_instructions: Domain-specific instructions for the last iteration.

    Returns:
        Tuple of (final_response, strange_loops_list).
    """
    local_count = strange_loop_count
    if domain_instructions:
        local_count += 1

    if local_count == 0:
        return response, []

    loops = []
    for i in range(local_count):
        if i == local_count - 1 and domain_instructions:
            loop_prompt = create_strange_loop_prompt(
                original_task, response, domain_instructions
            )
        else:
            loop_prompt = create_strange_loop_prompt(original_task, response)
        loop_response = agent.invoke(loop_prompt)
        loops.append({"prompt": loop_prompt, "response": loop_response})
        response = parse_strange_loop_response(loop_response)

    return response, loops


# =============================================================================
# Graph Construction
# =============================================================================


def create_execution_graph(
    agents: dict[str, Agent],
    cpp: int,
    depth: int,
    max_rounds: int = 3,
    convergence_threshold: float = 0.85,
    enable_downward_signals: bool = True,
    strange_loop_count: int = 0,
    domain_specific_instructions: str = "",
) -> Any:
    """Create a flat LangGraph StateGraph for MCA execution.

    Args:
        agents: Dict mapping node names to Agent instances.
        cpp: Children per parent.
        depth: Total tree depth.
        max_rounds: Maximum number of rounds.
        convergence_threshold: Jaccard similarity threshold for convergence.
        enable_downward_signals: Whether to enable parent-to-child signals.
        strange_loop_count: Number of strange loop iterations at finalization.
        domain_specific_instructions: Domain-specific instructions for strange loop.

    Returns:
        Compiled LangGraph.
    """
    if depth < 2:
        raise ValueError("depth must be >= 2 (need at least a root and leaf level)")

    # Precompute topology
    all_nodes = generate_all_nodes(cpp, depth)
    leaf_names = [n for n in all_nodes if is_leaf(parse_node_name(n)[0], depth)]
    root_name = "L1N1"

    # Compute internal levels (between root and leaves, exclusive)
    # Process bottom-up: highest level first so children are observed before parents
    internal_levels = list(range(depth - 1, 1, -1)) if depth > 2 else []

    builder = StateGraph(MCAState)

    # -------------------------------------------------------------------------
    # init node
    # -------------------------------------------------------------------------
    def init_node(state: MCAState) -> dict:
        history = {}
        for name in all_nodes:
            history[name] = []
        return {
            "agent_history": history,
            "current_round": 1,
            "max_rounds": max_rounds,
            "convergence_threshold": convergence_threshold,
            "enable_downward_signals": enable_downward_signals,
            "converged": False,
            "convergence_scores": [],
        }

    builder.add_node("init", init_node)

    # -------------------------------------------------------------------------
    # leaf_respond node
    # -------------------------------------------------------------------------
    def leaf_respond(state: MCAState) -> dict:
        round_num = state["current_round"]
        task = state["original_task"]
        updates = {}

        for name in leaf_names:
            agent = agents[name]
            history = list(state["agent_history"].get(name, []))
            previous = None
            parent_signal = None

            if round_num > 1 and history:
                last_round = history[-1]
                previous = last_round.get(
                    "lateral_response", last_round.get("response")
                )
                if state.get("enable_downward_signals"):
                    parent_signal = last_round.get("signal_received")

            prompt = create_initial_response_prompt(
                task,
                agent.config.perspective,
                round_num,
                previous_response=previous,
                parent_signal=parent_signal,
            )
            response = agent.invoke(prompt)
            round_data: AgentRoundData = {"response": response, "revised": False}
            updates[name] = history + [round_data]

        return {"agent_history": updates}

    builder.add_node("leaf_respond", leaf_respond)

    # -------------------------------------------------------------------------
    # leaf_lateral node
    # -------------------------------------------------------------------------
    def leaf_lateral(state: MCAState) -> dict:
        round_num = state["current_round"]
        task = state["original_task"]
        updates = {}

        for name in leaf_names:
            agent = agents[name]
            history = copy.deepcopy(state["agent_history"][name])
            own_response = history[-1]["response"]

            peer_responses = {}
            for sibling in agent.config.siblings:
                sib_history = state["agent_history"][sibling]
                peer_responses[sibling] = sib_history[-1]["response"]

            if not peer_responses:
                history[-1]["lateral_response"] = own_response
                history[-1]["revised"] = False
                updates[name] = history
                continue

            prompt = create_lateral_prompt(
                task, own_response, peer_responses, round_num
            )
            lateral_response = agent.invoke(prompt)

            revised = lateral_response.strip() != own_response.strip()
            history[-1]["lateral_response"] = lateral_response
            history[-1]["revised"] = revised
            updates[name] = history

        return {"agent_history": updates}

    builder.add_node("leaf_lateral", leaf_lateral)

    # -------------------------------------------------------------------------
    # Internal level observe/lateral nodes
    # -------------------------------------------------------------------------
    for int_level in internal_levels:
        level_node_count = count_nodes_at_level(int_level, cpp)

        def make_observe_fn(level):
            def observe_level(state: MCAState) -> dict:
                round_num = state["current_round"]
                task = state["original_task"]
                updates = {}
                node_count = count_nodes_at_level(level, cpp)

                for node_idx in range(1, node_count + 1):
                    name = make_node_name(level, node_idx)
                    agent = agents[name]
                    history = list(state["agent_history"].get(name, []))

                    # Gather children's latest responses
                    child_responses = {}
                    for child_name in agent.config.children:
                        child_hist = state["agent_history"][child_name]
                        if child_hist:
                            last = child_hist[-1]
                            child_responses[child_name] = last.get(
                                "lateral_response", last.get("response", "")
                            )

                    own_previous = None
                    if history:
                        last = history[-1]
                        own_previous = last.get(
                            "lateral_response", last.get("response")
                        )

                    prompt = create_observation_prompt(
                        task, child_responses, own_previous, round_num
                    )
                    response = agent.invoke(prompt)
                    round_data: AgentRoundData = {
                        "response": response,
                        "revised": False,
                    }
                    updates[name] = history + [round_data]

                return {"agent_history": updates}

            return observe_level

        def make_lateral_fn(level):
            def lateral_level(state: MCAState) -> dict:
                round_num = state["current_round"]
                task = state["original_task"]
                updates = {}
                node_count = count_nodes_at_level(level, cpp)

                for node_idx in range(1, node_count + 1):
                    name = make_node_name(level, node_idx)
                    agent = agents[name]
                    history = copy.deepcopy(state["agent_history"][name])
                    own_response = history[-1]["response"]

                    peer_responses = {}
                    for sibling in agent.config.siblings:
                        sib_history = state["agent_history"][sibling]
                        if sib_history:
                            peer_responses[sibling] = sib_history[-1]["response"]

                    if not peer_responses:
                        history[-1]["lateral_response"] = own_response
                        history[-1]["revised"] = False
                        updates[name] = history
                        continue

                    prompt = create_lateral_prompt(
                        task, own_response, peer_responses, round_num
                    )
                    lateral_response = agent.invoke(prompt)

                    revised = lateral_response.strip() != own_response.strip()
                    history[-1]["lateral_response"] = lateral_response
                    history[-1]["revised"] = revised
                    updates[name] = history

                return {"agent_history": updates}

            return lateral_level

        observe_node_name = f"observe_L{int_level}"
        lateral_node_name = f"lateral_L{int_level}"
        builder.add_node(observe_node_name, make_observe_fn(int_level))
        builder.add_node(lateral_node_name, make_lateral_fn(int_level))

    # -------------------------------------------------------------------------
    # observe_root node
    # -------------------------------------------------------------------------
    def observe_root(state: MCAState) -> dict:
        round_num = state["current_round"]
        task = state["original_task"]
        root_agent = agents[root_name]
        history = list(state["agent_history"].get(root_name, []))

        # Gather children's latest responses
        child_responses = {}
        for child_name in root_agent.config.children:
            child_hist = state["agent_history"][child_name]
            if child_hist:
                last = child_hist[-1]
                child_responses[child_name] = last.get(
                    "lateral_response", last.get("response", "")
                )

        own_previous = None
        if history:
            last = history[-1]
            own_previous = last.get("lateral_response", last.get("response"))

        prompt = create_observation_prompt(
            task, child_responses, own_previous, round_num
        )
        response = root_agent.invoke(prompt)
        round_data: AgentRoundData = {"response": response, "revised": False}

        return {"agent_history": {root_name: history + [round_data]}}

    builder.add_node("observe_root", observe_root)

    # -------------------------------------------------------------------------
    # signal_down node
    # -------------------------------------------------------------------------
    def signal_down(state: MCAState) -> dict:
        if not state.get("enable_downward_signals"):
            return {}

        task = state["original_task"]
        updates = {}

        # Gather all non-leaf agents that need to send signals
        non_leaf_names = [
            n for n in all_nodes if not is_leaf(parse_node_name(n)[0], depth)
        ]

        for name in non_leaf_names:
            agent = agents[name]
            # Use updates if this agent was already written (e.g., as a child
            # receiving signal_received), otherwise fall back to state.
            history = copy.deepcopy(updates.get(name, state["agent_history"][name]))
            if not history:
                continue

            # Get agent's latest synthesis
            last = history[-1]
            own_synthesis = last.get("lateral_response", last.get("response", ""))

            # Gather children's responses
            child_responses = {}
            for child_name in agent.config.children:
                child_hist = state["agent_history"][child_name]
                if child_hist:
                    c_last = child_hist[-1]
                    child_responses[child_name] = c_last.get(
                        "lateral_response", c_last.get("response", "")
                    )

            prompt = create_signal_prompt(task, child_responses, own_synthesis)
            signal = agent.invoke(prompt)

            # Store signal_sent on the parent
            history[-1]["signal_sent"] = signal
            updates[name] = history

            # Store signal_received on each child — use updates if the child
            # was already written, to avoid overwriting prior data.
            for child_name in agent.config.children:
                child_hist = copy.deepcopy(
                    updates.get(child_name, state["agent_history"].get(child_name, []))
                )
                if child_hist:
                    child_hist[-1]["signal_received"] = signal
                    updates[child_name] = child_hist

        return {"agent_history": updates} if updates else {}

    builder.add_node("signal_down", signal_down)

    # -------------------------------------------------------------------------
    # check_convergence node
    # -------------------------------------------------------------------------
    def check_convergence(state: MCAState) -> dict:
        round_num = state["current_round"]
        threshold = state["convergence_threshold"]
        max_r = state["max_rounds"]

        root_history = state["agent_history"].get(root_name, [])
        scores = list(state.get("convergence_scores", []))

        converged = False

        if len(root_history) >= 2:
            prev = root_history[-2].get(
                "lateral_response", root_history[-2].get("response", "")
            )
            curr = root_history[-1].get(
                "lateral_response", root_history[-1].get("response", "")
            )
            score = compute_jaccard_similarity(prev, curr)
            scores.append(score)
            converged = score >= threshold

        if round_num >= max_r:
            converged = True

        return {
            "converged": converged,
            "convergence_scores": scores,
            "current_round": round_num + 1,
        }

    builder.add_node("check_convergence", check_convergence)

    # -------------------------------------------------------------------------
    # finalize node
    # -------------------------------------------------------------------------
    def finalize(state: MCAState) -> dict:
        root_history = state["agent_history"].get(root_name, [])
        root_response = ""
        if root_history:
            last = root_history[-1]
            root_response = last.get("lateral_response", last.get("response", ""))

        if strange_loop_count > 0 or domain_specific_instructions:
            root_agent = agents[root_name]
            final_response, loops = _apply_strange_loop(
                root_agent,
                root_response,
                state["original_task"],
                strange_loop_count,
                domain_specific_instructions,
            )
        else:
            final_response = root_response
            loops = []

        result: dict = {"final_response": final_response}
        if loops:
            result["strange_loops"] = loops
        return result

    builder.add_node("finalize", finalize)

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------
    builder.set_entry_point("init")
    builder.add_edge("init", "leaf_respond")
    builder.add_edge("leaf_respond", "leaf_lateral")

    if internal_levels:
        # leaf_lateral → first internal level observe
        builder.add_edge("leaf_lateral", f"observe_L{internal_levels[0]}")

        # Chain internal levels
        for i, level in enumerate(internal_levels):
            builder.add_edge(f"observe_L{level}", f"lateral_L{level}")
            if i + 1 < len(internal_levels):
                builder.add_edge(
                    f"lateral_L{level}", f"observe_L{internal_levels[i + 1]}"
                )
            else:
                builder.add_edge(f"lateral_L{level}", "observe_root")
    else:
        builder.add_edge("leaf_lateral", "observe_root")

    builder.add_edge("observe_root", "signal_down")
    builder.add_edge("signal_down", "check_convergence")

    builder.add_conditional_edges(
        "check_convergence",
        lambda state: "finalize" if state.get("converged", False) else "leaf_respond",
        {"leaf_respond": "leaf_respond", "finalize": "finalize"},
    )

    builder.add_edge("finalize", END)

    # Compute recursion limit
    # init+leaf_respond+leaf_lateral, internal pairs, root+signal+convergence, finalize
    num_graph_nodes = 3 + len(internal_levels) * 2 + 3 + 1
    recursion_limit = max(50, max_rounds * (num_graph_nodes + 5))

    return builder.compile(), recursion_limit


# =============================================================================
# Public API
# =============================================================================


def run_execution_graph(
    graph,
    task: str,
    recursion_limit: int = 50,
    log_level: str = "warn",
    only_local_logs: bool = False,
) -> dict:
    """Run the execution graph on a task.

    Args:
        graph: The compiled graph to run.
        task: The task to perform.
        recursion_limit: Recursion limit for LangGraph.
        log_level: Logging level ("warn", "info", or "debug").
        only_local_logs: If True, only show logs from strange_mca logger.

    Returns:
        Result dictionary containing response and metadata.
    """
    setup_detailed_logging(log_level=log_level, only_local_logs=only_local_logs)
    callback_handler = DetailedLoggingCallbackHandler()

    logger.info(f"Running execution graph with task: {task[:100]}...")

    initial_state: MCAState = {
        "original_task": task,
    }

    config = RunnableConfig(
        callbacks=[callback_handler], recursion_limit=recursion_limit
    )

    result = graph.invoke(initial_state, config=config)

    if "final_response" not in result:
        result["final_response"] = ""

    return result
