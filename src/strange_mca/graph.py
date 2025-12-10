"""LangGraph implementation of the multiagent system using nested subgraphs.

Each agent is represented as a self-contained subgraph with:
- A 'down' node for task decomposition (non-leaf) or execution (leaf)
- Child subgraphs for each child agent (non-leaf only)
- An 'up' node for synthesizing children's responses (non-leaf only)

The graph structure mirrors the conceptual agent tree - no separate NetworkX graph needed.
"""

import logging
import operator
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from src.strange_mca.agents import Agent, AgentConfig
from src.strange_mca.logging_utils import (
    DetailedLoggingCallbackHandler,
    setup_detailed_logging,
)
from src.strange_mca.prompts import (
    create_strange_loop_prompt,
    create_synthesis_prompt,
    create_task_decomposition_prompt,
    parse_strange_loop_response,
)

# Set up logging
logger = logging.getLogger("strange_mca")


# =============================================================================
# Tree Helper Functions
# =============================================================================


def parse_node_name(node_name: str) -> tuple[int, int]:
    """Parse 'L{level}N{num}' into (level, num).

    Args:
        node_name: Node name like 'L1N1', 'L2N3', etc.

    Returns:
        Tuple of (level, node_number).

    Example:
        >>> parse_node_name('L2N3')
        (2, 3)
    """
    parts = node_name.split("N")
    level = int(parts[0][1:])
    num = int(parts[1])
    return level, num


def make_node_name(level: int, num: int) -> str:
    """Create node name from level and node number.

    Args:
        level: Tree level (1 = root).
        num: Node number within level (1-indexed).

    Returns:
        Node name like 'L1N1'.
    """
    return f"L{level}N{num}"


def get_children(node_name: str, cpp: int, depth: int) -> list[str]:
    """Get child node names for a given node.

    Args:
        node_name: Parent node name.
        cpp: Children per parent.
        depth: Total tree depth.

    Returns:
        List of child node names, empty if leaf.

    Example:
        >>> get_children('L1N1', 2, 3)
        ['L2N1', 'L2N2']
        >>> get_children('L2N2', 2, 3)
        ['L3N3', 'L3N4']
    """
    level, num = parse_node_name(node_name)
    if level >= depth:
        return []
    child_level = level + 1
    start = (num - 1) * cpp + 1
    return [make_node_name(child_level, start + i) for i in range(cpp)]


def is_leaf(level: int, depth: int) -> bool:
    """Check if a node at given level is a leaf."""
    return level == depth


def is_root(level: int) -> bool:
    """Check if a node at given level is root."""
    return level == 1


def count_nodes_at_level(level: int, cpp: int) -> int:
    """Count nodes at a given level."""
    return cpp ** (level - 1)


def total_nodes(cpp: int, depth: int) -> int:
    """Calculate total nodes in tree."""
    return sum(count_nodes_at_level(level, cpp) for level in range(1, depth + 1))


# =============================================================================
# State Schema
# =============================================================================


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right overwriting left."""
    if left is None:
        return right or {}
    if right is None:
        return left or {}
    return {**left, **right}


class State(TypedDict, total=False):
    """State for an agent subgraph.

    Note: depth and cpp are build-time constants captured in closures,
    not runtime state that needs to be passed through.
    """

    # The task for this agent (passed from parent or initial input)
    task: str

    # Original task (for context, passed to all descendants)
    original_task: str

    # This agent's response (set by down node for leaves, up node for non-leaves)
    response: str

    # Decomposition output from down pass (non-leaf nodes only)
    decomposition: str

    # Tasks assigned to each child (maps child name -> task string)
    child_tasks: Annotated[dict[str, str], merge_dicts]

    # Child responses stored as a dict with merge reducer
    child_responses: Annotated[dict[str, str], merge_dicts]

    # For root node only (set after strange loop)
    final_response: str
    strange_loops: list[dict[str, str]]

    # Legacy compatibility: nodes dict for result format
    nodes: dict[str, dict[str, str]]
    current_node: str


# =============================================================================
# Subgraph Construction
# =============================================================================


def _parse_subtask_for_child(decomposition_response: str, child_name: str) -> str:
    """Extract subtask for a specific child from decomposition response.

    Args:
        decomposition_response: The parent's decomposition containing subtasks.
        child_name: The child node name to extract subtask for.

    Returns:
        The subtask string for this child.
    """
    prefix = f"{child_name}: "
    for line in decomposition_response.split("\n"):
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    # Fallback: return empty string if not found
    logger.warning(f"No subtask found for {child_name} in decomposition response")
    return ""


def create_agent_subgraph(
    node_name: str,
    cpp: int,
    depth: int,
    model_name: str,
    domain_instructions: str = "",
    strange_loop_count: int = 0,
):
    """Create a subgraph for a single agent node.

    Args:
        node_name: The agent's name (e.g., 'L1N1').
        cpp: Children per parent.
        depth: Total tree depth.
        model_name: LLM model name to use.
        domain_instructions: Domain-specific instructions for strange loop.
        strange_loop_count: Number of strange loop iterations at root.

    Returns:
        Compiled LangGraph for this agent.
    """
    level, num = parse_node_name(node_name)
    is_leaf_node = is_leaf(level, depth)
    is_root_node = is_root(level)

    # Create agent for this node
    config = AgentConfig(name=node_name, level=level, node_number=num)
    agent = Agent(config, model_name=model_name)

    builder = StateGraph(State)

    # Get children for this node (empty list if leaf)
    children = get_children(node_name, cpp, depth)

    # -------------------------------------------------------------------------
    # DOWN node: decompose (non-leaf) or execute (leaf)
    # -------------------------------------------------------------------------
    def down_node(state: State) -> dict[str, Any]:
        task = state["task"]
        logger.debug(f"[{node_name}] Down pass with task: {task[:100]}...")

        if is_leaf_node:
            # Leaf: execute the task directly
            response = agent.run(task=task)
            logger.debug(f"[{node_name}] Leaf response: {response[:100]}...")
            return {"response": response}
        else:
            # Non-leaf: decompose into subtasks for children
            child_nodes_str = ", ".join(children)
            context = f"You are coordinating a task across {len(children)} agents: {child_nodes_str}."
            decomposition_prompt = create_task_decomposition_prompt(
                task, context, children
            )
            response = agent.run(task=decomposition_prompt)
            logger.debug(f"[{node_name}] Decomposition response: {response[:100]}...")
            # Store both response (for parsing) and decomposition (for output trace)
            return {"response": response, "decomposition": response}

    builder.add_node("down", down_node)

    if is_leaf_node:
        # Leaf: down -> END (response already set)
        builder.add_edge("down", END)
    else:
        # Non-leaf: down -> children (sequential) -> up

        # Create child subgraphs and add as nodes
        for child_name in children:
            child_subgraph = create_agent_subgraph(
                child_name,
                cpp,
                depth,
                model_name,
                domain_instructions,
                strange_loop_count=0,  # Only root has strange loops
            )

            # Create wrapper function to invoke child with transformed state
            def make_invoke_child(child: str, sg):
                def invoke_child(state: State) -> dict[str, Any]:
                    # Parse subtask for this child from parent's decomposition
                    subtask = _parse_subtask_for_child(state["response"], child)

                    # Include original task context
                    original_task = state.get("original_task", state["task"])
                    child_task = f"Original task context:\n{original_task}\n\nYour specific assignment:\n{subtask}"

                    # Invoke child subgraph (depth/cpp are captured at build time, not passed through state)
                    child_result = sg.invoke(
                        {
                            "task": child_task,
                            "original_task": original_task,
                        }
                    )

                    # Store child's task and response (both use merge reducer)
                    return {
                        "child_tasks": {child: child_task},
                        "child_responses": {child: child_result["response"]},
                    }

                return invoke_child

            builder.add_node(
                f"child_{child_name}", make_invoke_child(child_name, child_subgraph)
            )

        # Wire: down -> first child
        builder.add_edge("down", f"child_{children[0]}")

        # Wire: children sequentially
        for i in range(len(children) - 1):
            builder.add_edge(f"child_{children[i]}", f"child_{children[i + 1]}")

        # -------------------------------------------------------------------------
        # UP node: synthesize children's responses
        # -------------------------------------------------------------------------
        def up_node(state: State) -> dict[str, Any]:
            # Gather children's responses from child_responses dict
            stored_responses = state.get("child_responses", {})
            child_responses = {}
            for child in children:
                child_responses[child] = stored_responses.get(child, "")
                if not child_responses[child]:
                    logger.warning(f"[{node_name}] No response from child {child}")

            # Synthesize
            synthesis_prompt = create_synthesis_prompt(child_responses)
            response = agent.run(task=synthesis_prompt)
            logger.debug(f"[{node_name}] Synthesis response: {response[:100]}...")

            result: dict[str, Any] = {"response": response}

            # Strange loop at root
            if is_root_node:
                local_strange_loop_count = strange_loop_count
                if domain_instructions:
                    local_strange_loop_count += 1

                if local_strange_loop_count > 0:
                    loops = []
                    for i in range(local_strange_loop_count):
                        # Apply domain instructions on last iteration
                        if i == local_strange_loop_count - 1:
                            loop_prompt = create_strange_loop_prompt(
                                state.get("original_task", state["task"]),
                                response,
                                domain_instructions,
                            )
                        else:
                            loop_prompt = create_strange_loop_prompt(
                                state.get("original_task", state["task"]),
                                response,
                            )
                        loop_response = agent.run(task=loop_prompt)
                        loops.append({"prompt": loop_prompt, "response": loop_response})
                        response = parse_strange_loop_response(loop_response)

                    result["strange_loops"] = loops

                result["final_response"] = response

            return result

        builder.add_node("up", up_node)

        # Wire: last child -> up -> END
        builder.add_edge(f"child_{children[-1]}", "up")
        builder.add_edge("up", END)

    builder.set_entry_point("down")
    return builder.compile()


# =============================================================================
# Public API
# =============================================================================


def create_execution_graph(
    child_per_parent: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
    langgraph_viz_dir: Optional[str] = None,
    domain_specific_instructions: Optional[str] = "",
    strange_loop_count: int = 0,
):
    """Create a LangGraph for the multiagent system.

    This creates a nested subgraph structure where each agent is represented
    as a self-contained subgraph.

    Args:
        child_per_parent: Number of children each non-leaf node has.
        depth: Number of levels in the tree.
        model_name: LLM model name to use.
        langgraph_viz_dir: If provided, generate visualization in this directory.
        domain_specific_instructions: Domain-specific instructions for strange loop.
        strange_loop_count: Number of strange loop iterations.

    Returns:
        Compiled LangGraph.
    """
    logger.info(
        f"Creating execution graph with {child_per_parent} children per parent and {depth} levels"
    )

    # Create the root subgraph (which recursively creates all descendants)
    compiled_graph = create_agent_subgraph(
        "L1N1",  # root
        child_per_parent,
        depth,
        model_name,
        domain_specific_instructions or "",
        strange_loop_count,
    )

    logger.info("Compiled LangGraph successfully")

    # Generate visualization if requested
    if langgraph_viz_dir:
        from src.strange_mca.visualization import visualize_langgraph

        visualize_langgraph(compiled_graph, langgraph_viz_dir, child_per_parent, depth)

    return compiled_graph


def run_execution_graph(
    execution_graph,
    task: str,
    log_level: str = "warn",
    only_local_logs: bool = False,
    langgraph_viz_dir: Optional[str] = None,
) -> dict:
    """Run the execution graph on a task.

    Args:
        execution_graph: The compiled graph to run.
        task: The task to perform.
        log_level: Logging level ("warn", "info", or "debug").
        only_local_logs: If True, only show logs from strange_mca logger.
        langgraph_viz_dir: Directory where visualization was generated (unused).

    Returns:
        Result dictionary containing response and metadata.
    """
    # Set up logging
    setup_detailed_logging(log_level=log_level, only_local_logs=only_local_logs)

    # Create callback handler for detailed logging
    callback_handler = DetailedLoggingCallbackHandler()

    logger.info(f"Running execution graph with task: {task[:100]}...")

    # Initialize state
    initial_state: State = {
        "task": task,
        "original_task": task,
    }

    try:
        # Calculate recursion limit based on expected depth
        # Each subgraph has multiple nodes, so we need generous limits
        recursion_limit = 100

        config = RunnableConfig(
            callbacks=[callback_handler], recursion_limit=recursion_limit
        )

        result = execution_graph.invoke(initial_state, config=config)

        # Ensure backward-compatible result format
        if "final_response" not in result:
            result["final_response"] = result.get("response", "")

        return result

    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise
