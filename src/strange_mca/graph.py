"""LangGraph implementation of the multiagent system."""

from typing import Dict, List, TypedDict, Annotated, Literal, Any, cast, Optional
import logging
import json
import os
import networkx as nx

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from src.strange_mca.agents import Agent, AgentConfig, create_agent_configs, create_agent_tree, AgentTree
from src.strange_mca.logging_utils import DetailedLoggingCallbackHandler, setup_detailed_logging

# Set up logging
logger = logging.getLogger("strange_mca")

# Define a custom reducer for the responses dictionary
# def responses_reducer(current_dict: Dict[str, str], update: Dict[str, str]) -> Dict[str, str]:
#     """Update a dictionary with new key-value pairs."""
#     result = current_dict.copy()
#     result.update(update)
#     return result


class State(TypedDict):
    """State of the multiagent graph."""
    
    # The original task being processed
    original_task: str
    
    # The original context
    original_context: str
    
    # Dictionary mapping node names to their data
    # Each node entry contains 'task', 'context', and 'response'
    nodes: Dict[str, Dict[str, str]]
    
    # The current node being executed (using LangGraph node name, e.g., "L2N2_down")
    current_node: str
    
    # The final response
    final_response: str


#todo: move this to prompts.py
def create_task_decomposition_prompt(task: str, system_prompt: str, child_nodes: List[str]) -> str:
    """Create a prompt for task decomposition.
    
    Args:
        task: The original task to decompose.
        system_prompt: The system prompt of the agent that will decompose the task.
        child_nodes: List of child node names that will receive the subtasks.
        
    Returns:
        A prompt for task decomposition.
    """
    child_nodes_str = "\n".join([f"- {node}" for node in child_nodes])
    
    return f"""Based on your system prompt:

{system_prompt}

Your task is to break down the following task into subtasks for your team members:

{task}

You have the following team members that need assignments:
{child_nodes_str}

For each team member, create a specific subtask that:
1. Is clearly described and actionable
2. Includes any specific instructions or constraints
3. Contributes meaningfully to the overall task

Format your response with one subtask per team member, using their exact name as the prefix:

{child_nodes[0]}: [Subtask description for this team member]
{child_nodes[1] if len(child_nodes) > 1 else "[Next team member]"}: [Subtask description for this team member]
...and so on for all team members.

Make sure each team member has exactly one subtask assigned to them."""


def create_synthesis_prompt(child_responses: Dict[str, str]) -> str:
    """Create a prompt for synthesizing responses from child nodes.
    
    Args:
        child_responses: Dictionary mapping child node names to their responses.
        
    Returns:
        A prompt for synthesizing responses.
    """
    formatted_responses = "\n\n".join([
        f"Agent {child}: {response}"
        for child, response in child_responses.items()
    ])
    
    return f"""Synthesize the following responses from your team members into a coherent answer:

{formatted_responses}

Your task is to:
1. Integrate the key insights from each response
2. Resolve any contradictions or inconsistencies
3. Provide a comprehensive and coherent final answer

Format your response as a well-structured summary."""



def create_bidirectional_graph(
    child_per_parent: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
    langgraph_viz_dir: Optional[str] = None,
):
    """Create a LangGraph for bidirectional traversal of the multiagent system.
    
    This function creates a graph with clear separation between downward and upward passes.
    The downward pass decomposes tasks from parent to child nodes, and the upward pass
    synthesizes responses from child to parent nodes.
    
    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        langgraph_viz_dir: If provided, generate a visualization of the LangGraph structure
                          in this directory. If None, no visualization is generated.
        
    Returns:
        The compiled graph.
    """
    logger.info(f"Creating bidirectional graph with {child_per_parent} children per parent and {depth} levels")
    
    # Create agent tree using NetworkX
    agent_tree = create_agent_tree(child_per_parent, depth)
    
    # Create a dictionary to store agents for each node
    agents = {}
    for agent_name in agent_tree.graph.nodes():
        config = agent_tree.get_config(agent_name)
        agents[agent_name] = Agent(config, model_name=model_name)
        logger.debug(f"Created agent for {agent_name}")
    
    # Initialize the graph builder
    graph_builder = StateGraph(State)
    
    # Define the downward function (task decomposition)
    def down_function(state: State, lg_node_name: str) -> dict:
        """Process a node in the downward pass (task decomposition)."""
        # Extract the AgentTree node name from the LangGraph node name
        agent_name = lg_node_name.split("_down")[0]
        agent = agents[agent_name]
        
        # Update the current node in the state to use the LangGraph node name
        state_updates = {"current_node": lg_node_name}
        
        # Initialize nodes dictionary if it doesn't exist
        nodes = state.get("nodes", {}).copy()
        if lg_node_name not in nodes:
            nodes[lg_node_name] = {}
        
        # Get the task and context for this node
        original_task = state["original_task"]
        
        # Check if this node has a task, otherwise use parent's task or original task
        if "task" not in nodes.get(lg_node_name, {}):
            # If this is not the root, get the parent's task
            if not agent_tree.is_root(agent_name):
                agent_parent = agent_tree.get_parent(agent_name)
                lg_parent_down = f"{agent_parent}_down"
                if lg_parent_down in nodes and "task" in nodes[lg_parent_down]:
                    # Use the parent's task as a fallback
                    nodes[lg_node_name]["task"] = nodes[lg_parent_down]["task"]
                else:
                    # Use the original task as a fallback
                    nodes[lg_node_name]["task"] = original_task
            else:
                # Root node uses the original task
                nodes[lg_node_name]["task"] = original_task
        
        # Get the task and context for this node
        task = nodes[lg_node_name].get("task", original_task)
        context = nodes[lg_node_name].get("context", state["original_context"])
        
        # Check if this is a leaf node
        if agent_tree.is_leaf(agent_name):
            logger.debug(f"[{lg_node_name}] Processing leaf node task (downward pass)")
            # Leaf nodes directly process their task
            response = agent.run(context=context, task=task)
            
            # Update node response
            nodes[lg_node_name]["response"] = response
            state_updates["nodes"] = nodes
            
            # Return the updated state with next node being the upward pass of this node
            return {**state_updates, "next": f"{agent_name}_up"}
        else:
            logger.debug(f"[{lg_node_name}] Decomposing task for children (downward pass)")
            # Non-leaf nodes decompose the task for their children
            agent_children = agent_tree.get_children(agent_name)
            
            # Create a prompt for task decomposition
            child_nodes_str = ", ".join(agent_children)
            system_prompt = f"You are coordinating a task across {len(agent_children)} agents: {child_nodes_str}."
            decomposition_prompt = create_task_decomposition_prompt(task, system_prompt, agent_children)
            
            # Run the agent to decompose the task
            response = agent.run(context=context, task=decomposition_prompt)
            
            # Update node response
            nodes[lg_node_name]["response"] = response
            
            # Parse the response to extract tasks for children
            for agent_child in agent_children:
                lg_child_down = f"{agent_child}_down"
                if lg_child_down not in nodes:
                    nodes[lg_child_down] = {}
                
                # Look for the child's task in the response
                child_task_prefix = f"{agent_child}: "
                for line in response.split("\n"):
                    if line.startswith(child_task_prefix):
                        child_task = line[len(child_task_prefix):].strip()
                        nodes[lg_child_down]["task"] = child_task
                        break
            
            state_updates["nodes"] = nodes
            
            # Return the updated state with next node being the first child's downward pass
            return {**state_updates, "next": f"{agent_children[0]}_down"}
    
    # Define the upward function (response synthesis)
    def up_function(state: State, lg_node_name: str) -> dict:
        """Process a node in the upward pass (response synthesis)."""
        # Extract the AgentTree node name from the LangGraph node name
        agent_name = lg_node_name.split("_up")[0]
        agent = agents[agent_name]
        
        # Update the current node in the state to use the LangGraph node name
        state_updates = {"current_node": lg_node_name}
        
        # Initialize nodes dictionary if it doesn't exist
        nodes = state.get("nodes", {}).copy()
        if lg_node_name not in nodes:
            nodes[lg_node_name] = {}
        
        # Check if this is the root node
        if agent_tree.is_root(agent_name):
            logger.debug(f"[{lg_node_name}] Processing root node synthesis (upward pass)")
            
            # Get all child responses from their upward nodes
            agent_children = agent_tree.get_children(agent_name)
            child_responses = {}
            
            for agent_child in agent_children:
                # Use only the child's upward node response
                lg_child_up = f"{agent_child}_up"
                if lg_child_up in nodes and "response" in nodes[lg_child_up]:
                    child_responses[agent_child] = nodes[lg_child_up]["response"]
                else:
                    # If no response is available, use an empty string
                    child_responses[agent_child] = ""
                    logger.warning(f"No response found for {lg_child_up}")
            
            # Create synthesis prompt
            synthesis_prompt = create_synthesis_prompt(child_responses)
            
            # Run the agent with the synthesis prompt
            final_response = agent.run(context=state["original_context"], task=synthesis_prompt)
            
            # Update the node response and final response
            nodes[lg_node_name]["response"] = final_response
            state_updates["nodes"] = nodes
            state_updates["final_response"] = final_response
            
            # Root node completes the graph
            return {**state_updates, "next": END}
        elif agent_tree.is_leaf(agent_name):
            logger.debug(f"[{lg_node_name}] Processing leaf node (upward pass)")
            
            # Leaf nodes have already processed their task in the downward pass
            # Copy the response from the downward pass
            lg_down_node = f"{agent_name}_down"
            if lg_down_node in nodes and "response" in nodes[lg_down_node]:
                nodes[lg_node_name]["response"] = nodes[lg_down_node]["response"]
                state_updates["nodes"] = nodes
            else:
                logger.warning(f"No response found for {lg_down_node}")
            
            # Get the parent node
            agent_parent = agent_tree.get_parent(agent_name)
            
            # Get all siblings (including self)
            agent_siblings = agent_tree.get_children(agent_parent)
            
            # Find the next sibling
            try:
                idx = agent_siblings.index(agent_name)
                
                if idx < len(agent_siblings) - 1:
                    # If not the last sibling, go to next sibling's downward pass
                    agent_next_sibling = agent_siblings[idx + 1]
                    return {**state_updates, "next": f"{agent_next_sibling}_down"}
                else:
                    # If last sibling, go to parent's upward pass
                    return {**state_updates, "next": f"{agent_parent}_up"}
            except ValueError:
                # Fallback if node not found in siblings
                return {**state_updates, "next": f"{agent_parent}_up"}
        else:
            logger.debug(f"[{lg_node_name}] Processing non-leaf node synthesis (upward pass)")
            
            # Non-leaf, non-root nodes synthesize responses from their children's upward nodes
            agent_children = agent_tree.get_children(agent_name)
            child_responses = {}
            
            for agent_child in agent_children:
                # Use only the child's upward node response
                lg_child_up = f"{agent_child}_up"
                if lg_child_up in nodes and "response" in nodes[lg_child_up]:
                    child_responses[agent_child] = nodes[lg_child_up]["response"]
                else:
                    # If no response is available, use an empty string
                    child_responses[agent_child] = ""
                    logger.warning(f"No response found for {lg_child_up}")
            
            # Create synthesis prompt
            synthesis_prompt = create_synthesis_prompt(child_responses)
            
            # Run the agent with the synthesis prompt
            response = agent.run(context=state["original_context"], task=synthesis_prompt)
            
            # Update the node response
            nodes[lg_node_name]["response"] = response
            state_updates["nodes"] = nodes
            
            # Get the parent node
            agent_parent = agent_tree.get_parent(agent_name)
            
            # Get all siblings (including self)
            agent_siblings = agent_tree.get_children(agent_parent)
            
            # Find the next sibling
            try:
                idx = agent_siblings.index(agent_name)
                
                if idx < len(agent_siblings) - 1:
                    # If not the last sibling, go to next sibling's downward pass
                    agent_next_sibling = agent_siblings[idx + 1]
                    return {**state_updates, "next": f"{agent_next_sibling}_down"}
                else:
                    # If last sibling, go to parent's upward pass
                    return {**state_updates, "next": f"{agent_parent}_up"}
            except ValueError:
                # Fallback if node not found in siblings
                return {**state_updates, "next": f"{agent_parent}_up"}
    
    # Use callbacks to build the LangGraph nodes and edges
    def add_langgraph_node_callback(agent_name: str, tree: AgentTree) -> None:
        """Callback to add nodes to the LangGraph during traversal."""
        # Add downward node
        lg_down_node = f"{agent_name}_down"
        graph_builder.add_node(lg_down_node, lambda s: down_function(s, lg_down_node))
        logger.debug(f"Added downward node {lg_down_node} to LangGraph")
        
        # Add upward node
        lg_up_node = f"{agent_name}_up"
        graph_builder.add_node(lg_up_node, lambda s: up_function(s, lg_up_node))
        logger.debug(f"Added upward node {lg_up_node} to LangGraph")
    
    # Traverse the tree to add all nodes to the LangGraph
    agent_tree.perform_down_traversal(node_callback=add_langgraph_node_callback)
    
    # Add edges for downward traversal
    def add_downward_edges_callback(agent_name: str, tree: AgentTree) -> None:
        """Callback to add downward edges to the LangGraph during traversal."""
        if tree.is_leaf(agent_name):
            # Leaf nodes don't have children, so no downward edges needed
            return
        
        agent_children = tree.get_children(agent_name)
        if agent_children:
            # Connect this node's downward to its first child's downward
            agent_first_child = agent_children[0]
            lg_node_down = f"{agent_name}_down"
            lg_first_child_down = f"{agent_first_child}_down"
            graph_builder.add_edge(lg_node_down, lg_first_child_down)
            logger.debug(f"Added downward edge: {lg_node_down} -> {lg_first_child_down}")
    
    # Add edges for upward traversal and sibling transitions
    def add_upward_edges_callback(agent_name: str, tree: AgentTree) -> None:
        """Callback to add upward edges and sibling transitions to the LangGraph."""
        if tree.is_root(agent_name):
            # Root node's upward pass ends the graph
            lg_node_up = f"{agent_name}_up"
            graph_builder.add_edge(lg_node_up, END)
            logger.debug(f"Added edge: {lg_node_up} -> END")
            return
        
        agent_parent = tree.get_parent(agent_name)
        agent_siblings = tree.get_children(agent_parent)
        
        # Find this node's position among siblings
        try:
            idx = agent_siblings.index(agent_name)
            
            if idx < len(agent_siblings) - 1:
                # If not the last sibling, connect to next sibling's downward
                agent_next_sibling = agent_siblings[idx + 1]
                lg_node_up = f"{agent_name}_up"
                lg_next_sibling_down = f"{agent_next_sibling}_down"
                graph_builder.add_edge(lg_node_up, lg_next_sibling_down)
                logger.debug(f"Added sibling transition: {lg_node_up} -> {lg_next_sibling_down}")
            else:
                # If last sibling, connect to parent's upward
                lg_node_up = f"{agent_name}_up"
                lg_parent_up = f"{agent_parent}_up"
                graph_builder.add_edge(lg_node_up, lg_parent_up)
                logger.debug(f"Added upward edge: {lg_node_up} -> {lg_parent_up}")
        except ValueError:
            # Fallback if node not found in siblings
            lg_node_up = f"{agent_name}_up"
            lg_parent_up = f"{agent_parent}_up"
            graph_builder.add_edge(lg_node_up, lg_parent_up)
            logger.debug(f"Added fallback upward edge: {lg_node_up} -> {lg_parent_up}")
    
    # Add leaf node transitions from downward to upward
    def add_leaf_transitions_callback(agent_name: str, tree: AgentTree) -> None:
        """Callback to add transitions from downward to upward for leaf nodes."""
        if tree.is_leaf(agent_name):
            lg_node_down = f"{agent_name}_down"
            lg_node_up = f"{agent_name}_up"
            graph_builder.add_edge(lg_node_down, lg_node_up)
            logger.debug(f"Added leaf transition: {lg_node_down} -> {lg_node_up}")
    
    # Add all edges to the LangGraph
    agent_tree.perform_down_traversal(node_callback=add_downward_edges_callback)
    agent_tree.perform_down_traversal(node_callback=add_leaf_transitions_callback)
    agent_tree.perform_up_traversal(node_callback=add_upward_edges_callback)
    
    # Set the entry point to the root node's downward pass
    root_node = agent_tree.get_root()
    graph_builder.set_entry_point(f"{root_node}_down")
    logger.info(f"Set entry point to {root_node}_down")
    
    # Compile the graph
    compiled_graph = graph_builder.compile()
    logger.info("Compiled LangGraph successfully")
    
    # Generate visualization if requested
    if langgraph_viz_dir:
        # Create directory if it doesn't exist
        os.makedirs(langgraph_viz_dir, exist_ok=True)
        
        # Create langgraph-specific output path
        lg_viz_path = os.path.join(langgraph_viz_dir, "langgraph_structure")
        
        try:
            png_data = compiled_graph.get_graph().draw_mermaid_png()
            with open(f"{lg_viz_path}.png", "wb") as f:
                f.write(png_data)
            logger.info(f"LangGraph visualization saved to {lg_viz_path}.png")
        except Exception as e:
            logger.warn(f"Could not generate graph visualization: {e}")
    
    return compiled_graph


def run_bidirectional_graph(
    graph,
    task: str,
    context: str = "",
    log_level: str = "warn",
    only_local_logs: bool = False,
    langgraph_viz_dir: Optional[str] = None,
) -> dict:
    """Run the bidirectional graph on a task.
    
    Args:
        graph: The compiled graph to run.
        task: The task to perform.
        context: The context for the task.
        log_level: The level of logging detail using standard Python logging levels: "warn", "info", or "debug".
                  Default is "warn" which shows only warnings and errors.
        only_local_logs: If True, only show logs from the strange_mca logger and suppress logs from other loggers.
        langgraph_viz_dir: If provided, directory where LangGraph visualization was generated.
        
    Returns:
        The result dictionary containing all responses and the final response.
    """
    # Set up logging
    setup_detailed_logging(log_level=log_level, only_local_logs=only_local_logs)
    
    # Create a callback handler for detailed logging
    callback_handler = DetailedLoggingCallbackHandler()
    
    # Find the root node of the graph
    at_root_node = None
    for lg_node in graph.get_graph().nodes:
        if lg_node.endswith("_down") and "_" not in lg_node.split("_down")[0]:
            at_root_node = lg_node.split("_down")[0]
            break
    
    if not at_root_node:
        raise ValueError("Could not find root node in the graph")
    
    logger.info(f"Identified root node: {at_root_node}")
    
    # Create the initial state
    lg_root_down = f"{at_root_node}_down"
    initial_state = {
        "original_task": task,
        "original_context": context,
        "nodes": {
            lg_root_down: {
                "task": task,
                "context": context
            }
        },
        "current_node": lg_root_down,
        "node_responses": {},  # Keep for backward compatibility
        "node_tasks": {},      # Keep for backward compatibility
        "node_contexts": {}    # Keep for backward compatibility
    }
    
    logger.info("Running bidirectional graph...")
    
    try:
        # Run the graph with the initial state
        config = RunnableConfig(callbacks=[callback_handler])
        result = graph.invoke(initial_state, config=config)
        
        # For backward compatibility, extract responses from nodes dictionary
        # but don't include these in the final result
        if "nodes" in result:
            # Create backward compatibility dictionaries for internal use
            node_responses = {}
            node_tasks = {}
            node_contexts = {}
            
            for lg_node_name, node_data in result["nodes"].items():
                # Extract the AgentTree node name from the LangGraph node name
                if "_down" in lg_node_name:
                    agent_name = lg_node_name.split("_down")[0]
                elif "_up" in lg_node_name:
                    agent_name = lg_node_name.split("_up")[0]
                else:
                    agent_name = lg_node_name
                
                # Extract response, task, and context
                if "response" in node_data:
                    # For responses, prefer the upward node's response if available
                    lg_up_node = f"{agent_name}_up"
                    if lg_up_node in result["nodes"] and "response" in result["nodes"][lg_up_node]:
                        node_responses[agent_name] = result["nodes"][lg_up_node]["response"]
                    else:
                        node_responses[agent_name] = node_data["response"]
                
                if "task" in node_data and agent_name not in node_tasks:
                    node_tasks[agent_name] = node_data["task"]
                
                if "context" in node_data and agent_name not in node_contexts:
                    node_contexts[agent_name] = node_data["context"]
            
            # Don't include these in the final result to avoid confusion
            # result["node_responses"] = node_responses
            # result["node_tasks"] = node_tasks
            # result["node_contexts"] = node_contexts
        

        
        return result
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise


def test_tree_traversal(child_per_parent: int = 3, depth: int = 2):
    """Test the tree traversal functions.
    
    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
    """
    # Set up logging
    setup_detailed_logging(log_level="debug", only_local_logs=True)
    
    logger.info(f"Testing tree traversal with {child_per_parent} children per parent and {depth} levels")
    
    # Create agent tree using NetworkX
    agent_tree = create_agent_tree(child_per_parent, depth)
    
    # Log the tree structure
    logger.info("Tree structure:")
    for node_name in agent_tree.graph.nodes():
        parent = agent_tree.get_parent(node_name)
        children = agent_tree.get_children(node_name)
        parent_info = f"parent={parent}" if parent else "root"
        children_info = f"children={children}" if children else "leaf"
        logger.info(f"  {node_name}: {parent_info}, {children_info}")
    
    # Visualize the tree
    agent_tree.visualize()
    
    # Get the traversal paths
    downward_nodes = agent_tree.perform_down_traversal(
        node_callback=lambda node, tree: logger.debug(f"Downward traversal visiting node: {node}")
    )
    leaf_nodes = agent_tree.get_leaf_nodes()
    upward_nodes = agent_tree.perform_up_traversal(
        leaf_nodes,
        node_callback=lambda node, tree: logger.debug(f"Upward traversal visiting node: {node}")
    )
    
    # Log the results
    logger.info(f"Downward traversal visited {len(downward_nodes)} nodes: {downward_nodes}")
    logger.info(f"Found {len(leaf_nodes)} leaf nodes: {leaf_nodes}")
    logger.info(f"Upward traversal processed {len(upward_nodes)} nodes: {upward_nodes}")
    
    # Verify that all nodes were visited
    all_nodes = set(agent_tree.graph.nodes())
    if set(downward_nodes) == all_nodes and set(upward_nodes) == all_nodes:
        logger.info("All nodes were visited in both traversals")
    else:
        missing_down = all_nodes - set(downward_nodes)
        missing_up = all_nodes - set(upward_nodes)
        if missing_down:
            logger.warning(f"Nodes missed in downward traversal: {missing_down}")
        if missing_up:
            logger.warning(f"Nodes missed in upward traversal: {missing_up}")
    
    return agent_tree 