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
    
    # Dictionary mapping node names to their specific tasks
    node_tasks: Dict[str, str]
    
    # Dictionary mapping node names to their specific contexts
    node_contexts: Dict[str, str]
    
    # Dictionary mapping node names to their responses
    node_responses: Dict[str, str]
    
    # The current agent being executed
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


def process_agent(
    state: State,
    agent: Agent,
) -> dict:
    """Process an agent in the graph.
    
    Args:
        state: The current state of the graph.
        agent: The agent to process.
        
    Returns:
        The updated state.
    """
    original_task = state["original_task"]
    task = state.get("node_tasks", {}).get(agent.config.name, original_task)
    context = state.get("node_contexts", {}).get(agent.config.name, state["original_context"])
    current_node = state["current_node"]
    
    # Determine if this is a downward (task decomposition) or upward (synthesis) pass
    is_synthesis = task.startswith("Synthesize the following responses")
    
    if is_synthesis:
        logger.debug(f"[{current_node}] Processing synthesis task (upward pass)")
    else:
        # Check if this is a non-leaf node that needs to decompose the task
        if agent.config.children:
            logger.debug(f"[{current_node}] Processing task decomposition (downward pass)")
            # Create a task decomposition prompt
            decomposition_prompt = create_task_decomposition_prompt(task, agent.config.system_prompt, agent.config.children)
            # Run the agent with the decomposition prompt
            task = decomposition_prompt
        else:
            logger.debug(f"[{current_node}] Processing leaf node task (downward pass)")
    
    # Get the agent's response
    response = agent.run(context=context, task=task)
    
    # Update the node responses
    node_responses = state.get("node_responses", {}).copy()
    node_responses[agent.config.name] = response
    
    # If this is a task decomposition, parse the response to extract tasks for children
    if not is_synthesis and agent.config.children:
        node_tasks = state.get("node_tasks", {}).copy()
        
        # Extract tasks for each child from the response
        if response:
            logger.debug(f"[{current_node}] Parsing decomposition response")
            lines = response.strip().split('\n')
            for line in lines:
                for child in agent.config.children:
                    if line.startswith(f"{child}:"):
                        child_task = line[len(f"{child}:"):].strip()
                        node_tasks[child] = child_task
                        logger.debug(f"[{current_node}] Assigned task to {child}: {child_task[:50]}...")
        
        # If any child doesn't have a task assigned, give them a default task
        for child in agent.config.children:
            if child not in node_tasks:
                node_tasks[child] = f"Help solve this task: {task}"
                logger.debug(f"[{current_node}] Assigned default task to {child}")
        
        return {
            "node_responses": node_responses,
            "node_tasks": node_tasks
        }
    
    return {
        "node_responses": node_responses
    }


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
    # Ensure StateGraph is imported
    from langgraph.graph import StateGraph, END
    
    # Log the creation of the bidirectional traversal graph
    logger.info("Creating Tree with Bidirectional Graph Traversal")
    logger.info(f"Tree structure: {child_per_parent} children per parent, {depth} levels deep")
    logger.info("Downward pass: Task decomposition from parent nodes to children")
    logger.info("Upward pass: Response synthesis from children to parent nodes")
    
    # Create agent tree using NetworkX
    agent_tree = create_agent_tree(child_per_parent, depth)
    
    # Get the traversal paths
    downward_nodes = agent_tree.perform_down_traversal()
    leaf_nodes = agent_tree.get_leaf_nodes()
    upward_nodes = agent_tree.perform_up_traversal(leaf_nodes)
    
    logger.info(f"Downward traversal: {downward_nodes}")
    logger.info(f"Leaf nodes: {leaf_nodes}")
    logger.info(f"Upward traversal: {upward_nodes}")
    
    # Create agents
    agents = {name: Agent(agent_tree.get_config(name), model_name) 
              for name in agent_tree.graph.nodes()}
    
    # Create the graph
    graph_builder = StateGraph(State)
    
    # Define the downward function (task decomposition)
    def down_function(state: State, config: dict) -> dict:
        """Decompose tasks during the downward pass."""
        node_name = config["node_name"]
        agent = agents[node_name]
        
        logger.debug(f"[{node_name}_down] Processing downward pass")
        
        # Get the task and context for this node
        task = state["node_tasks"].get(node_name, state["original_task"])
        context = state["node_contexts"].get(node_name, state["original_context"])
        
        # Check if this is a non-leaf node that needs to decompose the task
        if not agent_tree.is_leaf(node_name):
            logger.debug(f"[{node_name}_down] Decomposing task for children")
            # Create a task decomposition prompt
            children = agent_tree.get_children(node_name)
            decomposition_prompt = create_task_decomposition_prompt(task, agent.config.system_prompt, children)
            # Run the agent with the decomposition prompt
            response = agent.run(context=context, task=decomposition_prompt)
        else:
            logger.debug(f"[{node_name}_down] Processing leaf node task")
            # Run the agent with the original task
            response = agent.run(context=context, task=task)
        
        # Update the node responses
        node_responses = state["node_responses"].copy()
        node_responses[node_name] = response
        
        # If this is a non-leaf node, parse the response to extract tasks for children
        if not agent_tree.is_leaf(node_name):
            # Parse the decomposition response to extract tasks for each child
            node_tasks = state["node_tasks"].copy()
            
            # Extract tasks for each child from the response
            if response:
                logger.debug(f"[{node_name}_down] Parsing decomposition response")
                lines = response.strip().split('\n')
                for line in lines:
                    for child in children:
                        if line.startswith(f"{child}:"):
                            child_task = line[len(f"{child}:"):].strip()
                            node_tasks[child] = child_task
                            logger.debug(f"[{node_name}_down] Assigned task to {child}: {child_task[:50]}...")
            
            # If any child doesn't have a task assigned, give them a default task
            for child in children:
                if child not in node_tasks:
                    node_tasks[child] = f"Help solve this task: {task}"
                    logger.debug(f"[{node_name}_down] Assigned default task to {child}")
            
            return {
                "node_responses": node_responses,
                "node_tasks": node_tasks
            }
        
        # For leaf nodes, check if all siblings have completed
        if agent_tree.is_leaf(node_name):
            parent = agent_tree.get_parent(node_name)
            siblings = agent_tree.get_children(parent)
            all_siblings_done = all(sib in node_responses for sib in siblings)
            
            if all_siblings_done:
                logger.debug(f"[{node_name}_down] All siblings completed, preparing synthesis for parent {parent}")
                
                # Create a dictionary of child responses for the parent
                child_responses = {child: node_responses[child] for child in siblings}
                
                # Create a synthesis task for the parent
                synthesis_prompt = create_synthesis_prompt(child_responses)
                
                # Update the parent's task
                node_tasks = state["node_tasks"].copy()
                node_tasks[parent] = synthesis_prompt
                
                return {
                    "node_responses": node_responses,
                    "node_tasks": node_tasks
                }
        
        return {
            "node_responses": node_responses
        }
    
    # Define the upward function (response synthesis)
    def up_function(state: State, config: dict) -> dict:
        """Synthesize responses during the upward pass."""
        node_name = config["node_name"]
        agent = agents[node_name]
        
        logger.debug(f"[{node_name}_up] Processing upward pass")
        
        # Skip processing for leaf nodes as per requirement
        if agent_tree.is_leaf(node_name):
            logger.debug(f"[{node_name}_up] Skipping leaf node in upward pass")
            return {}
        
        # Get the task and context for this node
        task = state["node_tasks"].get(node_name, state["original_task"])
        context = state["node_contexts"].get(node_name, state["original_context"])
        
        # Run the agent with the synthesis task
        response = agent.run(context=context, task=task)
        
        # Update the node responses
        node_responses = state["node_responses"].copy()
        node_responses[f"{node_name}_synthesis"] = response
        
        # Check if this is the root node
        is_root = agent_tree.is_root(node_name)
        
        if is_root:
            logger.debug(f"[{node_name}_up] Completed synthesis for root node")
            # For the root node, set the final response
            return {
                "node_responses": node_responses,
                "final_response": response
            }
        
        return {
            "node_responses": node_responses
        }
    
    # Add nodes for each agent in the downward and upward passes
    for node_name in agent_tree.graph.nodes():
        # Add downward node
        down_node = f"{node_name}_down"
        graph_builder.add_node(
            down_node, 
            lambda state, node_name=node_name: down_function(state, {"node_name": node_name})
        )
        
        # Add upward node
        up_node = f"{node_name}_up"
        graph_builder.add_node(
            up_node, 
            lambda state, node_name=node_name: up_function(state, {"node_name": node_name})
        )
    
    # Add edges based on the tree structure
    # For the downward pass
    for node_name in agent_tree.graph.nodes():
        down_node = f"{node_name}_down"
        
        if not agent_tree.is_leaf(node_name):
            # Connect to first child's downward node
            children = agent_tree.get_children(node_name)
            first_child_down = f"{children[0]}_down"
            graph_builder.add_edge(down_node, first_child_down)
        else:
            # For leaf nodes, connect to parent's upward node
            parent = agent_tree.get_parent(node_name)
            parent_up = f"{parent}_up"
            graph_builder.add_edge(down_node, parent_up)
    
    # For the upward pass
    for node_name in agent_tree.graph.nodes():
        if not agent_tree.is_leaf(node_name):
            up_node = f"{node_name}_up"
            
            if agent_tree.is_root(node_name):
                # Root node connects to END
                graph_builder.add_edge(up_node, END)
            else:
                # Non-root nodes connect to parent's upward node
                parent = agent_tree.get_parent(node_name)
                parent_up = f"{parent}_up"
                graph_builder.add_edge(up_node, parent_up)
    
    # Set the entry point to the root node's downward pass
    root_node = agent_tree.get_root()
    graph_builder.set_entry_point(f"{root_node}_down")
    
    # Compile the graph
    compiled_graph = graph_builder.compile()
    
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
            logger.debug(f"LangGraph visualization saved to {lg_viz_path}.png")
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
    root_node = None
    for node in graph.get_graph().nodes:
        if node.endswith("_down") and "_" not in node.split("_down")[0]:
            root_node = node.split("_down")[0]
            break
    
    if not root_node:
        raise ValueError("Could not find root node in the graph")
    
    logger.info(f"Identified root node: {root_node}")
    
    # Create the initial state
    initial_state = {
        "original_task": task,
        "original_context": context,
        "node_tasks": {root_node: task},
        "node_contexts": {root_node: context},
        "node_responses": {},
        "final_response": ""
    }
    
    # Log the initial state
    logger.info(f"Running bidirectional graph with task: {task}")
    if context:
        logger.info(f"Context: {context[:100]}..." if len(context) > 100 else f"Context: {context}")
    
    # Create the config with the callback
    config = RunnableConfig(callbacks=[callback_handler])
    
    # Run the graph
    try:
        logger.info("Starting graph execution")
        result = graph.invoke(initial_state, config=config)
        logger.info("Graph execution completed successfully")
        
        # Extract the final response from the root node's synthesis response
        final_response = result.get("node_responses", {}).get(f"{root_node}_synthesis", "")
        if not final_response:
            # If no synthesis response, use the root node's response
            final_response = result.get("node_responses", {}).get(root_node, "No final response generated")
        
        # Update the result with the final response if not already set
        if not result.get("final_response"):
            result["final_response"] = final_response
        
        logger.info(f"Final response: {final_response[:100]}..." if len(final_response) > 100 else f"Final response: {final_response}")
        
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
    
    # Perform downward traversal
    logger.info("--- DOWNWARD PASS ---")
    downward_nodes = agent_tree.perform_down_traversal()
    
    # Get leaf nodes
    leaf_nodes = agent_tree.get_leaf_nodes()
    
    # Perform upward traversal
    logger.info("--- UPWARD PASS ---")
    upward_nodes = agent_tree.perform_up_traversal(leaf_nodes)
    
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