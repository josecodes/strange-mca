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

class State(TypedDict):
    """State of the multiagent graph."""
    original_task: str
    nodes: Dict[str, Dict[str, str]]
    current_node: str
    final_response: str
    strange_loop_prompt: str
    strange_loop_response: str

def create_task_decomposition_prompt(task: str, context: str, child_nodes: List[str]) -> str:
    """Create a prompt for task decomposition.
    
    Args:
        task: The original task to decompose.
        context: The context for the agent that will decompose the task.
        child_nodes: List of child node names that will receive the subtasks.
        
    Returns:
        A prompt for task decomposition.
    """
    child_nodes_str = "\n".join([f"- {node}" for node in child_nodes])
    
    return f"""

{context}

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
    
    return f"""Synthesize the following responses from your team members:

{formatted_responses}

Your task is to:
1. Integrate the key insights from each response
2. Resolve any contradictions or inconsistencies
3. Provide a coherent and concise answer

Format your response as a well-structured summary."""



def create_strange_loop_prompt(original_task: str, tentative_response: str) -> str:
    """Create a prompt for the strange loop.
    
    Args:
        task: The original task to complete.
        tentative_response: The tentative response from the team.
    """
    return f"""
    
    I am the leader of a team of AI agents. 

    I was given the following task to complete:

    Task:     
    **************************************************
    {original_task}
    **************************************************

    My team and I produced this response:

     Response: 
    **************************************************
    {tentative_response}
    **************************************************
    
    Is that a great response for the task? If so, then simply provide that is the final response.

    If it could be improved upon, make some revisions and produce the final response.

    Format the revised final response  with in the following format:
    
    Final Response:
    **************************************************    
    [Final response]
    **************************************************

    After this section, you can provide an explanation of reasoning for revisions made (or lack thereof).
    """
def parse_strange_loop_response(response: str) -> str:
    """Extract the final response from the strange loop output.
    
    Args:
        response: The full response from the strange loop prompt.
        
    Returns:
        The extracted final response text, or the original response if no final response section found.
    """
    # Handle empty or None response
    if not response:
        return ""
    
    # Try to find the final response section using regex pattern matching
    import re
    pattern = r"Final Response:\s*\n\*{10,}\s*\n(.*?)\n\*{10,}"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If regex fails, try line-by-line parsing
    lines = response.split('\n')
    final_response_lines = []
    in_final_response = False
    found_asterisks = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if we're at the "Final Response:" line
        if line_stripped == "Final Response:" or line_stripped.startswith("Final Response:"):
            in_final_response = False  # Not yet in the response content
            found_asterisks = False
            continue
        
        # Check if we're at the first line of asterisks after "Final Response:"
        if not in_final_response and not found_asterisks and line_stripped.startswith('*****'):
            found_asterisks = True
            continue
        
        # Now we're in the actual response content
        if found_asterisks and not in_final_response:
            in_final_response = True
            
        # Check if we're at the closing line of asterisks
        if in_final_response and line_stripped.startswith('*****'):
            break
        
        # Add line to final response if we're in the final response section
        if in_final_response:
            final_response_lines.append(line)
    
    # If we found a final response, return it
    if final_response_lines:
        return '\n'.join(final_response_lines).strip()
    
    # If all else fails, return the original response
    return response.strip()



def create_execution_graph(
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
    for agent_name in agent_tree.mca_graph.nodes():
        config = agent_tree.get_config(agent_name)
        agents[agent_name] = Agent(config, model_name=model_name)
        logger.debug(f"Created agent for {agent_name}")
    
    # Initialize the graph builder
    lg_graph_builder = StateGraph(State)
    
    def down_function(state: State, lg_node_name: str) -> Dict[str, Any]:
        """Process a node in the downward pass (task decomposition)."""
        # Extract the AgentTree node name from the LangGraph node name
        agent_name = lg_node_name.split("_down")[0]
        agent = agents[agent_name]
        
        #to work with how langraph updates state
        nodes = state["nodes"].copy()
        state_updates: State = {"nodes": nodes, "current_node": lg_node_name}

        task = nodes[lg_node_name]["task"]
        
        if agent_tree.is_leaf(agent_name):
            logger.debug(f"[{lg_node_name}] Processing leaf node task (downward pass)")
            response = agent.run(task=task)
            nodes[lg_node_name]["response"] = response
            return state_updates
        else:
            logger.debug(f"[{lg_node_name}] Decomposing task for children (downward pass)")
            agent_children = agent_tree.get_children(agent_name)
            
            child_nodes_str = ", ".join(agent_children)
            context = f"You are coordinating a task across {len(agent_children)} agents: {child_nodes_str}."
            decomposition_prompt = create_task_decomposition_prompt(task, context, agent_children)
            nodes[lg_node_name]["decomposition_prompt"] = decomposition_prompt
            response = agent.run(task=decomposition_prompt)
            nodes[lg_node_name]["response"] = response
            
            # Parse the response to extract tasks for children
            for agent_child in agent_children:
                lg_child_down = f"{agent_child}_down"
                if lg_child_down not in nodes:
                    nodes[lg_child_down] = {}
                child_task_prefix = f"{agent_child}: "
                task_found = False
                for line in response.split("\n"):
                    if line.startswith(child_task_prefix):
                        child_task = line[len(child_task_prefix):].strip()
                        nodes[lg_child_down]["task"] = child_task
                        task_found = True
                        break
                if not task_found:
                    logger.warning(f"No task found for child agent {agent_child} in response")

            return state_updates
    
    def up_function(state: State, lg_node_name: str) -> Dict[str, Any]:
        """Process a node in the upward pass (response synthesis)."""
        agent_name = lg_node_name.split("_up")[0]
        agent = agents[agent_name]

        #to work with how langraph updates state
        nodes = state["nodes"].copy()
        state_updates: State = {"nodes": nodes, "current_node": lg_node_name}

        if lg_node_name not in nodes:
            nodes[lg_node_name] = {}
        else:
            logger.warning(f"[{lg_node_name}] up pass, already exists in nodes")
        if agent_tree.is_leaf(agent_name):
            logger.debug(f"[{lg_node_name}] Processing leaf node (upward pass)")
            lg_down_node = f"{agent_name}_down"
            if lg_down_node in nodes and "response" in nodes[lg_down_node]:
                nodes[lg_node_name]["response"] = nodes[lg_down_node]["response"]
            else:
                logger.warning(f"No response found for {lg_down_node}")
        else:
            logger.debug(f"[{lg_node_name}] Processing non-leaf node synthesis (upward pass)")
            agent_children = agent_tree.get_children(agent_name)
            child_responses = {}
            for agent_child in agent_children:
                lg_child_up = f"{agent_child}_up"
                if lg_child_up in nodes and "response" in nodes[lg_child_up]:
                    child_responses[agent_child] = nodes[lg_child_up]["response"]
                else:
                    child_responses[agent_child] = ""
                    logger.warning(f"No response found for {lg_child_up}")
            synthesis_prompt = create_synthesis_prompt(child_responses)
            nodes[lg_node_name]["synthesis_prompt"] = synthesis_prompt
            response = agent.run(task=synthesis_prompt)
            nodes[lg_node_name]["response"] = response
            if agent_tree.is_root(agent_name):
                logger.debug(f"[{lg_node_name}] Processing root node synthesis (upward pass)")
                strange_loop_prompt = create_strange_loop_prompt(state["original_task"], response)
                strange_loop_response = agent.run(task=strange_loop_prompt)
                final_response = parse_strange_loop_response(strange_loop_response)
                state_updates["strange_loop_prompt"] = strange_loop_prompt
                state_updates["strange_loop_response"] = strange_loop_response
                state_updates["final_response"] = final_response
        return state_updates
    

    def add_lg_node_edge(agent_name: str, predecessor_node: str, direction: Literal["down", "up"]) -> None:
        """Callback to add nodes to the LangGraph during traversal."""
        lg_node = f"{agent_name}_{direction}"
        if direction == "down":
            direction_function = down_function
        else:
            direction_function = up_function
        lg_graph_builder.add_node(lg_node, lambda s: direction_function(s, lg_node))
        if predecessor_node is not None:
            lg_graph_builder.add_edge(predecessor_node+f"_{direction}", lg_node)
            logger.debug(f"Added {direction} edge: {predecessor_node}_{direction} -> {lg_node}")
        
        
    down_list = agent_tree.perform_down_traversal(node_callback=add_lg_node_edge)
    up_list = agent_tree.perform_up_traversal(node_callback=add_lg_node_edge)
    lg_graph_builder.add_edge(down_list[-1]+'_down', up_list[0]+'_up')
    lg_graph_builder.add_edge(down_list[0]+'_up', END)

    root_node = agent_tree.get_root()
    lg_graph_builder.set_entry_point(f"{root_node}_down")
    logger.info(f"Set entry point to {root_node}_down")
    
    # Compile the graph
    compiled_lg_graph = lg_graph_builder.compile()
    logger.info("Compiled LangGraph successfully")
    
    # Generate visualization if requested
    if langgraph_viz_dir:
        from src.strange_mca.visualization import visualize_langgraph
        visualize_langgraph(compiled_lg_graph, langgraph_viz_dir)
    
    return compiled_lg_graph


def run_execution_graph(
    execution_graph,
    task: str,
    log_level: str = "warn",
    only_local_logs: bool = False,
    langgraph_viz_dir: Optional[str] = None,
) -> dict:
    """Run the bidirectional graph on a task.
    
    Args:
        graph: The compiled graph to run.
        task: The task to perform.
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
    for lg_node in execution_graph.get_graph().nodes:
        if lg_node.endswith("_down") and "_" not in lg_node.split("_down")[0]:
            at_root_node = lg_node.split("_down")[0]
            break
    
    if not at_root_node:
        raise ValueError("Could not find root node in the graph")
    
    logger.info(f"Identified root node: {at_root_node}")
    
    # initialize state
    initial_state: State = {
        "original_task": task,
        "nodes": {
            f"{at_root_node}_down": {
                "task": task,
            }
        },
    }
    
    logger.info("Running bidirectional graph...")
    
    try:
        # Run the graph with the initial state
        config = RunnableConfig(
            callbacks=[callback_handler],
            recursion_limit=100 #TODO: calculate this?
        )
        result = execution_graph.invoke(initial_state, config=config)
        return result
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise

