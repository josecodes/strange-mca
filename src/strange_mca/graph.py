"""LangGraph implementation of the multiagent system."""

from typing import Dict, List, TypedDict, Annotated, Literal, Any, cast, Optional
import logging
import json
import os

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from src.strange_mca.agents import Agent, AgentConfig, create_agent_configs
from src.strange_mca.logging_utils import DetailedLoggingCallbackHandler, setup_detailed_logging

# Set up logging
logger = logging.getLogger("strange_mca")

# Define a custom reducer for the responses dictionary
def responses_reducer(current_dict: Dict[str, str], update: Dict[str, str]) -> Dict[str, str]:
    """Update a dictionary with new key-value pairs."""
    result = current_dict.copy()
    result.update(update)
    return result


class State(TypedDict):
    """State of the multiagent graph."""
    
    # The current task being processed
    task: str
    
    # The context for the current task
    context: str
    
    # The current agent being executed
    current_agent: str
    
    # The responses from each agent - using Annotated with a custom reducer
    responses: Annotated[Dict[str, str], responses_reducer]
    
    # The final response
    final_response: str


def create_task_decomposition_prompt(task: str, system_prompt: str) -> str:
    """Create a prompt for task decomposition.
    
    Args:
        task: The original task to decompose.
        system_prompt: The system prompt of the agent that will decompose the task.
        
    Returns:
        A prompt for task decomposition.
    """
    return f"""Based on your system prompt:

{system_prompt}

Your task is to break down the following task into subtasks for your team members:

TASK: {task}

Please decompose this task into clear, focused subtasks that your team members can work on.
"""


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
    task = state["task"]
    context = state["context"]
    current_agent = state["current_agent"]
    
    # Determine if this is a downward (task decomposition) or upward (synthesis) pass
    is_synthesis = task.startswith("Synthesize the following responses")
    
    if is_synthesis:
        logger.debug(f"[{current_agent}] Processing synthesis task (upward pass)")
    else:
        # Check if this is a non-leaf node that needs to decompose the task
        if agent.config.children:
            logger.debug(f"[{current_agent}] Processing task decomposition (downward pass)")
            # Create a task decomposition prompt
            decomposition_prompt = create_task_decomposition_prompt(task, agent.config.system_prompt)
            # Run the agent with the decomposition prompt
            task = decomposition_prompt
        else:
            logger.debug(f"[{current_agent}] Processing leaf node task (downward pass)")
    
    # Get the agent's response
    response = agent.run(context=context, task=task)
    
    # Create a copy of the current responses
    updated_responses = state["responses"].copy()
    
    # Update the responses dictionary with the new response
    # Use both the current_agent and the agent's full_name as keys
    # This ensures we can find the response by either the node name or the agent name
    updated_responses[current_agent] = response
    updated_responses[agent.config.full_name] = response
    
    # Log the updated responses (simplified)
    logger.debug(f"[{current_agent}] Updated responses: {list(updated_responses.keys())}")
    
    # If this is a synthesis task for the root node, set the final response
    if is_synthesis and agent.config.parent is None:
        logger.debug(f"[{current_agent}] Completed synthesis for root node")
        return {
            "responses": updated_responses,
            "final_response": response
        }
    
    # For all other cases, just update the responses
    return {"responses": updated_responses}


def create_graph(
    child_per_parent: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
    langgraph_viz_dir: Optional[str] = None,
):
    """Create a LangGraph for the multiagent system.
    
    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        langgraph_viz_dir: If provided, generate a visualization of the LangGraph structure
                          in this directory. If None, no visualization is generated.
        
    Returns:
        The compiled graph.
    """
    # Log the creation of the bidirectional traversal graph
    logger.info("Creating Tree with Bidirectional Graph Traversal")
    logger.info(f"Tree structure: {child_per_parent} children per parent, {depth} levels deep")
    logger.info("Downward pass: Task decomposition from parent nodes to children")
    logger.info("Upward pass: Response synthesis from children to parent nodes")
    
    # Create agent configurations
    agent_configs = create_agent_configs(child_per_parent, depth)
    
    # Create agents
    agents = {name: Agent(config, model_name) for name, config in agent_configs.items()}
    
    # Create the graph
    graph_builder = StateGraph(State)
    
    # Add nodes for each agent
    for name, agent in agents.items():
        # Log the node name for debugging (simplified)
        logger.debug(f"[GRAPH] Adding node {name} to the graph")
        graph_builder.add_node(name, lambda state, agent=agent: process_agent(state, agent))
    
    # Create synthesis nodes for each non-leaf node
    for name, config in agent_configs.items():
        if config.children:
            # This is a non-leaf node, create a synthesis node for it
            synthesis_node_name = f"synthesize_{name}"
            
            # Create a synthesis function for this node
            def create_synthesis_function(parent_name, children_names):
                def synthesize_responses(state: State) -> dict:
                    """Synthesize responses from child nodes for a specific parent."""
                    # Check if all children have processed
                    all_processed = True
                    for child in children_names:
                        if child not in state["responses"]:
                            all_processed = False
                            break
                    
                    if all_processed:
                        logger.debug(f"[SYNTHESIZE:{parent_name}] Creating synthesis task from child responses")
                        
                        # Create the synthesis task for the parent
                        child_responses = "\n\n".join([
                            f"{child}: {state['responses'][child]}"
                            for child in children_names
                        ])
                        synthesis_task = (
                            f"Synthesize the following responses from your team members:\n\n"
                            f"{child_responses}"
                        )
                        logger.debug(f"[SYNTHESIZE:{parent_name}] Synthesis task: {synthesis_task}")
                        # Update the state for the next node (the parent)
                        return {
                            "task": synthesis_task,
                            "current_agent": parent_name
                        }
                    
                    # Log which children have not processed yet (debug level)
                    missing_children = [child for child in children_names if child not in state["responses"]]
                    logger.debug(f"[SYNTHESIZE:{parent_name}] Waiting for child responses: {missing_children}")
                    
                    # Return the state unchanged
                    return {}
                
                return synthesize_responses
            
            # Add the synthesis node with the created function
            synthesis_function = create_synthesis_function(name, config.children)
            graph_builder.add_node(synthesis_node_name, synthesis_function)
            logger.debug(f"[GRAPH] Adding synthesis node {synthesis_node_name} for parent {name}")
    
    # Add edges based on the tree structure
    for name, config in agent_configs.items():
        if config.children:
            # This is a parent node
            # Add edges from parent to children (downward pass)
            for child_name in config.children:
                logger.debug(f"[GRAPH] Adding edge from {name} to {child_name} (downward pass)")
                graph_builder.add_edge(name, child_name)
            
            # Add edges from children to synthesis node (upward pass preparation)
            synthesis_node_name = f"synthesize_{name}"
            for child_name in config.children:
                # If the child is a leaf node, connect directly to synthesis
                if not agent_configs[child_name].children:
                    logger.debug(f"[GRAPH] Adding edge from {child_name} to {synthesis_node_name} (upward pass)")
                    graph_builder.add_edge(child_name, synthesis_node_name)
                else:
                    # If the child is not a leaf, connect its synthesis node to this synthesis node
                    child_synthesis_node = f"synthesize_{child_name}"
                    logger.debug(f"[GRAPH] Adding edge from {child_synthesis_node} to {synthesis_node_name} (upward pass)")
                    graph_builder.add_edge(child_synthesis_node, synthesis_node_name)
            
            # If this is the root node, connect synthesis node directly to END
            if config.parent is None:
                logger.debug(f"[GRAPH] Adding edge from {synthesis_node_name} to END (final response)")
                graph_builder.add_edge(synthesis_node_name, END)
                
                # Add edge from root node to END for the case when it processes a synthesis task
                logger.debug(f"[GRAPH] Adding edge from {name} to END (root node final response)")
                graph_builder.add_edge(name, END)
            else:
                # For non-root nodes, add edge from synthesis node to parent (upward pass completion)
                logger.debug(f"[GRAPH] Adding edge from {synthesis_node_name} to {name} (upward pass completion)")
                graph_builder.add_edge(synthesis_node_name, name)
        else:
            # This is a leaf node with no children
            # No additional edges needed as they're already connected to synthesis nodes
            pass
    
    # Set the entry point
    graph_builder.set_entry_point("L1N1")
    
    # Compile the graph
    compiled_graph = graph_builder.compile()
    
    graph_dict = graph_builder.__dict__
    logger.debug(f"LangGraph nodes: {graph_dict.get('nodes', {}).keys()}")
    logger.debug(f"LangGraph edges: {graph_dict.get('edges', {})}")

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


def run_graph(
    graph,
    task: str,
    context: str = "",
    log_level: str = "warn",
    only_local_logs: bool = False,
    langgraph_viz_dir: Optional[str] = None,

) -> dict:
    """Run the graph on a task.
    
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
    # Create the initial state with the root node as the starting point
    # The graph will handle the bidirectional traversal based on its structure
    state = {
        "task": task,
        "context": context,
        "current_agent": "L1N1",  # Start with the root node
        "responses": {},
        "final_response": "",
    }
    
    # Set up callbacks with the appropriate log level
    invoke_config = {}
    callback_handler = DetailedLoggingCallbackHandler(debug_max=False, log_level=log_level)
    invoke_config["callbacks"] = [callback_handler]
    
    # Set up detailed logging with the appropriate log level
    setup_detailed_logging(log_level=log_level, only_local_logs=only_local_logs)
    
    # Run the graph
    result = graph.invoke(state, config=invoke_config)
    
    # Return the entire result
    return result 