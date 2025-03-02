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
    
    if task.startswith("Synthesize"):
        logger.debug(f"[{current_agent}] Processing synthesis task")
    else:
        logger.debug(f"[{current_agent}] Processing task")
    
    # Check if this is a synthesis task for a parent node
    is_synthesis = task.startswith("Synthesize the following responses")
    
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
    
    # If this is a synthesis task and the agent is the top-level node (no parent),
    # this is the final synthesized response
    if is_synthesis and agent.config.parent is None:
        logger.debug(f"[{current_agent}] Completed synthesis for top-level node")
        return {
            "responses": updated_responses,
            "final_response": response
        }
    
    # Otherwise, just update the responses
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
    
    # Add a special node for synthesis
    def synthesize_responses(state: State) -> dict:
        """Synthesize responses from child nodes."""
        # Get the root node and its children
        root_name = "L1N1"
        children = agent_configs[root_name].children
        
        # Check if all children have processed
        all_processed = True
        for child in children:
            if child not in state["responses"]:
                all_processed = False
                break
        
        if all_processed:
            logger.debug(f"[SYNTHESIZE] Creating synthesis task from child responses")
            
            # Create the synthesis task for the parent
            child_responses = "\n\n".join([
                f"{child}: {state['responses'][child]}"
                for child in children
            ])
            synthesis_task = (
                f"Synthesize the following responses from your team members:\n\n"
                f"{child_responses}"
            )
            
            # Get the root agent
            root_agent = agents[root_name]
            
            # Get the synthesized response
            response = root_agent.run(context=state["context"], task=synthesis_task)
            
            # Update the responses dictionary
            updated_responses = state["responses"].copy()
            updated_responses[root_name] = response
            
            # Return the updated state
            return {
                "responses": updated_responses,
                "final_response": response
            }
        
        # Log which children have not processed yet (debug level)
        missing_children = [child for child in children if child not in state["responses"]]
        logger.debug(f"[SYNTHESIZE] Waiting for child responses: {missing_children}")
        
        # Return the state unchanged
        return {}
    
    # Add the synthesis node
    graph_builder.add_node("synthesize", synthesize_responses)
    
    # Add edges based on the tree structure
    for name, config in agent_configs.items():
        if config.children:
            # This is a parent node
            # Add edges from parent to children
            for child_name in config.children:
                logger.debug(f"[GRAPH] Adding edge from {name} to {child_name}")
                graph_builder.add_edge(name, child_name)
            
            # Add edge from the last child to the synthesis node
            last_child = config.children[-1]
            logger.debug(f"[GRAPH] Adding edge from {last_child} to synthesize")
            graph_builder.add_edge(last_child, "synthesize")
            
            # Add edge from the synthesis node to END
            logger.debug(f"[GRAPH] Adding edge from synthesize to END")
            graph_builder.add_edge("synthesize", END)
        else:
            # This is a leaf node with no children
            # If it's not the last child of its parent, add edge to END
            # Otherwise, the edge to the synthesis node is already added
            parent = config.parent
            if parent:
                parent_children = agent_configs[parent].children
                if name != parent_children[-1]:
                    logger.debug(f"[GRAPH] Adding edge from {name} to END")
                    graph_builder.add_edge(name, END)
            else:
                # This is a leaf node with no parent, add edge to END
                logger.debug(f"[GRAPH] Adding edge from {name} to END")
                graph_builder.add_edge(name, END)
    
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
    # Create the initial state
    state = {
        "task": task,
        "context": context,
        "current_agent": "L1N1",
        "responses": {},
        "final_response": "",
    }
    
    # Set up callbacks with the appropriate log level
    config = {}
    callback_handler = DetailedLoggingCallbackHandler(debug_max=False, log_level=log_level)
    config["callbacks"] = [callback_handler]
    
    # Set up detailed logging with the appropriate log level
    setup_detailed_logging(log_level=log_level, only_local_logs=only_local_logs)
    
    # Run the graph
    result = graph.invoke(state, config=config)
    
    # Return the entire result
    return result 