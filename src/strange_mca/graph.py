"""LangGraph implementation of the multiagent system."""

from typing import Annotated, Dict, List, TypedDict, Literal, Any, cast

from langgraph.graph import END, StateGraph

from src.strange_mca.agents import Agent, AgentConfig, create_agent_configs


# Define a custom reducer for the responses dictionary
def responses_reducer(current_dict: Dict[str, str], update: Dict[str, str]) -> Dict[str, str]:
    """Update a dictionary with new key-value pairs."""
    result = current_dict.copy()
    result.update(update)
    return result


class AgentInput(TypedDict):
    """Input for an agent in the graph."""
    
    context: str
    task: str


class AgentOutput(TypedDict):
    """Output from an agent in the graph."""
    
    response: str


class MCAState(TypedDict):
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
    state: MCAState,
    agent: Agent,
) -> dict:
    """Process an agent in the graph.
    
    Args:
        state: The current state of the graph.
        agent: The agent to process.
        
    Returns:
        The updated state.
    """
    # Get the agent's response
    response = agent.run(context=state["context"], task=state["task"])
    
    # Return updates to the state - using a dictionary for responses
    return {"responses": {agent.config.full_name: response}}


def create_graph(
    child_per_parent: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
):
    """Create a LangGraph for the multiagent system.
    
    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        
    Returns:
        A compiled LangGraph.
    """
    # Create agent configurations
    agent_configs = create_agent_configs(child_per_parent, depth)
    
    # Create agents
    agents = {name: Agent(config, model_name) for name, config in agent_configs.items()}
    
    # Create the graph
    graph_builder = StateGraph(MCAState)
    
    # Add nodes for each agent
    for name, agent in agents.items():
        graph_builder.add_node(name, lambda state, agent=agent: process_agent(state, agent))
    
    # Add edges based on the tree structure
    for name, config in agent_configs.items():
        if config.children:
            # This is a parent node
            # Add edges from parent to children
            for child_name in config.children:
                graph_builder.add_edge(name, child_name)
            
            # Add conditional edge from the last child back to parent
            last_child = config.children[-1]
            parent_name = name  # Store the parent name for use in the closure
            
            # Define a condition to route back to parent after all children have processed
            def route_to_parent(state: MCAState, parent=parent_name, children=config.children) -> str:
                # Check if all children have processed
                if all(child in state["responses"] for child in children):
                    # Create the synthesis task for the parent
                    child_responses = "\n\n".join([
                        f"{child}: {state['responses'][child]}"
                        for child in children
                    ])
                    synthesis_task = (
                        f"Synthesize the following responses from your team members:\n\n"
                        f"{child_responses}"
                    )
                    # Return a dictionary with the updated task
                    return {"task": synthesis_task, "__return__": parent}
                return END
            
            # Add conditional edges
            graph_builder.add_conditional_edges(
                last_child,
                route_to_parent
            )
        else:
            # This is a leaf node, add edge to END
            graph_builder.add_edge(name, END)
    
    # Set the entry point
    graph_builder.set_entry_point("L1N1")
    
    # Compile the graph
    compiled_graph = graph_builder.compile()
    
    return compiled_graph


def run_graph(
    graph,
    task: str,
    context: str = "",
) -> str:
    """Run the graph on a task.
    
    Args:
        graph: The compiled graph to run.
        task: The task to perform.
        context: The context for the task.
        
    Returns:
        The final response.
    """
    # Create the initial state
    state = {
        "task": task,
        "context": context,
        "current_agent": "L1N1",
        "responses": {},
        "final_response": "",
    }
    
    # Run the graph
    result = graph.invoke(state)
    
    # Return the final response
    return result["responses"]["L1N1"] 