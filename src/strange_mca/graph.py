"""LangGraph implementation of the multiagent system."""

from typing import Annotated, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from strange_mca.agents import Agent, AgentConfig, create_agent_configs


class AgentInput(TypedDict):
    """Input for an agent in the graph."""
    
    context: str
    task: str


class AgentOutput(TypedDict):
    """Output from an agent in the graph."""
    
    response: str


class MCAState(BaseModel):
    """State of the multiagent graph."""
    
    # The current task being processed
    task: str
    
    # The context for the current task
    context: str
    
    # The current agent being executed
    current_agent: str
    
    # The responses from each agent
    responses: Dict[str, str] = {}
    
    # The final response
    final_response: str = ""


def process_agent(
    state: MCAState,
    agent: Agent,
) -> MCAState:
    """Process an agent in the graph.
    
    Args:
        state: The current state of the graph.
        agent: The agent to process.
        
    Returns:
        The updated state.
    """
    # Get the agent's response
    response = agent.run(context=state.context, task=state.task)
    
    # Update the state
    state.responses[agent.config.full_name] = response
    
    return state


def create_graph(
    team_size: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
):
    """Create a LangGraph for the multiagent system.
    
    Args:
        team_size: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        
    Returns:
        A compiled LangGraph.
    """
    # Create agent configurations
    agent_configs = create_agent_configs(team_size, depth)
    
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
            
            # Define a condition to route back to parent after all children have processed
            def route_to_parent(state: MCAState, parent=name, children=config.children):
                # Check if all children have processed
                if all(child in state.responses for child in children):
                    # Update the task for the parent to synthesize results
                    child_responses = "\n\n".join([
                        f"{child}: {state.responses[child]}"
                        for child in children
                    ])
                    state.task = (
                        f"Synthesize the following responses from your team members:\n\n"
                        f"{child_responses}"
                    )
                    return parent
                return END  # Return END instead of None
            
            graph_builder.add_conditional_edges(
                last_child,
                route_to_parent,
                {name: name, END: END},
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
    state = MCAState(
        task=task,
        context=context,
        current_agent="L1N1",
    )
    
    # Run the graph
    result = graph.invoke(state)
    
    # Return the final response
    return result.responses["L1N1"] 