"""
Script to run the Strange MCA system programmatically.

This script provides a function to run the Strange MCA system with specified parameters,
without needing to use command line arguments.
"""

import logging
import os
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import copy

from src.strange_mca.main import create_output_dir
from src.strange_mca.agents import create_agent_configs
from src.strange_mca.visualization import visualize_agent_graph, visualize_langgraph
from src.strange_mca.graph import create_execution_graph, run_execution_graph

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = logging.getLogger("strange_mca")


def run_strange_mca(
    task: str,
    child_per_parent: int = 2,
    depth: int = 2,
    model: str = "gpt-3.5-turbo",
    log_level: str = "info",
    viz: bool = True,
    all_logs: bool = False,
    print_details: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the Strange MCA system with the specified parameters.
    
    Args:
        task: The task to run.
        child_per_parent: The number of children per parent node.
        depth: The depth of the tree.
        model: The model to use.
        log_level: The log level to use.
        viz: Whether to generate visualizations.
        all_logs: Whether to show logs from dependencies in addition to local logs.
        print_details: Whether to print detailed information.
        output_dir: The output directory to use. If None, a directory will be created.
        
    Returns:
        The result of the execution.
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    if all_logs:
        logging.basicConfig(level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_output_dir(child_per_parent, depth, model)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("Running with the following parameters:")
    logger.info(f"  Task: {task}")
    logger.info(f"  Children per parent: {child_per_parent}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Create agent configurations
    agent_configs = create_agent_configs(child_per_parent, depth)
    logger.info(f"Total agents: {len(agent_configs)}")
    
    # Visualize the agent graph if requested
    if viz:
        output_file = visualize_agent_graph(
            agent_configs,
            output_path=os.path.join(output_dir, "agent_tree_nx"),
            format="png",
        )
        if output_file:
            logger.info(f"Agent tree visualization saved to {output_file}")
    
    # Create the execution graph
    logger.info("Creating execution graph...")
    graph = create_execution_graph(
        child_per_parent=child_per_parent,
        depth=depth,
        model_name=model,
        langgraph_viz_dir=None  # We'll handle visualization separately
    )
    
    # Generate LangGraph visualization if requested
    if viz:
        visualize_langgraph(graph, output_dir)
    
    # Run the execution graph
    logger.info("Running execution graph...")
    result = run_execution_graph(
        execution_graph=graph,
        task=task,
        log_level=log_level,
        only_local_logs=not all_logs,  # Invert the behavior
        langgraph_viz_dir=None  # We've already handled visualization
    )
    
    # Print the final response
    logger.info("Graph execution completed")
    print("\nFinal Response:")
    print("=" * 80)
    print(result.get("final_response", "No final response generated"))
    print("=" * 80)
    
    if print_details:
        print("\nFull State:")
        print("=" * 80)
        # Create a DEEP copy of the result to avoid modifying the original
        state_copy = copy.deepcopy(result)
        
        # Format nodes dictionary for better readability
        if "nodes" in state_copy:
            for node_name, node_data in state_copy["nodes"].items():
                # Truncate long responses for display
                if "response" in node_data and len(node_data["response"]) > 500:
                    state_copy["nodes"][node_name]["response"] = node_data["response"][:500] + "... [truncated]"
                
                # Truncate long tasks for display
                if "task" in node_data and len(node_data["task"]) > 500:
                    state_copy["nodes"][node_name]["task"] = node_data["task"][:500] + "... [truncated]"
        
        # Pretty print the state
        import pprint
        pp = pprint.PrettyPrinter(indent=2, width=100)
        pp.pprint(state_copy)
        print("=" * 80)
    
    # Save the final state to a file
    state_file = os.path.join(output_dir, "final_state.json")
    with open(state_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full state saved to: {state_file}")
    
    # Print execution summary
    print("\nExecution Summary:")
    print(f"Tree structure: {child_per_parent} children per parent, {depth} levels deep")
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Output saved to: {output_dir}")
    
    return result


if __name__ == "__main__":
    # Example usage
    result = run_strange_mca(
        task="if three cows go to a field, one has a baby, and clones itself, how many cows are there?",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=True,
        all_logs=False,
    ) 