"""Main module for the Strange MCA package."""

import argparse
import logging
from typing import Dict, List, Optional

from src.strange_mca.agents import Agent, create_agent_configs
from src.strange_mca.prompts import update_agent_prompts
from src.strange_mca.visualization import print_agent_details, print_agent_tree, visualize_agent_graph


def setup_logging():
    """Set up logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_multiagent_system(
    task: str,
    context: str = "",
    team_size: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
) -> str:
    """Run the multiagent system on a task.
    
    This is a simplified implementation that doesn't use LangGraph.
    
    Args:
        task: The task to perform.
        context: The context for the task.
        team_size: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        
    Returns:
        The final response.
    """
    # Create agent configurations
    agent_configs = create_agent_configs(team_size, depth)
    
    # Update agent prompts
    agent_configs = update_agent_prompts(agent_configs, team_size, depth)
    
    # Create agents
    agents = {name: Agent(config, model_name) for name, config in agent_configs.items()}
    
    # Process the task through the tree
    responses = {}
    
    # Start with the root node
    root_name = "L1N1"
    root_agent = agents[root_name]
    
    # Get the children of the root
    children = agent_configs[root_name].children
    
    # Process each child
    for child_name in children:
        child_agent = agents[child_name]
        child_response = child_agent.run(context=context, task=task)
        responses[child_name] = child_response
        logging.info(f"Agent {child_name} response: {child_response[:50]}...")
    
    # Create the synthesis task for the root
    child_responses = "\n\n".join([
        f"{child}: {responses[child]}"
        for child in children
    ])
    synthesis_task = (
        f"Synthesize the following responses from your team members:\n\n"
        f"{child_responses}"
    )
    
    # Get the final response from the root
    final_response = root_agent.run(context=context, task=synthesis_task)
    
    return final_response


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the multiagent system.")
    parser.add_argument(
        "--team_size",
        type=int,
        default=3,
        help="The number of children each non-leaf node has.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="The number of levels in the tree.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Explain the concept of a multiagent system and how it can be used to solve complex problems.",
        help="The task to perform.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="The context for the task.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the agent graph.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="agent_graph",
        help="The path to save the visualization.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="The format to save the visualization in.",
    )
    parser.add_argument(
        "--print_tree",
        action="store_true",
        help="Print the agent tree.",
    )
    parser.add_argument(
        "--print_details",
        action="store_true",
        help="Print details about each agent.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't run the system, just show the configuration.",
    )
    return parser.parse_args()


def main():
    """Run the main function."""
    # Set up logging
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Log configuration
    logging.info(f"Running with team_size={args.team_size}, depth={args.depth}")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Task: {args.task}")
    
    # Calculate the total number of agents
    total_agents = sum(args.team_size ** (i - 1) for i in range(1, args.depth + 1))
    logging.info(f"Total agents: {total_agents}")
    
    # Create agent configurations
    agent_configs = create_agent_configs(args.team_size, args.depth)
    
    # Update agent prompts
    agent_configs = update_agent_prompts(agent_configs, args.team_size, args.depth)
    
    # Print the agent tree if requested
    if args.print_tree:
        print_agent_tree(agent_configs)
        print()
    
    # Print agent details if requested
    if args.print_details:
        print_agent_details(agent_configs)
        print()
    
    # Visualize the agent graph if requested
    if args.visualize:
        output_file = visualize_agent_graph(
            agent_configs,
            output_path=args.output,
            format=args.format,
        )
        if output_file:
            logging.info(f"Visualization saved to {output_file}")
    
    # Run the system if not a dry run
    if not args.dry_run:
        # Run the system
        logging.info("Running the multiagent system...")
        response = run_multiagent_system(
            task=args.task,
            context=args.context,
            team_size=args.team_size,
            depth=args.depth,
            model_name=args.model,
        )
        
        # Print the response
        print("\n" + "=" * 80)
        print("FINAL RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)


if __name__ == "__main__":
    main() 