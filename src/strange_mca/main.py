"""Main module for the Strange MCA package."""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("dotenv package not found. Environment variables must be set manually.")

from src.strange_mca.agents import Agent, create_agent_configs
from src.strange_mca.graph import create_graph, run_graph
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
    child_per_parent: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
    verbose: bool = False,
) -> tuple[str, Dict[str, str]]:
    """Run the multiagent system on a task using LangGraph.
    
    Args:
        task: The task to perform.
        context: The context for the task.
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        verbose: Whether to enable verbose output.
        
    Returns:
        A tuple containing the final response and a dictionary of all agent responses.
    """
    try:
        # Create agent configurations
        agent_configs = create_agent_configs(child_per_parent, depth)
        
        # Update agent prompts
        agent_configs = update_agent_prompts(agent_configs, child_per_parent, depth)
        
        # Create the LangGraph
        graph = create_graph(child_per_parent, depth, model_name)
        
        # Run the graph
        final_response = run_graph(graph, task, context)
        
        # For compatibility with the verbose output, we need to collect all responses
        # Create agents to access their responses
        agents = {name: Agent(config, model_name) for name, config in agent_configs.items()}
        
        # Process each child agent to get their responses for verbose output
        responses = {}
        root_name = "L1N1"
        children = agent_configs[root_name].children
        
        for child_name in children:
            child_agent = agents[child_name]
            child_response = child_agent.run(context=context, task=task)
            responses[child_name] = child_response
            if verbose:
                logging.info(f"Agent {child_name} response: {child_response[:50]}...")
        
        # Add the final response to the responses dictionary
        responses[root_name] = final_response
        
        return final_response, responses
        
    except Exception as e:
        logging.error(f"Error running multiagent system: {e}")
        if "openai.error.AuthenticationError" in str(e) or "openai.AuthenticationError" in str(e):
            print("\nError: OpenAI API key is invalid or not set correctly.")
            print("Please check your .env file and ensure OPENAI_API_KEY is set properly.")
            print("You can find your API key at https://platform.openai.com/account/api-keys")
        elif "openai.error.RateLimitError" in str(e) or "openai.RateLimitError" in str(e):
            print("\nError: OpenAI API rate limit exceeded.")
            print("Possible solutions:")
            print("1. Check your billing status at https://platform.openai.com/account/billing")
            print("2. Use a different API key")
            print("3. Try again later")
            print("4. Upgrade your OpenAI plan")
        else:
            print(f"\nError: {e}")
        sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the multiagent system.")
    parser.add_argument(
        "--child_per_parent",
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
        default="viz_outputs/agent_graph",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with full agent responses.",
    )
    return parser.parse_args()


def main():
    """Run the main function."""
    # Set up logging
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Log configuration
    logging.info(f"Running with child_per_parent={args.child_per_parent}, depth={args.depth}")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Task: {args.task}")
    
    # Calculate the total number of agents
    total_agents = sum(args.child_per_parent ** (i - 1) for i in range(1, args.depth + 1))
    logging.info(f"Total agents: {total_agents}")
    
    # Create agent configurations
    agent_configs = create_agent_configs(args.child_per_parent, args.depth)
    
    # Update agent prompts
    agent_configs = update_agent_prompts(agent_configs, args.child_per_parent, args.depth)
    
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
        final_response, all_responses = run_multiagent_system(
            task=args.task,
            context=args.context,
            child_per_parent=args.child_per_parent,
            depth=args.depth,
            model_name=args.model,
            verbose=args.verbose,
        )
        
        # Print the response
        print("\n" + "=" * 80)
        print("FINAL RESPONSE:")
        print("=" * 80)
        print(final_response)
        print("=" * 80)
        
        # Print all agent responses if verbose mode is enabled
        if args.verbose:
            print("\n" + "=" * 80)
            print("ALL AGENT RESPONSES:")
            print("=" * 80)
            for agent_name, response in all_responses.items():
                print(f"\n--- {agent_name} ---")
                print(response)
                print("-" * 40)
            print("=" * 80)


if __name__ == "__main__":
    main() 