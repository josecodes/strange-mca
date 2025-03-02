"""Main module for the Strange MCA package."""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

from src.strange_mca.agents import Agent, create_agent_configs
from src.strange_mca.graph import create_graph, run_graph
from src.strange_mca.logging_utils import DetailedLoggingCallbackHandler, setup_detailed_logging
from src.strange_mca.prompts import update_agent_prompts
from src.strange_mca.visualization import print_agent_details, print_agent_tree, visualize_agent_graph

load_dotenv()


def setup_logging(log_level: str = "warn"):
    """Set up logging for the application.
    
    Args:
        log_level: The level of logging detail using standard Python logging levels: "warn", "info", or "debug".
                  Default is "warn" which shows only warnings and errors.
    """
    # Map string log levels to Python logging levels
    log_level_map = {
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG
    }
    
    # Determine the actual logging level to use
    actual_level = log_level_map.get(log_level, logging.WARNING)
    
    logging.basicConfig(
        level=actual_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_multiagent_system(
    task: str,
    context: str = "",
    child_per_parent: int = 3,
    depth: int = 2,
    model_name: str = "gpt-3.5-turbo",
    verbose: bool = False,
    log_level: str = "warn",
    only_local_logs: bool = False,
    langgraph_viz_dir: Optional[str] = None,
) -> tuple[str, Dict[str, str]]:
    """Run the multiagent system on a task using LangGraph.
    
    Args:
        task: The task to perform.
        context: The context for the task.
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        model_name: The name of the LLM model to use.
        verbose: Whether to enable verbose output.
        log_level: The level of logging detail using standard Python logging levels: "warn", "info", or "debug".
                  Default is "warn" which shows only warnings and errors.
        only_local_logs: If True, only show logs from the strange_mca logger and suppress logs from other loggers.
        langgraph_viz_dir: If provided, directory where to generate LangGraph visualization.
        
    Returns:
        A tuple containing the final response and a dictionary of all agent responses.
    """
    try:
        # Set up detailed logging
        setup_detailed_logging(log_level=log_level, only_local_logs=only_local_logs)
            
        # Create agent configurations
        agent_configs = create_agent_configs(child_per_parent, depth)
        
        # Update agent prompts
        agent_configs = update_agent_prompts(agent_configs, child_per_parent, depth)
        
        # Create the LangGraph
        graph = create_graph(
            child_per_parent=child_per_parent, 
            depth=depth, 
            model_name=model_name, 
            langgraph_viz_dir=langgraph_viz_dir
        )
        
        # Run the graph with the appropriate log level
        result = run_graph(
            graph=graph,
            task=task,
            context=context,
            log_level=log_level,
            only_local_logs=only_local_logs
        )
        
        # Get all responses from the graph
        responses = result["responses"]
        
        # Get the final response - prefer the final_response field if it exists
        final_response = result.get("final_response", "")
        if not final_response:
            # Fall back to the L1N1 response if no final_response is available
            final_response = responses.get("L1N1", "No response available")
        
        # Log responses if verbose
        if verbose:
            for agent_name, response in responses.items():
                if agent_name != "L1N1":  # Skip the root node since we'll show it as the final response
                    logging.debug(f"Agent {agent_name} response: {response[:50]}...")
        
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
        "--viz",
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
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["warn", "info", "debug"],
        default="warn",
        help="The level of logging detail to display using standard Python logging levels. Default is 'warn'.",
    )
    parser.add_argument(
        "--only_local_logs",
        action="store_true",
        help="Only show logs from the strange_mca logger and suppress logs from other loggers.",
    )
    return parser.parse_args()


def main():
    """Run the main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging with the specified log level
    setup_logging(log_level=args.log_level)
    
    # Log configuration
    logging.debug(f"Running with child_per_parent={args.child_per_parent}, depth={args.depth}")
    logging.debug(f"Using model: {args.model}")
    logging.debug(f"Task: {args.task}")
    
    # Calculate the total number of agents
    total_agents = sum(args.child_per_parent ** (i - 1) for i in range(1, args.depth + 1))
    logging.debug(f"Total agents: {total_agents}")
    
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
    if args.viz:
        # Generate agent configuration visualization
        output_file = visualize_agent_graph(
            agent_configs,
            output_path=args.output,
            format=args.format,
        )
        if output_file:
            logging.debug(f"Agent configuration visualization saved to {output_file}")
        
        # Set directory for LangGraph visualization
        langgraph_viz_dir = os.path.dirname(args.output)

    else:
        langgraph_viz_dir = None
    
    # Run the system if not a dry run
    if not args.dry_run:
        # Run the system
        logging.debug("Running the multiagent system...")
        final_response, all_responses = run_multiagent_system(
            task=args.task,
            context=args.context,
            child_per_parent=args.child_per_parent,
            depth=args.depth,
            model_name=args.model,
            verbose=args.verbose,
            log_level=args.log_level,
            only_local_logs=args.only_local_logs,
            langgraph_viz_dir=langgraph_viz_dir,
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