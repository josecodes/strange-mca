import argparse
import os
import sys

import textarena as ta
from dotenv import load_dotenv

# Add the project root to the Python path to allow importing from src
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from examples.arena.strangemca_textarena import StrangeMCAAgent

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a two-player chess game between Strange MCA and OpenAI"
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="Depth of the agent tree (default: 2)"
    )
    parser.add_argument(
        "--children",
        type=int,
        default=2,
        help="Number of children per parent node (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--strange-loops",
        type=int,
        default=1,
        help="Number of strange loop iterations (default: 1)",
    )
    return parser.parse_args()


def main():
    """Run a two-player game between a Strange MCA multi-agent team and an OpenAI model."""
    args = parse_args()

    # Define model names and create player name mapping
    openai_model = args.model
    strange_mca_config = {
        "child_per_parent": args.children,
        "depth": args.depth,
        "model": args.model,
    }

    player_names = {
        0: f"Strange MCA Team ({strange_mca_config['model']})",
        1: f"OpenAI ({openai_model})",
    }

    domain_specific_instructions = "I am playing in a game. I must win. My response must be an exact and valid move (no extra words!) that meets the game's rules."

    agents = {
        0: StrangeMCAAgent(
            child_per_parent=strange_mca_config["child_per_parent"],
            depth=strange_mca_config["depth"],
            model=strange_mca_config["model"],
            viz=False,
            print_details=True,
            domain_specific_instructions=domain_specific_instructions,
            strange_loop_count=args.strange_loops,
        ),
        1: ta.agents.OpenAIAgent(
            model_name=openai_model,
            system_prompt=domain_specific_instructions,
        ),
    }

    # Initialize environment and wrap it
    env = ta.make(env_id="Chess-v0")
    env = ta.wrappers.LLMObservationWrapper(env=env)

    # Reset the environment
    env.reset(num_players=len(agents))

    # Game log to track moves
    done = False
    while not done:
        print("=" * 50)
        player_id, observation = env.get_observation()
        print(f"Observation: {observation}")
        # Get the action from the agent (don't print the full observation)
        print("x" * 50)

        print(f"\n{player_names[player_id]} is thinking...")
        action = agents[player_id](observation)
        print(f"{player_names[player_id]} action: {action}")
        print("*" * 50)

        done, info = env.step(action=action)
        if info:
            print("Info:", info)
        print("-" * 50)

    rewards = env.close()
    print("Rewards:")
    if rewards:
        print(rewards)
    else:
        print("No results available.")


if __name__ == "__main__":
    main()
