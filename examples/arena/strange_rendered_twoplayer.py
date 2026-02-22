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


def main():
    """Run a Chess game between a Strange MCA multi-agent team and an OpenAI model."""
    # Define model names and create player name mapping
    openai_model = "gpt-4o-mini"
    strange_mca_config = {"child_per_parent": 2, "depth": 2, "model": "gpt-4o-mini"}
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
            max_rounds=2,
            enable_downward_signals=True,
            viz=False,
            print_details=False,
            domain_specific_instructions=domain_specific_instructions,
            strange_loop_count=2,
        ),
        1: ta.agents.OpenAIAgent(
            model_name=openai_model,
            system_prompt=domain_specific_instructions,
        ),
    }

    # Initialize environment and wrap it
    env = ta.make(env_id="Chess-v0")
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env = ta.wrappers.RenderWrappers.CursesRenderWrapper(
        env=env,
        player_names=player_names,
    )
    # Reset the environment
    env.reset(num_players=len(agents))
    # Game log to track moves
    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, info = env.step(action=action)
    rewards = env.close()
    print("Rewards:")
    if rewards:
        print(rewards)
    else:
        print("No results available.")


if __name__ == "__main__":
    main()
