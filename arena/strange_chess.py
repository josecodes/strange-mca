import os
import re
import sys
from dotenv import load_dotenv
import textarena as ta

# Add the project root to the Python path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strange_mca.run_strange_mca import run_strange_mca

# Load environment variables
load_dotenv()

class StrangeMCAAgent(ta.Agent):
    """Custom agent wrapper for the Strange MCA multi-agent system."""
    
    def __init__(
        self,
        child_per_parent: int = 2,
        depth: int = 2,
        model: str = "gpt-3.5-turbo",
        viz: bool = False,
        print_details: bool = False,
    ):
        """
        Initialize the Strange MCA agent.
        
        Args:
            child_per_parent: Number of children per parent node in the agent tree.
            depth: Depth of the agent tree.
            model: The model to use for the agents.
            viz: Whether to generate visualizations.
            print_details: Whether to print detailed information.
        """
        self.child_per_parent = child_per_parent
        self.depth = depth
        self.model = model
        self.viz = viz
        self.print_details = print_details
        
    def __call__(self, observation: str) -> str:
        """
        Process the observation and return an action using the Strange MCA system.
        
        Args:
            observation: The observation from the environment.
            
        Returns:
            The action to take.
        """
        # Create a chess-specific task for the Strange MCA system
        chess_task = f"""
        You are playing a game of chess. Analyze the board carefully and make a strategic move.
        
        Current game state:
        {observation}
        
        Your task is to decide on the best chess move to make in this position.
        Provide your move in standard algebraic notation (e.g., 'e2e4', 'Nf3', etc.) or in UCI format.
        Your final answer should be just the move notation, without any additional text.
        """
        
        # Run the Strange MCA system
        print(f"Running Strange MCA with {self.child_per_parent} children per parent, depth {self.depth}, model {self.model}")
        result = run_strange_mca(
            task=chess_task,
            child_per_parent=self.child_per_parent,
            depth=self.depth,
            model=self.model,
            viz=self.viz,
            print_details=self.print_details,
        )
        
        # Return the final answer from the Strange MCA system
        return result.get("final_answer", "")

def main():
    """Run a Chess game between a Strange MCA multi-agent team and an OpenAI model."""
    # Define the OpenAI agent with a chess-specific system prompt

    
    # Define model names and create player name mapping
    openai_model = "gpt-4o-mini"
    strange_mca_config = {
        "child_per_parent": 2,
        "depth": 2,
        "model": "gpt-3.5-turbo"
    }
    
    player_names = {
        0: f"Strange MCA Team ({strange_mca_config['model']})",
        1: f"OpenAI ({openai_model})"
    }
    
    agents = {
        0: StrangeMCAAgent(
            child_per_parent=strange_mca_config["child_per_parent"],
            depth=strange_mca_config["depth"],
            model=strange_mca_config["model"],
            viz=False,
            print_details=True
        ),
        1: ta.agents.OpenAIAgent(
            model_name=openai_model, 
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