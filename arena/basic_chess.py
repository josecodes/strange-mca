import os
from dotenv import load_dotenv
import textarena as ta

# Load environment variables
load_dotenv()

def main():
    model_player0 = "gpt-4o-mini"
    model_player1 = "gpt-3.5-turbo"
    
    player_names = {
        0: f"Player 0 ({model_player0})",
        1: f"Player 1 ({model_player1})"
    }
    
    agents = {
        0: ta.agents.OpenAIAgent(
            model_name=model_player0, 
        ),
        1: ta.agents.OpenAIAgent(
            model_name=model_player1, 
        ),
    }

    # Initialize environment and wrap it
    env = ta.make(env_id="Chess-v0")
    env = ta.wrappers.LLMObservationWrapper(env=env)
    
    # Option 1: Use no render wrapper for simple text output
    # This will just print the raw observations and actions
    
    # Option 2: Use CursesRenderWrapper for a nicer terminal UI
    # Uncomment the following lines to use CursesRenderWrapper
    # env = ta.wrappers.RenderWrappers.CursesRenderWrapper(
    #     env=env,
    #     player_names={0: "GPT-4o-mini", 1: "GPT-3.5-turbo"},
    # )

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