"""
SpellingBee game example using TextArena with OpenAI agents.

This script demonstrates how to use TextArena to run a SpellingBee game between two OpenAI models.
It uses the TextArena codebase directly from the local installation.
"""

import os
from dotenv import load_dotenv
import textarena as ta

# Load environment variables
load_dotenv()

def main():
    """Run a SpellingBee game between two OpenAI models."""
    # Initialize agents
    agents = {
        0: ta.agents.OpenAIAgent(model_name="gpt-4o-mini"),
        1: ta.agents.OpenAIAgent(model_name="gpt-3.5-turbo"),
    }

    # Initialize environment and wrap it
    env = ta.make(env_id="SpellingBee-v0")
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
    
    # Game log to track all actions and events
    game_log = []
    
    # Game loop
    done = False
    while not done:
        # Get the current player ID and observation
        player_id, observation = env.get_observation()
        
        # Display game state information
        print("\n" + "="*80)
        print(f"GAME STATE:")
        print(f"Current Player: {player_id} ({'GPT-4o-mini' if player_id == 0 else 'GPT-3.5-turbo'})")
        
        # Extract and display allowed letters from the environment state
        if hasattr(env.state, 'allowed_letters'):
            print(f"Allowed Letters: {', '.join(sorted(env.state.allowed_letters))}")
        
        # Display words played so far
        if hasattr(env.state, 'words_played'):
            print("\nWords Played:")
            for i, word in enumerate(env.state.words_played):
                print(f"  {i+1}. {word}")
        
        # Display game log
        print("\nGame Log:")
        for entry in game_log:
            print(f"  {entry}")
        
        print("\n" + "="*80)
        print(f"Player {player_id} observation:")
        print(observation)
        
        # Get the action from the agent
        action = agents[player_id](observation)
        print(f"Player {player_id} action: {action}")
        
        # Add to game log
        game_log.append(f"Player {player_id}: {action}")
        
        # Call step and handle the return values
        done, info = env.step(action=action)
        
        # Add any game messages to the log
        if hasattr(env.state, 'messages'):
            for message in env.state.messages:
                if message.startswith('[GAME]'):
                    game_log.append(message)
    
    # Close the environment and get rewards
    rewards = env.close()
    
    # Print the results
    print("\n" + "="*80)
    print("GAME RESULTS:")
    if rewards:
        for player_id, reward in rewards.items():
            player_name = 'GPT-4o-mini' if player_id == 0 else 'GPT-3.5-turbo'
            print(f"Player {player_id} ({player_name}): {reward}")
    else:
        print("No rewards available.")
    
    # Print final game log
    print("\nFinal Game Log:")
    for entry in game_log:
        print(f"  {entry}")

if __name__ == "__main__":
    main() 