"""
Chess game example using TextArena with OpenAI agents.

This script demonstrates how to use TextArena to run a Chess game between two OpenAI models.
It uses the TextArena codebase directly from the local installation.
"""

import os
from dotenv import load_dotenv
import textarena as ta

# Load environment variables
load_dotenv()

def main():
    """Run a Chess game between two OpenAI models."""
    # Initialize agents with chess-specific system prompts
    chess_prompt = """You are playing a game of chess. Analyze the board carefully before making your move.
    Submit your move in standard algebraic notation (e.g., 'e2e4', 'Nf3', etc.).
    Consider the strategic implications of your move and plan ahead."""
    
    # Define model names - easy to change here
    model_player0 = "gpt-4o-mini"
    model_player1 = "gpt-3.5-turbo"
    
    # Create a dictionary mapping player IDs to their names
    player_names = {
        0: f"Player 0 ({model_player0})",
        1: f"Player 1 ({model_player1})"
    }
    
    agents = {
        0: ta.agents.OpenAIAgent(
            model_name=model_player0, 
            system_prompt=chess_prompt
        ),
        1: ta.agents.OpenAIAgent(
            model_name=model_player1, 
            system_prompt=chess_prompt
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
    move_log = []
    
    # Print game header
    print("\n" + "="*50)
    print("CHESS GAME")
    print(f"White: {player_names[0]}")
    print(f"Black: {player_names[1]}")
    print("="*50)
    
    # Game loop
    done = False
    turn_count = 1
    while not done:
        # Get the current player ID and observation
        player_id, observation = env.get_observation()
        
        # Extract the chess board from the observation
        board_lines = []
        in_board = False
        for line in observation.split('\n'):
            if '|' in line and ('a' in line or 'b' in line or 'c' in line):
                in_board = True
                board_lines.append(line)
            elif in_board and '|' in line:
                board_lines.append(line)
            elif in_board and len(board_lines) >= 8:
                break
        
        # Print turn header
        print(f"\n--- Turn {turn_count} ---")
        print(f"Current Player: {'White' if player_id == 0 else 'Black'} ({player_names[player_id]})")
        
        # Print the current board state
        if board_lines:
            print("\nBoard:")
            for line in board_lines:
                print(f"  {line}")
        
        # Print move history in a compact format
        if move_log:
            print("\nMove History:")
            history_line = ""
            for i, move in enumerate(move_log):
                move_num = i // 2 + 1
                if i % 2 == 0:  # White's move
                    history_line += f"{move_num}. {move} "
                else:  # Black's move
                    history_line += f"{move} "
                    if (i + 1) % 6 == 0:  # Break line every 3 full moves
                        print(f"  {history_line}")
                        history_line = ""
            if history_line:
                print(f"  {history_line}")
        
        # Get the action from the agent (don't print the full observation)
        print(f"\n{player_names[player_id]} is thinking...")
        action = agents[player_id](observation)
        print(f"{player_names[player_id]} plays: {action}")
        
        # Add to move log
        move_log.append(action)
        
        # Call step and handle the return values
        done, info = env.step(action=action)
        
        # Print any game messages
        if hasattr(env.state, 'messages'):
            for message in env.state.messages:
                if message.startswith('[GAME]'):
                    print(f"Game: {message[7:].strip()}")  # Remove [GAME] prefix
        
        # Increment turn counter if black just played
        if player_id == 1:
            turn_count += 1
    
    # Close the environment and get rewards
    rewards = env.close()
    
    # Print the results
    print("\n" + "="*50)
    print("GAME RESULTS")
    if rewards:
        for player_id, reward in rewards.items():
            player_color = "White" if player_id == 0 else "Black"
            result = "Win" if reward > 0 else "Loss" if reward < 0 else "Draw"
            print(f"{player_color} ({player_names[player_id]}): {result}")
    else:
        print("No results available.")
    print("="*50)

if __name__ == "__main__":
    main() 