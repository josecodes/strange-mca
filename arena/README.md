# Strange MCA Arena

This directory contains scripts for running the Strange MCA agent team in an OpenAI Gym style environment using TextArena. TextArena provides a collection of competitive text-based games for language model evaluation and reinforcement learning.

## Prerequisites

Before running any of the scripts, make sure you have set up your environment variables. Create a `.env` file in the project root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

This project uses a local installation of TextArena from `/Users/jcortez/dev/TextArena`. The dependency is configured in the Poetry configuration, so it should be installed automatically when you run `poetry install`.

## Available Scripts

### 1. `direct_spelling_bee.py`

This script demonstrates how to use TextArena to run a SpellingBee game between two OpenAI models (GPT-4o-mini and GPT-3.5-Turbo).

To run the script:

```bash
poetry run python arena/direct_spelling_bee.py
```

### 2. `direct_tictactoe.py`

This script demonstrates how to use TextArena to run a TicTacToe game between two OpenAI models (GPT-4o-mini and GPT-3.5-Turbo).

To run the script:

```bash
poetry run python arena/direct_tictactoe.py
```

## Game State Display

The example scripts have been enhanced to display detailed game state information without relying on render wrappers. This includes:

- Current player information
- Game-specific state (e.g., allowed letters in SpellingBee, board state in TicTacToe)
- Game log tracking all actions and events
- Player observations and actions
- Game results with player rewards

This approach provides a clean and informative display that works in any terminal size.

## Render Wrapper Options

While the default configuration uses no render wrapper for reliability, you can optionally enable more interactive and visually appealing interfaces by using one of the following render wrappers:

1. **SimpleRenderWrapper**: A basic render wrapper that displays the game state in a formatted box. This can cause display issues in some terminals.

```python
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini", 1: "GPT-3.5-turbo"},
)
```

2. **CursesRenderWrapper**: A more advanced wrapper that uses the curses library for terminal rendering. This provides a nicer UI but requires a terminal that supports curses.

```python
env = ta.wrappers.RenderWrappers.CursesRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini", 1: "GPT-3.5-turbo"},
)
```

You can uncomment the appropriate lines in the example scripts to use these render wrappers.

## Available Games

TextArena provides a wide range of games, including:

- Single-player games: Crosswords, FifteenPuzzle, GuessTheNumber, Hangman, Mastermind, Sudoku, Wordle, etc.
- Two-player games: Chess, ConnectFour, Debate, TicTacToe, Battleship, Checkers, etc.
- Multi-player games: Poker, LiarsDice, Negotiation, etc.

For a complete list of available games, refer to the [TextArena GitHub repository](https://github.com/LeonGuertler/TextArena).

## Creating Custom Scripts

To create a custom script for a different game, follow this template:

```python
import os
from dotenv import load_dotenv
import textarena as ta

# Load environment variables
load_dotenv()

# Initialize agents
agents = {
    0: ta.agents.OpenAIAgent(model_name="gpt-3.5-turbo"),
    1: ta.agents.OpenAIAgent(model_name="gpt-4o-mini"),
}

# Initialize environment and wrap it
env = ta.make(env_id="GameName-v0")  # Replace with the desired game
env = ta.wrappers.LLMObservationWrapper(env=env)

# Game log to track all actions and events
game_log = []

# Reset the environment
env.reset(num_players=len(agents))

# Game loop
done = False
while not done:
    # Get the current player ID and observation
    player_id, observation = env.get_observation()
    
    # Display game state information
    print("\n" + "="*80)
    print(f"GAME STATE:")
    print(f"Current Player: {player_id}")
    
    # Display game-specific state information here
    # ...
    
    # Display game log
    print("\nGame Log:")
    for entry in game_log:
        print(f"  {entry}")
    
    print("\n" + "="*80)
    print(f"Player {player_id} observation:")
    print(observation)
    
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
        print(f"Player {player_id}: {reward}")
else:
    print("No rewards available.")

# Print final game log
print("\nFinal Game Log:")
for entry in game_log:
    print(f"  {entry}")
```

Replace `"GameName-v0"` with the ID of the game you want to play, and adjust the number of players and agent models as needed. 